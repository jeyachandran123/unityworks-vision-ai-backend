"""M4 Frame Buffer — own pixel memory and its lifetime; know nothing about pixels.

Frames are governed by **leases** rather than ownership, so several stages can
read the same frame without copying and memory can never grow without bound
(01_LAYERED §4.3).

Concurrency design, per 03_MODULES M4:

* **Per-camera slot rings.** Each camera writes only to its own ring, so there is
  no write contention between cameras and no single hot map.
* **Immutable published frames.** Once published, pixels are never mutated, so
  readers need no lock at all. This is why multi-consumer reads are free.
* **Reference counting** for leases and pins; reclamation happens when a frame
  has neither.
* **Lease deadlines with forced break.** One stuck stage must not exhaust the
  pool for every camera (invariant V9).

Buffer capacity is a function of *pipeline depth and jitter*, not of camera
count — which is why a 100-camera node does not need 100x the memory of a
1-camera node.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ...core.errors import FrameUnavailableError, PoolExhaustedError
from ...core.model.camera import SourceSemantics
from ...core.model.frame import (
    Frame,
    FrameDimensions,
    FrameQuality,
    PrivacyState,
    SourceMeta,
)
from ...core.model.ids import CameraId, FrameRef
from ...core.model.timebase import Duration, FrameTime, Instant
from ...core.ports.buffer import Allocation, AllocatorPort
from ...core.ports.clock import Clock
from ...kernel.config.schema import BufferSection
from ...kernel.events import EventBus, FrameEvicted, LeaseLeaked, PoolPressure
from ...kernel.metrics import MetricName, MetricsEngine


class FrameSlot:
    """A writable region handed to a decoder. Implements ``WritableSlot``."""

    __slots__ = ("_allocation", "_camera_id", "_released")

    def __init__(self, allocation: Allocation, camera_id: CameraId) -> None:
        self._allocation = allocation
        self._camera_id = camera_id
        self._released = False

    @property
    def capacity(self) -> int:
        return self._allocation.nbytes

    @property
    def camera_id(self) -> CameraId:
        return self._camera_id

    def memory(self) -> memoryview:
        if self._released:
            raise FrameUnavailableError("slot has already been released")
        return self._allocation.memory()

    @property
    def allocation(self) -> Allocation:
        return self._allocation


class _PublishedPixels:
    """Read-only pixel view over a published allocation. Implements ``PixelBuffer``."""

    __slots__ = ("_allocation", "_nbytes")

    def __init__(self, allocation: Allocation, nbytes: int) -> None:
        self._allocation = allocation
        self._nbytes = nbytes

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def readonly_view(self) -> memoryview:
        return self._allocation.memory()[: self._nbytes].toreadonly()


class FrameLease:
    """A read-only, deadline-bounded borrow of a published frame."""

    __slots__ = ("_entry", "_holder_id", "_deadline", "_released", "_ring")

    def __init__(
        self, entry: _Entry, holder_id: str, deadline: Instant, ring: _CameraRing
    ) -> None:
        self._entry = entry
        self._holder_id = holder_id
        self._deadline = deadline
        self._released = False
        self._ring = ring

    @property
    def frame(self) -> Frame:
        if self._released:
            raise FrameUnavailableError("lease has been released or force-broken")
        return self._entry.frame

    @property
    def holder_id(self) -> str:
        return self._holder_id

    @property
    def deadline(self) -> Instant:
        return self._deadline

    @property
    def released(self) -> bool:
        return self._released

    def pixels(self) -> memoryview:
        """A read-only view. No stage may mutate shared pixels."""
        return self.frame.pixels.readonly_view()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._ring._release_lease(self._entry, self)  # noqa: SLF001

    def _force_break(self) -> None:
        self._released = True

    def __enter__(self) -> FrameLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class PinHandle:
    """Extends retention so a late consumer can still reach a frame."""

    __slots__ = ("frame_ref", "expires_at", "reason", "_released")

    def __init__(self, frame_ref: FrameRef, expires_at: Instant, reason: str) -> None:
        self.frame_ref = frame_ref
        self.expires_at = expires_at
        self.reason = reason
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def _mark_released(self) -> None:
        self._released = True


@dataclass(slots=True)
class _Entry:
    frame: Frame
    allocation: Allocation
    published_at: Instant
    leases: list[FrameLease]
    pins: list[PinHandle]

    @property
    def reclaimable(self) -> bool:
        return not self.leases and not self.pins


@dataclass(frozen=True, slots=True)
class BufferStats:
    slots_total: int
    slots_in_use: int
    leases_active: int
    pins_active: int
    evictions: int
    lease_leaks: int
    pool_exhaustions: int


class _CameraRing:
    """One camera's slot ring. Single writer, own lock, no cross-camera sharing."""

    __slots__ = ("camera_id", "capacity", "_lock", "_entries", "_order", "_owner")

    def __init__(self, camera_id: CameraId, capacity: int, owner: FrameBuffer) -> None:
        self.camera_id = camera_id
        self.capacity = capacity
        self._lock = threading.RLock()
        self._entries: dict[FrameRef, _Entry] = {}
        self._order: list[FrameRef] = []
        self._owner = owner

    def _release_lease(self, entry: _Entry, lease: FrameLease) -> None:
        with self._lock:
            if lease in entry.leases:
                entry.leases.remove(lease)
        self._owner._note_lease_released()  # noqa: SLF001


class FrameBuffer:
    """Bounded, pooled frame storage with lease-governed lifetime."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        allocator: AllocatorPort,
        config: BufferSection,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._allocator = allocator
        self._config = config
        self._rings: dict[CameraId, _CameraRing] = {}
        self._rings_lock = threading.RLock()
        self._counters_lock = threading.Lock()
        self._leases_active = 0
        self._pins_active = 0
        self._evictions = 0
        self._lease_leaks = 0
        self._pool_exhaustions = 0

    # --- registration ------------------------------------------------------ #

    def register_camera(self, camera_id: CameraId) -> None:
        with self._rings_lock:
            if camera_id not in self._rings:
                self._rings[camera_id] = _CameraRing(
                    camera_id, self._config.slots_per_camera, self
                )

    def forget_camera(self, camera_id: CameraId) -> None:
        with self._rings_lock:
            ring = self._rings.pop(camera_id, None)
        if ring is None:
            return
        with ring._lock:  # noqa: SLF001
            for ref in list(ring._order):  # noqa: SLF001
                entry = ring._entries.pop(ref, None)  # noqa: SLF001
                if entry is not None:
                    self._allocator.release(entry.allocation)
            ring._order.clear()  # noqa: SLF001

    def _ring(self, camera_id: CameraId) -> _CameraRing:
        with self._rings_lock:
            ring = self._rings.get(camera_id)
            if ring is None:
                ring = _CameraRing(camera_id, self._config.slots_per_camera, self)
                self._rings[camera_id] = ring
            return ring

    # --- write side -------------------------------------------------------- #

    def acquire_slot(
        self, camera_id: CameraId, semantics: SourceSemantics = SourceSemantics.REALTIME
    ) -> FrameSlot:
        """Take a writable slot for this camera.

        On a full ring the behaviour is decided by source semantics:
        ``realtime`` evicts the oldest unpinned frame (latency is protected over
        completeness); ``archival`` raises ``PoolExhaustedError`` so the producer
        blocks (completeness is protected over latency).
        """
        ring = self._ring(camera_id)
        with ring._lock:  # noqa: SLF001
            if len(ring._order) >= ring.capacity:  # noqa: SLF001
                if not self._evict_oldest_locked(ring, semantics):
                    with self._counters_lock:
                        self._pool_exhaustions += 1
                    self._metrics.counter(
                        MetricName.POOL_EXHAUSTED, camera_id=str(camera_id)
                    ).increment()
                    raise PoolExhaustedError(
                        f"no capacity for camera '{camera_id}' "
                        f"({len(ring._order)}/{ring.capacity} slots held)",  # noqa: SLF001
                        camera_id=str(camera_id),
                    )
        allocation = self._allocator.allocate(self._config.bytes_per_slot)
        return FrameSlot(allocation, camera_id)

    def discard_slot(self, slot: FrameSlot) -> None:
        """Return an unpublished slot to the pool (decode failure, mask failure)."""
        self._allocator.release(slot.allocation)

    def publish(
        self,
        slot: FrameSlot,
        *,
        frame_ref: FrameRef,
        time: FrameTime,
        dimensions: FrameDimensions,
        privacy_state: PrivacyState,
        bytes_written: int,
        quality: FrameQuality | None = None,
        source_meta: SourceMeta | None = None,
    ) -> Frame:
        """Make a slot readable as an immutable Frame.

        Refuses to publish a frame whose masking failed — the ``Frame``
        constructor enforces this too, so the invariant holds at both levels
        (12_SECURITY §2.1).
        """
        if not privacy_state.emittable:
            self.discard_slot(slot)
            raise ValueError(
                f"refusing to publish {frame_ref} with privacy_state={privacy_state.value}"
            )

        frame = Frame(
            frame_ref=frame_ref,
            time=time,
            dimensions=dimensions,
            pixels=_PublishedPixels(slot.allocation, bytes_written),
            privacy_state=privacy_state,
            quality=quality or FrameQuality(),
            source_meta=source_meta or SourceMeta(),
        )
        entry = _Entry(
            frame=frame,
            allocation=slot.allocation,
            published_at=self._clock.monotonic(),
            leases=[],
            pins=[],
        )
        ring = self._ring(frame_ref.camera_id)
        with ring._lock:  # noqa: SLF001
            ring._entries[frame_ref] = entry  # noqa: SLF001
            ring._order.append(frame_ref)  # noqa: SLF001
            in_use = len(ring._order)  # noqa: SLF001

        self._metrics.gauge(
            MetricName.BUFFER_SLOTS_IN_USE, camera_id=str(frame_ref.camera_id)
        ).set(in_use)
        self._metrics.gauge(
            MetricName.BUFFER_SLOTS_TOTAL, camera_id=str(frame_ref.camera_id)
        ).set(ring.capacity)
        if in_use >= ring.capacity:
            self._bus.publish(
                PoolPressure(
                    occurred_at=self._clock.now(),
                    partition_key=str(frame_ref.camera_id),
                    location=self._allocator.location,
                    in_use=in_use,
                    total=ring.capacity,
                )
            )
        return frame

    # --- read side --------------------------------------------------------- #

    def acquire(
        self, frame_ref: FrameRef, holder_id: str, deadline: Duration | None = None
    ) -> FrameLease:
        """Borrow a published frame.

        Raises ``FrameUnavailableError`` when the frame has been evicted — a
        normal, expected outcome that callers degrade on and count, never an
        error that propagates upward.
        """
        ring = self._ring(frame_ref.camera_id)
        with ring._lock:  # noqa: SLF001
            entry = ring._entries.get(frame_ref)  # noqa: SLF001
            if entry is None:
                raise FrameUnavailableError(
                    f"frame {frame_ref} is not resident", frame_ref=str(frame_ref)
                )
            span = deadline or Duration.from_millis(self._config.lease_deadline_ms)
            lease = FrameLease(
                entry, holder_id, self._clock.monotonic().plus(span), ring
            )
            entry.leases.append(lease)
        with self._counters_lock:
            self._leases_active += 1
        self._metrics.gauge(
            MetricName.BUFFER_LEASES_ACTIVE, camera_id=str(frame_ref.camera_id)
        ).set(len(entry.leases))
        return lease

    def try_acquire(self, frame_ref: FrameRef, holder_id: str) -> FrameLease | None:
        try:
            return self.acquire(frame_ref, holder_id)
        except FrameUnavailableError:
            return None

    def pin(self, frame_ref: FrameRef, ttl: Duration, reason: str) -> PinHandle:
        """Extend retention past normal reclamation."""
        ring = self._ring(frame_ref.camera_id)
        with ring._lock:  # noqa: SLF001
            entry = ring._entries.get(frame_ref)  # noqa: SLF001
            if entry is None:
                raise FrameUnavailableError(
                    f"frame {frame_ref} is not resident", frame_ref=str(frame_ref)
                )
            handle = PinHandle(frame_ref, self._clock.monotonic().plus(ttl), reason)
            entry.pins.append(handle)
        with self._counters_lock:
            self._pins_active += 1
        self._metrics.gauge(
            MetricName.BUFFER_PINS_ACTIVE, camera_id=str(frame_ref.camera_id)
        ).set(len(entry.pins))
        return handle

    def unpin(self, handle: PinHandle) -> None:
        if handle.released:
            return
        ring = self._ring(handle.frame_ref.camera_id)
        with ring._lock:  # noqa: SLF001
            entry = ring._entries.get(handle.frame_ref)  # noqa: SLF001
            if entry is not None and handle in entry.pins:
                entry.pins.remove(handle)
        handle._mark_released()  # noqa: SLF001
        with self._counters_lock:
            self._pins_active -= 1

    def is_resident(self, frame_ref: FrameRef) -> bool:
        ring = self._ring(frame_ref.camera_id)
        with ring._lock:  # noqa: SLF001
            return frame_ref in ring._entries  # noqa: SLF001

    # --- maintenance -------------------------------------------------------- #

    def sweep(self) -> int:
        """Force-break expired leases, expire pins, and reclaim history.

        Returns the number of frames reclaimed. Called on the runtime's
        maintenance tick.
        """
        now = self._clock.monotonic()
        history_ns = self._config.history_window_ms * 1_000_000
        reclaimed = 0
        with self._rings_lock:
            rings = list(self._rings.values())

        for ring in rings:
            with ring._lock:  # noqa: SLF001
                for ref in list(ring._order):  # noqa: SLF001
                    entry = ring._entries.get(ref)  # noqa: SLF001
                    if entry is None:
                        continue
                    self._break_expired_leases(entry, now)
                    self._expire_pins(entry, now)
                    age = now.ns - entry.published_at.ns
                    if age > history_ns and entry.reclaimable:
                        ring._entries.pop(ref, None)  # noqa: SLF001
                        ring._order.remove(ref)  # noqa: SLF001
                        self._allocator.release(entry.allocation)
                        reclaimed += 1
        return reclaimed

    def _break_expired_leases(self, entry: _Entry, now: Instant) -> None:
        expired = [lease for lease in entry.leases if lease.deadline.ns < now.ns]
        for lease in expired:
            entry.leases.remove(lease)
            lease._force_break()  # noqa: SLF001
            with self._counters_lock:
                self._leases_active -= 1
                self._lease_leaks += 1
            self._metrics.counter(
                MetricName.LEASE_LEAKS, holder=lease.holder_id
            ).increment()
            self._bus.publish(
                LeaseLeaked(
                    occurred_at=self._clock.now(),
                    partition_key=str(entry.frame.frame_ref.camera_id),
                    holder_id=lease.holder_id,
                    frame_ref=str(entry.frame.frame_ref),
                )
            )

    def _expire_pins(self, entry: _Entry, now: Instant) -> None:
        expired = [pin for pin in entry.pins if pin.expires_at.ns < now.ns]
        for pin in expired:
            entry.pins.remove(pin)
            pin._mark_released()  # noqa: SLF001
            with self._counters_lock:
                self._pins_active -= 1

    def _evict_oldest_locked(self, ring: _CameraRing, semantics: SourceSemantics) -> bool:
        """Evict the oldest reclaimable frame. Never frees a pinned or leased frame.

        Semantics decide only whether an *unexpired* frame may be dropped:
        ``realtime`` evicts the oldest reclaimable frame to protect latency;
        ``archival`` reclaims only frames already past the retention horizon, so
        the producer blocks rather than losing a recorded frame.
        """
        now = self._clock.monotonic()
        history_ns = self._config.history_window_ms * 1_000_000
        for ref in list(ring._order):  # noqa: SLF001
            entry = ring._entries.get(ref)  # noqa: SLF001
            if entry is None:
                ring._order.remove(ref)  # noqa: SLF001
                continue
            self._break_expired_leases(entry, now)
            self._expire_pins(entry, now)
            if not entry.reclaimable:
                continue
            expired = (now.ns - entry.published_at.ns) > history_ns
            if not semantics.may_drop_frames and not expired:
                continue
            ring._entries.pop(ref, None)  # noqa: SLF001
            ring._order.remove(ref)  # noqa: SLF001
            self._allocator.release(entry.allocation)
            with self._counters_lock:
                self._evictions += 1
            self._metrics.counter(
                MetricName.FRAMES_EVICTED, camera_id=str(ring.camera_id)
            ).increment()
            self._bus.publish(
                FrameEvicted(
                    occurred_at=self._clock.now(),
                    partition_key=str(ring.camera_id),
                    camera_id=ring.camera_id,
                    frame_ref=str(ref),
                )
            )
            return True
        return False

    def _note_lease_released(self) -> None:
        with self._counters_lock:
            self._leases_active -= 1

    # --- telemetry ---------------------------------------------------------- #

    def stats(self) -> BufferStats:
        with self._rings_lock:
            rings = list(self._rings.values())
        in_use = 0
        total = 0
        for ring in rings:
            with ring._lock:  # noqa: SLF001
                in_use += len(ring._order)  # noqa: SLF001
                total += ring.capacity
        with self._counters_lock:
            return BufferStats(
                slots_total=total,
                slots_in_use=in_use,
                leases_active=self._leases_active,
                pins_active=self._pins_active,
                evictions=self._evictions,
                lease_leaks=self._lease_leaks,
                pool_exhaustions=self._pool_exhaustions,
            )

    def close(self) -> None:
        with self._rings_lock:
            camera_ids = list(self._rings)
        for camera_id in camera_ids:
            self.forget_camera(camera_id)
