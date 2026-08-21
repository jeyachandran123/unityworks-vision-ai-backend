"""The device broker (05_MODULES_PLATFORM_KERNEL M18).

**Consumers request capacity; the broker grants or denies. No consumer touches
device memory directly.** Without a broker, a detector and a VLM sharing one GPU
will eventually OOM each other at the worst possible moment, and the failure will
look like a random inference error rather than the resource conflict it is.

The broker refuses overcommit rather than discovering it mid-inference, and it
degrades to CPU rather than failing when an accelerator disappears (invariant V9).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from ...core.errors import DeviceOutOfMemoryError, DeviceUnavailableError
from ...core.ports.models import DeviceInfo, DeviceKind, DevicePort

CPU_DEVICE_ID = "cpu"


@dataclass(frozen=True, slots=True)
class Reservation:
    """A granted claim on device memory."""

    reservation_id: str
    device_id: str
    bytes_reserved: int
    owner: str
    priority_class: str = "default"
    pinned: bool = False
    """Pinned reservations are never evicted to satisfy another request."""


@dataclass(frozen=True, slots=True)
class DeviceState:
    info: DeviceInfo
    reserved_bytes: int
    reservation_count: int
    available: bool

    @property
    def free_bytes(self) -> int:
        return max(0, self.info.total_memory_bytes - self.reserved_bytes)

    @property
    def utilization(self) -> float:
        if self.info.total_memory_bytes <= 0:
            return 0.0
        return self.reserved_bytes / self.info.total_memory_bytes


@dataclass(frozen=True, slots=True)
class DeviceReport:
    devices: tuple[DeviceState, ...]
    total_reserved_bytes: int
    accelerators_available: int

    def find(self, device_id: str) -> DeviceState | None:
        for state in self.devices:
            if state.info.device_id == device_id:
                return state
        return None


@dataclass(slots=True)
class _DeviceRecord:
    info: DeviceInfo
    provider: DevicePort
    reservations: dict[str, Reservation] = field(default_factory=dict)

    @property
    def reserved(self) -> int:
        return sum(r.bytes_reserved for r in self.reservations.values())


class DeviceBroker:
    """Arbitrates device memory between competing consumers.

    Placement policy, in order:

    1. An explicitly requested device, if it can satisfy the reservation.
    2. The least-utilized available accelerator that can satisfy it.
    3. CPU, when CPU fallback is permitted.

    Preferring the least-utilized accelerator rather than the first one keeps a
    multi-GPU node balanced without a scheduler; preferring an explicit hint
    keeps a pinned model where an operator put it.
    """

    def __init__(
        self,
        providers: Sequence[DevicePort],
        *,
        allow_cpu_fallback: bool = True,
        headroom_fraction: float = 0.1,
    ) -> None:
        if not 0.0 <= headroom_fraction < 1.0:
            raise ValueError(f"headroom_fraction must be in [0,1), got {headroom_fraction}")
        self._providers = tuple(providers)
        self._allow_cpu_fallback = allow_cpu_fallback
        self._headroom = headroom_fraction
        self._lock = threading.RLock()
        self._devices: dict[str, _DeviceRecord] = {}
        self._counter = 0
        self.refresh()

    # --- inventory ------------------------------------------------------------ #

    def refresh(self) -> None:
        """Re-enumerate devices.

        A device that has disappeared keeps its record so existing reservations
        remain accountable, but stops being selectable — migration is the Model
        Manager's decision, not the broker's.
        """
        with self._lock:
            seen: set[str] = set()
            for provider in self._providers:
                try:
                    infos = provider.enumerate()
                except Exception:  # noqa: BLE001, S112 - a broken provider must not blind the broker
                    continue
                for info in infos:
                    seen.add(info.device_id)
                    record = self._devices.get(info.device_id)
                    if record is None:
                        self._devices[info.device_id] = _DeviceRecord(
                            info=info, provider=provider
                        )
                    else:
                        record.info = info
            for device_id, record in self._devices.items():
                if device_id not in seen:
                    record.info = DeviceInfo(
                        device_id=record.info.device_id,
                        kind=record.info.kind,
                        index=record.info.index,
                        total_memory_bytes=record.info.total_memory_bytes,
                        name=record.info.name,
                        available=False,
                    )

    def devices(self) -> tuple[DeviceInfo, ...]:
        with self._lock:
            return tuple(record.info for record in self._devices.values())

    def is_available(self, device_id: str) -> bool:
        with self._lock:
            record = self._devices.get(device_id)
        if record is None:
            return False
        if not record.info.available:
            return False
        try:
            return record.provider.is_available(device_id)
        except Exception:  # noqa: BLE001 - liveness checks never raise upward
            return False

    def report(self) -> DeviceReport:
        with self._lock:
            states = tuple(
                DeviceState(
                    info=record.info,
                    reserved_bytes=record.reserved,
                    reservation_count=len(record.reservations),
                    available=record.info.available,
                )
                for record in self._devices.values()
            )
            return DeviceReport(
                devices=states,
                total_reserved_bytes=sum(s.reserved_bytes for s in states),
                accelerators_available=sum(
                    1
                    for s in states
                    if s.available and s.info.kind.is_accelerator
                ),
            )

    # --- reservation ----------------------------------------------------------- #

    def reserve(
        self,
        *,
        owner: str,
        bytes_required: int,
        device_hint: str | None = None,
        priority_class: str = "default",
        pinned: bool = False,
    ) -> Reservation:
        """Claim device memory, or explain why not.

        Raises:
            DeviceOutOfMemoryError: no device can satisfy the request even after
                evicting non-pinned reservations of lower priority.
            DeviceUnavailableError: an explicit hint named a device that is gone
                and CPU fallback is disabled.
        """
        if bytes_required < 0:
            raise ValueError("bytes_required must be non-negative")

        with self._lock:
            if device_hint is not None:
                record = self._devices.get(device_hint)
                if record is not None and self._fits(record, bytes_required):
                    return self._grant(record, owner, bytes_required, priority_class, pinned)
                if record is None or not record.info.available:
                    if not self._allow_cpu_fallback:
                        raise DeviceUnavailableError(
                            f"device '{device_hint}' is unavailable and CPU fallback "
                            f"is disabled",
                            device_id=device_hint,
                        )

            candidate = self._select(bytes_required)
            if candidate is not None:
                return self._grant(candidate, owner, bytes_required, priority_class, pinned)

            evicted = self._evict_for(bytes_required, priority_class)
            if evicted:
                candidate = self._select(bytes_required)
                if candidate is not None:
                    return self._grant(
                        candidate, owner, bytes_required, priority_class, pinned
                    )

            report = self.report()
            raise DeviceOutOfMemoryError(
                f"no device can satisfy {bytes_required} bytes for '{owner}'; "
                f"{report.accelerators_available} accelerator(s) available, "
                f"{report.total_reserved_bytes} bytes already reserved",
                owner=owner,
                bytes_required=bytes_required,
            )

    def release(self, reservation: Reservation) -> None:
        """Return a reservation. Idempotent."""
        with self._lock:
            record = self._devices.get(reservation.device_id)
            if record is not None:
                record.reservations.pop(reservation.reservation_id, None)

    def reservations_on(self, device_id: str) -> tuple[Reservation, ...]:
        with self._lock:
            record = self._devices.get(device_id)
            return tuple(record.reservations.values()) if record else ()

    def evicted_candidates(self, device_id: str) -> tuple[Reservation, ...]:
        """Non-pinned reservations, oldest first — the eviction order."""
        with self._lock:
            record = self._devices.get(device_id)
            if record is None:
                return ()
            return tuple(r for r in record.reservations.values() if not r.pinned)

    # --- internals -------------------------------------------------------------- #

    def _fits(self, record: _DeviceRecord, bytes_required: int) -> bool:
        if not record.info.available:
            return False
        if record.info.kind is DeviceKind.CPU:
            return True  # host memory is not brokered by capacity here
        usable = int(record.info.total_memory_bytes * (1.0 - self._headroom))
        return record.reserved + bytes_required <= usable

    def _select(self, bytes_required: int) -> _DeviceRecord | None:
        accelerators = [
            record
            for record in self._devices.values()
            if record.info.kind.is_accelerator and self._fits(record, bytes_required)
        ]
        if accelerators:
            return min(
                accelerators,
                key=lambda r: (
                    r.reserved / r.info.total_memory_bytes
                    if r.info.total_memory_bytes
                    else 1.0
                ),
            )
        if not self._allow_cpu_fallback:
            return None
        for record in self._devices.values():
            if record.info.kind is DeviceKind.CPU and record.info.available:
                return record
        return None

    def _evict_for(self, bytes_required: int, priority_class: str) -> bool:
        """Evict non-pinned reservations to make room. Returns whether any went.

        Pinned reservations are never evicted: an operator who pinned a model
        meant it, and silently unloading it would be a surprise at the worst
        possible time.
        """
        evicted = False
        for record in self._devices.values():
            if not record.info.kind.is_accelerator or not record.info.available:
                continue
            candidates = [r for r in record.reservations.values() if not r.pinned]
            for reservation in candidates:
                if self._fits(record, bytes_required):
                    break
                record.reservations.pop(reservation.reservation_id, None)
                evicted = True
        return evicted

    def _grant(
        self,
        record: _DeviceRecord,
        owner: str,
        bytes_required: int,
        priority_class: str,
        pinned: bool,
    ) -> Reservation:
        self._counter += 1
        reservation = Reservation(
            reservation_id=f"res-{self._counter}",
            device_id=record.info.device_id,
            bytes_reserved=bytes_required,
            owner=owner,
            priority_class=priority_class,
            pinned=pinned,
        )
        record.reservations[reservation.reservation_id] = reservation
        return reservation
