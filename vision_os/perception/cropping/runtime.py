"""The Crop Runtime — the ``RegistryConsumer`` seam, one actor per camera.

> **Single responsibility:** *Own the Crop Manager's lifecycle, serialize each
> camera's trigger writes, and take the lease. Decide nothing yourself.*

``08_RUNTIME`` places M8 in the actor table, and §M8's Thread Safety section
splits its concurrency in two — a split this runtime makes real:

**Trigger state is per-camera single-writer**, matching M7. One lock per camera,
so two updates for one camera can never interleave and two cameras never contend.

**The budget is shared across cameras**, deliberately. Understanding cost is a
property of the node's GPU, not of any camera, and a per-camera cap cannot stop
100 cameras each staying under their own limit while collectively exhausting the
device. What is *not* shared is coordination: no camera ever waits on another,
because the critical section is a counter increment.

This runtime also owns the one thing the frame path does not drive: **demand
expiry**. A demand with a TTL must expire even at a camera that has gone quiet,
or a consumer's contract would outlive the window it was acknowledged for.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...core.errors import (
    CropError,
    CropExtractionError,
    FrameUnavailableError,
    GateRejectedError,
)
from ...core.model.crop import Crop, CropRequest, EvaluationResult, Skipped, SkipReason
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import CameraId, ModuleId, ObjectId
from ...core.model.space import Box
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...kernel.config.schema import CroppingSection
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from ..registry.engine import RegistryUpdate
from .engine import CropManager, FrameContext

CROP_RUNTIME_ID = ModuleId("crop_runtime")

_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)

#: How long a crop's source frame is pinned while extraction runs.
#:
#: Short: extraction is milliseconds, and a long pin holds a buffer slot that
#: acquisition needs. A frame evicted despite the pin produces
#: ``FRAME_UNAVAILABLE``, which §M8 calls *"a diagnosable configuration issue
#: rather than a mystery"*.
_LEASE_DEADLINE = Duration.from_millis(500)


@dataclass(frozen=True, slots=True)
class _Extraction:
    """One frame's extraction outcome.

    ``fulfilled`` and ``skips`` partition the admitted requests: a request lands
    in exactly one of them, which is what keeps the published accounting an
    exactly-once statement about every candidate.
    """

    fulfilled: tuple[CropRequest, ...]
    crops: list[Crop]
    skips: list[Skipped]


class CropRuntimeStats:
    """Mutable counters. Updated on the hot path, never published as a value."""

    __slots__ = (
        "crops_produced",
        "extraction_failures",
        "frames_consumed",
        "frames_failed",
        "gate_rejections",
        "requests_made",
        "sink_failures",
        "skips_recorded",
        "unavailable_frames",
    )

    def __init__(self) -> None:
        self.frames_consumed = 0
        self.frames_failed = 0
        self.requests_made = 0
        self.skips_recorded = 0
        self.crops_produced = 0
        self.gate_rejections = 0
        self.unavailable_frames = 0
        self.extraction_failures = 0
        self.sink_failures = 0

    @property
    def failure_rate(self) -> float:
        return self.frames_failed / self.frames_consumed if self.frames_consumed else 0.0


class CropRuntime:
    """Implements the registry-to-cropping seam; owns Crop Manager lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        health: HealthMonitor,
        manager: CropManager,
        config: CroppingSection,
        buffer=None,
        sink=None,
        regions_of=None,
        appearance_of=None,
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._health = health
        self._manager = manager
        self._config = config
        self._buffer = buffer
        self._sink = sink
        self._regions_of = regions_of
        self._appearance_of = appearance_of
        self._report_interval = report_interval

        self._stats = CropRuntimeStats()
        self._locks: dict[CameraId, asyncio.Lock] = {}
        self._started = False
        self._last_report_ns = 0
        self._last_expiry_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        self._started = True
        self._report_health(HealthState.HEALTHY, "crop manager ready")

    async def stop(self) -> None:
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seam ---------------------------------------------------------------- #

    async def on_registered(self, update: RegistryUpdate) -> None:
        """Consume one camera's registry update. **Never raises.**

        An attention failure may not stop the registry, which may not stop
        tracking, which may not stop detection, which may not stop acquisition
        (invariant V9).
        """
        if not self._started or not self._config.enabled:
            return

        self._stats.frames_consumed += 1
        if update.failed:
            # The registry could not run. Evaluating its (empty) object list
            # would manufacture a scene-is-empty conclusion from a broken
            # upstream — exactly the conflation V8 forbids.
            self._stats.frames_failed += 1
            return

        camera_id = update.camera_id
        lock = self._locks.get(camera_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[camera_id] = lock

        try:
            async with lock:
                result = self._evaluate(update)
                extraction = self._extract_all(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._stats.frames_failed += 1
            self._metrics.counter(
                MetricName.CROP_EXTRACTION_FAILURES,
                camera_id=str(camera_id),
                reason="runtime_guard",
            ).increment()
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        self._publish(result, extraction)
        self._maintain()

    def _evaluate(self, update: RegistryUpdate) -> EvaluationResult:
        frame = self._frame_context(update)
        if frame is None:
            # No metadata for this frame means no crop can be taken from it.
            # Every candidate is skipped with an attributed reason rather than
            # disappearing (V8).
            self._stats.unavailable_frames += 1
            self._metrics.counter(
                MetricName.CROP_FRAME_UNAVAILABLE, camera_id=str(update.camera_id)
            ).increment()
            return EvaluationResult(
                camera_id=update.camera_id,
                frame_ref=update.frame_ref,
                skipped=tuple(
                    Skipped(
                        object_id=obj.object_id,
                        camera_id=update.camera_id,
                        reason=SkipReason.FRAME_UNAVAILABLE,
                        detail="frame evicted before trigger evaluation",
                    )
                    for obj in update.present
                ),
            )
        return self._manager.evaluate(
            update.present,
            frame,
            regions_of=self._regions_of,
            appearance_of=self._appearance_of,
        )

    def _extract_all(self, result: EvaluationResult) -> _Extraction:
        """Take one lease and extract every admitted request from it.

        One lease per frame rather than one per request: a lease is a
        buffer-slot reservation, and taking N of them for N objects on the same
        frame would multiply pressure on the pool for no benefit.

        Returns the **fulfilled** requests alongside the crops and the new skips.
        A request that was admitted and then gate-rejected has *moved* to the
        skip column; reporting it in both would double-count the candidate and
        break the exactly-once identity that makes V8 checkable.
        """
        if not result.requests:
            return _Extraction((), [], [])
        if self._buffer is None:
            return _Extraction(
                (),
                [],
                [
                    self._unavailable(request, "no frame buffer bound")
                    for request in result.requests
                ],
            )

        try:
            lease = self._buffer.acquire(
                result.frame_ref, str(CROP_RUNTIME_ID), _LEASE_DEADLINE
            )
        except FrameUnavailableError:
            self._stats.unavailable_frames += len(result.requests)
            self._metrics.counter(
                MetricName.CROP_FRAME_UNAVAILABLE, camera_id=str(result.camera_id)
            ).increment(len(result.requests))
            return _Extraction(
                (),
                [],
                [
                    self._unavailable(request, "frame evicted before extraction")
                    for request in result.requests
                ],
            )

        crops: list[Crop] = []
        skips: list[Skipped] = []
        fulfilled: list[CropRequest] = []
        try:
            frame = lease.frame
            context = FrameContext(
                frame_ref=frame.frame_ref,
                width=frame.dimensions.width,
                height=frame.dimensions.height,
                t_capture=frame.time.t_capture,
                colour_space=frame.dimensions.colour_space,
            )
            pixels = lease.pixels()
            neighbours = tuple(request.source_box for request in result.requests)
            channels = self._channels(frame.dimensions.colour_space)

            for request in result.requests:
                others = tuple(box for box in neighbours if box is not request.source_box)
                crop_or_skip = self._extract_one(request, pixels, context, channels, others)
                if isinstance(crop_or_skip, Skipped):
                    skips.append(crop_or_skip)
                else:
                    crops.append(crop_or_skip)
                    fulfilled.append(request)
        finally:
            lease.release()

        return _Extraction(tuple(fulfilled), crops, skips)

    def _extract_one(
        self,
        request: CropRequest,
        pixels,
        context: FrameContext,
        channels: int,
        neighbours: tuple[Box, ...],
    ) -> Crop | Skipped:
        """Extract one crop, converting each documented failure into a record.

        The three outcomes are kept distinct all the way out, because they call
        for three different operator responses: a gate rejection means the
        *input* was poor, an eviction means the *buffer* is too shallow, and an
        extraction error means the *code* is faulty.
        """
        try:
            crop = self._manager.extract(
                request,
                pixels=pixels,
                frame=context,
                channels=channels,
                neighbour_boxes=neighbours,
            )
        except GateRejectedError as exc:
            self._stats.gate_rejections += 1
            return Skipped(
                object_id=request.object_id,
                camera_id=request.camera_id,
                reason=SkipReason.QUALITY_INSUFFICIENT,
                detail=str(exc.context.get("reason", "")) or exc.message,
                attribute_keys=request.required_attributes,
            )
        except FrameUnavailableError:
            self._stats.unavailable_frames += 1
            return self._unavailable(request, "frame evicted during extraction")
        except CropExtractionError as exc:
            self._stats.extraction_failures += 1
            return Skipped(
                object_id=request.object_id,
                camera_id=request.camera_id,
                reason=SkipReason.QUALITY_INSUFFICIENT,
                detail=f"extraction failed: {exc.message}",
                attribute_keys=request.required_attributes,
            )
        except CropError as exc:
            self._stats.extraction_failures += 1
            return Skipped(
                object_id=request.object_id,
                camera_id=request.camera_id,
                reason=SkipReason.QUALITY_INSUFFICIENT,
                detail=f"{type(exc).__name__}: {exc.message}",
                attribute_keys=request.required_attributes,
            )
        self._stats.crops_produced += 1
        return crop

    def _unavailable(self, request: CropRequest, detail: str) -> Skipped:
        return Skipped(
            object_id=request.object_id,
            camera_id=request.camera_id,
            reason=SkipReason.FRAME_UNAVAILABLE,
            detail=detail,
            attribute_keys=request.required_attributes,
        )

    def _publish(self, result: EvaluationResult, extraction: _Extraction) -> None:
        """Emit the settled accounting: fulfilled requests, crops, and skips.

        ``requests`` narrows to what actually produced evidence, so a candidate
        admitted and then gate-rejected is reported once, on the skip side.
        """
        self._stats.requests_made += len(extraction.fulfilled)
        self._stats.skips_recorded += len(result.skipped) + len(extraction.skips)

        if self._sink is None:
            return
        try:
            self._sink(
                EvaluationResult(
                    camera_id=result.camera_id,
                    frame_ref=result.frame_ref,
                    requests=extraction.fulfilled,
                    skipped=result.skipped + tuple(extraction.skips),
                ),
                tuple(extraction.crops),
            )
        except Exception:  # noqa: BLE001 - a bad sink must not break attention
            self._stats.sink_failures += 1

    # --- schedules ---------------------------------------------------------------- #

    def _maintain(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_expiry_ns >= 1_000_000_000:
            self._last_expiry_ns = now
            try:
                self._manager.expire_demands()
            except Exception:  # noqa: BLE001 - maintenance never stops attention
                self._metrics.counter(
                    MetricName.CROP_EXTRACTION_FAILURES, reason="demand_expiry"
                ).increment()
        self._maybe_report()

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._manager.health()
        self._report_health(health.state, health.detail)

        status = self._manager.budget_status()
        self._metrics.gauge(MetricName.UNDERSTANDING_BUDGET_PRESSURE).set(status.pressure)
        self._metrics.counter(MetricName.UNDERSTANDING_BUDGET_SPENT).increment(0)
        self._metrics.gauge(MetricName.DEMANDS_ACTIVE).set(
            float(len(self._manager.demands.active()))
        )
        self._metrics.gauge(MetricName.CROP_CACHE_EVICTIONS).set(
            float(self._manager.cache.stats().evictions)
        )

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=CROP_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "frames_consumed": float(self._stats.frames_consumed),
                    "failure_rate": self._stats.failure_rate,
                    "crops_produced": float(self._stats.crops_produced),
                    "gate_rejections": float(self._stats.gate_rejections),
                },
            )
        )

    # --- helpers ------------------------------------------------------------------- #

    def _frame_context(self, update: RegistryUpdate) -> FrameContext | None:
        """Frame metadata, without holding the frame.

        Peeks at the buffer for dimensions and capture time and releases
        immediately. Trigger evaluation must not hold a lease: the whole point of
        the two-phase design is that the expensive resource is taken only for the
        few candidates that survive.
        """
        if self._buffer is None:
            return None
        lease = self._buffer.try_acquire(update.frame_ref, str(CROP_RUNTIME_ID))
        if lease is None:
            return None
        try:
            frame = lease.frame
            return FrameContext(
                frame_ref=frame.frame_ref,
                width=frame.dimensions.width,
                height=frame.dimensions.height,
                t_capture=frame.time.t_capture,
                colour_space=frame.dimensions.colour_space,
            )
        finally:
            lease.release()

    @staticmethod
    def _channels(colour_space: str) -> int:
        """Bytes per pixel for the declared colour space.

        Declared, never guessed from the buffer length: inferring channel count
        from `len(pixels) / (w*h)` would silently succeed on a truncated buffer
        and produce a crop of garbage that looks like evidence.
        """
        return 1 if colour_space in ("gray", "gray8", "mono") else 3

    # --- access ---------------------------------------------------------------------- #

    @property
    def stats(self) -> CropRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    @property
    def cameras_seen(self) -> int:
        return len(self._locks)

    def health(self) -> ComponentHealth:
        return self._manager.health()

    def forget(self, camera_id: CameraId) -> None:
        """Release a camera's lock and trigger state after it detaches.

        Without this the lock table grows with every camera the process has ever
        seen — a slow leak visible only on long-lived nodes with churning camera
        sets, which is where it is hardest to diagnose.
        """
        self._locks.pop(camera_id, None)
        self._manager.forget_camera(camera_id)

    def reconcile(self, camera_id: CameraId, live: frozenset[ObjectId]) -> int:
        """Drop trigger state for objects the registry no longer knows about."""
        return self._manager.trigger_state.partition(camera_id).retain_only(live)
