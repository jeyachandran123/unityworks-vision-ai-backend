"""The Tracking Runtime — the ``DetectionConsumer`` seam, one actor per camera.

> **Single responsibility:** *Own the tracking layer's lifecycle and serialize
> each camera's frames. Track nothing yourself.*

Tracking is **strictly sequential per camera** (08_RUNTIME section 3.2, port
obligation T1): frame N's association depends on frame N−1's state. This runtime
is what makes that true regardless of how the caller behaves — each camera gets
its own lock, so two frames from one camera can never interleave, while two
cameras never contend at all.

That per-camera-lock shape is exactly the actor model the architecture specifies
for M6: *"Each camera's tracker is an actor processing its frames in order.
There is no cross-camera state, so cameras run fully in parallel with zero
contention."*

**Backpressure, not dropping.** 08_RUNTIME section 5.2 specifies the
Detection-to-Tracking edge as ``block`` — *"ordering matters; dropping here
corrupts tracks"*. So ``on_detected`` awaits its turn rather than shedding, and
the wait propagates back through detection to the frame scheduler, which is the
component whose job it is to shed.
"""

from __future__ import annotations

import asyncio

from ...core.model.detection import DetectionOutcome
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import CameraId, ModuleId
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...kernel.config.schema import TrackingSection
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from .engine import TrackingEngine, TrackingOutcome

TRACKING_RUNTIME_ID = ModuleId("tracking_runtime")

_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)


class TrackingRuntimeStats:
    """Mutable counters. Plain object rather than a dataclass because it is
    updated on the hot path and never published as a value."""

    __slots__ = (
        "frames_consumed",
        "frames_failed",
        "frames_timed_out",
        "frames_tracked",
        "sink_failures",
        "tracks_emitted",
    )

    def __init__(self) -> None:
        self.frames_consumed = 0
        self.frames_tracked = 0
        self.frames_failed = 0
        self.frames_timed_out = 0
        self.tracks_emitted = 0
        self.sink_failures = 0

    @property
    def failure_rate(self) -> float:
        return self.frames_failed / self.frames_consumed if self.frames_consumed else 0.0


class TrackingRuntime:
    """Implements ``DetectionConsumer``; owns tracking lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        health: HealthMonitor,
        engine: TrackingEngine,
        config: TrackingSection,
        sink=None,
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._health = health
        self._engine = engine
        self._config = config
        self._sink = sink
        self._report_interval = report_interval

        self._stats = TrackingRuntimeStats()
        self._locks: dict[CameraId, asyncio.Lock] = {}
        self._started = False
        self._last_report_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        self._started = True
        self._report_health(HealthState.HEALTHY, "tracking ready")

    async def stop(self) -> None:
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seam --------------------------------------------------------------- #

    async def on_detected(self, outcome: DetectionOutcome) -> None:
        """Consume one frame's detections. **Never raises.**

        A tracking failure may not stop detection, which may not stop
        acquisition (invariant V9). Everything is absorbed, counted, and
        reported through health.
        """
        if not self._started:
            return

        camera_id = outcome.frame_ref.camera_id
        self._stats.frames_consumed += 1

        # One lock per camera: serialization where it is required (T1), zero
        # contention where it is not.
        lock = self._locks.get(camera_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[camera_id] = lock

        try:
            # Backpressure here is blocking by design, so the wait must be
            # bounded: a tracker wedged on one frame would otherwise stall its
            # camera's whole pipeline indefinitely rather than degrade.
            async with asyncio.timeout(self._config.frame_timeout_ms / 1000):
                async with lock:
                    result = self._engine.track(outcome)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._stats.frames_failed += 1
            self._stats.frames_timed_out += 1
            self._metrics.counter(
                MetricName.TRACKING_FAILURES,
                camera_id=str(camera_id),
                reason="timeout",
            ).increment()
            self._report_health(
                HealthState.DEGRADED,
                f"tracking exceeded {self._config.frame_timeout_ms}ms on {camera_id}",
            )
            return
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._stats.frames_failed += 1
            self._metrics.counter(
                MetricName.TRACKING_FAILURES,
                camera_id=str(camera_id),
                reason="runtime_guard",
            ).increment()
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        self._publish(result)

    def _publish(self, result: TrackingOutcome) -> None:
        if result.failed:
            self._stats.frames_failed += 1
        else:
            self._stats.frames_tracked += 1
            self._stats.tracks_emitted += result.count

        if self._sink is not None:
            try:
                self._sink(result)
            except Exception:  # noqa: BLE001 - a bad sink must not break tracking
                self._stats.sink_failures += 1

        self._maybe_report()

    # --- health ------------------------------------------------------------------ #

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._engine.health()
        self._report_health(health.state, health.detail)

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=TRACKING_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "frames_consumed": float(self._stats.frames_consumed),
                    "failure_rate": self._stats.failure_rate,
                },
            )
        )

    # --- access -------------------------------------------------------------------- #

    @property
    def stats(self) -> TrackingRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    @property
    def cameras_seen(self) -> int:
        return len(self._locks)

    def health(self) -> ComponentHealth:
        return self._engine.health()

    def forget(self, camera_id: CameraId) -> None:
        """Release a camera's lock after it detaches.

        Without this the lock table grows with every camera the process has ever
        seen — a slow leak that only shows up on long-lived nodes with churning
        camera sets, which is precisely where it is hardest to diagnose.
        """
        self._locks.pop(camera_id, None)
