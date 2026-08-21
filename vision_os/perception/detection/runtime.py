"""The Detection Runtime — the seam consumer.

Single responsibility: *own the detection layer's lifecycle and resume the
admitted-frame path. Detect nothing yourself.*

This is the only Flow 2 module the Flow 1 Runtime ever holds, and it holds it as
an ``AdmittedFrameConsumer`` protocol — so Flow 1 never learns that detection
exists, let alone which detector is bound.

**Two distinct downstream paths, deliberately not the same mechanism:**

* ``DetectionCompleted`` / ``DetectionFailed`` go to the **Event Bus**, for
  observability. The bus is bounded and lossy by design, which is correct for
  notification and wrong for pipeline data.
* The next pipeline stage receives every ``DetectionOutcome`` through the
  ``DetectionConsumer`` seam — a direct sideways-within-layer handoff
  (``01_LAYERED`` section 2.1) with backpressure rather than dropping
  (``08_RUNTIME`` section 5.2, *"ordering matters; dropping here corrupts
  tracks"*).

The runtime holds the consumer as a protocol and never learns what implements it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...core.model.detection import Detection, DetectionOutcome
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import FrameRef, ModuleId
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...core.ports.pipeline import DetectionConsumer
from ...core.ports.scheduling import Fidelity
from ...kernel.events import EventBus
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from .engine import DetectionEngine

DETECTION_RUNTIME_ID = ModuleId("detection_runtime")

#: How often the runtime re-reports engine health.
_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)

DetectionSink = Callable[[Sequence[Detection]], None]
"""A synchronous tap on the *non-empty* detection stream.

For diagnostics and co-located inspection only. It is **not** the pipeline seam:
it never fires on empty or failed frames, so a stage that needs to age state must
use ``DetectionConsumer`` instead.
"""


@dataclass(slots=True)
class DetectionRuntimeStats:
    frames_consumed: int = 0
    frames_detected: int = 0
    frames_failed: int = 0
    detections_emitted: int = 0
    consumer_failures: int = 0
    """Downstream-seam faults. Counted here rather than swallowed, so a broken
    consumer is visible without being able to stop detection."""

    @property
    def failure_rate(self) -> float:
        return self.frames_failed / self.frames_consumed if self.frames_consumed else 0.0


class DetectionRuntime:
    """Implements ``AdmittedFrameConsumer``; owns detection lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        engine: DetectionEngine,
        sink: DetectionSink | None = None,
        consumer: DetectionConsumer | None = None,
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._health = health
        self._engine = engine
        self._sink = sink
        # The documented pipeline seam. ``None`` is the Flow 2 behaviour: an
        # outcome is published and dropped. The runtime holds a protocol and
        # never learns what implements it.
        self._consumer = consumer
        self._report_interval = report_interval
        self._stats = DetectionRuntimeStats()
        self._started = False
        self._last_report_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        """Warm the detector and report readiness.

        Warmup happens here rather than lazily on the first frame, so a cold
        first inference never masquerades as a performance regression.
        """
        started = self._clock.monotonic().ns
        await self._engine.warm()
        warmup_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._metrics.histogram(MetricName.DETECTOR_WARMUP_MS).record(warmup_ms)
        self._started = True
        self._report_health(HealthState.HEALTHY, "warm")

    async def stop(self) -> None:
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seam ---------------------------------------------------------------- #

    async def on_admitted(self, frame_ref: FrameRef, fidelity: Fidelity) -> None:
        """Resume the pipeline after admission.

        **Never raises.** A detection failure may not terminate a source actor or
        the Vision Runtime, so everything is absorbed, counted, and published
        (invariant V9).
        """
        if not self._started:
            return
        self._stats.frames_consumed += 1
        try:
            outcome = await self._engine.detect(frame_ref, fidelity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._stats.frames_failed += 1
            self._metrics.counter(
                MetricName.DETECTION_FAILURES,
                camera_id=str(frame_ref.camera_id),
                reason="runtime_guard",
            ).increment()
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        await self._publish(outcome)

    async def _publish(self, outcome: DetectionOutcome) -> None:
        if outcome.failed:
            self._stats.frames_failed += 1
        else:
            self._stats.frames_detected += 1
            self._stats.detections_emitted += outcome.count

        # The engine already published DetectionCompleted / DetectionFailed on the
        # bus for observability. The diagnostic sink fires only on non-empty
        # frames, by contract.
        if self._sink is not None and outcome.detections:
            try:
                self._sink(outcome.detections)
            except Exception:  # noqa: BLE001, S110 - a bad sink must not break detection
                pass

        # The pipeline seam. Awaited inline so backpressure reaches the scheduler
        # and per-camera frame order is preserved (08_RUNTIME section 5.2).
        # Fires on EVERY outcome — empty and failed included — because that is
        # when a downstream stage ages or invalidates its state.
        if self._consumer is not None:
            try:
                await self._consumer.on_detected(outcome)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the seam is a firewall
                self._stats.consumer_failures += 1
                self._metrics.counter(
                    MetricName.PIPELINE_CONSUMER_FAILURES,
                    camera_id=str(outcome.frame_ref.camera_id),
                ).increment()

        self._maybe_report()

    # --- health ------------------------------------------------------------------- #

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._engine.health()
        self._report_health(health.state, health.detail)
        self._metrics.gauge(MetricName.DETECTION_IN_FLIGHT).set(
            float(self._stats.frames_consumed - self._stats.frames_detected
                  - self._stats.frames_failed)
        )

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=DETECTION_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "frames_consumed": float(self._stats.frames_consumed),
                    "failure_rate": self._stats.failure_rate,
                },
            )
        )

    @property
    def stats(self) -> DetectionRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    def health(self) -> ComponentHealth:
        return self._engine.health()
