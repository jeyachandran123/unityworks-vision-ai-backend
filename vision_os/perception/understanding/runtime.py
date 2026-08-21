"""The Understanding Runtime — the ``CropConsumer`` seam and the batch worker.

> **Single responsibility:** *Own the engine's lifecycle, batch compatible
> requests, and take the lease. Understand nothing yourself.*

08_RUNTIME §1 assigns M9 a **batch coordinator + device worker**, and §M9
explains why the shape differs from every other module's:

> *VLM calls are long (100 ms - 2 s), so the module is concurrency-bound rather
> than throughput-bound.*

Two consequences this runtime makes real.

**Requests group by `(model, prompt_version)`.** Only compatible requests batch
together. Two prompts are two questions, and answering one while attributing it
to both is fabrication with extra steps.

**Nothing here blocks the layer beneath.** 08_RUNTIME §5.2 gives the
crop-to-understanding queue `drop_oldest` because *"losing an enrichment is
acceptable"* — the asymmetry against `Builder → State`'s `block` is, in that
document's words, *"the whole philosophy of the platform expressed as queue
configuration."*
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from ...core.errors import FrameUnavailableError
from ...core.model.crop import Crop, EvaluationResult
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import (
    AttributeKey,
    CameraId,
    ModuleId,
    RequestId,
    new_ulid,
)
from ...core.model.timebase import Duration
from ...core.model.understanding import (
    UnderstandingOutcome,
    UnderstandingRequest,
    UnderstandingResult,
)
from ...core.ports.clock import Clock
from ...kernel.config.schema import UnderstandingSection
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine
from .engine import UnderstandingEngine

UNDERSTANDING_RUNTIME_ID = ModuleId("understanding_runtime")

_DEFAULT_REPORT_INTERVAL = Duration.from_millis(1_000)


@dataclass(slots=True)
class _Pending:
    """One crop waiting for a batch slot."""

    request: UnderstandingRequest
    crops: tuple[Crop, ...]
    enqueued_ns: int


class UnderstandingRuntimeStats:
    """Mutable counters. Updated on the hot path, never published as a value."""

    __slots__ = (
        "attributes_produced",
        "batches_run",
        "crops_consumed",
        "dropped_on_overflow",
        "frames_consumed",
        "frames_failed",
        "requests_made",
        "results_failed",
        "results_produced",
        "sink_failures",
    )

    def __init__(self) -> None:
        self.frames_consumed = 0
        self.frames_failed = 0
        self.crops_consumed = 0
        self.requests_made = 0
        self.results_produced = 0
        self.results_failed = 0
        self.attributes_produced = 0
        self.batches_run = 0
        self.dropped_on_overflow = 0
        self.sink_failures = 0

    @property
    def failure_rate(self) -> float:
        return self.results_failed / self.results_produced if self.results_produced else 0.0


class UnderstandingRuntime:
    """Implements the cropping-to-understanding seam; owns engine lifecycle."""

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        health: HealthMonitor,
        engine: UnderstandingEngine,
        config: UnderstandingSection,
        sink=None,
        queue_capacity: int = 64,
        report_interval: Duration = _DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._health = health
        self._engine = engine
        self._config = config
        self._sink = sink
        self._report_interval = report_interval

        self._stats = UnderstandingRuntimeStats()
        self._queue: deque[_Pending] = deque(maxlen=max(1, queue_capacity))
        self._lock = asyncio.Lock()
        self._started = False
        self._last_report_ns = 0

    # --- lifecycle -------------------------------------------------------------- #

    async def start(self) -> None:
        self._started = True
        self._report_health(HealthState.HEALTHY, "understanding ready")

    async def stop(self) -> None:
        """Drain before shutting down.

        A clean shutdown that discarded queued crops would make the last few
        seconds of a run depend on when the operator pressed stop.
        """
        await self.drain()
        self._started = False
        self._report_health(HealthState.DRAINING, "stopped")

    # --- the seam ---------------------------------------------------------------- #

    async def on_crops(
        self, result: EvaluationResult, crops: Sequence[Crop]
    ) -> None:
        """Consume one camera's crops from M8. **Never raises.**

        An understanding failure may not stop the Crop Manager, which may not stop
        the registry, which may not stop tracking, which may not stop detection,
        which may not stop acquisition (V9).
        """
        if not self._started or not self._config.enabled:
            return

        self._stats.frames_consumed += 1
        if not crops:
            # Not an error: M8 correctly produced nothing, and the skips it
            # recorded already explain why (V8). Nothing to understand.
            return

        try:
            async with self._lock:
                self._enqueue(result, crops)
                await self._run_ready()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the seam is a firewall
            self._stats.frames_failed += 1
            self._metrics.counter(
                MetricName.UNDERSTANDING_FAILURES,
                camera_id=str(result.camera_id),
                reason="runtime_guard",
            ).increment()
            self._report_health(
                HealthState.DEGRADED, f"unhandled {type(exc).__name__}: {exc}"
            )
            return

        self._maybe_report()

    def _enqueue(self, result: EvaluationResult, crops: Sequence[Crop]) -> None:
        """Queue crops for understanding, dropping the oldest under pressure.

        `drop_oldest` per 08_RUNTIME §5.2, and **counted** per §5.1: *"overflow
        signal: counter + event — ALWAYS, NEVER silent."* A dropped crop is a lost
        enrichment the platform admits to, not one it hides.
        """
        by_object = {
            request.object_id: request
            for request in result.requests
            if request.object_id is not None
        }
        now_ns = self._clock.monotonic().ns

        for crop in crops:
            source = by_object.get(crop.object_id)
            if source is None:
                continue
            if len(self._queue) == self._queue.maxlen:
                self._stats.dropped_on_overflow += 1
                self._metrics.counter(
                    MetricName.UNDERSTANDING_QUEUE_DEPTH,
                    camera_id=str(result.camera_id),
                    event="overflow",
                ).increment()
            self._queue.append(
                _Pending(
                    request=self._request_for(crop, source),
                    crops=(crop,),
                    enqueued_ns=now_ns,
                )
            )
            self._stats.crops_consumed += 1

    def _request_for(self, crop: Crop, source) -> UnderstandingRequest:
        """Build the engine's request from a crop and the demand that caused it.

        Note what is carried over verbatim: the trigger reason, the attribute set
        and the demand ids. M9 never re-derives any of them — the reason the
        platform looked was M8's decision, and reproducing it here would create a
        second answer to a question already settled.
        """
        return UnderstandingRequest(
            request_id=RequestId(new_ulid(now_ms=self._clock.now().ns // 1_000_000)),
            tenant_id=crop.tenant_id,
            site_id=crop.site_id,
            camera_id=crop.camera_id,
            object_id=crop.object_id,
            class_id=source.class_id,
            crop_ids=(crop.crop_id,),
            frame_ref=crop.source_frame,
            requested_attributes=tuple(
                AttributeKey(key) for key in source.required_attributes
            ),
            trigger_reason=crop.trigger_reason,
            t_capture=crop.t_capture,
            quality=crop.quality,
            priority_class=source.priority_class,
            demand_ids=source.demand_ids,
        )

    async def _run_ready(self) -> None:
        """Run whatever is queued, in compatible batches."""
        if not self._queue:
            return
        pending = list(self._queue)
        self._queue.clear()

        requests = [item.request for item in pending]
        crops = {item.request.request_id: item.crops for item in pending}
        groups = self._engine.plan_batches(requests)
        self._stats.batches_run += len(groups)
        if groups:
            self._metrics.histogram(MetricName.UNDERSTANDING_BATCH_SIZE).record(
                sum(group.size for group in groups) / len(groups)
            )

        self._stats.requests_made += len(requests)
        results = self._engine.understand_batch(requests, crops=crops)
        self._publish(tuple(results.values()))

    def _publish(self, results: Sequence[UnderstandingResult]) -> None:
        for result in results:
            self._stats.results_produced += 1
            if result.outcome.is_failure:
                self._stats.results_failed += 1
            self._stats.attributes_produced += len(result.attributes)
            self._metrics.counter(
                MetricName.UNDERSTANDING_COST_UNITS, camera_id=str(result.camera_id)
            ).increment(int(result.cost_units))

        if self._sink is None:
            return
        try:
            # The control-plane form. `01_LAYERED` §3.2 sizes this edge at ~3 KB:
            # structured claims plus a raw-output *reference*, never the bytes.
            self._sink(tuple(r.without_raw_output() for r in results))
        except Exception:  # noqa: BLE001 - a bad sink must not break understanding
            self._stats.sink_failures += 1

    async def drain(self) -> None:
        """Run everything queued. Used at shutdown and by tests."""
        async with self._lock:
            await self._run_ready()

    # --- observability ------------------------------------------------------------ #

    def _maybe_report(self) -> None:
        now = self._clock.monotonic().ns
        if now - self._last_report_ns < self._report_interval.ns:
            return
        self._last_report_ns = now
        health = self._engine.health()
        self._report_health(health.state, health.detail)

        stats = self._engine.cache.stats()
        self._metrics.gauge(MetricName.UNDERSTANDING_CACHE_EVICTIONS).set(
            float(stats.evictions)
        )
        self._metrics.gauge(MetricName.UNDERSTANDING_QUEUE_DEPTH).set(
            float(len(self._queue))
        )

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=UNDERSTANDING_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
                metrics={
                    "crops_consumed": float(self._stats.crops_consumed),
                    "results_produced": float(self._stats.results_produced),
                    "failure_rate": self._stats.failure_rate,
                    "dropped_on_overflow": float(self._stats.dropped_on_overflow),
                },
            )
        )

    # --- access ---------------------------------------------------------------------- #

    @property
    def stats(self) -> UnderstandingRuntimeStats:
        return self._stats

    @property
    def started(self) -> bool:
        return self._started

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def health(self) -> ComponentHealth:
        return self._engine.health()

    @property
    def engine(self) -> UnderstandingEngine:
        return self._engine


@dataclass(frozen=True, slots=True)
class UnderstandingBatchReport:
    """What one drain produced. For operators and tests.

    Deliberately counts outcomes rather than summarising them into a single
    "success" number: 04_MODULES §M9's six outcomes are six different situations,
    and collapsing them is how a platform stops being able to explain itself.
    """

    results: tuple[UnderstandingResult, ...] = ()
    by_outcome: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, results: Sequence[UnderstandingResult]) -> UnderstandingBatchReport:
        counts: dict[str, int] = {}
        for result in results:
            counts[result.outcome.value] = counts.get(result.outcome.value, 0) + 1
        return cls(results=tuple(results), by_outcome=counts)

    @property
    def attributes(self) -> int:
        return sum(len(result.attributes) for result in self.results)

    @property
    def succeeded(self) -> int:
        return self.by_outcome.get(UnderstandingOutcome.SUCCEEDED.value, 0)


def frame_unavailable(camera_id: CameraId) -> FrameUnavailableError:
    """The typed error for a crop whose pixels went away before understanding.

    Kept here rather than raised inline so the message says the same thing every
    time it is produced.
    """
    return FrameUnavailableError(
        f"crop pixels for camera '{camera_id}' were released before understanding "
        f"ran; the crop's lease is held by M8 and understanding is asynchronous",
        camera_id=str(camera_id),
    )
