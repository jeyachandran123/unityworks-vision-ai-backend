"""M5 Detection Engine — find things in a frame.

> **Single responsibility:** *Nothing else — no memory, no identity, no meaning.*

The engine answers exactly one question: "what objects are visible in this
frame?" It never remembers a previous frame, never assigns a persistent identity,
never enriches an attribute, and never interprets what anything means. Those are
Flows 3, 5 and the consumer's business layer respectively.

Being memoryless is not a limitation but the property that keeps this module
trivially testable and its detector freely replaceable: frame N's result cannot
depend on frame N-1, so a swap can never carry hidden state across.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence

from ...acquisition import FrameBuffer
from ...core.errors import (
    DetectionFailedError,
    DetectionQueueFullError,
    DetectionTimeoutError,
    DetectorContractError,
    FrameUnavailableError,
    NotFoundError,
)
from ...core.model.detection import Detection, DetectionOutcome
from ...core.model.frame import Frame
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import CameraId, ClassId, FrameRef, ModuleId
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...core.ports.detection import (
    DetectionRequest,
    DetectionResult,
    DetectorCapabilities,
    FrameView,
)
from ...core.ports.scheduling import Fidelity
from ...kernel.config.schema import DetectionSection
from ...kernel.events import (
    DetectionCompleted,
    EventBus,
    PerformanceWarning,
    ThresholdExceeded,
)
from ...kernel.events import DetectionFailed as DetectionFailedEvent
from ...kernel.metrics import MetricName, MetricsEngine
from ...taxonomy import TaxonomyRegistry
from .binding import DetectorBinding
from .normalizer import DetectionNormalizer, NormalizationPolicy
from .scheduler import BatchKey, DetectionScheduler
from .worker import DeviceWorker

DETECTION_ENGINE_ID = ModuleId("detection_engine")
LEASE_HOLDER = "detection_engine"


#: Re-exported for the many call sites that import it from the engine. The type
#: itself lives in ``core.model.detection`` because it is the payload of the
#: Detection-to-Tracking handoff and a port may not name a flow-layer type.
__all__ = ["DETECTION_ENGINE_ID", "DetectionEngine", "DetectionOutcome"]


class DetectionEngine:
    """Converts admitted frames into standardized detections."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        buffer: FrameBuffer,
        camera_manager,  # CameraManager — L1, called downward
        taxonomy: TaxonomyRegistry,
        binding: DetectorBinding,
        scheduler: DetectionScheduler,
        worker: DeviceWorker,
        config: DetectionSection,
        config_revision: str,
        deterministic: bool = False,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._buffer = buffer
        self._cameras = camera_manager
        self._taxonomy = taxonomy
        self._binding = binding
        self._scheduler = scheduler
        self._worker = worker
        self._config = config
        self._normalizer = DetectionNormalizer(
            taxonomy=taxonomy,
            policy=NormalizationPolicy(
                confidence_threshold=config.confidence_threshold,
                max_detections=config.max_detections_per_frame,
                iou_threshold=config.iou_threshold,
                apply_platform_nms=config.apply_platform_nms,
            ),
            config_revision=config_revision,
            deterministic=deterministic,
        )
        self._deterministic = deterministic

    # --- public API ------------------------------------------------------------- #

    async def detect(
        self,
        frame_ref: FrameRef,
        fidelity: Fidelity | None = None,
        *,
        target_classes: Sequence[ClassId] = (),
    ) -> DetectionOutcome:
        """Detect on one admitted frame.

        Never raises. Every failure mode — an evicted frame, a dead GPU, a
        timeout, a contract violation — degrades to a failed outcome that is
        counted and published, because detection failure must never terminate the
        Vision Runtime (invariant V9).
        """
        lease = None
        try:
            lease = self._buffer.acquire(frame_ref, LEASE_HOLDER)
            return await self._detect_leased(lease.frame, fidelity, target_classes)
        except FrameUnavailableError:
            # Normal: the frame was evicted between admission and detection.
            # Counted, never escalated.
            return self._fail(frame_ref, "frame_unavailable", "transient")
        except DetectionQueueFullError as exc:
            return self._fail(frame_ref, "queue_full", "systemic", detail=exc.message)
        except DetectionTimeoutError as exc:
            self._metrics.counter(
                MetricName.DETECTION_TIMEOUTS, camera_id=str(frame_ref.camera_id)
            ).increment()
            return self._fail(frame_ref, "timeout", "transient", detail=exc.message)
        except DetectorContractError as exc:
            return self._fail(frame_ref, "contract_violation", "byzantine", detail=exc.message)
        except DetectionFailedError as exc:
            return self._fail(frame_ref, "detector_failed", "transient", detail=exc.message)
        except NotFoundError as exc:
            return self._fail(frame_ref, "camera_unknown", "persistent", detail=exc.message)
        except Exception as exc:  # noqa: BLE001 - detection never kills the runtime
            return self._fail(
                frame_ref,
                "unexpected",
                "systemic",
                detail=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if lease is not None:
                lease.release()

    async def detect_batch(
        self, frame_refs: Sequence[FrameRef], fidelity: Fidelity | None = None
    ) -> dict[FrameRef, DetectionOutcome]:
        """Detect on several frames. Each outcome is independent."""
        outcomes = await asyncio.gather(
            *(self.detect(ref, fidelity) for ref in frame_refs)
        )
        return {outcome.frame_ref: outcome for outcome in outcomes}

    def capabilities(self) -> DetectorCapabilities:
        """What the currently bound detector can produce.

        Published so a consumer asking for a class this site cannot produce gets
        an explicit capability gap rather than silence (invariant V8).
        """
        return self._binding.capabilities

    def capability_gap(self, requested: Sequence[ClassId]) -> tuple[ClassId, ...]:
        capabilities = self._binding.capabilities
        return tuple(
            class_id for class_id in requested if not capabilities.can_produce(class_id)
        )

    async def warm(self) -> None:
        await self._worker.warm()

    def health(self) -> ComponentHealth:
        worker_health = self._worker.health()
        stats = self._worker.stats
        if stats.batches and stats.failures / max(1, stats.batches) > 0.5:
            return ComponentHealth(
                component_id=DETECTION_ENGINE_ID,
                state=HealthState.DEGRADED,
                reported_at=self._clock.now(),
                detail=f"{stats.failures}/{stats.batches} batches failed",
                metrics={"failure_rate": stats.failures / max(1, stats.batches)},
            )
        return ComponentHealth(
            component_id=DETECTION_ENGINE_ID,
            state=worker_health.state,
            reported_at=self._clock.now(),
            detail=worker_health.detail,
            metrics={
                "mean_inference_ms": stats.mean_inference_ms,
                "queue_depth": float(self._scheduler.depth),
            },
        )

    # --- internals -------------------------------------------------------------- #

    async def _detect_leased(
        self,
        frame: Frame,
        fidelity: Fidelity | None,
        target_classes: Sequence[ClassId],
    ) -> DetectionOutcome:
        camera = self._cameras.get(frame.frame_ref.camera_id)
        binding = self._binding
        view = FrameView(
            frame_ref=frame.frame_ref,
            dimensions=frame.dimensions,
            pixels=frame.pixels.readonly_view(),
        )
        width, height = self._inference_size(fidelity, binding)
        request = DetectionRequest(
            target_classes=tuple(target_classes),
            min_confidence=self._config.confidence_threshold,
            max_detections=self._config.max_detections_per_frame,
            inference_width=width,
            inference_height=height,
            tier=fidelity.tier if fidelity else "primary",
            deterministic=self._deterministic,
        )

        submitted_ns = self._clock.monotonic().ns
        result = await self._scheduler.submit(
            key=BatchKey(
                model_id=str(binding.model_handle.model_id),
                model_version=binding.model_handle.version,
                precision=binding.model_handle.precision,
                inference_width=width,
                inference_height=height,
                tier=request.tier,
            ),
            frame_ref=frame.frame_ref,
            camera_id=frame.frame_ref.camera_id,
            view=view,
            request=request,
        )
        queued_ms = (self._clock.monotonic().ns - submitted_ns) / 1_000_000

        outcome = self._normalizer.normalize(
            result=result,
            frame_ref=frame.frame_ref,
            dimensions=frame.dimensions,
            tenant_id=camera.tenant_id,
            site_id=camera.site_id,
            t_capture=frame.time.t_capture,
            t_capture_uncertainty=frame.time.t_capture_uncertainty,
            adapter_id=binding.adapter_id,
            adapter_version=binding.adapter_version,
            nms=binding.capabilities.nms,
            calibration=binding.calibration,
            input_hash=_input_hash(view),
            queued_ms=queued_ms,
            target_classes=target_classes,
        )

        self._record(frame, result, outcome.detections, outcome.rejected, queued_ms)
        return DetectionOutcome(
            frame_ref=frame.frame_ref,
            detections=outcome.detections,
            rejected=outcome.rejected,
        )

    def _inference_size(
        self, fidelity: Fidelity | None, binding: DetectorBinding
    ) -> tuple[int, int]:
        """Honour the scheduler's fidelity tier when dynamic resolution is on.

        Lowering resolution under pressure changes *what the platform can see* —
        small and distant objects disappear — so the tier travels on the request
        and onto the result rather than being applied invisibly.
        """
        constraints = binding.capabilities.input_constraints
        if fidelity is not None and self._config.dynamic_resolution:
            return (
                max(constraints.min_width, fidelity.inference_width),
                max(constraints.min_height, fidelity.inference_height),
            )
        return (constraints.max_width, constraints.max_height)

    def _record(
        self,
        frame: Frame,
        result: DetectionResult,
        detections: Sequence[Detection],
        rejected: Sequence[tuple[str, str]],
        queued_ms: float,
    ) -> None:
        camera_id = frame.frame_ref.camera_id
        label = str(camera_id)

        self._metrics.counter(
            MetricName.DETECTION_FRAMES_PROCESSED, camera_id=label
        ).increment()
        self._metrics.counter(
            MetricName.DETECTIONS_EMITTED, camera_id=label
        ).increment(len(detections))
        self._metrics.histogram(
            MetricName.DETECTION_INFERENCE_MS, camera_id=label
        ).record(result.timing.inference_ms)
        self._metrics.histogram(MetricName.DETECTION_QUEUE_MS, camera_id=label).record(
            queued_ms
        )
        self._metrics.histogram(MetricName.DETECTION_BATCH_SIZE).record(
            result.timing.batch_size
        )
        self._metrics.gauge(MetricName.DETECTION_QUEUE_DEPTH).set(self._scheduler.depth)

        for reason, _ in rejected:
            self._metrics.counter(
                MetricName.DETECTIONS_REJECTED, camera_id=label, reason=reason
            ).increment()

        for detection in detections:
            self._metrics.counter(
                MetricName.DETECTIONS_EMITTED,
                camera_id=label,
                class_id=str(detection.class_id),
            ).increment()

        self._bus.publish(
            DetectionCompleted(
                occurred_at=self._clock.now(),
                partition_key=label,
                camera_id=camera_id,
                frame_ref=str(frame.frame_ref),
                detection_count=len(detections),
                inference_ms=result.timing.inference_ms,
                batch_size=result.timing.batch_size,
                model_id=str(result.model_meta.model_id),
            )
        )

        if result.timing.inference_ms > self._config.slow_inference_warn_ms:
            self._bus.publish(
                PerformanceWarning(
                    occurred_at=self._clock.now(),
                    partition_key=label,
                    camera_id=camera_id,
                    metric="inference_ms",
                    observed=result.timing.inference_ms,
                    expected=float(self._config.slow_inference_warn_ms),
                )
            )

        if len(detections) >= self._config.max_detections_per_frame:
            self._bus.publish(
                ThresholdExceeded(
                    occurred_at=self._clock.now(),
                    partition_key=label,
                    camera_id=camera_id,
                    threshold_name="max_detections_per_frame",
                    limit=float(self._config.max_detections_per_frame),
                    observed=float(len(detections)),
                )
            )

    def _fail(
        self, frame_ref: FrameRef, reason: str, failure_class: str, *, detail: str = ""
    ) -> DetectionOutcome:
        camera_id: CameraId = frame_ref.camera_id
        self._metrics.counter(
            MetricName.DETECTION_FAILURES, camera_id=str(camera_id), reason=reason
        ).increment()
        self._bus.publish(
            DetectionFailedEvent(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                frame_ref=str(frame_ref),
                reason=reason,
                failure_class=failure_class,
            )
        )
        return DetectionOutcome(
            frame_ref=frame_ref,
            detections=(),
            failed=True,
            reason=detail or reason,
        )

    def stale_after(self) -> Duration:
        return Duration.from_millis(self._config.inference_timeout_ms)


def _input_hash(view: FrameView) -> str:
    """Hash of the exact pixels the model saw.

    Stride-sampled: a full hash at 3000 frames a second would cost more than the
    explainability is worth, and a sampled digest is sufficient to prove which
    image produced which result.
    """
    data = view.pixels
    step = max(1, len(data) // 4096)
    digest = hashlib.blake2b(bytes(data[::step]), digest_size=16)
    digest.update(str(view.frame_ref).encode("utf-8"))
    return f"blake2b:{digest.hexdigest()}"

