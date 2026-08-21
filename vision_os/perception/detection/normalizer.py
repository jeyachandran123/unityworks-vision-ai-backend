"""Assemble platform ``Detection`` objects from adapter output.

Single responsibility: *turn what an adapter said into what the platform
guarantees, and refuse anything that is not one.*

This is the contract-enforcement choke point of the detection layer. An adapter
declares it honours obligations D1-D8; this module **verifies** the ones that are
checkable at runtime and rejects the rest, because an adapter that quietly
returns a native label or a box outside normalized space produces results that
are structurally plausible and silently wrong (the Byzantine failure class).

It also adds everything an adapter must not be able to forge: the detection
identity, the provenance chain, the calibrated confidence, and the evidence
record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import DetectorContractError
from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.detection import (
    DecisionStep,
    Detection,
    DetectionEvidence,
    QualityGrades,
)
from ...core.model.frame import FrameDimensions
from ...core.model.ids import (
    AdapterId,
    ClassId,
    DetectionId,
    FrameRef,
    SiteId,
    TenantId,
    new_ulid,
)
from ...core.model.provenance import InferenceTiming, ModelMeta, Provenance
from ...core.model.space import Box, FrameOfReference, SpatialInfo
from ...core.model.timebase import Duration, Instant
from ...core.ports.detection import DetectionResult, NmsDeclaration, RawDetection
from ...kernel.models.calibration import CalibrationProfile
from ...taxonomy import TaxonomyRegistry

PRODUCER_MODULE = "detection_engine"
PRODUCER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """How strictly adapter output is policed, and what the platform adds."""

    confidence_threshold: float = 0.25
    max_detections: int = 300
    iou_threshold: float = 0.45
    apply_platform_nms: bool = True
    clamp_tolerance: float = 1e-3
    """Boxes escaping [0,1] by less than this are clamped and noted; anything
    larger is a contract violation, because a real letterbox-inverse bug produces
    errors of percent, not of epsilon."""


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    detections: tuple[Detection, ...]
    rejected: tuple[tuple[str, str], ...]
    """``(reason, detail)`` per rejected raw detection.

    Never a "false detection" count — whether a detection is false requires
    ground truth the platform does not have, and claiming otherwise would breach
    the Semantic Ceiling. What is countable is what the platform refused, and
    why.
    """


class DetectionNormalizer:
    """Validate adapter output and assemble platform detections."""

    def __init__(
        self,
        *,
        taxonomy: TaxonomyRegistry,
        policy: NormalizationPolicy,
        config_revision: str,
        deterministic: bool = False,
    ) -> None:
        self._taxonomy = taxonomy
        self._policy = policy
        self._config_revision = config_revision
        self._deterministic = deterministic

    def normalize(
        self,
        *,
        result: DetectionResult,
        frame_ref: FrameRef,
        dimensions: FrameDimensions,
        tenant_id: TenantId,
        site_id: SiteId,
        t_capture: Instant,
        t_capture_uncertainty: Duration,
        adapter_id: AdapterId,
        adapter_version: str,
        nms: NmsDeclaration,
        calibration: CalibrationProfile | None,
        input_hash: str,
        queued_ms: float = 0.0,
        target_classes: Sequence[ClassId] = (),
    ) -> NormalizationOutcome:
        """Assemble detections for one frame.

        Raises:
            DetectorContractError: when the adapter violated a checkable
                obligation. Rejected here rather than propagated, because a
                contract breach that reaches state is undetectable downstream.
        """
        if result.frame_ref != frame_ref:
            raise DetectorContractError(
                f"adapter returned a result for {result.frame_ref} while asked for "
                f"{frame_ref}; batch results must map 1:1 and in order (D6)",
                adapter_id=str(adapter_id),
            )

        base_path: list[DecisionStep] = []
        candidates = list(result.detections)

        candidates, threshold_rejects = self._apply_threshold(candidates)
        if threshold_rejects:
            base_path.append(DecisionStep.THRESHOLD_APPLIED)

        if nms.applied:
            base_path.append(DecisionStep.NMS_APPLIED_BY_ADAPTER)
        elif self._policy.apply_platform_nms:
            candidates, nms_rejects = self._suppress(candidates)
            if nms_rejects:
                base_path.append(DecisionStep.NMS_APPLIED_BY_PLATFORM)
                threshold_rejects = (*threshold_rejects, *nms_rejects)

        if target_classes:
            candidates, class_rejects = self._filter_classes(candidates, target_classes)
            threshold_rejects = (*threshold_rejects, *class_rejects)

        truncated = False
        if len(candidates) > self._policy.max_detections:
            overflow = len(candidates) - self._policy.max_detections
            candidates.sort(key=lambda d: d.score, reverse=True)
            candidates = candidates[: self._policy.max_detections]
            threshold_rejects = (
                *threshold_rejects,
                ("max_detections", f"{overflow} beyond limit"),
            )
            truncated = True

        base_path.append(
            DecisionStep.CALIBRATION_APPLIED
            if calibration is not None
            else DecisionStep.CALIBRATION_UNAVAILABLE
        )
        if truncated:
            base_path.append(DecisionStep.MAX_DETECTIONS_TRUNCATED)

        provenance = Provenance(
            producer_module=PRODUCER_MODULE,
            producer_version=PRODUCER_VERSION,
            config_revision=self._config_revision,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            model_id=result.model_meta.model_id,
            model_version=result.model_meta.model_version,
            model_artifact_hash=result.model_meta.artifact_hash,
            deterministic=self._deterministic,
        )
        timing = self._timing(result.timing, queued_ms)

        detections: list[Detection] = []
        rejected = list(threshold_rejects)

        for raw in candidates:
            try:
                detections.append(
                    self._build(
                        raw=raw,
                        frame_ref=frame_ref,
                        dimensions=dimensions,
                        tenant_id=tenant_id,
                        site_id=site_id,
                        t_capture=t_capture,
                        t_capture_uncertainty=t_capture_uncertainty,
                        provenance=provenance,
                        timing=timing,
                        model_meta=result.model_meta,
                        calibration=calibration,
                        input_hash=input_hash,
                        adapter_id=adapter_id,
                        decision_path=base_path,
                    )
                )
            except DetectorContractError:
                raise
            except ValueError as exc:
                rejected.append(("invalid_geometry", str(exc)))

        return NormalizationOutcome(
            detections=tuple(detections), rejected=tuple(rejected)
        )

    # --- assembly ------------------------------------------------------------- #

    def _build(
        self,
        *,
        raw: RawDetection,
        frame_ref: FrameRef,
        dimensions: FrameDimensions,
        tenant_id: TenantId,
        site_id: SiteId,
        t_capture: Instant,
        t_capture_uncertainty: Duration,
        provenance: Provenance,
        timing: InferenceTiming,
        model_meta: ModelMeta,
        calibration: CalibrationProfile | None,
        input_hash: str,
        adapter_id: AdapterId,
        decision_path: Sequence[DecisionStep],
    ) -> Detection:
        class_id = self._verify_class(raw, adapter_id)
        box, clamped = self._verify_box(raw, adapter_id)

        path = list(decision_path)
        path.append(DecisionStep.TAXONOMY_MAPPED)
        if clamped:
            path.append(DecisionStep.COORDINATES_CLAMPED)

        confidence = self._calibrate(raw.score, calibration)
        quality = self._grade(box, raw, dimensions)

        return Detection(
            detection_id=DetectionId(new_ulid(now_ms=t_capture.millis)),
            frame_ref=frame_ref,
            tenant_id=tenant_id,
            site_id=site_id,
            t_capture=t_capture,
            t_capture_uncertainty=t_capture_uncertainty,
            class_id=class_id,
            taxonomy_version=self._taxonomy.version,
            confidence=confidence,
            spatial=SpatialInfo(
                frame_of_reference=FrameOfReference.NORMALIZED, bbox=box
            ),
            provenance=provenance,
            timing=timing,
            evidence=DetectionEvidence(
                input_hash=input_hash,
                decision_path=tuple(dict.fromkeys(path)),
            ),
            quality=quality,
            class_scores=self._verify_scores(raw, adapter_id),
            geometry_kind=raw.geometry_kind,
            labels={"device": model_meta.device_id, "precision": model_meta.precision},
        )

    # --- contract verification -------------------------------------------------- #

    def _verify_class(self, raw: RawDetection, adapter_id: AdapterId) -> ClassId:
        """Obligation D2: a native label must never escape the adapter."""
        if not self._taxonomy.has(raw.class_id):
            raise DetectorContractError(
                f"adapter '{adapter_id}' emitted class '{raw.class_id}', which is not "
                f"in the platform taxonomy. A model-native label must never escape "
                f"the adapter (obligation D2).",
                adapter_id=str(adapter_id),
                class_id=str(raw.class_id),
            )
        return self._taxonomy.resolve(raw.class_id)

    def _verify_box(self, raw: RawDetection, adapter_id: AdapterId) -> tuple[Box, bool]:
        """Obligation D1: normalized coordinates with letterboxing exactly inverted.

        A small overshoot is clamped and recorded; a large one is a contract
        violation, because a genuine letterbox-inverse bug is off by percent and
        would otherwise be invisible for months while tracking quietly degrades.
        """
        box = raw.box
        if box.is_within_unit():
            return box, False

        tolerance = self._policy.clamp_tolerance
        overshoot = max(
            -box.x1, -box.y1, box.x2 - 1.0, box.y2 - 1.0, 0.0
        )
        if overshoot > tolerance:
            raise DetectorContractError(
                f"adapter '{adapter_id}' returned box {box} escaping normalized "
                f"[0,1] space by {overshoot:.4f}. Coordinates must be normalized "
                f"against the rectified source image with letterboxing exactly "
                f"inverted (obligation D1).",
                adapter_id=str(adapter_id),
                overshoot=overshoot,
            )
        return box.clamped_to_unit(), True

    def _verify_scores(
        self, raw: RawDetection, adapter_id: AdapterId
    ) -> tuple[tuple[ClassId, float], ...]:
        """Class distributions must also use platform taxonomy."""
        verified: list[tuple[ClassId, float]] = []
        for class_id, score in raw.class_scores:
            if not self._taxonomy.has(class_id):
                raise DetectorContractError(
                    f"adapter '{adapter_id}' emitted class distribution entry "
                    f"'{class_id}', which is not in the platform taxonomy (D2)",
                    adapter_id=str(adapter_id),
                )
            verified.append((self._taxonomy.resolve(class_id), score))
        return tuple(verified)

    # --- platform additions ------------------------------------------------------ #

    def _calibrate(
        self, raw_score: float, calibration: CalibrationProfile | None
    ) -> Confidence:
        """Obligation D3, plus the platform's calibration responsibility.

        ``raw_score`` is preserved either way, so a profile refitted next year can
        re-calibrate history without re-running inference.
        """
        if calibration is None:
            return Confidence.uncalibrated(
                raw_score, ConfidenceSemantics.DETECTION_PRESENCE
            )
        return Confidence(
            value=calibration.apply(raw_score),
            semantics=ConfidenceSemantics.DETECTION_PRESENCE,
            calibrated=True,
            calibration_id=calibration.calibration_id,
            raw_score=raw_score,
        )

    def _grade(
        self, box: Box, raw: RawDetection, dimensions: FrameDimensions
    ) -> QualityGrades:
        """Obligation D8: populate what is derivable at this stage.

        Occlusion, blur and crowding stay ``None`` — they need a crop or
        neighbour context and belong to Flow 4. "Not measured" and "measured as
        zero" are different claims.
        """
        return QualityGrades(
            scale_pixels=box.height * dimensions.height,
            truncation=raw.truncation if raw.truncation is not None else _truncation(box),
        )

    def _timing(self, timing: InferenceTiming, queued_ms: float) -> InferenceTiming:
        return InferenceTiming(
            queued_ms=queued_ms,
            preprocess_ms=timing.preprocess_ms,
            inference_ms=timing.inference_ms,
            postprocess_ms=timing.postprocess_ms,
            batch_size=timing.batch_size,
            device_id=timing.device_id,
            model_load_state=timing.model_load_state,
        )

    # --- filtering ----------------------------------------------------------------- #

    def _apply_threshold(
        self, candidates: list[RawDetection]
    ) -> tuple[list[RawDetection], tuple[tuple[str, str], ...]]:
        kept: list[RawDetection] = []
        rejected: list[tuple[str, str]] = []
        for raw in candidates:
            if raw.score < self._policy.confidence_threshold:
                rejected.append(("below_threshold", f"{raw.class_id}@{raw.score:.3f}"))
            else:
                kept.append(raw)
        return kept, tuple(rejected)

    def _filter_classes(
        self, candidates: list[RawDetection], target_classes: Sequence[ClassId]
    ) -> tuple[list[RawDetection], tuple[tuple[str, str], ...]]:
        kept: list[RawDetection] = []
        rejected: list[tuple[str, str]] = []
        for raw in candidates:
            if any(self._taxonomy.is_a(raw.class_id, target) for target in target_classes):
                kept.append(raw)
            else:
                rejected.append(("not_requested", str(raw.class_id)))
        return kept, tuple(rejected)

    def _suppress(
        self, candidates: list[RawDetection]
    ) -> tuple[list[RawDetection], tuple[tuple[str, str], ...]]:
        """Class-wise greedy NMS, applied only when the adapter declared it did not.

        A platform cannot correct for suppression it does not know about (D4), but
        it can act on an honest declaration that none was applied.
        """
        ordered = sorted(candidates, key=lambda d: d.score, reverse=True)
        kept: list[RawDetection] = []
        rejected: list[tuple[str, str]] = []
        for raw in ordered:
            if any(
                other.class_id == raw.class_id
                and _iou(other.box, raw.box) >= self._policy.iou_threshold
                for other in kept
            ):
                rejected.append(("nms_suppressed", str(raw.class_id)))
            else:
                kept.append(raw)
        return kept, tuple(rejected)


def _iou(first: Box, second: Box) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _truncation(box: Box) -> float:
    """How much of the box would lie outside the frame if it were not clamped.

    Zero for any box already inside, which is the common case; a positive value
    means the object continues past the frame edge and anything measured from it
    understates its true extent.
    """
    full_width = box.x2 - box.x1
    full_height = box.y2 - box.y1
    if full_width <= 0 or full_height <= 0:
        return 0.0
    visible_width = min(box.x2, 1.0) - max(box.x1, 0.0)
    visible_height = min(box.y2, 1.0) - max(box.y1, 0.0)
    visible = max(0.0, visible_width) * max(0.0, visible_height)
    total = full_width * full_height
    return max(0.0, min(1.0, 1.0 - (visible / total)))
