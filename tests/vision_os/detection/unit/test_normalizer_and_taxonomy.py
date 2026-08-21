"""The contract-enforcement choke point, and the taxonomy it enforces against.

The normalizer is where an adapter's claim to honour D1-D8 is *verified*. An
adapter that quietly returns a native label or a box outside normalized space
produces results that are structurally plausible and silently wrong — the
Byzantine failure class, and the one nothing downstream can detect.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import DetectorContractError, TaxonomyError
from vision_os.core.model.detection import DecisionStep
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import (
    AdapterId,
    CalibrationId,
    ClassId,
    ConfigRevision,
    ModelId,
    SiteId,
    TenantId,
)
from vision_os.core.model.provenance import InferenceTiming, ModelMeta
from vision_os.core.model.space import Box
from vision_os.core.model.taxonomy import (
    UNKNOWN_CLASS,
    ClassStatus,
    MappingEntry,
    TaxonomyClass,
    TaxonomyMapping,
)
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.detection import (
    DetectionResult,
    NmsDeclaration,
    RawDetection,
)
from vision_os.kernel.models.calibration import (
    CalibrationMethod,
    CalibrationProfile,
)
from vision_os.perception.detection import DetectionNormalizer, NormalizationPolicy
from vision_os.taxonomy import TaxonomyRegistry

from ..conftest import ADAPTER_ID, MODEL_ID, frame_ref

DIMENSIONS = FrameDimensions(width=640, height=480)


def _result(detections: tuple[RawDetection, ...], **overrides) -> DetectionResult:
    return DetectionResult(
        frame_ref=overrides.get("frame_ref", frame_ref()),
        detections=detections,
        model_meta=ModelMeta(
            model_id=MODEL_ID, model_version="1.0.0", artifact_hash="blake2b:abc"
        ),
        timing=InferenceTiming(inference_ms=8.0, batch_size=1),
    )


def _normalize(
    normalizer: DetectionNormalizer,
    detections: tuple[RawDetection, ...],
    *,
    nms: NmsDeclaration | None = None,
    calibration: CalibrationProfile | None = None,
    target_classes=(),
):
    return normalizer.normalize(
        result=_result(detections),
        frame_ref=frame_ref(),
        dimensions=DIMENSIONS,
        tenant_id=TenantId("acme"),
        site_id=SiteId("site-sg-01"),
        t_capture=Instant(1_000_000_000),
        t_capture_uncertainty=Duration.from_millis(10),
        adapter_id=ADAPTER_ID,
        adapter_version="1.0.0",
        nms=nms or NmsDeclaration(applied=False),
        calibration=calibration,
        input_hash="blake2b:frame",
        target_classes=target_classes,
    )


@pytest.fixture
def normalizer(taxonomy: TaxonomyRegistry) -> DetectionNormalizer:
    return DetectionNormalizer(
        taxonomy=taxonomy,
        policy=NormalizationPolicy(confidence_threshold=0.25, max_detections=10),
        config_revision=ConfigRevision("cfg-test"),
    )


class TestContractEnforcement:
    def test_native_label_is_rejected(self, normalizer: DetectionNormalizer) -> None:
        """Obligation D2 — a model-native label must never escape the adapter."""
        with pytest.raises(DetectorContractError, match="not in the platform taxonomy"):
            _normalize(
                normalizer,
                (RawDetection(ClassId("pedestrian"), Box(0.1, 0.1, 0.2, 0.2), 0.9),),
            )

    def test_grossly_out_of_range_box_is_rejected(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """A real letterbox-inverse bug is off by percent, not by epsilon."""
        with pytest.raises(DetectorContractError, match="escaping normalized"):
            _normalize(
                normalizer,
                (RawDetection(ClassId("person"), Box(0.5, 0.5, 1.4, 1.4), 0.9),),
            )

    def test_epsilon_overshoot_is_clamped_and_recorded(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """Float noise at the frame edge is normal; a percent error is not."""
        outcome = _normalize(
            normalizer,
            (RawDetection(ClassId("person"), Box(0.1, 0.1, 1.0000004, 0.9), 0.9),),
        )
        detection = outcome.detections[0]
        assert detection.spatial.bbox.is_within_unit()
        assert DecisionStep.COORDINATES_CLAMPED in detection.evidence.decision_path

    def test_mismatched_frame_ref_is_rejected(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """Obligation D6 — results must map 1:1 and in order."""
        with pytest.raises(DetectorContractError, match="1:1 and in order"):
            normalizer.normalize(
                result=_result(
                    (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.9),),
                    frame_ref=frame_ref(99),
                ),
                frame_ref=frame_ref(0),
                dimensions=DIMENSIONS,
                tenant_id=TenantId("acme"),
                site_id=SiteId("s"),
                t_capture=Instant(1),
                t_capture_uncertainty=Duration.from_millis(1),
                adapter_id=ADAPTER_ID,
                adapter_version="1.0.0",
                nms=NmsDeclaration(applied=False),
                calibration=None,
                input_hash="h",
            )

    def test_class_distribution_is_verified_too(
        self, normalizer: DetectionNormalizer
    ) -> None:
        with pytest.raises(DetectorContractError, match="class distribution"):
            _normalize(
                normalizer,
                (
                    RawDetection(
                        ClassId("person"),
                        Box(0.1, 0.1, 0.2, 0.2),
                        0.9,
                        class_scores=((ClassId("invented"), 0.4),),
                    ),
                ),
            )

    def test_native_label_never_reaches_the_detection(
        self, normalizer: DetectionNormalizer
    ) -> None:
        outcome = _normalize(
            normalizer,
            (
                RawDetection(
                    ClassId("person"),
                    Box(0.1, 0.1, 0.2, 0.2),
                    0.9,
                    native_label="person",
                ),
            ),
        )
        detection = outcome.detections[0]
        assert not hasattr(detection, "native_label")
        assert detection.class_id == "person"


class TestThresholdAndSuppression:
    def test_below_threshold_is_rejected_with_a_reason(
        self, normalizer: DetectionNormalizer
    ) -> None:
        outcome = _normalize(
            normalizer,
            (
                RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.9),
                RawDetection(ClassId("person"), Box(0.3, 0.3, 0.4, 0.4), 0.05),
            ),
        )
        assert len(outcome.detections) == 1
        assert ("below_threshold", "person@0.050") in outcome.rejected

    def test_platform_nms_runs_only_when_the_adapter_declares_none(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """A platform cannot correct for suppression it does not know about (D4)."""
        overlapping = (
            RawDetection(ClassId("person"), Box(0.1, 0.1, 0.5, 0.5), 0.9),
            RawDetection(ClassId("person"), Box(0.11, 0.11, 0.51, 0.51), 0.8),
        )
        suppressed = _normalize(normalizer, overlapping)
        assert len(suppressed.detections) == 1
        assert any(r[0] == "nms_suppressed" for r in suppressed.rejected)

        declared = _normalize(
            normalizer, overlapping, nms=NmsDeclaration(applied=True, iou_threshold=0.45)
        )
        assert len(declared.detections) == 2, (
            "double-suppressing an adapter that already applied NMS would silently "
            "halve object counts"
        )

    def test_nms_is_class_wise(self, normalizer: DetectionNormalizer) -> None:
        """Overlapping objects of different kinds are both real."""
        outcome = _normalize(
            normalizer,
            (
                RawDetection(ClassId("person"), Box(0.1, 0.1, 0.5, 0.5), 0.9),
                RawDetection(ClassId("vehicle.forklift"), Box(0.11, 0.11, 0.51, 0.51), 0.8),
            ),
        )
        assert len(outcome.detections) == 2

    def test_max_detections_truncates_by_score(
        self, taxonomy: TaxonomyRegistry
    ) -> None:
        normalizer = DetectionNormalizer(
            taxonomy=taxonomy,
            policy=NormalizationPolicy(
                confidence_threshold=0.0, max_detections=2, apply_platform_nms=False
            ),
            config_revision=ConfigRevision("cfg"),
        )
        outcome = _normalize(
            normalizer,
            tuple(
                RawDetection(
                    ClassId("person"), Box(i * 0.05, 0.1, i * 0.05 + 0.02, 0.2), 0.1 * i
                )
                for i in range(1, 8)
            ),
        )
        assert len(outcome.detections) == 2
        scores = [d.confidence.raw_score for d in outcome.detections]
        assert scores == sorted(scores, reverse=True)
        assert any(r[0] == "max_detections" for r in outcome.rejected)

    def test_target_class_filter_is_hierarchical(
        self, normalizer: DetectionNormalizer
    ) -> None:
        outcome = _normalize(
            normalizer,
            (
                RawDetection(ClassId("vehicle.forklift"), Box(0.1, 0.1, 0.2, 0.2), 0.9),
                RawDetection(ClassId("person"), Box(0.3, 0.3, 0.4, 0.4), 0.9),
            ),
            target_classes=(ClassId("vehicle"),),
        )
        assert len(outcome.detections) == 1
        assert outcome.detections[0].class_id == "vehicle.forklift"


class TestPlatformAdditions:
    def test_uncalibrated_confidence_is_declared_as_such(
        self, normalizer: DetectionNormalizer
    ) -> None:
        outcome = _normalize(
            normalizer, (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.8),)
        )
        confidence = outcome.detections[0].confidence
        assert not confidence.calibrated
        assert confidence.raw_score == 0.8
        assert confidence.value == 0.8
        assert (
            DecisionStep.CALIBRATION_UNAVAILABLE
            in outcome.detections[0].evidence.decision_path
        )

    def test_calibration_transforms_the_value_and_keeps_the_raw_score(
        self, normalizer: DetectionNormalizer
    ) -> None:
        profile = CalibrationProfile(
            calibration_id=CalibrationId("cal-1"),
            model_id=MODEL_ID,
            model_version="1.0.0",
            method=CalibrationMethod.TEMPERATURE,
            temperature=2.0,
        )
        outcome = _normalize(
            normalizer,
            (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.9),),
            calibration=profile,
        )
        confidence = outcome.detections[0].confidence
        assert confidence.calibrated
        assert confidence.calibration_id == "cal-1"
        assert confidence.raw_score == 0.9
        assert confidence.value != 0.9

    def test_quality_grades_populate_scale_and_truncation(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """Obligation D8 — what is derivable here, and nothing more."""
        outcome = _normalize(
            normalizer,
            (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
        )
        quality = outcome.detections[0].quality
        assert quality.scale_pixels == pytest.approx(0.5 * 480)
        assert quality.truncation == pytest.approx(0.0)
        assert quality.occlusion is None
        assert quality.blur is None

    def test_provenance_is_assembled_by_the_platform(
        self, normalizer: DetectionNormalizer
    ) -> None:
        """The adapter cannot forge identity or provenance."""
        outcome = _normalize(
            normalizer, (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.9),)
        )
        provenance = outcome.detections[0].provenance
        assert provenance.producer_module == "detection_engine"
        assert provenance.adapter_id == ADAPTER_ID
        assert provenance.model_artifact_hash == "blake2b:abc"
        assert provenance.config_revision == "cfg-test"

    def test_every_detection_gets_a_unique_id(
        self, normalizer: DetectionNormalizer
    ) -> None:
        outcome = _normalize(
            normalizer,
            (
                RawDetection(ClassId("person"), Box(0.1, 0.1, 0.2, 0.2), 0.9),
                RawDetection(ClassId("person"), Box(0.6, 0.6, 0.7, 0.7), 0.9),
            ),
        )
        ids = {d.detection_id for d in outcome.detections}
        assert len(ids) == 2

    def test_superseded_class_is_resolved_forward(
        self, taxonomy: TaxonomyRegistry
    ) -> None:
        """A rename must not orphan detections still using the old name."""
        taxonomy.register_class(TaxonomyClass(ClassId("cart"), taxonomy.version))
        taxonomy.deprecate(ClassId("cart"), superseded_by=ClassId("container"))
        normalizer = DetectionNormalizer(
            taxonomy=taxonomy,
            policy=NormalizationPolicy(confidence_threshold=0.0),
            config_revision=ConfigRevision("cfg"),
        )
        outcome = _normalize(
            normalizer, (RawDetection(ClassId("cart"), Box(0.1, 0.1, 0.2, 0.2), 0.9),)
        )
        assert outcome.detections[0].class_id == "container"


class TestTaxonomyRegistry:
    def test_unknown_class_always_exists(self, taxonomy: TaxonomyRegistry) -> None:
        """Without it, ``emit_as_unknown`` could not be honoured and unmapped
        detections would vanish silently."""
        assert taxonomy.has(UNKNOWN_CLASS)

    def test_orphan_class_is_rejected(self, taxonomy: TaxonomyRegistry) -> None:
        with pytest.raises(TaxonomyError, match="parent"):
            taxonomy.register_class(
                TaxonomyClass(ClassId("furniture.bed"), taxonomy.version)
            )

    def test_registration_order_does_not_matter(self) -> None:
        registry = TaxonomyRegistry()
        registry.register_classes(
            (
                TaxonomyClass(ClassId("a.b.c"), registry.version),
                TaxonomyClass(ClassId("a"), registry.version),
                TaxonomyClass(ClassId("a.b"), registry.version),
            )
        )
        assert registry.has(ClassId("a.b.c"))

    def test_hierarchical_matching(self, taxonomy: TaxonomyRegistry) -> None:
        assert taxonomy.is_a(ClassId("vehicle.forklift"), ClassId("vehicle"))
        assert not taxonomy.is_a(ClassId("person"), ClassId("vehicle"))
        assert set(taxonomy.descendants(ClassId("vehicle"))) == {
            "vehicle",
            "vehicle.forklift",
        }

    def test_mapping_with_undefined_class_is_rejected(
        self, taxonomy: TaxonomyRegistry
    ) -> None:
        """Validated at load, not at first frame."""
        with pytest.raises(TaxonomyError, match="does not define"):
            taxonomy.register_mapping(
                TaxonomyMapping(
                    adapter_id=AdapterId("bad"),
                    model_id=ModelId("m"),
                    entries=(MappingEntry("thing", ClassId("nonexistent")),),
                )
            )

    def test_a_specific_class_covers_its_ancestors(
        self, taxonomy: TaxonomyRegistry, mapping: TaxonomyMapping
    ) -> None:
        """A detector producing ``vehicle.forklift`` satisfies a demand for
        ``vehicle`` (02_VOM section 8.3 rule 1).

        This is what lets a model upgrade *increase* specificity without breaking
        an existing consumer query.
        """
        report = taxonomy.register_mapping(mapping)
        assert report.valid
        assert report.can_produce(ClassId("person"))
        assert report.can_produce(ClassId("vehicle"))
        assert report.can_produce(ClassId("container"))
        assert ClassId("vehicle") not in report.absent

    def test_coverage_reports_what_cannot_be_produced(
        self, taxonomy: TaxonomyRegistry, mapping: TaxonomyMapping
    ) -> None:
        """A capability gap must be explicit, never silence (invariant V8)."""
        taxonomy.register_class(TaxonomyClass(ClassId("furniture"), taxonomy.version))
        report = taxonomy.register_mapping(mapping)
        assert ClassId("furniture") in report.absent
        assert not report.can_produce(ClassId("furniture"))

    def test_capability_gap_is_reported(
        self, taxonomy: TaxonomyRegistry, mapping: TaxonomyMapping
    ) -> None:
        taxonomy.register_class(TaxonomyClass(ClassId("furniture"), taxonomy.version))
        taxonomy.register_mapping(mapping)
        gap = taxonomy.capability_gap(
            (ClassId("person"), ClassId("furniture"), ClassId("vehicle"))
        )
        assert ClassId("furniture") in gap
        assert ClassId("person") not in gap
        assert ClassId("vehicle") not in gap

    def test_supersession_cycle_is_detected(self, taxonomy: TaxonomyRegistry) -> None:
        taxonomy.register_class(TaxonomyClass(ClassId("x"), taxonomy.version))
        taxonomy.register_class(TaxonomyClass(ClassId("y"), taxonomy.version))
        taxonomy.deprecate(ClassId("x"), superseded_by=ClassId("y"))
        taxonomy.deprecate(ClassId("y"), superseded_by=ClassId("x"))
        with pytest.raises(TaxonomyError, match="cycle"):
            taxonomy.resolve(ClassId("x"))

    def test_deprecated_class_is_retained(self, taxonomy: TaxonomyRegistry) -> None:
        taxonomy.deprecate(ClassId("container.tray"))
        assert taxonomy.has(ClassId("container.tray"))
        assert taxonomy.get(ClassId("container.tray")).status is ClassStatus.DEPRECATED

    def test_geometry_kinds_are_enforced_per_class(
        self, taxonomy: TaxonomyRegistry
    ) -> None:
        from vision_os.core.model.taxonomy import GeometryKind

        assert taxonomy.supports_geometry(ClassId("container"), GeometryKind.MASK)
        assert not taxonomy.supports_geometry(ClassId("person"), GeometryKind.MASK)
