"""The Flow 2 object model — Detection, Confidence, Provenance, taxonomy.

These are the contracts every future detector and every future consumer
integrates against. The tests that matter most defend properties whose violation
is silent: a box that escapes normalized space, a confidence compared across
incomparable models, a result with no provenance.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.detection import (
    DecisionStep,
    Detection,
    DetectionEvidence,
    QualityGrades,
)
from vision_os.core.model.ids import (
    AdapterId,
    CalibrationId,
    ClassId,
    ConfigRevision,
    DetectionId,
    ModelId,
    ModuleId,
    SiteId,
    TenantId,
)
from vision_os.core.model.provenance import InferenceTiming, ModelMeta, Provenance
from vision_os.core.model.space import Box, FrameOfReference, SpatialInfo
from vision_os.core.model.taxonomy import (
    ClassStatus,
    MappingEntry,
    TaxonomyClass,
    TaxonomyMapping,
    UnmappedPolicy,
)
from vision_os.core.model.timebase import Duration, Instant

from ..conftest import ADAPTER_ID, MODEL_ID, frame_ref


def _provenance(**overrides) -> Provenance:
    defaults = {
        "producer_module": ModuleId("detection_engine"),
        "producer_version": "1.0.0",
        "config_revision": ConfigRevision("cfg-abc"),
        "adapter_id": ADAPTER_ID,
        "adapter_version": "1.0.0",
        "model_id": MODEL_ID,
        "model_version": "1.0.0",
        "model_artifact_hash": "blake2b:deadbeef",
    }
    defaults.update(overrides)
    return Provenance(**defaults)


def _detection(**overrides) -> Detection:
    defaults = {
        "detection_id": DetectionId("01JQ8F3K2P7XN4V9WBHZ3TDCE1"),
        "frame_ref": frame_ref(),
        "tenant_id": TenantId("acme"),
        "site_id": SiteId("site-sg-01"),
        "t_capture": Instant(1_000_000_000),
        "t_capture_uncertainty": Duration.from_millis(12),
        "class_id": ClassId("person"),
        "taxonomy_version": "1.0.0",
        "confidence": Confidence.uncalibrated(
            0.91, ConfidenceSemantics.DETECTION_PRESENCE
        ),
        "spatial": SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED,
            bbox=Box(0.31, 0.44, 0.38, 0.71),
        ),
        "provenance": _provenance(),
        "timing": InferenceTiming(inference_ms=14.0, batch_size=4),
        "evidence": DetectionEvidence(input_hash="blake2b:cafe"),
    }
    defaults.update(overrides)
    return Detection(**defaults)


class TestConfidence:
    def test_raw_score_is_always_preserved(self) -> None:
        """A profile refitted next year must re-calibrate history without re-running
        inference — which is only possible if the raw score survived."""
        confidence = Confidence.uncalibrated(0.87, ConfidenceSemantics.DETECTION_PRESENCE)
        assert confidence.raw_score == 0.87
        assert not confidence.calibrated

    def test_calibrated_confidence_must_name_its_profile(self) -> None:
        with pytest.raises(ValueError, match="must name the calibration"):
            Confidence(
                value=0.9,
                semantics=ConfidenceSemantics.DETECTION_PRESENCE,
                calibrated=True,
            )

    def test_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            Confidence(value=1.4, semantics=ConfidenceSemantics.DETECTION_PRESENCE)

    def test_uncalibrated_scores_are_not_comparable(self) -> None:
        """The rule that keeps a 2029 detector from silently changing a 2026
        consumer's threshold behaviour."""
        first = Confidence.uncalibrated(0.8, ConfidenceSemantics.DETECTION_PRESENCE)
        second = Confidence.uncalibrated(0.7, ConfidenceSemantics.DETECTION_PRESENCE)
        assert not first.comparable_with(second)

    def test_same_profile_is_comparable(self) -> None:
        profile = CalibrationId("cal-person-2026Q2")
        first = Confidence(0.8, ConfidenceSemantics.DETECTION_PRESENCE, True, profile, 0.7)
        second = Confidence(0.6, ConfidenceSemantics.DETECTION_PRESENCE, True, profile, 0.5)
        assert first.comparable_with(second)

    def test_different_profiles_are_not_comparable(self) -> None:
        first = Confidence(
            0.8, ConfidenceSemantics.DETECTION_PRESENCE, True, CalibrationId("a"), 0.7
        )
        second = Confidence(
            0.6, ConfidenceSemantics.DETECTION_PRESENCE, True, CalibrationId("b"), 0.5
        )
        assert not first.comparable_with(second)

    def test_self_reported_is_never_comparable(self) -> None:
        """A model's opinion about itself is not a probability."""
        assert not ConfidenceSemantics.SELF_REPORTED.is_comparable_across_models
        assert ConfidenceSemantics.DETECTION_PRESENCE.is_comparable_across_models


class TestProvenance:
    def test_config_revision_is_mandatory(self) -> None:
        """Without it, no result is reproducible six months later."""
        with pytest.raises(ValueError, match="config_revision"):
            _provenance(config_revision=ConfigRevision(""))

    def test_naming_a_model_without_its_hash_is_rejected(self) -> None:
        """The exact weights that produced a result are mandatory (invariant V4)."""
        with pytest.raises(ValueError, match="artifact hash"):
            _provenance(model_artifact_hash=None)

    def test_detector_name_and_version_are_surfaced(self) -> None:
        provenance = _provenance()
        assert provenance.detector_name == "detector.reference"
        assert provenance.detector_version == "1.0.0"

    def test_timing_totals(self) -> None:
        timing = InferenceTiming(
            queued_ms=1.0, preprocess_ms=2.0, inference_ms=10.0, postprocess_ms=0.5
        )
        assert timing.total_ms == pytest.approx(13.5)


class TestDetectionContract:
    def test_carries_every_required_field(self) -> None:
        """The standardized contract every detector must emit."""
        detection = _detection()
        assert detection.detection_id
        assert detection.frame_ref
        assert detection.t_capture
        assert detection.spatial.bbox is not None
        assert detection.confidence.value
        assert detection.class_id
        assert detection.detector_name == "detector.reference"
        assert detection.detector_version == "1.0.0"
        assert detection.inference_ms == 14.0
        assert detection.coordinate_space == "normalized"
        assert detection.evidence.input_hash
        assert detection.taxonomy_version

    def test_rejects_a_box_outside_normalized_space(self) -> None:
        """Obligation D1, enforced by the type rather than by review."""
        with pytest.raises(ValueError, match="escapes normalized"):
            _detection(
                spatial=SpatialInfo(
                    frame_of_reference=FrameOfReference.NORMALIZED,
                    bbox=Box(0.5, 0.5, 1.4, 1.4),
                )
            )

    def test_rejects_wrong_confidence_semantics(self) -> None:
        """A detection asserts presence, not classification or identity (D3)."""
        with pytest.raises(ValueError, match="DETECTION_PRESENCE"):
            _detection(
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.CLASSIFICATION
                )
            )

    def test_requires_a_bounding_box(self) -> None:
        with pytest.raises(ValueError, match="bounding box"):
            _detection(
                spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED)
            )

    def test_requires_a_taxonomy_version(self) -> None:
        with pytest.raises(ValueError, match="taxonomy version"):
            _detection(taxonomy_version="")

    def test_holds_no_identity_or_temporal_state(self) -> None:
        """Detection is memoryless: no track, no object, no history.

        The structural boundary between Flow 2 and Flow 3.
        """
        fields = set(Detection.__dataclass_fields__)
        for forbidden in ("track_id", "object_id", "previous", "history", "age"):
            assert forbidden not in fields, (
                f"Detection carries '{forbidden}'; identity and temporal state "
                f"belong to Flow 3"
            )

    def test_hierarchical_class_matching(self) -> None:
        detection = _detection(class_id=ClassId("vehicle.forklift"))
        assert detection.is_a(ClassId("vehicle"))
        assert detection.is_a(ClassId("vehicle.forklift"))
        assert not detection.is_a(ClassId("person"))

    def test_class_scores_are_retained(self) -> None:
        """Retained so class flapping can later be resolved by distribution
        rather than a majority vote over discarded information."""
        detection = _detection(
            class_scores=((ClassId("person"), 0.9), (ClassId("container"), 0.05))
        )
        assert len(detection.class_scores) == 2

    def test_labels_are_opaque(self) -> None:
        detection = _detection(labels={"device": "cuda:0"})
        assert detection.labels["device"] == "cuda:0"


class TestQualityGrades:
    def test_unmeasured_grades_are_none_not_zero(self) -> None:
        """"Not measured" and "measured as zero" are different claims."""
        grades = QualityGrades(scale_pixels=218.0, truncation=0.0)
        assert grades.occlusion is None
        assert grades.blur is None
        assert grades.crowding is None
        assert grades.truncation == 0.0

    def test_out_of_range_grades_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="truncation"):
            QualityGrades(truncation=1.5)
        with pytest.raises(ValueError, match="scale_pixels"):
            QualityGrades(scale_pixels=-1.0)


class TestDecisionPath:
    def test_records_why_the_result_looks_as_it_does(self) -> None:
        evidence = DetectionEvidence(
            input_hash="blake2b:abc",
            decision_path=(
                DecisionStep.THRESHOLD_APPLIED,
                DecisionStep.CALIBRATION_UNAVAILABLE,
                DecisionStep.TAXONOMY_MAPPED,
            ),
        )
        assert DecisionStep.CALIBRATION_UNAVAILABLE in evidence.decision_path


class TestTaxonomyModel:
    def test_hierarchy_is_derived_from_the_dotted_path(self) -> None:
        forklift = TaxonomyClass(ClassId("vehicle.forklift"), "1.0.0")
        assert forklift.parent == "vehicle"
        assert forklift.ancestry == ("vehicle", "vehicle.forklift")
        assert forklift.is_a(ClassId("vehicle"))

    def test_root_class_has_no_parent(self) -> None:
        assert TaxonomyClass(ClassId("person"), "1.0.0").parent is None

    def test_superseded_class_must_name_a_successor(self) -> None:
        """Classes are deprecated, never deleted, so history stays readable."""
        with pytest.raises(ValueError, match="names no successor"):
            TaxonomyClass(
                ClassId("old"), "1.0.0", status=ClassStatus.SUPERSEDED
            )

    def test_malformed_class_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            TaxonomyClass(ClassId("vehicle."), "1.0.0")

    def test_mapping_rejects_duplicate_native_labels(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            TaxonomyMapping(
                adapter_id=AdapterId("a"),
                model_id=ModelId("m"),
                entries=(
                    MappingEntry("person", ClassId("person")),
                    MappingEntry("person", ClassId("vehicle")),
                ),
            )

    def test_mapping_publishes_producible_classes(self) -> None:
        mapping = TaxonomyMapping(
            adapter_id=AdapterId("a"),
            model_id=ModelId("m"),
            entries=(
                MappingEntry("person", ClassId("person")),
                MappingEntry("pedestrian", ClassId("person")),
                MappingEntry("forklift", ClassId("vehicle.forklift")),
            ),
        )
        assert mapping.producible_classes == ("person", "vehicle.forklift")

    def test_unmapped_policy_is_declared(self) -> None:
        mapping = TaxonomyMapping(
            adapter_id=AdapterId("a"),
            model_id=ModelId("m"),
            entries=(MappingEntry("person", ClassId("person")),),
            unmapped_policy=UnmappedPolicy.EMIT_AS_UNKNOWN,
        )
        assert mapping.unmapped_policy is UnmappedPolicy.EMIT_AS_UNKNOWN

    def test_mapping_confidence_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="mapping_confidence"):
            MappingEntry("person", ClassId("person"), mapping_confidence=1.5)


class TestModelMeta:
    def test_carries_the_weights_identity(self) -> None:
        meta = ModelMeta(
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:abc",
            precision="fp16",
            device_id="cuda:0",
        )
        assert meta.artifact_hash == "blake2b:abc"
        assert meta.precision == "fp16"
