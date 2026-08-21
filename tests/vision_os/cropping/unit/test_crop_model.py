"""The Crop object — 02_VOM section 10.7.

A crop is **evidence**, not an image: pixels plus everything needed to defend a
claim made from them. Every test here guards a way that defence could quietly
fail — a crop that names the wrong camera, a rejection with no reason, imagery
retained forever, or a transform record that disagrees with its own pixels.

``without_pixels`` is invariant V12 made mechanical. A crop reference is ~1 KB
and may cross a node boundary; the pixels are megabytes and may not.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.crop import (
    Crop,
    CropTransform,
    EvaluationResult,
    GateRejection,
    GateResult,
    PrivacyClass,
    RetentionMode,
    Skipped,
    SkipReason,
    TriggerReason,
)
from vision_os.core.model.detection import QualityGrades, QualityLevel
from vision_os.core.model.ids import (
    ConfigRevision,
    CropId,
    ModuleId,
    ObjectId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration, Instant

from ..conftest import (
    CAMERA,
    OTHER_CAMERA,
    SITE,
    TENANT,
    at,
    frame_ref,
    make_request,
)

PROVENANCE = Provenance(
    producer_module=ModuleId("crop_manager"),
    producer_version="1.0.0",
    config_revision=ConfigRevision("test"),
)

TRANSFORM = CropTransform(
    source_width=640,
    source_height=480,
    output_width=64,
    output_height=64,
    crop_x=100,
    crop_y=100,
    crop_width=64,
    crop_height=64,
)


def make_crop(**overrides) -> Crop:
    payload = {
        "crop_id": CropId("a" * 64),
        "tenant_id": TENANT,
        "site_id": SITE,
        "camera_id": CAMERA,
        "source_frame": frame_ref(3),
        "object_id": ObjectId("obj-1"),
        "source_box": Box(0.4, 0.3, 0.55, 0.85),
        "padding_applied": 0.15,
        "output_size": (64, 64),
        "transform": TRANSFORM,
        "quality": QualityGrades(scale_pixels=264.0, overall=QualityLevel.GOOD),
        "gate_result": GateResult.accept(),
        "retention": RetentionMode.EPHEMERAL,
        "privacy_class": PrivacyClass.C1_IMAGERY,
        "t_capture": at(3),
        "trigger_reason": TriggerReason.FIRST_SIGHT,
        "provenance": PROVENANCE,
    }
    payload.update(overrides)
    return Crop(**payload)


class TestTraceability:
    def test_a_crop_names_its_own_frames_camera(self) -> None:
        """Evidence that claims the wrong camera is worse than no evidence."""
        with pytest.raises(ValueError, match="traceable"):
            make_crop(camera_id=OTHER_CAMERA)

    def test_a_crop_requires_a_content_hash(self) -> None:
        with pytest.raises(ValueError, match="content-addressed"):
            make_crop(crop_id=CropId(""))

    def test_a_crop_carries_its_trigger_reason(self) -> None:
        """02_VOM section 10.9: evidence records *why this was computed at all*.

        Without it, a result six months old is a number with no story.
        """
        crop = make_crop(trigger_reason=TriggerReason.APPEARANCE_CHANGED)
        assert crop.trigger_reason is TriggerReason.APPEARANCE_CHANGED

    def test_capture_time_is_the_frames_not_the_extractions(self) -> None:
        """V11. A dwell assembled from crops must reflect the world."""
        crop = make_crop(t_capture=at(3))
        assert crop.t_capture == at(3)


class TestTransformIntegrity:
    def test_output_size_must_match_the_transform(self) -> None:
        """A record that disagrees with its pixels invites an invalid comparison."""
        with pytest.raises(ValueError, match="looks valid and is not"):
            make_crop(output_size=(128, 128))

    def test_a_degenerate_output_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="degenerate output size"):
            make_crop(output_size=(0, 64))

    def test_padding_is_bounded(self) -> None:
        with pytest.raises(ValueError, match=r"padding_applied must be in"):
            make_crop(padding_applied=9.0)


class TestGateResult:
    def test_a_rejection_must_name_a_reason(self) -> None:
        from vision_os.core.model.crop import GateOutcome

        with pytest.raises(ValueError, match="must name its reason"):
            GateResult(outcome=GateOutcome.REJECTED)

    def test_a_rejected_crop_is_still_constructible(self) -> None:
        """A rejection is *counted evidence*, not an absence.

        The crop exists with its grades and its reason so the rejection becomes a
        statistic; discarding it would make the gate invisible.
        """
        crop = make_crop(
            gate_result=GateResult.reject(GateRejection.TOO_SMALL, "18px"),
            quality=QualityGrades(scale_pixels=18.0, overall=QualityLevel.INSUFFICIENT),
        )
        assert not crop.passed_gate
        assert crop.gate_result.reason is GateRejection.TOO_SMALL


class TestRetentionAndPrivacy:
    def test_evidence_retention_requires_a_ttl(self) -> None:
        """Imagery retained forever is a compliance liability (12_SECURITY)."""
        with pytest.raises(ValueError, match="compliance liability"):
            make_crop(retention=RetentionMode.EVIDENCE, retention_ttl=None)

    def test_never_persist_cannot_carry_a_ttl(self) -> None:
        with pytest.raises(ValueError, match="never_persist cannot carry a TTL"):
            make_crop(
                retention=RetentionMode.NEVER_PERSIST,
                retention_ttl=Duration.from_millis(1_000),
            )

    def test_expiry_is_computed_from_capture_time(self) -> None:
        crop = make_crop(
            retention=RetentionMode.EVIDENCE,
            retention_ttl=Duration.from_millis(1_000),
        )
        assert crop.expires_at() == Instant(at(3).ns + 1_000_000_000)

    def test_an_ephemeral_crop_expires_immediately(self) -> None:
        assert make_crop().expires_at() is None

    def test_only_evidence_is_persistable(self) -> None:
        assert not make_crop().is_persistable
        assert make_crop(
            retention=RetentionMode.EVIDENCE, retention_ttl=Duration.from_millis(1)
        ).is_persistable

    def test_every_crop_is_classified_imagery(self) -> None:
        """C1, fixed rather than inferred (12_SECURITY section 3).

        Inferring a data classification is how imagery ends up in an
        unclassified path.
        """
        assert make_crop().privacy_class is PrivacyClass.C1_IMAGERY


class TestTheDataPlaneBoundary:
    def test_without_pixels_strips_the_imagery(self) -> None:
        """Invariant V12: the reference travels, the pixels do not."""
        crop = make_crop(pixels=memoryview(b"\x00" * (64 * 64 * 3)))
        reference = crop.without_pixels()
        assert reference.pixels is None
        assert reference.crop_id == crop.crop_id
        assert reference.transform == crop.transform

    def test_stripping_pixels_preserves_every_other_field(self) -> None:
        crop = make_crop(pixels=memoryview(b"\x00" * 12))
        reference = crop.without_pixels()
        for field in Crop.__dataclass_fields__:
            if field == "pixels":
                continue
            assert getattr(reference, field) == getattr(crop, field)


class TestEvaluationResult:
    def test_every_candidate_appears_exactly_once(self) -> None:
        """The V8 accounting identity."""
        result = EvaluationResult(
            camera_id=CAMERA,
            frame_ref=frame_ref(1),
            requests=(make_request(object_id="a"), make_request(object_id="b")),
            skipped=(
                Skipped(
                    object_id=ObjectId("c"),
                    camera_id=CAMERA,
                    reason=SkipReason.NO_DEMAND,
                ),
            ),
        )
        assert result.candidate_count == 3

    def test_skips_are_countable_by_reason(self) -> None:
        """A deployment where every skip is NO_DEMAND is healthy.

        One where they are BUDGET_EXHAUSTED is under-provisioned; one where they
        are QUALITY_INSUFFICIENT has a camera problem. None of that is visible
        without an attributed reason.
        """
        result = EvaluationResult(
            camera_id=CAMERA,
            frame_ref=frame_ref(1),
            skipped=tuple(
                Skipped(
                    object_id=ObjectId(f"o{i}"),
                    camera_id=CAMERA,
                    reason=SkipReason.BUDGET_EXHAUSTED
                    if i % 2
                    else SkipReason.NO_DEMAND,
                )
                for i in range(4)
            ),
        )
        counts = result.skips_by_reason()
        assert counts[SkipReason.NO_DEMAND] == 2
        assert counts[SkipReason.BUDGET_EXHAUSTED] == 2


class TestCropRequest:
    def test_a_request_names_its_frames_own_camera(self) -> None:
        with pytest.raises(ValueError, match="the frame's own camera"):
            make_request(camera=OTHER_CAMERA).__class__(
                object_id=ObjectId("x"),
                camera_id=OTHER_CAMERA,
                frame_ref=frame_ref(1, camera=CAMERA),
                source_box=Box(0.1, 0.1, 0.2, 0.2),
                trigger_reason=TriggerReason.FIRST_SIGHT,
                tenant_id=TENANT,
                site_id=SITE,
            )

    def test_tenancy_travels_with_the_request(self) -> None:
        """12_SECURITY section 4 needs it in the cache key, so it must be here."""
        assert make_request().tenant_id == TENANT
