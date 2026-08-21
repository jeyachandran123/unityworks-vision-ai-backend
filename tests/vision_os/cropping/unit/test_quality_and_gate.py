"""Quality estimation and the gate.

Two modules, deliberately separate. The estimator **measures**; the gate
**judges**. Keeping them apart means a deployment can swap in a learned quality
predictor without touching the thresholds that decide what is affordable, and a
threshold change never silently alters what "blur" means.

The test that earns its keep here is ``test_unmeasured_grades_are_none``. A
zeroed blur grade reads as *perfectly sharp*: an estimator that reports 0.0 for
"I did not look" tells the gate to spend money on an input nobody assessed, and
every dashboard stays green while it happens.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import HeuristicQualityEstimator
from vision_os.core.model.crop import GateOutcome, GateRejection, GateResult
from vision_os.core.model.detection import (
    ExposureLevel,
    QualityGrades,
    QualityLevel,
)
from vision_os.core.model.space import Box
from vision_os.core.ports.cropping import QualityRequest
from vision_os.perception.cropping import GateThresholds, QualityGate

from ..conftest import (
    CAMERA,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    bright_frame,
    dark_frame,
    blurred_frame,
    noise_frame,
    flat_frame,
    sharp_frame,
)


def request_for(box: Box, **overrides) -> QualityRequest:
    payload = {
        "camera_id": CAMERA,
        "box": box,
        "source_width": FRAME_WIDTH,
        "source_height": FRAME_HEIGHT,
    }
    payload.update(overrides)
    return QualityRequest(**payload)


class TestScaleAndTruncation:
    def test_scale_is_object_height_in_source_pixels(
        self, estimator: HeuristicQualityEstimator
    ) -> None:
        """Height, not area. The models downstream care how tall a person is."""
        grades = estimator.estimate(request_for(Box(0.4, 0.2, 0.5, 0.7)))
        assert grades.scale_pixels == pytest.approx(0.5 * FRAME_HEIGHT, rel=1e-6)

    def test_a_larger_box_grades_larger(self, estimator) -> None:
        small = estimator.estimate(request_for(Box(0.5, 0.5, 0.52, 0.55)))
        large = estimator.estimate(request_for(Box(0.2, 0.1, 0.5, 0.9)))
        assert large.scale_pixels > small.scale_pixels

    def test_truncation_measures_what_lies_outside(self, estimator) -> None:
        """Measured against the *unclamped* box; after clamping the evidence is gone."""
        grades = estimator.estimate(request_for(Box(-0.1, 0.2, 0.1, 0.8)))
        assert grades.truncation == pytest.approx(0.5, abs=0.01)

    def test_a_fully_visible_object_is_untruncated(self, estimator) -> None:
        grades = estimator.estimate(request_for(Box(0.3, 0.3, 0.6, 0.8)))
        assert grades.truncation == pytest.approx(0.0, abs=1e-9)


class TestCrowdingAndOcclusion:
    def test_crowding_measures_neighbour_overlap(self, estimator) -> None:
        box = Box(0.4, 0.4, 0.6, 0.8)
        grades = estimator.estimate(
            request_for(box, neighbour_boxes=(Box(0.5, 0.4, 0.7, 0.8),))
        )
        assert grades.crowding == pytest.approx(0.5, abs=0.01)

    def test_no_neighbours_means_no_crowding(self, estimator) -> None:
        grades = estimator.estimate(request_for(Box(0.4, 0.4, 0.6, 0.8)))
        assert grades.crowding == 0.0

    def test_crowding_is_bounded(self, estimator) -> None:
        """Several overlapping neighbours cannot push the grade past 1."""
        box = Box(0.4, 0.4, 0.6, 0.8)
        neighbours = tuple(Box(0.4, 0.4, 0.6, 0.8) for _ in range(5))
        grades = estimator.estimate(request_for(box, neighbour_boxes=neighbours))
        assert grades.crowding <= 1.0


class TestPixelGrades:
    def test_unmeasured_grades_are_none(self, estimator) -> None:
        """Without pixels, blur and exposure must be ``None`` — never zero.

        The single most consequential contract in this module: zero means
        *perfectly sharp*, and reporting it for an input never examined sends
        unanswerable questions to expensive models.
        """
        grades = estimator.estimate(request_for(Box(0.4, 0.3, 0.55, 0.85)))
        assert grades.blur is None
        assert grades.exposure is None
        assert grades.scale_pixels is not None, "geometry is always measurable"

    def test_a_sharp_crop_grades_low_blur(self, estimator) -> None:
        grades = estimator.estimate(
            request_for(
                Box(0.4, 0.3, 0.55, 0.85),
                pixels=sharp_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        assert grades.blur is not None
        assert grades.blur < 0.5, f"a checkerboard graded {grades.blur} blur"

    def test_a_featureless_crop_grades_maximum_blur(self, estimator) -> None:
        grades = estimator.estimate(
            request_for(
                Box(0.4, 0.3, 0.55, 0.85),
                pixels=flat_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        assert grades.blur == pytest.approx(1.0, abs=1e-6)

    def test_blur_separates_a_smeared_crop_from_a_sharp_one(self, estimator) -> None:
        """The axis must respond to *blur*, not merely to featurelessness.

        Regression test for a defect found on real kitchen footage: the estimator
        sampled with a stride spanning the whole crop and then treated those
        far-apart samples as neighbours. Pixels that distant are uncorrelated in
        any textured scene, so the variance saturated and every crop graded
        0.0 — perfectly sharp. Only a uniform frame ever scored as blurred, which
        made the whole axis, and every threshold built on it, inert.

        A texture smeared by repeated local averaging must grade measurably
        worse than the texture itself.
        """
        sharp = estimator.estimate(
            request_for(
                Box(0.4, 0.3, 0.55, 0.85),
                pixels=noise_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        smeared = estimator.estimate(
            request_for(
                Box(0.4, 0.3, 0.55, 0.85),
                pixels=blurred_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        assert sharp.blur is not None and smeared.blur is not None
        assert smeared.blur > sharp.blur + 0.3, (
            f"blurring moved the grade only from {sharp.blur} to {smeared.blur}; "
            "the axis is not measuring focus"
        )

    def test_exposure_is_classified(self, estimator) -> None:
        def exposure_of(pixels):
            return estimator.estimate(
                request_for(
                    Box(0.4, 0.3, 0.55, 0.85),
                    pixels=pixels,
                    crop_width=64,
                    crop_height=64,
                )
            ).exposure

        assert exposure_of(dark_frame(64, 64)) is ExposureLevel.UNDER
        assert exposure_of(bright_frame(64, 64)) is ExposureLevel.OVER
        assert exposure_of(flat_frame(64, 64, value=128)) is ExposureLevel.OK


class TestTheOverallVerdict:
    def test_worst_axis_wins(self, estimator) -> None:
        """A sharp, well-exposed, 12-pixel-tall crop is not a good crop.

        Averaging the axes would say otherwise and would send an unanswerable
        question to an expensive model.
        """
        grades = estimator.estimate(
            request_for(
                Box(0.5, 0.5, 0.52, 0.525),
                pixels=sharp_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        assert grades.overall is QualityLevel.INSUFFICIENT

    def test_geometry_alone_never_grades_excellent(self, estimator) -> None:
        """"Good" is the honest ceiling for a verdict that has not seen pixels."""
        grades = estimator.estimate(request_for(Box(0.2, 0.05, 0.5, 0.95)))
        assert grades.overall is QualityLevel.GOOD

    def test_a_large_sharp_crop_can_be_excellent(self, estimator) -> None:
        grades = estimator.estimate(
            request_for(
                Box(0.2, 0.05, 0.5, 0.95),
                pixels=sharp_frame(64, 64),
                crop_width=64,
                crop_height=64,
            )
        )
        assert grades.overall is QualityLevel.EXCELLENT

    def test_overall_is_always_set(self, estimator) -> None:
        """It is the gate's only input (obligation Q3)."""
        for box in (
            Box(0.4, 0.3, 0.55, 0.85),
            Box(0.0, 0.0, 0.001, 0.001),
            Box(0.9, 0.9, 1.5, 1.5),
        ):
            assert estimator.estimate(request_for(box)).overall is not None


class TestEstimatorRobustness:
    def test_determinism(self, estimator) -> None:
        request = request_for(Box(0.31, 0.22, 0.55, 0.81))
        assert estimator.estimate(request) == estimator.estimate(request)

    def test_a_zero_dimension_frame_grades_rather_than_raises(self, estimator) -> None:
        grades = estimator.estimate(
            request_for(Box(0.4, 0.3, 0.55, 0.85), source_width=0, source_height=0)
        )
        assert grades.overall is QualityLevel.INSUFFICIENT

    def test_a_short_pixel_buffer_is_refused_not_misread(self, estimator) -> None:
        """A truncated buffer must not produce grades that look measured.

        Reading past a short buffer would either raise deep in the sampler or —
        worse — silently grade whatever bytes happened to be adjacent.
        """
        grades = estimator.estimate(
            request_for(
                Box(0.4, 0.3, 0.55, 0.85),
                pixels=memoryview(b"\x00" * 12),
                crop_width=64,
                crop_height=64,
            )
        )
        assert grades.blur is None or grades.blur == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_scale_pixels": 0},
            {"min_scale_pixels": 100, "good_scale_pixels": 50},
            {"max_blur": 1.5},
            {"max_truncation": -0.1},
        ],
    )
    def test_invalid_configuration_is_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            HeuristicQualityEstimator(**kwargs)


class TestTheGate:
    def test_a_rejection_always_names_its_reason(self) -> None:
        """*"The VLM never answers for far-away people"* must be a statistic."""
        with pytest.raises(ValueError, match="must name its reason"):
            GateResult(outcome=GateOutcome.REJECTED)

    def test_a_pass_carries_no_reason(self) -> None:
        with pytest.raises(ValueError, match="cannot carry a rejection reason"):
            GateResult(outcome=GateOutcome.PASSED, reason=GateRejection.TOO_SMALL)

    def test_too_small(self, gate: QualityGate) -> None:
        result = gate.evaluate(QualityGrades(scale_pixels=20.0))
        assert result.reason is GateRejection.TOO_SMALL
        assert "20.0px" in result.detail, "the measurement must be in the record"

    def test_too_truncated(self, gate) -> None:
        result = gate.evaluate(QualityGrades(scale_pixels=200.0, truncation=0.9))
        assert result.reason is GateRejection.TOO_TRUNCATED

    def test_too_occluded(self, gate) -> None:
        result = gate.evaluate(QualityGrades(scale_pixels=200.0, occlusion=0.95))
        assert result.reason is GateRejection.TOO_OCCLUDED

    def test_too_blurry(self, gate) -> None:
        result = gate.evaluate(QualityGrades(scale_pixels=200.0, blur=0.99))
        assert result.reason is GateRejection.TOO_BLURRY

    def test_degenerate_geometry(self, gate) -> None:
        result = gate.evaluate(QualityGrades(scale_pixels=0.0))
        assert result.reason is GateRejection.DEGENERATE_GEOMETRY

    def test_scale_wins_the_attribution_over_blur(self, gate) -> None:
        """A 9-pixel object is TOO_SMALL, not TOO_BLURRY.

        It is also unavoidably blurry — but a human reading the alarm needs the
        cause, and the cause is that the object is 40 metres away.
        """
        result = gate.evaluate(QualityGrades(scale_pixels=9.0, blur=0.99))
        assert result.reason is GateRejection.TOO_SMALL

    def test_exposure_does_not_reject_by_default(self, gate) -> None:
        """V9: degrade, never die. Rejecting on exposure blinds a site at dusk."""
        result = gate.evaluate(
            QualityGrades(scale_pixels=200.0, exposure=ExposureLevel.UNDER)
        )
        assert result.passed

    def test_exposure_rejection_is_opt_in(self) -> None:
        gate = QualityGate(GateThresholds(reject_extreme_exposure=True))
        result = gate.evaluate(
            QualityGrades(scale_pixels=200.0, exposure=ExposureLevel.OVER)
        )
        assert result.reason is GateRejection.EXPOSURE_UNUSABLE

    def test_ungraded_input_passes(self, gate) -> None:
        """An estimator that measured nothing must not silently block everything.

        The gate refuses what it *knows* is bad. Refusing what it does not know
        would turn a missing estimator into a total outage (V9).
        """
        assert gate.evaluate(QualityGrades()).passed

    def test_the_gate_is_pure(self, gate) -> None:
        grades = QualityGrades(scale_pixels=200.0, blur=0.1)
        assert gate.evaluate(grades) == gate.evaluate(grades)

    def test_would_pass_agrees_with_evaluate(self, gate) -> None:
        """Two implementations of "good enough" would drift; there is only one."""
        for grades in (
            QualityGrades(scale_pixels=200.0),
            QualityGrades(scale_pixels=10.0),
            QualityGrades(scale_pixels=200.0, blur=0.99),
        ):
            assert gate.would_pass(grades) == gate.evaluate(grades).passed

    @pytest.mark.parametrize(
        "kwargs",
        [{"min_scale_pixels": -1}, {"max_blur": 2.0}, {"max_truncation": -0.5}],
    )
    def test_invalid_thresholds_are_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            GateThresholds(**kwargs)
