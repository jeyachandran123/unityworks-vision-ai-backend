"""Motion prediction — and the elapsed-time obligation (T2).

06_PORTS calls integrating over frame count instead of elapsed time *"the single
most common way an off-the-shelf tracker misbehaves inside UWV"*, because the
platform drops frames by design (V7). These tests hold the predictors to real
seconds, and the key one feeds the same physical motion at two different frame
rates and requires the same velocity out.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.tracking.motion import (
    DEFAULT_UNCERTAINTY_GROWTH,
    VELOCITY_NOISE_FLOOR,
    LinearPredictor,
    StationaryPredictor,
    heading_of,
    speed_of,
)
from vision_os.core.model.timebase import Duration
from vision_os.core.ports.tracking import MotionObservation, Prediction


def observe(predictor, x: float, y: float = 0.4, *, elapsed_ms: int = 200, size: float = 0.1):
    predictor.observe(
        MotionObservation(x, y, x + size, y + 0.4, Duration.from_millis(elapsed_ms))
    )


class TestPredictionValidation:
    def test_negative_uncertainty_is_refused(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            Prediction(0.1, 0.1, 0.2, 0.2, uncertainty=-0.1)

    def test_a_prediction_may_leave_the_unit_square(self) -> None:
        """Clamping here would fold an exiting object back into the frame and
        make the association gate lie about where it expects the object."""
        assert Prediction(-0.2, 0.1, 0.1, 0.4).x1 == pytest.approx(-0.2)


class TestStationaryPredictor:
    def test_predicts_no_movement(self) -> None:
        predictor = StationaryPredictor()
        observe(predictor, 0.3)
        prediction = predictor.predict(Duration.from_millis(1_000))
        assert prediction.x1 == pytest.approx(0.3)

    def test_reports_zero_velocity(self) -> None:
        predictor = StationaryPredictor()
        observe(predictor, 0.3)
        observe(predictor, 0.5)
        assert predictor.velocity() == (0.0, 0.0)

    def test_acceleration_is_unknown_not_zero(self) -> None:
        predictor = StationaryPredictor()
        observe(predictor, 0.3)
        assert predictor.acceleration() is None

    def test_uncertainty_grows_with_elapsed_time(self) -> None:
        predictor = StationaryPredictor()
        observe(predictor, 0.3)
        near = predictor.predict(Duration.from_millis(200))
        far = predictor.predict(Duration.from_millis(2_000))
        assert far.uncertainty > near.uncertainty

    def test_predicting_before_observing_is_an_explicit_error(self) -> None:
        with pytest.raises(ValueError, match="before any observation"):
            StationaryPredictor().predict(Duration.from_millis(100))

    def test_declares_a_model_id(self) -> None:
        assert StationaryPredictor().model_id == "motion.stationary"


class TestLinearPredictor:
    def test_extrapolates_along_the_observed_velocity(self) -> None:
        predictor = LinearPredictor(smoothing=1.0)
        observe(predictor, 0.10)
        observe(predictor, 0.20)  # +0.10 over 200 ms -> 0.5 /s
        prediction = predictor.predict(Duration.from_millis(200))
        assert prediction.x1 == pytest.approx(0.30, abs=1e-6)

    def test_velocity_is_per_second_not_per_frame(self) -> None:
        predictor = LinearPredictor(smoothing=1.0)
        observe(predictor, 0.10)
        observe(predictor, 0.20, elapsed_ms=200)
        assert predictor.velocity()[0] == pytest.approx(0.5, abs=1e-6)

    def test_same_physical_motion_at_two_frame_rates_gives_the_same_velocity(self) -> None:
        """**Port obligation T2.** The scheduler drops frames by design, so a
        tracker keying on frame count changes its physics under load."""
        fast = LinearPredictor(smoothing=1.0)
        observe(fast, 0.10)
        observe(fast, 0.15, elapsed_ms=100)

        slow = LinearPredictor(smoothing=1.0)
        observe(slow, 0.10)
        observe(slow, 0.25, elapsed_ms=300)

        assert fast.velocity()[0] == pytest.approx(slow.velocity()[0], abs=1e-6)

    def test_a_zero_elapsed_observation_does_not_divide_by_zero(self) -> None:
        predictor = LinearPredictor()
        observe(predictor, 0.10, elapsed_ms=0)
        observe(predictor, 0.20, elapsed_ms=0)
        assert predictor.velocity() == (0.0, 0.0)

    def test_smoothing_damps_a_single_jitter_spike(self) -> None:
        """Differentiating raw detector jitter produces wild velocity swings."""
        smooth = LinearPredictor(smoothing=0.3)
        raw = LinearPredictor(smoothing=1.0)
        for predictor in (smooth, raw):
            observe(predictor, 0.10)
            observe(predictor, 0.15)
            observe(predictor, 0.40)  # the spike
        assert abs(smooth.velocity()[0]) < abs(raw.velocity()[0])

    def test_acceleration_is_none_until_enough_observations(self) -> None:
        predictor = LinearPredictor()
        observe(predictor, 0.10)
        assert predictor.acceleration() is None
        observe(predictor, 0.20)
        assert predictor.acceleration() is None

    def test_acceleration_appears_after_three_observations(self) -> None:
        predictor = LinearPredictor(smoothing=1.0)
        observe(predictor, 0.10)
        observe(predictor, 0.20)
        observe(predictor, 0.40)
        assert predictor.acceleration() is not None

    def test_box_size_is_not_extrapolated(self) -> None:
        """Extrapolating growth compounds error, and a too-large gate captures
        neighbours — the cost of being wrong is asymmetric."""
        predictor = LinearPredictor(smoothing=1.0)
        observe(predictor, 0.10, size=0.10)
        observe(predictor, 0.20, size=0.20)
        prediction = predictor.predict(Duration.from_millis(1_000))
        assert (prediction.x2 - prediction.x1) == pytest.approx(0.20, abs=1e-6)

    def test_uncertainty_grows_with_the_prediction_horizon(self) -> None:
        predictor = LinearPredictor()
        observe(predictor, 0.10)
        observe(predictor, 0.20)
        near = predictor.predict(Duration.from_millis(200))
        far = predictor.predict(Duration.from_millis(2_000))
        assert far.uncertainty == pytest.approx(near.uncertainty * 10, rel=1e-6)
        assert near.uncertainty == pytest.approx(DEFAULT_UNCERTAINTY_GROWTH * 0.2)

    def test_a_long_horizon_extrapolates_proportionally(self) -> None:
        """A five-frame coast must predict five frames ahead, not one."""
        predictor = LinearPredictor(smoothing=1.0)
        observe(predictor, 0.10)
        observe(predictor, 0.20)
        one = predictor.predict(Duration.from_millis(200))
        five = predictor.predict(Duration.from_millis(1_000))
        assert (five.x1 - 0.20) == pytest.approx((one.x1 - 0.20) * 5, abs=1e-6)

    def test_invalid_smoothing_is_refused(self) -> None:
        for value in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="smoothing"):
                LinearPredictor(smoothing=value)

    def test_predicting_before_observing_is_an_explicit_error(self) -> None:
        with pytest.raises(ValueError, match="before any observation"):
            LinearPredictor().predict(Duration.from_millis(100))

    def test_declares_a_model_id(self) -> None:
        assert LinearPredictor().model_id == "motion.linear"

    def test_diagonal_motion_is_tracked_on_both_axes(self) -> None:
        predictor = LinearPredictor(smoothing=1.0)
        predictor.observe(MotionObservation(0.1, 0.1, 0.2, 0.2, Duration.from_millis(200)))
        predictor.observe(MotionObservation(0.2, 0.3, 0.3, 0.4, Duration.from_millis(200)))
        vx, vy = predictor.velocity()
        assert vx == pytest.approx(0.5, abs=1e-6)
        assert vy == pytest.approx(1.0, abs=1e-6)


class TestSpeedAndHeading:
    def test_speed_is_the_magnitude(self) -> None:
        assert speed_of((0.3, 0.4)) == pytest.approx(0.5)

    def test_heading_is_none_below_the_noise_floor(self) -> None:
        """At rest atan2 returns a precise angle computed entirely from noise."""
        assert heading_of((0.0, 0.0)) is None
        assert heading_of((VELOCITY_NOISE_FLOOR / 2, 0.0)) is None

    def test_heading_east_is_zero(self) -> None:
        assert heading_of((1.0, 0.0)) == pytest.approx(0.0)

    def test_heading_is_clockwise_from_positive_x(self) -> None:
        """Image space: +y is down, so a downward vector reads as 90 degrees."""
        assert heading_of((0.0, 1.0)) == pytest.approx(90.0)

    def test_heading_wraps_into_the_circle(self) -> None:
        heading = heading_of((0.0, -1.0))
        assert heading is not None
        assert 0.0 <= heading < 360.0
        assert heading == pytest.approx(270.0)

    def test_heading_is_defined_just_above_the_floor(self) -> None:
        assert heading_of((VELOCITY_NOISE_FLOOR * 2, 0.0)) is not None


class TestPredictorsAreIndependent:
    def test_two_predictors_do_not_share_state(self) -> None:
        """Each track owns its predictor; shared state would cross-contaminate."""
        first, second = LinearPredictor(), LinearPredictor()
        observe(first, 0.10)
        observe(first, 0.50)
        observe(second, 0.10)
        observe(second, 0.11)
        assert abs(first.velocity()[0]) > abs(second.velocity()[0])

    def test_both_predictors_satisfy_the_same_port(self) -> None:
        for predictor in (StationaryPredictor(), LinearPredictor()):
            assert predictor.model_id
            observe(predictor, 0.2)
            assert predictor.predict(Duration.from_millis(100)) is not None
            assert isinstance(predictor.velocity(), tuple)
