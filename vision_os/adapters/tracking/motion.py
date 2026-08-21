"""Motion predictors behind P-motion. Pure arithmetic, no dependencies.

Two models ship:

``StationaryPredictor``
    Predicts no movement. The honest baseline for a fixed-camera scene of mostly
    static objects, and the correct choice when velocity cannot be estimated.

``LinearPredictor``
    Constant velocity with exponentially smoothed estimates. Covers the large
    majority of surveillance motion without tuning constants that vary by camera
    geometry.

**A Kalman filter is deliberately not here.** It belongs behind
``MotionPredictorPort`` as an adapter, because its process- and measurement-noise
parameters are correct for one camera geometry and wrong for another; baking it
into the platform would freeze constants that must stay deployable. The seam
exists and is proven by ``ScriptedPredictor`` in the test suite.

Everything integrates over **elapsed seconds**, never frame count — port
obligation T2, and the single most common way an off-the-shelf tracker
misbehaves inside a platform that drops frames by design.
"""

from __future__ import annotations

import math

from ...core.model.timebase import Duration
from ...core.ports.tracking import MotionObservation, Prediction

#: Below this, velocity is noise rather than motion and heading is meaningless.
VELOCITY_NOISE_FLOOR = 1e-4

#: Positional uncertainty accrued per second of prediction, in normalized units.
#: Chosen so a one-second coast widens the gate by roughly a small object's
#: width — enough to re-acquire, not so much that it swallows a neighbour.
DEFAULT_UNCERTAINTY_GROWTH = 0.05


class StationaryPredictor:
    """Predicts that nothing moves.

    Not a placeholder: for a fixed camera watching mostly static objects this
    outperforms a velocity model, which amplifies detector jitter into phantom
    motion and then extrapolates it.
    """

    __slots__ = ("_box", "_uncertainty_growth")

    def __init__(self, *, uncertainty_growth: float = DEFAULT_UNCERTAINTY_GROWTH) -> None:
        self._box: tuple[float, float, float, float] | None = None
        self._uncertainty_growth = uncertainty_growth

    @property
    def model_id(self) -> str:
        return "motion.stationary"

    def observe(self, observation: MotionObservation) -> None:
        self._box = (observation.x1, observation.y1, observation.x2, observation.y2)

    def predict(self, elapsed: Duration) -> Prediction:
        if self._box is None:
            raise ValueError("predict() before any observation")
        x1, y1, x2, y2 = self._box
        return Prediction(
            x1, y1, x2, y2, uncertainty=self._uncertainty_growth * elapsed.seconds
        )

    def velocity(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def acceleration(self) -> tuple[float, float] | None:
        return None


class LinearPredictor:
    """Constant-velocity prediction with exponentially smoothed estimates.

    Smoothing rather than raw finite differences because a detector's box jitters
    by a pixel or two every frame; differentiating that directly produces
    velocities that swing wildly and predictions that overshoot. The smoothing
    factor is the one tuning knob, and it is a platform-neutral number rather
    than a camera-geometry-dependent one.

    Box **size** is smoothed but not extrapolated. An object walking toward the
    camera does grow, but extrapolating growth compounds error quickly and a
    too-large gate captures neighbours — the cost of being wrong is asymmetric.
    """

    __slots__ = (
        "_acceleration",
        "_box",
        "_observations",
        "_smoothing",
        "_uncertainty_growth",
        "_velocity",
    )

    def __init__(
        self,
        *,
        smoothing: float = 0.6,
        uncertainty_growth: float = DEFAULT_UNCERTAINTY_GROWTH,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0,1], got {smoothing}")
        self._smoothing = smoothing
        self._uncertainty_growth = uncertainty_growth
        self._box: tuple[float, float, float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._acceleration: tuple[float, float] | None = None
        self._observations = 0

    @property
    def model_id(self) -> str:
        return "motion.linear"

    def observe(self, observation: MotionObservation) -> None:
        box = (observation.x1, observation.y1, observation.x2, observation.y2)
        seconds = observation.elapsed.seconds

        if self._box is not None and seconds > 0.0:
            previous_cx = (self._box[0] + self._box[2]) / 2.0
            previous_cy = (self._box[1] + self._box[3]) / 2.0
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            # Per second, never per frame (T2).
            instantaneous = ((cx - previous_cx) / seconds, (cy - previous_cy) / seconds)
            previous_velocity = self._velocity
            alpha = self._smoothing
            self._velocity = (
                alpha * instantaneous[0] + (1.0 - alpha) * previous_velocity[0],
                alpha * instantaneous[1] + (1.0 - alpha) * previous_velocity[1],
            )
            if self._observations >= 2:
                self._acceleration = (
                    (self._velocity[0] - previous_velocity[0]) / seconds,
                    (self._velocity[1] - previous_velocity[1]) / seconds,
                )

        self._box = box
        self._observations += 1

    def predict(self, elapsed: Duration) -> Prediction:
        if self._box is None:
            raise ValueError("predict() before any observation")
        x1, y1, x2, y2 = self._box
        seconds = elapsed.seconds
        dx = self._velocity[0] * seconds
        dy = self._velocity[1] * seconds
        return Prediction(
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
            uncertainty=self._uncertainty_growth * seconds,
        )

    def velocity(self) -> tuple[float, float]:
        return self._velocity

    def acceleration(self) -> tuple[float, float] | None:
        return self._acceleration


def speed_of(velocity: tuple[float, float]) -> float:
    return math.hypot(velocity[0], velocity[1])


def heading_of(velocity: tuple[float, float]) -> float | None:
    """Clockwise degrees from +x, or ``None`` below the noise floor.

    Returning ``None`` rather than 0.0 matters: at rest, ``atan2`` returns a
    perfectly precise angle derived entirely from noise, and a consumer has no
    way to tell that from a real heading.
    """
    if speed_of(velocity) < VELOCITY_NOISE_FLOOR:
        return None
    degrees = math.degrees(math.atan2(velocity[1], velocity[0]))
    return degrees % 360.0
