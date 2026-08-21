"""Confidence calibration profiles (02_VISION_OBJECT_MODEL section 7.2 rule 4).

> **Calibration is a platform capability, not a model capability.** A new
> detector gains calibrated confidence by fitting a profile, not by being
> trusted.

A consumer that wrote ``if confidence > 0.7`` against a 2026 detector will
silently change behaviour when a 2029 detector with a different score
distribution is swapped in — unless confidence means the same thing across both.
That is what a profile buys, and it is why a model swap does not require every
consumer to re-tune.

``raw_score`` is always preserved alongside the calibrated value, so a profile
refitted next year can re-calibrate history without re-running inference.
"""

from __future__ import annotations

import enum
import math
import threading
from dataclasses import dataclass

from ...core.model.ids import CalibrationId, ModelId


class CalibrationMethod(enum.Enum):
    IDENTITY = "identity"
    """No transformation. Declared, not assumed — an identity profile still marks
    a confidence ``calibrated`` only if it was genuinely validated."""

    TEMPERATURE = "temperature"
    """Logistic temperature scaling; the standard single-parameter fit."""

    PIECEWISE = "piecewise"
    """Monotone piecewise-linear mapping, for isotonic-style fits."""


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """A fitted mapping from a model's raw score to a probability."""

    calibration_id: CalibrationId
    model_id: ModelId
    model_version: str
    method: CalibrationMethod = CalibrationMethod.IDENTITY
    temperature: float = 1.0
    knots: tuple[tuple[float, float], ...] = ()
    """``(raw, calibrated)`` pairs for ``PIECEWISE``, ascending in ``raw``."""

    fitted_on: str = ""
    """The validation set this was fitted against. A profile with no stated
    provenance is a guess wearing a number."""

    def __post_init__(self) -> None:
        if self.method is CalibrationMethod.TEMPERATURE and self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.method is CalibrationMethod.PIECEWISE:
            if len(self.knots) < 2:
                raise ValueError("a piecewise profile needs at least two knots")
            raws = [k[0] for k in self.knots]
            if raws != sorted(raws):
                raise ValueError("piecewise knots must ascend in raw score")
            for raw, calibrated in self.knots:
                if not 0.0 <= raw <= 1.0 or not 0.0 <= calibrated <= 1.0:
                    raise ValueError(f"knot ({raw}, {calibrated}) escapes [0,1]")

    def apply(self, raw_score: float) -> float:
        """Map a raw score to a calibrated probability, clamped to [0,1]."""
        if self.method is CalibrationMethod.IDENTITY:
            return _clamp(raw_score)
        if self.method is CalibrationMethod.TEMPERATURE:
            return _clamp(_temperature_scale(raw_score, self.temperature))
        return _clamp(_piecewise(raw_score, self.knots))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _temperature_scale(raw_score: float, temperature: float) -> float:
    """Logistic temperature scaling.

    Guards the logit at the boundaries: a raw score of exactly 0 or 1 would
    otherwise produce an infinite logit and a NaN result — which in practice is
    the one input a detector is most likely to emit.
    """
    epsilon = 1e-6
    bounded = min(1.0 - epsilon, max(epsilon, raw_score))
    logit = math.log(bounded / (1.0 - bounded))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _piecewise(raw_score: float, knots: tuple[tuple[float, float], ...]) -> float:
    if raw_score <= knots[0][0]:
        return knots[0][1]
    if raw_score >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False):
        if x0 <= raw_score <= x1:
            span = x1 - x0
            if span <= 0:
                return y1
            ratio = (raw_score - x0) / span
            return y0 + ratio * (y1 - y0)
    return raw_score


class CalibrationRegistry:
    """Holds profiles per (model, version). Read-mostly, snapshot-swapped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[tuple[ModelId, str], CalibrationProfile] = {}

    def register(self, profile: CalibrationProfile) -> None:
        with self._lock:
            profiles = dict(self._profiles)
            profiles[(profile.model_id, profile.model_version)] = profile
            self._profiles = profiles

    def get(self, model_id: ModelId, version: str) -> CalibrationProfile | None:
        """``None`` means uncalibrated — which the platform states rather than
        papering over with an identity transform."""
        return self._profiles.get((model_id, version))

    def profiles(self) -> tuple[CalibrationProfile, ...]:
        return tuple(self._profiles.values())
