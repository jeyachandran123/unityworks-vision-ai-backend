"""The confidence model (02_VISION_OBJECT_MODEL section 7).

Confidence is the most abused field in computer vision. A YOLO objectness score, a
tracker association cost, and a VLM's self-reported certainty are three
incomparable quantities that every system stuffs into one float named
``confidence`` and then compares.

Two rules are load-bearing:

* **``raw_score`` is always preserved.** Calibration mappings improve over time;
  keeping the raw score means historical records can be re-calibrated without
  re-running inference.
* **Uncalibrated scores are never compared across models.** ``calibrated`` is a
  field, not an assumption, so a consumer can tell the difference.

A consumer that wrote ``if confidence > 0.7`` against a 2026 detector will
silently change behaviour when a 2029 detector with a different score
distribution is swapped in — unless confidence means the same thing across both.
That is what this type exists to guarantee.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .ids import CalibrationId


class ConfidenceSemantics(enum.Enum):
    """What the number means (02_VOM section 7.1)."""

    DETECTION_PRESENCE = "detection_presence"
    """P(an object of this class is present here)."""

    CLASSIFICATION = "classification"
    """P(class | object present)."""

    ASSOCIATION = "association"
    """P(this detection continues this track). Flow 3."""

    IDENTITY = "identity"
    """P(this track is this object). Flow 3/7."""

    ATTRIBUTE = "attribute"
    """P(this attribute claim is true). Flow 5."""

    SELF_REPORTED = "self_reported"
    """The model said so about itself. Weakest; never comparable."""

    @property
    def is_comparable_across_models(self) -> bool:
        """Whether ordering across heterogeneous producers is ever meaningful.

        Even for these, comparison additionally requires ``calibrated`` — the
        semantics tell you *what* is being measured, calibration tells you
        whether the scale is shared.
        """
        return self is not ConfidenceSemantics.SELF_REPORTED


@dataclass(frozen=True, slots=True)
class Confidence:
    """A score with its meaning, its provenance, and its original value."""

    value: float
    semantics: ConfidenceSemantics
    calibrated: bool = False
    calibration_id: CalibrationId | None = None
    raw_score: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.value}")
        if self.calibrated and self.calibration_id is None:
            raise ValueError("a calibrated confidence must name the calibration applied")
        if self.raw_score is not None and not 0.0 <= self.raw_score <= 1.0:
            raise ValueError(f"raw_score must be in [0,1], got {self.raw_score}")

    @classmethod
    def uncalibrated(
        cls, raw_score: float, semantics: ConfidenceSemantics
    ) -> Confidence:
        """The honest default: the model's own number, declared uncalibrated."""
        return cls(
            value=raw_score,
            semantics=semantics,
            calibrated=False,
            calibration_id=None,
            raw_score=raw_score,
        )

    def comparable_with(self, other: Confidence) -> bool:
        """Whether two confidences may be ordered against one another.

        Requires the same semantics, both calibrated, and the same calibration
        profile. Anything less and the comparison is meaningless — which is a
        fact the platform states rather than letting a consumer discover it.
        """
        return (
            self.semantics is other.semantics
            and self.semantics.is_comparable_across_models
            and self.calibrated
            and other.calibrated
            and self.calibration_id == other.calibration_id
        )
