"""Ground truth, predictions, and the vocabulary they share.

**Ground truth describes reality.** It is never derived from a detector, a VLM,
or a compliance result — if it were, an evaluation would measure the system
against itself and report agreement as accuracy. Every label here comes from a
human looking at a frame.

### Why three states and not two

``PRESENT`` / ``ABSENT`` / ``NOT_VISIBLE`` are three different facts about the
world, and collapsing the third into the second is the failure this whole
evaluation exists to catch. A chef whose hands are inside a pot is not a chef
without gloves. Measuring with two states would score the system's most dangerous
error as correct.

### ``NOT_VISIBLE`` and ``UNKNOWN`` are also different

``NOT_VISIBLE`` — the evidence region could not be adequately observed. The
annotator could not see the hands.

``UNKNOWN`` — the semantic identity cannot be established, or the thing is
outside the taxonomy. The annotator saw the object clearly and it is not a class
this deployment recognises.

They are kept apart because they call for different engineering: the first is a
camera, crop or quality problem, the second is a taxonomy or model problem.

This module lives under ``tools/`` and is imported by no runtime code. It reads
Vision OS output; Vision OS does not know it exists.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"


class AttributeState(enum.Enum):
    """What is true of one attribute of one subject in one frame."""

    PRESENT = "present"
    ABSENT = "absent"
    NOT_VISIBLE = "not_visible"
    """The evidence region could not be adequately observed."""

    UNKNOWN = "unknown"
    """The identity cannot be established, or it is outside the taxonomy.
    Distinct from ``NOT_VISIBLE``: the annotator could see it and still cannot
    name it."""

    @property
    def is_decided(self) -> bool:
        """Whether this state asserts something about the world.

        The two undecided states are not failures — an annotator who marks
        ``NOT_VISIBLE`` has recorded a fact about the footage, and a system that
        agrees is correct.
        """
        return self in (AttributeState.PRESENT, AttributeState.ABSENT)


class FailureCategory(enum.Enum):
    """Where a wrong answer came from.

    ``UNKNOWN_FAILURE_REASON`` exists and is used deliberately. Forcing every
    failure into a named cause would produce a tidy distribution built partly on
    guesses, and the whole point of this measurement is to find out where the
    errors actually are.
    """

    DETECTION_FAILURE = "detection_failure"
    TRACKING_FAILURE = "tracking_failure"
    CROP_FAILURE = "crop_failure"
    LOW_RESOLUTION = "low_resolution"
    MOTION_BLUR = "motion_blur"
    OCCLUSION = "occlusion"
    TRUNCATION = "truncation"
    SEMANTIC_MODEL_FAILURE = "semantic_model_failure"
    VLM_FAILURE = "vlm_failure"
    COMPLIANCE_LOGIC_FAILURE = "compliance_logic_failure"
    UNKNOWN_HANDLING_FAILURE = "unknown_handling_failure"
    TEMPORAL_FAILURE = "temporal_failure"
    MODEL_CONFLICT = "model_conflict"
    UNKNOWN_FAILURE_REASON = "unknown_failure_reason"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized, matching the platform's own convention."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x1 < self.x2 <= 1.0 and 0.0 <= self.y1 < self.y2 <= 1.0):
            raise ValueError(f"degenerate or out-of-range box {self}")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union — how a prediction is matched to a truth.

        Matching by IoU rather than by id, because the system's object ids are
        its own invention and an annotator has no way to know them. A prediction
        is scored against the truth it overlaps most.
        """
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        overlap = (x2 - x1) * (y2 - y1)
        return overlap / (self.area + other.area - overlap)


@dataclass(frozen=True, slots=True)
class AnnotatedSubject:
    """One person a human looked at, in one frame."""

    subject_id: str
    """The **annotator's** id, not the platform's. Stable within a frame only."""

    box: BoundingBox
    attributes: Mapping[str, AttributeState]
    note: str = ""

    def state(self, attribute: str) -> AttributeState | None:
        return self.attributes.get(attribute)


@dataclass(frozen=True, slots=True)
class AnnotatedFrame:
    """One frame of real footage and everything a human recorded about it."""

    frame_id: str
    video_id: str
    camera_id: str
    restaurant_id: str
    frame_index: int
    timestamp_ms: float
    subjects: tuple[AnnotatedSubject, ...] = ()
    image_path: str = ""
    difficulty: str = ""
    """Free-text tag for the conditions — `motion_blur`, `distant`, `crowded`.
    Used to slice metrics by difficulty rather than to change any behaviour."""

    def to_wire(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "camera_id": self.camera_id,
            "restaurant_id": self.restaurant_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "image_path": self.image_path,
            "difficulty": self.difficulty,
            "subjects": [
                {
                    "subject_id": s.subject_id,
                    "box": asdict(s.box),
                    "attributes": {k: v.value for k, v in s.attributes.items()},
                    "note": s.note,
                }
                for s in self.subjects
            ],
        }

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> AnnotatedFrame:
        return cls(
            frame_id=str(raw["frame_id"]),
            video_id=str(raw["video_id"]),
            camera_id=str(raw.get("camera_id", "")),
            restaurant_id=str(raw.get("restaurant_id", "")),
            frame_index=int(raw.get("frame_index", 0)),
            timestamp_ms=float(raw.get("timestamp_ms", 0.0)),
            image_path=str(raw.get("image_path", "")),
            difficulty=str(raw.get("difficulty", "")),
            subjects=tuple(
                AnnotatedSubject(
                    subject_id=str(s["subject_id"]),
                    box=BoundingBox(**s["box"]),
                    attributes={
                        k: AttributeState(v) for k, v in s.get("attributes", {}).items()
                    },
                    note=str(s.get("note", "")),
                )
                for s in raw.get("subjects", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class PredictedSubject:
    """What Vision OS said about one subject, with the evidence behind it.

    Every field beyond the states exists so a wrong answer can be traced to the
    pixels that produced it. A failure nobody can open is a failure nobody can
    fix.
    """

    object_id: str
    box: BoundingBox | None
    attributes: Mapping[str, AttributeState] = field(default_factory=dict)
    raw_values: Mapping[str, str] = field(default_factory=dict)
    """The platform's own vocabulary before mapping — `hairnet`, `not_visible`.
    Kept so a mapping bug is distinguishable from a model error."""

    crop_ids: Mapping[str, str] = field(default_factory=dict)
    crop_size: Mapping[str, str] = field(default_factory=dict)
    quality: Mapping[str, str] = field(default_factory=dict)
    """Per attribute: the gate's verdict, or its rejection reason."""

    skip_reason: Mapping[str, str] = field(default_factory=dict)
    model_id: str = ""
    model_version: str = ""
    vlm_used: bool = False
    detector_class: str = ""
    detector_confidence: float = 0.0

    def state(self, attribute: str) -> AttributeState | None:
        return self.attributes.get(attribute)


@dataclass(frozen=True, slots=True)
class PredictedFrame:
    frame_id: str
    video_id: str
    subjects: tuple[PredictedSubject, ...] = ()
    vlm_calls: int = 0
    vlm_call_reasons: Mapping[str, int] = field(default_factory=dict)


def load_annotations(path: Path | str) -> tuple[AnnotatedFrame, ...]:
    """Read an annotation file. Raises rather than guessing at a bad one.

    A malformed annotation silently skipped would shrink the evaluation set
    without saying so, and the resulting metric would describe a dataset nobody
    chose.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"annotation file declares schema_version "
            f"{document.get('schema_version')!r}; this tool reads {SCHEMA_VERSION}"
        )
    return tuple(AnnotatedFrame.from_wire(f) for f in document.get("frames", ()))


def save_annotations(
    frames: Iterable[AnnotatedFrame], path: Path | str, *, source: str = ""
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": source,
                "annotated_by": "human_visual_inspection",
                "frames": [f.to_wire() for f in frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


__all__ = [
    "SCHEMA_VERSION",
    "AnnotatedFrame",
    "AnnotatedSubject",
    "AttributeState",
    "BoundingBox",
    "FailureCategory",
    "PredictedFrame",
    "PredictedSubject",
    "load_annotations",
    "save_annotations",
]
