"""Phase 5 — a dataset that can measure whether a reported violation is real.

Every dataset before this one was built from **detector proposals**: a human
confirmed the people YOLO found. That makes two questions unanswerable, and both
matter more than anything the old dataset could measure.

*A person the detector never proposed was never annotated*, so detection recall
cannot be computed — a system that finds one worker perfectly scores 100 %.

*And there were no violations in it at all.* With zero uncovered heads, every
`ABSENT` the system produced was wrong by construction, so `ABSENT` precision —
**how often a reported violation is real**, the number that decides whether this
system is safe to deploy — was undefined rather than poor.

So here a frame is annotated **independently of any model**. The annotator marks
every person they can see; the detector's opinion is recorded next to that, never
instead of it.

### The rule this module exists to enforce

    ABSENT means "the region was observable and the PPE was not there".

    It never means "I could not see".

A worker whose hands are inside a stockpot is not a worker without gloves.
`validate()` rejects that annotation rather than trusting an annotator to
remember it at 2 a.m. on the four-hundredth frame — the failure it guards against
is one of attention, not of understanding.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import AttributeState, BoundingBox

SCHEMA_VERSION = "2.0.0"


class PersonVisibility(enum.Enum):
    """How much of a person the camera shows."""

    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"
    """Known to be present — an arm behind a counter, a body behind a door — but
    not usable as a subject. Recorded rather than dropped so a detector is not
    penalised for missing someone no camera could resolve."""


class Observability(enum.Enum):
    """Whether the body region an attribute lives on can be inspected.

    Separate from the attribute state, and the separation is the point: a head
    can be plainly observable and its covering still be ambiguous, and a head can
    be entirely unobservable while the worker is certainly wearing something.
    Collapsing the two produces exactly the false violations this phase measures.
    """

    OBSERVABLE = "observable"
    PARTIALLY_OBSERVABLE = "partially_observable"
    NOT_OBSERVABLE = "not_observable"

    @property
    def permits_a_decided_state(self) -> bool:
        """Whether PRESENT/ABSENT may be asserted for this region.

        ``PARTIALLY_OBSERVABLE`` permits it: half a head is often enough to see a
        hairnet, and refusing there would discard real evidence. The annotator
        still has ``UNKNOWN`` for when it is not enough.
        """
        return self is not Observability.NOT_OBSERVABLE


@dataclass(frozen=True, slots=True)
class PpeAnnotation:
    """One attribute of one person: what was seen, and whether it could be."""

    state: AttributeState
    observability: Observability
    note: str = ""
    ambiguous: bool = False
    """Set when the annotator reached a state but would not defend it. Kept so a
    borderline call can be excluded from a headline metric without being deleted
    from the record."""


@dataclass(frozen=True, slots=True)
class AnnotatedPerson:
    """One person a human saw, whether or not any model found them."""

    person_id: str
    box: BoundingBox
    visibility: PersonVisibility = PersonVisibility.VISIBLE
    occluded: bool = False
    truncated: bool = False
    ppe: Mapping[str, PpeAnnotation] = field(default_factory=dict)
    note: str = ""

    detected_by_reference_model: bool | None = None
    """Whether the bundled detector proposed this person. **Recorded after
    annotation, never before**, and used only to compute detection recall. An
    annotator who saw the model's proposals first would be anchored by them."""

    def state(self, attribute: str) -> AttributeState | None:
        entry = self.ppe.get(attribute)
        return entry.state if entry else None


@dataclass(frozen=True, slots=True)
class PpeFrame:
    """One frame, and every person in it."""

    frame_id: str
    video_id: str
    camera_id: str
    restaurant_id: str
    frame_index: int
    timestamp_ms: float
    persons: tuple[AnnotatedPerson, ...] = ()
    image_path: str = ""
    conditions: tuple[str, ...] = ()
    """Free-text condition tags — `motion_blur`, `backlit`, `crowded`, `distant`.
    Used to slice metrics by difficulty, never to change behaviour."""

    annotated_by: str = ""
    annotation_source: str = "human_visual_inspection"


# --- validation ------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    frame_id: str
    person_id: str
    attribute: str
    rule: str
    detail: str

    def __str__(self) -> str:
        where = f"{self.frame_id}/{self.person_id}"
        if self.attribute:
            where += f".{self.attribute}"
        return f"[{self.rule}] {where}: {self.detail}"


def validate(
    frames: Sequence[PpeFrame], *, attributes: Sequence[str] = ()
) -> list[ValidationIssue]:
    """Every rule that keeps a violation metric honest. Returns all issues.

    Returns rather than raises, and returns *all* of them: an annotator fixing a
    batch needs the whole list, not the first error repeated four hundred times.
    """
    issues: list[ValidationIssue] = []
    known = set(attributes)
    seen_frames: set[str] = set()

    for frame in frames:
        if frame.frame_id in seen_frames:
            issues.append(ValidationIssue(
                frame.frame_id, "", "", "duplicate_frame",
                "this frame_id appears more than once; one of them will be "
                "silently discarded when the file is loaded into a dict",
            ))
        seen_frames.add(frame.frame_id)

        if frame.annotation_source != "human_visual_inspection":
            issues.append(ValidationIssue(
                frame.frame_id, "", "", "non_human_annotation",
                f"annotation_source is {frame.annotation_source!r}; ground truth "
                f"derived from a model measures the system against itself",
            ))

        seen_people: set[str] = set()
        for person in frame.persons:
            if not person.person_id:
                issues.append(ValidationIssue(
                    frame.frame_id, "", "", "missing_person_id",
                    "a person without an id cannot be matched across frames",
                ))
            if person.person_id in seen_people:
                issues.append(ValidationIssue(
                    frame.frame_id, person.person_id, "", "duplicate_person",
                    "the same person_id is annotated twice in one frame",
                ))
            seen_people.add(person.person_id)

            for attribute, entry in person.ppe.items():
                if known and attribute not in known:
                    issues.append(ValidationIssue(
                        frame.frame_id, person.person_id, attribute,
                        "unknown_attribute",
                        f"not declared for this dataset; declared: {sorted(known)}",
                    ))

                # THE rule. An unobservable region can never carry a decided
                # state, because that is precisely how missing evidence becomes
                # a violation against a compliant worker.
                if entry.state.is_decided and not entry.observability.permits_a_decided_state:
                    issues.append(ValidationIssue(
                        frame.frame_id, person.person_id, attribute,
                        "decided_state_without_observability",
                        f"state is {entry.state.value.upper()} while observability is "
                        f"{entry.observability.value}; if the region could not be "
                        f"observed the state must be NOT_VISIBLE, never "
                        f"{entry.state.value.upper()}",
                    ))

                # The mirror image: claiming the region was unobservable while
                # recording a state that asserts otherwise.
                if (
                    entry.state is AttributeState.NOT_VISIBLE
                    and entry.observability is Observability.OBSERVABLE
                    and not entry.note
                ):
                    issues.append(ValidationIssue(
                        frame.frame_id, person.person_id, attribute,
                        "unexplained_refusal",
                        "marked NOT_VISIBLE while the region is fully observable; "
                        "this may be right (a covered head behind a pillar) but it "
                        "needs a note saying why",
                    ))

            if person.visibility is PersonVisibility.NOT_VISIBLE and person.ppe:
                issues.append(ValidationIssue(
                    frame.frame_id, person.person_id, "", "ppe_on_invisible_person",
                    "PPE recorded for a person marked NOT_VISIBLE",
                ))

    return issues


# --- quality report --------------------------------------------------------- #

#: Below this many examples of a state, a rate computed from it is noise.
#:
#: Not a statistical result — a working floor, chosen so that one annotation
#: error cannot move a reported percentage by more than a few points. Reported
#: alongside every verdict so a reader can apply their own.
MINIMUM_FOR_A_RATE = 20


@dataclass(frozen=True, slots=True)
class AttributeCoverage:
    attribute: str
    counts: Mapping[str, int]

    def count(self, state: AttributeState) -> int:
        return self.counts.get(state.value, 0)

    @property
    def can_measure_absent_precision(self) -> bool:
        """Whether "how often is a reported violation real" is answerable.

        Needs real violations in the footage. Without them every ABSENT the
        system emits is wrong by construction and the metric is undefined, not
        poor — which is the trap this whole phase exists to escape.
        """
        return self.count(AttributeState.ABSENT) >= MINIMUM_FOR_A_RATE

    @property
    def can_measure_absent_recall(self) -> bool:
        return self.count(AttributeState.ABSENT) >= MINIMUM_FOR_A_RATE

    @property
    def can_measure_present_precision(self) -> bool:
        return self.count(AttributeState.PRESENT) >= MINIMUM_FOR_A_RATE

    def verdict(self) -> str:
        if not self.can_measure_absent_precision:
            return (
                f"INSUFFICIENT DATA — {self.count(AttributeState.ABSENT)} ABSENT "
                f"examples, need {MINIMUM_FOR_A_RATE}"
            )
        return "measurable"


def coverage(
    frames: Sequence[PpeFrame], attributes: Sequence[str]
) -> tuple[AttributeCoverage, ...]:
    out = []
    for attribute in attributes:
        counts: dict[str, int] = {}
        for frame in frames:
            for person in frame.persons:
                state = person.state(attribute)
                if state is not None:
                    counts[state.value] = counts.get(state.value, 0) + 1
        out.append(AttributeCoverage(attribute, counts))
    return tuple(out)


def quality_report(
    frames: Sequence[PpeFrame], attributes: Sequence[str]
) -> dict[str, Any]:
    """What this dataset can and cannot measure — written before any model runs.

    Deliberately produced first. A quality report written after the metrics is a
    rationalisation of them.
    """
    people = [p for f in frames for p in f.persons]
    undetected = sum(1 for p in people if p.detected_by_reference_model is False)
    known_detection = sum(1 for p in people if p.detected_by_reference_model is not None)
    cov = coverage(frames, attributes)

    return {
        "schema_version": SCHEMA_VERSION,
        "restaurants": sorted({f.restaurant_id for f in frames}),
        "cameras": sorted({f.camera_id for f in frames}),
        "videos": sorted({f.video_id for f in frames}),
        "frames": len(frames),
        "annotated_persons": len(people),
        "unique_person_ids": len({p.person_id for p in people}),
        "conditions_present": sorted({c for f in frames for c in f.conditions}),
        "attributes": {
            c.attribute: {
                "counts": dict(c.counts),
                "verdict": c.verdict(),
                "absent_precision_measurable": c.can_measure_absent_precision,
                "absent_recall_measurable": c.can_measure_absent_recall,
                "present_precision_measurable": c.can_measure_present_precision,
            }
            for c in cov
        },
        "detection_recall_measurable": known_detection == len(people) and len(people) > 0,
        "persons_missed_by_reference_detector": undetected,
        "minimum_examples_for_a_rate": MINIMUM_FOR_A_RATE,
    }


# --- persistence ------------------------------------------------------------ #


def to_wire(frame: PpeFrame) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "video_id": frame.video_id,
        "camera_id": frame.camera_id,
        "restaurant_id": frame.restaurant_id,
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "image_path": frame.image_path,
        "conditions": list(frame.conditions),
        "annotated_by": frame.annotated_by,
        "annotation_source": frame.annotation_source,
        "persons": [
            {
                "person_id": p.person_id,
                "box": asdict(p.box),
                "visibility": p.visibility.value,
                "occluded": p.occluded,
                "truncated": p.truncated,
                "note": p.note,
                "detected_by_reference_model": p.detected_by_reference_model,
                "ppe": {
                    key: {
                        "state": entry.state.value,
                        "observability": entry.observability.value,
                        "note": entry.note,
                        "ambiguous": entry.ambiguous,
                    }
                    for key, entry in p.ppe.items()
                },
            }
            for p in frame.persons
        ],
    }


def from_wire(raw: Mapping[str, Any]) -> PpeFrame:
    return PpeFrame(
        frame_id=str(raw["frame_id"]),
        video_id=str(raw["video_id"]),
        camera_id=str(raw.get("camera_id", "")),
        restaurant_id=str(raw.get("restaurant_id", "")),
        frame_index=int(raw.get("frame_index", 0)),
        timestamp_ms=float(raw.get("timestamp_ms", 0.0)),
        image_path=str(raw.get("image_path", "")),
        conditions=tuple(raw.get("conditions", ())),
        annotated_by=str(raw.get("annotated_by", "")),
        annotation_source=str(raw.get("annotation_source", "human_visual_inspection")),
        persons=tuple(
            AnnotatedPerson(
                person_id=str(p["person_id"]),
                box=BoundingBox(**p["box"]),
                visibility=PersonVisibility(p.get("visibility", "visible")),
                occluded=bool(p.get("occluded", False)),
                truncated=bool(p.get("truncated", False)),
                note=str(p.get("note", "")),
                detected_by_reference_model=p.get("detected_by_reference_model"),
                ppe={
                    key: PpeAnnotation(
                        state=AttributeState(entry["state"]),
                        observability=Observability(entry["observability"]),
                        note=str(entry.get("note", "")),
                        ambiguous=bool(entry.get("ambiguous", False)),
                    )
                    for key, entry in p.get("ppe", {}).items()
                },
            )
            for p in raw.get("persons", ())
        ),
    )


def save(frames: Iterable[PpeFrame], path: Path | str, *, source: str = "") -> Path:
    """Write annotations, refusing to write an invalid set.

    Validation happens on the way out rather than on the way in: an invalid file
    that reaches the repository will eventually be loaded by something that does
    not check.
    """
    frames = list(frames)
    issues = validate(frames)
    blocking = [i for i in issues if i.rule == "decided_state_without_observability"]
    if blocking:
        raise ValueError(
            "refusing to write annotations that assert PPE state for an "
            f"unobservable region ({len(blocking)} of them):\n  "
            + "\n  ".join(str(i) for i in blocking[:5])
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": source,
                "frames": [to_wire(f) for f in frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def load(path: Path | str) -> tuple[PpeFrame, ...]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"annotation file declares schema_version "
            f"{document.get('schema_version')!r}; this tool reads {SCHEMA_VERSION}"
        )
    return tuple(from_wire(f) for f in document.get("frames", ()))


__all__ = [
    "MINIMUM_FOR_A_RATE",
    "SCHEMA_VERSION",
    "AnnotatedPerson",
    "AttributeCoverage",
    "Observability",
    "PersonVisibility",
    "PpeAnnotation",
    "PpeFrame",
    "ValidationIssue",
    "coverage",
    "load",
    "quality_report",
    "save",
    "validate",
]
