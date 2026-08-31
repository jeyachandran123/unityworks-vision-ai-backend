"""The authoritative PPE annotation contract — P9 schema v2.0.0.

### What changed from v1.0.0, and why

`datasets/kitchen-01/annotations/kitchen-01.json` (schema 1.0.0) records one
value per attribute:

```json
"attributes": {"head_covering": "present", "hand_covering": "not_visible"}
```

That conflates two different questions. `not_visible` is doing double duty as an
*observability* answer inside an *attribute* domain, which is why an annotator —
and a model — can slide between "I could not see it" and "it was not there". The
whole programme's central failure mode lives in that ambiguity.

v2.0.0 separates them. Every region carries **two** independent fields:

```
observability   VISIBLE | NOT_VISIBLE | UNCERTAIN     was the region assessable?
state           PRESENT | ABSENT | UNCERTAIN | NOT_EVALUATED    what was on it?
```

and the validator refuses the combinations that cannot both be true. An
annotation asserting `NOT_VISIBLE` + `ABSENT` is not a judgement call to be
weighed later; it is a contradiction, and it is rejected at write time.

### Hands are two regions, not one

v1.0.0 had a single `hand_covering`. A worker with one gloved hand and one bare
hand could not be described. P9 splits `LEFT_HAND` and `RIGHT_HAND`, because the
policy question *"are the hands covered"* has no single answer for that worker,
and forcing one destroys the example that matters most.

Left and right are from **the camera's point of view**, stated here because the
alternative — the subject's own left and right — is unrecoverable from a single
frame when the subject faces away.

### Extensibility

`Region` and the attribute vocabulary are open by design: adding `APRON` or
`FOOTWEAR` later adds enum members and leaves every existing annotation valid.
Nothing in the validator enumerates the *set* of regions an annotation must
carry — only that the regions it does carry are internally consistent.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any

#: Bumped when the meaning of a field changes. Never reused.
ANNOTATION_SCHEMA_VERSION = "2.0.0"


class Observability(enum.Enum):
    """Could this region be assessed at all?

    Answered **before** and **independently of** what is on it. A region can be
    perfectly visible and carry nothing, or be entirely hidden while carrying
    something — and only one of those is knowable from the image.
    """

    VISIBLE = "visible"
    """The region is in frame and legible enough to judge."""

    NOT_VISIBLE = "not_visible"
    """Out of frame, cut off, turned away, occluded, too dark, too small, or
    too blurred. **Not a statement about PPE.**"""

    UNCERTAIN = "uncertain"
    """The annotator could not decide between the two above. A first-class
    answer, not a failure to answer: it is the honest label for the marginal
    crops that produce most disagreement, and hiding them inside VISIBLE is how
    a benchmark quietly acquires noise it cannot see."""


class AttributeState(enum.Enum):
    """What was on the region — only meaningful when it was observable."""

    PRESENT = "present"
    """The required item is there."""

    ABSENT = "absent"
    """The region was seen, and the item is **not** there. The violation class."""

    UNCERTAIN = "uncertain"
    """Visible, but the annotator could not tell what was on it."""

    NOT_EVALUATED = "not_evaluated"
    """The region was not observable, so no attribute question was asked. The
    **only** legal state when observability is not VISIBLE."""


class Region(enum.Enum):
    """Body regions a PPE requirement can attach to.

    Open set. Adding a member does not invalidate existing annotations.
    """

    HEAD = "head"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"

    @property
    def attribute(self) -> str:
        """The PPE item this region is inspected for."""
        return {
            Region.HEAD: "head_covering",
            Region.FACE: "face_covering",
            Region.LEFT_HAND: "glove",
            Region.RIGHT_HAND: "glove",
        }[self]


class LabelProvenance(enum.Enum):
    """Where a label came from. **The most important field in the schema.**

    A benchmark whose labels came from the thing it is benchmarking measures
    nothing. This field makes that failure visible instead of invisible, and the
    validator refuses to admit anything but `HUMAN_VERIFIED` into an evaluation
    split.
    """

    HUMAN_VERIFIED = "human_verified"
    """A person looked at this crop and decided. The only provenance admissible
    as ground truth."""

    HUMAN_ADJUDICATED = "human_adjudicated"
    """Two annotators disagreed and a third resolved it. Records that the
    example was hard, which is worth knowing when it later fails."""

    MACHINE_PROPOSED = "machine_proposed"
    """A model suggested this label and **no human has confirmed it**. Usable to
    prioritise review queues. **Never** ground truth, never in a test split, and
    never counted in a published metric."""

    DETECTOR_DERIVED = "detector_derived"
    """Geometry that came from a detector rather than from a person. The
    kitchen-01 boxes are this, which is why that dataset cannot measure
    detection recall — the annotation says so rather than leaving it to be
    rediscovered."""


class QualityStatus(enum.Enum):
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    DISPUTED = "disputed"
    """Annotators disagreed and no adjudication has happened yet. Excluded from
    evaluation until resolved — never silently resolved by picking one."""

    REJECTED = "rejected"


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HARD_TEST = "hard_test"
    """Safety-critical and difficult cases. **Protected from iterative tuning**:
    a run against it is recorded, and repeated tuning against it is a process
    failure the manifest can evidence."""

    UNASSIGNED = "unassigned"
    """In the pool, not yet allocated. The default, so a new sample cannot leak
    into a split by forgetting to set one."""


#: Hard-case tags. Free to extend; the validator warns on unknown tags rather
#: than rejecting, because a new failure mode should be cheap to record.
HARD_CASE_TAGS = frozenset(
    {
        # head
        "head_partial", "head_cut_off", "head_out_of_frame", "profile",
        "back_of_head", "head_bent_down", "cap_not_hairnet", "covering_partial",
        # face
        "face_occluded", "face_side", "face_back", "face_too_small",
        "hand_over_face",
        # hands
        "hand_out_of_frame", "hand_behind_object", "hand_behind_person",
        "hand_holding_equipment", "hand_holding_food", "hand_in_container",
        "crossed_arms", "glove_partial", "glove_skin_coloured", "glove_dark",
        # global
        "low_resolution", "motion_blur", "low_light", "strong_light",
        "occluded_by_equipment", "occluded_by_person", "multiple_people",
        "small_subject", "large_subject", "distant", "truncated_by_frame",
    }
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RegionAnnotation:
    """One region of one subject, in one frame.

    The invariant this type exists to carry is checked in `__post_init__` rather
    than by a linter or a reviewer: **a region that was not observable has no
    attribute state**. Constructing the contradiction raises.
    """

    region: Region
    observability: Observability
    state: AttributeState
    hard_case_tags: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.observability is Observability.VISIBLE:
            if self.state is AttributeState.NOT_EVALUATED:
                raise ValueError(
                    f"{self.region.value}: observable regions must be evaluated — "
                    f"VISIBLE + NOT_EVALUATED is an annotation that was skipped, "
                    f"not an observation. Use UNCERTAIN if the answer was unclear."
                )
        elif self.state is not AttributeState.NOT_EVALUATED:
            raise ValueError(
                f"{self.region.value}: observability is "
                f"{self.observability.value!r} but state is {self.state.value!r}. "
                f"A region nobody could see cannot carry a decided attribute — "
                f"this is the exact confusion the whole schema exists to prevent."
            )

    @property
    def is_ground_truth_positive(self) -> bool:
        return (
            self.observability is Observability.VISIBLE
            and self.state is AttributeState.PRESENT
        )

    @property
    def is_ground_truth_violation(self) -> bool:
        """A genuine, observed PPE breach. The class the corpus is short of."""
        return (
            self.observability is Observability.VISIBLE
            and self.state is AttributeState.ABSENT
        )


@dataclass(frozen=True, slots=True)
class SubjectAnnotation:
    """One person, in one frame, with every region that was annotated."""

    sample_id: str
    subject_id: str
    session_id: str
    camera_id: str
    frame_id: str
    source_video: str
    box: tuple[float, float, float, float]
    """Normalised `(x1, y1, x2, y2)`. Provenance is `box_provenance`, because a
    detector-proposed box and a human-drawn one support different claims."""

    regions: tuple[RegionAnnotation, ...]
    label_provenance: LabelProvenance
    box_provenance: LabelProvenance
    annotator: str
    annotated_at: str
    annotation_version: str = ANNOTATION_SCHEMA_VERSION
    split: Split = Split.UNASSIGNED
    quality_status: QualityStatus = QualityStatus.NEEDS_REVIEW
    ambiguous: bool = False
    note: str = ""

    identity_verified: bool = False
    """Whether `subject_id` names a **person**, or merely a slot in a frame.

    Defaults to False, and the default is load-bearing. kitchen-01's `s0..s4`
    are detection-order indices assigned per frame: nothing in that dataset
    asserts that `s0` in frame 60 and `s0` in frame 1740 are the same human, and
    the boxes are only *suggestive* of it.

    Splitting on an unverified id would put what might be one person on both
    sides of a train/test boundary while every leakage check reported clean —
    the failure would be invisible precisely because the check would pass. So an
    unverified id does not participate in `group_key` at all, which collapses
    such a session into a single indivisible group and makes the thinness
    visible instead."""

    def __post_init__(self) -> None:
        for name in ("sample_id", "subject_id", "session_id", "camera_id", "frame_id"):
            value = getattr(self, name)
            if not _ID.match(str(value)):
                raise ValueError(f"{name}={value!r} is not a legal identifier")
        if not self.regions:
            raise ValueError(
                f"{self.sample_id}: a subject with no annotated region records "
                f"nothing; omit the subject instead of writing an empty one"
            )
        seen = [r.region for r in self.regions]
        if len(seen) != len(set(seen)):
            raise ValueError(f"{self.sample_id}: duplicate region annotations")
        x1, y1, x2, y2 = self.box
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"{self.sample_id}: box {self.box} is not a valid unit box")
        if not self.annotator:
            raise ValueError(
                f"{self.sample_id}: every annotation names its annotator — an "
                f"unattributable label cannot be audited or re-reviewed"
            )

    @property
    def group_key(self) -> str:
        """The unit a split must not straddle.

        Session **and** subject when the identity is verified: the same person on
        two different days is two appearances, but the same person seconds apart
        is one example photographed twice, and putting those either side of a
        boundary is the leakage this key exists to prevent.

        When the identity is *not* verified the subject is dropped from the key
        and the whole session becomes one group. That is deliberately
        pessimistic: it is the only assumption that cannot produce silent
        leakage, and a corpus that becomes unsplittable under it was never
        splittable — it just looked splittable.
        """
        if self.identity_verified:
            return f"{self.session_id}::{self.subject_id}"
        return f"{self.session_id}::*"

    @property
    def is_evaluation_grade(self) -> bool:
        return (
            self.label_provenance
            in (LabelProvenance.HUMAN_VERIFIED, LabelProvenance.HUMAN_ADJUDICATED)
            and self.quality_status is QualityStatus.ACCEPTED
        )

    def region_of(self, region: Region) -> RegionAnnotation | None:
        return next((r for r in self.regions if r.region is region), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "source_video": self.source_video,
            "box": list(self.box),
            "split": self.split.value,
            "label_provenance": self.label_provenance.value,
            "box_provenance": self.box_provenance.value,
            "quality_status": self.quality_status.value,
            "ambiguous": self.ambiguous,
            "identity_verified": self.identity_verified,
            "annotator": self.annotator,
            "annotated_at": self.annotated_at,
            "annotation_version": self.annotation_version,
            "note": self.note,
            "regions": [
                {
                    "region": r.region.value,
                    "attribute": r.region.attribute,
                    "observability": r.observability.value,
                    "state": r.state.value,
                    "hard_case_tags": list(r.hard_case_tags),
                    "note": r.note,
                }
                for r in self.regions
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SubjectAnnotation:
        return cls(
            sample_id=payload["sample_id"],
            subject_id=payload["subject_id"],
            session_id=payload["session_id"],
            camera_id=payload["camera_id"],
            frame_id=payload["frame_id"],
            source_video=payload.get("source_video", ""),
            box=tuple(payload["box"]),
            regions=tuple(
                RegionAnnotation(
                    region=Region(r["region"]),
                    observability=Observability(r["observability"]),
                    state=AttributeState(r["state"]),
                    hard_case_tags=tuple(r.get("hard_case_tags", ())),
                    note=r.get("note", ""),
                )
                for r in payload["regions"]
            ),
            label_provenance=LabelProvenance(payload["label_provenance"]),
            box_provenance=LabelProvenance(payload["box_provenance"]),
            annotator=payload["annotator"],
            annotated_at=payload["annotated_at"],
            annotation_version=payload.get("annotation_version", ANNOTATION_SCHEMA_VERSION),
            split=Split(payload.get("split", "unassigned")),
            quality_status=QualityStatus(payload.get("quality_status", "needs_review")),
            ambiguous=bool(payload.get("ambiguous", False)),
            identity_verified=bool(payload.get("identity_verified", False)),
            note=payload.get("note", ""),
        )
