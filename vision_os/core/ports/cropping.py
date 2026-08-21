"""P12 ``TriggerPolicyPort``, P13 ``QualityEstimatorPort``, P14 ``CropStrategyPort``.

All three belong to M8 (06_PORTS section 2), and all three ship with default
adapters — unlike P11, the roadmap does not defer them. §M8 describes the trigger
set as *"a default policy, fully replaceable"* and the quality estimator as
*"heuristic sharpness/scale today; learned quality predictors later"*.

**The ceiling holds inside every one of them.** §M8 is explicit:

> *A trigger policy may say "re-look because appearance changed by 0.4 cosine
> distance." It may never say "re-look because this is the kitchen." Priority is
> expressed as an opaque class supplied by configuration; the reason a class
> exists lives with the consumer (V1/V2).*

That is why every port below takes ids and measurements and returns decisions —
and why none of them receives a region *label*, a class *meaning*, or a
business justification for a priority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.confidence import Confidence
from ..model.crop import CropTransform, SkipReason, TriggerReason
from ..model.detection import QualityGrades
from ..model.ids import AttributeKey, CameraId, ClassId, ObjectId, RegionId
from ..model.space import Box
from ..model.timebase import Duration, Instant

TRIGGER_POLICY_PORT_VERSION = "1.0.0"
QUALITY_ESTIMATOR_PORT_VERSION = "1.0.0"
CROP_STRATEGY_PORT_VERSION = "1.0.0"


# --- P12 TriggerPolicyPort ---------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttributeStatus:
    """What the platform already knows about one attribute of one object.

    ``observed_at is None`` means never computed — distinct from computed and
    stale, which is why the field is optional rather than defaulting to zero.
    """

    key: AttributeKey
    observed_at: Instant | None = None
    confidence: float | None = None
    valid_until: Instant | None = None

    def age(self, now: Instant) -> Duration | None:
        if self.observed_at is None:
            return None
        return Duration(max(0, now.ns - self.observed_at.ns))

    def is_stale(self, now: Instant, freshness: Duration) -> bool:
        """Stale against a *demand's* freshness, not a global one.

        Two demands can want the same attribute at different rates, and the
        stricter one wins — which only works if staleness is evaluated per
        demand rather than stamped on the attribute.
        """
        age = self.age(now)
        return age is None or age.ns > freshness.ns


#: A detector's label space **behaves differently** when it meets something
#: outside itself, and a policy that cannot tell the two apart cannot reason about
#: how much a class claim is worth.
#:
#: A closed-set detector scores a fixed vocabulary and returns the argmax; it has
#: no index meaning "none of these", so an object it was never trained on still
#: receives the nearest word it knows. An open-vocabulary detector is scored
#: against labels supplied at query time, so an absence from its answer carries
#: information a closed-set absence does not.
#:
#: These are the same two constants the detector provider already declares. They
#: are restated here rather than imported because ``core`` may not import an
#: adapter, and a policy reasoning about capability needs the vocabulary.
CLOSED_SET = "closed_set"
OPEN_VOCABULARY = "open_vocabulary"


@dataclass(frozen=True, slots=True)
class LabelSpaceView:
    """What the bound detector can and cannot name.

    Injected into M8 at composition rather than discovered, for the same reason
    ``CapabilityView`` is: the capability question has one owner, and the adapter
    that knows declares it honestly (adapter obligation A1).

    The empty default is the honest unknown. A deployment that has not declared
    its detector's label space gets ``covers() is None`` everywhere, and a policy
    written against that reads "cannot tell" rather than "outside the vocabulary"
    — the same distinction ``QualityGrades`` draws between unmeasured and zero
    (obligation Q2).
    """

    kind: str = ""
    """``closed_set`` | ``open_vocabulary`` | ``""`` for undeclared."""

    producible_classes: frozenset[ClassId] = frozenset()
    """The detector's vocabulary **as platform classes**, after any narrowing by
    configuration.

    Platform classes rather than the model's native labels, because a native
    label must never escape its adapter (port obligation D2) and because a policy
    compares this against a candidate's ``class_id``. Empty means undeclared,
    never "names nothing"."""

    @property
    def is_closed_set(self) -> bool:
        return self.kind == CLOSED_SET

    @property
    def is_declared(self) -> bool:
        return bool(self.kind)

    def covers(self, class_id: ClassId) -> bool | None:
        """Whether this class is inside the detector's declared vocabulary.

        ``None`` when nothing was declared. Returning ``False`` in that case
        would let an unconfigured deployment read as "every class is outside the
        detector's capability", which would send every object to verification —
        the brute-force pipeline this whole seam exists to prevent.

        Matching is hierarchical for the same reason class filters are: a
        detector producing ``vehicle.forklift`` covers a demand for ``vehicle``.
        """
        if not self.producible_classes:
            return None
        return any(
            class_id == produced or class_id.startswith(f"{produced}.")
            for produced in self.producible_classes
        )


@dataclass(frozen=True, slots=True)
class TriggerCandidate:
    """One object considered for analysis this frame.

    Carries measurements and ids only. Note what is absent: no region *label*,
    no class *meaning*, no business context. A policy cannot breach the ceiling
    because it is never handed the material to breach it with.
    """

    object_id: ObjectId
    camera_id: CameraId
    class_id: ClassId
    box: Box
    lifecycle: str
    identity_confidence: float

    first_seen: Instant
    last_confirmed: Instant
    observation_count: int
    region_ids: tuple[RegionId, ...] = ()

    attributes: Mapping[AttributeKey, AttributeStatus] = field(default_factory=dict)
    appearance_delta: float | None = None
    """How much the object's appearance changed since it was last seen, in [0,1].

    A **delta**, not a signature: the caller holds the previous scalar and hands
    the policy the difference, which keeps the policy stateless (obligation G6).

    ``None`` means *not measured* — the first sighting, or no appearance detector
    bound. Distinct from a measured delta of zero, which claims the appearance
    did not change; the platform cannot make that claim about something it has
    seen once (V8). The *meaning* of a change is never inferred."""

    last_analysed: Instant | None = None
    last_gate_rejection: bool = False
    """Whether the previous attempt was gate-rejected, which is what makes
    ``QUALITY_IMPROVED`` expressible."""

    entered_region_this_frame: bool = False
    lifecycle_changed_this_frame: bool = False
    estimated_quality: QualityGrades = QualityGrades()

    # --- what the detector claimed, and how much that claim is worth --------- #
    #
    # A policy deciding whether a class claim needs corroborating cannot reason
    # from ``identity_confidence``: that is P(this track is this object), and
    # presenting it as a classification score would compare two incomparable
    # quantities (02_VOM section 7.2). These four fields carry the detector's own
    # evidence, typed so the confusion is not expressible.

    class_confidence: Confidence | None = None
    """The detector's score for the class claim, with its semantics attached.

    ``CLASSIFICATION`` — P(class | object present) — as recorded on the object's
    ``class_history``. ``None`` means no class evidence has been retained, which
    is distinct from a low score and must not be read as one."""

    class_alternatives: tuple[tuple[ClassId, float], ...] = ()
    """Other classes this object has been called, with each one's share of the
    accumulated evidence, strongest first.

    A stable object seen forty times as one class and a flapping object split
    across three are different claims, and a single confidence number cannot tell
    them apart. Empty means no alternative was ever asserted."""

    label_space_kind: str = ""
    """The bound detector's label space, from ``LabelSpaceView.kind``. Empty when
    undeclared."""

    class_in_native_vocabulary: bool | None = None
    """Whether this class is inside the detector's declared vocabulary.

    ``None`` is *undeclared*, not *outside*. A policy must be able to tell "the
    detector cannot name this kind of thing" from "nobody told us what the
    detector can name", because the first is a reason to look again and the
    second is a reason to fix the configuration."""


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Whether to analyse a candidate, and why — or why not.

    **Exactly one of ``reason`` and ``skip`` is set.** A decision that carries
    neither is a candidate that vanishes, which is the failure invariant V8 and
    responsibility 7 exist to prevent.
    """

    object_id: ObjectId
    reason: TriggerReason | None = None
    skip: SkipReason | None = None
    attributes: tuple[AttributeKey, ...] = ()
    """Which attributes prompted the trigger, or went unsatisfied on a skip."""

    priority_class: str = ""
    demand_ids: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if (self.reason is None) == (self.skip is None):
            raise ValueError(
                f"a trigger decision for '{self.object_id}' must carry exactly one "
                f"of reason or skip; a candidate that produces neither becomes "
                f"invisible, which is what invariant V8 forbids"
            )

    @property
    def fires(self) -> bool:
        return self.reason is not None


@runtime_checkable
class TriggerPolicyPort(Protocol):
    """P12 — decide which candidates deserve expensive analysis.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **G1** | **Every candidate produces exactly one decision.** None may be dropped; a skip is a decision. |
    | **G2** | A decision carries either a ``TriggerReason`` or a ``SkipReason``, never both and never neither. |
    | **G3** | Deterministic: identical candidates and demands yield identical decisions, in identical order (V13). |
    | **G4** | The ceiling holds — a policy reasons about measurements and ids, never about what a region or class *means* (V1/V2). |
    | **G5** | Priority is an opaque class. A policy may order by it; it may not interpret it. |
    | **G6** | Stateless across calls, or state is per-camera and reset with the partition. |
    """

    @property
    def policy_id(self) -> str:
        ...

    def evaluate(
        self,
        candidates: Sequence[TriggerCandidate],
        *,
        now: Instant,
        demands: Sequence[object],
    ) -> Sequence[TriggerDecision]:
        """Decide for every candidate. Returns one decision per candidate."""
        ...


# --- P13 QualityEstimatorPort -------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class QualityRequest:
    """What the estimator is asked to grade.

    ``pixels`` is present when a crop already exists and absent when the
    estimate is pre-extraction — the cheap path that avoids paying for a crop
    that cannot pass. An estimator must handle both.
    """

    camera_id: CameraId
    box: Box
    source_width: int
    source_height: int
    neighbour_boxes: Sequence[Box] = ()
    """For crowding. Passed rather than discovered, so the estimator holds no
    state and stays a pure function."""

    pixels: memoryview | None = None
    crop_width: int = 0
    crop_height: int = 0


@runtime_checkable
class QualityEstimatorPort(Protocol):
    """P13 — grade input quality (02_VOM section 10.8).

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **Q1** | Grades in ``[0,1]``; ``scale_pixels`` in source pixels. |
    | **Q2** | **Unmeasured is ``None``, never zero.** "Not measured" and "measured as zero" are different claims and a consumer reads a zeroed grade as a good one. |
    | **Q3** | ``overall`` is set whenever any grade is; it is the gate's input. |
    | **Q4** | Deterministic and pure: identical input yields identical grades (V13). |
    | **Q5** | Never raises on legal-but-extreme geometry; a degenerate box grades as insufficient. |
    """

    @property
    def estimator_id(self) -> str:
        ...

    def estimate(self, request: QualityRequest) -> QualityGrades:
        """Grade the input. Pure, deterministic, never raising."""
        ...


# --- P14 CropStrategyPort ------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class CropPlan:
    """How to turn an object box into crop pixels.

    A **plan**, not pixels: extraction is *"a pure function of (frame, box,
    transform)"* (§M8 Thread Safety), so the decision and the work are separable
    and the work is trivially parallel.
    """

    source_box: Box
    """The object's box, before padding — recorded on the crop."""

    padded_box: Box
    """What will actually be read, after padding and clamping to the frame."""

    padding_applied: float
    output_width: int
    output_height: int
    preserve_aspect: bool = True
    """Letterbox rather than squash. A squashed crop produces attributes about a
    distorted object, and the distortion is invisible in the output."""

    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("a crop plan must have positive output dimensions")
        if self.padding_applied < 0.0:
            raise ValueError("padding_applied must be non-negative")


@runtime_checkable
class CropStrategyPort(Protocol):
    """P14 — decide the crop geometry.

    Tight box, context-padded, multi-scale, part-focused (head region for
    headwear, torso for hi-vis), temporal stacks. Today the platform ships tight
    and padded; the rest plug in without M8 changing.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **C1** | The plan's ``padded_box`` lies inside the frame after clamping. |
    | **C2** | ``output_size`` is the model's native input size — **never larger**, which is pure waste (§M8 Performance). |
    | **C3** | Deterministic: identical object and configuration yield an identical plan (V13). |
    | **C4** | Aspect handling is declared, so a comparison across strategies stays fair. |
    | **C5** | A strategy may use the object's class to choose geometry (a head region for a person); it may never use what the class *means* to a business. |
    """

    @property
    def strategy_id(self) -> str:
        ...

    def plan(
        self,
        *,
        box: Box,
        class_id: ClassId,
        source_width: int,
        source_height: int,
        attributes: Sequence[AttributeKey] = (),
    ) -> CropPlan:
        """Produce the geometry. Pure and deterministic."""
        ...


@runtime_checkable
class CropExtractorPort(Protocol):
    """Turn a plan plus frame pixels into normalized crop bytes.

    Separated from ``CropStrategyPort`` because the *decision* is cheap and
    per-object while the *work* is expensive and batchable. A deployment that
    moves extraction to a GPU replaces this and nothing else.
    """

    @property
    def extractor_id(self) -> str:
        ...

    def extract(
        self,
        pixels: memoryview,
        *,
        plan: CropPlan,
        source_width: int,
        source_height: int,
        channels: int,
        colour_space: str,
    ) -> tuple[bytes, CropTransform]:
        """Return the crop bytes and **what was actually applied**.

        The transform is returned rather than assumed, because a crop whose
        record and reality disagree invites a comparison that looks valid and is
        not (02_VOM section 10.7).
        """
        ...
