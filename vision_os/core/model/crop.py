"""The Crop object (02_VOM section 10.7) — canonical visual evidence.

A crop is **evidence**, not an image. The distinction is the whole module: an
image is pixels, and evidence is pixels plus everything needed to defend a claim
made from them — which camera, which frame, which object, what transform was
applied, how good the input was, and whether it passed the gate.

Three fields carry that weight, and each exists because omitting it produces a
failure nobody detects:

``crop_id`` is a content hash
    The same pixels cropped twice must be one crop. Content addressing gives
    free deduplication, free cache keys, free integrity checking, and a stable
    evidence reference that survives storage migration.

``transform`` records what was *actually* applied
    02_VOM section 10.7: *"two models evaluated on differently-letterboxed crops
    are not comparable, and without this field nobody finds out."* This is the
    field that makes a model comparison fair and a result reproducible.

``gate_result`` carries a reason
    *"`gate_result` with a reason means rejected crops are counted and
    explicable, so 'the VLM never answers for far-away people' becomes a visible
    statistic rather than a mystery."*
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from .detection import QualityGrades
from .ids import CameraId, ClassId, CropId, FrameRef, ObjectId, SiteId, TenantId
from .provenance import Provenance
from .space import Box
from .timebase import Duration, Instant

CROP_SCHEMA_VERSION = "1.0.0"


class PrivacyClass(enum.Enum):
    """Data classification (12_SECURITY section 3).

    A crop is **C1 · Imagery**: encrypted at rest and in transit with per-tenant
    keys, retained 24–72 hours, and requiring a separate evidence privilege to
    read. The class travels with the crop so no downstream component has to
    infer it — and inferring it is how imagery ends up in an unclassified path.
    """

    C0_OPERATIONAL = "c0_operational"
    """Metrics, health, config. No personal data."""

    C1_IMAGERY = "c1_imagery"
    """Frames, crops, evidence. What every crop is."""

    C2_BIOMETRIC = "c2_biometric"
    """Appearance embeddings, identity galleries. **Disabled by default.**"""

    C3_BEHAVIOURAL = "c3_behavioural"
    """Observations, trajectories, dwell, attributes."""


class RetentionMode(enum.Enum):
    """How long a crop may live (02_VOM section 10.7).

    Set by M8, honoured by the evidence store. M8 decides the *policy*; it does
    not persist anything itself.
    """

    EPHEMERAL = "ephemeral"
    """Lives only long enough to be analysed. The default, and the cheapest."""

    EVIDENCE = "evidence"
    """Retained for a bounded TTL so a claim can be audited later."""

    NEVER_PERSIST = "never_persist"
    """Must not touch disk. 12_SECURITY section 2.3's no-evidence mode: the crop
    exists in memory for the duration of one inference and nowhere else."""


class GateOutcome(enum.Enum):
    PASSED = "passed"
    REJECTED = "rejected"


class GateRejection(enum.Enum):
    """Why a crop was refused. Every rejection is attributable.

    Deliberately narrow: these are the conditions under which an expensive model
    cannot produce a defensible answer, not judgments about what the pixels mean.
    """

    TOO_SMALL = "too_small"
    """Below the scale floor. The strongest single predictor of a useless claim."""

    TOO_BLURRY = "too_blurry"
    TOO_OCCLUDED = "too_occluded"
    TOO_TRUNCATED = "too_truncated"
    """Most of the object is outside the frame; the crop shows a fragment."""

    EXPOSURE_UNUSABLE = "exposure_unusable"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    """The requested box has no area after clamping to the frame."""


class TriggerReason(enum.Enum):
    """Why a candidate was selected for analysis (03_MODULES section M8).

    Travels onto the evidence, because 02_VOM section 10.9 requires evidence to
    record *why this was computed at all* — without it, a result six months old
    is a number with no story.
    """

    FIRST_SIGHT = "first_sight"
    """Object newly confirmed; no attributes yet."""

    ATTRIBUTE_MISSING = "attribute_missing"
    """A demand requires an attribute never computed."""

    ATTRIBUTE_STALE = "attribute_stale"
    """Beyond the demand's freshness SLA."""

    APPEARANCE_CHANGED = "appearance_changed"
    """Visual change exceeds threshold — a re-look is warranted."""

    LOW_CONFIDENCE = "low_confidence"
    """A prior claim was weak; a better crop is available now."""

    IDENTITY_UNVERIFIED = "identity_unverified"
    """The detector's class claim is not admissible for what a demand needs, and
    no corroborating evidence exists yet.

    Distinct from ``LOW_CONFIDENCE``, which is about an attribute *already
    computed* being weak. This fires when there is no claim at all and the
    detector's own answer cannot stand alone — a closed-set model naming an
    object outside its vocabulary returns the nearest word it knows, and that is
    a guess wearing the clothes of an identification.

    Kept separate because this value travels onto ``Evidence.trigger_reason``,
    which is where *"why did the platform look at all?"* is answered six months
    later. Collapsing the two would make the reasons indistinguishable in the
    audit trail and in the metrics that count them."""

    QUALITY_IMPROVED = "quality_improved"
    """Previously gate-rejected; conditions are now adequate."""

    PERIODIC_REFRESH = "periodic_refresh"
    """Cadence floor."""

    EXPLICIT_REQUEST = "explicit_request"
    """Bounded, rate-limited on-demand API call."""

    LIFECYCLE_TRANSITION = "lifecycle_transition"
    """Object entered or left a region, or changed lifecycle state."""


class SkipReason(enum.Enum):
    """Why a candidate was **not** analysed (03_MODULES section M8).

    Responsibility 7 and invariant V8: *"a consumer must be able to tell 'no
    attribute because nothing was there' from 'no attribute because we could not
    afford to look'."* Every candidate that produces no crop produces one of
    these instead. Silence is never an answer.
    """

    NO_DEMAND = "no_demand"
    """Nothing asked for this. The largest bucket in a healthy deployment, and
    the reason cost is `demands × changes` rather than `cameras × fps`."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    QUALITY_INSUFFICIENT = "quality_insufficient"
    FRESH_ENOUGH = "fresh_enough"
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    """A demand wanted corroborating evidence and the evidence already held was
    good enough, so no model was called.

    Distinct from ``FRESH_ENOUGH``, which means *"the attribute is present and
    within its freshness window"*. Here the attribute may be entirely absent —
    what the policy decided is that the evidence the platform already has
    supports the claim without paying for more. Recording it as ``FRESH_ENOUGH``
    would make an attribute that was never computed look like one that was, and
    would hide the largest and most valuable skip bucket in a deployment that
    uses verification."""
    DEDUPLICATED = "deduplicated"
    PRIORITY_PREEMPTED = "priority_preempted"
    FRAME_UNAVAILABLE = "frame_unavailable"
    """The frame was evicted before the crop could be taken. Signals that pin TTL
    or buffer depth needs tuning — a diagnosable configuration issue rather than
    a mystery."""


@dataclass(frozen=True, slots=True)
class CropTransform:
    """What was **actually** applied to produce the crop pixels.

    Not what was requested — what happened. A crop whose transform record and
    real transform disagree is worse than one with no record, because it invites
    a comparison that looks valid and is not.
    """

    source_width: int
    source_height: int
    output_width: int
    output_height: int

    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    """The source-pixel rectangle actually read, after padding and clamping."""

    scale: float = 1.0

    drawn_width: int = 0
    drawn_height: int = 0
    """The image area actually written, before padding. Zero means "the whole
    output", which is the un-letterboxed case.

    Recorded rather than inferred from the padding. Letterbox padding splits by
    integer division, so an odd leftover pixel goes to one side — and
    reconstructing the drawn size as ``output - 2 * pad`` is then wrong by a
    pixel, which is enough to make an aspect check report a distortion that never
    happened."""

    pad_left: int = 0
    pad_top: int = 0
    """Letterbox padding added to reach the output size while preserving aspect."""

    interpolation: str = "bilinear"
    colour_space: str = "bgr24"
    letterboxed: bool = False

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "output_width", "output_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.crop_width <= 0 or self.crop_height <= 0:
            raise ValueError(
                f"degenerate crop rectangle {self.crop_width}x{self.crop_height}; "
                f"a crop with no area is not evidence"
            )
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")

    @property
    def drawn_size(self) -> tuple[int, int]:
        """The written area, defaulting to the full output when unrecorded."""
        width = self.drawn_width or self.output_width
        height = self.drawn_height or self.output_height
        return width, height

    @property
    def aspect_preserved(self) -> bool:
        """Whether the source aspect ratio survived the resize.

        A crop that was squashed to fit produces attributes about a distorted
        object, and the distortion is invisible in the output.

        Tolerance is **one pixel**, not floating-point epsilon. Drawn dimensions
        are integers, so preserving a 32x384 aspect inside a 64x64 output means
        drawing 5x64 rather than the exact 5.33x64 — a sub-pixel rounding, not a
        distortion. An epsilon comparison would call every real letterbox
        squashed and make the flag useless.
        """
        drawn_width, drawn_height = self.drawn_size
        if self.crop_height <= 0 or drawn_height <= 0:
            return False
        ideal_width = self.crop_width * drawn_height / self.crop_height
        return abs(drawn_width - ideal_width) <= 1.0


@dataclass(frozen=True, slots=True)
class GateResult:
    """The quality gate's verdict, with its reason.

    ``passed`` carries no reason; ``rejected`` always does. A rejection without a
    reason is a statistic nobody can act on.
    """

    outcome: GateOutcome
    reason: GateRejection | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.outcome is GateOutcome.REJECTED and self.reason is None:
            raise ValueError(
                "a rejected crop must name its reason; 'the VLM never answers for "
                "far-away people' must be a visible statistic, not a mystery "
                "(02_VOM section 10.7)"
            )
        if self.outcome is GateOutcome.PASSED and self.reason is not None:
            raise ValueError("a passed crop cannot carry a rejection reason")

    @property
    def passed(self) -> bool:
        return self.outcome is GateOutcome.PASSED

    @classmethod
    def accept(cls) -> GateResult:
        return cls(outcome=GateOutcome.PASSED)

    @classmethod
    def reject(cls, reason: GateRejection, detail: str = "") -> GateResult:
        return cls(outcome=GateOutcome.REJECTED, reason=reason, detail=detail)


@dataclass(frozen=True, slots=True)
class Crop:
    """One piece of canonical visual evidence (02_VOM section 10.7).

    Immutable, and content-addressed by ``crop_id``. Traceable to camera, frame,
    track (through the object) and object — every field needed to defend a claim
    six months after it was made.

    ``pixels`` is a **data-plane** handle and never crosses a node boundary
    (invariant V12). The crop *reference* travels; the imagery does not.
    """

    crop_id: CropId
    tenant_id: TenantId
    site_id: SiteId
    camera_id: CameraId

    source_frame: FrameRef
    object_id: ObjectId | None
    """``None`` only for a crop taken without an object — reserved for future
    whole-frame evidence. Every crop M8 produces today names its object."""

    source_box: Box
    """Normalized, in the source frame. The object's box before padding."""

    padding_applied: float
    """Context ratio added around the box. Recorded because an attribute computed
    on a tight crop and on a padded one are different measurements."""

    output_size: tuple[int, int]
    transform: CropTransform
    quality: QualityGrades
    gate_result: GateResult
    retention: RetentionMode
    privacy_class: PrivacyClass

    t_capture: Instant
    """Capture time of the source frame, not extraction time. A dwell or a
    sequence assembled from crops must reflect the world (V11)."""

    trigger_reason: TriggerReason
    provenance: Provenance
    pixels: memoryview | None = None
    """The normalized crop bytes. Data plane: present on the producing node,
    absent once the crop reference travels."""

    retention_ttl: Duration | None = None
    schema_version: str = CROP_SCHEMA_VERSION
    labels: Mapping[str, str] = field(default_factory=dict)
    """Opaque operational tags. No platform logic may branch on these."""

    def __post_init__(self) -> None:
        if not self.crop_id:
            raise ValueError("a crop requires a content-addressed crop_id")
        if self.source_frame.camera_id != self.camera_id:
            raise ValueError(
                f"crop claims camera {self.camera_id} but its frame is from "
                f"{self.source_frame.camera_id}; evidence must be traceable"
            )
        if not 0.0 <= self.padding_applied <= 4.0:
            raise ValueError(
                f"padding_applied must be in [0,4], got {self.padding_applied}"
            )
        width, height = self.output_size
        if width <= 0 or height <= 0:
            raise ValueError(f"degenerate output size {width}x{height}")
        if (width, height) != (self.transform.output_width, self.transform.output_height):
            raise ValueError(
                "output_size disagrees with the recorded transform; a crop whose "
                "transform record does not match its pixels invites a comparison "
                "that looks valid and is not"
            )
        if self.retention is RetentionMode.EVIDENCE and self.retention_ttl is None:
            raise ValueError(
                "evidence retention requires a TTL; imagery retained forever is a "
                "compliance liability (12_SECURITY section 3)"
            )
        if self.retention is RetentionMode.NEVER_PERSIST and self.retention_ttl is not None:
            raise ValueError("never_persist cannot carry a TTL")

    # --- derived, all pure --------------------------------------------------- #

    @property
    def passed_gate(self) -> bool:
        return self.gate_result.passed

    @property
    def is_persistable(self) -> bool:
        return self.retention is RetentionMode.EVIDENCE

    @property
    def width(self) -> int:
        return self.output_size[0]

    @property
    def height(self) -> int:
        return self.output_size[1]

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def without_pixels(self) -> Crop:
        """The control-plane form: everything except the imagery.

        Invariant V12 made mechanical. A crop reference is ~1 KB and may cross a
        node boundary; the pixels are megabytes and may not.
        """
        from dataclasses import replace

        return replace(self, pixels=None)

    def expires_at(self) -> Instant | None:
        """When the evidence store may discard this. ``None`` means immediately."""
        if self.retention_ttl is None:
            return None
        return Instant(self.t_capture.ns + self.retention_ttl.ns)


@dataclass(frozen=True, slots=True)
class CropRequest:
    """A candidate that passed trigger evaluation and is worth extracting.

    Separating request from extraction is what lets the expensive step be
    batched, scheduled and parallelised — extraction is a pure function of
    (frame, box, transform), so a request is fully described by its inputs.
    """

    object_id: ObjectId
    camera_id: CameraId
    frame_ref: FrameRef
    source_box: Box
    trigger_reason: TriggerReason

    tenant_id: TenantId
    site_id: SiteId
    """Inherited from the object, never from a module-level constant. Tenancy is
    a property of the data, and 12_SECURITY section 4 requires it in every cache
    key — so the request that produces a crop must already know it."""

    class_id: ClassId = ClassId("")
    """The object's class. A crop strategy may use it to choose geometry — a head
    region for a person — and may never use what the class *means* to a business
    (port obligation C5)."""

    required_attributes: tuple[str, ...] = ()
    """Which attributes the demands want. Passed to M9; M8 never interprets them."""

    priority_class: str = ""
    """Opaque. The platform orders by it and never interprets it (V1)."""

    demand_ids: tuple[str, ...] = ()
    """Which demands this request serves, for cost attribution."""

    def __post_init__(self) -> None:
        if self.frame_ref.camera_id != self.camera_id:
            raise ValueError("a crop request must name the frame's own camera")


@dataclass(frozen=True, slots=True)
class Skipped:
    """A candidate that was **not** analysed, and why.

    The other half of ``evaluate``'s return, and the one that makes V8 true. A
    deployment where every skip is ``NO_DEMAND`` is healthy; one where they are
    ``BUDGET_EXHAUSTED`` is under-provisioned; one where they are
    ``QUALITY_INSUFFICIENT`` has a camera problem. None of that is visible
    without an attributed reason.
    """

    object_id: ObjectId
    camera_id: CameraId
    reason: SkipReason
    detail: str = ""
    attribute_keys: tuple[str, ...] = ()
    """Which attributes went unsatisfied, when the skip was demand-specific."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One frame's trigger evaluation for one camera.

    **Every candidate appears exactly once**, in ``requests`` or in ``skipped``.
    A candidate that appears in neither is invisible, which is the failure mode
    responsibility 7 exists to prevent.
    """

    camera_id: CameraId
    frame_ref: FrameRef
    requests: tuple[CropRequest, ...] = ()
    skipped: tuple[Skipped, ...] = ()

    @property
    def candidate_count(self) -> int:
        return len(self.requests) + len(self.skipped)

    def skips_by_reason(self) -> dict[SkipReason, int]:
        counts: dict[SkipReason, int] = {}
        for skip in self.skipped:
            counts[skip.reason] = counts.get(skip.reason, 0) + 1
        return counts
