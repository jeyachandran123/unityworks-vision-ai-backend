"""The Observation Envelope (02_VOM §11) — the platform's single published unit.

> *Everything above exists to produce this, and every consumer integrates against
> this.*

An observation is a **statement about a moment that has passed**. 07_STATE §2.1 is
explicit that its immutability is not a constraint imposed to enable event
sourcing — *"it is the natural nature of the data. An observation... cannot become
untrue."* Everything in this module follows from that one sentence.

Four properties are enforced at construction rather than by convention, because
each has a failure mode that is invisible afterwards:

``provenance`` is mandatory
    §M11: *"an unexplainable observation is worse than no observation — it is a
    fact nobody can audit (V4)."* An envelope without a producer and a config
    revision cannot be constructed at all.

``measurement_basis`` is mandatory
    V8 at field level. A predicted position presented as a measured one is a
    confident lie, and no consumer can detect it after the fact.

``supersedes`` replaces mutation
    07_STATE §2.2. A correction is a **new** observation; the corrected one
    survives unchanged forever, because *"a consumer that acted on `obs-A` at the
    time acted correctly on the information then available — and can prove it."*

``coverage`` is a first-class type
    02_VOM §11.2: *"This type is the difference between a platform that is honest
    about its limits and one that is dangerously silent — and it is not
    optional."*

**Nothing here interprets.** There is no severity, no alert, no verdict. A
restaurant, a hospital and a factory read the same observation and reach entirely
different conclusions, which is the design goal.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .confidence import Confidence, ConfidenceSemantics
from .crop import TriggerReason
from .detection import QualityGrades
from .ids import (
    AttributeKey,
    BlobRef,
    CameraId,
    ClassId,
    CropId,
    DemandId,
    EvidenceId,
    FrameRef,
    ObjectId,
    ObservationId,
    SiteId,
    TenantId,
    TrackId,
)
from .provenance import Provenance
from .space import SpatialInfo
from .timebase import ClockQuality, Duration, Instant
from .understanding import DecisionRecord, Timing
from .visual_object import Attribute, LifecycleState

OBSERVATION_SCHEMA_VERSION = "1.0.0"


class ObservationType(enum.Enum):
    """02_VOM §11.2. Closed set — seven types, each with distinct content.

    The set is closed because 02_VOM §12 classifies adding an enum value as
    *additive* only when consumers tolerate unknown values; the platform may add
    a type later, but it may never quietly repurpose one.
    """

    PRESENCE = "presence"
    """An object was detected in a frame. Class, spatial, detection confidence."""

    SPATIAL = "spatial"
    """Position, motion or region membership changed materially."""

    ATTRIBUTE = "attribute"
    """An attribute was computed or revised. Carries evidence."""

    IDENTITY = "identity"
    """An identity assertion was made or revised. Binding, method, confidence."""

    LIFECYCLE = "lifecycle"
    """A VisualObject changed lifecycle state. Old, new, reason."""

    QUALITY = "quality"
    """Input quality changed materially."""

    COVERAGE = "coverage"
    """**The invariant V8 object.** How the platform says *"between 09:14 and
    09:21, camera 7 was blind."* Without it a consumer querying that window sees
    no observations and concludes nothing happened."""

    @property
    def concerns_an_object(self) -> bool:
        """Whether this type is about a specific object.

        ``coverage`` is the exception: it is a statement about the *platform's
        ability to see*, not about anything seen.
        """
        return self is not ObservationType.COVERAGE


class MeasurementBasis(enum.Enum):
    """V8 at field level (02_VOM §11).

    A predicted position and a measured one are different claims. Systems that
    present both as "the position" produce confident answers about objects that
    were not actually observed, and nothing downstream can tell.
    """

    MEASURED = "measured"
    """Directly observed this frame."""

    PREDICTED = "predicted"
    """Extrapolated from motion. The object was not measured."""

    INTERPOLATED = "interpolated"
    """Derived between two measurements. Only valid in retrospect."""

    @property
    def is_direct(self) -> bool:
        return self is MeasurementBasis.MEASURED


class ObservabilityStatus(enum.Enum):
    """07_STATE §7.2. Can the platform see, and how well."""

    OBSERVING = "observing"
    DEGRADED = "degraded"
    BLIND = "blind"
    DISCONNECTED = "disconnected"

    @property
    def is_observing(self) -> bool:
        return self is ObservabilityStatus.OBSERVING


class ObservabilityReason(enum.Enum):
    """07_STATE §7.2's closed reason set.

    Every value names a *mechanism*, never a conclusion. The platform reports
    that it cannot see and why; what that means for a consumer is a consumer's
    judgment (V1).
    """

    NORMAL = "normal"
    STREAM_DISCONNECTED = "stream_disconnected"
    DECODE_FAILING = "decode_failing"
    PRIVACY_MASK_FAILED = "privacy_mask_failed"
    """M2 — the platform's only fail-closed path."""

    SCHEDULER_SHEDDING = "scheduler_shedding"
    DETECTOR_UNAVAILABLE = "detector_unavailable"
    UNDERSTANDING_BUDGET_EXHAUSTED = "understanding_budget_exhausted"
    """M8 — attributes thinned, presence unaffected. A partial blindness that a
    consumer must be able to distinguish from total."""

    SCENE_OBSCURED = "scene_obscured"
    CALIBRATION_SUSPECT = "calibration_suspect"
    MODEL_CAPABILITY_GAP = "model_capability_gap"
    PARTITION_UNAVAILABLE = "partition_unavailable"
    RESTART = "restart"
    """07_STATE §9.3: *"The restart gap is recorded as a coverage observation, so
    consumers see the discontinuity as data rather than inferring it from a
    suspicious silence."*"""


class ViolationKind(enum.Enum):
    """Why an observation, or part of one, was refused.

    Split deliberately between **attribute-level** and **envelope-level**,
    because §M11's failure table prescribes opposite responses: an unregistered
    attribute is dropped and the observation survives; an incomplete envelope
    rejects the whole thing.
    """

    UNREGISTERED_ATTRIBUTE = "unregistered_attribute"
    """The ceiling's third and final gate (00_CHARTER §4.3)."""

    DEPRECATED_ATTRIBUTE = "deprecated_attribute"
    ATTRIBUTE_NOT_APPLICABLE = "attribute_not_applicable"
    ATTRIBUTE_VALUE_INVALID = "attribute_value_invalid"
    WRONG_CONFIDENCE_SEMANTICS = "wrong_confidence_semantics"
    """An attribute carrying DETECTION_PRESENCE confidence is claiming something
    it cannot support (02_VOM §7.2)."""

    MISSING_PROVENANCE = "missing_provenance"
    MISSING_TIMING = "missing_timing"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_MEASUREMENT_BASIS = "missing_measurement_basis"
    TAXONOMY_VERSION_MISMATCH = "taxonomy_version_mismatch"
    """*"This indicates a partial deployment and must be loud."*"""

    UNREGISTERED_CLASS = "unregistered_class"

    @property
    def rejects_the_envelope(self) -> bool:
        """Whether this violation kills the whole observation.

        §M11: *"Attribute key not in registry — **drop the attribute**, keep the
        rest of the observation."* versus *"Envelope incomplete — **reject the
        observation entirely** and alarm."* Two failures, two responses, and
        conflating them either loses valid facts or publishes unauditable ones.
        """
        return self in (
            ViolationKind.MISSING_PROVENANCE,
            ViolationKind.MISSING_TIMING,
            ViolationKind.MISSING_EVIDENCE,
            ViolationKind.MISSING_MEASUREMENT_BASIS,
            ViolationKind.TAXONOMY_VERSION_MISMATCH,
            ViolationKind.UNREGISTERED_CLASS,
        )


@dataclass(frozen=True, slots=True)
class Violation:
    """One reason something was refused, with enough detail to act on.

    ``detail`` is required rather than optional: a violation nobody can diagnose
    produces an alarm nobody can clear, and the sustained-rate alarm §M11 asks
    for is only useful if each instance names its cause.
    """

    kind: ViolationKind
    detail: str
    field_name: str = ""

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError(
                f"violation '{self.kind.value}' carries no detail; an alarm "
                f"nobody can diagnose is an alarm nobody clears"
            )

    @property
    def rejects_the_envelope(self) -> bool:
        return self.kind.rejects_the_envelope


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """What was retained, and whether it is actually reachable.

    ``status`` exists because §M11's failure table requires honesty about a
    failed evidence write: *"Emit the observation with `evidence_ref` marked
    `pending`; retry asynchronously; if permanently failed, mark
    `evidence_unavailable` — **honest rather than silent**."*
    """

    evidence_id: EvidenceId
    status: str = "stored"
    """``stored`` | ``pending`` | ``unavailable``."""

    crop_ref: CropId | None = None
    raw_output_ref: BlobRef | None = None
    retention: str = "evidence"

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("an evidence reference requires an id")
        if self.status not in ("stored", "pending", "unavailable"):
            raise ValueError(
                f"unknown evidence status '{self.status}'; the three states are "
                f"stored, pending and unavailable, and each means something "
                f"different to a consumer trying to retrieve it"
            )

    @property
    def is_retrievable(self) -> bool:
        return self.status == "stored"


@dataclass(frozen=True, slots=True)
class Evidence:
    """02_VOM §10.9's ``Evidence``, **complete**.

    Flow 6 produced everything except ``observation_id``, because M9 may not mint
    an identifier for an object it is forbidden to create. M11 stamps it here —
    the promised half of a two-part construction.
    """

    evidence_id: EvidenceId
    observation_id: ObservationId
    trigger_reason: TriggerReason
    input_hash: str
    frame_ref: FrameRef
    crop_ref: CropId | None = None
    raw_output_ref: BlobRef | None = None
    unstructured_note: str | None = None
    """02_VOM §9.3 — inspectable, **never promoted**. Not queryable as fact,
    never entering state as an attribute, never filterable by the API."""

    decision_path: tuple[DecisionRecord, ...] = ()
    timing: Timing = Timing()
    retention: str = "evidence"

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError(
                "evidence must name the observation it explains; unattached "
                "evidence cannot be retrieved through the evidence contract "
                "(09_API_CONTRACTS section 6)"
            )
        if not self.input_hash:
            raise ValueError("evidence requires an input hash (invariant V4)")


@dataclass(frozen=True, slots=True)
class CoverageWindow:
    """The content of a ``coverage`` observation (07_STATE §7.2).

    Note ``regions_affected``: blindness is often *partial* — a parked truck
    occludes one corner of a view. Reporting the whole camera as blind would
    overstate the gap; reporting nothing would understate it.
    """

    status: ObservabilityStatus
    reason: ObservabilityReason
    since: Instant
    effective_rate: float = 1.0
    regions_affected: tuple[str, ...] = ()
    capability_gaps: tuple[tuple[str, str], ...] = ()
    """``(class or attribute, reason)``. A consumer learns the platform cannot
    produce something *here*, rather than waiting for it forever (V8)."""

    until: Instant | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.effective_rate <= 1.0:
            raise ValueError(
                f"effective_rate must be in [0,1], got {self.effective_rate}"
            )
        if self.status.is_observing and self.reason is not ObservabilityReason.NORMAL:
            raise ValueError(
                f"status 'observing' cannot carry reason '{self.reason.value}'; a "
                f"camera that is observing has no observability problem to name"
            )

    @property
    def is_gap(self) -> bool:
        return not self.status.is_observing


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    """The content of a ``lifecycle`` observation."""

    previous: LifecycleState
    current: LifecycleState
    trigger: str = ""

    def __post_init__(self) -> None:
        if self.previous is self.current:
            raise ValueError(
                f"a lifecycle observation must record a change; "
                f"'{self.current.value}' to itself is not a transition"
            )


@dataclass(frozen=True, slots=True)
class IdentityAssertionRef:
    """The content of an ``identity`` observation.

    A **claim**, not a truth (02_VOM §4.2). ``alternatives`` is non-zero exactly
    when candidates existed but none was decisive — the case M7 cares most about,
    published rather than guessed.
    """

    binding_id: str
    track_id: TrackId | None = None
    method: str = ""
    ambiguous: bool = False
    alternatives: int = 0

    def __post_init__(self) -> None:
        if self.ambiguous and self.alternatives < 1:
            raise ValueError(
                "an ambiguous assertion must name how many alternatives it "
                "declined to choose between; ambiguity with no alternatives is "
                "not ambiguity"
            )


@dataclass(frozen=True, slots=True)
class Observation:
    """02_VOM §11's envelope, implemented field for field.

    **Immutable.** A correction is a new observation carrying ``supersedes``; the
    corrected one survives unchanged forever (V5, 07_STATE §2.2).

    Construction enforces what §M11 calls envelope completeness. An observation
    that cannot be audited cannot be built — which is a stronger guarantee than
    validating one after the fact, because there is no code path that produces an
    incomplete envelope to validate.
    """

    # ---- identity ----
    observation_id: ObservationId
    observation_type: ObservationType
    tenant_id: TenantId
    site_id: SiteId
    camera_id: CameraId

    # ---- temporal anchor ----
    frame_ref: FrameRef
    t_capture: Instant
    t_capture_unc: Duration
    clock_quality: ClockQuality
    t_published: Instant

    # ---- explainability (mandatory: V4) ----
    provenance: Provenance
    timing: Timing

    # ---- subject ----
    object_id: ObjectId | None = None
    """``None`` only for ``coverage``, which is a statement about the platform
    rather than about anything seen."""

    track_id: TrackId | None = None
    class_id: ClassId | None = None
    taxonomy_version: str = ""
    lifecycle_state: LifecycleState | None = None

    # ---- content ----
    confidence: Confidence | None = None
    spatial: SpatialInfo | None = None
    attributes: tuple[Attribute, ...] = ()
    measurement_basis: MeasurementBasis = MeasurementBasis.MEASURED
    quality: QualityGrades | None = None

    coverage: CoverageWindow | None = None
    lifecycle_transition: LifecycleTransition | None = None
    identity: IdentityAssertionRef | None = None

    # ---- evidence ----
    evidence_ref: EvidenceRef | None = None

    # ---- relationships ----
    lineage: tuple[ObservationId, ...] = ()
    supersedes: ObservationId | None = None
    demand_ids: tuple[DemandId, ...] = ()

    schema_version: str = OBSERVATION_SCHEMA_VERSION
    labels: Mapping[str, str] = field(default_factory=dict)
    """Opaque operational tags. No platform logic may branch on these."""

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("an observation requires an id")

        # V4. §M11: an unexplainable observation is worse than no observation.
        if self.provenance is None:
            raise ValueError(
                "an observation requires provenance; a record nobody can trace "
                "to a producer is not auditable (invariant V4)"
            )
        if not self.provenance.producer_module:
            raise ValueError(
                "an observation without a producer module cannot be audited; "
                "an unexplainable observation is worse than no observation "
                "(04_MODULES section M11)"
            )
        if not self.provenance.config_revision:
            raise ValueError(
                "an observation without a config revision is not reproducible "
                "(invariant V4)"
            )

        if self.frame_ref.camera_id != self.camera_id:
            raise ValueError(
                f"observation claims camera {self.camera_id} but its frame is "
                f"from {self.frame_ref.camera_id}; a fact must be traceable to "
                f"the instant it describes"
            )

        if self.t_capture_unc.ns < 0:
            raise ValueError("capture uncertainty cannot be negative")

        self._check_type_payload()
        self._check_attributes()

        if self.supersedes == self.observation_id:
            raise ValueError("an observation cannot supersede itself")
        if self.observation_id in self.lineage:
            raise ValueError(
                "an observation cannot be its own ancestor; a lineage cycle "
                "makes provenance unresolvable"
            )

    def _check_type_payload(self) -> None:
        """Each type must carry the content 02_VOM §11.2 assigns it.

        An observation of a type whose payload is missing is a fact with no
        content — indistinguishable from a bug, and impossible for a consumer to
        act on.
        """
        kind = self.observation_type

        if kind.concerns_an_object and self.object_id is None:
            raise ValueError(
                f"a '{kind.value}' observation must name its object; only "
                f"coverage is a statement about the platform rather than about "
                f"something seen"
            )
        if kind is ObservationType.COVERAGE:
            if self.coverage is None:
                raise ValueError("a coverage observation must carry a coverage window")
            if self.object_id is not None:
                raise ValueError(
                    "a coverage observation describes the platform's ability to "
                    "see, not an object; attaching one would make a blindness "
                    "report look like a sighting"
                )
            return

        if kind is ObservationType.LIFECYCLE and self.lifecycle_transition is None:
            raise ValueError("a lifecycle observation must carry its transition")
        if kind is ObservationType.IDENTITY and self.identity is None:
            raise ValueError("an identity observation must carry its assertion")
        if kind is ObservationType.ATTRIBUTE and not self.attributes:
            raise ValueError(
                "an attribute observation with no attributes says nothing; "
                "'the model produced nothing' is not published as a fact — it is "
                "an understanding outcome (04_MODULES section M9)"
            )
        if kind is ObservationType.QUALITY and self.quality is None:
            raise ValueError("a quality observation must carry its grades")
        if kind in (ObservationType.PRESENCE, ObservationType.SPATIAL):
            if self.spatial is None:
                raise ValueError(
                    f"a '{kind.value}' observation must carry spatial information"
                )

    def _check_attributes(self) -> None:
        """Attribute confidence must mean what an attribute claim means.

        02_VOM §7.2: a detector's objectness score and a VLM's self-report are
        incomparable quantities. An attribute carrying DETECTION_PRESENCE
        confidence is borrowing a number that answers a different question.
        """
        for attribute in self.attributes:
            if attribute.confidence.semantics not in (
                ConfidenceSemantics.ATTRIBUTE,
                ConfidenceSemantics.SELF_REPORTED,
            ):
                raise ValueError(
                    f"attribute '{attribute.key}' carries "
                    f"{attribute.confidence.semantics.value} confidence; an "
                    f"attribute claim must carry ATTRIBUTE or SELF_REPORTED "
                    f"(02_VOM section 9.2)"
                )

    # --- derived, all pure --------------------------------------------------- #

    @property
    def is_correction(self) -> bool:
        return self.supersedes is not None

    @property
    def attribute_keys(self) -> tuple[AttributeKey, ...]:
        return tuple(a.key for a in self.attributes)

    @property
    def is_measured(self) -> bool:
        return self.measurement_basis.is_direct

    @property
    def has_ground_position(self) -> bool:
        """Whether a world-frame position is available.

        False is normal, not a failure: §M11 degrades to normalized coordinates
        when calibration is unavailable rather than dropping the observation.
        """
        return self.spatial is not None and self.spatial.ground_point is not None

    def attribute(self, key: AttributeKey) -> Attribute | None:
        for candidate in self.attributes:
            if candidate.key == key:
                return candidate
        return None

    def stale_at(self, now: Instant) -> Duration:
        """Age of this observation's content, measured from capture.

        From capture rather than publication: a consumer cares how old the *world*
        was, not how long the platform took (V11).
        """
        return Duration(max(0, now.ns - self.t_capture.ns))


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The outcome of putting a candidate observation through the final gate.

    ``observation`` is present when something publishable survived — which may be
    a *narrowed* observation, with offending attributes dropped. That is §M11's
    prescribed response and it is why this is not a boolean.
    """

    observation: Observation | None = None
    violations: tuple[Violation, ...] = ()
    dropped_attributes: tuple[AttributeKey, ...] = ()

    @property
    def published(self) -> bool:
        return self.observation is not None

    @property
    def rejected(self) -> bool:
        return self.observation is None

    @property
    def envelope_violations(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.rejects_the_envelope)

    @property
    def attribute_violations(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if not v.rejects_the_envelope)


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """What one build pass produced, with everything it refused.

    Carrying the refusals alongside the observations is what makes the
    sustained-rate alarm §M11 asks for possible: a caller can count rejections by
    kind without re-deriving them.
    """

    observations: tuple[Observation, ...] = ()
    violations: tuple[Violation, ...] = ()
    suppressed: int = 0
    """Observations correctly not published because nothing changed. A
    *success*, not a loss — §M11 puts the typical reduction at 10-50x."""

    @property
    def count(self) -> int:
        return len(self.observations)

    def by_type(self) -> dict[ObservationType, int]:
        counts: dict[ObservationType, int] = {}
        for observation in self.observations:
            counts[observation.observation_type] = (
                counts.get(observation.observation_type, 0) + 1
            )
        return counts


def coverage_gap(
    windows: Sequence[CoverageWindow], *, since: Instant, until: Instant
) -> float:
    """Fraction of a window that was actually observable (07_STATE §7.3).

    The number a consumer needs alongside an empty result: without it, *"the
    region was empty"* and *"the camera was disconnected"* are the same answer.
    """
    span = max(1, until.ns - since.ns)
    blind = 0
    for window in windows:
        if not window.is_gap:
            continue
        start = max(window.since.ns, since.ns)
        end = min((window.until or until).ns, until.ns)
        blind += max(0, end - start)
    return max(0.0, min(1.0, 1.0 - blind / span))
