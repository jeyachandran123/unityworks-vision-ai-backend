"""The Observation API's contract types (09_API_CONTRACTS).

> §M14's single responsibility: *"Serve facts safely under contract. Produce
> nothing; accept no writes to state."*

Everything a consumer can send or receive is declared here, and **nothing that
would let them write**. 09_API §1.1 lists five contracts — Query, Subscribe,
Demand, Coverage/Capability, Evidence — and a sixth that *"does not exist"*:

> *"~~Mutate~~ — **Does not exist** (V6). There is no endpoint, no field, no
> admin override."*

That absence is structural rather than enforced. There is no `Mutation` type in
this module to reject, no disabled flag to re-enable, and no privileged variant
of a query. A future contributor looking for the write path will not find one
disabled — they will find that it was never expressed.

**Why the contract lives in `core/model` rather than in the API package.** The
same reason 02_VOM's envelope does: these are the shapes the platform promises
external consumers for a decade, and 09_API §7.3's nine-month deprecation
timeline only means something if the shapes are versioned artefacts rather than
implementation details of whatever serves them today.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .confidence import Confidence
from .ids import (
    AttributeKey,
    CameraId,
    ClassId,
    DemandId,
    ObjectId,
    ObservationId,
    RegionId,
    SiteId,
    TenantId,
)
from .observation import Observation, ObservationType
from .timebase import Duration, Instant
from .vision_state import (
    CoverageReport,
    ObjectState,
    StateSnapshot,
)
from .visual_object import LifecycleState

API_VERSION = "1.0.0"
API_MAJOR = 1

SUPPORTED_MAJORS: frozenset[int] = frozenset({1})
"""Majors this build serves.

09_API §7.1 requires *"two adjacent majors concurrently"* during a migration
window. One is declared today because only one exists — the set is the mechanism,
and it holds two the day a v2 ships without any other change."""


# --- scope and identity ---------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Scope:
    """Where a request applies (09_API §2.1).

    ``tenant_id`` is **required and first**. 12_SECURITY §4.2 forbids post-filtering:
    *"Constructing the query already scoped means there is no moment at which
    cross-tenant data exists in memory to leak."* A scope with no tenant cannot be
    constructed, so no code path can produce an unscoped query by omission.
    """

    tenant_id: TenantId
    site_ids: tuple[SiteId, ...] = ()
    camera_ids: tuple[CameraId, ...] = ()
    region_ids: tuple[RegionId, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError(
                "a scope must name a tenant; tenancy is part of identity rather "
                "than a filter applied afterwards (12_SECURITY §4.1)"
            )

    def covers_camera(self, camera_id: CameraId) -> bool:
        return not self.camera_ids or camera_id in self.camera_ids

    def narrowed_to(self, cameras: Sequence[CameraId]) -> Scope:
        """A scope restricted to what the caller is authorized to see.

        Used by the authorization step to *construct* the query already scoped,
        rather than to check a result afterwards.
        """
        allowed = tuple(c for c in cameras if self.covers_camera(c))
        return Scope(
            tenant_id=self.tenant_id,
            site_ids=self.site_ids,
            camera_ids=allowed,
            region_ids=self.region_ids,
        )


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is asking. Exists **only** at this layer.

    12_SECURITY §5.1: *"External identity exists only at the Observation API...
    There is no ambient user context inside the pipeline, which means no pipeline
    component can accidentally make an authorization decision."*

    Nothing constructed from a `Principal` travels downward. M12 never learns a
    request had an actor.
    """

    subject: str
    tenant_id: TenantId
    scopes: tuple[str, ...] = ()
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a principal must have a subject")
        if not self.tenant_id:
            raise ValueError("a principal must belong to a tenant")


class Action(enum.Enum):
    """The permission model's actions (12_SECURITY §5.2).

    Separated exactly as §5.3 requires. ``READ_EVIDENCE`` is not a stronger form
    of ``READ_OBSERVATIONS``; it is a different act: *"Reading 'a person was
    here' and viewing their image are categorically different acts. Most
    consumers need the first and must never have the second."*
    """

    READ_STATE = "read_state"
    READ_OBSERVATIONS = "read_observations"
    READ_EVIDENCE = "read_evidence"
    SUBSCRIBE = "subscribe"
    REGISTER_DEMAND = "register_demand"
    READ_CAPABILITY = "read_capability"
    READ_COVERAGE = "read_coverage"

    @property
    def is_read(self) -> bool:
        """Whether this action only reads.

        ``REGISTER_DEMAND`` is the one that is not. 12_SECURITY §5.3: *"Demands
        spend money and cause computation; they are not a read."*
        """
        return self is not Action.REGISTER_DEMAND


# --- query contracts ------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AttributePredicate:
    """One attribute condition (09_API §2.1).

    Deliberately limited to equality and presence. A comparison operator would
    invite ``dwell_seconds > 300``, which is a threshold with business meaning —
    V1 puts that in the consumer's rule engine, not in a query language.
    """

    key: AttributeKey
    equals: object = None
    present: bool = True

    def matches(self, attributes: Mapping[AttributeKey, object]) -> bool:
        held = attributes.get(self.key)
        if held is None:
            return not self.present
        return True if self.equals is None else held == self.equals


@dataclass(frozen=True, slots=True)
class StateFilter:
    """What to include in a state query (09_API §2.1)."""

    class_ids: tuple[ClassId, ...] = ()
    lifecycle: tuple[LifecycleState, ...] = (
        LifecycleState.ACTIVE,
        LifecycleState.OCCLUDED,
    )
    """§2.1's default. A consumer asking "what is here" means present things."""

    min_confidence: float | None = None
    """§2.1: *"applies only to calibrated confidence."* An uncalibrated score is
    not comparable across models (02_VOM §7.2), so thresholding one would compare
    two different quantities."""

    attributes: tuple[AttributePredicate, ...] = ()
    max_staleness: Duration | None = None
    object_ids: tuple[ObjectId, ...] = ()

    def matches_class(self, class_id: ClassId | None) -> bool:
        """Hierarchical matching (§2.1): ``vehicle`` matches ``vehicle.forklift``.

        Without it, a new taxonomy class would silently break every existing
        query — which is why §7.2 can classify a new class as a *minor* change
        needing no consumer action.
        """
        if not self.class_ids:
            return True
        if class_id is None:
            return False
        return any(
            class_id == known or class_id.startswith(f"{known}.")
            for known in self.class_ids
        )


@dataclass(frozen=True, slots=True)
class QueryOptions:
    """What a state result carries (09_API §2.1)."""

    include_attributes: tuple[AttributeKey, ...] | None = None
    """``None`` means all; an empty tuple means none."""

    include_spatial: bool = True
    include_trajectory: bool = False
    include_provenance: bool = True
    """§2.1: *"default true — explainability is not opt-out by accident."*"""

    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A historical query's bounds (09_API §2.2).

    Against ``t_capture``, *"not ingest time"* — V11. A window against ingest time
    would return different results depending on how backed up the pipeline was.
    """

    start: Instant
    end: Instant

    def __post_init__(self) -> None:
        if self.end.ns < self.start.ns:
            raise ValueError("a time window must not end before it starts")

    @property
    def span(self) -> Duration:
        return Duration(self.end.ns - self.start.ns)

    def contains(self, moment: Instant) -> bool:
        return self.start.ns <= moment.ns <= self.end.ns


@dataclass(frozen=True, slots=True)
class ObservationFilter:
    """What to include in a historical query (09_API §2.2)."""

    observation_types: tuple[ObservationType, ...] = ()
    object_ids: tuple[ObjectId, ...] = ()
    class_ids: tuple[ClassId, ...] = ()
    attribute_keys: tuple[AttributeKey, ...] = ()
    min_confidence: float | None = None
    producer: str | None = None
    """§2.2: *"e.g. 'everything the old detector produced'"* — the query a
    deployment needs after discovering a model was misbehaving."""

    include_superseded: bool = False
    """§2.2: *"Excluded by default; retrievable for audit."*"""

    def matches(self, observation: Observation) -> bool:
        if self.observation_types and observation.observation_type not in self.observation_types:
            return False
        if self.object_ids and observation.object_id not in self.object_ids:
            return False
        if self.class_ids and not _class_matches(observation.class_id, self.class_ids):
            return False
        if self.attribute_keys:
            held = {a.key for a in observation.attributes}
            if not held & set(self.attribute_keys):
                return False
        if self.producer and str(observation.provenance.model_id or "") != self.producer:
            return False
        if self.min_confidence is not None:
            confidence = observation.confidence
            if confidence is None or not confidence.calibrated:
                return False
            if confidence.value < self.min_confidence:
                return False
        return True


def _class_matches(class_id: ClassId | None, wanted: Sequence[ClassId]) -> bool:
    if class_id is None:
        return False
    return any(class_id == known or class_id.startswith(f"{known}.") for known in wanted)


# --- result views ---------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ObjectView:
    """One object as a consumer sees it (09_API §2.3).

    A *view*, not the internal `ObjectState`. `01_LAYERED` §1.2: the L6/L7 split
    exists so *"internal representation evolve[s] while the contract holds"*, and
    that is only true if the wire shape is a separate type. Returning
    `ObjectState` directly would make every internal refactor a breaking API
    change.
    """

    object_id: ObjectId
    class_id: ClassId
    class_confidence: Confidence
    lifecycle: LifecycleState
    camera_id: CameraId
    first_seen: Instant
    last_seen: Instant
    last_confirmed: Instant
    attributes: Mapping[AttributeKey, AttributeView] = field(default_factory=dict)
    spatial: object | None = None
    trajectory: tuple[tuple[Instant, object], ...] = ()
    provenance: object | None = None
    observation_count: int = 0

    @property
    def is_stale(self) -> bool:
        """Whether the last sighting was believed rather than measured.

        Derived rather than stored so a consumer cannot receive a stale flag that
        is itself stale.
        """
        return self.last_confirmed.ns < self.last_seen.ns


@dataclass(frozen=True, slots=True)
class AttributeView:
    """One attribute value as a consumer sees it."""

    key: AttributeKey
    value: object
    confidence: Confidence
    observed_at: Instant
    valid_until: Instant | None = None
    evidence_ref: str | None = None
    """The handle for `get_evidence`. Present even when the caller lacks the
    evidence privilege: knowing an explanation exists is not the same as reading
    it, and hiding the reference would make V4 look unavailable rather than
    unauthorized."""

    def is_stale(self, now: Instant) -> bool:
        return self.valid_until is not None and now.ns > self.valid_until.ns


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Coverage as returned on **every** state query (09_API §2.1).

    > *"``coverage`` is returned on every state query, unconditionally. A consumer
    > must not be able to receive an empty or thin result without simultaneously
    > receiving the information required to interpret it (V8)."*
    """

    observable_fraction: float = 1.0
    cameras_observing: int = 0
    cameras_blind: int = 0
    cameras_degraded: int = 0
    unavailable: tuple[tuple[str, str], ...] = ()

    @property
    def fully_observable(self) -> bool:
        return self.observable_fraction >= 1.0 and not self.unavailable


@dataclass(frozen=True, slots=True)
class CapabilitySummary:
    """What the platform can currently produce (09_API §5.2).

    *"Capability is live state, not documentation"* — a model evicted under memory
    pressure changes this report, and a consumer polling it can react rather than
    silently degrading.
    """

    taxonomy_version: str = ""
    producible_classes: tuple[ClassId, ...] = ()
    producible_attributes: tuple[AttributeKey, ...] = ()
    gaps: tuple[tuple[str, str], ...] = ()
    models_in_use: tuple[tuple[str, str, str], ...] = ()
    effective_since: Instant | None = None


@dataclass(frozen=True, slots=True)
class StateResult:
    """The answer to `query_state` (09_API §2.1).

    ``coverage`` has no default and cannot be omitted — the field is required by
    construction, which is the mechanical form of §2.1's "unconditionally".
    """

    objects: tuple[ObjectView, ...]
    snapshot: SnapshotView
    coverage: CoverageSummary
    capabilities: CapabilitySummary = field(default_factory=CapabilitySummary)
    cursor: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every requested partition answered.

        A consumer concluding "nothing is here" must check this and ``coverage``
        first — obligation C5.
        """
        return self.coverage.fully_observable and not self.snapshot.incomplete


@dataclass(frozen=True, slots=True)
class SnapshotView:
    """The consistency statement accompanying every state result (07_STATE §5.2)."""

    partitions: tuple[CameraId, ...] = ()
    consistency: str = "strong"
    max_lag_ms: float = 0.0
    incomplete: tuple[tuple[str, str], ...] = ()
    taken_at: Instant | None = None

    @classmethod
    def of(cls, snapshot: StateSnapshot) -> SnapshotView:
        return cls(
            partitions=tuple(snapshot.partitions),
            consistency=snapshot.consistency.value,
            max_lag_ms=snapshot.max_lag.millis,
            incomplete=tuple(
                (str(camera), reason) for camera, reason in snapshot.incomplete
            ),
            taken_at=snapshot.taken_at,
        )


@dataclass(frozen=True, slots=True)
class Page:
    """A page of historical observations (09_API §2.2).

    > *"Cursor stability is a direct dividend of immutability. In a mutable store,
    > paginating through a changing dataset either repeats or skips records."*
    """

    observations: tuple[Observation, ...] = ()
    cursor: str | None = None
    window_fully_observable: bool = True
    """§2.2: *"The page reports whether the window was fully observable."*"""

    coverage: CoverageSummary = field(default_factory=CoverageSummary)

    @property
    def count(self) -> int:
        return len(self.observations)


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """The justification behind one observation (09_API §6).

    > *"This contract is what makes V4 usable rather than theoretical. An
    > observation that is explainable in principle but whose explanation cannot be
    > retrieved is not explainable in practice."*
    """

    observation_id: ObservationId
    trigger_reason: str = ""
    crop: bytes | None = None
    raw_model_output: str | None = None
    unstructured_note: str | None = None
    decision_path: tuple[str, ...] = ()
    provenance: object | None = None
    timing: object | None = None
    quality: object | None = None


# --- subscription contracts ------------------------------------------------ #


class DeliveryMode(enum.Enum):
    """09_API §3.2."""

    ALL = "all"
    """Every matching observation, at-least-once. For anything that must not miss."""

    CONFLATED = "conflated"
    """Latest per object. For dashboards and overlays."""

    SAMPLED = "sampled"
    """Rate-limited sample. For monitoring."""


class OverflowPolicy(enum.Enum):
    """What to do with a slow subscriber (09_API §3.4).

    Every option is explicit about loss. The three §3.4 forbids — unbounded
    buffering, silent drop, stalling the platform — are not members of this enum,
    so they cannot be configured.
    """

    CONFLATE = "conflate"
    DROP_WITH_GAP = "drop_with_gap"
    DISCONNECT = "disconnect"


class GapReason(enum.Enum):
    """Why a subscriber missed messages (09_API §3.3)."""

    SLOW_CONSUMER = "slow_consumer"
    PLATFORM_BLIND = "platform_blind"
    BUDGET_SHED = "budget_shed"
    PARTITION_UNAVAILABLE = "partition_unavailable"
    RETENTION_EXPIRED = "retention_expired"

    @property
    def recoverable(self) -> bool:
        """Whether `query_observations` can backfill the window.

        A consumer that was slow can fetch what it missed; a platform that was
        blind has nothing to give. §3.3 makes this a field on the Gap because
        *"a well-built consumer does exactly that."*
        """
        return self in (
            GapReason.SLOW_CONSUMER,
            GapReason.BUDGET_SHED,
            GapReason.PARTITION_UNAVAILABLE,
        )


@dataclass(frozen=True, slots=True)
class DeliveryPolicy:
    """How a subscription behaves under pressure (09_API §3.1)."""

    mode: DeliveryMode = DeliveryMode.ALL
    overflow: OverflowPolicy = OverflowPolicy.DROP_WITH_GAP
    max_lag: Duration = Duration.from_millis(5_000)
    batch_window: Duration | None = None
    heartbeat: Duration = Duration.from_millis(10_000)
    """§3.1's default. A subscriber that hears nothing for longer than this knows
    the connection is dead rather than the scene being quiet."""

    resume_from: str | None = None
    queue_capacity: int = 1_024
    """Bounded, always. §3.4: *"Never: unbounded buffering."*"""

    def __post_init__(self) -> None:
        if self.queue_capacity < 1:
            raise ValueError("a subscription queue must have positive capacity")
        if self.heartbeat.ns <= 0:
            raise ValueError(
                "a subscription must have a positive heartbeat; without one, a "
                "dead connection and a quiet scene are indistinguishable (V8)"
            )


@dataclass(frozen=True, slots=True)
class Gap:
    """The most important message type in the subscription contract (09_API §3.3).

    > *"A subscriber is never silently skipped. If the platform drops messages for
    > any reason — the consumer was slow, the camera was blind, the budget was
    > exhausted — an explicit `Gap` is delivered. This is V8 applied to delivery."*
    """

    start: Instant
    end: Instant
    reason: GapReason
    cameras: tuple[CameraId, ...] = ()
    observations_missed: int | None = None
    recoverable: bool = True

    @property
    def span(self) -> Duration:
        return Duration(max(0, self.end.ns - self.start.ns))


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """Liveness. Carries the cursor so a reconnect loses nothing."""

    at: Instant
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageChange:
    """Observability changed inside the subscription's scope."""

    camera_id: CameraId
    status: str
    reason: str
    at: Instant
    effective_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class StateDeltaMessage:
    """What changed in state, for subscribers watching a projection."""

    camera_id: CameraId
    changed_objects: tuple[ObjectId, ...] = ()
    changed_regions: tuple[RegionId, ...] = ()
    at: Instant | None = None


#: Everything a subscriber can receive (09_API §3.1).
Message = Observation | StateDeltaMessage | Gap | Heartbeat | CoverageChange


@dataclass(frozen=True, slots=True)
class SubscriptionFilter:
    """What a subscription matches."""

    observation_types: tuple[ObservationType, ...] = ()
    class_ids: tuple[ClassId, ...] = ()
    object_ids: tuple[ObjectId, ...] = ()
    attribute_keys: tuple[AttributeKey, ...] = ()

    def matches(self, observation: Observation) -> bool:
        if self.observation_types and observation.observation_type not in self.observation_types:
            return False
        if self.class_ids and not _class_matches(observation.class_id, self.class_ids):
            return False
        if self.object_ids and observation.object_id not in self.object_ids:
            return False
        if self.attribute_keys:
            held = {a.key for a in observation.attributes}
            if not held & set(self.attribute_keys):
                return False
        return True


# --- audit ------------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Who read what (§M14 responsibility 8, 12_SECURITY §8).

    ``purpose`` is populated for evidence access, where 12_SECURITY §5.4 requires
    it: *"This does not technically prevent misuse — nothing at this layer can —
    but it converts imagery access from an invisible act into an attributable one,
    which is the control that actually changes behaviour."*
    """

    at: Instant
    principal: str
    tenant_id: TenantId
    action: Action
    resource: str = ""
    granted: bool = True
    purpose: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ApiErrorView:
    """The error envelope a consumer receives (09_API §8).

    ``retryable`` is a **field**, never a property the consumer derives. §8:
    *"Consumers must never infer retryability from a status code or a message
    string. Inferring it is how retry storms begin."*
    """

    code: str
    message: str
    retryable: bool
    retry_after: Duration | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class DemandView:
    """A registered demand as its owner sees it."""

    demand_id: DemandId
    subscriber: str
    status: str
    required_attributes: tuple[AttributeKey, ...] = ()
    effective_freshness: Duration | None = None
    unsatisfiable: tuple[tuple[AttributeKey, str], ...] = ()


def object_view_of(
    state: ObjectState,
    camera_id: CameraId,
    *,
    options: QueryOptions,
) -> ObjectView:
    """Project internal state onto the wire shape.

    The one place the two representations meet. Everything the consumer sees is
    constructed here, so a field added to `ObjectState` is invisible until someone
    deliberately exposes it — which is the L6/L7 split doing its job.
    """
    attributes: dict[AttributeKey, AttributeView] = {}
    wanted = options.include_attributes
    for key, held in state.attributes.items():
        if wanted is not None and key not in wanted:
            continue
        attributes[key] = AttributeView(
            key=key,
            value=held.value,
            confidence=held.confidence,
            observed_at=held.observed_at,
            valid_until=held.valid_until,
            evidence_ref=(
                str(held.evidence_ref.evidence_id) if held.evidence_ref else None
            ),
        )

    return ObjectView(
        object_id=state.object_id,
        class_id=state.class_id,
        class_confidence=state.class_confidence,
        lifecycle=state.lifecycle,
        camera_id=camera_id,
        first_seen=state.first_seen,
        last_seen=state.last_seen,
        last_confirmed=state.last_confirmed,
        attributes=attributes,
        spatial=state.spatial if options.include_spatial else None,
        trajectory=tuple(state.trajectory) if options.include_trajectory else (),
        provenance=state.provenance_summary if options.include_provenance else None,
        observation_count=state.observation_count,
    )


def coverage_summary_of(report: CoverageReport, *, unavailable=()) -> CoverageSummary:
    """Build the summary every state query carries."""
    return CoverageSummary(
        observable_fraction=report.observable_fraction,
        unavailable=tuple(unavailable),
    )
