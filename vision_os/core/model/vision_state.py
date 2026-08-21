"""The Vision State structure (07_STATE §3).

> **The Vision State is the platform's answer to one question: *what is visible
> right now, where, since when, and how confident are we?*** — plus the honest
> admission of where it cannot see.

It is a **materialized projection**, not a database of record. The system of
record is the immutable observation log; state is what the log means at this
instant, kept current because recomputing it per query would be absurd.

Three properties define it (§1.1), and each shows up as a design decision here:

**Owned** — every structure in this module is frozen. There is no setter, no
mutating method, no write path a consumer could reach. §1.2: allowing an external
write would destroy derivability, explainability, the Semantic Ceiling and the
single-writer design *simultaneously*.

**Derived** — nothing here is constructed except by projecting an observation.
The types carry no factory that invents state.

**Honest** — `staleness`, `is_stale`, `measurement_basis`, `ObservabilityState`
and `incomplete` are all V8 made structural. §3.1: *"Systems that omit this force
every consumer to reimplement staleness reasoning, and most get it wrong."*

**What is deliberately absent** (§10): business entities, thresholds with business
meaning, alerts, aggregations, raw video, cross-tenant anything. The test for any
proposed field is *"would this mean the same thing in a hospital, a warehouse and
a city street?"*
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from .confidence import Confidence
from .ids import (
    AttributeKey,
    CalibrationId,
    CameraId,
    ClassId,
    LogPosition,
    ObjectId,
    ObservationId,
    PartitionVersion,
    RegionId,
    SiteId,
    StreamEpoch,
    TrackerEpoch,
)
from .observation import (
    EvidenceRef,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
)
from .provenance import Provenance
from .space import Point, SpatialInfo
from .timebase import Duration, Instant
from .visual_object import LifecycleState

STATE_SCHEMA_VERSION = "1.0.0"

#: Default ring bounds (07_STATE §6.3).
#:
#: *"All in-memory history is bounded by both count and time, and the bound is a
#: structural property of the ring buffers rather than a tunable that might be
#: misconfigured to infinity."* This is the platform's principal defence against
#: the 30-day soak failure where memory grows imperceptibly until a node dies.
DEFAULT_TRAJECTORY_POINTS = 64
DEFAULT_ATTRIBUTE_HISTORY = 8
DEFAULT_CLASS_HISTORY = 16


class ConsistencyLevel(enum.Enum):
    """07_STATE §5.2. What a read actually guarantees.

    Reported rather than assumed, because *"the platform never fabricates a
    global instant. In a distributed deployment there is no such moment, and
    pretending otherwise produces answers that are wrong in ways nobody can
    detect."*
    """

    STRONG = "strong"
    """One object or one partition. One writer, one atomic version."""

    SNAPSHOT_SET = "snapshot_set"
    """Several partitions, each at its own version. **Not a global instant.**"""

    EVENTUAL = "eventual"
    """Across nodes, or a site aggregate. Per-partition versions and a lag bound
    are reported alongside."""


@dataclass(frozen=True, slots=True)
class AttributeState:
    """07_STATE §3.1's ``AttributeState``.

    ``is_stale`` is derived rather than stored: a stored flag would be wrong the
    moment the clock moved past it, and a consumer reading a stale flag that says
    "fresh" is worse off than one computing it.
    """

    key: AttributeKey
    value: object
    confidence: Confidence
    observed_at: Instant
    valid_until: Instant | None = None
    evidence_ref: EvidenceRef | None = None
    producer: Provenance | None = None
    previous: tuple[tuple[object, Instant], ...] = ()
    """Bounded ring of prior values. Short attribute history, for perception —
    not for analytics (§6.1)."""

    def is_stale(self, now: Instant) -> bool:
        """The object-level expression of V8.

        A consumer reading ``headwear_present: false`` observed 40 minutes ago is
        reading something quite different from a fresh measurement, and the state
        says so without being asked.
        """
        return self.valid_until is not None and now.ns > self.valid_until.ns

    def age(self, now: Instant) -> Duration:
        return Duration(max(0, now.ns - self.observed_at.ns))

    def revised_with(
        self, value: object, confidence: Confidence, at: Instant, **rest
    ) -> AttributeState:
        """A new state carrying this one into ``previous``. Never a mutation."""
        history = ((self.value, self.observed_at), *self.previous)
        return replace(
            self,
            value=value,
            confidence=confidence,
            observed_at=at,
            previous=history[:DEFAULT_ATTRIBUTE_HISTORY],
            **rest,
        )


@dataclass(frozen=True, slots=True)
class RegionMembership:
    """One object's presence in one region, with its dwell accumulator.

    ``dwell`` is a **duration**, never a verdict. 07_STATE §3.3 is emphatic that
    this is where the ceiling is most tempting to breach.
    """

    region_id: RegionId
    geometry_version: str
    entered_at: Instant
    last_confirmed: Instant
    containment: float = 1.0
    method: str = ""

    def dwell(self, now: Instant) -> Duration:
        return Duration(max(0, now.ns - self.entered_at.ns))


@dataclass(frozen=True, slots=True)
class MotionState:
    """Descriptive motion. No prediction, no intent."""

    velocity: Point | None = None
    heading_degrees: float | None = None
    motion_state: str = "unknown"
    """``stationary`` | ``moving`` | ``unknown``. A description of measured
    displacement, never an inference about purpose."""


@dataclass(frozen=True, slots=True)
class IdentitySummary:
    """What the registry asserted, carried into state as a claim.

    ``method`` and ``confidence`` travel because 02_VOM §4.2 makes identity an
    *assertion* rather than a truth — and a consumer ranking objects by identity
    confidence must know how the claim was made.
    """

    binding_count: int = 0
    assertion_confidence: Confidence | None = None
    method: str = ""
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class ObjectState:
    """07_STATE §3.1 — the primary entity.

    Frozen, and updated only by producing a new value. That is what makes
    §5.1's structural sharing work: an unchanged object is the *same object*,
    shared between snapshots rather than copied into each.
    """

    object_id: ObjectId
    class_id: ClassId
    class_confidence: Confidence
    lifecycle: LifecycleState

    first_seen: Instant
    last_seen: Instant
    last_confirmed: Instant
    """The last **measured** sighting. Distinct from ``last_seen``, which
    includes predicted updates — conflating them would let a coasting object look
    freshly observed."""

    spatial: SpatialInfo | None = None
    measurement_basis: MeasurementBasis = MeasurementBasis.MEASURED
    motion: MotionState = MotionState()
    trajectory: tuple[tuple[Instant, Point, MeasurementBasis], ...] = ()

    class_history: tuple[tuple[ClassId, Instant, Confidence], ...] = ()
    regions: Mapping[RegionId, RegionMembership] = field(default_factory=dict)
    attributes: Mapping[AttributeKey, AttributeState] = field(default_factory=dict)
    identity: IdentitySummary = IdentitySummary()

    provenance_summary: Mapping[str, str] = field(default_factory=dict)
    """``{last_detector, last_tracker, last_understander}``. A summary, so a
    consumer can see which producers shaped this object without walking the log."""

    observation_count: int = 0
    last_observation: ObservationId | None = None
    schema_version: str = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object state requires an object_id")
        if self.last_confirmed.ns > self.last_seen.ns:
            raise ValueError(
                "last_confirmed cannot be later than last_seen; a measured "
                "sighting is a kind of sighting"
            )
        if self.first_seen.ns > self.last_seen.ns:
            raise ValueError("first_seen cannot be later than last_seen")

    def staleness(self, now: Instant) -> Duration:
        """Time since the last **measured** sighting (§3.1).

        From ``last_confirmed`` rather than ``last_seen``, deliberately: an object
        coasting on prediction for 40 seconds has not been seen for 40 seconds,
        however recently its position was updated.
        """
        return Duration(max(0, now.ns - self.last_confirmed.ns))

    def stale_attributes(self, now: Instant) -> tuple[AttributeKey, ...]:
        return tuple(
            key for key, state in self.attributes.items() if state.is_stale(now)
        )

    @property
    def is_present(self) -> bool:
        return self.lifecycle in (
            LifecycleState.PROVISIONAL,
            LifecycleState.ACTIVE,
            LifecycleState.OCCLUDED,
        )

    def attribute(self, key: AttributeKey) -> AttributeState | None:
        return self.attributes.get(key)


@dataclass(frozen=True, slots=True)
class RegionState:
    """07_STATE §3.3.

    > *This is exactly where the Semantic Ceiling is most tempting to breach.*
    > *`occupancy` is a count. `dwell_stats` are descriptive statistics over
    > durations. There is no `is_crowded`, no `exceeds_capacity`, no
    > `queue_forming` — each of those requires a threshold or a definition that
    > only a consumer possesses (V1).*
    """

    region_id: RegionId
    geometry_version: str
    occupancy: Mapping[ClassId, int] = field(default_factory=dict)
    present_objects: tuple[ObjectId, ...] = ()
    dwell_current_max: Duration = Duration(0)
    dwell_current_mean: Duration = Duration(0)
    last_transition: Instant | None = None

    @property
    def total_occupancy(self) -> int:
        return sum(self.occupancy.values())


@dataclass(frozen=True, slots=True)
class ObservabilityState:
    """07_STATE §7.2 — the structural implementation of V8.

    A consumer querying a window with an empty result must be able to tell
    *"the region was observed and was empty"* from *"the camera was blind"*.
    Without this they are the same answer, and in a hospital or a factory that is
    a safety issue rather than a data-quality one.
    """

    camera_id: CameraId
    status: ObservabilityStatus = ObservabilityStatus.OBSERVING
    since: Instant = Instant(0)
    reason: ObservabilityReason = ObservabilityReason.NORMAL
    effective_rate: float = 1.0
    regions_affected: tuple[RegionId, ...] = ()
    capability_gaps: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.effective_rate <= 1.0:
            raise ValueError("effective_rate must be in [0,1]")

    @property
    def is_observing(self) -> bool:
        return self.status.is_observing


@dataclass(frozen=True, slots=True)
class CameraStatus:
    """Per-partition health, epoch and clock quality (07_STATE §3.2)."""

    camera_id: CameraId
    stream_epoch: StreamEpoch = StreamEpoch(0)
    tracker_epoch: TrackerEpoch = TrackerEpoch(0)
    calibration_id: CalibrationId | None = None
    health: str = "healthy"
    last_observation_at: Instant | None = None


@dataclass(frozen=True, slots=True)
class CameraPartition:
    """07_STATE §3.2 — the unit of ownership.

    > *The camera is the partition. Each partition has exactly one writer.*

    Frozen and replaced wholesale on every applied observation. ``version``
    increments monotonically so a snapshot holder can name exactly what it holds,
    and ``log_position`` is the projection watermark that makes *"is the
    projection caught up?"* answerable.
    """

    camera_id: CameraId
    objects: Mapping[ObjectId, ObjectState] = field(default_factory=dict)
    regions: Mapping[RegionId, RegionState] = field(default_factory=dict)
    status: CameraStatus | None = None
    observability: ObservabilityState | None = None
    log_position: LogPosition = LogPosition(0)
    version: PartitionVersion = PartitionVersion(0)
    quarantined: int = 0
    """Observations that could not be projected. §M12: *"Quarantine that
    observation, continue the projection, alarm. One bad record must not stop the
    world."*"""

    degraded_reason: str = ""
    """Non-empty when the partition has stopped accepting observations. §4.4 step
    4 halts loudly rather than dropping facts silently."""

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded_reason)

    def present_objects(self) -> tuple[ObjectState, ...]:
        return tuple(o for o in self.objects.values() if o.is_present)

    def object_state(self, object_id: ObjectId) -> ObjectState | None:
        return self.objects.get(object_id)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """07_STATE §7.3's ``CoverageReport``.

    > *A consumer querying a historical window should always request coverage
    > alongside results... an empty result without its coverage context is not an
    > answer — it is half of one.*
    """

    observable_fraction: float
    gaps: tuple[tuple[Instant, Instant, ObservabilityReason, str], ...] = ()
    effective_rate: float = 1.0
    capability_gaps: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.observable_fraction <= 1.0:
            raise ValueError("observable_fraction must be in [0,1]")

    @property
    def fully_observed(self) -> bool:
        return self.observable_fraction >= 1.0 and not self.gaps


@dataclass(frozen=True, slots=True)
class CoverageMap:
    """Live coverage across a site (07_STATE §7.3): *"can we see right now?"*"""

    by_camera: Mapping[CameraId, ObservabilityState] = field(default_factory=dict)
    at: Instant = Instant(0)

    @property
    def observing(self) -> tuple[CameraId, ...]:
        return tuple(
            camera for camera, state in self.by_camera.items() if state.is_observing
        )

    @property
    def blind(self) -> tuple[CameraId, ...]:
        return tuple(
            camera for camera, state in self.by_camera.items() if not state.is_observing
        )

    @property
    def observable_fraction(self) -> float:
        if not self.by_camera:
            return 0.0
        return len(self.observing) / len(self.by_camera)


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """07_STATE §3.4 — what the *currently loaded* models can produce here.

    > *`capabilities` is state, not documentation... When a model is evicted under
    > memory pressure, or a prompt pack fails to load, the capability report
    > changes — and a consumer discovers the gap instead of waiting indefinitely
    > for an attribute that will never arrive (V8).*
    """

    producible_classes: frozenset[ClassId] = frozenset()
    producible_attributes: frozenset[AttributeKey] = frozenset()
    gaps: tuple[tuple[str, str], ...] = ()

    def can_produce(self, key: AttributeKey) -> bool:
        return key in self.producible_attributes


@dataclass(frozen=True, slots=True)
class SiteContext:
    """07_STATE §3.4. Aggregated **eventually**, never transactionally."""

    site_id: SiteId
    coverage: CoverageMap = CoverageMap()
    capabilities: CapabilityReport = CapabilityReport()
    camera_count: int = 0
    object_count: int = 0
    aggregate_health: str = "healthy"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """07_STATE §5.2's ``StateSnapshot``. **O(1) to take.**

    A pointer to immutable roots, not a copy. Readers hold one as long as they
    need; the writer continues producing new versions. *"No reader blocks a
    writer; no writer blocks a reader; no reader blocks another reader."*

    ``incomplete`` is the V8 property of snapshots: a query spanning 40 cameras
    that could only reach 37 returns 37 **and says which three are missing** —
    never a silently smaller answer.
    """

    partitions: Mapping[CameraId, CameraPartition] = field(default_factory=dict)
    site: SiteContext | None = None
    consistency: ConsistencyLevel = ConsistencyLevel.STRONG
    max_lag: Duration = Duration(0)
    incomplete: tuple[tuple[CameraId, str], ...] = ()
    taken_at: Instant = Instant(0)

    def __post_init__(self) -> None:
        if len(self.partitions) > 1 and self.consistency is ConsistencyLevel.STRONG:
            raise ValueError(
                "a multi-partition snapshot cannot claim strong consistency; "
                "there is no global instant to be strongly consistent with, and "
                "claiming one produces answers that are wrong in ways nobody can "
                "detect (07_STATE section 5.2)"
            )

    @property
    def object_count(self) -> int:
        return sum(p.object_count for p in self.partitions.values())

    @property
    def is_complete(self) -> bool:
        return not self.incomplete

    def partition(self, camera_id: CameraId) -> CameraPartition | None:
        return self.partitions.get(camera_id)

    def object_state(self, object_id: ObjectId) -> ObjectState | None:
        for partition in self.partitions.values():
            found = partition.objects.get(object_id)
            if found is not None:
                return found
        return None

    def versions(self) -> dict[CameraId, PartitionVersion]:
        """Per-partition versions, so a reader can say exactly what it holds."""
        return {camera: p.version for camera, p in self.partitions.items()}


@dataclass(frozen=True, slots=True)
class StateDelta:
    """What changed in one partition, for subscribers (§M12 responsibility 8).

    Carries ids rather than whole objects: a subscriber that wants the new state
    reads a snapshot, and pushing full objects would make every subscriber a copy
    of the projection.
    """

    camera_id: CameraId
    version: PartitionVersion
    log_position: LogPosition
    changed_objects: tuple[ObjectId, ...] = ()
    changed_regions: tuple[RegionId, ...] = ()
    coverage_changed: bool = False
    at: Instant = Instant(0)

    @property
    def is_empty(self) -> bool:
        return not (
            self.changed_objects or self.changed_regions or self.coverage_changed
        )


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What `append` did (§M12 public API).

    ``rejected`` and ``quarantined`` are separate: a rejected observation never
    entered the log, while a quarantined one is *in* the log and could not be
    projected. The first is a producer problem, the second a projection bug — and
    §M12 prescribes different responses.
    """

    accepted: int = 0
    rejected: tuple[tuple[ObservationId, str], ...] = ()
    quarantined: tuple[tuple[ObservationId, str], ...] = ()
    log_position: LogPosition = LogPosition(0)
    version: PartitionVersion = PartitionVersion(0)
    degraded: bool = False
    detail: str = ""

    @property
    def committed(self) -> bool:
        return self.accepted > 0 and not self.degraded

    @property
    def total(self) -> int:
        return self.accepted + len(self.rejected) + len(self.quarantined)


def bounded_ring(
    existing: Sequence, item: object, *, limit: int
) -> tuple:
    """Prepend into a bounded ring. The structural bound of §6.3.

    Newest first, so the common read — "what is the current value's predecessor"
    — is index zero, and the bound is applied at every write rather than swept
    later. A ring that is trimmed on a schedule is a ring that is unbounded
    between sweeps.
    """
    if limit < 1:
        raise ValueError("a ring bound must be at least 1")
    return ((item, *existing))[:limit]
