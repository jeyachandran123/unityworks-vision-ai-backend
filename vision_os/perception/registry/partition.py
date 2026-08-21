"""The camera partition — M7's single-writer object store.

> **Single responsibility:** *Own one camera's objects. Be the only thing that
> writes them.*

``07_STATE`` section 4.1: *"The camera is the partition. Each partition has
exactly one writer. Cross-partition operations are explicitly eventually
consistent."* ``08_RUNTIME`` section 2 places M7 in the actor table: *"A camera's
objects are owned by one registry actor."*

Three properties are structural rather than conventional:

**Objects are frozen; every mutation produces a new instance.** A consumer holding
a ``VisualObject`` holds a snapshot that cannot drift under it. This is what makes
"single writer, multiple readers" true without readers taking locks.

**Ids are minted here and nowhere else.** ``01_LAYERED`` section 8: *"Exactly one
module may mint or retire an object identity. Diffusing this is how ID chaos
begins."*

**Every bound is finite.** The population is capped, spatial history is a ring,
and class history is a ring. Section M7 calls unbounded history here *"the most
likely long-run memory leak in the entire platform"*.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from ...core.errors import (
    IdentityConflictError,
    ObjectNotFoundError,
    RegistryCapacityError,
)
from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.ids import (
    AttributeKey,
    BindingId,
    CameraId,
    ClassId,
    ObjectId,
    SiteId,
    TenantId,
    TrackId,
    new_ulid,
)
from ...core.model.provenance import Provenance
from ...core.model.space import SpatialInfo
from ...core.model.timebase import Duration, Instant
from ...core.model.visual_object import (
    Attribute,
    BindingMethod,
    ClassObservation,
    LifecycleState,
    TrackBinding,
    VisualObject,
)
from .lifecycle import LifecyclePolicy

#: Retained spatial samples per object. Bounded structurally (section M7).
DEFAULT_SPATIAL_HISTORY = 64

#: Retained class observations per object. Enough to resolve flapping over a
#: realistic window without becoming a second history store.
DEFAULT_CLASS_HISTORY = 32


@dataclass(slots=True)
class ClassDistribution:
    """Accumulated class evidence for one object.

    Section M7 responsibility 4: *"Resolve class flapping using the retained
    class distribution."* A tracked object detected as ``person`` then
    ``person.child`` then ``person`` must not flip its published class on every
    frame — but neither may the registry *rewrite* what was previously asserted.
    So the distribution decides the current best class, and ``class_history``
    records what was seen, unedited.
    """

    weights: dict[ClassId, float] = field(default_factory=dict)
    observations: int = 0

    def observe(self, class_id: ClassId, confidence: float) -> None:
        self.weights[class_id] = self.weights.get(class_id, 0.0) + max(0.0, confidence)
        self.observations += 1

    def best(self) -> tuple[ClassId, float] | None:
        """Highest-weight class and its share of total evidence.

        Ties break on the class id so the answer is deterministic (V13); an
        arbitrary tie-break would make the published class depend on dict order.
        """
        if not self.weights:
            return None
        total = sum(self.weights.values())
        if total <= 0.0:
            return None
        best_id = max(sorted(self.weights), key=lambda c: self.weights[c])
        return best_id, self.weights[best_id] / total

    def share(self, class_id: ClassId) -> float:
        total = sum(self.weights.values())
        return self.weights.get(class_id, 0.0) / total if total > 0 else 0.0


@dataclass(slots=True)
class ObjectRecord:
    """Mutable per-object bookkeeping. Internal; never leaves the layer.

    The immutable ``VisualObject`` handed to consumers is projected from this.
    Keeping them separate is what lets the published record be frozen (V5) while
    the partition still updates cheaply in place.
    """

    object_id: ObjectId
    camera_id: CameraId
    tenant_id: TenantId
    site_id: SiteId

    lifecycle: LifecycleState
    class_id: ClassId
    distribution: ClassDistribution

    spatial: SpatialInfo
    first_seen: Instant
    last_seen: Instant
    last_confirmed: Instant

    identity_confidence: float
    observation_count: int = 0
    unmeasured_frames: int = 0

    bindings: list[TrackBinding] = field(default_factory=list)
    class_history: deque[ClassObservation] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_CLASS_HISTORY)
    )
    spatial_history: deque[tuple[Instant, SpatialInfo]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_SPATIAL_HISTORY)
    )
    attributes: dict[AttributeKey, Attribute] = field(default_factory=dict)

    merged_into: ObjectId | None = None
    lineage: tuple[ObjectId, ...] = ()

    @property
    def open_binding(self) -> TrackBinding | None:
        for binding in reversed(self.bindings):
            if binding.is_open and not binding.is_superseded:
                return binding
        return None

    @property
    def bound_track(self) -> TrackId | None:
        binding = self.open_binding
        return binding.track_id if binding else None


class RegistryPartition:
    """One camera's objects. Single-writer by construction.

    Not thread-safe, and deliberately so: safety comes from the actor owning it,
    not from locks inside it. A lock here would suggest concurrent writers are
    expected, which would quietly license exactly the design the sharding model
    exists to prevent.
    """

    __slots__ = (
        "_camera_id",
        "_class_history",
        "_policy",
        "_provenance",
        "_records",
        "_sequence",
        "_site_id",
        "_spatial_history",
        "_tenant_id",
        "_version",
    )

    def __init__(
        self,
        camera_id: CameraId,
        *,
        tenant_id: TenantId,
        site_id: SiteId,
        policy: LifecyclePolicy,
        provenance: Provenance,
        spatial_history: int = DEFAULT_SPATIAL_HISTORY,
        class_history: int = DEFAULT_CLASS_HISTORY,
    ) -> None:
        if spatial_history < 1 or class_history < 1:
            raise ValueError("history bounds must be >= 1")
        self._camera_id = camera_id
        self._tenant_id = tenant_id
        self._site_id = site_id
        self._policy = policy
        # Injected, never invented: without the exact config revision that
        # governed a result, no object is reproducible six months later (V4).
        self._provenance = provenance
        self._spatial_history = spatial_history
        self._class_history = class_history
        self._records: dict[ObjectId, ObjectRecord] = {}
        self._sequence = 0
        self._version = 0

    # --- identity ------------------------------------------------------------ #

    @property
    def camera_id(self) -> CameraId:
        return self._camera_id

    @property
    def site_id(self) -> SiteId:
        return self._site_id

    @property
    def version(self) -> int:
        """Monotonic, incremented on every mutation. Lets a reader detect drift."""
        return self._version

    @property
    def sequence(self) -> int:
        return self._sequence

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._records

    # --- reads --------------------------------------------------------------- #

    def get(self, object_id: ObjectId) -> VisualObject:
        """Project one object.

        Raises:
            ObjectNotFoundError: no such object in this partition. A *merged*
                object is still found and reports where it went — history stays
                resolvable (V5).
        """
        record = self._records.get(object_id)
        if record is None:
            raise ObjectNotFoundError(
                f"object '{object_id}' is not in partition '{self._camera_id}'",
                object_id=str(object_id),
                camera_id=str(self._camera_id),
            )
        return self._project(record)

    def find(self, object_id: ObjectId) -> VisualObject | None:
        record = self._records.get(object_id)
        return self._project(record) if record is not None else None

    def resolve(self, object_id: ObjectId) -> VisualObject | None:
        """Follow ``merged_into`` to the surviving object.

        The reason merge is not deletion: an observation recorded against the old
        id remains answerable.
        """
        seen: set[ObjectId] = set()
        current = self._records.get(object_id)
        while current is not None and current.merged_into is not None:
            if current.object_id in seen:
                return None  # a cycle cannot occur, but never loop on corruption
            seen.add(current.object_id)
            current = self._records.get(current.merged_into)
        return self._project(current) if current is not None else None

    def objects(self) -> tuple[VisualObject, ...]:
        """Every object in stable id order, terminal ones included."""
        return tuple(self._project(self._records[key]) for key in sorted(self._records))

    def active(self) -> tuple[VisualObject, ...]:
        """Objects the platform believes are present."""
        return tuple(o for o in self.objects() if o.is_present)

    def record_for(self, object_id: ObjectId) -> ObjectRecord | None:
        """Internal handle. Callers inside the layer only."""
        return self._records.get(object_id)

    def records(self) -> tuple[ObjectRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def by_track(self, track_id: TrackId) -> ObjectRecord | None:
        """The object currently bound to this track, if any."""
        for key in sorted(self._records):
            record = self._records[key]
            if record.bound_track == track_id and not record.lifecycle.is_terminal:
                return record
        return None

    # --- writes -------------------------------------------------------------- #

    def mint(
        self,
        *,
        class_id: ClassId,
        confidence: float,
        spatial: SpatialInfo,
        now: Instant,
        class_confidence: float = 1.0,
    ) -> ObjectRecord:
        """Create an object. **The only place an ``ObjectId`` comes into being.**

        Raises:
            RegistryCapacityError: the population is capped and nothing could be
                shed. Refusing is the bounded behaviour section M7 requires; the
                alternative is a registry that grows until the node dies.
        """
        if len(self._records) >= self._policy.max_objects_per_camera:
            raise RegistryCapacityError(
                f"partition '{self._camera_id}' holds "
                f"{len(self._records)} objects (max "
                f"{self._policy.max_objects_per_camera}); refusing to grow",
                camera_id=str(self._camera_id),
                capacity=self._policy.max_objects_per_camera,
            )

        # ULID minted from the injected clock, not wall time: identity generation
        # must not reintroduce hidden time (V13).
        object_id = ObjectId(new_ulid(now_ms=now.ns // 1_000_000))
        self._sequence += 1

        distribution = ClassDistribution()
        distribution.observe(class_id, class_confidence)

        record = ObjectRecord(
            object_id=object_id,
            camera_id=self._camera_id,
            tenant_id=self._tenant_id,
            site_id=self._site_id,
            lifecycle=LifecycleState.PROVISIONAL,
            class_id=class_id,
            distribution=distribution,
            spatial=spatial,
            first_seen=now,
            last_seen=now,
            last_confirmed=now,
            identity_confidence=confidence,
            observation_count=1,
            class_history=deque(maxlen=self._class_history),
            spatial_history=deque(maxlen=self._spatial_history),
        )
        record.class_history.append(
            ClassObservation(
                class_id=class_id,
                observed_at=now,
                confidence=Confidence.uncalibrated(
                    class_confidence, ConfidenceSemantics.CLASSIFICATION
                ),
            )
        )
        record.spatial_history.append((now, spatial))
        self._records[object_id] = record
        self._version += 1
        return record

    def observe(
        self,
        record: ObjectRecord,
        *,
        class_id: ClassId,
        class_confidence: float,
        spatial: SpatialInfo,
        now: Instant,
        measured: bool,
    ) -> None:
        """Fold one sighting into an object.

        ``measured`` is what separates ``last_confirmed`` from ``last_seen`` —
        *measured* versus *believed*, the object-level expression of V8. A
        predicted position advances the object's clock but not its evidence.
        """
        record.last_seen = now
        record.observation_count += 1

        if measured:
            record.last_confirmed = now
            record.unmeasured_frames = 0
            record.spatial = spatial
            record.spatial_history.append((now, spatial))
            record.distribution.observe(class_id, class_confidence)
            record.class_history.append(
                ClassObservation(
                    class_id=class_id,
                    observed_at=now,
                    confidence=Confidence.uncalibrated(
                        class_confidence, ConfidenceSemantics.CLASSIFICATION
                    ),
                )
            )
            resolved = record.distribution.best()
            if resolved is not None:
                # The published class follows the accumulated distribution, never
                # the latest frame. class_history keeps what was actually seen —
                # section M7: "never silently rewrite past class assertions".
                record.class_id = resolved[0]
        else:
            record.unmeasured_frames += 1

        self._version += 1

    def set_lifecycle(self, record: ObjectRecord, state: LifecycleState) -> None:
        record.lifecycle = state
        self._version += 1

    def open_binding(
        self,
        record: ObjectRecord,
        *,
        track_id: TrackId,
        method: BindingMethod,
        confidence: float,
        now: Instant,
        supersedes: BindingId | None = None,
    ) -> TrackBinding:
        """Bind a track to an object, closing any previous open binding."""
        self.close_bindings(record, now=now)
        binding = TrackBinding(
            binding_id=BindingId(new_ulid(now_ms=now.ns // 1_000_000)),
            track_id=track_id,
            bound_from=now,
            confidence=Confidence.uncalibrated(confidence, ConfidenceSemantics.IDENTITY),
            method=method,
        )
        if supersedes is not None:
            record.bindings = [
                replace(b, superseded_by=binding.binding_id)
                if b.binding_id == supersedes
                else b
                for b in record.bindings
            ]
        record.bindings.append(binding)
        record.identity_confidence = confidence
        self._version += 1
        return binding

    def close_bindings(self, record: ObjectRecord, *, now: Instant) -> None:
        """Close every open binding. Retained, not deleted (V5)."""
        if not any(b.is_open for b in record.bindings):
            return
        record.bindings = [
            replace(b, bound_to=now) if b.is_open else b for b in record.bindings
        ]
        self._version += 1

    def apply_attribute(self, record: ObjectRecord, attribute: Attribute) -> None:
        """Hold an attribute value. **Holding, not producing** — see M7 vs M9."""
        record.attributes[attribute.key] = attribute
        self._version += 1

    def merge(
        self, *, source: ObjectRecord, target: ObjectRecord, now: Instant
    ) -> None:
        """Point ``source`` at ``target``. Neither record is deleted.

        Raises:
            IdentityConflictError: the merge would corrupt history — an object
                into itself, across partitions, or into an already-merged target
                (which would make the surviving id ambiguous).
        """
        if source.object_id == target.object_id:
            raise IdentityConflictError(
                "an object cannot be merged into itself",
                object_id=str(source.object_id),
            )
        if source.camera_id != target.camera_id:
            raise IdentityConflictError(
                f"cross-partition merge from '{source.camera_id}' to "
                f"'{target.camera_id}' must be a two-phase, event-driven "
                f"operation at the site layer; taking a lock across camera "
                f"partitions reintroduces exactly the global contention the "
                f"sharding model eliminates (03_MODULES M7)",
                source_camera=str(source.camera_id),
                target_camera=str(target.camera_id),
            )
        if target.merged_into is not None:
            raise IdentityConflictError(
                f"target '{target.object_id}' has itself been merged; merging into "
                f"it would leave the surviving identity ambiguous",
                object_id=str(target.object_id),
            )

        self.close_bindings(source, now=now)
        source.merged_into = target.object_id
        source.lifecycle = LifecycleState.MERGED_INTO

        # The survivor inherits evidence, not identity: its own id and first_seen
        # stand, but the merged object's history becomes part of what it rests on.
        target.lineage = (*target.lineage, source.object_id)
        target.first_seen = Instant(min(target.first_seen.ns, source.first_seen.ns))
        target.observation_count += source.observation_count
        for class_id, weight in source.distribution.weights.items():
            target.distribution.weights[class_id] = (
                target.distribution.weights.get(class_id, 0.0) + weight
            )
        target.distribution.observations += source.distribution.observations
        resolved = target.distribution.best()
        if resolved is not None:
            target.class_id = resolved[0]
        self._version += 1

    def adopt(self, record: ObjectRecord) -> None:
        """Insert a record built elsewhere — split results and reloads.

        Raises:
            RegistryCapacityError: the population is capped.
        """
        if record.object_id in self._records:
            raise IdentityConflictError(
                f"object '{record.object_id}' already exists in this partition",
                object_id=str(record.object_id),
            )
        if len(self._records) >= self._policy.max_objects_per_camera:
            raise RegistryCapacityError(
                f"partition '{self._camera_id}' is at capacity",
                camera_id=str(self._camera_id),
                capacity=self._policy.max_objects_per_camera,
            )
        self._records[record.object_id] = record
        self._version += 1

    def next_object_id(self, now: Instant) -> ObjectId:
        """Mint an id without creating a record — used by ``split``."""
        self._sequence += 1
        return ObjectId(new_ulid(now_ms=now.ns // 1_000_000))

    def evict(self, object_id: ObjectId) -> ObjectRecord | None:
        """Remove a record entirely.

        Reserved for expiry past the retention horizon and for shedding
        provisional objects under pressure. Never used to resolve an identity
        error — that is ``merge``, which preserves history.
        """
        record = self._records.pop(object_id, None)
        if record is not None:
            self._version += 1
        return record

    def shed_candidates(self) -> tuple[ObjectRecord, ...]:
        """Provisional objects, oldest first — the only shedding order allowed.

        A confirmed object has been asserted to consumers; withdrawing that to
        save memory would make the platform's claims a function of its load.
        """
        return tuple(
            sorted(
                (r for r in self._records.values() if r.lifecycle is LifecycleState.PROVISIONAL),
                key=lambda r: (r.first_seen.ns, r.object_id),
            )
        )

    @property
    def at_capacity(self) -> bool:
        return len(self._records) >= self._policy.max_objects_per_camera

    # --- projection ----------------------------------------------------------- #

    def _project(self, record: ObjectRecord) -> VisualObject:
        """Freeze a record into the published immutable object."""
        return VisualObject(
            object_id=record.object_id,
            tenant_id=record.tenant_id,
            site_id=record.site_id,
            camera_id=record.camera_id,
            class_id=record.class_id,
            confidence=Confidence.uncalibrated(
                max(0.0, min(1.0, record.identity_confidence)),
                ConfidenceSemantics.IDENTITY,
            ),
            lifecycle=record.lifecycle,
            class_history=tuple(record.class_history),
            track_bindings=tuple(record.bindings),
            current_spatial=record.spatial,
            spatial_history=tuple(record.spatial_history),
            attributes=dict(record.attributes),
            first_seen=record.first_seen,
            last_seen=record.last_seen,
            last_confirmed=record.last_confirmed,
            observation_count=record.observation_count,
            provenance=self._provenance,
            merged_into=record.merged_into,
            lineage=record.lineage,
        )

    def stats(self) -> PartitionStats:
        by_state: dict[LifecycleState, int] = {}
        for record in self._records.values():
            by_state[record.lifecycle] = by_state.get(record.lifecycle, 0) + 1
        return PartitionStats(
            camera_id=self._camera_id,
            total=len(self._records),
            capacity=self._policy.max_objects_per_camera,
            version=self._version,
            ids_minted=self._sequence,
            by_state=by_state,
        )


@dataclass(frozen=True, slots=True)
class PartitionStats:
    camera_id: CameraId
    total: int
    capacity: int
    version: int
    ids_minted: int
    by_state: dict[LifecycleState, int] = field(default_factory=dict)

    @property
    def saturation(self) -> float:
        return self.total / self.capacity if self.capacity else 0.0

    @property
    def present(self) -> int:
        return sum(
            count for state, count in self.by_state.items() if state.is_present
        )


def spatial_distance(first: SpatialInfo, second: SpatialInfo) -> float:
    """Normalized centre separation between two spatial claims.

    Pure geometry. Returns 1.0 — maximally distant — when either lacks a box,
    because "cannot compare" must not read as "identical".
    """
    if first.bbox is None or second.bbox is None:
        return 1.0
    return min(1.0, first.bbox.centre.distance_to(second.bbox.centre) / 1.4142135623730951)


def elapsed_between(earlier: Instant, later: Instant) -> Duration:
    return Duration(max(0, later.ns - earlier.ns))


def iter_open_bindings(records: Iterable[ObjectRecord]) -> tuple[TrackBinding, ...]:
    return tuple(b for r in records if (b := r.open_binding) is not None)
