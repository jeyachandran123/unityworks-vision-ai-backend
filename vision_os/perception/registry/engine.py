"""M7 Object Registry — the sole authority over object identity.

> **Single responsibility:** *Decide what is the same thing over time, and be the
> only module allowed to decide it.*

The public API is 03_MODULES section M7's, implemented verbatim::

    ingest(camera_id, track_update)  -> RegistryUpdate
    get(object_id)                   -> VisualObject !NotFound
    active(scope)                    -> VisualObject[]
    bind(track_id, object_id, ...)   -> BindingId
    merge(source, target, evidence)  -> ObjectId
    split(object_id, at, evidence)   -> (ObjectId, ObjectId)
    apply_attribute(object_id, attr) -> void
    expire_stale(now)                -> ObjectId[]

``ingest`` **never raises**: a registry failure may not stop tracking, which may
not stop detection, which may not stop acquisition (V9). Everything else raises
on misuse, because those are direct API calls where a caller can and should
handle the error.

**What this module does not do**, and why each absence is load-bearing:

*It produces no attributes.* Responsibility 6 says "hold current attribute values
as they are produced" — holding is storage, producing is inference, and inference
is M9 at L4. Fusing them makes cost proportional to frame rate and both
components untestable (01_LAYERED section 1.2).

*It writes no Vision State.* Vision State is a projection of the immutable
observation log, built by M13 at L6 from what M11 published. M7's durable objects
*feed* that projection; they are not it.

*It builds no observations.* Schema and ceiling enforcement is M11's single choke
point. A second producer would be a second, unenforced one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ...core.errors import (
    IdentityConflictError,
    ObjectNotFoundError,
    RegistryCapacityError,
)
from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import (
    BindingId,
    CameraId,
    FrameRef,
    ObjectId,
    SiteId,
    TenantId,
    TrackId,
)
from ...core.model.provenance import Provenance
from ...core.model.region import Region
from ...core.model.timebase import Duration, Instant
from ...core.model.track import Track, TrackUpdate
from ...core.model.visual_object import (
    Attribute,
    BindingMethod,
    IdentityAssertion,
    LifecycleState,
    RevisionReason,
    VisualObject,
)
from ...core.ports.clock import Clock
from ...core.ports.registry import PartitionSnapshot
from ...kernel.config.schema import RegistrySection
from ...kernel.events import (
    EventBus,
    IdentityAsserted,
    IdentityRevised,
    ObjectCreated,
    ObjectLifecycleChanged,
    ObjectPopulationCapped,
)
from ...kernel.events import RegionTransition as RegionTransitionEvent
from ...kernel.metrics import MetricName, MetricsEngine
from .attributes import AttributeRegistry
from .binding import BindingPolicy, TrackBinder
from .lifecycle import (
    LifecyclePolicy,
    LifecycleTransition,
    ObjectLifecycleMachine,
)
from .partition import ObjectRecord, PartitionStats, RegistryPartition
from .regions import RegionTracker, RegionTransition

REGISTRY_ENGINE_ID = "object_registry"


@dataclass(frozen=True, slots=True)
class RegistryUpdate:
    """What one ``TrackUpdate`` did to a camera's objects.

    ``failed`` distinguishes "the registry could not run" from "the registry ran
    and nothing changed". Conflating them would make a broken registry look like
    an empty scene (invariant V8).
    """

    camera_id: CameraId
    frame_ref: FrameRef
    """The frame this update came from, **typed**.

    Widened from ``str`` in Flow 5. The Crop Manager attaches to this update as
    its declared extension point and must lease pixels for exactly this frame;
    reconstructing a ``FrameRef`` by parsing its own ``__str__`` would be a
    silent correctness hazard the type system is right there to prevent. Anything
    wanting the old text calls ``str(update.frame_ref)``.
    """

    objects: tuple[VisualObject, ...] = ()
    created: tuple[ObjectId, ...] = ()
    lifecycle_changes: tuple[tuple[ObjectId, LifecycleState, LifecycleState], ...] = ()
    assertions: tuple[IdentityAssertion, ...] = ()
    region_transitions: tuple[RegionTransition, ...] = ()
    expired: tuple[ObjectId, ...] = ()
    shed: tuple[ObjectId, ...] = ()
    failed: bool = False
    reason: str = ""
    latency_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.objects)

    @property
    def present(self) -> tuple[VisualObject, ...]:
        return tuple(o for o in self.objects if o.is_present)

    @property
    def ambiguous_assertions(self) -> tuple[IdentityAssertion, ...]:
        return tuple(a for a in self.assertions if a.is_ambiguous)


class ObjectRegistry:
    """Converts tracks into canonical, durable visual objects."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        config: RegistrySection,
        tenant_id: TenantId,
        site_id: SiteId,
        provenance: Provenance,
        lifecycle: LifecyclePolicy | None = None,
        binding: BindingPolicy | None = None,
        attributes: AttributeRegistry | None = None,
        resolver=None,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._config = config
        self._tenant_id = tenant_id
        self._site_id = site_id
        self._provenance = provenance

        self._lifecycle_policy = lifecycle or LifecyclePolicy()
        self._machine = ObjectLifecycleMachine(self._lifecycle_policy)
        self._binder = TrackBinder(binding or BindingPolicy())
        self._attribute_registry = attributes or AttributeRegistry()
        # P11. ``None`` in Phase 1 — no implementations ship (15_ROADMAP section
        # 3). The engine consults it when bound and works entirely without it.
        self._resolver = resolver

        self._partitions: dict[CameraId, RegistryPartition] = {}
        self._regions: dict[CameraId, RegionTracker] = {}
        self._epochs: dict[CameraId, int] = {}
        self._camera_time: dict[CameraId, Instant] = {}
        self._frames = 0
        self._failures = 0
        self._degraded_reason = ""

    # --- the public API (03_MODULES section M7) -------------------------------- #

    def ingest(self, camera_id: CameraId, update: TrackUpdate) -> RegistryUpdate:
        """Fold one frame's tracks into the object population. **Never raises.**"""
        started = self._clock.monotonic().ns
        try:
            return self._ingest(camera_id, update, started)
        except RegistryCapacityError as exc:
            self._metrics.counter(
                MetricName.REGISTRY_CAPACITY_REFUSALS, camera_id=str(camera_id)
            ).increment()
            return self._failure(camera_id, update, f"capacity: {exc}", started)
        except Exception as exc:  # noqa: BLE001 - the engine is a firewall
            self._failures += 1
            self._degraded_reason = f"{type(exc).__name__}: {exc}"
            self._metrics.counter(
                MetricName.REGISTRY_FAILURES, camera_id=str(camera_id), reason="ingest"
            ).increment()
            return self._failure(camera_id, update, self._degraded_reason, started)

    def get(self, object_id: ObjectId) -> VisualObject:
        """Fetch one object.

        Raises:
            ObjectNotFoundError: unknown id. A *merged* object is found and
                reports where it went — history stays resolvable (V5).
        """
        for partition in self._partitions.values():
            found = partition.find(object_id)
            if found is not None:
                return found
        raise ObjectNotFoundError(
            f"object '{object_id}' is not known to this registry",
            object_id=str(object_id),
        )

    def resolve(self, object_id: ObjectId) -> VisualObject | None:
        """Follow ``merged_into`` to the surviving object."""
        for partition in self._partitions.values():
            if object_id in partition:
                return partition.resolve(object_id)
        return None

    def active(self, scope: CameraId | None = None) -> tuple[VisualObject, ...]:
        """Objects believed present, for one camera or the whole site."""
        if scope is not None:
            partition = self._partitions.get(scope)
            return partition.active() if partition else ()
        return tuple(
            obj
            for camera_id in sorted(self._partitions)
            for obj in self._partitions[camera_id].active()
        )

    def objects(self, scope: CameraId | None = None) -> tuple[VisualObject, ...]:
        """Every object including terminal ones, for merge resolution."""
        if scope is not None:
            partition = self._partitions.get(scope)
            return partition.objects() if partition else ()
        return tuple(
            obj
            for camera_id in sorted(self._partitions)
            for obj in self._partitions[camera_id].objects()
        )

    def bind(
        self,
        track_id: TrackId,
        object_id: ObjectId,
        *,
        method: BindingMethod = BindingMethod.MANUAL,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> BindingId:
        """Assert that a track belongs to an object.

        Raises:
            ObjectNotFoundError: the object does not exist.
            IdentityConflictError: the object is terminal.
        """
        partition, record = self._locate(object_id)
        if record.lifecycle.is_terminal:
            raise IdentityConflictError(
                f"object '{object_id}' is {record.lifecycle.value} and cannot accept "
                f"a binding; terminal states are final (V5)",
                object_id=str(object_id),
            )
        now = self._clock.now()
        binding = partition.open_binding(
            record, track_id=track_id, method=method, confidence=confidence, now=now
        )
        self._publish_assertion(
            partition.camera_id,
            IdentityAssertion(
                binding_id=binding.binding_id,
                object_id=object_id,
                track_id=track_id,
                asserted_at=now,
                confidence=Confidence.uncalibrated(
                    confidence, ConfidenceSemantics.IDENTITY
                ),
                method=method,
                evidence=evidence,
            ),
        )
        return binding.binding_id

    def merge(
        self, source_object: ObjectId, target_object: ObjectId, *, evidence: str = ""
    ) -> ObjectId:
        """Merge ``source`` into ``target``. Returns the surviving id.

        **Preserves history** (V5). The source is not deleted: it enters
        ``merged_into`` and observations referencing it stay resolvable.

        Raises:
            ObjectNotFoundError / IdentityConflictError.
        """
        source_partition, source = self._locate(source_object)
        target_partition, target = self._locate(target_object)

        if source_partition.camera_id != target_partition.camera_id:
            raise IdentityConflictError(
                f"cross-partition merge from '{source_partition.camera_id}' to "
                f"'{target_partition.camera_id}' must be a two-phase, "
                f"event-driven operation at the site layer, never a synchronous "
                f"cross-partition write (03_MODULES M7 Thread Safety)",
                source_camera=str(source_partition.camera_id),
                target_camera=str(target_partition.camera_id),
            )

        now = self._clock.now()
        transition = self._machine.on_merged(state=source.lifecycle)
        source_partition.merge(source=source, target=target, now=now)

        self._metrics.counter(
            MetricName.OBJECTS_MERGED, camera_id=str(source_partition.camera_id)
        ).increment()
        # A merge is an identity *revision*, not a deletion: the source survives
        # in MERGED_INTO so observations referencing it stay resolvable (V5).
        self._bus.publish(
            IdentityRevised(
                occurred_at=now,
                camera_id=source_partition.camera_id,
                object_id=str(source_object),
                successor_id=str(target_object),
                reason=RevisionReason.MERGE.value,
                evidence=evidence,
            )
        )
        self._publish_lifecycle(
            source_partition.camera_id,
            source_object,
            transition.previous,
            transition.current,
            transition.trigger.value,
        )
        return target_object

    def split(
        self, object_id: ObjectId, *, at: Instant, evidence: str = ""
    ) -> tuple[ObjectId, ObjectId]:
        """Split one object into two at a point in time.

        The original keeps its id and everything before ``at``; a new object
        takes everything after. History is preserved on both sides — the new
        object records the original in its ``lineage``.

        Raises:
            IdentityConflictError: ``at`` lies outside the object's lifetime, so
                one side of the split would be empty and the operation would be a
                rename pretending to be a correction.
        """
        partition, record = self._locate(object_id)
        if not (record.first_seen.ns < at.ns < record.last_seen.ns):
            raise IdentityConflictError(
                f"split point lies outside object '{object_id}' lifetime "
                f"[{record.first_seen.ns}, {record.last_seen.ns}]; one side would "
                f"be empty",
                object_id=str(object_id),
            )

        now = self._clock.now()
        successor_id = partition.next_object_id(now)
        successor = ObjectRecord(
            object_id=successor_id,
            camera_id=record.camera_id,
            tenant_id=record.tenant_id,
            site_id=record.site_id,
            lifecycle=record.lifecycle,
            class_id=record.class_id,
            distribution=record.distribution,
            spatial=record.spatial,
            first_seen=at,
            last_seen=record.last_seen,
            last_confirmed=record.last_confirmed,
            identity_confidence=record.identity_confidence * 0.5,
            observation_count=max(1, record.observation_count // 2),
            lineage=(*record.lineage, object_id),
        )
        successor.spatial_history.extend(
            (t, s) for t, s in record.spatial_history if t.ns >= at.ns
        )
        successor.class_history.extend(
            c for c in record.class_history if c.observed_at.ns >= at.ns
        )
        successor.bindings = [b for b in record.bindings if b.bound_from.ns >= at.ns]

        # The original keeps only what precedes the split point.
        record.spatial_history = type(record.spatial_history)(
            ((t, s) for t, s in record.spatial_history if t.ns < at.ns),
            maxlen=record.spatial_history.maxlen,
        )
        record.class_history = type(record.class_history)(
            (c for c in record.class_history if c.observed_at.ns < at.ns),
            maxlen=record.class_history.maxlen,
        )
        record.bindings = [b for b in record.bindings if b.bound_from.ns < at.ns]
        record.last_seen = at
        record.last_confirmed = Instant(min(record.last_confirmed.ns, at.ns))
        record.observation_count = max(1, record.observation_count // 2)
        partition.adopt(successor)

        self._metrics.counter(
            MetricName.OBJECTS_SPLIT, camera_id=str(partition.camera_id)
        ).increment()
        self._bus.publish(
            IdentityRevised(
                occurred_at=now,
                camera_id=partition.camera_id,
                object_id=str(object_id),
                successor_id=str(successor_id),
                reason=RevisionReason.SPLIT.value,
                evidence=evidence,
            )
        )
        return object_id, successor_id

    def apply_attribute(self, object_id: ObjectId, attribute: Attribute) -> None:
        """Hold an attribute value against an object.

        **Holds; does not produce.** Extraction is M9 Understanding at L4. This
        method exists in M7's documented API and will be called from Flow 5.

        Raises:
            AttributeRejectedError: the attribute is unregistered or fails the
                neutrality gate (02_VOM section 9.1).
            ObjectNotFoundError: unknown object.
        """
        partition, record = self._locate(object_id)
        self._attribute_registry.require(attribute.key)
        if not self._attribute_registry.applies(attribute.key, record.class_id):
            from ...core.errors import AttributeRejectedError

            raise AttributeRejectedError(
                f"attribute '{attribute.key}' does not apply to class "
                f"'{record.class_id}'",
                attribute_key=str(attribute.key),
                class_id=str(record.class_id),
            )
        partition.apply_attribute(record, attribute)
        self._metrics.counter(
            MetricName.ATTRIBUTES_APPLIED, camera_id=str(partition.camera_id)
        ).increment()

    def expire_stale(self, now: Instant | None = None) -> tuple[ObjectId, ...]:
        """Advance horizons and remove objects past retention.

        Returns the ids removed. Called on a schedule rather than per frame,
        because a camera that goes quiet must still see its objects age.
        """
        moment = now or self._clock.now()
        removed: list[ObjectId] = []
        for camera_id in sorted(self._partitions):
            partition = self._partitions[camera_id]
            for record in partition.records():
                if record.lifecycle.is_terminal:
                    if record.lifecycle is LifecycleState.EXPIRED:
                        partition.evict(record.object_id)
                        removed.append(record.object_id)
                    continue
                transition = self._drive_horizons(record, moment)
                if transition is not None:
                    # An object the sweep moves out of a measurable state has no
                    # live track; closing the binding is what lets it be a
                    # re-entry candidate later.
                    if not transition.current.is_measurable:
                        partition.close_bindings(record, now=moment)
                    partition.set_lifecycle(record, transition.current)
                    self._publish_lifecycle(
                        camera_id,
                        record.object_id,
                        transition.previous,
                        transition.current,
                        transition.trigger.value,
                    )
                    if transition.current is LifecycleState.EXPIRED:
                        partition.evict(record.object_id)
                        tracker = self._regions.get(camera_id)
                        if tracker is not None:
                            tracker.forget(record.object_id, at=moment)
                        removed.append(record.object_id)
        return tuple(removed)

    # --- configuration --------------------------------------------------------- #

    def set_regions(self, camera_id: CameraId, regions: tuple[Region, ...]) -> None:
        """Adopt region geometry for a camera.

        Section M7: existing dwell accumulations are closed out against the old
        version and new ones opened against the new, so time spent in the old
        shape is never attributed to the new one.
        """
        tracker = self._region_tracker(camera_id)
        closed = tracker.set_regions(regions, now=self._clock.now())
        for transition in closed:
            self._publish_region(camera_id, transition)

    def restore(self, snapshot: PartitionSnapshot) -> int:
        """Reload durable objects after a restart. Returns how many were restored.

        ``07_STATE`` section 9.3: *"object identity survives, tracks do not"*.
        Every restored object is therefore **unbound** — its bindings are closed,
        because the tracks that produced them died with the process. The first
        track to match one re-binds through ``EPOCH_REBIND`` with explicitly
        reduced confidence, which is what makes the discontinuity visible rather
        than pretending continuity was observed.

        Terminal objects are restored too: an observation referencing a merged id
        must stay resolvable across a restart, or V5 holds only until the next
        deployment.
        """
        partition = self._partition(snapshot.camera_id)
        restored = 0
        now = self._clock.now()

        for obj in snapshot.objects:
            if obj.object_id in partition:
                continue
            record = _record_from(obj)
            # Bindings are closed on reload: the tracks are gone, and leaving a
            # binding open would let a new track with a recycled id inherit it.
            record.bindings = [
                b if not b.is_open else replace(b, bound_to=snapshot.taken_at)
                for b in record.bindings
            ]
            try:
                partition.adopt(record)
            except (RegistryCapacityError, IdentityConflictError):
                break
            restored += 1

        if restored:
            self._metrics.gauge(
                MetricName.OBJECTS_TOTAL, camera_id=str(snapshot.camera_id)
            ).set(float(len(partition)))
        if restored < snapshot.count:
            # Some objects did not fit. Alarm rather than absorb: a partition
            # that silently drops history on reload is data loss presented as a
            # fresh start.
            self._bus.publish(
                ObjectPopulationCapped(
                    occurred_at=now,
                    camera_id=snapshot.camera_id,
                    population=len(partition),
                    capacity=self._lifecycle_policy.max_objects_per_camera,
                    shed=0,
                    detail=(
                        f"restored {restored} of {snapshot.count} durable objects; "
                        f"the remainder exceeded the partition cap"
                    ),
                )
            )
        return restored

    @property
    def site_id(self) -> SiteId:
        return self._site_id

    @property
    def tenant_id(self) -> TenantId:
        return self._tenant_id

    @property
    def attribute_registry(self) -> AttributeRegistry:
        return self._attribute_registry

    def partition_stats(self, camera_id: CameraId) -> PartitionStats | None:
        partition = self._partitions.get(camera_id)
        return partition.stats() if partition else None

    def health(self) -> ComponentHealth:
        state = HealthState.HEALTHY
        detail = "registry"
        if self._degraded_reason:
            state = HealthState.DEGRADED
            detail = self._degraded_reason
        return ComponentHealth(
            component_id=REGISTRY_ENGINE_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "frames": float(self._frames),
                "failures": float(self._failures),
                "objects": float(sum(len(p) for p in self._partitions.values())),
            },
        )

    # --- ingestion internals ---------------------------------------------------- #

    def _ingest(
        self, camera_id: CameraId, update: TrackUpdate, started: int
    ) -> RegistryUpdate:
        # **Capture time, not processing time.** 02_VOM section 5.2 rule 5: a
        # dwell of 45 s means the object was present for 45 s *in the world*,
        # regardless of whether the platform was keeping up. Every object
        # timestamp here becomes a duration a consumer reasons about, so all of
        # them derive from the tracks' own times; the injected clock is used only
        # for scheduling and latency measurement.
        #
        # A frame with no tracks carries no capture time, so it does not advance
        # the camera's clock. Aging a quiet camera is ``expire_stale``'s job —
        # it takes a real ``now`` precisely because ingestion cannot invent one.
        now = self._advance_time(camera_id, update)
        partition = self._partition(camera_id)
        regions = self._region_tracker(camera_id)

        crossing_epoch = self._epochs.get(camera_id, update.tracker_epoch) != update.tracker_epoch
        self._epochs[camera_id] = update.tracker_epoch

        created: list[ObjectId] = []
        changes: list[tuple[ObjectId, LifecycleState, LifecycleState]] = []
        assertions: list[IdentityAssertion] = []
        transitions: list[RegionTransition] = []
        shed: list[ObjectId] = []

        touched: set[ObjectId] = set()
        records = partition.records()

        for track in update.active:
            record, assertion, was_created = self._absorb(
                partition=partition,
                records=records,
                track=track,
                now=now,
                crossing_epoch=crossing_epoch,
                shed=shed,
            )
            if record is None:
                continue
            touched.add(record.object_id)
            if was_created:
                created.append(record.object_id)
            if assertion is not None:
                assertions.append(assertion)
                self._publish_assertion(camera_id, assertion)

            measured = not track.is_predicted
            transition = self._machine.on_measured(
                state=record.lifecycle, observation_count=record.observation_count
            ) if measured else self._machine.on_unmeasured(
                state=record.lifecycle,
                since_confirmed=Duration(max(0, now.ns - record.last_confirmed.ns)),
            )
            if transition.changed:
                partition.set_lifecycle(record, transition.current)
                changes.append(
                    (record.object_id, transition.previous, transition.current)
                )
                self._publish_lifecycle(
                    camera_id,
                    record.object_id,
                    transition.previous,
                    transition.current,
                    transition.trigger.value,
                )

            if record.spatial.bbox is not None and record.lifecycle.is_present:
                transitions.extend(
                    regions.update(record.object_id, record.spatial.bbox, at=track.last_seen)
                )
            records = partition.records()

        # Objects with no track this frame age toward departure.
        for record in partition.records():
            if record.object_id in touched or record.lifecycle.is_terminal:
                continue

            # The track that owned this object is gone from the update, so its
            # binding closes. Leaving it open would make the object permanently
            # unavailable for re-entry: the binder only considers *unbound*
            # candidates, since a bound object already has a track.
            partition.close_bindings(record, now=now)

            since = Duration(max(0, now.ns - record.last_confirmed.ns))
            left = self._left_field_of_view(record)
            transition = self._machine.on_unmeasured(
                state=record.lifecycle, since_confirmed=since, left_field_of_view=left
            )
            partition.observe(
                record,
                class_id=record.class_id,
                class_confidence=0.0,
                spatial=record.spatial,
                now=now,
                measured=False,
            )
            if transition.changed:
                partition.set_lifecycle(record, transition.current)
                changes.append((record.object_id, transition.previous, transition.current))
                self._publish_lifecycle(
                    camera_id,
                    record.object_id,
                    transition.previous,
                    transition.current,
                    transition.trigger.value,
                )
                if not transition.current.is_present:
                    transitions.extend(regions.forget(record.object_id, at=now))

        for transition_event in transitions:
            self._publish_region(camera_id, transition_event)

        self._frames += 1
        latency_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._record_metrics(camera_id, partition, latency_ms)

        return RegistryUpdate(
            camera_id=camera_id,
            frame_ref=update.frame_ref,
            objects=partition.objects(),
            created=tuple(created),
            lifecycle_changes=tuple(changes),
            assertions=tuple(assertions),
            region_transitions=tuple(transitions),
            shed=tuple(shed),
            latency_ms=latency_ms,
        )

    def _absorb(
        self,
        *,
        partition: RegistryPartition,
        records: Sequence[ObjectRecord],
        track: Track,
        now: Instant,
        crossing_epoch: bool,
        shed: list[ObjectId],
    ) -> tuple[ObjectRecord | None, IdentityAssertion | None, bool]:
        """Bind one track to an object, minting one if nothing matches."""
        continuity = self._binder.bind_continuing(records, track.track_id)
        if continuity.matched is not None:
            record = partition.record_for(continuity.matched.object_id)
            if record is not None:
                partition.observe(
                    record,
                    class_id=track.class_id,
                    class_confidence=track.confidence.value,
                    spatial=track.spatial,
                    now=now,
                    measured=not track.is_predicted,
                )
                return record, None, False

        decision = self._binder.bind_reentry(
            records,
            spatial=track.spatial,
            class_id=track.class_id,
            now=now,
            crossing_epoch=crossing_epoch,
        )

        if decision.matched is not None:
            record = partition.record_for(decision.matched.object_id)
            if record is not None:
                partition.observe(
                    record,
                    class_id=track.class_id,
                    class_confidence=track.confidence.value,
                    spatial=track.spatial,
                    now=now,
                    measured=not track.is_predicted,
                )
                binding = partition.open_binding(
                    record,
                    track_id=track.track_id,
                    method=decision.matched.method,
                    confidence=decision.matched.score,
                    now=now,
                )
                return (
                    record,
                    IdentityAssertion(
                        binding_id=binding.binding_id,
                        object_id=record.object_id,
                        track_id=track.track_id,
                        asserted_at=now,
                        confidence=Confidence.uncalibrated(
                            decision.matched.score, ConfidenceSemantics.IDENTITY
                        ),
                        method=decision.matched.method,
                        evidence=decision.matched.rationale,
                        alternatives=decision.alternatives,
                    ),
                    False,
                )

        # Nothing matched, or the match was too ambiguous to assert. Section M7:
        # create a NEW object and emit a low-confidence assertion linking the
        # candidates. Never guess silently.
        record = self._mint(partition, track, now, shed)
        if record is None:
            return None, None, False

        binding = partition.open_binding(
            record,
            track_id=track.track_id,
            method=BindingMethod.FIRST_SIGHT,
            confidence=1.0 if not decision.ambiguous else decision.candidates[0].score,
            now=now,
        )
        assertion = IdentityAssertion(
            binding_id=binding.binding_id,
            object_id=record.object_id,
            track_id=track.track_id,
            asserted_at=now,
            confidence=Confidence.uncalibrated(
                decision.candidates[0].score if decision.ambiguous else 1.0,
                ConfidenceSemantics.IDENTITY,
            ),
            method=BindingMethod.FIRST_SIGHT,
            evidence=decision.reason,
            alternatives=decision.alternatives,
        )
        if decision.ambiguous:
            self._metrics.counter(
                MetricName.IDENTITY_AMBIGUITIES, camera_id=str(partition.camera_id)
            ).increment()
        return record, assertion, True

    def _mint(
        self,
        partition: RegistryPartition,
        track: Track,
        now: Instant,
        shed: list[ObjectId],
    ) -> ObjectRecord | None:
        """Create an object, shedding provisional ones if the partition is full."""
        try:
            return partition.mint(
                class_id=track.class_id,
                confidence=track.confidence.value,
                spatial=track.spatial,
                now=now,
                class_confidence=track.confidence.value,
            )
        except RegistryCapacityError:
            for candidate in partition.shed_candidates():
                transition = self._machine.on_shed(state=candidate.lifecycle)
                partition.evict(candidate.object_id)
                shed.append(candidate.object_id)
                self._metrics.counter(
                    MetricName.OBJECTS_SHED, camera_id=str(partition.camera_id)
                ).increment()
                self._publish_lifecycle(
                    partition.camera_id,
                    candidate.object_id,
                    transition.previous,
                    transition.current,
                    transition.trigger.value,
                )
                self._bus.publish(
                    ObjectPopulationCapped(
                        occurred_at=now,
                        camera_id=partition.camera_id,
                        population=len(partition),
                        capacity=self._lifecycle_policy.max_objects_per_camera,
                        shed=1,
                        detail="shed the oldest provisional object to make room",
                    )
                )
                break
            else:
                # Nothing sheddable: the population is entirely confirmed
                # objects. Refuse and alarm rather than withdraw an assertion —
                # section M7 requires the alarm, and withdrawing a confirmed
                # object would make the platform's claims depend on its memory.
                self._bus.publish(
                    ObjectPopulationCapped(
                        occurred_at=now,
                        camera_id=partition.camera_id,
                        population=len(partition),
                        capacity=self._lifecycle_policy.max_objects_per_camera,
                        shed=0,
                        detail=(
                            "no provisional objects to shed; refusing new objects "
                            "rather than withdrawing a confirmed assertion"
                        ),
                    )
                )
                return None
            try:
                return partition.mint(
                    class_id=track.class_id,
                    confidence=track.confidence.value,
                    spatial=track.spatial,
                    now=now,
                    class_confidence=track.confidence.value,
                )
            except RegistryCapacityError:
                return None

    def _left_field_of_view(self, record: ObjectRecord) -> bool:
        """Whether the object was last seen at a frame edge.

        Pure geometry, and the reason ``active -> dormant`` and ``active ->
        occluded`` are separate edges: an object that walks out of frame is a
        different claim from one that stops being measurable in place.
        """
        box = record.spatial.bbox
        if box is None:
            return False
        margin = self._config.edge_margin
        return (
            box.x1 <= margin
            or box.y1 <= margin
            or box.x2 >= 1.0 - margin
            or box.y2 >= 1.0 - margin
        )

    # --- infrastructure --------------------------------------------------------- #

    def _advance_time(self, camera_id: CameraId, update: TrackUpdate) -> Instant:
        """The capture instant this frame represents, monotonic per camera.

        **Never regresses.** An object's ``last_seen`` moving backwards would
        make every duration derived from it meaningless, and the object model
        refuses to be constructed that way — correctly, because a measurement
        that is newer than the most recent update is incoherent.
        """
        times = [t.last_updated.ns for t in update.active]
        previous = self._camera_time.get(camera_id)
        if times:
            candidate = max(times)
            current = Instant(
                candidate if previous is None else max(previous.ns, candidate)
            )
        else:
            current = previous or self._clock.now()
        self._camera_time[camera_id] = current
        return current

    def _drive_horizons(
        self, record: ObjectRecord, moment: Instant
    ) -> LifecycleTransition | None:
        """Advance an object's lifecycle to a fixed point.

        One evaluation moves one edge, but a long gap can cross several horizons
        at once — after twenty minutes an object is *expired*, not merely
        occluded. Driving to a fixed point is what makes a sweep's result depend
        on elapsed time rather than on how often the sweep happens to run.
        """
        first: LifecycleTransition | None = None
        last: LifecycleTransition | None = None
        state = record.lifecycle
        since = Duration(max(0, moment.ns - record.last_confirmed.ns))

        for _ in range(len(LifecycleState)):
            if state.is_terminal:
                break
            transition = self._machine.on_unmeasured(
                state=state, since_confirmed=since
            )
            if not transition.changed:
                break
            first = first or transition
            last = transition
            state = transition.current

        if first is None or last is None:
            return None
        return LifecycleTransition(
            previous=first.previous, current=last.current, trigger=last.trigger
        )

    def _partition(self, camera_id: CameraId) -> RegistryPartition:
        partition = self._partitions.get(camera_id)
        if partition is None:
            partition = RegistryPartition(
                camera_id,
                tenant_id=self._tenant_id,
                site_id=self._site_id,
                policy=self._lifecycle_policy,
                provenance=self._provenance,
                spatial_history=self._config.spatial_history_length,
                class_history=self._config.class_history_length,
            )
            self._partitions[camera_id] = partition
        return partition

    def _region_tracker(self, camera_id: CameraId) -> RegionTracker:
        tracker = self._regions.get(camera_id)
        if tracker is None:
            tracker = RegionTracker()
            self._regions[camera_id] = tracker
        return tracker

    def _locate(self, object_id: ObjectId) -> tuple[RegistryPartition, ObjectRecord]:
        for camera_id in sorted(self._partitions):
            partition = self._partitions[camera_id]
            record = partition.record_for(object_id)
            if record is not None:
                return partition, record
        raise ObjectNotFoundError(
            f"object '{object_id}' is not known to this registry",
            object_id=str(object_id),
        )

    def _failure(
        self, camera_id: CameraId, update: TrackUpdate, reason: str, started: int
    ) -> RegistryUpdate:
        return RegistryUpdate(
            camera_id=camera_id,
            frame_ref=update.frame_ref,
            failed=True,
            reason=reason,
            latency_ms=(self._clock.monotonic().ns - started) / 1_000_000,
        )

    def _record_metrics(
        self, camera_id: CameraId, partition: RegistryPartition, latency_ms: float
    ) -> None:
        label = str(camera_id)
        stats = partition.stats()
        self._metrics.gauge(MetricName.OBJECTS_ACTIVE, camera_id=label).set(
            float(stats.present)
        )
        self._metrics.gauge(MetricName.OBJECTS_TOTAL, camera_id=label).set(
            float(stats.total)
        )
        self._metrics.gauge(MetricName.REGISTRY_SATURATION, camera_id=label).set(
            stats.saturation
        )
        self._metrics.histogram(MetricName.REGISTRY_LATENCY_MS, camera_id=label).record(
            latency_ms
        )
        self._metrics.counter(
            MetricName.REGISTRY_FRAMES_PROCESSED, camera_id=label
        ).increment()

    def _publish_lifecycle(
        self,
        camera_id: CameraId,
        object_id: ObjectId,
        previous: LifecycleState,
        current: LifecycleState,
        trigger: str = "",
    ) -> None:
        """Every transition is observable. No state change is silent."""
        now = self._clock.now()
        if previous is LifecycleState.PROVISIONAL and current is LifecycleState.ACTIVE:
            self._metrics.counter(
                MetricName.OBJECTS_CONFIRMED, camera_id=str(camera_id)
            ).increment()
        if current is LifecycleState.EXPIRED:
            self._metrics.counter(
                MetricName.OBJECTS_EXPIRED, camera_id=str(camera_id)
            ).increment()
        self._bus.publish(
            ObjectLifecycleChanged(
                occurred_at=now,
                camera_id=camera_id,
                object_id=str(object_id),
                previous=previous.value,
                current=current.value,
                trigger=trigger,
            )
        )

    def _publish_assertion(self, camera_id: CameraId, assertion: IdentityAssertion) -> None:
        if assertion.method is BindingMethod.FIRST_SIGHT and not assertion.is_ambiguous:
            self._metrics.counter(
                MetricName.OBJECTS_CREATED, camera_id=str(camera_id)
            ).increment()
            self._bus.publish(
                ObjectCreated(
                    occurred_at=assertion.asserted_at,
                    camera_id=camera_id,
                    object_id=str(assertion.object_id),
                    track_id=str(assertion.track_id),
                )
            )
        self._bus.publish(
            IdentityAsserted(
                occurred_at=assertion.asserted_at,
                camera_id=camera_id,
                object_id=str(assertion.object_id),
                track_id=str(assertion.track_id),
                method=assertion.method.value,
                confidence=assertion.confidence.value,
                ambiguous=assertion.is_ambiguous,
                alternatives=len(assertion.alternatives),
            )
        )

    def _publish_region(self, camera_id: CameraId, transition: RegionTransition) -> None:
        self._metrics.counter(
            MetricName.REGION_TRANSITIONS,
            camera_id=str(camera_id),
            direction="entry" if transition.entered else "exit",
        ).increment()
        self._bus.publish(
            RegionTransitionEvent(
                occurred_at=transition.at,
                camera_id=camera_id,
                object_id=str(transition.object_id),
                region_id=str(transition.region_id),
                geometry_version=transition.geometry_version,
                entered=transition.entered,
                dwell_ms=transition.dwell.millis,
            )
        )

    @property
    def frames_processed(self) -> int:
        return self._frames

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def partitions(self) -> tuple[CameraId, ...]:
        return tuple(sorted(self._partitions))

    def region_tracker(self, camera_id: CameraId) -> RegionTracker | None:
        """Read access for occupancy reporting. Never mutated by a consumer."""
        return self._regions.get(camera_id)


def _record_from(obj: VisualObject) -> ObjectRecord:
    """Rebuild internal bookkeeping from a published object.

    Used only on reload. The class distribution is reconstructed from
    ``class_history`` rather than persisted separately: history is the durable
    fact, and a distribution derived from it cannot disagree with it.
    """
    from collections import deque

    from .partition import ClassDistribution

    distribution = ClassDistribution()
    for observation in obj.class_history:
        distribution.observe(observation.class_id, observation.confidence.value)

    record = ObjectRecord(
        object_id=obj.object_id,
        camera_id=obj.camera_id,
        tenant_id=obj.tenant_id,
        site_id=obj.site_id,
        lifecycle=obj.lifecycle,
        class_id=obj.class_id,
        distribution=distribution,
        spatial=obj.current_spatial,
        first_seen=obj.first_seen,
        last_seen=obj.last_seen,
        last_confirmed=obj.last_confirmed,
        identity_confidence=obj.confidence.value,
        observation_count=obj.observation_count,
        bindings=list(obj.track_bindings),
        attributes=dict(obj.attributes),
        merged_into=obj.merged_into,
        lineage=obj.lineage,
    )
    record.class_history = deque(obj.class_history, maxlen=len(obj.class_history) or 1)
    record.spatial_history = deque(
        obj.spatial_history, maxlen=len(obj.spatial_history) or 1
    )
    return record
