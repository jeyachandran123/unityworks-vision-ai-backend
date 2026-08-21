"""Projecting observations into state — the pure half of M12.

> 07_STATE §2: *State is what the log means at this instant.*

Every function here is a **pure transformation**: ``(partition, observation) ->
partition``. No I/O, no clock, no logging, no events. That is what makes
07_STATE §9.1's strongest claim true — a projection bug is fixed by *"fix,
rebuild into a shadow projection, atomic swap"* with **no data loss** — because
replaying the same log through corrected code produces the corrected state and
nothing else was ever authoritative.

**Structural sharing, not copying.** §5.1: a snapshot is *"a pointer to an
immutable root"*, O(1) to take, and *"memory cost is proportional to change since
the snapshot, not to state size, because unchanged subtrees are shared."* Every
update here replaces only the objects on the path that changed; the rest of the
mapping is the same mapping. That is why heavy query load cannot slow perception.

**Every ring is bounded at the write.** §6.3: the bound is *"a structural
property of the ring buffers rather than a tunable that might be misconfigured to
infinity."* A ring trimmed on a schedule is unbounded between sweeps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..core.errors import ProjectionError
from ..core.model.ids import (
    AttributeKey,
    ClassId,
    LogPosition,
    ObjectId,
    PartitionVersion,
    RegionId,
)
from ..core.model.observation import (
    Observation,
    ObservationType,
)
from ..core.model.timebase import Duration, Instant
from ..core.model.vision_state import (
    AttributeState,
    CameraPartition,
    CameraStatus,
    IdentitySummary,
    ObjectState,
    ObservabilityState,
    RegionState,
    bounded_ring,
)


@dataclass(frozen=True, slots=True)
class ProjectionBounds:
    """The ring bounds of 07_STATE §6.3, injected rather than assumed.

    Injected so a deployment's steady-state memory is *calculable before
    deployment* from its configuration — which is what makes the capacity
    planning in 13_DEPLOYMENT meaningful.
    """

    trajectory_points: int = 64
    attribute_history: int = 8
    class_history: int = 16
    max_objects: int = 512

    def __post_init__(self) -> None:
        for name in (
            "trajectory_points",
            "attribute_history",
            "class_history",
            "max_objects",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class ProjectionOutcome:
    """What projecting one observation did.

    ``changed_objects`` and ``changed_regions`` feed the delta a subscriber
    receives, so a subscriber learns *what* moved without diffing two snapshots.
    """

    partition: CameraPartition
    changed_objects: tuple[ObjectId, ...] = ()
    changed_regions: tuple[RegionId, ...] = ()
    coverage_changed: bool = False
    evicted: int = 0


def project(
    partition: CameraPartition,
    observation: Observation,
    *,
    bounds: ProjectionBounds,
    position: LogPosition,
) -> ProjectionOutcome:
    """Fold one observation into a partition. **Pure.**

    Raises:
        ProjectionError: this observation cannot be absorbed. §M12's response is
            to *"quarantine that observation, continue the projection, alarm"* —
            so the caller catches this, records it, and moves on. One bad record
            must not stop the world.
    """
    if observation.camera_id != partition.camera_id:
        raise ProjectionError(
            f"observation for camera '{observation.camera_id}' cannot be "
            f"projected into partition '{partition.camera_id}'; the camera is "
            f"the partition (07_STATE section 4.1)",
            observation_id=str(observation.observation_id),
        )

    kind = observation.observation_type

    if kind is ObservationType.COVERAGE:
        return _project_coverage(partition, observation, position)

    if observation.object_id is None:
        raise ProjectionError(
            f"a '{kind.value}' observation reached the projection with no object",
            observation_id=str(observation.observation_id),
        )

    previous = partition.objects.get(observation.object_id)
    updated = _apply(previous, observation, bounds)
    objects = dict(partition.objects)
    objects[observation.object_id] = updated

    evicted = _shed(objects, bounds.max_objects, keep=observation.object_id)
    regions = _project_regions(partition, updated, previous)

    return ProjectionOutcome(
        partition=replace(
            partition,
            objects=objects,
            regions=regions,
            status=_advance_status(partition, observation),
            log_position=position,
            version=PartitionVersion(partition.version + 1),
        ),
        changed_objects=(observation.object_id,),
        changed_regions=tuple(sorted(set(regions) ^ set(partition.regions)))
        or _touched_regions(updated, previous),
        evicted=evicted,
    )


# --- object projection ------------------------------------------------------- #


def _apply(
    previous: ObjectState | None,
    observation: Observation,
    bounds: ProjectionBounds,
) -> ObjectState:
    """Fold an observation into one object's state.

    Every branch produces a **new** ``ObjectState``. Nothing is mutated, which is
    what lets an unchanged object be shared between snapshots rather than copied
    into each.
    """
    if previous is None:
        return _create(observation, bounds)

    base = replace(
        previous,
        last_seen=_later(previous.last_seen, observation.t_capture),
        observation_count=previous.observation_count + 1,
        last_observation=observation.observation_id,
        provenance_summary=_merge_provenance(previous, observation),
    )

    if observation.is_measured:
        base = replace(
            base, last_confirmed=_later(previous.last_confirmed, observation.t_capture)
        )

    kind = observation.observation_type

    if kind in (ObservationType.PRESENCE, ObservationType.SPATIAL):
        base = _apply_spatial(base, observation, bounds)
    if kind is ObservationType.ATTRIBUTE:
        base = _apply_attributes(base, observation, bounds)
    if kind is ObservationType.LIFECYCLE and observation.lifecycle_transition:
        base = replace(base, lifecycle=observation.lifecycle_transition.current)
    if kind is ObservationType.IDENTITY and observation.identity:
        base = replace(
            base,
            identity=IdentitySummary(
                binding_count=previous.identity.binding_count + 1,
                assertion_confidence=observation.confidence,
                method=observation.identity.method,
                ambiguous=observation.identity.ambiguous,
            ),
        )

    if observation.class_id is not None and observation.class_id != previous.class_id:
        base = _apply_class_change(base, previous, observation, bounds)
    elif (
        observation.confidence is not None
        and observation.class_id == previous.class_id
        and kind is ObservationType.PRESENCE
    ):
        base = replace(base, class_confidence=observation.confidence)

    if observation.lifecycle_state is not None and kind is not ObservationType.LIFECYCLE:
        base = replace(base, lifecycle=observation.lifecycle_state)

    return base


def _create(observation: Observation, bounds: ProjectionBounds) -> ObjectState:
    """First observation for an object. State begins here and nowhere else.

    07_STATE §1.1: *"Nothing is in state that was not first a published fact."*
    There is no other constructor, which is what makes derivability structural
    rather than a rule someone must remember.
    """
    from ..core.model.confidence import Confidence, ConfidenceSemantics
    from ..core.model.visual_object import LifecycleState

    confidence = observation.confidence or Confidence.uncalibrated(
        0.5, ConfidenceSemantics.IDENTITY
    )
    class_id = observation.class_id or ClassId("unknown")

    state = ObjectState(
        object_id=observation.object_id,
        class_id=class_id,
        class_confidence=confidence,
        lifecycle=observation.lifecycle_state or LifecycleState.PROVISIONAL,
        first_seen=observation.t_capture,
        last_seen=observation.t_capture,
        last_confirmed=observation.t_capture
        if observation.is_measured
        else observation.t_capture,
        spatial=observation.spatial,
        measurement_basis=observation.measurement_basis,
        class_history=((class_id, observation.t_capture, confidence),),
        observation_count=1,
        last_observation=observation.observation_id,
        provenance_summary=_merge_provenance(None, observation),
    )
    if observation.observation_type is ObservationType.ATTRIBUTE:
        state = _apply_attributes(state, observation, bounds)
    return state


def _apply_spatial(
    state: ObjectState, observation: Observation, bounds: ProjectionBounds
) -> ObjectState:
    """Update position and extend the bounded trajectory ring.

    ``measurement_basis`` travels **into the ring**, not just onto the object: a
    consumer reading a trajectory must be able to tell which points were seen and
    which were extrapolated, or a coasted path looks like an observed one.
    """
    if observation.spatial is None:
        return state

    point = (
        observation.spatial.bbox.bottom_centre
        if observation.spatial.bbox is not None
        else observation.spatial.ground_point
    )
    trajectory = state.trajectory
    if point is not None:
        trajectory = bounded_ring(
            state.trajectory,
            (observation.t_capture, point, observation.measurement_basis),
            limit=bounds.trajectory_points,
        )

    return replace(
        state,
        spatial=observation.spatial,
        measurement_basis=observation.measurement_basis,
        trajectory=trajectory,
    )


def _apply_attributes(
    state: ObjectState, observation: Observation, bounds: ProjectionBounds
) -> ObjectState:
    """Fold attribute values in, carrying displaced values into ``previous``.

    A revision never discards the prior value: 07_STATE §3.1 keeps a short
    attribute history so a consumer can see that ``posture`` flipped twice in
    four seconds, which is a different situation from one stable reading.
    """
    if not observation.attributes:
        return state

    attributes: dict[AttributeKey, AttributeState] = dict(state.attributes)
    for attribute in observation.attributes:
        existing = attributes.get(attribute.key)
        if existing is None:
            attributes[attribute.key] = AttributeState(
                key=attribute.key,
                value=attribute.value,
                confidence=attribute.confidence,
                observed_at=attribute.observed_at,
                valid_until=attribute.valid_until,
                evidence_ref=observation.evidence_ref,
                producer=attribute.producer,
            )
            continue
        if attribute.observed_at.ns < existing.observed_at.ns:
            # An out-of-order arrival must not overwrite a newer value. The log
            # is ordered per partition, but a rebuild or a late correction can
            # deliver an older observation, and taking it would move state
            # backwards in time.
            continue
        attributes[attribute.key] = AttributeState(
            key=attribute.key,
            value=attribute.value,
            confidence=attribute.confidence,
            observed_at=attribute.observed_at,
            valid_until=attribute.valid_until,
            evidence_ref=observation.evidence_ref,
            producer=attribute.producer,
            previous=bounded_ring(
                existing.previous,
                (existing.value, existing.observed_at),
                limit=bounds.attribute_history,
            ),
        )
    return replace(state, attributes=attributes)


def _apply_class_change(
    base: ObjectState,
    previous: ObjectState,
    observation: Observation,
    bounds: ProjectionBounds,
) -> ObjectState:
    """A reclassification. Both classes survive in ``class_history``.

    07_STATE §2.2's worked example: an object reclassified from ``person`` to
    ``object.mannequin`` keeps both, *"the state reflects the current best
    understanding; the log retains how that understanding was reached."*
    """
    confidence = observation.confidence or previous.class_confidence
    return replace(
        base,
        class_id=observation.class_id,
        class_confidence=confidence,
        class_history=bounded_ring(
            previous.class_history,
            (observation.class_id, observation.t_capture, confidence),
            limit=bounds.class_history,
        ),
    )


def _merge_provenance(
    previous: ObjectState | None, observation: Observation
) -> Mapping[str, str]:
    """Track which producer last shaped this object, per role.

    A *summary*, not a history: 07_STATE §3.1 wants a consumer to see which
    detector and which understander are behind the current state without walking
    the log, and the log is where the full record lives.
    """
    summary = dict(previous.provenance_summary) if previous else {}
    role = {
        ObservationType.PRESENCE: "last_detector",
        ObservationType.SPATIAL: "last_tracker",
        ObservationType.ATTRIBUTE: "last_understander",
    }.get(observation.observation_type)
    if role and observation.provenance.model_id:
        summary[role] = str(observation.provenance.model_id)
    return summary


def _shed(
    objects: dict[ObjectId, ObjectState], capacity: int, *, keep: ObjectId
) -> int:
    """Cap the population, shedding the least recently seen.

    07_STATE §6.3: *"Objects per camera partition — capped; `provisional` objects
    shed first under pressure."* Provisional first because shedding a confirmed
    object would withdraw an assertion the platform made, and an assertion that
    depends on memory pressure is not an assertion.
    """
    if len(objects) <= capacity:
        return 0
    evicted = 0
    from ..core.model.visual_object import LifecycleState

    def rank(object_id: ObjectId) -> tuple[int, int]:
        state = objects[object_id]
        provisional = 0 if state.lifecycle is LifecycleState.PROVISIONAL else 1
        return (provisional, state.last_seen.ns)

    while len(objects) > capacity:
        doomed = min((o for o in objects if o != keep), key=rank, default=None)
        if doomed is None:
            break
        del objects[doomed]
        evicted += 1
    return evicted


# --- region projection ------------------------------------------------------- #


def _project_regions(
    partition: CameraPartition,
    updated: ObjectState,
    previous: ObjectState | None,
) -> Mapping[RegionId, RegionState]:
    """Recompute occupancy for regions this object entered or left.

    07_STATE §3.3: occupancy is *"pure counting, no interpretation"*. There is no
    threshold, no `is_crowded`, no capacity — each of those requires a definition
    only a consumer possesses.
    """
    touched = _touched_regions(updated, previous)
    if not touched:
        return partition.regions

    regions = dict(partition.regions)
    for region_id in touched:
        members = [
            state
            for state in _with(partition.objects, updated).values()
            if region_id in state.regions and state.is_present
        ]
        occupancy: dict[ClassId, int] = {}
        dwells: list[int] = []
        for state in members:
            occupancy[state.class_id] = occupancy.get(state.class_id, 0) + 1
            membership = state.regions[region_id]
            dwells.append(max(0, state.last_seen.ns - membership.entered_at.ns))

        existing = partition.regions.get(region_id)
        regions[region_id] = RegionState(
            region_id=region_id,
            geometry_version=(
                existing.geometry_version
                if existing
                else next(
                    (s.regions[region_id].geometry_version for s in members), "1.0.0"
                )
            ),
            occupancy=occupancy,
            present_objects=tuple(sorted(s.object_id for s in members)),
            dwell_current_max=Duration(max(dwells) if dwells else 0),
            dwell_current_mean=Duration(sum(dwells) // len(dwells) if dwells else 0),
            last_transition=updated.last_seen,
        )
    return regions


def _touched_regions(
    updated: ObjectState, previous: ObjectState | None
) -> tuple[RegionId, ...]:
    before = set(previous.regions) if previous else set()
    after = set(updated.regions)
    return tuple(sorted(before ^ after))


def _with(
    objects: Mapping[ObjectId, ObjectState], updated: ObjectState
) -> Mapping[ObjectId, ObjectState]:
    merged = dict(objects)
    merged[updated.object_id] = updated
    return merged


# --- coverage projection ----------------------------------------------------- #


def _project_coverage(
    partition: CameraPartition, observation: Observation, position: LogPosition
) -> ProjectionOutcome:
    """Fold a coverage observation into observability state.

    07_STATE §7.3: coverage is *"both live state and historical observations"*.
    M11 emits the observation; this projects it into the live view. Same
    producer/projector split as every other type, which is what keeps
    *"observation is the only write path"* true for coverage too.
    """
    window = observation.coverage
    if window is None:
        raise ProjectionError(
            "a coverage observation reached the projection with no window",
            observation_id=str(observation.observation_id),
        )
    return ProjectionOutcome(
        partition=replace(
            partition,
            observability=ObservabilityState(
                camera_id=partition.camera_id,
                status=window.status,
                since=window.since,
                reason=window.reason,
                effective_rate=window.effective_rate,
                regions_affected=tuple(RegionId(r) for r in window.regions_affected),
                capability_gaps=window.capability_gaps,
            ),
            log_position=position,
            version=PartitionVersion(partition.version + 1),
        ),
        coverage_changed=True,
    )


def _advance_status(
    partition: CameraPartition, observation: Observation
) -> CameraStatus:
    existing = partition.status or CameraStatus(camera_id=partition.camera_id)
    return replace(
        existing,
        stream_epoch=observation.frame_ref.stream_epoch,
        last_observation_at=observation.t_capture,
    )


def _later(a: Instant, b: Instant) -> Instant:
    """Monotonic time advance.

    Never moves backwards: a late-arriving observation from an earlier instant
    must not make an object look less recently seen than it is.
    """
    return a if a.ns >= b.ns else b
