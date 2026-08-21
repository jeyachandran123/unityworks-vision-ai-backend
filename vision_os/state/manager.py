"""M12 Vision State Manager — the single writer of visual truth.

> **Single responsibility:** *Be the single writer of visual truth, and never
> interpret it.*

The public API is 04_MODULES §M12's, implemented verbatim::

    append(observations)              -> CommitResult !CommitFailed
    snapshot(scope)                   -> StateSnapshot
    object_state(object_id)           -> ObjectState !NotFound
    history(object_id, window)        -> Observation[]
    coverage(scope, at?)              -> CoverageMap
    subscribe(filter)                 => StateDelta
    rebuild(scope, from_log_position) -> RebuildHandle
    retention_sweep(policy)           -> SweepReport

**Append is the only write path.** 07_STATE §1.1: state is *derived*, and
*"nothing is in state that was not first a published fact."* There is no setter,
no patch, no direct object write — not because one is discouraged but because
none exists. §1.2 explains what allowing one would cost: derivability,
explainability, the Semantic Ceiling and the entire lock-free design, all at once.

**Log first, then project.** The order is the architecture. If projection ran
first and the append failed, state would hold something the log does not — and
§9.1's *"state/log divergence detected → rebuild the partition from the log; the
log is authoritative, always"* would silently produce a different world.

**A full buffer halts the partition.** 10_RELIABILITY §4.4 step 4. Dropping
observations to keep running would leave *"a permanent, undetectable hole in the
record"*, and §R1 prefers an admitted gap to a quiet lie.

**What this module never does:** interpret, aggregate for business purposes,
alert, predict, or accept a write from anything but an observation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

from ..core.errors import (
    CommitFailedError,
    LogUnavailableError,
    PartitionDegradedError,
    ProjectionError,
    StateNotFoundError,
)
from ..core.model.health import ComponentHealth, HealthState
from ..core.model.ids import (
    CameraId,
    LogPosition,
    ObjectId,
    ObservationId,
    PartitionVersion,
    SiteId,
)
from ..core.model.observation import Observation, ObservationType
from ..core.model.timebase import Duration, Instant
from ..core.model.vision_state import (
    CameraPartition,
    CameraStatus,
    CapabilityReport,
    CommitResult,
    ConsistencyLevel,
    CoverageMap,
    CoverageReport,
    ObjectState,
    ObservabilityState,
    SiteContext,
    StateDelta,
    StateSnapshot,
)
from ..core.ports.clock import Clock
from ..core.ports.synthesis import ObservationLogPort
from ..kernel.config.schema import StateSection
from ..kernel.events import (
    EventBus,
    ObservationQuarantined,
    PartitionDegraded,
    PartitionRecovered,
    StateRebuilt,
)
from ..kernel.metrics import MetricName, MetricsEngine
from .projection import ProjectionBounds, project

VISION_STATE_ID = "vision_state"
VISION_STATE_VERSION = "1.0.0"

#: The head of a partition's log. Rebuilding from here replays everything.
LOG_START = LogPosition(0)


@dataclass(frozen=True, slots=True)
class RebuildHandle:
    """The outcome of a rebuild (§M12 public API).

    07_STATE §9.2: rebuild runs into a shadow projection and swaps atomically, so
    *"consumers see a version change, never an outage."* The handle reports what
    was replayed so an operator can confirm the shadow caught up.
    """

    camera_id: CameraId
    replayed: int
    from_position: LogPosition
    to_position: LogPosition
    quarantined: int = 0

    @property
    def clean(self) -> bool:
        return self.quarantined == 0


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What a retention sweep removed (§M12 public API).

    §M12's failure table: *"Retry; alarm; **never delete without confirming what
    was deleted**."* The report is that confirmation.
    """

    partitions: int = 0
    records_removed: int = 0
    before: Instant | None = None
    failures: tuple[tuple[CameraId, str], ...] = ()

    @property
    def clean(self) -> bool:
        return not self.failures


@dataclass(slots=True)
class _Buffer:
    """A partition's bounded durability buffer (10_RELIABILITY §4.4 step 3).

    Bounded, and its filling is what triggers step 4. An unbounded buffer would
    turn a storage outage into an OOM, which is a worse failure than the one it
    was meant to survive.
    """

    pending: list[Observation] = field(default_factory=list)
    degraded_since: Instant | None = None

    @property
    def depth(self) -> int:
        return len(self.pending)


class VisionStateManager:
    """M12. Appends, projects, snapshots — and interprets nothing."""

    __slots__ = (
        "_bounds",
        "_buffers",
        "_clock",
        "_config",
        "_events",
        "_log",
        "_metrics",
        "_partitions",
        "_quarantined",
        "_seen",
        "_site_id",
        "_subscribers",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        events: EventBus,
        config: StateSection,
        log: ObservationLogPort,
        site_id: SiteId,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._events = events
        self._config = config
        self._log = log
        self._site_id = site_id
        self._bounds = ProjectionBounds(
            trajectory_points=config.trajectory_points,
            attribute_history=config.attribute_history,
            class_history=config.class_history,
            max_objects=config.max_objects_per_partition,
        )
        self._partitions: dict[CameraId, CameraPartition] = {}
        self._buffers: dict[CameraId, _Buffer] = {}
        self._seen: dict[CameraId, set[ObservationId]] = {}
        self._subscribers: list[Callable[[StateDelta], None]] = []
        self._quarantined = 0

    # --- the only write path --------------------------------------------------- #

    def append(self, observations: Sequence[Observation]) -> CommitResult:
        """Append to the log, then project. **The only way state changes.**

        Ordered deliberately: log first, project second. A projection that ran
        before a failed append would leave state holding something the log does
        not, and §9.1 makes the log authoritative — so a rebuild would silently
        produce a different world.

        Raises:
            CommitFailedError: the durability buffer filled and the partition has
                stopped accepting observations. 10_RELIABILITY §4.4 step 4 halts
                loudly rather than dropping facts, because *"losing observations
                invisibly is a V8 violation of the worst kind."*
            PartitionDegradedError: the partition is already halted.
        """
        if not observations:
            return CommitResult()

        cameras = {o.camera_id for o in observations}
        if len(cameras) > 1:
            # Per-partition append, always. A batch spanning cameras would need a
            # cross-partition commit, and §4.4 refuses those by design.
            return self._append_many(observations)

        camera_id = next(iter(cameras))
        return self._append_partition(camera_id, observations)

    def _append_many(self, observations: Sequence[Observation]) -> CommitResult:
        """Split a mixed batch by camera and commit each independently.

        Independently, not transactionally: §4.4's *"neither takes a
        cross-partition lock"*. One camera's storage problem must not stop
        another's facts from being recorded.
        """
        by_camera: dict[CameraId, list[Observation]] = {}
        for observation in observations:
            by_camera.setdefault(observation.camera_id, []).append(observation)

        accepted = 0
        rejected: list[tuple[ObservationId, str]] = []
        quarantined: list[tuple[ObservationId, str]] = []
        degraded = False
        position = LogPosition(0)
        version = PartitionVersion(0)

        for camera_id, batch in by_camera.items():
            try:
                result = self._append_partition(camera_id, batch)
            except (CommitFailedError, PartitionDegradedError) as exc:
                degraded = True
                rejected.extend((o.observation_id, exc.message) for o in batch)
                continue
            accepted += result.accepted
            rejected.extend(result.rejected)
            quarantined.extend(result.quarantined)
            position = result.log_position
            version = result.version

        return CommitResult(
            accepted=accepted,
            rejected=tuple(rejected),
            quarantined=tuple(quarantined),
            log_position=position,
            version=version,
            degraded=degraded,
        )

    def _append_partition(
        self, camera_id: CameraId, observations: Sequence[Observation]
    ) -> CommitResult:
        buffer = self._buffers.setdefault(camera_id, _Buffer())
        partition = self._partition(camera_id)

        if partition.is_degraded:
            raise PartitionDegradedError(
                f"partition '{camera_id}' has stopped accepting observations: "
                f"{partition.degraded_reason}",
                camera_id=str(camera_id),
            )

        started = self._clock.monotonic().ns
        pending = [*buffer.pending, *observations]

        try:
            appended = self._log.append(camera_id, pending)
        except Exception as exc:  # noqa: BLE001 - any store failure is the same here
            return self._buffer_or_halt(camera_id, buffer, pending, exc)

        buffer.pending.clear()
        if buffer.degraded_since is not None:
            self._recover(camera_id, buffer, drained=len(pending))

        self._metrics.histogram(
            MetricName.STATE_COMMIT_MS, camera_id=str(camera_id)
        ).record((self._clock.monotonic().ns - started) / 1_000_000)
        self._metrics.counter(
            MetricName.OBSERVATIONS_APPENDED, camera_id=str(camera_id)
        ).increment(appended.appended)
        if appended.duplicates:
            self._metrics.counter(
                MetricName.OBSERVATIONS_DUPLICATE, camera_id=str(camera_id)
            ).increment(len(appended.duplicates))

        return self._project_batch(camera_id, pending, appended, set(appended.duplicates))

    def _project_batch(
        self,
        camera_id: CameraId,
        observations: Sequence[Observation],
        appended,
        duplicates: set[ObservationId],
    ) -> CommitResult:
        """Project what was appended. Quarantine what will not project.

        §M12: *"Quarantine that observation, continue the projection, alarm. One
        bad record must not stop the world."* The observation stays in the log —
        it is part of the record — and only the projection could not absorb it,
        which is a projection bug rather than a producer one.
        """
        partition = self._partition(camera_id)
        seen = self._seen.setdefault(camera_id, set())
        started = self._clock.monotonic().ns

        accepted = 0
        quarantined: list[tuple[ObservationId, str]] = []
        changed_objects: list[ObjectId] = []
        changed_regions: list = []
        coverage_changed = False

        for index, observation in enumerate(observations):
            if observation.observation_id in duplicates or observation.observation_id in seen:
                # Idempotent by observation_id (§M13 append, §9.1 recovery).
                # Replaying is safe, which is what makes at-least-once workable.
                continue
            position = LogPosition(appended.position - len(observations) + index + 1)
            try:
                outcome = project(
                    partition, observation, bounds=self._bounds, position=position
                )
            except ProjectionError as exc:
                self._quarantined += 1
                quarantined.append((observation.observation_id, exc.message))
                self._publish_quarantine(camera_id, observation, exc.message)
                continue

            partition = outcome.partition
            accepted += 1
            seen.add(observation.observation_id)
            changed_objects.extend(outcome.changed_objects)
            changed_regions.extend(outcome.changed_regions)
            coverage_changed = coverage_changed or outcome.coverage_changed
            if outcome.evicted:
                self._metrics.counter(
                    MetricName.STATE_HISTORY_EVICTIONS, camera_id=str(camera_id)
                ).increment(outcome.evicted)

        self._partitions[camera_id] = partition
        self._trim_seen(camera_id, seen)

        self._metrics.histogram(
            MetricName.STATE_PROJECTION_MS, camera_id=str(camera_id)
        ).record((self._clock.monotonic().ns - started) / 1_000_000)
        self._metrics.gauge(
            MetricName.STATE_OBJECTS, camera_id=str(camera_id)
        ).set(float(partition.object_count))
        self._metrics.gauge(
            MetricName.STATE_LOG_POSITION, camera_id=str(camera_id)
        ).set(float(partition.log_position))

        if accepted or coverage_changed:
            self._notify(
                StateDelta(
                    camera_id=camera_id,
                    version=partition.version,
                    log_position=partition.log_position,
                    changed_objects=tuple(dict.fromkeys(changed_objects)),
                    changed_regions=tuple(dict.fromkeys(changed_regions)),
                    coverage_changed=coverage_changed,
                    at=self._clock.now(),
                )
            )

        return CommitResult(
            accepted=accepted,
            quarantined=tuple(quarantined),
            log_position=partition.log_position,
            version=partition.version,
        )

    # --- the durability ladder --------------------------------------------------- #

    def _buffer_or_halt(
        self,
        camera_id: CameraId,
        buffer: _Buffer,
        pending: list[Observation],
        cause: Exception,
    ) -> CommitResult:
        """10_RELIABILITY §4.4 steps 3 and 4.

        Buffer while there is room; **halt** when there is not. Halting is the
        correct trade under R1: an admitted gap is recoverable, and a hole nobody
        can see is not.
        """
        if len(pending) <= self._config.log_buffer_capacity:
            buffer.pending = list(pending)
            if buffer.degraded_since is None:
                buffer.degraded_since = self._clock.now()
            self._metrics.gauge(
                MetricName.STATE_BUFFER_DEPTH, camera_id=str(camera_id)
            ).set(float(buffer.depth))
            raise LogUnavailableError(
                f"log unavailable for partition '{camera_id}'; {buffer.depth} "
                f"observation(s) buffered locally ({cause})",
                camera_id=str(camera_id),
                buffered=buffer.depth,
            )

        reason = (
            f"durability buffer full at {self._config.log_buffer_capacity}; "
            f"the partition stops accepting observations rather than dropping "
            f"facts silently (10_RELIABILITY section 4.4 step 4)"
        )
        self._partitions[camera_id] = replace(
            self._partition(camera_id), degraded_reason=reason
        )
        self._metrics.counter(
            MetricName.STATE_PARTITIONS_DEGRADED, camera_id=str(camera_id)
        ).increment()
        self._events.publish(
            PartitionDegraded(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                reason="buffer_full",
                buffered=buffer.depth,
                detail=reason,
            )
        )
        raise CommitFailedError(reason, camera_id=str(camera_id))

    def _recover(self, camera_id: CameraId, buffer: _Buffer, *, drained: int) -> None:
        """A degraded partition drained and resumed.

        §4.4 step 5: *"drain buffer, resume, **emit coverage for the gap**"*. The
        coverage observation is M11's to build — this publishes the event that
        tells the runtime to ask for one, keeping "observation is the only write
        path" true even for the recovery record.
        """
        gap_ms = (
            (self._clock.now().ns - buffer.degraded_since.ns) / 1_000_000
            if buffer.degraded_since
            else 0.0
        )
        buffer.degraded_since = None
        self._events.publish(
            PartitionRecovered(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                drained=drained,
                gap_ms=gap_ms,
            )
        )

    def resume(self, camera_id: CameraId) -> None:
        """Clear a degraded partition after an operator confirms the store is back.

        Manual rather than automatic: a partition that halted because durability
        was at risk should not resume on a hopeful retry, because a second halt
        would lose the buffered facts it was holding.
        """
        partition = self._partition(camera_id)
        if not partition.is_degraded:
            return
        self._partitions[camera_id] = replace(partition, degraded_reason="")

    # --- read paths -------------------------------------------------------------- #

    def snapshot(self, scope: Sequence[CameraId] | None = None) -> StateSnapshot:
        """An immutable, consistent view. **O(1).**

        07_STATE §5.1: a snapshot is *"a pointer to an immutable root"* — no
        copying, because every partition is already a frozen value. This is the
        mechanism behind M14's claim that heavy query load cannot slow
        perception: the read path and the write path never touch the same
        mutable memory.
        """
        started = self._clock.monotonic().ns
        cameras = tuple(scope) if scope is not None else tuple(self._partitions)
        included: dict[CameraId, CameraPartition] = {}
        incomplete: list[tuple[CameraId, str]] = []

        for camera_id in cameras:
            partition = self._partitions.get(camera_id)
            if partition is None:
                incomplete.append((camera_id, "no state for this partition"))
                continue
            if partition.is_degraded:
                incomplete.append((camera_id, partition.degraded_reason))
                continue
            included[camera_id] = partition

        consistency = (
            ConsistencyLevel.STRONG
            if len(included) <= 1
            else ConsistencyLevel.SNAPSHOT_SET
        )
        self._metrics.histogram(MetricName.STATE_SNAPSHOT_MS).record(
            (self._clock.monotonic().ns - started) / 1_000_000
        )
        return StateSnapshot(
            partitions=included,
            site=self.site_context(),
            consistency=consistency,
            max_lag=self._max_lag(included),
            incomplete=tuple(incomplete),
            taken_at=self._clock.now(),
        )

    def object_state(self, object_id: ObjectId) -> ObjectState:
        """One object's current state.

        Raises:
            StateNotFoundError: no such object. Distinct from an object that
                exists and is empty — §7.1 is built on a consumer being able to
                tell those apart.
        """
        for partition in self._partitions.values():
            found = partition.objects.get(object_id)
            if found is not None:
                return found
        raise StateNotFoundError(
            f"no state for object '{object_id}'", object_id=str(object_id)
        )

    def history(
        self, object_id: ObjectId, *, window: Duration | None = None, limit: int = 100
    ) -> tuple[Observation, ...]:
        """Observations for one object, read from the **log**.

        From the log rather than from state, deliberately: 07_STATE §6.2 puts
        operational history in the log and only *working* history in memory.
        Serving history from state would make the projection a time-series
        database, which §6.1 explicitly refuses.
        """
        matches: list[Observation] = []
        cutoff = (
            Instant(self._clock.now().ns - window.ns) if window is not None else None
        )
        for camera_id in self._partitions:
            for observation in self._log.read(camera_id, limit=limit * 4):
                if observation.object_id != object_id:
                    continue
                if cutoff is not None and observation.t_capture.ns < cutoff.ns:
                    continue
                matches.append(observation)
        matches.sort(key=lambda o: o.t_capture.ns)
        return tuple(matches[-limit:])

    def observations_in(
        self,
        camera_id: CameraId,
        *,
        since: Instant,
        until: Instant,
        limit: int = 10_000,
    ) -> tuple[Observation, ...]:
        """Observations for one partition over a window, read from the **log**.

        The L6 → L7 seam for 09_API §2.2's historical query. M14 reads through
        M12 rather than holding P20 itself: §M14's Dependencies name the Vision
        State Manager, not the log, and giving L7 a storage port would let a
        query bypass the layer that owns partitioning and consistency.

        Against ``t_capture``, never ingest time (V11, §2.2) — a window against
        ingest time would return different results depending on how backed up the
        pipeline happened to be.

        Ordered by ``(t_capture, observation_id)``, which §2.2 requires be *"total
        and stable"*: total because two observations can share a capture instant,
        stable because a cursor over an immutable log must land in the same place
        every time.
        """
        matches = [
            observation
            for observation in self._log.read(camera_id, limit=limit)
            if since.ns <= observation.t_capture.ns <= until.ns
        ]
        matches.sort(key=lambda o: (o.t_capture.ns, str(o.observation_id)))
        return tuple(matches)

    def coverage(self, scope: Sequence[CameraId] | None = None) -> CoverageMap:
        """Live coverage — *"can we see right now?"* (07_STATE §7.3)."""
        cameras = tuple(scope) if scope is not None else tuple(self._partitions)
        by_camera = {}
        for camera_id in cameras:
            partition = self._partitions.get(camera_id)
            if partition is None:
                continue
            by_camera[camera_id] = partition.observability or ObservabilityState(
                camera_id=camera_id
            )
        return CoverageMap(by_camera=by_camera, at=self._clock.now())

    def coverage_report(
        self, camera_id: CameraId, *, since: Instant, until: Instant
    ) -> CoverageReport:
        """Historical coverage over a window, reconstructed from the log.

        §7.3: *"a query over any past window can reconstruct exactly what was
        observable then."* The answer a consumer needs alongside an empty result,
        because without it *"the region was empty"* and *"the camera was blind"*
        are the same answer.
        """
        windows = [
            observation.coverage
            for observation in self._log.read(camera_id, limit=10_000)
            if observation.observation_type is ObservationType.COVERAGE
            and observation.coverage is not None
        ]
        from ..core.model.observation import coverage_gap

        gaps = tuple(
            (w.since, w.until or until, w.reason, str(camera_id))
            for w in windows
            if w.is_gap
        )
        return CoverageReport(
            observable_fraction=coverage_gap(windows, since=since, until=until),
            gaps=gaps,
            effective_rate=windows[-1].effective_rate if windows else 1.0,
        )

    def site_context(self) -> SiteContext:
        """Site aggregation. **Eventually consistent, by design** (§4.4)."""
        return SiteContext(
            site_id=self._site_id,
            coverage=CoverageMap(
                by_camera={
                    camera: (p.observability or ObservabilityState(camera_id=camera))
                    for camera, p in self._partitions.items()
                },
                at=self._clock.now(),
            ),
            capabilities=CapabilityReport(),
            camera_count=len(self._partitions),
            object_count=sum(p.object_count for p in self._partitions.values()),
        )

    def subscribe(self, listener: Callable[[StateDelta], None]) -> Callable[[], None]:
        """Receive state deltas. Returns an unsubscribe callable.

        A slow subscriber must never stall the writer, so delivery is a direct
        call the writer makes defensively — a raising listener is dropped from the
        batch, not allowed to propagate into the commit path.
        """
        self._subscribers.append(listener)

        def unsubscribe() -> None:
            if listener in self._subscribers:
                self._subscribers.remove(listener)

        return unsubscribe

    # --- recovery ----------------------------------------------------------------- #

    def rebuild(
        self, camera_id: CameraId, *, from_position: LogPosition = LOG_START
    ) -> RebuildHandle:
        """Rebuild a partition's projection from the log.

        07_STATE §9.2: into a **shadow** projection, then swapped atomically, so
        *"consumers see a version change, never an outage."* The live partition
        keeps serving until the last statement of this method.

        This is the capability §9.1 calls *"the strongest argument for event
        sourcing here"*: a projection bug costs a rebuild, not data.
        """
        shadow = CameraPartition(camera_id=camera_id)
        replayed = 0
        quarantined = 0
        position = from_position

        for observation in self._log.read(camera_id, start=from_position):
            position = LogPosition(position + 1)
            try:
                outcome = project(
                    shadow, observation, bounds=self._bounds, position=position
                )
            except ProjectionError:
                quarantined += 1
                continue
            shadow = outcome.partition
            replayed += 1

        live = self._partitions.get(camera_id)
        # The swap. One assignment, so no reader ever observes a half-rebuilt
        # partition — the old root stays valid for anyone already holding it.
        self._partitions[camera_id] = replace(
            shadow,
            version=PartitionVersion((live.version if live else 0) + 1),
            status=shadow.status or (live.status if live else None),
        )
        self._seen[camera_id] = set()

        self._metrics.counter(
            MetricName.STATE_REBUILDS, camera_id=str(camera_id)
        ).increment()
        self._events.publish(
            StateRebuilt(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                observations_replayed=replayed,
                from_position=int(from_position),
                detail="shadow projection swapped atomically",
            )
        )
        return RebuildHandle(
            camera_id=camera_id,
            replayed=replayed,
            from_position=from_position,
            to_position=position,
            quarantined=quarantined,
        )

    def retention_sweep(self, *, before: Instant | None = None) -> SweepReport:
        """Truncate the log's cold prefix.

        §M12's failure table: *"never delete without confirming what was
        deleted."* The report is the confirmation, and a failure names its
        partition rather than being swallowed.
        """
        cutoff = before or Instant(
            self._clock.now().ns - self._config.log_retention_ms * 1_000_000
        )
        removed = 0
        failures: list[tuple[CameraId, str]] = []
        for camera_id in tuple(self._partitions):
            try:
                removed += self._log.truncate(camera_id, cutoff)
            except Exception as exc:  # noqa: BLE001 - a store may fail many ways
                failures.append((camera_id, f"{type(exc).__name__}: {exc}"))
        return SweepReport(
            partitions=len(self._partitions),
            records_removed=removed,
            before=cutoff,
            failures=tuple(failures),
        )

    # --- internals ------------------------------------------------------------------ #

    def _partition(self, camera_id: CameraId) -> CameraPartition:
        partition = self._partitions.get(camera_id)
        if partition is None:
            partition = CameraPartition(
                camera_id=camera_id, status=CameraStatus(camera_id=camera_id)
            )
            self._partitions[camera_id] = partition
            self._metrics.gauge(MetricName.STATE_PARTITIONS).set(
                float(len(self._partitions))
            )
        return partition

    def _trim_seen(self, camera_id: CameraId, seen: set[ObservationId]) -> None:
        """Bound the idempotency set.

        Unbounded it would grow with every observation ever appended — the same
        leak the history rings exist to prevent. Trimming risks re-projecting a
        very old duplicate, which is harmless: projection is idempotent in effect
        because an older observation cannot move state backwards.
        """
        limit = self._config.log_buffer_capacity * 4
        if len(seen) <= limit:
            return
        self._seen[camera_id] = set(list(seen)[-limit:])

    def _max_lag(self, partitions: dict[CameraId, CameraPartition]) -> Duration:
        """Worst staleness across included partitions (§5.2's ``max_lag``)."""
        now = self._clock.now().ns
        lags = [
            now - p.status.last_observation_at.ns
            for p in partitions.values()
            if p.status and p.status.last_observation_at
        ]
        return Duration(max(lags) if lags else 0)

    def _notify(self, delta: StateDelta) -> None:
        self._metrics.counter(
            MetricName.STATE_SUBSCRIBER_DELTAS, camera_id=str(delta.camera_id)
        ).increment()
        for listener in tuple(self._subscribers):
            # A raising or slow subscriber is dropped from this batch rather than
            # allowed into the commit path: 07_STATE section 5.2 requires that one
            # slow consumer never stall the platform.
            with contextlib.suppress(Exception):
                listener(delta)

    def _publish_quarantine(
        self, camera_id: CameraId, observation: Observation, reason: str
    ) -> None:
        self._metrics.counter(
            MetricName.OBSERVATIONS_QUARANTINED, camera_id=str(camera_id)
        ).increment()
        self._events.publish(
            ObservationQuarantined(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                observation_id=str(observation.observation_id),
                reason=reason,
            )
        )

    # --- observability ----------------------------------------------------------------- #

    def health(self) -> ComponentHealth:
        degraded = [
            str(camera)
            for camera, partition in self._partitions.items()
            if partition.is_degraded
        ]
        state = HealthState.HEALTHY
        detail = "state nominal"
        if degraded:
            state = HealthState.DEGRADED
            detail = (
                f"partition(s) {', '.join(sorted(degraded))} stopped accepting "
                f"observations; facts are not being recorded for them"
            )
        elif self._quarantined:
            state = HealthState.DEGRADED
            detail = f"{self._quarantined} observation(s) quarantined"

        return ComponentHealth(
            component_id=VISION_STATE_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "partitions": float(len(self._partitions)),
                "objects": float(
                    sum(p.object_count for p in self._partitions.values())
                ),
                "quarantined": float(self._quarantined),
                "degraded_partitions": float(len(degraded)),
            },
        )

    @property
    def partitions(self) -> tuple[CameraId, ...]:
        return tuple(sorted(self._partitions))

    @property
    def quarantined(self) -> int:
        return self._quarantined

    def buffer_depth(self, camera_id: CameraId) -> int:
        buffer = self._buffers.get(camera_id)
        return buffer.depth if buffer else 0

    def forget(self, camera_id: CameraId) -> None:
        """Release a camera's partition after it is retired.

        Not a retention operation: the log is untouched, so the record survives
        and the partition can be rebuilt from it at any time.
        """
        self._partitions.pop(camera_id, None)
        self._buffers.pop(camera_id, None)
        self._seen.pop(camera_id, None)


def observations_of(batch: Iterable) -> tuple[Observation, ...]:
    """Flatten a build pass into an append-ready sequence."""
    return tuple(o for o in batch if isinstance(o, Observation))
