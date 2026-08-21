"""Replay verification — proving a rebuild reproduces the world (V13).

> 07_STATE §9.1: every recovery scenario that reports *"no data loss"* reports it
> **because** the log is authoritative and replay reconstructs what was there.

The brief: *"Replay must reconstruct **identical** Vision State from the
Observation Log. Replay must be deterministic, repeatable, auditable. No replay
shortcut may exist."*

**There is no shortcut, and that is architectural rather than disciplinary.**
Replay does not have its own code path: it calls the same `project` function the
live write path calls, over the same log, into a fresh partition. A faster path
that skipped or approximated the projection would be a second implementation of
the world, and two implementations diverge — quietly, and in a way nothing would
detect, because the projection was the only other copy of what the log means.

What this module adds is the **comparison**: a structural diff of two partitions
that says exactly where they disagree. Without it, "replay reproduces state" is a
claim; with it, it is a test that names the field that drifted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.model.ids import CameraId, LogPosition, ObjectId
from ..core.model.observation import Observation
from ..core.model.vision_state import CameraPartition, ObjectState
from ..core.ports.synthesis import ObservationLogPort
from .projection import ProjectionBounds, project


@dataclass(frozen=True, slots=True)
class Divergence:
    """One field on which two projections of the same log disagree.

    Named down to the field because *"replay produced different state"* is not
    actionable. *"Object o-14's ``last_confirmed`` is 200ms earlier"* points at
    the projection rule that drifted.
    """

    scope: str
    field_name: str
    live: object
    replayed: object

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{self.scope}.{self.field_name}: live={self.live!r} replayed={self.replayed!r}"


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """The outcome of replaying one partition.

    ``identical`` is the whole point. Everything else is here so that a failure
    is diagnosable rather than merely detected.
    """

    camera_id: CameraId
    observations: int = 0
    objects_live: int = 0
    objects_replayed: int = 0
    divergences: tuple[Divergence, ...] = field(default_factory=tuple)
    duration_ms: float = 0.0
    log_position: LogPosition = LogPosition(0)

    @property
    def identical(self) -> bool:
        return not self.divergences

    @property
    def observations_per_second(self) -> float:
        return (
            self.observations / (self.duration_ms / 1000.0)
            if self.duration_ms > 0
            else 0.0
        )

    def summary(self) -> str:
        if self.identical:
            return (
                f"{self.camera_id}: {self.observations} observations replayed "
                f"identically in {self.duration_ms:.1f}ms"
            )
        return (
            f"{self.camera_id}: {len(self.divergences)} divergences after "
            f"{self.observations} observations"
        )


def replay_partition(
    log: ObservationLogPort,
    camera_id: CameraId,
    *,
    bounds: ProjectionBounds,
    start: LogPosition | None = None,
) -> tuple[CameraPartition, int]:
    """Rebuild one partition from the log. Returns the partition and its count.

    **The same `project` the live path uses.** Not a similar one, not an
    optimized one — the identical function, so there is no second definition of
    what an observation means to state.
    """
    partition = CameraPartition(camera_id=camera_id)
    count = 0
    for index, observation in enumerate(log.read(camera_id, start=start, limit=1_000_000)):
        outcome = project(
            partition, observation, bounds=bounds, position=LogPosition(index + 1)
        )
        partition = outcome.partition
        count += 1
    return partition, count


def compare_partitions(
    live: CameraPartition, replayed: CameraPartition
) -> tuple[Divergence, ...]:
    """Structural diff of two projections of the same log.

    Compares the fields that carry *meaning*, not the bookkeeping that
    legitimately differs. ``version`` counts writes and a replay performs a
    different number of them; ``log_position`` is a watermark into a log the
    replay may have started partway through. Comparing those would report a
    divergence on every successful replay, and a test that always fails is a test
    that gets deleted.
    """
    found: list[Divergence] = []

    if set(live.objects) != set(replayed.objects):
        missing = sorted(set(live.objects) - set(replayed.objects))
        extra = sorted(set(replayed.objects) - set(live.objects))
        found.append(
            Divergence(
                scope=str(live.camera_id),
                field_name="objects",
                live=f"{len(live.objects)} objects, missing from replay: {missing[:5]}",
                replayed=f"{len(replayed.objects)} objects, extra in replay: {extra[:5]}",
            )
        )

    for object_id in sorted(set(live.objects) & set(replayed.objects)):
        found.extend(
            _compare_object(object_id, live.objects[object_id], replayed.objects[object_id])
        )

    # ``observability`` is ``None`` until a coverage observation arrives, and a
    # partition that never went blind legitimately has none. Comparing the
    # *presence* as well as the value matters: a replay that invented a default
    # observability state would be manufacturing a coverage claim the log never
    # made, which is the exact inverse of V8.
    live_status = live.observability.status if live.observability else None
    replayed_status = replayed.observability.status if replayed.observability else None
    if live_status is not replayed_status:
        found.append(
            Divergence(
                scope=str(live.camera_id),
                field_name="observability.status",
                live=live_status.value if live_status else None,
                replayed=replayed_status.value if replayed_status else None,
            )
        )

    if set(live.regions) != set(replayed.regions):
        found.append(
            Divergence(
                scope=str(live.camera_id),
                field_name="regions",
                live=sorted(live.regions),
                replayed=sorted(replayed.regions),
            )
        )

    return tuple(found)


def _compare_object(
    object_id: ObjectId, live: ObjectState, replayed: ObjectState
) -> list[Divergence]:
    """Every field of an object that must survive a replay.

    ``last_confirmed`` is here for a specific reason: it is the field that
    distinguishes a measured sighting from a believed one (V8), and a projection
    bug that confirmed on a prediction would show up nowhere else — the object
    would still be present, at the right place, with the right class.
    """
    out: list[Divergence] = []
    scope = str(object_id)

    for name in (
        "class_id",
        "lifecycle",
        "first_seen",
        "last_seen",
        "last_confirmed",
        "observation_count",
        "measurement_basis",
    ):
        a, b = getattr(live, name), getattr(replayed, name)
        if a != b:
            out.append(Divergence(scope=scope, field_name=name, live=a, replayed=b))

    if set(live.attributes) != set(replayed.attributes):
        out.append(
            Divergence(
                scope=scope,
                field_name="attributes",
                live=sorted(str(k) for k in live.attributes),
                replayed=sorted(str(k) for k in replayed.attributes),
            )
        )
    else:
        for key in live.attributes:
            if live.attributes[key].value != replayed.attributes[key].value:
                out.append(
                    Divergence(
                        scope=scope,
                        field_name=f"attributes[{key}]",
                        live=live.attributes[key].value,
                        replayed=replayed.attributes[key].value,
                    )
                )

    if set(live.regions) != set(replayed.regions):
        out.append(
            Divergence(
                scope=scope,
                field_name="regions",
                live=sorted(str(r) for r in live.regions),
                replayed=sorted(str(r) for r in replayed.regions),
            )
        )

    return out


class ReplayVerifier:
    """Replays a log and proves the result matches live state.

    Used by an operator before trusting a rebuild, by CI on every change to the
    projection, and by a deployment validating a storage migration. §9.1's
    *"projection bug: fix, rebuild into a shadow projection, atomic swap"* is
    only safe if somebody checked that the shadow agrees.
    """

    __slots__ = ("_bounds", "_clock", "_log", "_metrics")

    def __init__(self, *, clock, metrics, log: ObservationLogPort, bounds: ProjectionBounds) -> None:
        self._clock = clock
        self._metrics = metrics
        self._log = log
        self._bounds = bounds

    def verify(self, camera_id: CameraId, live: CameraPartition) -> ReplayReport:
        """Replay one partition and compare it to the live projection."""
        from ..kernel.metrics import MetricName

        started = self._clock.monotonic().ns
        replayed, count = replay_partition(self._log, camera_id, bounds=self._bounds)
        elapsed = (self._clock.monotonic().ns - started) / 1_000_000

        divergences = compare_partitions(live, replayed)

        self._metrics.counter(MetricName.REPLAY_RUNS).increment()
        self._metrics.counter(MetricName.REPLAY_OBSERVATIONS).increment(count)
        self._metrics.histogram(MetricName.REPLAY_MS).record(elapsed)
        if divergences:
            self._metrics.counter(
                MetricName.REPLAY_MISMATCHES, camera_id=str(camera_id)
            ).increment(len(divergences))

        return ReplayReport(
            camera_id=camera_id,
            observations=count,
            objects_live=len(live.objects),
            objects_replayed=len(replayed.objects),
            divergences=divergences,
            duration_ms=elapsed,
            log_position=self._log.position(camera_id),
        )

    def verify_all(self, partitions: Sequence[tuple[CameraId, CameraPartition]]):
        return tuple(self.verify(camera_id, live) for camera_id, live in partitions)


def deterministic_digest(observations: Sequence[Observation]) -> str:
    """A stable fingerprint of a log's semantic content.

    Two runs producing the same digest produced the same facts in the same order.
    Excludes ``t_published`` — when the platform *said* something is not part of
    what it said, and a replay legitimately publishes at a different wall time.
    """
    import hashlib

    digest = hashlib.sha256()
    for observation in observations:
        digest.update(str(observation.observation_id).encode())
        digest.update(observation.observation_type.value.encode())
        digest.update(str(observation.t_capture.ns).encode())
        digest.update(str(observation.object_id or "").encode())
        digest.update(str(observation.class_id or "").encode())
        for attribute in observation.attributes:
            digest.update(f"{attribute.key}={attribute.value}".encode())
    return digest.hexdigest()
