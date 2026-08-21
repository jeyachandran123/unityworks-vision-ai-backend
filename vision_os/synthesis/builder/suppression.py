"""Change suppression — the only state M11 owns.

> 04_MODULES §M11: *"**Change suppression is the main performance feature**, and
> it is a correctness feature too: without it, a stationary object publishes an
> identical observation at full frame rate forever, which floods storage,
> subscribers, and consumers with no information. Typical reduction is 10-50x."*

Read the second clause carefully. Suppression is not only cheaper — an
observation stream where every frame repeats the same fact carries *less*
information than one that publishes on change, because a consumer cannot tell
which entries mean anything.

**The heartbeat is what makes suppression safe.** §M11 again: *"a consumer must
be able to distinguish 'unchanged' from 'stopped observing,' so unchanged objects
still publish at a slow floor rate (V8)."* Suppression without a heartbeat
converts a working camera and a dead one into the same silence.

**State ownership.** §M11: *"last-published signature per (object, observation
type)... Small, ephemeral, per-camera."* And the reason it must stay small:
*"the builder must be a pure, heavily-testable function of its inputs, and giving
it durable state would compromise that."*
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.model.ids import CameraId, ObjectId, ObservationId
from ...core.model.observation import Observation, ObservationType
from ...core.model.timebase import Duration, Instant

#: Bound on tracked subjects per camera.
#:
#: An unbounded signature map grows with every object a camera has ever seen —
#: the same leak §M7 calls *"a memory leak with a face"*. An evicted subject
#: simply republishes, which is the conservative failure the restart path already
#: accepts.
DEFAULT_SUBJECT_CAPACITY = 4096

#: How long an unchanged subject may go without publishing.
#:
#: The V8 floor. Long enough that suppression still saves 10-50x; short enough
#: that a consumer notices silence well before it would matter.
DEFAULT_HEARTBEAT = Duration.from_millis(30_000)


#: The key suppression is tracked against: one signature per subject *per type*.
#:
#: Per type, deliberately: an object whose position changed but whose attributes
#: did not should publish a `spatial` observation and suppress the `attribute`
#: one. A single key per object would couple them and lose one or the other.
SubjectKey = tuple[ObjectId | None, ObservationType]


@dataclass(slots=True)
class PublishedSignature:
    """What was last published for one subject and type."""

    signature: str
    observation_id: ObservationId
    published_at: Instant
    sequence: int = 0
    """How many observations of this type this subject has published. Feeds
    lineage without a separate counter."""


@dataclass(slots=True)
class CameraSuppressionState:
    """One camera's suppression memory. **Single-writer.**

    Bounded by ``capacity``. Eviction is least-recently-published, and an evicted
    subject republishes on its next observation — brief duplication, which §M11
    explicitly prefers: *"brief duplication is harmless, missing data is not."*
    """

    camera_id: CameraId
    capacity: int = DEFAULT_SUBJECT_CAPACITY
    published: dict[SubjectKey, PublishedSignature] = field(default_factory=dict)
    evictions: int = 0
    suppressed: int = 0
    heartbeats: int = 0

    def last(self, key: SubjectKey) -> PublishedSignature | None:
        return self.published.get(key)

    def record(
        self, key: SubjectKey, signature: str, observation: Observation
    ) -> int:
        """Note a publication. Returns the subject's new sequence number."""
        previous = self.published.get(key)
        sequence = (previous.sequence + 1) if previous else 1
        self.published[key] = PublishedSignature(
            signature=signature,
            observation_id=observation.observation_id,
            published_at=observation.t_capture,
            sequence=sequence,
        )
        self._evict_if_needed()
        return sequence

    def forget(self, object_id: ObjectId) -> int:
        """Drop a departed object's signatures across every type."""
        doomed = [key for key in self.published if key[0] == object_id]
        for key in doomed:
            del self.published[key]
        return len(doomed)

    def _evict_if_needed(self) -> None:
        while len(self.published) > self.capacity:
            oldest = min(
                self.published,
                key=lambda key: self.published[key].published_at.ns,
            )
            del self.published[oldest]
            self.evictions += 1

    @property
    def tracked(self) -> int:
        return len(self.published)


class SuppressionStateStore:
    """Every camera's suppression state.

    A plain dict with no lock, because the runtime serializes each camera's
    writes — the same arrangement M7 and M8 use, for the same reason. Adding a
    lock here would suggest cross-camera access is expected, and it is not.

    **Deliberately not durable.** §M11's failure table: *"Suppression state lost
    (restart) — publish a full snapshot for active objects; brief duplication is
    harmless, missing data is not."* Persisting it would create a second writer
    of something derivable, and a stale signature after a restart would suppress
    the very republication the restart made necessary.
    """

    __slots__ = ("_capacity", "_partitions")

    def __init__(self, *, capacity_per_camera: int = DEFAULT_SUBJECT_CAPACITY) -> None:
        if capacity_per_camera < 1:
            raise ValueError("capacity_per_camera must be >= 1")
        self._capacity = capacity_per_camera
        self._partitions: dict[CameraId, CameraSuppressionState] = {}

    def partition(self, camera_id: CameraId) -> CameraSuppressionState:
        partition = self._partitions.get(camera_id)
        if partition is None:
            partition = CameraSuppressionState(
                camera_id=camera_id, capacity=self._capacity
            )
            self._partitions[camera_id] = partition
        return partition

    def drop(self, camera_id: CameraId) -> None:
        self._partitions.pop(camera_id, None)

    @property
    def cameras(self) -> tuple[CameraId, ...]:
        return tuple(sorted(self._partitions))

    @property
    def tracked_subjects(self) -> int:
        return sum(p.tracked for p in self._partitions.values())

    @property
    def suppressed(self) -> int:
        return sum(p.suppressed for p in self._partitions.values())


def subject_key(observation: Observation) -> SubjectKey:
    return (observation.object_id, observation.observation_type)
