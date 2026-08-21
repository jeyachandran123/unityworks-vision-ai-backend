"""Per-object trigger state, partitioned by camera.

> §M8 State Ownership: *"per-object trigger state (last analysis time per
> attribute, last appearance signature), budget accounting, crop deduplication
> cache, priority queues. **Ephemeral and node-local**; rebuilt from registry
> state after restart, with the conservative consequence that a restart causes
> one round of ``FIRST_SIGHT`` re-analysis. Acceptable and bounded."*

Two design consequences follow from that paragraph, and both are load-bearing.

**Nothing here is durable.** There is no store, no snapshot, no reload. A restart
loses trigger state and the platform re-analyses once — which is a bounded,
predictable, *conservative* cost. Persisting it would create a second writer of
something the registry already owns, and a stale trigger record after a restart
would suppress the analysis that the restart made necessary.

**The partition is the camera**, matching M7. Per-camera single-writer means a
camera's trigger state is touched by one caller at a time and never by another
camera, so there are no cross-camera locks in this file — only in the budget,
which is genuinely shared and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.model.crop import GateRejection
from ...core.model.ids import AttributeKey, CameraId, CropId, ObjectId
from ...core.model.timebase import Duration, Instant


@dataclass(slots=True)
class ObjectTriggerState:
    """What M8 remembers about one object between frames.

    Mutable and node-local. Deliberately tiny: everything durable about the
    object lives in M7, and duplicating it here would create a second version of
    the truth that drifts.
    """

    object_id: ObjectId
    first_seen_by_manager: Instant
    last_analysed: Instant | None = None
    last_analysed_per_attribute: dict[AttributeKey, Instant] = field(default_factory=dict)
    last_appearance: float | None = None
    """The previous appearance scalar. The *delta* is what triggers; keeping the
    absolute value is what makes a delta computable."""

    last_gate_rejection: GateRejection | None = None
    consecutive_gate_rejections: int = 0
    """Feeds the capability-gap detector: one rejection is weather, fifty in a row
    at the same camera is a mounting problem the consumer should be told about."""

    last_crop_id: CropId | None = None
    last_lifecycle: str = ""
    last_region_ids: frozenset = frozenset()
    analyses: int = 0
    skips: int = 0

    def note_analysis(
        self, now: Instant, attributes: tuple[AttributeKey, ...], crop_id: CropId | None
    ) -> None:
        self.last_analysed = now
        for key in attributes:
            self.last_analysed_per_attribute[key] = now
        if crop_id is not None:
            self.last_crop_id = crop_id
        self.analyses += 1
        self.last_gate_rejection = None
        self.consecutive_gate_rejections = 0

    def note_gate_rejection(self, reason: GateRejection) -> None:
        self.last_gate_rejection = reason
        self.consecutive_gate_rejections += 1

    def note_appearance(self, signature: float | None) -> float | None:
        """Store the new signature and return the delta from the previous one.

        ``None`` on the first observation — *no measurement*, distinct from a
        measured delta of zero. A zero here would say "the appearance did not
        change", which is a claim the platform cannot make about an object it
        has seen once (V8).
        """
        previous = self.last_appearance
        self.last_appearance = signature
        if previous is None or signature is None:
            return None
        return abs(signature - previous)

    def since_analysis(self, now: Instant) -> Duration | None:
        if self.last_analysed is None:
            return None
        return Duration(max(0, now.ns - self.last_analysed.ns))


@dataclass(slots=True)
class CameraTriggerPartition:
    """One camera's trigger state. **Single-writer.**

    Bounded by ``capacity``: an unbounded map here would grow with every object a
    camera has ever seen, which is the same memory leak §M7 calls *"a memory leak
    with a face"* wearing different clothes. Eviction is least-recently-analysed,
    and an evicted object simply re-triggers as ``FIRST_SIGHT`` — the same
    conservative failure the restart path already accepts.
    """

    camera_id: CameraId
    capacity: int = 4096
    objects: dict[ObjectId, ObjectTriggerState] = field(default_factory=dict)
    evictions: int = 0
    frames_evaluated: int = 0
    candidates_evaluated: int = 0

    def state_for(self, object_id: ObjectId, *, now: Instant) -> ObjectTriggerState:
        state = self.objects.get(object_id)
        if state is None:
            state = ObjectTriggerState(object_id=object_id, first_seen_by_manager=now)
            self.objects[object_id] = state
            self._evict_if_needed()
        return state

    def forget(self, object_id: ObjectId) -> None:
        """Drop an object that left the registry's population."""
        self.objects.pop(object_id, None)

    def retain_only(self, live: frozenset) -> int:
        """Drop everything the registry no longer knows about.

        Called on the maintenance schedule rather than per frame: reconciling
        against the whole population every frame would make M8's cost
        proportional to population rather than to candidates.
        """
        doomed = [key for key in self.objects if key not in live]
        for key in doomed:
            del self.objects[key]
        return len(doomed)

    def _evict_if_needed(self) -> None:
        while len(self.objects) > self.capacity:
            oldest = min(
                self.objects,
                key=lambda key: (
                    self.objects[key].last_analysed.ns
                    if self.objects[key].last_analysed
                    else self.objects[key].first_seen_by_manager.ns
                ),
            )
            del self.objects[oldest]
            self.evictions += 1

    @property
    def tracked_objects(self) -> int:
        return len(self.objects)


class TriggerStateStore:
    """Every camera's trigger partitions.

    A plain dict of partitions with no lock, because the runtime serializes each
    camera's writes — the same arrangement M7 uses, for the same reason. Adding a
    lock here would suggest cross-camera access is expected, and it is not.
    """

    __slots__ = ("_capacity", "_partitions")

    def __init__(self, *, capacity_per_camera: int = 4096) -> None:
        if capacity_per_camera < 1:
            raise ValueError("capacity_per_camera must be >= 1")
        self._capacity = capacity_per_camera
        self._partitions: dict[CameraId, CameraTriggerPartition] = {}

    def partition(self, camera_id: CameraId) -> CameraTriggerPartition:
        partition = self._partitions.get(camera_id)
        if partition is None:
            partition = CameraTriggerPartition(
                camera_id=camera_id, capacity=self._capacity
            )
            self._partitions[camera_id] = partition
        return partition

    def drop(self, camera_id: CameraId) -> None:
        """Release a camera's state when it detaches."""
        self._partitions.pop(camera_id, None)

    @property
    def cameras(self) -> tuple[CameraId, ...]:
        return tuple(sorted(self._partitions))

    @property
    def tracked_objects(self) -> int:
        return sum(p.tracked_objects for p in self._partitions.values())

    @property
    def evictions(self) -> int:
        return sum(p.evictions for p in self._partitions.values())


@dataclass(slots=True)
class GateRejectionWindow:
    """A rolling count of gate outcomes for one camera.

    Exists to make ``GateRejectionSpike`` reportable without storing every
    rejection: two integers and a reason tally answer "is the rate abnormal and
    what is the dominant cause", which is all the alarm needs.
    """

    camera_id: CameraId
    window: int = 100
    outcomes: list[bool] = field(default_factory=list)
    by_reason: dict[GateRejection, int] = field(default_factory=dict)
    alarm_active: bool = False

    def record(self, *, passed: bool, reason: GateRejection | None = None) -> None:
        self.outcomes.append(passed)
        if len(self.outcomes) > self.window:
            self.outcomes.pop(0)
        if reason is not None:
            self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    @property
    def sample_size(self) -> int:
        return len(self.outcomes)

    @property
    def rejection_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for passed in self.outcomes if not passed) / len(self.outcomes)

    def dominant_reason(self) -> GateRejection | None:
        """The most common rejection, which is what makes the alarm actionable.

        Ties resolve on the enum's value so the reported cause is stable across
        runs rather than dependent on dict ordering (V13).
        """
        if not self.by_reason:
            return None
        return max(self.by_reason, key=lambda r: (self.by_reason[r], r.value))
