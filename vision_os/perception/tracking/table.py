"""The per-camera track table — the tracker's mutable state.

> **Single responsibility:** *Hold the live tracks for one camera. Associate
> nothing, predict nothing, decide no lifecycle.*

This is the platform's most volatile state and is deliberately **not durable**
(03_MODULES M6 state ownership). On restart tracks do not survive; a new
``TrackerEpoch`` is minted and consumers see the discontinuity rather than
inferring that objects teleported.

Two bounds are structural rather than advisory (port obligation T8):

* ``max_tracks`` caps the table. A crowd degrades by refusing new tracks, never
  by growing without limit.
* ``history_length`` caps each track's frame history to a ring. An hour-long
  track holds the same memory as a one-second track.

**Single-writer.** One table belongs to one camera actor and is never shared, so
there are no locks here and none are needed (08_RUNTIME section 2).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ...core.errors import TrackerCapacityError
from ...core.model.ids import (
    CameraId,
    ClassId,
    FrameRef,
    LocalTrackId,
    SiteId,
    TenantId,
    TrackerEpoch,
    TrackId,
)
from ...core.model.space import Box
from ...core.model.timebase import Instant
from ...core.model.track import (
    AssociationMethod,
    BreakReason,
    MotionState,
    TrackState,
)
from ...core.ports.tracking import MotionPredictorPort

#: Frames retained per track. Bounded so memory is a function of track count,
#: not of how long any track has lived.
DEFAULT_HISTORY_LENGTH = 32

#: The epoch a camera starts at before any reset has occurred.
INITIAL_EPOCH = TrackerEpoch(0)


@dataclass(slots=True)
class TrackRecord:
    """One live track's mutable state. Internal; never leaves the layer.

    The immutable ``Track`` handed to consumers is projected from this. Keeping
    the two separate is what lets the published object be frozen (invariant V5)
    while the tracker still updates cheaply in place.
    """

    track_id: TrackId
    class_id: ClassId
    box: Box
    state: TrackState
    predictor: MotionPredictorPort

    tenant_id: TenantId
    site_id: SiteId
    """Inherited from the detections that formed this track, never from global
    configuration. Tenancy is declared per camera, so a tracker serving two
    tenants on one node must carry each track's own — a single construction-time
    value would silently stamp the wrong tenant on half of them and breach the
    platform's hard isolation boundary."""

    first_seen: Instant
    last_seen: Instant
    """Last **measured** sighting. Not advanced while coasting."""

    last_updated: Instant

    age_frames: int = 0
    hit_count: int = 0
    coast_frames: int = 0

    since_measurement_ns: int = 0
    """Elapsed time since the last **measured** position, accumulated across
    missed frames.

    Load-bearing. A motion model extrapolates from the last observation, so
    predicting with only the *current* frame's elapsed leaves the gate anchored
    one step ahead of a position that is now many steps stale — and a moving
    object is never re-acquired. The bug is invisible on stationary objects,
    which is what makes it easy to ship."""

    association_confidence: float = 0.0
    association_method: AssociationMethod = AssociationMethod.REINITIALIZED
    association_cost: float = 0.0
    runner_up_cost: float | None = None
    gated_candidates: int = 0

    motion_state: MotionState = MotionState.UNKNOWN
    break_reason: BreakReason = BreakReason.NONE
    history: deque[FrameRef] = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_LENGTH))

    #: Consecutive frames whose measured displacement stayed below the motion
    #: floor. Hysteresis: a single still frame does not make a walking person
    #: "stationary", and a single jitter does not make a parked car "moving".
    still_frames: int = 0
    moving_frames: int = 0
    direction_changes: int = 0
    """Heading reversals in recent history — the signal for ``erratic``."""

    last_heading: float | None = None

    @property
    def is_measured_this_frame(self) -> bool:
        return self.coast_frames == 0


class TrackTable:
    """Bounded, single-writer track storage for one camera.

    Mints ``LocalTrackId`` from a counter that **never resets within an epoch**
    (port obligation T3). Reusing an id inside an epoch would let a consumer
    join two unrelated objects into one continuous history — a corruption that
    is invisible downstream and unrecoverable after the fact.
    """

    __slots__ = (
        "_camera_id",
        "_epoch",
        "_history_length",
        "_max_tracks",
        "_next_local_id",
        "_records",
        "_retired",
    )

    def __init__(
        self,
        camera_id: CameraId,
        *,
        epoch: TrackerEpoch = INITIAL_EPOCH,
        max_tracks: int = 256,
        history_length: int = DEFAULT_HISTORY_LENGTH,
    ) -> None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be >= 1")
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        self._camera_id = camera_id
        self._epoch = epoch
        self._max_tracks = max_tracks
        self._history_length = history_length
        self._records: dict[TrackId, TrackRecord] = {}
        self._next_local_id = 0
        self._retired: set[TrackId] = set()

    # --- identity ---------------------------------------------------------- #

    @property
    def camera_id(self) -> CameraId:
        return self._camera_id

    @property
    def epoch(self) -> TrackerEpoch:
        return self._epoch

    @property
    def capacity(self) -> int:
        return self._max_tracks

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, track_id: object) -> bool:
        return track_id in self._records

    # --- access ------------------------------------------------------------ #

    def records(self) -> tuple[TrackRecord, ...]:
        """Live records in stable id order.

        Ordering is deterministic rather than dict-insertion-dependent, because
        association indexes into this sequence and a reordering would change
        which track wins a tie (invariant V13).
        """
        return tuple(self._records[key] for key in sorted(self._records))

    def get(self, track_id: TrackId) -> TrackRecord | None:
        return self._records.get(track_id)

    # --- mutation ---------------------------------------------------------- #

    def create(
        self,
        *,
        class_id: ClassId,
        box: Box,
        predictor: MotionPredictorPort,
        now: Instant,
        frame_ref: FrameRef,
        tenant_id: TenantId,
        site_id: SiteId,
    ) -> TrackRecord:
        """Start a new tentative track.

        Raises:
            TrackerCapacityError: the table is full. Refusing is the bounded
                behaviour T8 requires; the alternative is a crowd scene growing
                memory without limit.
        """
        if len(self._records) >= self._max_tracks:
            raise TrackerCapacityError(
                f"camera {self._camera_id} already holds {len(self._records)} tracks "
                f"(max {self._max_tracks}); refusing new tracks keeps memory bounded",
                camera_id=str(self._camera_id),
                capacity=self._max_tracks,
            )

        track_id = TrackId(
            self._camera_id, self._epoch, LocalTrackId(self._next_local_id)
        )
        self._next_local_id += 1

        record = TrackRecord(
            track_id=track_id,
            class_id=class_id,
            box=box,
            state=TrackState.TENTATIVE,
            predictor=predictor,
            tenant_id=tenant_id,
            site_id=site_id,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            age_frames=1,
            hit_count=1,
            history=deque(maxlen=self._history_length),
        )
        record.history.append(frame_ref)
        self._records[track_id] = record
        return record

    def remove(self, track_id: TrackId) -> TrackRecord | None:
        """Retire a track. Its id is never reissued within this epoch."""
        record = self._records.pop(track_id, None)
        if record is not None:
            self._retired.add(track_id)
        return record

    def evict_weakest(self) -> TrackRecord | None:
        """Drop the least-defensible track to make room.

        Ordering: tentative before confirmed, then longest-coasting, then lowest
        association confidence, then oldest id. A tentative track has asserted
        nothing yet, so dropping it costs the least — and dropping *something*
        deterministically is better than either refusing all new tracks forever
        or growing without bound.
        """
        if not self._records:
            return None
        weakest = min(
            self._records.values(),
            key=lambda r: (
                r.state is not TrackState.TENTATIVE,
                -r.coast_frames,
                r.association_confidence,
                r.track_id.local_id,
            ),
        )
        return self.remove(weakest.track_id)

    def reset(self, epoch: TrackerEpoch) -> tuple[TrackRecord, ...]:
        """Discard everything and adopt a new epoch.

        Returns the discarded records so the caller can publish termination
        events — a silent reset would look downstream like every object
        simultaneously vanishing for no reason.
        """
        discarded = tuple(self._records[key] for key in sorted(self._records))
        self._records.clear()
        self._retired.clear()
        self._epoch = epoch
        self._next_local_id = 0
        return discarded

    def was_retired(self, track_id: TrackId) -> bool:
        """Whether this id belonged to a track that has since terminated."""
        return track_id in self._retired

    def stats(self) -> TableStats:
        by_state: dict[TrackState, int] = {}
        for record in self._records.values():
            by_state[record.state] = by_state.get(record.state, 0) + 1
        return TableStats(
            camera_id=self._camera_id,
            epoch=self._epoch,
            live=len(self._records),
            capacity=self._max_tracks,
            retired=len(self._retired),
            ids_issued=self._next_local_id,
            by_state=by_state,
        )


@dataclass(frozen=True, slots=True)
class TableStats:
    camera_id: CameraId
    epoch: TrackerEpoch
    live: int
    capacity: int
    retired: int
    ids_issued: int
    by_state: dict[TrackState, int] = field(default_factory=dict)

    @property
    def saturation(self) -> float:
        return self.live / self.capacity if self.capacity else 0.0
