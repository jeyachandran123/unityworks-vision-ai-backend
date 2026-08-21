"""The track lifecycle — a closed state machine (03_MODULES M6 R2).

> **Single responsibility:** *Decide what state a track is in next. Associate
> nothing, predict nothing, own no tracks.*

The machine is defined as an explicit transition table rather than as scattered
``if`` statements, for one reason: **illegal transitions must be impossible, not
merely unlikely**. A resurrected terminated track is the kind of bug that
produces a track which appears to teleport across the scene after an unrelated
object leaves — plausible-looking output that no downstream consumer can detect
as wrong.

Every transition is observable. The caller receives the reason alongside the new
state, so the event stream explains itself without the caller re-deriving intent.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ...core.errors import IllegalTransitionError
from ...core.model.track import BreakReason, TrackState


class TransitionReason(enum.Enum):
    """Why the machine moved. Reported with every transition."""

    CREATED = "created"
    HIT = "hit"
    """Associated with a detection this frame."""

    CONFIRMED = "confirmed"
    """Reached the minimum-hits bar."""

    MISS = "miss"
    """No detection associated this frame."""

    RECOVERED = "recovered"
    """Re-associated after coasting or being lost."""

    COAST_EXCEEDED = "coast_exceeded"
    LOST_WINDOW_EXPIRED = "lost_window_expired"
    MAX_AGE_EXCEEDED = "max_age_exceeded"
    LEFT_FRAME = "left_frame"
    EPOCH_RESET = "epoch_reset"
    CAPACITY_EVICTED = "capacity_evicted"
    """Evicted to keep the track table bounded (port obligation T8)."""

    AMBIGUOUS_ASSOCIATION = "ambiguous_association"
    """Terminated rather than risk a wrong association (03_MODULES M6: *prefer
    terminating a track over a wrong association*)."""


#: The closed transition table. A pair absent from this set cannot occur.
#:
#: Read the absences as carefully as the presences:
#:
#: * ``TERMINATED`` appears only as a destination. It is final; nothing leaves.
#: * ``TENTATIVE -> LOST`` is absent. An unconfirmed track that stops being seen
#:   was probably a detector false positive and is dropped outright rather than
#:   held in a recovery window, where it would compete for association and
#:   invent continuity that was never established.
#: * ``LOST -> COASTING`` is absent. Recovery is a return to measurement; a
#:   track cannot be re-lost without first being re-seen.
_ALLOWED: frozenset[tuple[TrackState, TrackState]] = frozenset(
    {
        # birth
        (TrackState.TENTATIVE, TrackState.TENTATIVE),
        (TrackState.TENTATIVE, TrackState.CONFIRMED),
        (TrackState.TENTATIVE, TrackState.TERMINATED),
        # steady state
        (TrackState.CONFIRMED, TrackState.CONFIRMED),
        (TrackState.CONFIRMED, TrackState.COASTING),
        (TrackState.CONFIRMED, TrackState.TERMINATED),
        # gap handling
        (TrackState.COASTING, TrackState.CONFIRMED),
        (TrackState.COASTING, TrackState.COASTING),
        (TrackState.COASTING, TrackState.LOST),
        (TrackState.COASTING, TrackState.TERMINATED),
        # recovery window
        (TrackState.LOST, TrackState.CONFIRMED),
        (TrackState.LOST, TrackState.LOST),
        (TrackState.LOST, TrackState.TERMINATED),
    }
)


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    """Thresholds governing the machine. Strongly typed, validated on build.

    These are the whole of "track memory" (03_MODULES M6 state ownership).
    Tracking owns short temporal continuity and must never become long-term
    memory, so every bound here is finite and none may be disabled.
    """

    min_hits_to_confirm: int = 3
    """Frames a track must be measured before it is asserted. The direct
    trade-off between reacting to real objects and promoting detector noise."""

    max_coast_frames: int = 5
    """Consecutive predicted frames before the track is considered lost."""

    max_lost_frames: int = 15
    """Recovery window. After this the track terminates and its id is retired
    for the epoch."""

    max_age_frames: int = 36_000
    """Absolute ceiling regardless of health. A track that has lived two hours
    on a 5 fps camera is far more likely to be a stuck association than a
    genuinely persistent object, and an unbounded age is how tracking quietly
    becomes long-term memory."""

    max_tracks_per_camera: int = 256
    """Bounded track table (T8). A crowd degrades by refusing new tracks."""

    def __post_init__(self) -> None:
        if self.min_hits_to_confirm < 1:
            raise ValueError("min_hits_to_confirm must be >= 1")
        if self.max_coast_frames < 0:
            raise ValueError("max_coast_frames must be >= 0")
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must be >= 0")
        if self.max_age_frames < 1:
            raise ValueError("max_age_frames must be >= 1")
        if self.max_tracks_per_camera < 1:
            raise ValueError("max_tracks_per_camera must be >= 1")


@dataclass(frozen=True, slots=True)
class Transition:
    """One state change, with everything needed to explain it."""

    previous: TrackState
    current: TrackState
    reason: TransitionReason
    break_reason: BreakReason = BreakReason.NONE

    @property
    def changed(self) -> bool:
        return self.previous is not self.current

    @property
    def is_recovery(self) -> bool:
        return self.reason is TransitionReason.RECOVERED

    @property
    def is_terminal(self) -> bool:
        return self.current is TrackState.TERMINATED


def is_legal(previous: TrackState, current: TrackState) -> bool:
    """Whether the machine permits this edge."""
    return (previous, current) in _ALLOWED


def check_transition(previous: TrackState, current: TrackState) -> None:
    """Raise unless the edge is legal.

    Raises:
        IllegalTransitionError: always, for an edge outside the table. This is
            byzantine rather than transient — the lifecycle is closed, so an
            illegal edge means a tracker is constructing state incorrectly.
    """
    if not is_legal(previous, current):
        raise IllegalTransitionError(
            f"illegal track transition {previous.value} -> {current.value}; "
            f"the lifecycle is a closed machine (03_MODULES M6 R2)",
            previous=previous.value,
            current=current.value,
        )


class LifecycleMachine:
    """Computes the next state. Holds no track state of its own.

    Deliberately stateless: it receives the counters and returns a transition,
    so it is exhaustively testable without constructing a tracker, and two
    trackers cannot disagree about what "confirmed" means.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: LifecyclePolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> LifecyclePolicy:
        return self._policy

    def on_hit(
        self, *, state: TrackState, hit_count: int, age_frames: int
    ) -> Transition:
        """A detection was associated with this track.

        ``hit_count`` and ``age_frames`` are the values **after** counting this
        frame, so a track whose first hit satisfies ``min_hits_to_confirm`` is
        confirmed immediately rather than one frame late.
        """
        if state is TrackState.TERMINATED:
            raise IllegalTransitionError(
                "a terminated track cannot be associated; ids are retired for the epoch",
                previous=state.value,
                current="hit",
            )

        if age_frames >= self._policy.max_age_frames:
            return self._to(state, TrackState.TERMINATED, TransitionReason.MAX_AGE_EXCEEDED)

        was_absent = state in (TrackState.COASTING, TrackState.LOST)
        if was_absent:
            return self._to(state, TrackState.CONFIRMED, TransitionReason.RECOVERED)

        if state is TrackState.TENTATIVE:
            if hit_count >= self._policy.min_hits_to_confirm:
                return self._to(state, TrackState.CONFIRMED, TransitionReason.CONFIRMED)
            return self._to(state, TrackState.TENTATIVE, TransitionReason.HIT)

        return self._to(state, TrackState.CONFIRMED, TransitionReason.HIT)

    def on_miss(
        self,
        *,
        state: TrackState,
        coast_frames: int,
        age_frames: int,
        break_reason: BreakReason = BreakReason.DETECTOR_MISS,
    ) -> Transition:
        """No detection was associated with this track this frame.

        ``coast_frames`` is the value **after** counting this frame.

        A detection gap is a **normal operating condition**, not an exception
        path (03_MODULES M6 R6): the scheduler drops frames by design, so this
        is the common case rather than the error case.
        """
        if state is TrackState.TERMINATED:
            raise IllegalTransitionError(
                "a terminated track cannot miss; it no longer exists",
                previous=state.value,
                current="miss",
            )

        if age_frames >= self._policy.max_age_frames:
            return self._to(
                state, TrackState.TERMINATED, TransitionReason.MAX_AGE_EXCEEDED, break_reason
            )

        # An unconfirmed track that stops being seen is dropped, not held. It
        # never established continuity, so a recovery window would let it
        # compete for associations it has no claim to.
        if state is TrackState.TENTATIVE:
            return self._to(
                state, TrackState.TERMINATED, TransitionReason.MISS, break_reason
            )

        if state is TrackState.CONFIRMED:
            if self._policy.max_coast_frames == 0:
                return self._to(
                    state,
                    TrackState.TERMINATED,
                    TransitionReason.COAST_EXCEEDED,
                    break_reason,
                )
            return self._to(state, TrackState.COASTING, TransitionReason.MISS, break_reason)

        if state is TrackState.COASTING:
            if coast_frames > self._policy.max_coast_frames:
                if self._policy.max_lost_frames == 0:
                    return self._to(
                        state,
                        TrackState.TERMINATED,
                        TransitionReason.COAST_EXCEEDED,
                        break_reason,
                    )
                return self._to(
                    state, TrackState.LOST, TransitionReason.COAST_EXCEEDED, break_reason
                )
            return self._to(state, TrackState.COASTING, TransitionReason.MISS, break_reason)

        # LOST
        lost_frames = coast_frames - self._policy.max_coast_frames
        if lost_frames > self._policy.max_lost_frames:
            return self._to(
                state,
                TrackState.TERMINATED,
                TransitionReason.LOST_WINDOW_EXPIRED,
                break_reason,
            )
        return self._to(state, TrackState.LOST, TransitionReason.MISS, break_reason)

    def terminate(
        self, *, state: TrackState, reason: TransitionReason, break_reason: BreakReason
    ) -> Transition:
        """Force termination — exit, reset, eviction, or refused association."""
        if state is TrackState.TERMINATED:
            raise IllegalTransitionError(
                "a terminated track cannot be terminated again",
                previous=state.value,
                current=TrackState.TERMINATED.value,
            )
        return self._to(state, TrackState.TERMINATED, reason, break_reason)

    def _to(
        self,
        previous: TrackState,
        current: TrackState,
        reason: TransitionReason,
        break_reason: BreakReason = BreakReason.NONE,
    ) -> Transition:
        check_transition(previous, current)
        return Transition(
            previous=previous,
            current=current,
            reason=reason,
            break_reason=break_reason if current is not TrackState.CONFIRMED else BreakReason.NONE,
        )
