"""The object lifecycle — a closed state machine (02_VOM section 10.6).

> **Single responsibility:** *Decide what lifecycle state an object is in next.
> Bind nothing, own no objects, compute no geometry.*

The transition table is transcribed directly from the state diagram in 02_VOM
section 10.6, edge for edge. It is a table rather than scattered conditionals for
the same reason the track lifecycle is: **illegal transitions must be impossible,
not merely unlikely.** An object that resurrects from ``expired``, or that departs
and comes back without passing through re-entry, corrupts every count derived from
it — and the corruption is invisible downstream.

Read the absences as carefully as the presences. The diagram permits merging from
``active`` and ``dormant`` only: an ``occluded`` object is mid-claim and must
resolve to active or dormant before its identity can be revised, and a
``departed`` object has left, so a merge would be asserting a continuity nobody
observed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ...core.errors import IllegalTransitionError
from ...core.model.timebase import Duration
from ...core.model.visual_object import LifecycleState


class LifecycleTrigger(enum.Enum):
    """Why the machine moved. Reported with every transition."""

    FIRST_DETECTIONS = "first_detections"
    CONFIRMATION_MET = "confirmation_threshold_met"
    NEVER_CONFIRMED = "never_confirmed"
    MEASUREMENT_LOST = "measurement_lost"
    REASSOCIATED = "re_association_succeeded"
    OCCLUSION_HORIZON = "occlusion_exceeded_horizon"
    LEFT_FIELD_OF_VIEW = "left_field_of_view"
    RE_ENTRY_MATCHED = "re_entry_matched"
    DEPARTURE_HORIZON = "departure_horizon_exceeded"
    IDENTITY_RESOLUTION = "identity_resolution"
    RETENTION_HORIZON = "retention_horizon"
    POPULATION_SHED = "population_shed"
    """Evicted to keep the per-camera population capped. Only ``provisional``
    objects are shed, which is why this reaches ``expired`` from there alone."""


#: The closed transition table, transcribed from the 02_VOM section 10.6 diagram.
#:
#: Every entry maps to exactly one edge in that diagram. Self-transitions are
#: permitted where a state persists across frames; they are not in the diagram
#: because a diagram shows changes, but "stay occluded another frame" is a real
#: outcome the caller needs to express without a special case.
_ALLOWED: frozenset[tuple[LifecycleState, LifecycleState]] = frozenset(
    {
        # birth
        (LifecycleState.PROVISIONAL, LifecycleState.PROVISIONAL),
        (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE),
        (LifecycleState.PROVISIONAL, LifecycleState.EXPIRED),
        # observed
        (LifecycleState.ACTIVE, LifecycleState.ACTIVE),
        (LifecycleState.ACTIVE, LifecycleState.OCCLUDED),
        (LifecycleState.ACTIVE, LifecycleState.DORMANT),
        (LifecycleState.ACTIVE, LifecycleState.MERGED_INTO),
        # believed present, unmeasured
        (LifecycleState.OCCLUDED, LifecycleState.OCCLUDED),
        (LifecycleState.OCCLUDED, LifecycleState.ACTIVE),
        (LifecycleState.OCCLUDED, LifecycleState.DORMANT),
        # retained for re-entry
        (LifecycleState.DORMANT, LifecycleState.DORMANT),
        (LifecycleState.DORMANT, LifecycleState.ACTIVE),
        (LifecycleState.DORMANT, LifecycleState.DEPARTED),
        (LifecycleState.DORMANT, LifecycleState.MERGED_INTO),
        # gone
        (LifecycleState.DEPARTED, LifecycleState.DEPARTED),
        (LifecycleState.DEPARTED, LifecycleState.EXPIRED),
    }
)


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    """The horizons that drive the machine. Strongly typed, validated on build.

    Every horizon is finite. An object retained forever is the runaway registry
    section M7 warns about — *"a memory leak with a face"*.
    """

    min_observations_to_confirm: int = 3
    """Sightings before a provisional object is asserted as real. Below this it
    is probably tracker noise, and confirming it would put a phantom into the
    platform's first durable state."""

    provisional_horizon: Duration = Duration.from_millis(3_000)
    """How long a provisional object may fail to confirm before expiring."""

    occlusion_horizon: Duration = Duration.from_millis(10_000)
    """Believed-present without measurement. Past this the claim weakens from
    "occluded" to "dormant" — still retained, no longer asserted present."""

    dormant_horizon: Duration = Duration.from_millis(120_000)
    """Retained for re-entry. Past this the object is believed departed."""

    retention_horizon: Duration = Duration.from_millis(600_000)
    """How long a departed object is kept before expiry. This is the bound on
    the registry's memory, and it is why the population cannot grow without
    limit even under sustained churn."""

    max_objects_per_camera: int = 512
    """Cap per partition. A crowd degrades by shedding provisional objects and
    alarming, never by growing (section M7 failure handling)."""

    def __post_init__(self) -> None:
        if self.min_observations_to_confirm < 1:
            raise ValueError("min_observations_to_confirm must be >= 1")
        for name in (
            "provisional_horizon",
            "occlusion_horizon",
            "dormant_horizon",
            "retention_horizon",
        ):
            horizon: Duration = getattr(self, name)
            if horizon.ns <= 0:
                raise ValueError(f"{name} must be positive; every horizon is finite")
        if self.max_objects_per_camera < 1:
            raise ValueError("max_objects_per_camera must be >= 1")


@dataclass(frozen=True, slots=True)
class LifecycleTransition:
    """One state change, with everything needed to explain it."""

    previous: LifecycleState
    current: LifecycleState
    trigger: LifecycleTrigger

    @property
    def changed(self) -> bool:
        return self.previous is not self.current

    @property
    def is_terminal(self) -> bool:
        return self.current.is_terminal


def is_legal(previous: LifecycleState, current: LifecycleState) -> bool:
    """Whether the machine permits this edge."""
    return (previous, current) in _ALLOWED


def check_transition(previous: LifecycleState, current: LifecycleState) -> None:
    """Raise unless the edge is legal.

    Raises:
        IllegalTransitionError: always, for an edge outside the table. The
            lifecycle is closed, so an illegal edge means a caller is
            constructing state incorrectly rather than encountering bad input.
    """
    if not is_legal(previous, current):
        raise IllegalTransitionError(
            f"illegal object lifecycle transition {previous.value} -> "
            f"{current.value}; the lifecycle is a closed machine "
            f"(02_VOM section 10.6)",
            previous=previous.value,
            current=current.value,
        )


class ObjectLifecycleMachine:
    """Computes the next lifecycle state. Holds no object state of its own.

    Deliberately stateless: it receives the counters and elapsed times and
    returns a transition, so it is exhaustively testable without constructing a
    registry, and two partitions cannot disagree about what "confirmed" means.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: LifecyclePolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> LifecyclePolicy:
        return self._policy

    def on_measured(
        self, *, state: LifecycleState, observation_count: int
    ) -> LifecycleTransition:
        """The object was measured this frame.

        ``observation_count`` is the value **after** counting this sighting, so
        an object whose confirming observation arrives is confirmed immediately
        rather than one frame late.
        """
        self._reject_terminal(state, "measured")

        if state is LifecycleState.PROVISIONAL:
            if observation_count >= self._policy.min_observations_to_confirm:
                return self._to(state, LifecycleState.ACTIVE, LifecycleTrigger.CONFIRMATION_MET)
            return self._to(
                state, LifecycleState.PROVISIONAL, LifecycleTrigger.FIRST_DETECTIONS
            )

        if state is LifecycleState.OCCLUDED:
            return self._to(state, LifecycleState.ACTIVE, LifecycleTrigger.REASSOCIATED)

        if state is LifecycleState.DORMANT:
            return self._to(state, LifecycleState.ACTIVE, LifecycleTrigger.RE_ENTRY_MATCHED)

        if state is LifecycleState.DEPARTED:
            # The diagram has no departed -> active edge. A departed object that
            # reappears is a *new* object plus an identity assertion linking the
            # two, which is the registry's job and not a lifecycle edge.
            raise IllegalTransitionError(
                "a departed object cannot be re-measured; re-entry after departure "
                "mints a new object and emits an identity assertion linking them "
                "(02_VOM section 4.2)",
                previous=state.value,
                current="measured",
            )

        return self._to(state, LifecycleState.ACTIVE, LifecycleTrigger.CONFIRMATION_MET)

    def on_unmeasured(
        self,
        *,
        state: LifecycleState,
        since_confirmed: Duration,
        left_field_of_view: bool = False,
    ) -> LifecycleTransition:
        """No measurement was associated with this object this frame.

        ``left_field_of_view`` distinguishes "the object walked out of frame"
        from "the object stopped being measurable where it was". The diagram has
        separate edges for these — ``active -> dormant`` and ``active ->
        occluded`` — because they are different claims and a consumer that
        cannot tell them apart cannot reason about egress.
        """
        self._reject_terminal(state, "unmeasured")
        policy = self._policy

        if state is LifecycleState.PROVISIONAL:
            if since_confirmed.ns >= policy.provisional_horizon.ns:
                return self._to(
                    state, LifecycleState.EXPIRED, LifecycleTrigger.NEVER_CONFIRMED
                )
            return self._to(
                state, LifecycleState.PROVISIONAL, LifecycleTrigger.FIRST_DETECTIONS
            )

        if state is LifecycleState.ACTIVE:
            if left_field_of_view:
                return self._to(
                    state, LifecycleState.DORMANT, LifecycleTrigger.LEFT_FIELD_OF_VIEW
                )
            return self._to(
                state, LifecycleState.OCCLUDED, LifecycleTrigger.MEASUREMENT_LOST
            )

        if state is LifecycleState.OCCLUDED:
            if since_confirmed.ns >= policy.occlusion_horizon.ns:
                return self._to(
                    state, LifecycleState.DORMANT, LifecycleTrigger.OCCLUSION_HORIZON
                )
            return self._to(
                state, LifecycleState.OCCLUDED, LifecycleTrigger.MEASUREMENT_LOST
            )

        if state is LifecycleState.DORMANT:
            if since_confirmed.ns >= policy.dormant_horizon.ns:
                return self._to(
                    state, LifecycleState.DEPARTED, LifecycleTrigger.DEPARTURE_HORIZON
                )
            return self._to(state, LifecycleState.DORMANT, LifecycleTrigger.OCCLUSION_HORIZON)

        # DEPARTED
        if since_confirmed.ns >= policy.retention_horizon.ns:
            return self._to(state, LifecycleState.EXPIRED, LifecycleTrigger.RETENTION_HORIZON)
        return self._to(state, LifecycleState.DEPARTED, LifecycleTrigger.DEPARTURE_HORIZON)

    def on_merged(self, *, state: LifecycleState) -> LifecycleTransition:
        """Identity resolution merged this object into another.

        Legal from ``active`` and ``dormant`` only, per the diagram. An
        ``occluded`` object is mid-claim; a ``departed`` one has left, and
        merging it would assert a continuity nobody observed.
        """
        self._reject_terminal(state, "merge")
        return self._to(
            state, LifecycleState.MERGED_INTO, LifecycleTrigger.IDENTITY_RESOLUTION
        )

    def on_shed(self, *, state: LifecycleState) -> LifecycleTransition:
        """Evicted under population pressure.

        Only ``provisional`` objects are shed. A confirmed object has been
        asserted to consumers, and withdrawing that assertion to save memory
        would make the platform's claims a function of its load.
        """
        if state is not LifecycleState.PROVISIONAL:
            raise IllegalTransitionError(
                f"only provisional objects may be shed under population pressure; "
                f"{state.value} has been asserted to consumers and withdrawing it "
                f"would make the platform's claims a function of its memory usage",
                previous=state.value,
                current=LifecycleState.EXPIRED.value,
            )
        return self._to(state, LifecycleState.EXPIRED, LifecycleTrigger.POPULATION_SHED)

    # --- internals ------------------------------------------------------------ #

    def _reject_terminal(self, state: LifecycleState, action: str) -> None:
        if state.is_terminal:
            raise IllegalTransitionError(
                f"a {state.value} object cannot be {action}; terminal states are "
                f"final and the record is retained only so history stays "
                f"resolvable (V5)",
                previous=state.value,
                current=action,
            )

    def _to(
        self,
        previous: LifecycleState,
        current: LifecycleState,
        trigger: LifecycleTrigger,
    ) -> LifecycleTransition:
        check_transition(previous, current)
        return LifecycleTransition(previous=previous, current=current, trigger=trigger)
