"""The lifecycle machine — illegal transitions must be impossible.

The machine is the reason a resurrected terminated track cannot exist. That bug
produces a track which appears to teleport across the scene after an unrelated
object leaves: plausible output that no downstream consumer can detect as wrong,
which is exactly the class of failure a closed transition table eliminates.
"""

from __future__ import annotations

import itertools

import pytest

from vision_os.core.errors import IllegalTransitionError
from vision_os.core.model.track import BreakReason, TrackState
from vision_os.perception.tracking.lifecycle import (
    LifecycleMachine,
    LifecyclePolicy,
    TransitionReason,
    check_transition,
    is_legal,
)


@pytest.fixture
def policy() -> LifecyclePolicy:
    return LifecyclePolicy(
        min_hits_to_confirm=3,
        max_coast_frames=5,
        max_lost_frames=10,
        max_age_frames=1_000,
        max_tracks_per_camera=32,
    )


@pytest.fixture
def machine(policy: LifecyclePolicy) -> LifecycleMachine:
    return LifecycleMachine(policy)


class TestPolicyValidation:
    def test_a_valid_policy_constructs(self, policy: LifecyclePolicy) -> None:
        assert policy.min_hits_to_confirm == 3

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("min_hits_to_confirm", 0),
            ("max_coast_frames", -1),
            ("max_lost_frames", -1),
            ("max_age_frames", 0),
            ("max_tracks_per_camera", 0),
        ],
    )
    def test_invalid_bounds_are_refused(self, field: str, value: int) -> None:
        with pytest.raises(ValueError):
            LifecyclePolicy(**{field: value})

    def test_every_bound_is_finite(self, policy: LifecyclePolicy) -> None:
        """Track memory is short by construction; no bound may be disabled.

        An unbounded age is how tracking quietly becomes long-term memory, which
        is M7's job and not this module's.
        """
        for field in (
            "min_hits_to_confirm",
            "max_age_frames",
            "max_tracks_per_camera",
        ):
            assert getattr(policy, field) < float("inf")
            assert getattr(policy, field) >= 1


class TestTransitionTable:
    def test_terminated_is_final(self) -> None:
        """Nothing leaves TERMINATED. The resurrection bug, made impossible."""
        for state in TrackState:
            assert not is_legal(TrackState.TERMINATED, state), (
                f"terminated -> {state.value} must be illegal"
            )

    def test_tentative_cannot_become_lost(self) -> None:
        """An unconfirmed track never established continuity to recover."""
        assert not is_legal(TrackState.TENTATIVE, TrackState.LOST)

    def test_tentative_cannot_coast(self) -> None:
        assert not is_legal(TrackState.TENTATIVE, TrackState.COASTING)

    def test_lost_cannot_return_to_coasting(self) -> None:
        """Recovery is a return to measurement; a track cannot be re-lost first."""
        assert not is_legal(TrackState.LOST, TrackState.COASTING)

    def test_confirmed_cannot_regress_to_tentative(self) -> None:
        assert not is_legal(TrackState.CONFIRMED, TrackState.TENTATIVE)

    def test_lost_can_be_recovered(self) -> None:
        assert is_legal(TrackState.LOST, TrackState.CONFIRMED)

    def test_coasting_can_be_recovered(self) -> None:
        assert is_legal(TrackState.COASTING, TrackState.CONFIRMED)

    def test_every_state_can_terminate_except_terminated(self) -> None:
        for state in TrackState:
            if state is TrackState.TERMINATED:
                continue
            assert is_legal(state, TrackState.TERMINATED)

    def test_check_transition_raises_on_an_illegal_edge(self) -> None:
        with pytest.raises(IllegalTransitionError, match="illegal track transition"):
            check_transition(TrackState.TERMINATED, TrackState.CONFIRMED)

    def test_check_transition_is_silent_on_a_legal_edge(self) -> None:
        check_transition(TrackState.TENTATIVE, TrackState.CONFIRMED)

    def test_the_table_is_closed(self) -> None:
        """Every pair is either explicitly allowed or explicitly rejected."""
        for previous, current in itertools.product(TrackState, TrackState):
            legal = is_legal(previous, current)
            assert isinstance(legal, bool)
            if not legal:
                with pytest.raises(IllegalTransitionError):
                    check_transition(previous, current)


class TestConfirmation:
    def test_a_track_confirms_at_the_hit_threshold(self, machine: LifecycleMachine) -> None:
        transition = machine.on_hit(state=TrackState.TENTATIVE, hit_count=3, age_frames=3)
        assert transition.current is TrackState.CONFIRMED
        assert transition.reason is TransitionReason.CONFIRMED

    def test_a_track_stays_tentative_below_the_threshold(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.on_hit(state=TrackState.TENTATIVE, hit_count=2, age_frames=2)
        assert transition.current is TrackState.TENTATIVE
        assert transition.reason is TransitionReason.HIT

    def test_confirmation_is_immediate_not_one_frame_late(self) -> None:
        """Counters are post-increment, so the confirming frame confirms."""
        machine = LifecycleMachine(LifecyclePolicy(min_hits_to_confirm=1))
        transition = machine.on_hit(state=TrackState.TENTATIVE, hit_count=1, age_frames=1)
        assert transition.current is TrackState.CONFIRMED

    def test_a_confirmed_track_stays_confirmed_on_a_hit(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.on_hit(state=TrackState.CONFIRMED, hit_count=9, age_frames=9)
        assert transition.current is TrackState.CONFIRMED
        assert not transition.is_recovery


class TestMissHandling:
    def test_an_unconfirmed_track_is_dropped_on_a_miss(
        self, machine: LifecycleMachine
    ) -> None:
        """Probably a detector false positive; holding it would let it compete
        for associations it has no claim to."""
        transition = machine.on_miss(
            state=TrackState.TENTATIVE, coast_frames=1, age_frames=2
        )
        assert transition.current is TrackState.TERMINATED
        assert transition.is_terminal

    def test_a_confirmed_track_coasts_on_a_miss(self, machine: LifecycleMachine) -> None:
        transition = machine.on_miss(
            state=TrackState.CONFIRMED, coast_frames=1, age_frames=10
        )
        assert transition.current is TrackState.COASTING
        assert transition.break_reason is BreakReason.DETECTOR_MISS

    def test_coasting_continues_within_budget(self, machine: LifecycleMachine) -> None:
        transition = machine.on_miss(
            state=TrackState.COASTING, coast_frames=3, age_frames=13
        )
        assert transition.current is TrackState.COASTING

    def test_coasting_becomes_lost_past_budget(self, machine: LifecycleMachine) -> None:
        transition = machine.on_miss(
            state=TrackState.COASTING, coast_frames=6, age_frames=16
        )
        assert transition.current is TrackState.LOST
        assert transition.reason is TransitionReason.COAST_EXCEEDED

    def test_lost_terminates_after_the_recovery_window(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.on_miss(
            state=TrackState.LOST, coast_frames=5 + 10 + 1, age_frames=40
        )
        assert transition.current is TrackState.TERMINATED
        assert transition.reason is TransitionReason.LOST_WINDOW_EXPIRED

    def test_lost_survives_inside_the_recovery_window(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.on_miss(
            state=TrackState.LOST, coast_frames=5 + 3, age_frames=20
        )
        assert transition.current is TrackState.LOST

    def test_a_break_reason_travels_with_the_transition(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.on_miss(
            state=TrackState.CONFIRMED,
            coast_frames=1,
            age_frames=10,
            break_reason=BreakReason.OCCLUSION,
        )
        assert transition.break_reason is BreakReason.OCCLUSION


class TestRecovery:
    def test_a_coasting_track_recovers_on_a_hit(self, machine: LifecycleMachine) -> None:
        transition = machine.on_hit(state=TrackState.COASTING, hit_count=8, age_frames=12)
        assert transition.current is TrackState.CONFIRMED
        assert transition.is_recovery

    def test_a_lost_track_recovers_on_a_hit(self, machine: LifecycleMachine) -> None:
        transition = machine.on_hit(state=TrackState.LOST, hit_count=8, age_frames=20)
        assert transition.current is TrackState.CONFIRMED
        assert transition.is_recovery

    def test_recovery_clears_the_break_reason(self, machine: LifecycleMachine) -> None:
        """A recovered track is not broken; a stale reason would mislead."""
        transition = machine.on_hit(state=TrackState.COASTING, hit_count=8, age_frames=12)
        assert transition.break_reason is BreakReason.NONE

    def test_recovery_is_a_transition_not_a_state(self) -> None:
        """It is fully expressible as coasting|lost -> confirmed."""
        assert "recovered" not in {s.value for s in TrackState}
        assert TransitionReason.RECOVERED in set(TransitionReason)


class TestBoundsAreEnforced:
    def test_max_age_terminates_even_a_healthy_track(self) -> None:
        """A stuck association is far likelier than a genuinely eternal object."""
        machine = LifecycleMachine(LifecyclePolicy(max_age_frames=10))
        transition = machine.on_hit(state=TrackState.CONFIRMED, hit_count=10, age_frames=10)
        assert transition.current is TrackState.TERMINATED
        assert transition.reason is TransitionReason.MAX_AGE_EXCEEDED

    def test_max_age_terminates_on_a_miss_too(self) -> None:
        machine = LifecycleMachine(LifecyclePolicy(max_age_frames=10))
        transition = machine.on_miss(
            state=TrackState.COASTING, coast_frames=2, age_frames=10
        )
        assert transition.current is TrackState.TERMINATED

    def test_zero_coast_budget_terminates_immediately(self) -> None:
        """A deployment that wants no coasting gets none — no hidden minimum."""
        machine = LifecycleMachine(LifecyclePolicy(max_coast_frames=0))
        transition = machine.on_miss(
            state=TrackState.CONFIRMED, coast_frames=1, age_frames=5
        )
        assert transition.current is TrackState.TERMINATED

    def test_zero_lost_window_terminates_at_the_coast_boundary(self) -> None:
        machine = LifecycleMachine(
            LifecyclePolicy(max_coast_frames=2, max_lost_frames=0)
        )
        transition = machine.on_miss(
            state=TrackState.COASTING, coast_frames=3, age_frames=9
        )
        assert transition.current is TrackState.TERMINATED


class TestTerminatedTracksAreInert:
    def test_a_terminated_track_cannot_be_hit(self, machine: LifecycleMachine) -> None:
        with pytest.raises(IllegalTransitionError, match="cannot be associated"):
            machine.on_hit(state=TrackState.TERMINATED, hit_count=1, age_frames=1)

    def test_a_terminated_track_cannot_miss(self, machine: LifecycleMachine) -> None:
        with pytest.raises(IllegalTransitionError, match="cannot miss"):
            machine.on_miss(state=TrackState.TERMINATED, coast_frames=1, age_frames=1)

    def test_a_terminated_track_cannot_be_terminated_again(
        self, machine: LifecycleMachine
    ) -> None:
        with pytest.raises(IllegalTransitionError, match="again"):
            machine.terminate(
                state=TrackState.TERMINATED,
                reason=TransitionReason.LEFT_FRAME,
                break_reason=BreakReason.EXIT,
            )


class TestForcedTermination:
    def test_exit_terminates_with_its_reason(self, machine: LifecycleMachine) -> None:
        transition = machine.terminate(
            state=TrackState.CONFIRMED,
            reason=TransitionReason.LEFT_FRAME,
            break_reason=BreakReason.EXIT,
        )
        assert transition.current is TrackState.TERMINATED
        assert transition.break_reason is BreakReason.EXIT

    def test_epoch_reset_terminates_with_its_reason(
        self, machine: LifecycleMachine
    ) -> None:
        transition = machine.terminate(
            state=TrackState.COASTING,
            reason=TransitionReason.EPOCH_RESET,
            break_reason=BreakReason.EPOCH_RESET,
        )
        assert transition.break_reason is BreakReason.EPOCH_RESET

    def test_ambiguous_association_can_terminate_a_track(
        self, machine: LifecycleMachine
    ) -> None:
        """M6: prefer terminating a track over a wrong association."""
        transition = machine.terminate(
            state=TrackState.CONFIRMED,
            reason=TransitionReason.AMBIGUOUS_ASSOCIATION,
            break_reason=BreakReason.ASSOCIATION_FAILURE,
        )
        assert transition.is_terminal


class TestTransitionRecord:
    def test_changed_detects_a_real_move(self, machine: LifecycleMachine) -> None:
        moved = machine.on_hit(state=TrackState.TENTATIVE, hit_count=3, age_frames=3)
        stayed = machine.on_hit(state=TrackState.CONFIRMED, hit_count=9, age_frames=9)
        assert moved.changed
        assert not stayed.changed

    def test_every_transition_names_its_reason(self, machine: LifecycleMachine) -> None:
        """Every transition must be observable, so none may be unexplained."""
        transitions = [
            machine.on_hit(state=TrackState.TENTATIVE, hit_count=1, age_frames=1),
            machine.on_hit(state=TrackState.TENTATIVE, hit_count=3, age_frames=3),
            machine.on_miss(state=TrackState.CONFIRMED, coast_frames=1, age_frames=5),
            machine.on_miss(state=TrackState.COASTING, coast_frames=6, age_frames=11),
            machine.on_hit(state=TrackState.LOST, hit_count=4, age_frames=15),
        ]
        for transition in transitions:
            assert transition.reason in set(TransitionReason)


class TestFullLifecycleWalk:
    def test_the_documented_path_is_walkable(self, machine: LifecycleMachine) -> None:
        """tentative -> confirmed -> coasting -> lost -> terminated (M6 R2)."""
        state = TrackState.TENTATIVE
        visited = [state]

        for hit in range(1, 4):
            state = machine.on_hit(state=state, hit_count=hit, age_frames=hit).current
        visited.append(state)

        coast = 0
        for _ in range(6):
            coast += 1
            state = machine.on_miss(
                state=state, coast_frames=coast, age_frames=3 + coast
            ).current
            if state is TrackState.COASTING and TrackState.COASTING not in visited:
                visited.append(state)
            if state is TrackState.LOST and TrackState.LOST not in visited:
                visited.append(state)

        for _ in range(20):
            coast += 1
            state = machine.on_miss(
                state=state, coast_frames=coast, age_frames=3 + coast
            ).current
            if state is TrackState.TERMINATED:
                visited.append(state)
                break

        assert visited == [
            TrackState.TENTATIVE,
            TrackState.CONFIRMED,
            TrackState.COASTING,
            TrackState.LOST,
            TrackState.TERMINATED,
        ]
