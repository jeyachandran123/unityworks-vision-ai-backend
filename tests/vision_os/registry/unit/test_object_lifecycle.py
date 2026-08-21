"""The object lifecycle machine — the 02_VOM section 10.6 state diagram.

The transition table is transcribed edge-for-edge from that diagram, so these
tests read it back the same way: every permitted edge is exercised, and every
edge *absent* from the diagram is asserted illegal.

The absences carry the design. An object that resurrects from ``expired``, or
that merges while occluded, corrupts every count derived from it — and the
corruption is invisible downstream, which is exactly why a closed table beats
scattered conditionals.
"""

from __future__ import annotations

import itertools

import pytest

from vision_os.core.errors import IllegalTransitionError
from vision_os.core.model.timebase import Duration
from vision_os.core.model.visual_object import LifecycleState
from vision_os.perception.registry.lifecycle import (
    LifecyclePolicy,
    LifecycleTrigger,
    ObjectLifecycleMachine,
    check_transition,
    is_legal,
)


@pytest.fixture
def policy() -> LifecyclePolicy:
    return LifecyclePolicy(
        min_observations_to_confirm=3,
        provisional_horizon=Duration.from_millis(2_000),
        occlusion_horizon=Duration.from_millis(4_000),
        dormant_horizon=Duration.from_millis(10_000),
        retention_horizon=Duration.from_millis(30_000),
        max_objects_per_camera=32,
    )


@pytest.fixture
def machine(policy: LifecyclePolicy) -> ObjectLifecycleMachine:
    return ObjectLifecycleMachine(policy)


class TestPolicyValidation:
    def test_a_valid_policy_constructs(self, policy) -> None:
        assert policy.min_observations_to_confirm == 3

    def test_zero_confirmations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="min_observations_to_confirm"):
            LifecyclePolicy(min_observations_to_confirm=0)

    @pytest.mark.parametrize(
        "field",
        ["provisional_horizon", "occlusion_horizon", "dormant_horizon", "retention_horizon"],
    )
    def test_every_horizon_must_be_positive(self, field: str) -> None:
        """A retained-forever object is the runaway registry section M7 warns of."""
        with pytest.raises(ValueError, match=field):
            LifecyclePolicy(**{field: Duration(0)})

    def test_zero_capacity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_objects_per_camera"):
            LifecyclePolicy(max_objects_per_camera=0)


class TestTransitionTable:
    """Read back from the 02_VOM section 10.6 diagram."""

    @pytest.mark.parametrize(
        ("previous", "current"),
        [
            (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE),
            (LifecycleState.PROVISIONAL, LifecycleState.EXPIRED),
            (LifecycleState.ACTIVE, LifecycleState.OCCLUDED),
            (LifecycleState.OCCLUDED, LifecycleState.ACTIVE),
            (LifecycleState.OCCLUDED, LifecycleState.DORMANT),
            (LifecycleState.ACTIVE, LifecycleState.DORMANT),
            (LifecycleState.DORMANT, LifecycleState.ACTIVE),
            (LifecycleState.DORMANT, LifecycleState.DEPARTED),
            (LifecycleState.ACTIVE, LifecycleState.MERGED_INTO),
            (LifecycleState.DORMANT, LifecycleState.MERGED_INTO),
            (LifecycleState.DEPARTED, LifecycleState.EXPIRED),
        ],
    )
    def test_every_documented_edge_is_legal(self, previous, current) -> None:
        assert is_legal(previous, current), (
            f"{previous.value} -> {current.value} is in the 02_VOM diagram"
        )

    def test_terminal_states_are_final(self) -> None:
        for terminal in (LifecycleState.MERGED_INTO, LifecycleState.EXPIRED):
            for target in LifecycleState:
                assert not is_legal(terminal, target), (
                    f"{terminal.value} -> {target.value} must be illegal; a "
                    f"resurrected object corrupts every count derived from it"
                )

    def test_an_occluded_object_cannot_merge(self) -> None:
        """Not in the diagram: an occluded object is mid-claim.

        It must resolve to active or dormant before its identity is revised.
        """
        assert not is_legal(LifecycleState.OCCLUDED, LifecycleState.MERGED_INTO)

    def test_a_departed_object_cannot_merge(self) -> None:
        """Merging a departed object asserts a continuity nobody observed."""
        assert not is_legal(LifecycleState.DEPARTED, LifecycleState.MERGED_INTO)

    def test_a_departed_object_cannot_reactivate(self) -> None:
        assert not is_legal(LifecycleState.DEPARTED, LifecycleState.ACTIVE)

    def test_a_provisional_object_cannot_skip_to_dormant(self) -> None:
        assert not is_legal(LifecycleState.PROVISIONAL, LifecycleState.DORMANT)
        assert not is_legal(LifecycleState.PROVISIONAL, LifecycleState.OCCLUDED)

    def test_an_active_object_cannot_regress_to_provisional(self) -> None:
        assert not is_legal(LifecycleState.ACTIVE, LifecycleState.PROVISIONAL)

    def test_check_transition_raises_on_an_illegal_edge(self) -> None:
        with pytest.raises(IllegalTransitionError, match="closed machine"):
            check_transition(LifecycleState.EXPIRED, LifecycleState.ACTIVE)

    def test_the_table_is_closed(self) -> None:
        for previous, current in itertools.product(LifecycleState, LifecycleState):
            if is_legal(previous, current):
                check_transition(previous, current)
            else:
                with pytest.raises(IllegalTransitionError):
                    check_transition(previous, current)


class TestConfirmation:
    def test_an_object_confirms_at_the_observation_threshold(self, machine) -> None:
        transition = machine.on_measured(
            state=LifecycleState.PROVISIONAL, observation_count=3
        )
        assert transition.current is LifecycleState.ACTIVE
        assert transition.trigger is LifecycleTrigger.CONFIRMATION_MET

    def test_it_stays_provisional_below_the_threshold(self, machine) -> None:
        transition = machine.on_measured(
            state=LifecycleState.PROVISIONAL, observation_count=2
        )
        assert transition.current is LifecycleState.PROVISIONAL

    def test_confirmation_is_immediate_not_one_frame_late(self) -> None:
        machine = ObjectLifecycleMachine(LifecyclePolicy(min_observations_to_confirm=1))
        assert (
            machine.on_measured(
                state=LifecycleState.PROVISIONAL, observation_count=1
            ).current
            is LifecycleState.ACTIVE
        )

    def test_an_active_object_stays_active_when_measured(self, machine) -> None:
        transition = machine.on_measured(
            state=LifecycleState.ACTIVE, observation_count=50
        )
        assert transition.current is LifecycleState.ACTIVE
        assert not transition.changed


class TestOcclusionAndDeparture:
    def test_an_unmeasured_active_object_becomes_occluded(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.ACTIVE, since_confirmed=Duration.from_millis(200)
        )
        assert transition.current is LifecycleState.OCCLUDED
        assert transition.trigger is LifecycleTrigger.MEASUREMENT_LOST

    def test_leaving_the_frame_goes_straight_to_dormant(self, machine) -> None:
        """Different claim from occlusion, and the diagram has separate edges.

        A consumer that cannot tell them apart cannot reason about egress.
        """
        transition = machine.on_unmeasured(
            state=LifecycleState.ACTIVE,
            since_confirmed=Duration.from_millis(200),
            left_field_of_view=True,
        )
        assert transition.current is LifecycleState.DORMANT
        assert transition.trigger is LifecycleTrigger.LEFT_FIELD_OF_VIEW

    def test_occlusion_persists_within_its_horizon(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.OCCLUDED, since_confirmed=Duration.from_millis(1_000)
        )
        assert transition.current is LifecycleState.OCCLUDED

    def test_occlusion_becomes_dormant_past_its_horizon(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.OCCLUDED, since_confirmed=Duration.from_millis(5_000)
        )
        assert transition.current is LifecycleState.DORMANT
        assert transition.trigger is LifecycleTrigger.OCCLUSION_HORIZON

    def test_dormant_becomes_departed_past_its_horizon(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.DORMANT, since_confirmed=Duration.from_millis(11_000)
        )
        assert transition.current is LifecycleState.DEPARTED

    def test_departed_expires_past_retention(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.DEPARTED, since_confirmed=Duration.from_millis(31_000)
        )
        assert transition.current is LifecycleState.EXPIRED
        assert transition.trigger is LifecycleTrigger.RETENTION_HORIZON

    def test_a_provisional_object_expires_if_never_confirmed(self, machine) -> None:
        transition = machine.on_unmeasured(
            state=LifecycleState.PROVISIONAL, since_confirmed=Duration.from_millis(3_000)
        )
        assert transition.current is LifecycleState.EXPIRED
        assert transition.trigger is LifecycleTrigger.NEVER_CONFIRMED


class TestReEntry:
    def test_an_occluded_object_reactivates_when_measured(self, machine) -> None:
        transition = machine.on_measured(
            state=LifecycleState.OCCLUDED, observation_count=20
        )
        assert transition.current is LifecycleState.ACTIVE
        assert transition.trigger is LifecycleTrigger.REASSOCIATED

    def test_a_dormant_object_reactivates_on_re_entry(self, machine) -> None:
        transition = machine.on_measured(
            state=LifecycleState.DORMANT, observation_count=20
        )
        assert transition.current is LifecycleState.ACTIVE
        assert transition.trigger is LifecycleTrigger.RE_ENTRY_MATCHED

    def test_a_departed_object_cannot_be_re_measured(self, machine) -> None:
        """Re-entry after departure mints a *new* object plus an assertion.

        The diagram has no departed -> active edge, and adding one would let the
        registry claim a continuity it never observed.
        """
        with pytest.raises(IllegalTransitionError, match="new object"):
            machine.on_measured(state=LifecycleState.DEPARTED, observation_count=1)


class TestTerminalStatesAreInert:
    @pytest.mark.parametrize(
        "state", [LifecycleState.MERGED_INTO, LifecycleState.EXPIRED]
    )
    def test_a_terminal_object_cannot_be_measured(self, machine, state) -> None:
        with pytest.raises(IllegalTransitionError, match="terminal"):
            machine.on_measured(state=state, observation_count=1)

    @pytest.mark.parametrize(
        "state", [LifecycleState.MERGED_INTO, LifecycleState.EXPIRED]
    )
    def test_a_terminal_object_cannot_age(self, machine, state) -> None:
        with pytest.raises(IllegalTransitionError, match="terminal"):
            machine.on_unmeasured(state=state, since_confirmed=Duration.from_millis(1))

    def test_a_terminal_object_cannot_be_merged(self, machine) -> None:
        with pytest.raises(IllegalTransitionError, match="terminal"):
            machine.on_merged(state=LifecycleState.EXPIRED)


class TestMerging:
    def test_an_active_object_can_be_merged(self, machine) -> None:
        transition = machine.on_merged(state=LifecycleState.ACTIVE)
        assert transition.current is LifecycleState.MERGED_INTO
        assert transition.trigger is LifecycleTrigger.IDENTITY_RESOLUTION

    def test_a_dormant_object_can_be_merged(self, machine) -> None:
        assert (
            machine.on_merged(state=LifecycleState.DORMANT).current
            is LifecycleState.MERGED_INTO
        )

    def test_an_occluded_object_cannot_be_merged(self, machine) -> None:
        with pytest.raises(IllegalTransitionError, match="illegal"):
            machine.on_merged(state=LifecycleState.OCCLUDED)


class TestShedding:
    def test_only_provisional_objects_may_be_shed(self, machine) -> None:
        transition = machine.on_shed(state=LifecycleState.PROVISIONAL)
        assert transition.current is LifecycleState.EXPIRED
        assert transition.trigger is LifecycleTrigger.POPULATION_SHED

    @pytest.mark.parametrize(
        "state",
        [LifecycleState.ACTIVE, LifecycleState.OCCLUDED, LifecycleState.DORMANT],
    )
    def test_a_confirmed_object_is_never_shed(self, machine, state) -> None:
        """Withdrawing an assertion to save memory would make the platform's
        claims a function of its load."""
        with pytest.raises(IllegalTransitionError, match="asserted to consumers"):
            machine.on_shed(state=state)


class TestTransitionRecord:
    def test_changed_detects_a_real_move(self, machine) -> None:
        moved = machine.on_measured(
            state=LifecycleState.PROVISIONAL, observation_count=3
        )
        stayed = machine.on_measured(state=LifecycleState.ACTIVE, observation_count=9)
        assert moved.changed
        assert not stayed.changed

    def test_every_transition_names_its_trigger(self, machine) -> None:
        """Every transition must be observable, so none may be unexplained."""
        transitions = [
            machine.on_measured(state=LifecycleState.PROVISIONAL, observation_count=1),
            machine.on_measured(state=LifecycleState.PROVISIONAL, observation_count=3),
            machine.on_unmeasured(
                state=LifecycleState.ACTIVE, since_confirmed=Duration.from_millis(100)
            ),
            machine.on_unmeasured(
                state=LifecycleState.OCCLUDED, since_confirmed=Duration.from_millis(9_000)
            ),
            machine.on_merged(state=LifecycleState.ACTIVE),
            machine.on_shed(state=LifecycleState.PROVISIONAL),
        ]
        for transition in transitions:
            assert transition.trigger in set(LifecycleTrigger)

    def test_terminal_transitions_are_flagged(self, machine) -> None:
        assert machine.on_merged(state=LifecycleState.ACTIVE).is_terminal
        assert machine.on_shed(state=LifecycleState.PROVISIONAL).is_terminal


class TestFullLifecycleWalk:
    def test_the_documented_path_is_walkable(self, machine) -> None:
        """provisional -> active -> occluded -> dormant -> departed -> expired."""
        visited = [LifecycleState.PROVISIONAL]
        state = LifecycleState.PROVISIONAL

        state = machine.on_measured(state=state, observation_count=3).current
        visited.append(state)

        state = machine.on_unmeasured(
            state=state, since_confirmed=Duration.from_millis(200)
        ).current
        visited.append(state)

        state = machine.on_unmeasured(
            state=state, since_confirmed=Duration.from_millis(5_000)
        ).current
        visited.append(state)

        state = machine.on_unmeasured(
            state=state, since_confirmed=Duration.from_millis(11_000)
        ).current
        visited.append(state)

        state = machine.on_unmeasured(
            state=state, since_confirmed=Duration.from_millis(31_000)
        ).current
        visited.append(state)

        assert visited == [
            LifecycleState.PROVISIONAL,
            LifecycleState.ACTIVE,
            LifecycleState.OCCLUDED,
            LifecycleState.DORMANT,
            LifecycleState.DEPARTED,
            LifecycleState.EXPIRED,
        ]

    def test_the_recovery_path_is_walkable(self, machine) -> None:
        """active -> occluded -> active, and active -> dormant -> active."""
        state = machine.on_unmeasured(
            state=LifecycleState.ACTIVE, since_confirmed=Duration.from_millis(200)
        ).current
        assert state is LifecycleState.OCCLUDED
        assert (
            machine.on_measured(state=state, observation_count=10).current
            is LifecycleState.ACTIVE
        )

        state = machine.on_unmeasured(
            state=LifecycleState.ACTIVE,
            since_confirmed=Duration.from_millis(200),
            left_field_of_view=True,
        ).current
        assert state is LifecycleState.DORMANT
        assert (
            machine.on_measured(state=state, observation_count=10).current
            is LifecycleState.ACTIVE
        )
