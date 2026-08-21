"""Track-to-object binding — and the refusal to guess.

Section M7's failure table is unambiguous:

> *Re-entry ambiguity (two candidates match) → Create a **new** object and emit a
> low-confidence identity assertion linking candidates. **Never guess silently**;
> let the consumer choose a confidence threshold (V1).*

So the binder's most important output is not its winner but the case where it
declines to pick one *and keeps the alternatives*. A binder that returned only a
match would make the ambiguity unrecoverable one call after it was known.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import ClassId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.model.visual_object import BindingMethod, LifecycleState
from vision_os.perception.registry.binding import (
    BindingPolicy,
    TrackBinder,
    _class_compatible,
)
from vision_os.perception.registry.lifecycle import LifecyclePolicy
from vision_os.perception.registry.partition import RegistryPartition

from ..conftest import CAMERA, PERSON, SITE, TENANT, at, spatial, track_id


@pytest.fixture
def binder() -> TrackBinder:
    return TrackBinder(BindingPolicy())


def make_partition(registry_provenance) -> RegistryPartition:
    return RegistryPartition(
        CAMERA,
        tenant_id=TENANT,
        site_id=SITE,
        policy=LifecyclePolicy(),
        provenance=registry_provenance,
    )


def add(
    partition: RegistryPartition,
    *,
    x: float,
    seq: int = 0,
    state: LifecycleState = LifecycleState.DORMANT,
    class_id=PERSON,
    bound: bool = False,
):
    record = partition.mint(
        class_id=class_id,
        confidence=0.9,
        spatial=spatial(Box(x, 0.4, x + 0.1, 0.8)),
        now=at(seq),
        class_confidence=0.9,
    )
    partition.set_lifecycle(record, state)
    if bound:
        partition.open_binding(
            record,
            track_id=track_id(99),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(seq),
        )
    return record


class TestPolicyValidation:
    def test_a_default_policy_is_valid(self) -> None:
        assert BindingPolicy().max_reentry_distance > 0

    @pytest.mark.parametrize(
        "field", ["max_reentry_distance", "ambiguity_margin", "min_binding_confidence"]
    )
    def test_out_of_range_values_are_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            BindingPolicy(**{field: 1.5})

    def test_a_zero_epoch_penalty_is_refused(self) -> None:
        """Zero would make every epoch re-bind impossible rather than weaker."""
        with pytest.raises(ValueError, match="epoch_rebind_penalty"):
            BindingPolicy(epoch_rebind_penalty=0.0)

    def test_a_non_positive_gap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_reentry_gap"):
            BindingPolicy(max_reentry_gap=Duration(0))


class TestTrackContinuity:
    def test_a_track_that_owns_an_object_binds_to_it(
        self, binder, registry_provenance
    ) -> None:
        partition = make_partition(registry_provenance)
        record = add(partition, x=0.3, state=LifecycleState.ACTIVE)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(0),
        )
        decision = binder.bind_continuing(partition.records(), track_id(7))
        assert decision.matched is not None
        assert decision.matched.object_id == record.object_id
        assert decision.matched.method is BindingMethod.TRACK_CONTINUITY

    def test_continuity_is_asserted_at_full_confidence(
        self, binder, registry_provenance
    ) -> None:
        """M6 already asserted it; M7 is only recording it, not re-deciding."""
        partition = make_partition(registry_provenance)
        record = add(partition, x=0.3, state=LifecycleState.ACTIVE)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.5,
            now=at(0),
        )
        decision = binder.bind_continuing(partition.records(), track_id(7))
        assert decision.matched.score == 1.0

    def test_an_unknown_track_matches_nothing(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, state=LifecycleState.ACTIVE)
        assert binder.bind_continuing(partition.records(), track_id(99)).matched is None

    def test_a_terminal_object_is_never_matched(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        record = add(partition, x=0.3, state=LifecycleState.ACTIVE)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(0),
        )
        partition.set_lifecycle(record, LifecycleState.EXPIRED)
        assert binder.bind_continuing(partition.records(), track_id(7)).matched is None


class TestReEntry:
    def test_a_nearby_dormant_object_is_matched(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        record = add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is not None
        assert decision.matched.object_id == record.object_id
        assert decision.matched.method is BindingMethod.SPATIO_TEMPORAL

    def test_an_occluded_object_is_a_candidate(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.OCCLUDED)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is not None

    def test_an_active_object_is_not_a_candidate(self, binder, registry_provenance) -> None:
        """It already has a track; two tracks on one object is a merge, not a bind."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.ACTIVE)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is None

    def test_a_departed_object_is_not_a_candidate(self, binder, registry_provenance) -> None:
        """Re-binding a departed object asserts a continuity nobody observed."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DEPARTED)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is None

    def test_an_already_bound_object_is_not_a_candidate(
        self, binder, registry_provenance
    ) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT, bound=True)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is None

    def test_a_distant_object_is_gated_out(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.05, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.85, 0.4, 0.95, 0.8)),
            class_id=PERSON,
            now=at(5),
        )
        assert decision.matched is None
        assert decision.reason == "no_candidates"

    def test_a_stale_object_is_gated_out(self, binder, registry_provenance) -> None:
        """Past the gap, position tells you almost nothing."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=PERSON,
            now=at(1_000),
        )
        assert decision.matched is None

    def test_a_different_class_is_gated_out(self, binder, registry_provenance) -> None:
        """A person does not become a forklift."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=ClassId("vehicle.forklift"),
            now=at(5),
        )
        assert decision.matched is None

    def test_a_refined_class_is_compatible(self, binder, registry_provenance) -> None:
        """A detector refining ``person`` to ``person.child`` is normal."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=ClassId("person.child"),
            now=at(5),
        )
        assert decision.matched is not None

    def test_class_matching_can_be_disabled(self, registry_provenance) -> None:
        binder = TrackBinder(BindingPolicy(class_must_match=False))
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.32, 0.4, 0.42, 0.8)),
            class_id=ClassId("vehicle"),
            now=at(5),
        )
        assert decision.matched is not None


class TestAmbiguityIsRefused:
    """Section M7: never guess silently."""

    def test_two_equally_good_candidates_are_refused(
        self, binder, registry_provenance
    ) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.32, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
        )
        assert decision.matched is None
        assert decision.ambiguous
        assert "ambiguous" in decision.reason

    def test_a_refusal_retains_every_candidate(self, binder, registry_provenance) -> None:
        """The alternatives must survive the decision, or the ambiguity is lost."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.32, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
        )
        assert len(decision.candidates) == 2
        assert len(decision.alternatives) == 2
        assert all(0.0 <= score <= 1.0 for _, score in decision.alternatives)

    def test_a_clear_winner_is_matched(self, binder, registry_provenance) -> None:
        """The refusal must be selective, or it would refuse everything."""
        partition = make_partition(registry_provenance)
        near = add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.50, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
            class_id=PERSON,
            now=at(1),
        )
        assert decision.matched is not None
        assert decision.matched.object_id == near.object_id
        assert not decision.ambiguous

    def test_a_zero_margin_disables_refusal(self, registry_provenance) -> None:
        binder = TrackBinder(BindingPolicy(ambiguity_margin=0.0))
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.32, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
        )
        assert decision.matched is not None
        assert not decision.ambiguous

    def test_the_margin_is_reported(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.45, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
            class_id=PERSON,
            now=at(1),
        )
        assert decision.margin is not None
        assert decision.margin >= 0.0


class TestEpochRebinding:
    def test_crossing_an_epoch_reduces_confidence(self, binder, registry_provenance) -> None:
        """07_STATE section 9.3 requires explicitly reduced confidence."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)

        same = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
            crossing_epoch=False,
        )
        crossed = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
            crossing_epoch=True,
        )
        assert crossed.matched.score < same.matched.score

    def test_crossing_an_epoch_is_reported_as_its_own_method(
        self, binder, registry_provenance
    ) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.3, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(2),
            crossing_epoch=True,
        )
        assert decision.matched.method is BindingMethod.EPOCH_REBIND


class TestScoring:
    def test_a_closer_candidate_scores_higher(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        near = add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.45, seq=0, state=LifecycleState.DORMANT)

        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
            class_id=PERSON,
            now=at(1),
        )
        assert decision.candidates[0].object_id == near.object_id

    def test_a_more_recent_candidate_scores_higher(
        self, binder, registry_provenance
    ) -> None:
        partition = make_partition(registry_provenance)
        stale = add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        fresh = add(partition, x=0.30, seq=50, state=LifecycleState.DORMANT)
        # Both at the same place; only recency separates them.
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
            class_id=PERSON,
            now=at(55),
        )
        scores = {c.object_id: c.score for c in decision.candidates}
        assert scores[fresh.object_id] > scores[stale.object_id]

    def test_candidate_order_is_deterministic(self, binder, registry_provenance) -> None:
        """An arbitrary tie-break would make identity depend on dict order (V13)."""
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)

        runs = set()
        for _ in range(20):
            decision = binder.bind_reentry(
                partition.records(),
                spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
                class_id=PERSON,
                now=at(1),
            )
            runs.add(tuple(c.object_id for c in decision.candidates))
        assert len(runs) == 1

    def test_every_candidate_carries_its_evidence(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.31, 0.4, 0.41, 0.8)),
            class_id=PERSON,
            now=at(3),
        )
        candidate = decision.candidates[0]
        assert candidate.rationale
        assert candidate.distance >= 0.0
        assert candidate.gap.ns >= 0

    def test_a_low_scoring_candidate_is_dropped(self, registry_provenance) -> None:
        binder = TrackBinder(BindingPolicy(min_binding_confidence=0.95))
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.36, 0.4, 0.46, 0.8)),
            class_id=PERSON,
            now=at(20),
        )
        assert decision.matched is None


class TestClassCompatibility:
    def test_identical_classes_are_compatible(self) -> None:
        assert _class_compatible("person", "person")

    def test_a_refinement_is_compatible_both_ways(self) -> None:
        assert _class_compatible("person", "person.child")
        assert _class_compatible("person.child", "person")

    def test_unrelated_classes_are_incompatible(self) -> None:
        assert not _class_compatible("person", "vehicle")

    def test_a_bare_prefix_is_not_a_refinement(self) -> None:
        assert not _class_compatible("person", "personnel")


class TestEmptyCases:
    def test_no_records_yields_no_candidates(self, binder) -> None:
        decision = binder.bind_reentry(
            (), spatial=spatial(Box(0.3, 0.4, 0.4, 0.8)), class_id=PERSON, now=at(0)
        )
        assert decision.matched is None
        assert decision.candidates == ()

    def test_no_records_has_no_continuity_match(self, binder) -> None:
        assert binder.bind_continuing((), track_id(1)).matched is None

    def test_a_single_candidate_has_no_margin(self, binder, registry_provenance) -> None:
        partition = make_partition(registry_provenance)
        add(partition, x=0.30, seq=0, state=LifecycleState.DORMANT)
        decision = binder.bind_reentry(
            partition.records(),
            spatial=spatial(Box(0.30, 0.4, 0.40, 0.8)),
            class_id=PERSON,
            now=at(1),
        )
        assert decision.margin is None
