"""Association — cost, gating, assignment, and refusing to guess.

This is where ID switches are born. Two objects crossing produce two plausible
assignments, and a tracker that picks confidently is wrong half the time while
looking clean. The architecture is explicit (03_MODULES M6): *prefer terminating
a track over a wrong association*, and *never hide uncertainty to look clean*.

The optimal assigner is checked against exhaustive search, because a subtly wrong
Hungarian implementation produces assignments that are merely suboptimal — which
looks like ordinary tracker noise and is nearly impossible to spot in the field.
"""

from __future__ import annotations

import itertools
import random

import pytest

from vision_os.core.model.space import Box
from vision_os.core.ports.tracking import AssociationCandidate, Prediction
from vision_os.perception.tracking.association import (
    AssociationPolicy,
    CostMatrixBuilder,
    GreedyAssociator,
    OptimalAssociator,
    _hungarian,
    centre_distance,
    iou,
    scale_ratio_penalty,
)


def prediction_of(box: Box, uncertainty: float = 0.0) -> Prediction:
    return Prediction(box.x1, box.y1, box.x2, box.y2, uncertainty)


class TestIou:
    def test_identical_boxes_overlap_completely(self) -> None:
        box = Box(0.1, 0.1, 0.3, 0.3)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self) -> None:
        assert iou(Box(0.0, 0.0, 0.1, 0.1), Box(0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_touching_edges_do_not_overlap(self) -> None:
        assert iou(Box(0.0, 0.0, 0.1, 0.1), Box(0.1, 0.0, 0.2, 0.1)) == 0.0

    def test_half_overlap_is_one_third(self) -> None:
        """Two unit boxes overlapping by half: 0.5 / 1.5."""
        a = Box(0.0, 0.0, 1.0, 1.0)
        b = Box(0.5, 0.0, 1.5, 1.0)
        assert iou(a, b) == pytest.approx(1 / 3)

    def test_containment_is_the_area_ratio(self) -> None:
        outer = Box(0.0, 0.0, 1.0, 1.0)
        inner = Box(0.25, 0.25, 0.75, 0.75)
        assert iou(outer, inner) == pytest.approx(0.25)

    def test_is_symmetric(self) -> None:
        a, b = Box(0.1, 0.1, 0.4, 0.4), Box(0.2, 0.2, 0.6, 0.6)
        assert iou(a, b) == pytest.approx(iou(b, a))


class TestDistanceAndScale:
    def test_identical_centres_are_zero_apart(self) -> None:
        box = Box(0.1, 0.1, 0.3, 0.3)
        assert centre_distance(box, box) == 0.0

    def test_distance_is_normalized_into_the_unit_range(self) -> None:
        """Corner to corner of the unit square is the maximum, mapped to 1.0."""
        a = Box(0.0, 0.0, 0.01, 0.01)
        b = Box(0.99, 0.99, 1.0, 1.0)
        assert 0.9 < centre_distance(a, b) <= 1.0

    def test_same_size_boxes_carry_no_scale_penalty(self) -> None:
        a, b = Box(0.0, 0.0, 0.2, 0.2), Box(0.5, 0.5, 0.7, 0.7)
        assert scale_ratio_penalty(a, b) == pytest.approx(0.0)

    def test_a_quarter_sized_box_is_penalized(self) -> None:
        """A person does not halve in size between consecutive frames."""
        big = Box(0.0, 0.0, 0.4, 0.4)
        small = Box(0.0, 0.0, 0.2, 0.2)
        assert scale_ratio_penalty(big, small) == pytest.approx(0.75)

    def test_scale_penalty_is_symmetric(self) -> None:
        big, small = Box(0.0, 0.0, 0.4, 0.4), Box(0.0, 0.0, 0.2, 0.2)
        assert scale_ratio_penalty(big, small) == pytest.approx(
            scale_ratio_penalty(small, big)
        )


class TestPolicyValidation:
    def test_a_default_policy_is_valid(self) -> None:
        assert AssociationPolicy().weight_total > 0

    def test_all_zero_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            AssociationPolicy(iou_weight=0.0, distance_weight=0.0, scale_weight=0.0)

    def test_negative_weights_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AssociationPolicy(iou_weight=-0.1)

    @pytest.mark.parametrize("field", ["max_cost", "min_iou"])
    def test_out_of_range_thresholds_are_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            AssociationPolicy(**{field: 1.5})

    def test_negative_ambiguity_margin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ambiguity_margin"):
            AssociationPolicy(ambiguity_margin=-0.1)

    def test_confidence_weight_defaults_to_zero(self) -> None:
        """Mixing a presence score into association cost without calibration is
        exactly the confidence conflation the Confidence type prevents."""
        assert AssociationPolicy().confidence_weight == 0.0


class TestCostMatrix:
    @pytest.fixture
    def builder(self) -> CostMatrixBuilder:
        return CostMatrixBuilder(AssociationPolicy())

    def test_a_perfect_overlap_costs_nearly_nothing(self, builder) -> None:
        box = Box(0.1, 0.1, 0.3, 0.5)
        candidates = builder.build(predictions=[prediction_of(box)], detection_boxes=[box])
        assert len(candidates) == 1
        assert candidates[0].cost < 0.05

    def test_distant_boxes_are_gated_out(self, builder) -> None:
        candidates = builder.build(
            predictions=[prediction_of(Box(0.0, 0.0, 0.1, 0.1))],
            detection_boxes=[Box(0.9, 0.9, 1.0, 1.0)],
        )
        assert candidates == ()

    def test_prediction_uncertainty_widens_the_gate(self, builder) -> None:
        """A long coast must stay recoverable without the gate being permanently
        loose — so the gate widens with the prediction's own uncertainty."""
        predicted = Box(0.1, 0.4, 0.2, 0.8)
        detected = Box(0.3, 0.4, 0.4, 0.8)

        tight = builder.build(
            predictions=[prediction_of(predicted, 0.0)], detection_boxes=[detected]
        )
        wide = builder.build(
            predictions=[prediction_of(predicted, 0.2)], detection_boxes=[detected]
        )
        assert tight == ()
        assert len(wide) == 1

    def test_candidates_are_ordered_deterministically(self, builder) -> None:
        boxes = [Box(0.1 + i * 0.02, 0.4, 0.2 + i * 0.02, 0.8) for i in range(4)]
        predictions = [prediction_of(b) for b in boxes]
        first = builder.build(predictions=predictions, detection_boxes=boxes)
        second = builder.build(predictions=predictions, detection_boxes=boxes)
        assert first == second
        assert list(first) == sorted(
            first, key=lambda c: (c.cost, c.track_index, c.detection_index)
        )

    def test_a_degenerate_prediction_gates_nothing_rather_than_crashing(
        self, builder
    ) -> None:
        """An object predicted past a frame edge must not take the frame down."""
        broken = Prediction(0.5, 0.5, 0.5, 0.5)
        candidates = builder.build(
            predictions=[broken], detection_boxes=[Box(0.1, 0.1, 0.2, 0.2)]
        )
        assert candidates == ()

    def test_costs_stay_within_the_unit_range(self, builder) -> None:
        boxes = [Box(0.0, 0.0, 0.3, 0.3), Box(0.05, 0.05, 0.5, 0.5)]
        candidates = builder.build(
            predictions=[prediction_of(boxes[0])], detection_boxes=boxes
        )
        for candidate in candidates:
            assert 0.0 <= candidate.cost <= 1.0

    def test_empty_input_yields_no_candidates(self, builder) -> None:
        assert builder.build(predictions=[], detection_boxes=[]) == ()
        assert builder.build(predictions=[], detection_boxes=[Box(0, 0, 1, 1)]) == ()


class TestGreedyAssignment:
    @pytest.fixture
    def associator(self) -> GreedyAssociator:
        return GreedyAssociator()

    def test_a_single_pair_matches(self, associator) -> None:
        result = associator.assign(
            track_count=1,
            detection_count=1,
            candidates=[AssociationCandidate(0, 0, 0.1)],
            max_cost=0.7,
        )
        assert result.matches == ((0, 0),)
        assert result.unmatched_tracks == ()
        assert result.unmatched_detections == ()

    def test_cheapest_wins_a_contested_detection(self, associator) -> None:
        result = associator.assign(
            track_count=2,
            detection_count=1,
            candidates=[
                AssociationCandidate(0, 0, 0.4),
                AssociationCandidate(1, 0, 0.1),
            ],
            max_cost=0.7,
        )
        assert result.matches == ((1, 0),)
        assert result.unmatched_tracks == (0,)

    def test_a_track_matches_at_most_one_detection(self, associator) -> None:
        result = associator.assign(
            track_count=1,
            detection_count=2,
            candidates=[
                AssociationCandidate(0, 0, 0.1),
                AssociationCandidate(0, 1, 0.2),
            ],
            max_cost=0.7,
        )
        assert len(result.matches) == 1
        assert result.unmatched_detections == (1,)

    def test_candidates_above_max_cost_are_ignored(self, associator) -> None:
        result = associator.assign(
            track_count=1,
            detection_count=1,
            candidates=[AssociationCandidate(0, 0, 0.9)],
            max_cost=0.7,
        )
        assert result.matches == ()
        assert result.unmatched_tracks == (0,)

    def test_runner_up_is_recorded_for_a_contested_track(self, associator) -> None:
        """The margin to the runner-up is the ambiguity signal M6 requires."""
        result = associator.assign(
            track_count=1,
            detection_count=2,
            candidates=[
                AssociationCandidate(0, 0, 0.20),
                AssociationCandidate(0, 1, 0.22),
            ],
            max_cost=0.7,
        )
        assert result.runner_up[0] == pytest.approx(0.22)

    def test_is_deterministic_under_ties(self, associator) -> None:
        candidates = [
            AssociationCandidate(0, 0, 0.5),
            AssociationCandidate(0, 1, 0.5),
            AssociationCandidate(1, 0, 0.5),
            AssociationCandidate(1, 1, 0.5),
        ]
        results = {
            associator.assign(
                track_count=2, detection_count=2, candidates=candidates, max_cost=0.7
            ).matches
            for _ in range(50)
        }
        assert len(results) == 1

    def test_input_order_does_not_change_the_outcome(self, associator) -> None:
        candidates = [
            AssociationCandidate(0, 1, 0.3),
            AssociationCandidate(1, 0, 0.2),
            AssociationCandidate(0, 0, 0.4),
        ]
        baseline = associator.assign(
            track_count=2, detection_count=2, candidates=candidates, max_cost=0.9
        ).matches
        for permutation in itertools.permutations(candidates):
            assert (
                associator.assign(
                    track_count=2,
                    detection_count=2,
                    candidates=list(permutation),
                    max_cost=0.9,
                ).matches
                == baseline
            )

    def test_empty_input_reports_everything_unmatched(self, associator) -> None:
        result = associator.assign(
            track_count=2, detection_count=3, candidates=[], max_cost=0.7
        )
        assert result.unmatched_tracks == (0, 1)
        assert result.unmatched_detections == (0, 1, 2)


class TestOptimalAssignment:
    @pytest.fixture
    def associator(self) -> OptimalAssociator:
        return OptimalAssociator()

    def test_it_beats_greedy_on_the_classic_trap(self, associator) -> None:
        """Greedy takes the cheap pair first and strands the rest.

        Track 0 can match detection 0 at 0.10 or detection 1 at 0.60.
        Track 1 can match detection 0 at 0.15 only.
        Greedy: (0,0)=0.10, then track 1 has nothing -> total 0.10, one unmatched.
        Optimal: (0,1)=0.60 + (1,0)=0.15 = 0.75, both matched.
        """
        candidates = [
            AssociationCandidate(0, 0, 0.10),
            AssociationCandidate(0, 1, 0.60),
            AssociationCandidate(1, 0, 0.15),
        ]
        greedy = GreedyAssociator().assign(
            track_count=2, detection_count=2, candidates=candidates, max_cost=0.7
        )
        optimal = associator.assign(
            track_count=2, detection_count=2, candidates=candidates, max_cost=0.7
        )
        assert len(greedy.matches) == 1
        assert len(optimal.matches) == 2

    def test_matches_brute_force_on_random_problems(self, associator) -> None:
        """A subtly wrong solver produces merely suboptimal assignments, which
        look like ordinary tracker noise and are nearly unfindable in the field."""
        rng = random.Random(20260803)  # noqa: S311 - test fixtures, not crypto
        for _ in range(60):
            tracks = rng.randint(1, 4)
            detections = rng.randint(1, 4)
            candidates = [
                AssociationCandidate(t, d, round(rng.uniform(0.0, 0.6), 3))
                for t in range(tracks)
                for d in range(detections)
            ]
            result = associator.assign(
                track_count=tracks,
                detection_count=detections,
                candidates=candidates,
                max_cost=1.0,
            )
            got = sum(result.costs.values())

            lookup = {(c.track_index, c.detection_index): c.cost for c in candidates}
            best = float("inf")
            size = min(tracks, detections)
            for track_subset in itertools.permutations(range(tracks), size):
                for detection_subset in itertools.permutations(range(detections), size):
                    total = sum(
                        lookup[(track_subset[i], detection_subset[i])] for i in range(size)
                    )
                    best = min(best, total)
            assert got == pytest.approx(best), (
                f"suboptimal assignment: {got} vs optimal {best}"
            )

    def test_is_deterministic(self, associator) -> None:
        candidates = [
            AssociationCandidate(t, d, 0.3)
            for t in range(3)
            for d in range(3)
        ]
        results = {
            associator.assign(
                track_count=3, detection_count=3, candidates=candidates, max_cost=0.7
            ).matches
            for _ in range(30)
        }
        assert len(results) == 1

    def test_empty_input_is_handled(self, associator) -> None:
        result = associator.assign(
            track_count=0, detection_count=0, candidates=[], max_cost=0.7
        )
        assert result.matches == ()

    def test_no_feasible_candidate_matches_nothing(self, associator) -> None:
        result = associator.assign(
            track_count=2,
            detection_count=2,
            candidates=[AssociationCandidate(0, 0, 0.99)],
            max_cost=0.5,
        )
        assert result.matches == ()
        assert result.unmatched_tracks == (0, 1)

    def test_rectangular_problems_match_the_smaller_side(self, associator) -> None:
        candidates = [
            AssociationCandidate(t, d, 0.1 + 0.01 * (t + d))
            for t in range(2)
            for d in range(5)
        ]
        result = associator.assign(
            track_count=2, detection_count=5, candidates=candidates, max_cost=0.9
        )
        assert len(result.matches) == 2
        assert len(result.unmatched_detections) == 3

    def test_both_associators_satisfy_the_same_port(self) -> None:
        for associator in (GreedyAssociator(), OptimalAssociator()):
            assert associator.method_id
            result = associator.assign(
                track_count=1,
                detection_count=1,
                candidates=[AssociationCandidate(0, 0, 0.1)],
                max_cost=0.7,
            )
            assert result.matches == ((0, 0),)


class TestHungarianCore:
    def test_square_optimum(self) -> None:
        matrix = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]
        assignment = _hungarian([row[:] for row in matrix])
        assert sum(matrix[r][c] for r, c in assignment) == pytest.approx(5.0)

    def test_wide_matrix_matches_every_row(self) -> None:
        matrix = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        assignment = _hungarian([row[:] for row in matrix])
        assert len(assignment) == 2
        assert {r for r, _ in assignment} == {0, 1}

    def test_tall_matrix_matches_every_column(self) -> None:
        matrix = [[1.0, 2.0], [4.0, 3.0], [2.0, 5.0], [0.5, 9.0]]
        assignment = _hungarian([row[:] for row in matrix])
        assert len(assignment) == 2
        assert {c for _, c in assignment} == {0, 1}

    def test_single_cell(self) -> None:
        assert _hungarian([[0.42]]) == [(0, 0)]

    def test_all_ties_resolve_identically_every_run(self) -> None:
        matrix = [[0.5] * 4 for _ in range(4)]
        runs = {tuple(_hungarian([row[:] for row in matrix])) for _ in range(30)}
        assert len(runs) == 1
