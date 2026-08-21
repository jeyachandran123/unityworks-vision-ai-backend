"""Association — deciding which detection continues which track.

> **Single responsibility:** *Build a cost matrix and resolve it. Own no tracks,
> mutate no state, predict nothing.*

This module is where ID switches are born. Two objects crossing produce two
plausible assignments, and a tracker that picks confidently is wrong half the
time while looking clean. The architecture is explicit about the trade
(03_MODULES M6): *"Prefer terminating a track over a wrong association"*, and
*"the tracker never hides uncertainty to look clean — a confidently wrong
association is far more damaging downstream than an admitted uncertain one."*

That principle is implemented literally here:

* the **margin** between the best and second-best candidate is computed and
  retained, not discarded;
* a match whose margin falls below ``ambiguity_margin`` is **refused**, and the
  track coasts rather than binding to a detection it might not own.

Cost is normalized to [0,1] so geometry, motion and (later) appearance combine
meaningfully. Assignment is deterministic including tie-breaks — non-determinism
here silently changes which object keeps which id, producing ID switches no test
can reproduce (invariant V13).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.model.space import Box
from ...core.ports.tracking import (
    AssignmentResult,
    AssociationCandidate,
    Prediction,
)


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Pure geometry, no domain meaning."""
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a.area + b.area - intersection
    return intersection / union if union > 0.0 else 0.0


def centre_distance(a: Box, b: Box) -> float:
    """Normalized centre separation, clamped to [0,1].

    Divided by the diagonal of the unit square so the value is comparable with
    an IoU-derived cost rather than being on an arbitrary scale.
    """
    separation = a.centre.distance_to(b.centre)
    return min(1.0, separation / 1.4142135623730951)


def scale_ratio_penalty(a: Box, b: Box) -> float:
    """How differently sized two boxes are, in [0,1].

    A person does not halve in size between consecutive frames. A candidate that
    matches well on position but badly on scale is usually a different object at
    a different depth, and this is what separates them.
    """
    area_a, area_b = a.area, b.area
    if area_a <= 0.0 or area_b <= 0.0:
        return 1.0
    ratio = min(area_a, area_b) / max(area_a, area_b)
    return 1.0 - ratio


@dataclass(frozen=True, slots=True)
class AssociationPolicy:
    """Weights and gates. Strongly typed, validated on construction."""

    iou_weight: float = 0.6
    distance_weight: float = 0.25
    scale_weight: float = 0.15
    confidence_weight: float = 0.0
    """Optional bias toward higher-confidence detections. Zero by default: a
    detector's presence score measures something different from association
    likelihood, and mixing them without calibration is exactly the confidence
    conflation the platform's ``Confidence`` type exists to prevent."""

    max_cost: float = 0.7
    """Above this a pair is not a candidate at all."""

    min_iou: float = 0.1
    """Hard geometric gate applied before cost is computed."""

    gate_multiplier: float = 3.0
    """How many prediction standard deviations wide the motion gate is. A larger
    prediction uncertainty legitimately widens the gate, which is how a long
    coast stays recoverable without the gate being permanently loose."""

    ambiguity_margin: float = 0.05
    """Minimum cost gap between the best and second-best candidate for a match
    to be asserted. Below this the association is refused and the track coasts —
    the architecture's *"prefer terminating a track over a wrong association"*
    made executable."""

    def __post_init__(self) -> None:
        for name in ("iou_weight", "distance_weight", "scale_weight", "confidence_weight"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        total = (
            self.iou_weight + self.distance_weight + self.scale_weight + self.confidence_weight
        )
        if total <= 0.0:
            raise ValueError("association weights must sum to a positive value")
        if not 0.0 <= self.max_cost <= 1.0:
            raise ValueError(f"max_cost must be in [0,1], got {self.max_cost}")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError(f"min_iou must be in [0,1], got {self.min_iou}")
        if self.gate_multiplier < 0.0:
            raise ValueError("gate_multiplier must be non-negative")
        if self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be non-negative")

    @property
    def weight_total(self) -> float:
        return (
            self.iou_weight + self.distance_weight + self.scale_weight + self.confidence_weight
        )


class CostMatrixBuilder:
    """Turns predicted track positions and detections into gated candidates.

    Gating first, cost second. At 100 objects the full matrix is 10,000 pairs;
    gating by predicted position reduces it to the handful that are geometrically
    possible, which is what keeps association effectively linear at realistic
    densities (03_MODULES M6 performance).
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: AssociationPolicy) -> None:
        self._policy = policy

    def build(
        self,
        *,
        predictions: Sequence[Prediction],
        detection_boxes: Sequence[Box],
        detection_scores: Sequence[float] = (),
    ) -> tuple[AssociationCandidate, ...]:
        """Candidates that survive gating, ordered deterministically by cost."""
        policy = self._policy
        candidates: list[AssociationCandidate] = []

        for track_index, prediction in enumerate(predictions):
            try:
                predicted = Box(prediction.x1, prediction.y1, prediction.x2, prediction.y2)
            except ValueError:
                # A degenerate prediction (an object predicted past a frame edge)
                # gates nothing rather than crashing the frame.
                continue

            gate = policy.gate_multiplier * prediction.uncertainty
            for detection_index, box in enumerate(detection_boxes):
                overlap = iou(predicted, box)
                separation = centre_distance(predicted, box)

                # The motion gate widens with prediction uncertainty, so a track
                # coasting through an occlusion stays recoverable.
                if overlap < policy.min_iou and separation > gate:
                    continue

                score = (
                    detection_scores[detection_index]
                    if detection_index < len(detection_scores)
                    else 1.0
                )
                cost = self._cost(predicted, box, overlap, separation, score)
                if cost <= policy.max_cost:
                    candidates.append(
                        AssociationCandidate(track_index, detection_index, cost)
                    )

        # Sorted by (cost, track, detection) so ties resolve identically on every
        # run and on every machine (invariant V13).
        candidates.sort(key=lambda c: (c.cost, c.track_index, c.detection_index))
        return tuple(candidates)

    def _cost(
        self, predicted: Box, box: Box, overlap: float, separation: float, score: float
    ) -> float:
        policy = self._policy
        weighted = (
            policy.iou_weight * (1.0 - overlap)
            + policy.distance_weight * separation
            + policy.scale_weight * scale_ratio_penalty(predicted, box)
            + policy.confidence_weight * (1.0 - score)
        )
        return weighted / policy.weight_total


class GreedyAssociator:
    """Deterministic greedy assignment. The default ``AssociationPort``.

    Greedy rather than optimal by default because at realistic densities — after
    gating, a handful of candidates per track — greedy and Hungarian agree, and
    greedy is O(k log k) in surviving candidates rather than O(n^3). The optimal
    solver ships alongside for the crowded case; both satisfy the same port.
    """

    __slots__ = ()

    @property
    def method_id(self) -> str:
        return "association.greedy"

    def assign(
        self,
        *,
        track_count: int,
        detection_count: int,
        candidates: Sequence[AssociationCandidate],
        max_cost: float,
    ) -> AssignmentResult:
        ordered = sorted(
            candidates, key=lambda c: (c.cost, c.track_index, c.detection_index)
        )

        matches: list[tuple[int, int]] = []
        costs: dict[tuple[int, int], float] = {}
        runner_up: dict[int, float] = {}
        taken_tracks: set[int] = set()
        taken_detections: set[int] = set()

        for candidate in ordered:
            if candidate.cost > max_cost:
                break
            if candidate.track_index in taken_tracks:
                continue
            if candidate.detection_index in taken_detections:
                # This track's best alternative, recorded before it is discarded.
                # The gap to the winner is the ambiguity signal the tracker is
                # required to publish rather than hide.
                runner_up.setdefault(candidate.track_index, candidate.cost)
                continue
            matches.append((candidate.track_index, candidate.detection_index))
            costs[(candidate.track_index, candidate.detection_index)] = candidate.cost
            taken_tracks.add(candidate.track_index)
            taken_detections.add(candidate.detection_index)

        # A losing candidate for an already-matched track is also a runner-up.
        for candidate in ordered:
            if candidate.track_index in taken_tracks:
                won = costs.get((candidate.track_index, candidate.detection_index))
                if won is None:
                    existing = runner_up.get(candidate.track_index)
                    if existing is None or candidate.cost < existing:
                        runner_up[candidate.track_index] = candidate.cost

        return AssignmentResult(
            matches=tuple(matches),
            unmatched_tracks=tuple(i for i in range(track_count) if i not in taken_tracks),
            unmatched_detections=tuple(
                i for i in range(detection_count) if i not in taken_detections
            ),
            costs=costs,
            runner_up=runner_up,
        )


class OptimalAssociator:
    """Hungarian (Jonker-Volgenant style) assignment on the gated sub-problem.

    Implemented over the *gated* candidate set rather than the dense matrix: the
    full n x m matrix is mostly infeasible pairs, and solving it wastes the work
    gating just saved. Falls back to a large finite cost for absent pairs so the
    rectangular problem stays well-formed.

    Pure Python and dependency-free — ``scipy.optimize.linear_sum_assignment``
    would be faster but would make an optional numeric stack a hard requirement
    of the tracking layer, which the port structure exists to avoid.
    """

    __slots__ = ()

    @property
    def method_id(self) -> str:
        return "association.optimal"

    def assign(
        self,
        *,
        track_count: int,
        detection_count: int,
        candidates: Sequence[AssociationCandidate],
        max_cost: float,
    ) -> AssignmentResult:
        feasible = [c for c in candidates if c.cost <= max_cost]
        if not feasible or track_count == 0 or detection_count == 0:
            return AssignmentResult(
                unmatched_tracks=tuple(range(track_count)),
                unmatched_detections=tuple(range(detection_count)),
            )

        # Restrict to rows and columns that actually have a feasible pair; the
        # rest are unmatched by construction and only inflate the matrix.
        rows = sorted({c.track_index for c in feasible})
        cols = sorted({c.detection_index for c in feasible})
        row_of = {t: i for i, t in enumerate(rows)}
        col_of = {d: j for j, d in enumerate(cols)}

        infeasible = 1e6
        matrix = [[infeasible] * len(cols) for _ in rows]
        for candidate in feasible:
            matrix[row_of[candidate.track_index]][col_of[candidate.detection_index]] = (
                candidate.cost
            )

        assignment = _hungarian(matrix)

        matches: list[tuple[int, int]] = []
        costs: dict[tuple[int, int], float] = {}
        for row, col in assignment:
            if matrix[row][col] >= infeasible:
                continue
            track_index, detection_index = rows[row], cols[col]
            matches.append((track_index, detection_index))
            costs[(track_index, detection_index)] = matrix[row][col]

        matches.sort()
        matched_tracks = {t for t, _ in matches}
        matched_detections = {d for _, d in matches}

        runner_up: dict[int, float] = {}
        for candidate in sorted(feasible, key=lambda c: (c.cost, c.track_index)):
            if costs.get((candidate.track_index, candidate.detection_index)) is not None:
                continue
            existing = runner_up.get(candidate.track_index)
            if existing is None or candidate.cost < existing:
                runner_up[candidate.track_index] = candidate.cost

        return AssignmentResult(
            matches=tuple(matches),
            unmatched_tracks=tuple(
                i for i in range(track_count) if i not in matched_tracks
            ),
            unmatched_detections=tuple(
                i for i in range(detection_count) if i not in matched_detections
            ),
            costs=costs,
            runner_up=runner_up,
        )


def _hungarian(matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost perfect matching on a rectangular matrix.

    The O(n^3) shortest-augmenting-path formulation. Deterministic: it visits
    rows and columns in index order, so equal-cost alternatives always resolve
    the same way (invariant V13).
    """
    rows, cols = len(matrix), len(matrix[0])
    transposed = rows > cols
    if transposed:
        matrix = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
        rows, cols = cols, rows

    infinity = float("inf")
    potential_row = [0.0] * (rows + 1)
    potential_col = [0.0] * (cols + 1)
    match_col = [rows] * (cols + 1)  # column -> row, `rows` means unmatched

    for row in range(rows):
        match_col[cols] = row
        col_marker = cols
        min_cost = [infinity] * (cols + 1)
        previous = [cols] * (cols + 1)
        visited = [False] * (cols + 1)

        while True:
            visited[col_marker] = True
            current_row = match_col[col_marker]
            delta = infinity
            next_col = cols

            for col in range(cols):
                if visited[col]:
                    continue
                cost = (
                    matrix[current_row][col]
                    - potential_row[current_row]
                    - potential_col[col]
                )
                if cost < min_cost[col]:
                    min_cost[col] = cost
                    previous[col] = col_marker
                if min_cost[col] < delta:
                    delta = min_cost[col]
                    next_col = col

            for col in range(cols + 1):
                if visited[col]:
                    potential_row[match_col[col]] += delta
                    potential_col[col] -= delta
                else:
                    min_cost[col] -= delta

            col_marker = next_col
            if match_col[col_marker] == rows:
                break

        while col_marker != cols:
            moved = previous[col_marker]
            match_col[col_marker] = match_col[moved]
            col_marker = moved

    result: list[tuple[int, int]] = []
    for col in range(cols):
        if match_col[col] < rows:
            result.append((col, match_col[col]) if transposed else (match_col[col], col))
    result.sort()
    return result
