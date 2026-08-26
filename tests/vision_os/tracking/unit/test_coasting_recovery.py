"""A COASTING track must not be re-measured onto a different person.

The LOST gate (``test_lost_recovery.py``) fixed the wrong stage. Measured with
production values afterwards, a departed person's track still never reached
LOST: on every frame it won some other person's detection, ``_on_hit`` reset
``coast_frames`` to zero, and the track re-entered CONFIRMED. LOST was the state
these tracks never got to.

Why any detection at all was reachable is arithmetic, not bad luck. The
association gate widens with prediction uncertainty so a track can survive an
occlusion::

    gate = gate_multiplier * uncertainty = 3.0 * (0.05 * seconds_unmeasured)

Live analysis measured 1.72 s per frame, so after two missed frames the gate is
0.51 — wider than half the frame. Every person present is a geometric
candidate by then, and the only remaining bar is ``max_cost``.

Measured on 2,286 continuous real frames of camera 12 — about twenty minutes of
a working kitchen — through the production detector and the production
association policy, with identities taken from unbroken dense-IoU chains in the
footage rather than from any model's opinion:

    scenario                              n     min    med    p95    max
    SAME_PERSON genuine recovery       8490   0.002  0.123  0.482  0.669
    DIFFERENT_PERSON wrong association  795   0.446  0.626  0.696  0.700

No wrong association was cheaper than 0.446; the ordinary gate admits
everything up to 0.700, so all 795 were takeable. The band between those two
numbers is where ``COAST_RECOVERY_COST_FACTOR`` sits.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.tracking.geometric import (
    COAST_RECOVERY_COST_FACTOR,
    LOST_RECOVERY_COST_FACTOR,
)
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.ports.tracking import MotionObservation
from vision_os.perception.tracking.association import CostMatrixBuilder
from vision_os.adapters.tracking.motion import StationaryPredictor

from tests.vision_os.tracking.conftest import drive, make_request, walking_box

#: Where ``drive(tracker, 8)`` leaves the tracked person: ``walking_box(7)``.
LAST_MEASURED = Box(0.38, 0.4, 0.48, 0.8)

#: The same person, one stride further along their own path. Costs 0.350 —
#: inside the genuine-recovery band measured above.
SAME_PERSON_CONTINUING = Box(0.42, 0.4, 0.52, 0.8)

#: A **different** person standing where a second chef stands: shoulder to
#: shoulder, same size, slightly overlapping. Costs 0.528 — inside the
#: wrong-association band measured above, and comfortably inside the ordinary
#: 0.700 gate, which is why the old code took it.
OTHER_PERSON_ALONGSIDE = Box(0.455, 0.4, 0.555, 0.8)


class Seq:
    """Monotonic frame numbers (obligation T1 rejects a frame that goes back)."""

    def __init__(self, start: int = 0) -> None:
        self._n = start

    def next(self) -> int:
        self._n += 1
        return self._n


def _cost(policy, target: Box, *, seconds: float) -> float | None:
    """The cost the production builder emits for this pair, or None if gated."""
    predictor = StationaryPredictor()
    predictor.observe(
        MotionObservation(
            LAST_MEASURED.x1, LAST_MEASURED.y1, LAST_MEASURED.x2, LAST_MEASURED.y2,
            Duration(0),
        )
    )
    candidates = CostMatrixBuilder(policy).build(
        predictions=[predictor.predict(Duration(int(seconds * 1e9)))],
        detection_boxes=[target],
        detection_scores=[0.9],
    )
    return candidates[0].cost if candidates else None


def _live_ids(tracker, seq: Seq) -> set:
    update = tracker.update(make_request(seq.next(), []))
    return {t.track_id for t in update.active}


class TestTheScenarioMatchesRealFootage:
    """The two boxes above are not chosen to make the gate look good.

    They are pinned against the distributions measured on camera 12, so if the
    association weights are ever retuned this fails loudly rather than leaving
    the behavioural tests quietly testing something else.
    """

    #: The pooled bands from the capture described in the module docstring.
    SAME_PERSON_BAND = (0.002, 0.669)
    DIFFERENT_PERSON_BAND = (0.446, 0.700)

    def test_the_same_person_box_sits_in_the_genuine_recovery_band(
        self, association_policy
    ):
        cost = _cost(association_policy, SAME_PERSON_CONTINUING, seconds=0.4)
        assert cost is not None
        low, high = self.SAME_PERSON_BAND
        assert low <= cost <= high, (
            f"the 'same person' fixture costs {cost:.3f}, outside the "
            f"{low:.3f}-{high:.3f} band measured for real recoveries"
        )

    def test_the_other_person_box_sits_in_the_wrong_association_band(
        self, association_policy
    ):
        cost = _cost(association_policy, OTHER_PERSON_ALONGSIDE, seconds=0.4)
        assert cost is not None
        low, high = self.DIFFERENT_PERSON_BAND
        assert low <= cost <= high, (
            f"the 'other person' fixture costs {cost:.3f}, outside the "
            f"{low:.3f}-{high:.3f} band measured for real wrong associations"
        )

    def test_the_ordinary_gate_would_have_accepted_the_wrong_person(
        self, association_policy
    ):
        """The defect itself, as a number. Without the fix this is the match."""
        cost = _cost(association_policy, OTHER_PERSON_ALONGSIDE, seconds=0.4)
        assert cost <= association_policy.max_cost, (
            "the wrong-person fixture is outside the ordinary gate, so this "
            "suite would prove nothing"
        )

    def test_the_gate_separates_them(self, association_policy):
        ceiling = association_policy.max_cost * COAST_RECOVERY_COST_FACTOR
        assert _cost(association_policy, SAME_PERSON_CONTINUING, seconds=0.4) <= ceiling
        assert _cost(association_policy, OTHER_PERSON_ALONGSIDE, seconds=0.4) > ceiling

    def test_coasting_is_more_forgiving_than_lost(self):
        """A shorter absence earns a wider bar, never a narrower one.

        Both are fractions of the one configured ``max_cost`` so they cannot
        drift apart, and their ordering is the whole reason there are two.
        """
        assert LOST_RECOVERY_COST_FACTOR < COAST_RECOVERY_COST_FACTOR < 1.0


class TestCoastingRecovery:
    """TEST B and TEST C, on the tracker production actually binds.

    ``tracker.iou`` is what ``app/vision/runtime.py`` names, and its
    ``StationaryPredictor`` is why the numbers above are the numbers: the
    prediction stays where the person was last measured while the gate around
    it widens.
    """

    def test_a_coasting_track_recovers_the_same_person(self, iou_tracker):
        """TEST B. Short occlusion must still cost nobody their identity."""
        drive(iou_tracker, 8)
        seq = Seq(50)
        before = _live_ids(iou_tracker, seq)
        assert before, "nobody was tracked to begin with"

        # One more empty frame: two misses total, so the track is COASTING and
        # nowhere near the coast horizon.
        iou_tracker.update(make_request(seq.next(), []))

        update = iou_tracker.update(
            make_request(seq.next(), [SAME_PERSON_CONTINUING])
        )
        after = {t.track_id for t in update.active}

        assert before & after, (
            "a person missed for two frames lost their identity — the coast "
            "window is what those frames are for"
        )
        assert set(update.recovered) & before, (
            f"the recovery was not reported as one: {update.recovered}"
        )

    def test_a_coasting_track_does_not_take_a_different_person(self, iou_tracker):
        """TEST C. The defect, exactly.

        Person A stops being detected. Person B is standing alongside where A
        was. A must coast on toward termination and B must get their own
        identity — the old gate handed A's track to B's detection at 0.528.
        """
        drive(iou_tracker, 8)
        seq = Seq(50)
        a_ids = _live_ids(iou_tracker, seq)
        assert a_ids, "person A was never tracked"

        update = iou_tracker.update(
            make_request(seq.next(), [OTHER_PERSON_ALONGSIDE])
        )

        assert not (set(update.recovered) & a_ids), (
            f"person A's COASTING track was recovered onto person B: "
            f"{set(update.recovered) & a_ids}"
        )
        assert set(update.new), "person B did not receive a track of their own"

        # A is still live — refused, not terminated on the spot — and still
        # unmeasured, which is what lets the lifecycle age it out.
        a_track = next((t for t in update.active if t.track_id in a_ids), None)
        assert a_track is not None, "person A's track vanished without terminating"
        assert a_track.coast_frames >= 2, (
            "person A's coast counter was reset, so the track will never reach "
            "the horizon that terminates it"
        )

    def test_the_refused_track_terminates_while_the_other_person_stays(
        self, iou_tracker, lifecycle_policy
    ):
        """TEST C, continued: denying the match must actually end the track.

        This is the property that was false in production — 97 tracks created,
        zero terminated, because something was always close enough to grab.
        """
        drive(iou_tracker, 8)
        seq = Seq(50)
        a_ids = _live_ids(iou_tracker, seq)

        terminated: list = []
        b_ids: set = set()
        total = lifecycle_policy.max_coast_frames + lifecycle_policy.max_lost_frames + 3
        for _ in range(total):
            update = iou_tracker.update(
                make_request(seq.next(), [OTHER_PERSON_ALONGSIDE])
            )
            terminated.extend(update.terminated)
            b_ids |= set(update.new)

        assert a_ids & {track_id for track_id, _ in terminated}, (
            "person A's track never terminated while person B stood alongside — "
            "it is still being kept alive by wrong associations"
        )

        live = {t.track_id for t in iou_tracker.update(make_request(seq.next(), [
            OTHER_PERSON_ALONGSIDE
        ])).active}
        assert not (live & a_ids), "person A still holds an identity"
        assert live & b_ids, "person B lost their own identity"

    def test_person_b_never_inherits_person_a_identity(self, iou_tracker):
        """The downstream consequence, at the layer that causes it.

        M7 keys attributes by the object a track produced, so an id that is
        never reused is a PPE history that is never transplanted.
        """
        drive(iou_tracker, 8)
        seq = Seq(50)
        a_ids = _live_ids(iou_tracker, seq)

        recovered: list = []
        for _ in range(12):
            update = iou_tracker.update(
                make_request(seq.next(), [OTHER_PERSON_ALONGSIDE])
            )
            recovered.extend(update.recovered)

        assert not (set(recovered) & a_ids), (
            f"person A's identity was handed to person B: {set(recovered) & a_ids}"
        )


class TestUntouchedBehaviour:
    """What the fix must not cost. Every tracker, because the gate is shared."""

    def test_an_ordinary_association_is_unaffected(self, any_tracker):
        """A track measured every frame never meets the recovery gate at all."""
        updates = drive(any_tracker, 10)
        ids = {t.track_id for t in updates[-1].active}
        assert len(ids) == 1, (
            f"a continuously visible person fragmented into {len(ids)} tracks"
        )
        assert not updates[-1].terminated

    def test_a_confirmed_track_may_still_match_at_ordinary_cost(self, iou_tracker):
        """CONFIRMED keeps the full ``max_cost`` budget.

        The ceiling applies to a track that is *unmeasured*, not to one being
        measured — narrowing the live case would fragment ordinary tracking.
        """
        drive(iou_tracker, 8)
        seq = Seq(50)
        before = {
            t.track_id
            for t in iou_tracker.update(
                make_request(seq.next(), [walking_box(8)])
            ).active
        }
        update = iou_tracker.update(
            make_request(seq.next(), [OTHER_PERSON_ALONGSIDE])
        )
        assert before & {t.track_id for t in update.active}, (
            "a measured track was refused a match it was entitled to"
        )

    @pytest.mark.parametrize("empty_frames", [1, 3, 6, 20])
    def test_an_empty_scene_still_settles_to_nothing(self, iou_tracker, empty_frames):
        """No detections means no association at any gate. Unchanged."""
        drive(iou_tracker, 8)
        seq = Seq(50)
        for _ in range(empty_frames):
            update = iou_tracker.update(make_request(seq.next(), []))
        if empty_frames >= 20:
            assert update.active == (), f"tracks survive an empty scene: {update.active}"
