"""A LOST track must not be resurrected onto a different person.

Measured in production before this was fixed:

    tracking.created     +59
    tracking.recovered  +135      (twice the creation rate)
    tracking.terminated    0      (across 301 consumed frames)

Every state shared one association gate, so a LOST track for someone who had
walked out could win the nearest new detection on ordinary cost and be
"recovered" onto whoever appeared next. A track that keeps being re-matched
never accumulates the consecutive misses termination needs, so nothing ever
terminated, active objects grew without bound, and PPE attributes belonging to
one person were carried onto another.

These tests pin the three properties that failure violated.
"""

from __future__ import annotations

from vision_os.core.model.space import Box
from vision_os.perception.tracking.lifecycle import TrackState

from tests.vision_os.tracking.conftest import coast, drive, make_request

#: Far enough that no honest observer would call it the same person walking —
#: opposite side of the frame, no overlap with where the first one was.
ELSEWHERE = Box(0.75, 0.05, 0.85, 0.45)


class Seq:
    """Monotonic frame numbers.

    The tracker enforces per-camera ordering (obligation T1) and raises on a
    frame that goes backwards, so a test must never reuse or rewind a sequence
    number. Handing out numbers from one counter keeps every helper honest.
    """

    def __init__(self, start: int = 0) -> None:
        self._n = start

    def next(self) -> int:
        self._n += 1
        return self._n


def _live_ids(tracker, seq: Seq):
    """Track ids the tracker still considers live, without disturbing order."""
    update = tracker.update(make_request(seq.next(), []))
    return {t.track_id for t in update.active}


class TestLostCannotStealADetection:
    def test_a_lost_track_does_not_recover_onto_a_different_person(
        self, sort_tracker, lifecycle_policy
    ):
        """TEST C. The defect, exactly.

        Person A is tracked, then leaves. Once A is LOST, person B appears
        somewhere else entirely. A must not be handed B's detection.
        """
        seq = Seq()
        drive(sort_tracker, 8)
        seq = Seq(50)
        first = _live_ids(sort_tracker, seq)
        assert first, "person A was never tracked"

        # Miss for long enough to pass COASTING and become LOST, but not so
        # long that the lost window expires — the exact window in which the
        # old code performed its wrong recovery.
        coast(sort_tracker, lifecycle_policy.max_coast_frames + 2, start=seq.next())
        seq = Seq(200)

        # Person B, on the other side of the frame.
        update = sort_tracker.update(make_request(seq.next(), [ELSEWHERE]))

        assert not update.recovered, (
            "a LOST track was recovered onto a detection that is not the same "
            f"person: {update.recovered}"
        )
        new_ids = {t.track_id for t in update.active} - first
        assert new_ids, "person B did not receive a track of their own"

    def test_the_lost_track_terminates_instead(self, sort_tracker, lifecycle_policy):
        """TEST D. Denying recovery must actually end the track.

        Verified rather than assumed: `lost_frames = coast_frames -
        max_coast_frames`, and `coast_frames` only ever increases while a track
        goes unmatched, so a track that can no longer be matched must reach
        TERMINATED.
        """
        drive(sort_tracker, 8)
        seq = Seq(50)
        before = _live_ids(sort_tracker, seq)

        terminated: list = []
        total = lifecycle_policy.max_coast_frames + lifecycle_policy.max_lost_frames + 3
        for step in range(total):
            # An unrelated person is present throughout, so the old track has
            # something to wrongly grab at on every single frame.
            update = sort_tracker.update(make_request(seq.next(), [ELSEWHERE]))
            terminated.extend(update.terminated)

        assert terminated, (
            "no track terminated while an unrelated detection was available — "
            "the LOST track is still being kept alive by wrong associations"
        )
        assert before & {track_id for track_id, _ in terminated}, (
            "the terminated track was not the departed person's"
        )

    def test_a_brief_miss_by_the_same_person_still_recovers(self, sort_tracker):
        """COASTING is untouched. This is the behaviour the fix must not break.

        A detector that drops one or two frames on a person who is still there
        must not cost them their identity — that is what the coast window is
        for. Only tracks already declared LOST lose the right to recover.
        """
        drive(sort_tracker, 8)
        seq = Seq(50)
        before = _live_ids(sort_tracker, seq)

        coast(sort_tracker, 2, start=seq.next())
        seq = Seq(60)

        # The same person, continuing along the same path.
        update = sort_tracker.update(
            make_request(seq.next(), [Box(0.42, 0.4, 0.52, 0.8)])
        )
        after = {t.track_id for t in update.active}

        assert before & after, (
            "a person who was briefly missed lost their identity — the coast "
            "window has been broken"
        )


class TestNoIdentityContamination:
    def test_person_b_does_not_inherit_person_a_identity(
        self, sort_tracker, lifecycle_policy
    ):
        """TEST G, at the identity layer.

        Attribute inheritance downstream is a consequence of track identity
        reuse: M7 keys attributes by the object the track produced. If B never
        receives A's track id, B cannot receive A's PPE history.
        """
        drive(sort_tracker, 8)
        seq = Seq(50)
        a_ids = _live_ids(sort_tracker, seq)

        coast(sort_tracker, lifecycle_policy.max_coast_frames + 2, start=seq.next())
        seq = Seq(200)

        # Person B is present from here on. Run past the lost window so A's
        # track resolves one way or the other rather than being judged while
        # it is still legitimately coasting.
        recovered: list = []
        total = lifecycle_policy.max_lost_frames + 4
        for _ in range(total):
            update = sort_tracker.update(make_request(seq.next(), [ELSEWHERE]))
            recovered.extend(update.recovered)

        # A's identity must never have been handed to B's detection. This is
        # the contamination itself — `active` containing both A (coasting out)
        # and B (newly created) at the same moment is co-existence, not reuse.
        assert not (set(recovered) & a_ids), (
            f"person A's identity was recovered onto person B: {set(recovered) & a_ids}"
        )

        final = _live_ids(sort_tracker, seq)
        assert final, "person B is not being tracked"
        assert not (final & a_ids), (
            f"person A's track outlived its window and still holds an identity: {final & a_ids}"
        )


class TestEmptySceneSettles:
    def test_an_empty_scene_ends_with_no_active_tracks(
        self, sort_tracker, lifecycle_policy
    ):
        """A kitchen nobody is standing in must hold no live subjects.

        This is what stops a departed person continuing to be a compliance
        subject, and with it the empty-room alerts.
        """
        drive(sort_tracker, 8)
        seq = Seq(50)
        assert _live_ids(sort_tracker, seq), "nobody was tracked to begin with"

        total = lifecycle_policy.max_coast_frames + lifecycle_policy.max_lost_frames + 3
        updates = coast(sort_tracker, total, start=seq.next())

        assert any(u.terminated for u in updates), "nothing terminated on an empty scene"
        assert updates[-1].active == (), (
            f"tracks survive an empty scene: {updates[-1].active}"
        )


class TestWallClockHorizon:
    """A departed person must not stay live because frames arrive slowly.

    The frame horizons (`max_coast_frames` + `max_lost_frames` = 20) assume a
    known frame rate. Live analysis measured ~0.5 fps, at which those 20 frames
    are 40+ seconds — and on a slower camera, minutes. For the whole of that
    window a person who had left remained a compliance subject.
    """

    @staticmethod
    def _policy(**overrides):
        from vision_os.perception.tracking.lifecycle import LifecyclePolicy

        base = {
            "min_hits_to_confirm": 3,
            "max_coast_frames": 5,
            "max_lost_frames": 10,
            "max_age_frames": 1_000,
            "max_tracks_per_camera": 32,
        }
        return LifecyclePolicy(**{**base, **overrides})

    @staticmethod
    def _machine(policy):
        from vision_os.perception.tracking.lifecycle import LifecycleMachine

        return LifecycleMachine(policy)

    def test_a_slow_camera_terminates_on_elapsed_time(self):
        """TEST A. Far fewer than 20 missed frames, but a long time gone."""
        from vision_os.perception.tracking.lifecycle import TrackState

        policy = self._policy(max_unmeasured_ns=30_000_000_000)
        machine = self._machine(policy)

        transition = machine.on_miss(
            state=TrackState.COASTING,
            coast_frames=2,          # nowhere near the frame horizon
            age_frames=10,
            since_measurement_ns=31_000_000_000,   # 31s unmeasured
        )
        assert transition.current is TrackState.TERMINATED
        assert transition.is_terminal

    def test_a_normal_frame_rate_still_uses_the_frame_horizon(self):
        """TEST B. The guard only ever shortens a life; it never extends one.

        Two frames missed at a healthy rate is a blink, and the track must
        stay recoverable exactly as before.
        """
        from vision_os.perception.tracking.lifecycle import TrackState

        machine = self._machine(self._policy(max_unmeasured_ns=30_000_000_000))
        transition = machine.on_miss(
            state=TrackState.CONFIRMED,
            coast_frames=1,
            age_frames=10,
            since_measurement_ns=250_000_000,      # 0.25s
        )
        assert transition.current is TrackState.COASTING
        assert not transition.is_terminal

    def test_a_brief_occlusion_is_not_terminated(self):
        """TEST C. Someone walks in front of a chef for a couple of seconds."""
        from vision_os.perception.tracking.lifecycle import TrackState

        machine = self._machine(self._policy(max_unmeasured_ns=30_000_000_000))
        transition = machine.on_miss(
            state=TrackState.COASTING,
            coast_frames=3,
            age_frames=20,
            since_measurement_ns=2_000_000_000,    # 2s
        )
        assert not transition.is_terminal, "a brief occlusion ended a track"

    def test_the_guard_can_be_disabled(self):
        """0 restores pure frame-counted behaviour, for a deployment that
        wants the original semantics."""
        from vision_os.perception.tracking.lifecycle import TrackState

        machine = self._machine(self._policy(max_unmeasured_ns=0))
        transition = machine.on_miss(
            state=TrackState.COASTING,
            coast_frames=2,
            age_frames=10,
            since_measurement_ns=10 * 60 * 1_000_000_000,   # ten minutes
        )
        assert not transition.is_terminal

    def test_a_confirmed_track_is_not_spared_by_its_state(self):
        """TEST D. Gone long enough is gone, whatever state it was in."""
        from vision_os.perception.tracking.lifecycle import TrackState

        machine = self._machine(self._policy(max_unmeasured_ns=30_000_000_000))
        for state in (TrackState.CONFIRMED, TrackState.COASTING, TrackState.LOST):
            transition = machine.on_miss(
                state=state,
                coast_frames=1,
                age_frames=10,
                since_measurement_ns=45_000_000_000,
            )
            assert transition.is_terminal, f"{state} survived the wall-clock horizon"
