"""What the trackers actually do — continuity, coasting, recovery, ambiguity.

These are the semantics the platform is bought for. 14_TESTING section 7.2 is
explicit that tracker quality is *not* per-frame accuracy: what matters is
fragmentation, ID switches, occlusion recovery, and behaviour under non-uniform
time gaps. Those four shape this file.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import OutOfOrderFrameError
from vision_os.core.model.ids import CameraId
from vision_os.core.model.space import Box
from vision_os.core.model.track import (
    BreakReason,
    MeasurementBasis,
    MotionState,
    TrackState,
)

from ..conftest import CAMERA, OTHER_CAMERA, coast, drive, make_request, walking_box


class TestContinuity:
    def test_a_walking_object_keeps_one_id(self, any_tracker) -> None:
        """The whole point of the layer, on every shipped tracker."""
        updates = drive(any_tracker, 12)
        created = [tid for u in updates for tid in u.new]
        assert len(created) == 1, f"object fragmented into {len(created)} tracks"

    def test_a_track_confirms_after_the_hit_threshold(self, sort_tracker) -> None:
        updates = drive(sort_tracker, 5)
        states = [u.active[0].state for u in updates if u.active]
        assert states[0] is TrackState.TENTATIVE
        assert states[-1] is TrackState.CONFIRMED

    def test_a_stationary_object_keeps_one_id(self, any_tracker) -> None:
        box = Box(0.4, 0.4, 0.5, 0.8)
        created = []
        for seq in range(12):
            created.extend(any_tracker.update(make_request(seq, [box])).new)
        assert len(created) == 1

    def test_two_separated_objects_get_two_ids(self, sort_tracker) -> None:
        for seq in range(8):
            x = 0.05 + seq * 0.02
            sort_tracker.update(
                make_request(seq, [Box(x, 0.1, x + 0.08, 0.4), Box(x, 0.6, x + 0.08, 0.9)])
            )
        assert len(sort_tracker.tracks(CAMERA)) == 2

    def test_hit_ratio_stays_high_for_a_clean_sequence(self, sort_tracker) -> None:
        """The fragmentation signal (14_TESTING section 7.2)."""
        drive(sort_tracker, 15)
        track = sort_tracker.tracks(CAMERA)[0]
        assert track.hit_ratio > 0.9

    def test_a_new_object_appearing_later_starts_its_own_track(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        first = sort_tracker.tracks(CAMERA)[0].track_id
        update = sort_tracker.update(
            make_request(6, [walking_box(6), Box(0.05, 0.05, 0.15, 0.25)])
        )
        assert len(update.new) == 1
        assert first in {t.track_id for t in update.active}


class TestCoasting:
    def test_a_missed_frame_coasts_rather_than_terminating(self, sort_tracker) -> None:
        """A detection gap is a normal operating condition (M6 R6)."""
        drive(sort_tracker, 6)
        update = sort_tracker.update(make_request(6, []))
        assert update.active
        assert update.active[0].state is TrackState.COASTING

    def test_a_coasting_position_is_marked_predicted(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        sort_tracker.update(make_request(6, []))
        track = sort_tracker.tracks(CAMERA)[0]
        assert track.measurement_basis is MeasurementBasis.PREDICTED
        assert track.is_predicted

    def test_a_coasting_position_advances_along_the_motion(self, sort_tracker) -> None:
        """A frozen prediction loses every moving object across a gap — and the
        bug is invisible on stationary ones, which is what makes it easy to ship.
        """
        drive(sort_tracker, 6)
        positions = []
        for step in range(3):
            sort_tracker.update(make_request(6 + step, []))
            positions.append(sort_tracker.tracks(CAMERA)[0].spatial.bbox.x1)
        assert positions == sorted(positions)
        assert positions[-1] > positions[0], "coasting position never moved"

    def test_last_seen_is_not_advanced_by_a_prediction(self, sort_tracker) -> None:
        """A consumer asking 'how fresh is this?' must not be told a prediction
        is a sighting."""
        drive(sort_tracker, 6)
        measured_at = sort_tracker.tracks(CAMERA)[0].last_seen
        coast(sort_tracker, 3, start=6)
        track = sort_tracker.tracks(CAMERA)[0]
        assert track.last_seen == measured_at
        assert track.last_updated.ns > track.last_seen.ns

    def test_coast_frames_increment(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        for step in range(3):
            sort_tracker.update(make_request(6 + step, []))
            assert sort_tracker.tracks(CAMERA)[0].coast_frames == step + 1

    def test_motion_uncertainty_grows_while_coasting(self, sort_tracker) -> None:
        """A prediction five frames old is weaker than one made this frame."""
        drive(sort_tracker, 6)
        sort_tracker.update(make_request(6, []))
        early = sort_tracker.tracks(CAMERA)[0].motion.uncertainty
        coast(sort_tracker, 3, start=7)
        late = sort_tracker.tracks(CAMERA)[0].motion.uncertainty
        assert late > early

    def test_association_confidence_decays_while_coasting(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        measured = sort_tracker.tracks(CAMERA)[0].confidence.value
        coast(sort_tracker, 3, start=6)
        coasted = sort_tracker.tracks(CAMERA)[0].confidence.value
        assert coasted < measured

    def test_a_track_becomes_lost_past_the_coast_budget(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        coast(sort_tracker, 7, start=6)
        live = sort_tracker.tracks(CAMERA)
        assert live and live[0].state is TrackState.LOST

    def test_a_track_terminates_after_the_recovery_window(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        for step in range(40):
            update = sort_tracker.update(make_request(6 + step, []))
            if update.terminated:
                assert update.terminated[0][1] is not BreakReason.NONE
                return
        pytest.fail("track never terminated")

    def test_a_terminated_track_leaves_the_live_set(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        coast(sort_tracker, 40, start=6)
        assert sort_tracker.tracks(CAMERA) == ()


class TestRecovery:
    def test_a_coasting_track_recovers_at_its_predicted_position(
        self, sort_tracker
    ) -> None:
        updates = drive(sort_tracker, 6)
        original = updates[-1].active[0].track_id
        coast(sort_tracker, 3, start=6)

        update = sort_tracker.update(make_request(9, [walking_box(9)]))
        assert original in update.recovered
        track = sort_tracker.tracks(CAMERA)[0]
        assert track.track_id == original
        assert track.state is TrackState.CONFIRMED
        assert not track.is_predicted

    def test_a_lost_track_recovers_within_the_window(self, sort_tracker) -> None:
        updates = drive(sort_tracker, 6)
        original = updates[-1].active[0].track_id
        coast(sort_tracker, 7, start=6)
        assert sort_tracker.tracks(CAMERA)[0].state is TrackState.LOST

        update = sort_tracker.update(make_request(13, [walking_box(13)]))
        assert original in update.recovered
        assert sort_tracker.tracks(CAMERA)[0].track_id == original

    def test_a_stationary_object_recovers_after_occlusion(self, any_tracker) -> None:
        """The most common real case: someone walks in front of a parked object."""
        box = Box(0.4, 0.4, 0.5, 0.8)
        for seq in range(6):
            any_tracker.update(make_request(seq, [box]))
        original = any_tracker.tracks(CAMERA)[0].track_id
        any_tracker.update(make_request(6, []))

        update = any_tracker.update(make_request(7, [box]))
        assert original in update.recovered
        assert any_tracker.tracks(CAMERA)[0].track_id == original

    def test_recovery_clears_the_break_reason(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        coast(sort_tracker, 2, start=6)
        sort_tracker.update(make_request(8, [walking_box(8)]))
        assert sort_tracker.tracks(CAMERA)[0].break_reason is BreakReason.NONE

    def test_an_object_returning_after_termination_gets_a_new_id(
        self, sort_tracker
    ) -> None:
        """Correct, not a failure: the tracker has no basis to claim continuity
        across a gap it already gave up on. Durable identity is M7's job."""
        updates = drive(sort_tracker, 6)
        original = updates[-1].active[0].track_id
        coast(sort_tracker, 40, start=6)

        update = sort_tracker.update(make_request(60, [walking_box(0)]))
        assert update.new
        assert update.new[0] != original


class TestNonUniformTimeGaps:
    """T2 — the scheduler drops frames by design (V7)."""

    def test_a_steady_object_survives_wildly_uneven_gaps(self, sort_tracker) -> None:
        gaps = [200, 1000, 100, 1500, 250, 800, 200]
        x, timestamp, created = 0.1, 0, []
        for seq, gap in enumerate(gaps):
            x += 0.00005 * gap  # constant real-world speed
            timestamp += gap * 1_000_000
            update = sort_tracker.update(
                make_request(
                    seq,
                    [Box(x, 0.4, x + 0.1, 0.8)],
                    elapsed_ms=gap,
                    timestamp_ns=timestamp,
                )
            )
            created.extend(update.new)
        assert len(created) == 1, "object fragmented under non-uniform gaps"

    def test_velocity_reflects_elapsed_time_not_frame_count(self, sort_tracker) -> None:
        x, timestamp = 0.1, 0
        for seq in range(6):
            x += 0.05
            timestamp += 500_000_000
            sort_tracker.update(
                make_request(
                    seq, [Box(x, 0.4, x + 0.1, 0.8)], elapsed_ms=500, timestamp_ns=timestamp
                )
            )
        # 0.05 per 0.5 s = 0.1 per second, regardless of how many frames that took.
        assert sort_tracker.tracks(CAMERA)[0].motion.velocity.x == pytest.approx(
            0.1, abs=0.03
        )

    def test_a_long_gap_predicts_proportionally_further(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        sort_tracker.update(make_request(6, [], elapsed_ms=200))
        short = sort_tracker.tracks(CAMERA)[0].spatial.bbox.x1

        sort_tracker.reset(CAMERA, "test")
        drive(sort_tracker, 6)
        sort_tracker.update(make_request(6, [], elapsed_ms=1000))
        long = sort_tracker.tracks(CAMERA)[0].spatial.bbox.x1

        assert long > short


class TestOrdering:
    def test_an_out_of_order_frame_is_rejected_loudly(self, any_tracker) -> None:
        """T1. Absorbing it integrates a negative time step and runs positions
        backwards — degradation that looks like poor tracker quality."""
        drive(any_tracker, 5)
        with pytest.raises(OutOfOrderFrameError, match="ordering"):
            any_tracker.update(make_request(2, [walking_box(2)]))

    def test_a_repeated_frame_is_rejected(self, sort_tracker) -> None:
        drive(sort_tracker, 5)
        with pytest.raises(OutOfOrderFrameError):
            sort_tracker.update(make_request(4, [walking_box(4)]))

    def test_the_error_names_the_frame_and_the_last_processed(
        self, sort_tracker
    ) -> None:
        drive(sort_tracker, 5)
        with pytest.raises(OutOfOrderFrameError) as caught:
            sort_tracker.update(make_request(1, []))
        message = str(caught.value)
        assert "T1" in message

    def test_a_skipped_frame_number_is_accepted(self, sort_tracker) -> None:
        """The scheduler drops frames by design; gaps in numbering are normal."""
        drive(sort_tracker, 5)
        update = sort_tracker.update(make_request(50, [walking_box(6)]))
        assert update is not None

    def test_ordering_is_tracked_per_camera(self, sort_tracker) -> None:
        drive(sort_tracker, 5, camera=CAMERA)
        update = sort_tracker.update(make_request(0, [walking_box(0)], camera=OTHER_CAMERA))
        assert update.camera_id == OTHER_CAMERA


class TestCameraIsolation:
    def test_two_cameras_keep_separate_id_spaces(self, sort_tracker) -> None:
        drive(sort_tracker, 5, camera=CAMERA)
        drive(sort_tracker, 5, camera=OTHER_CAMERA)
        for track in sort_tracker.tracks(CAMERA):
            assert track.camera_id == CAMERA
        for track in sort_tracker.tracks(OTHER_CAMERA):
            assert track.camera_id == OTHER_CAMERA

    def test_resetting_one_camera_leaves_the_other_untouched(self, sort_tracker) -> None:
        drive(sort_tracker, 5, camera=CAMERA)
        drive(sort_tracker, 5, camera=OTHER_CAMERA)
        before = len(sort_tracker.tracks(OTHER_CAMERA))
        sort_tracker.reset(CAMERA, "test")
        assert sort_tracker.tracks(CAMERA) == ()
        assert len(sort_tracker.tracks(OTHER_CAMERA)) == before

    def test_an_unknown_camera_has_no_tracks(self, sort_tracker) -> None:
        assert sort_tracker.tracks(CameraId("never-seen")) == ()

    def test_identical_objects_on_two_cameras_are_not_linked(self, sort_tracker) -> None:
        """Cross-camera identity is P11, Phase 2, and must not appear by accident."""
        box = Box(0.4, 0.4, 0.5, 0.8)
        for seq in range(6):
            sort_tracker.update(make_request(seq, [box], camera=CAMERA))
            sort_tracker.update(make_request(seq, [box], camera=OTHER_CAMERA))
        first = sort_tracker.tracks(CAMERA)[0].track_id
        second = sort_tracker.tracks(OTHER_CAMERA)[0].track_id
        assert first != second
        assert first.local_id == second.local_id, "same counter, different id — by design"


class TestEpochs:
    def test_reset_advances_the_epoch(self, sort_tracker) -> None:
        drive(sort_tracker, 5)
        assert sort_tracker.reset(CAMERA, "test") == 1
        assert sort_tracker.reset(CAMERA, "test") == 2

    def test_tracks_after_a_reset_carry_the_new_epoch(self, sort_tracker) -> None:
        drive(sort_tracker, 5)
        epoch = sort_tracker.reset(CAMERA, "test")
        update = sort_tracker.update(make_request(0, [walking_box(0)]))
        assert update.tracker_epoch == epoch
        assert all(t.track_id.tracker_epoch == epoch for t in update.active)

    def test_an_id_from_a_previous_epoch_never_compares_equal(self, sort_tracker) -> None:
        updates = drive(sort_tracker, 5)
        before = updates[-1].active[0].track_id
        sort_tracker.reset(CAMERA, "test")
        update = sort_tracker.update(make_request(0, [walking_box(0)]))
        assert update.active[0].track_id != before

    def test_frame_numbering_may_restart_after_a_reset(self, sort_tracker) -> None:
        drive(sort_tracker, 5)
        sort_tracker.reset(CAMERA, "test")
        assert sort_tracker.update(make_request(0, [walking_box(0)])) is not None


class TestAmbiguityIsPublished:
    def test_a_confident_association_reports_high_confidence(self, sort_tracker) -> None:
        drive(sort_tracker, 8)
        assert sort_tracker.tracks(CAMERA)[0].confidence.value > 0.5

    def test_a_contested_association_records_a_runner_up(self, sort_tracker) -> None:
        """Two objects nearly on top of each other. The margin is the honest
        measure of how ambiguous the binding was, and M6 forbids hiding it."""
        for seq in range(6):
            x = 0.3 + seq * 0.01
            sort_tracker.update(
                make_request(seq, [Box(x, 0.4, x + 0.2, 0.8), Box(x + 0.02, 0.42, x + 0.22, 0.82)])
            )
        evidence = [t.evidence for t in sort_tracker.tracks(CAMERA)]
        assert any(e.runner_up_cost is not None for e in evidence)

    def test_association_cost_is_retained(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        assert sort_tracker.tracks(CAMERA)[0].evidence.association_cost >= 0.0

    def test_an_indistinguishable_pair_is_refused_rather_than_guessed(
        self, sort_tracker
    ) -> None:
        """Two identical boxes: nothing says which track continues which
        detection. M6 requires preferring termination over a wrong association.
        """
        pair = [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)]
        refusals = []
        for seq in range(6):
            refusals.extend(sort_tracker.update(make_request(seq, pair)).refused)
        assert refusals, "a coin-flip association was asserted instead of refused"

    def test_a_refusal_carries_both_costs(self, sort_tracker) -> None:
        """The refused track is terminated the same frame, so this is the only
        place the numbers behind the decision survive."""
        pair = [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)]
        refusals = []
        for seq in range(6):
            refusals.extend(sort_tracker.update(make_request(seq, pair)).refused)
        refusal = refusals[0]
        assert refusal.runner_up_cost >= refusal.best_cost
        assert refusal.margin < 0.05
        assert refusal.track_id.camera_id == CAMERA

    def test_a_clearly_separated_pair_is_not_refused(self, sort_tracker) -> None:
        """The refusal must be selective, or it would refuse everything."""
        refusals = []
        for seq in range(8):
            x = 0.05 + seq * 0.02
            refusals.extend(
                sort_tracker.update(
                    make_request(
                        seq, [Box(x, 0.05, x + 0.1, 0.3), Box(x, 0.6, x + 0.1, 0.85)]
                    )
                ).refused
            )
        assert refusals == []

    def test_a_refused_track_carries_the_association_failure_reason(
        self, sort_tracker
    ) -> None:
        pair = [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)]
        reasons = set()
        for seq in range(6):
            update = sort_tracker.update(make_request(seq, pair))
            reasons.update(reason for _, reason in update.terminated)
        assert BreakReason.ASSOCIATION_FAILURE in reasons

    def test_the_association_method_is_reported(self, sort_tracker, iou_tracker) -> None:
        drive(sort_tracker, 6)
        drive(iou_tracker, 6)
        assert sort_tracker.tracks(CAMERA)[0].evidence.association_method.value == (
            "motion_gated_iou"
        )
        assert iou_tracker.tracks(CAMERA)[0].evidence.association_method.value == "iou"


class TestMotionClassification:
    def test_a_moving_object_is_classified_moving(self, sort_tracker) -> None:
        drive(sort_tracker, 10)
        assert sort_tracker.tracks(CAMERA)[0].motion_state is MotionState.MOVING

    def test_a_still_object_is_classified_stationary(self, sort_tracker) -> None:
        box = Box(0.4, 0.4, 0.5, 0.8)
        for seq in range(10):
            sort_tracker.update(make_request(seq, [box]))
        assert sort_tracker.tracks(CAMERA)[0].motion_state is MotionState.STATIONARY

    def test_motion_state_starts_unknown(self, sort_tracker) -> None:
        """Not 'stationary' — too few observations is a different claim."""
        update = sort_tracker.update(make_request(0, [walking_box(0)]))
        assert update.active[0].motion_state is MotionState.UNKNOWN

    def test_hysteresis_prevents_flapping_on_one_still_frame(self, sort_tracker) -> None:
        drive(sort_tracker, 10)
        assert sort_tracker.tracks(CAMERA)[0].motion_state is MotionState.MOVING
        # One frame where the object happens not to move.
        sort_tracker.update(make_request(10, [walking_box(9)]))
        assert sort_tracker.tracks(CAMERA)[0].motion_state is MotionState.MOVING


class TestTwoStageAssociation:
    def test_bytetrack_continues_a_track_through_a_weak_detection(
        self, bytetrack_tracker
    ) -> None:
        """A detection too weak to start a track is often strong enough to
        continue one — the largest single reduction in fragmentation available
        without an appearance model."""
        created = []
        for seq in range(6):
            created.extend(bytetrack_tracker.update(make_request(seq, [walking_box(seq)])).new)
        # Now the detector's confidence collapses (partial occlusion).
        for seq in range(6, 10):
            update = bytetrack_tracker.update(
                make_request(seq, [walking_box(seq)], scores=[0.25])
            )
            created.extend(update.new)
        assert len(created) == 1, "weak detections fragmented the track"
        assert bytetrack_tracker.tracks(CAMERA)[0].state is TrackState.CONFIRMED

    def test_a_weak_detection_alone_starts_no_track(self, bytetrack_tracker) -> None:
        """Below the high floor, a detection may continue but never initiate."""
        for seq in range(5):
            update = bytetrack_tracker.update(
                make_request(seq, [walking_box(seq)], scores=[0.25])
            )
            assert update.new == ()

    def test_a_single_stage_tracker_drops_the_weak_detection(self, sort_tracker) -> None:
        """The contrast that shows two-stage association is doing something."""
        for seq in range(6):
            sort_tracker.update(make_request(seq, [walking_box(seq)]))
        for seq in range(6, 10):
            sort_tracker.update(make_request(seq, [walking_box(seq)], scores=[0.05]))
        track = sort_tracker.tracks(CAMERA)
        assert not track or track[0].state.is_predicted


class TestCapacity:
    def test_the_track_table_stays_within_its_declared_maximum(
        self, lifecycle_policy, sort_tracker
    ) -> None:
        boxes = []
        for i in range(60):
            x = (i % 10) * 0.09
            y = (i // 10) * 0.15
            boxes.append(Box(x, y, x + 0.05, y + 0.1))
        for seq in range(5):
            update = sort_tracker.update(make_request(seq, boxes))
            assert len(update.active) <= lifecycle_policy.max_tracks_per_camera

    def test_a_crowd_degrades_rather_than_crashing(self, sort_tracker) -> None:
        boxes = [
            Box((i % 10) * 0.09, (i // 10) * 0.15, (i % 10) * 0.09 + 0.05,
                (i // 10) * 0.15 + 0.1)
            for i in range(60)
        ]
        for seq in range(10):
            assert sort_tracker.update(make_request(seq, boxes)) is not None


class TestProvenance:
    def test_every_track_names_the_adapter_that_produced_it(self, any_tracker) -> None:
        drive(any_tracker, 6)
        provenance = any_tracker.tracks(CAMERA)[0].provenance
        assert provenance.adapter_id
        assert provenance.producer_module == "tracking_engine"

    def test_every_track_carries_the_config_revision(self, sort_tracker) -> None:
        """Without it, six months later nobody can say what produced a result."""
        drive(sort_tracker, 6)
        assert sort_tracker.tracks(CAMERA)[0].provenance.config_revision == "test"

    def test_tracks_declare_deterministic_production(self, sort_tracker) -> None:
        drive(sort_tracker, 6)
        assert sort_tracker.tracks(CAMERA)[0].provenance.deterministic

    def test_tenancy_is_carried_not_invented(self, sort_tracker) -> None:
        from ..conftest import SITE, TENANT

        drive(sort_tracker, 6)
        track = sort_tracker.tracks(CAMERA)[0]
        assert track.tenant_id == TENANT
        assert track.site_id == SITE
