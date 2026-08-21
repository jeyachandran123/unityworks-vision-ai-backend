"""The Track object model — what a track may and may not claim.

Most of these assert *refusals*. A track that can be constructed in an incoherent
state is one every downstream stage must defend against; making the state
unconstructible moves the check from N consumers to one constructor.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    ConfigRevision,
    FrameRef,
    FrameSeq,
    LocalTrackId,
    ModuleId,
    StreamEpoch,
    TrackerEpoch,
    TrackId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box, FrameOfReference, Point, SpatialInfo
from vision_os.core.model.timebase import Instant
from vision_os.core.model.track import (
    Association,
    AssociationMethod,
    BreakReason,
    MeasurementBasis,
    MotionEstimate,
    MotionState,
    Track,
    TrackEvidence,
    TrackState,
    TrackUpdate,
)

from ..conftest import PERSON, SITE, TENANT

CAMERA = CameraId("cam-01")
EPOCH = TrackerEpoch(0)


def make_track_id(local: int = 0, *, camera: CameraId = CAMERA, epoch: int = 0) -> TrackId:
    return TrackId(camera, TrackerEpoch(epoch), LocalTrackId(local))


def make_track(**overrides) -> Track:
    track_id = overrides.pop("track_id", make_track_id())
    defaults = dict(
        track_id=track_id,
        camera_id=track_id.camera_id,
        tenant_id=TENANT,
        site_id=SITE,
        state=TrackState.CONFIRMED,
        class_id=PERSON,
        confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.ASSOCIATION),
        spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.3, 0.5)
        ),
        measurement_basis=MeasurementBasis.MEASURED,
        motion=MotionEstimate(),
        motion_state=MotionState.UNKNOWN,
        first_seen=Instant(0),
        last_seen=Instant(1_000),
        last_updated=Instant(1_000),
        age_frames=5,
        hit_count=5,
        coast_frames=0,
        detections=(FrameRef(CAMERA, StreamEpoch(1), FrameSeq(0)),),
        evidence=TrackEvidence(association_method=AssociationMethod.IOU),
        provenance=Provenance(
            producer_module=ModuleId("tracking_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
    )
    defaults.update(overrides)
    return Track(**defaults)


class TestTrackId:
    """A composite id is what stops a track handle becoming an identity."""

    def test_carries_camera_epoch_and_local_id(self) -> None:
        track_id = make_track_id(7, epoch=3)
        assert track_id.camera_id == CAMERA
        assert track_id.tracker_epoch == 3
        assert track_id.local_id == 7

    def test_same_local_id_on_two_cameras_is_not_equal(self) -> None:
        """A bare integer would compare equal here — the corruption V10 prevents."""
        first = make_track_id(5, camera=CameraId("cam-01"))
        second = make_track_id(5, camera=CameraId("cam-02"))
        assert first != second

    def test_same_local_id_in_two_epochs_is_not_equal(self) -> None:
        """After a reset, a recycled local id must not appear to continue a track."""
        assert make_track_id(5, epoch=0) != make_track_id(5, epoch=1)

    def test_same_epoch_as_detects_comparability(self) -> None:
        assert make_track_id(1).same_epoch_as(make_track_id(2))
        assert not make_track_id(1).same_epoch_as(make_track_id(1, epoch=1))
        assert not make_track_id(1).same_epoch_as(make_track_id(1, camera=CameraId("cam-02")))

    def test_negative_epoch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tracker_epoch"):
            TrackId(CAMERA, TrackerEpoch(-1), LocalTrackId(0))

    def test_negative_local_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="local_id"):
            TrackId(CAMERA, EPOCH, LocalTrackId(-1))

    def test_str_is_readable_and_unambiguous(self) -> None:
        assert str(make_track_id(88, epoch=3)) == "cam-01/t3/#88"

    def test_is_hashable_and_orderable(self) -> None:
        ids = {make_track_id(2), make_track_id(1), make_track_id(1)}
        assert len(ids) == 2
        assert sorted(ids)[0].local_id == 1


class TestTrackConstruction:
    def test_a_valid_track_constructs(self) -> None:
        assert make_track().state is TrackState.CONFIRMED

    def test_association_confidence_is_required(self) -> None:
        """T4 — a detector's presence score measures something else entirely."""
        with pytest.raises(ValueError, match="ASSOCIATION"):
            make_track(
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.DETECTION_PRESENCE
                )
            )

    def test_identity_confidence_is_also_refused(self) -> None:
        """Identity is M7's claim, not a track's."""
        with pytest.raises(ValueError, match="ASSOCIATION"):
            make_track(
                confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.IDENTITY)
            )

    def test_track_id_camera_must_match_the_track(self) -> None:
        with pytest.raises(ValueError, match="names camera"):
            make_track(track_id=make_track_id(camera=CameraId("cam-99")), camera_id=CAMERA)

    def test_a_box_is_required(self) -> None:
        with pytest.raises(ValueError, match="bounding box"):
            make_track(spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED))

    def test_box_outside_unit_space_is_refused(self) -> None:
        with pytest.raises(ValueError, match="normalized"):
            make_track(
                spatial=SpatialInfo(
                    frame_of_reference=FrameOfReference.NORMALIZED,
                    bbox=Box(0.1, 0.1, 1.4, 0.5),
                )
            )

    def test_hit_count_cannot_exceed_age(self) -> None:
        """A track cannot have been measured more often than it existed."""
        with pytest.raises(ValueError, match="exceeds age_frames"):
            make_track(age_frames=3, hit_count=5)

    def test_negative_counters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_track(coast_frames=-1)

    def test_last_seen_cannot_follow_last_updated(self) -> None:
        with pytest.raises(ValueError, match="last_seen"):
            make_track(last_seen=Instant(5_000), last_updated=Instant(1_000))

    def test_empty_taxonomy_class_is_permitted_but_state_is_not_invented(self) -> None:
        """The model does not police class vocabulary — the taxonomy registry does."""
        assert make_track(class_id=ClassId("vehicle.forklift")).class_id


class TestPredictedPositionsAreMarked:
    """T5 / V8 at object scale — the corruption no consumer can detect alone."""

    def test_coasting_track_claiming_measurement_is_refused(self) -> None:
        with pytest.raises(ValueError, match="V8"):
            make_track(
                state=TrackState.COASTING,
                coast_frames=2,
                measurement_basis=MeasurementBasis.MEASURED,
            )

    def test_lost_track_claiming_measurement_is_refused(self) -> None:
        with pytest.raises(ValueError, match="V8"):
            make_track(
                state=TrackState.LOST,
                coast_frames=8,
                measurement_basis=MeasurementBasis.MEASURED,
            )

    def test_coasting_track_with_predicted_basis_is_accepted(self) -> None:
        track = make_track(
            state=TrackState.COASTING,
            coast_frames=2,
            measurement_basis=MeasurementBasis.PREDICTED,
        )
        assert track.is_predicted

    def test_coasting_requires_at_least_one_coasted_frame(self) -> None:
        with pytest.raises(ValueError, match="coasted at least one frame"):
            make_track(
                state=TrackState.COASTING,
                coast_frames=0,
                measurement_basis=MeasurementBasis.PREDICTED,
            )

    def test_confirmed_measured_track_is_not_predicted(self) -> None:
        assert not make_track().is_predicted


class TestTrackCarriesNoIdentity:
    """The absences are the design (invariant V10)."""

    def test_track_has_no_object_id(self) -> None:
        assert not hasattr(make_track(), "object_id")

    def test_track_has_no_person_or_name_field(self) -> None:
        fields = set(Track.__dataclass_fields__)
        for forbidden in ("person_id", "name", "identity", "face_id", "global_id"):
            assert forbidden not in fields

    def test_track_has_no_cross_camera_field(self) -> None:
        fields = set(Track.__dataclass_fields__)
        for forbidden in ("cameras", "camera_ids", "site_track_id", "global_track_id"):
            assert forbidden not in fields

    def test_track_has_no_embedding_field(self) -> None:
        """Appearance embeddings are C2 biometric data (12_SECURITY section 4.3)."""
        fields = set(Track.__dataclass_fields__)
        for forbidden in ("embedding", "appearance", "features", "descriptor"):
            assert forbidden not in fields


class TestDerivedProperties:
    def test_hit_ratio_is_the_fragmentation_signal(self) -> None:
        assert make_track(age_frames=10, hit_count=7).hit_ratio == pytest.approx(0.7)

    def test_hit_ratio_of_a_new_track_is_zero_not_an_error(self) -> None:
        assert make_track(age_frames=0, hit_count=0).hit_ratio == 0.0

    def test_lifetime_spans_first_to_last_update(self) -> None:
        track = make_track(first_seen=Instant(1_000), last_seen=Instant(5_000),
                           last_updated=Instant(5_000))
        assert track.lifetime().ns == 4_000

    def test_staleness_measures_from_the_last_measurement_not_the_last_update(self) -> None:
        """A prediction is not a sighting; a consumer must be able to tell."""
        track = make_track(
            state=TrackState.COASTING,
            coast_frames=3,
            measurement_basis=MeasurementBasis.PREDICTED,
            last_seen=Instant(1_000),
            last_updated=Instant(4_000),
        )
        assert track.staleness(Instant(5_000)).ns == 4_000

    def test_staleness_never_goes_negative(self) -> None:
        assert make_track(last_seen=Instant(9_000), last_updated=Instant(9_000)).staleness(
            Instant(1_000)
        ).ns == 0

    def test_tracker_epoch_is_exposed(self) -> None:
        assert make_track(track_id=make_track_id(epoch=4)).tracker_epoch == 4

    def test_is_a_matches_hierarchically(self) -> None:
        track = make_track(class_id=ClassId("vehicle.forklift"))
        assert track.is_a(ClassId("vehicle"))
        assert track.is_a(ClassId("vehicle.forklift"))
        assert not track.is_a(ClassId("person"))

    def test_is_a_does_not_match_a_prefix_that_is_not_a_path_segment(self) -> None:
        assert not make_track(class_id=ClassId("vehicles")).is_a(ClassId("vehicle"))


class TestTrackState:
    def test_only_terminated_is_dead(self) -> None:
        for state in TrackState:
            assert state.is_alive is (state is not TrackState.TERMINATED)

    def test_coasting_and_lost_are_predicted_states(self) -> None:
        assert TrackState.COASTING.is_predicted
        assert TrackState.LOST.is_predicted
        assert not TrackState.CONFIRMED.is_predicted
        assert not TrackState.TENTATIVE.is_predicted

    def test_the_state_set_is_exactly_the_architecture_s_five(self) -> None:
        """03_MODULES M6 R2. ``NEW`` and ``RECOVERED`` are events, not states."""
        assert {s.value for s in TrackState} == {
            "tentative",
            "confirmed",
            "coasting",
            "lost",
            "terminated",
        }


class TestMotionEstimate:
    def test_defaults_to_no_motion_and_unknown_acceleration(self) -> None:
        motion = MotionEstimate()
        assert motion.speed == 0.0
        assert motion.acceleration is None, (
            "acceleration must be None until measurable; zero is a different claim"
        )

    def test_heading_is_none_rather_than_zero_when_at_rest(self) -> None:
        """At rest, atan2 returns a precise angle derived entirely from noise."""
        assert MotionEstimate().heading_degrees is None

    def test_negative_speed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            MotionEstimate(speed=-1.0)

    def test_negative_uncertainty_is_refused(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            MotionEstimate(uncertainty=-0.1)

    def test_heading_outside_the_circle_is_refused(self) -> None:
        with pytest.raises(ValueError, match="heading"):
            MotionEstimate(heading_degrees=400.0)

    def test_velocity_is_a_point_per_second(self) -> None:
        motion = MotionEstimate(velocity=Point(0.1, -0.05), speed=0.112)
        assert motion.velocity.x == pytest.approx(0.1)


class TestMotionStateIsDescriptive:
    def test_no_judgment_states_exist(self) -> None:
        """``loitering`` and ``queueing`` are judgments the ceiling rejects (V1)."""
        values = {s.value for s in MotionState}
        assert values == {"stationary", "moving", "erratic", "unknown"}
        for judgment in ("loitering", "queueing", "waiting", "abandoned", "suspicious"):
            assert judgment not in values

    def test_unknown_is_distinct_from_stationary(self) -> None:
        assert MotionState.UNKNOWN is not MotionState.STATIONARY


class TestTrackEvidence:
    def test_margin_is_the_ambiguity_measure(self) -> None:
        evidence = TrackEvidence(
            association_method=AssociationMethod.IOU,
            association_cost=0.2,
            runner_up_cost=0.5,
        )
        assert evidence.margin == pytest.approx(0.3)

    def test_margin_is_none_when_there_was_no_contest(self) -> None:
        evidence = TrackEvidence(association_method=AssociationMethod.IOU, association_cost=0.2)
        assert evidence.margin is None

    def test_a_narrow_margin_is_visible(self) -> None:
        """The ID-switch risk M6 requires the tracker not hide."""
        evidence = TrackEvidence(
            association_method=AssociationMethod.IOU,
            association_cost=0.40,
            runner_up_cost=0.41,
        )
        assert evidence.margin == pytest.approx(0.01, abs=1e-9)


class TestAssociation:
    def test_requires_association_semantics(self) -> None:
        with pytest.raises(ValueError, match="ASSOCIATION"):
            Association(
                track_id=make_track_id(),
                detection_index=0,
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.DETECTION_PRESENCE
                ),
                method=AssociationMethod.IOU,
            )

    def test_unassociated_tracks_use_index_minus_one(self) -> None:
        association = Association(
            track_id=make_track_id(),
            detection_index=-1,
            confidence=Confidence.uncalibrated(0.0, ConfidenceSemantics.ASSOCIATION),
            method=AssociationMethod.REINITIALIZED,
        )
        assert association.detection_index == -1


class TestTrackUpdate:
    def _update(self, **overrides) -> TrackUpdate:
        defaults = dict(
            camera_id=CAMERA,
            frame_ref=FrameRef(CAMERA, StreamEpoch(1), FrameSeq(3)),
            tracker_epoch=0,
        )
        defaults.update(overrides)
        return TrackUpdate(**defaults)

    def test_empty_update_is_valid(self) -> None:
        update = self._update()
        assert update.active_count == 0
        assert not update.failed

    def test_confirmed_filters_by_state(self) -> None:
        update = self._update(
            active=(
                make_track(state=TrackState.CONFIRMED),
                make_track(track_id=make_track_id(1), state=TrackState.TENTATIVE),
            )
        )
        assert len(update.confirmed) == 1

    def test_measured_excludes_predicted_tracks(self) -> None:
        update = self._update(
            active=(
                make_track(),
                make_track(
                    track_id=make_track_id(1),
                    state=TrackState.COASTING,
                    coast_frames=2,
                    measurement_basis=MeasurementBasis.PREDICTED,
                ),
            )
        )
        assert len(update.measured) == 1
        assert len(update.active) == 2

    def test_terminated_pairs_carry_a_reason(self) -> None:
        update = self._update(terminated=((make_track_id(), BreakReason.EXIT),))
        assert update.terminated[0][1] is BreakReason.EXIT

    def test_failed_is_distinct_from_empty(self) -> None:
        """Invariant V8: 'could not track' is not 'nothing is here'."""
        empty = self._update()
        broken = self._update(failed=True, reason="adapter blew up")
        assert empty.active_count == broken.active_count == 0
        assert not empty.failed
        assert broken.failed and broken.reason


class TestBreakReason:
    def test_none_is_distinct_from_a_real_reason(self) -> None:
        assert BreakReason.NONE.value == "none"

    def test_epoch_reset_is_a_reason(self) -> None:
        """A reset is a discontinuity consumers must see, not a tracking failure."""
        assert BreakReason.EPOCH_RESET in set(BreakReason)

    def test_exit_is_the_healthy_ending(self) -> None:
        assert BreakReason.EXIT in set(BreakReason)
