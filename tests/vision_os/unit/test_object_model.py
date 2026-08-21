"""Vision Object Model — identity, time, space, and the closed vocabulary.

These are the contracts the next decade integrates against (02_VOM). The tests
that matter most here defend the properties whose violation is silent:
``FrameRef`` uniqueness across reconnects, mandatory uncertainty on time and
ground projection, and the refusal to construct an unmasked frame.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.camera import (
    NativeProfile,
    PipelineProfile,
    SourceSemantics,
    SourceSpec,
)
from vision_os.core.model.frame import (
    Frame,
    FrameDimensions,
    FrameQuality,
    PrivacyState,
)
from vision_os.core.model.ids import (
    CameraId,
    FrameRef,
    FrameSeq,
    ProfileId,
    StreamEpoch,
    new_ulid,
    ulid_timestamp_ms,
)
from vision_os.core.model.region import Region
from vision_os.core.model.space import (
    Box,
    Calibration,
    FrameOfReference,
    Homography,
    Point,
    Polygon,
    SpatialInfo,
)
from vision_os.core.model.timebase import (
    ClockQuality,
    Duration,
    FrameTime,
    Instant,
)


class _Pixels:
    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def nbytes(self) -> int:
        return len(self._data)

    def readonly_view(self) -> memoryview:
        return memoryview(self._data).toreadonly()


class TestFrameRef:
    def test_epoch_makes_frame_refs_unique_across_reconnects(self) -> None:
        """The bug class the epoch exists to prevent.

        Every naive RTSP implementation restarts frame numbering at zero on
        reconnect, so frame 100 before and after describe different instants
        while comparing equal — producing time travel in state and observations
        that reference the wrong pixels.
        """
        before = FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(100))
        after = FrameRef(CameraId("cam-01"), StreamEpoch(2), FrameSeq(100))
        assert before != after
        assert before < after

    def test_ordering_is_total_within_a_camera(self) -> None:
        refs = [
            FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(2)),
            FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(0)),
            FrameRef(CameraId("cam-01"), StreamEpoch(2), FrameSeq(0)),
            FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(1)),
        ]
        ordered = sorted(refs)
        assert [(r.stream_epoch, r.frame_seq) for r in ordered] == [(1, 0), (1, 1), (1, 2), (2, 0)]

    def test_follows_in_same_epoch_rejects_cross_epoch_comparison(self) -> None:
        first = FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(5))
        later_epoch = FrameRef(CameraId("cam-01"), StreamEpoch(2), FrameSeq(6))
        assert not later_epoch.follows_in_same_epoch(first)

    def test_negative_components_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="frame_seq"):
            FrameRef(CameraId("cam-01"), StreamEpoch(0), FrameSeq(-1))


class TestUlid:
    def test_is_lexicographically_time_sortable(self) -> None:
        early = new_ulid(now_ms=1_000)
        late = new_ulid(now_ms=2_000)
        assert early < late

    def test_round_trips_its_timestamp(self) -> None:
        assert ulid_timestamp_ms(new_ulid(now_ms=1_762_000_000_000)) == 1_762_000_000_000

    def test_is_unique_without_coordination(self) -> None:
        minted = {new_ulid(now_ms=1_000) for _ in range(2_000)}
        assert len(minted) == 2_000

    def test_malformed_ulid_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            ulid_timestamp_ms("too-short")


class TestTimeModel:
    def test_uncertainty_is_mandatory_and_non_negative(self) -> None:
        with pytest.raises(ValueError, match="uncertainty"):
            FrameTime(
                pts=0,
                t_capture=Instant(1_000),
                t_capture_uncertainty=Duration(-1),
                t_ingest=Instant(2_000),
                t_decoded=Instant(3_000),
                clock_quality=ClockQuality.NTP_SYNCED,
            )

    def test_unknown_clock_quality_never_fuses(self) -> None:
        """Admitting ignorance beats a confident wrong ordering (02_VOM §5.2)."""
        assert not ClockQuality.UNKNOWN.fusable
        assert ClockQuality.PTP_LOCKED.fusable

    def test_fusion_refused_when_combined_uncertainty_exceeds_phenomenon(self) -> None:
        def at(uncertainty_ms: float, quality: ClockQuality) -> FrameTime:
            return FrameTime(
                pts=0,
                t_capture=Instant(1_000_000_000),
                t_capture_uncertainty=Duration.from_millis(uncertainty_ms),
                t_ingest=Instant(1_000_000_000),
                t_decoded=Instant(1_000_000_000),
                clock_quality=quality,
            )

        precise = at(20, ClockQuality.NTP_SYNCED)
        sloppy = at(800, ClockQuality.ESTIMATED)
        phenomenon = Duration.from_millis(100)
        assert precise.may_fuse_with(at(20, ClockQuality.NTP_SYNCED), phenomenon)
        assert not precise.may_fuse_with(sloppy, phenomenon)

    def test_nanosecond_precision_is_preserved(self) -> None:
        """Float seconds silently destroy sub-millisecond timing at epoch values."""
        instant = Instant(1_762_000_000_123_456_789)
        assert instant.ns == 1_762_000_000_123_456_789


class TestSpaceModel:
    def test_degenerate_box_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            Box(0.5, 0.5, 0.5, 0.9)

    def test_normalized_coordinates_need_no_calibration(self) -> None:
        info = SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=Box(0.1, 0.1, 0.2, 0.2)
        )
        assert info.calibration_id is None

    def test_metric_coordinates_require_a_calibration_id(self) -> None:
        """A metre measurement without its calibration is unverifiable."""
        with pytest.raises(ValueError, match="calibration_id"):
            SpatialInfo(frame_of_reference=FrameOfReference.CAMERA_GROUND)

    def test_ground_point_requires_uncertainty(self) -> None:
        """Projection error grows sharply with distance (02_VOM §6.2)."""
        with pytest.raises(ValueError, match="ground_uncertainty"):
            SpatialInfo(
                frame_of_reference=FrameOfReference.NORMALIZED,
                ground_point=Point(1.0, 2.0),
            )

    def test_polygon_containment_is_pure_geometry(self) -> None:
        square = Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)))
        assert square.contains(Point(0.5, 0.5))
        assert not square.contains(Point(1.5, 0.5))

    def test_polygon_requires_three_vertices(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            Polygon((Point(0, 0), Point(1, 1)))

    def test_degenerate_homography_is_detected(self) -> None:
        singular = Homography(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 0.0, 1.0)))
        assert singular.is_degenerate()

    def test_projection_returns_uncertainty_that_grows_with_distance(self) -> None:
        calibration = Calibration(
            calibration_id="cal-1",
            homography=Homography(((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0))),
            ground_uncertainty_at_unit_distance=0.05,
        )
        near, near_error = calibration.project_to_ground(Point(0.1, 0.1))
        far, far_error = calibration.project_to_ground(Point(0.9, 0.9))
        assert far.x > near.x
        assert far_error.semi_major > near_error.semi_major

    def test_suspect_calibration_inflates_uncertainty_rather_than_failing(self) -> None:
        """A false drift positive must not blind a working site (03_MODULES M1)."""
        homography = Homography(((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0)))
        trusted = Calibration("cal-1", homography, 0.05, suspect=False)
        suspect = Calibration("cal-1", homography, 0.05, suspect=True)
        _, trusted_error = trusted.project_to_ground(Point(0.5, 0.5))
        _, suspect_error = suspect.project_to_ground(Point(0.5, 0.5))
        assert suspect_error.semi_major > trusted_error.semi_major


class TestRegionNeutrality:
    def test_label_is_opaque_and_carries_no_platform_meaning(self) -> None:
        """Invariant V2: the platform never interprets a region label.

        ``Z3`` is "the pass" in a kitchen and "bed 12 bay" in a ward. The
        platform holds the geometry; the meaning lives with the consumer.
        """
        region = Region(
            region_id="Z3",
            geometry=Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))),
            frame_of_reference=FrameOfReference.NORMALIZED,
            label="Z3",
        )
        assert region.contains(Point(0.5, 0.5))
        assert not hasattr(region, "zone_type")
        assert not hasattr(region, "purpose")


class TestFrame:
    def _time(self) -> FrameTime:
        return FrameTime(
            pts=0,
            t_capture=Instant(1_000),
            t_capture_uncertainty=Duration.from_millis(10),
            t_ingest=Instant(2_000),
            t_decoded=Instant(3_000),
            clock_quality=ClockQuality.NTP_SYNCED,
        )

    def test_refuses_to_construct_with_failed_masking(self) -> None:
        """The fail-closed invariant, enforced at the type level (12_SECURITY)."""
        with pytest.raises(ValueError, match="masking failures must drop the frame"):
            Frame(
                frame_ref=FrameRef(CameraId("cam-01"), StreamEpoch(0), FrameSeq(0)),
                time=self._time(),
                dimensions=FrameDimensions(2, 2),
                pixels=_Pixels(b"\x00" * 12),
                privacy_state=PrivacyState.MASK_FAILED,
            )

    def test_masked_and_unmasked_permitted_are_both_emittable(self) -> None:
        for state in (PrivacyState.MASKED, PrivacyState.UNMASKED_PERMITTED):
            frame = Frame(
                frame_ref=FrameRef(CameraId("cam-01"), StreamEpoch(0), FrameSeq(0)),
                time=self._time(),
                dimensions=FrameDimensions(2, 2),
                pixels=_Pixels(b"\x00" * 12),
                privacy_state=state,
            )
            assert frame.privacy_state.emittable

    def test_pixels_are_read_only_to_consumers(self) -> None:
        """Published frames are immutable, which is why readers need no lock."""
        frame = Frame(
            frame_ref=FrameRef(CameraId("cam-01"), StreamEpoch(0), FrameSeq(0)),
            time=self._time(),
            dimensions=FrameDimensions(2, 2),
            pixels=_Pixels(b"\x00" * 12),
            privacy_state=PrivacyState.MASKED,
            quality=FrameQuality(),
        )
        view = frame.pixels.readonly_view()
        assert view.readonly
        with pytest.raises(TypeError):
            view[0] = 1


class TestSourceSpecSecrecy:
    def test_inline_credentials_in_uri_are_rejected(self) -> None:
        """Camera records reach logs, diagnostics and support bundles."""
        with pytest.raises(ValueError, match="credential_ref"):
            SourceSpec(uri="rtsp://admin:p4ssw0rd@10.0.0.5/stream", transport="rtsp")

    def test_credential_reference_is_permitted(self) -> None:
        spec = SourceSpec(
            uri="rtsp://10.0.0.5/stream", transport="rtsp", credential_ref="cam-01-creds"
        )
        assert spec.credential_ref == "cam-01-creds"
        assert "p4ssw0rd" not in spec.uri


class TestSourceSemantics:
    def test_only_realtime_may_drop_frames(self) -> None:
        assert SourceSemantics.REALTIME.may_drop_frames
        assert not SourceSemantics.ARCHIVAL.may_drop_frames
        assert not SourceSemantics.DISCRETE.may_drop_frames

    def test_non_realtime_semantics_are_reproducible(self) -> None:
        assert SourceSemantics.ARCHIVAL.is_deterministic
        assert not SourceSemantics.REALTIME.is_deterministic


class TestPipelineProfile:
    def test_rejects_non_positive_cadence(self) -> None:
        with pytest.raises(ValueError, match="target_fps"):
            PipelineProfile(profile_id=ProfileId("bad"), target_fps=0.0)

    def test_priority_class_is_an_opaque_label(self) -> None:
        """Invariant V1: the platform orders by it, never interprets it."""
        profile = PipelineProfile(
            profile_id=ProfileId("p"), target_fps=5.0, priority_class="A"
        )
        assert profile.priority_class == "A"


def test_native_profile_is_data_only() -> None:
    profile = NativeProfile(width=1920, height=1080, fps=25.0, codec="h264")
    assert profile.colour_space == "bgr24"
