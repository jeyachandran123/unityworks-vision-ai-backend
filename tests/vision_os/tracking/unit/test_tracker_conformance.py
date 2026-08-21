"""The tracker conformance kit — and proof that it fails when it should.

The second half of this file matters more than the first. A kit that passes
everything it is shown is indistinguishable from no kit, so every obligation is
paired with a tracker that violates *exactly* that obligation and nothing else,
and the kit is required to reject it.

The broken trackers are built by wrapping a real one and corrupting a single
behaviour, so an unrelated failure cannot mask the obligation under test.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.tracking import (
    build_bytetrack_tracker,
    build_iou_tracker,
    build_sort_tracker,
)
from vision_os.conformance import TRACKER_KIT
from vision_os.conformance.kit import KitSection
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import CameraId, TrackerEpoch
from vision_os.core.model.track import BreakReason, MeasurementBasis, TrackUpdate
from vision_os.core.ports.tracking import TrackerCapabilities, TrackerPort


def build(kind: str = "sort"):
    factories = {
        "iou": build_iou_tracker,
        "sort": build_sort_tracker,
        "bytetrack": build_bytetrack_tracker,
    }
    return factories[kind](config_revision="test")


def failed_checks(report) -> set[str]:
    names = set()
    for failure in report.failures:
        head = failure.split(":", 1)[0]
        names.add(head.split("] ")[-1])
    return names


# --- the shipped trackers all conform ---------------------------------------- #


class TestShippedTrackersConform:
    @pytest.mark.parametrize("kind", ["iou", "sort", "bytetrack"])
    def test_tracker_passes_its_kit(self, kind: str) -> None:
        report = TRACKER_KIT.run(build(kind))
        assert report.passed, report.failures

    def test_the_kit_runs_every_check(self) -> None:
        report = TRACKER_KIT.run(build())
        assert len(report.executed) == len(TRACKER_KIT.checks)
        assert not report.skipped

    def test_the_kit_covers_every_section_that_matters(self) -> None:
        sections = TRACKER_KIT.sections_covered()
        assert KitSection.SHAPE in sections
        assert KitSection.SEMANTICS in sections
        assert KitSection.FAILURE in sections
        assert KitSection.RESOURCE in sections

    def test_the_kit_covers_every_port_obligation_it_claims(self) -> None:
        """T1-T8 must each be represented by an executable check."""
        obligations = {c.obligation for c in TRACKER_KIT.checks if c.obligation}
        for required in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"):
            assert required in obligations, f"no executable check for {required}"

    def test_the_fast_subset_gates_at_load(self) -> None:
        """The catastrophic classes must be caught before the first real frame."""
        report = TRACKER_KIT.run(build(), fast_only=True)
        assert report.passed, report.failures
        assert report.fast_subset_only
        executed = " ".join(report.executed)
        for critical in (
            "ids_are_unique_within_epoch",
            "out_of_order_is_rejected",
            "coasting_is_marked_predicted",
            "no_fabricated_tracks",
        ):
            assert critical in executed

    def test_a_pass_summary_names_the_port(self) -> None:
        summary = TRACKER_KIT.run(build()).summary()
        assert "PASS" in summary
        assert "P9" in summary


# --- broken trackers, one obligation each ------------------------------------- #


class _Wrapper(TrackerPort):
    """Delegates everything to a real tracker. Subclasses break one thing."""

    def __init__(self, kind: str = "sort") -> None:
        self._inner = build(kind)

    def update(self, request) -> TrackUpdate:
        return self._inner.update(request)

    def tracks(self, camera_id):
        return self._inner.tracks(camera_id)

    def reset(self, camera_id, reason):
        return self._inner.reset(camera_id, reason)

    def capabilities(self) -> TrackerCapabilities:
        return self._inner.capabilities()


class _DriftingCapabilities(_Wrapper):
    """Capabilities change after processing — callers cache them."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def capabilities(self) -> TrackerCapabilities:
        self._calls += 1
        inner = self._inner.capabilities()
        return TrackerCapabilities(
            tracker_id=inner.tracker_id,
            version=inner.version,
            max_objects=inner.max_objects + self._calls,
        )


class _AcceptsOutOfOrder(_Wrapper):
    """Absorbs an out-of-order frame instead of rejecting it (breaks T1)."""

    def update(self, request) -> TrackUpdate:
        try:
            return self._inner.update(request)
        except Exception:  # noqa: BLE001 - the defect under test
            return TrackUpdate(
                camera_id=request.camera_id,
                frame_ref=request.frame_ref,
                tracker_epoch=0,
            )


class _ReusesIds(_Wrapper):
    """Mints a fresh tracker per frame, so local ids restart at 0 (breaks T3)."""

    def update(self, request) -> TrackUpdate:
        self._inner = build()
        return self._inner.update(request)


class _MislabelsPredictions(_Wrapper):
    """Presents a coasted position as measured (breaks T5).

    The corruption V8 exists to prevent, and one no consumer can detect alone.
    """

    def update(self, request) -> TrackUpdate:
        update = self._inner.update(request)
        active = tuple(
            _replace(track, measurement_basis=MeasurementBasis.MEASURED)
            if track.state.is_predicted
            else track
            for track in update.active
        )
        return _replace(update, active=active)


class _TerminatesWithoutReason(_Wrapper):
    """Drops the break_reason (breaks T6)."""

    def update(self, request) -> TrackUpdate:
        update = self._inner.update(request)
        return _replace(
            update,
            terminated=tuple((tid, BreakReason.NONE) for tid, _ in update.terminated),
        )


class _WrongConfidenceSemantics(_Wrapper):
    """Reports the detector's presence score as association confidence (T4)."""

    def update(self, request) -> TrackUpdate:
        update = self._inner.update(request)
        active = tuple(
            _replace(
                track,
                confidence=Confidence(
                    value=track.confidence.value,
                    semantics=ConfidenceSemantics.DETECTION_PRESENCE,
                ),
            )
            for track in update.active
        )
        return _replace(update, active=active)


class _ResetDoesNotAdvanceEpoch(_Wrapper):
    """Clears state but keeps the epoch (breaks T7).

    A recycled local id then lets a consumer infer that an object teleported.
    """

    def reset(self, camera_id, reason) -> TrackerEpoch:
        self._inner.reset(camera_id, reason)
        return TrackerEpoch(0)


class _LeaksAcrossCameras(_Wrapper):
    """Resetting one camera wipes them all (breaks T7)."""

    def reset(self, camera_id, reason) -> TrackerEpoch:
        epoch = self._inner.reset(camera_id, reason)
        for other in (CameraId("kit-cam-01"), CameraId("kit-cam-02"), CameraId("kit-reset")):
            if other != camera_id:
                self._inner.reset(other, "leak")
        return epoch


class _RejectsEmptyFrames(_Wrapper):
    """Treats a frame with no detections as an error.

    Fatal in this platform: an empty frame is exactly when tracks coast and
    terminate, so such a tracker freezes every track during a detector outage.
    """

    def update(self, request) -> TrackUpdate:
        if not request.detections:
            raise ValueError("no detections")
        return self._inner.update(request)


class _FabricatesTracks(_Wrapper):
    """Invents a track from an empty frame."""

    def update(self, request) -> TrackUpdate:
        update = self._inner.update(request)
        if not request.detections and not update.active:
            from vision_os.core.model.ids import LocalTrackId, TrackId

            from .test_track_model import make_track

            phantom = make_track(
                track_id=TrackId(request.camera_id, TrackerEpoch(0), LocalTrackId(999))
            )
            return _replace(update, active=(phantom,))
        return update


class _OverstatesCapacity(_Wrapper):
    """Declares room for two objects but holds as many as it likes (breaks T8)."""

    def capabilities(self) -> TrackerCapabilities:
        inner = self._inner.capabilities()
        return TrackerCapabilities(
            tracker_id=inner.tracker_id, version=inner.version, max_objects=2
        )


class _NeverTerminates(_Wrapper):
    """Retains tracks forever (breaks T8)."""

    def update(self, request) -> TrackUpdate:
        update = self._inner.update(request)
        return _replace(update, terminated=())

    def tracks(self, camera_id):
        return self._inner.tracks(camera_id)


def _replace(instance, **changes):
    """``dataclasses.replace`` for frozen slotted types."""
    import dataclasses

    return dataclasses.replace(instance, **changes)


class TestKitRejectsBrokenTrackers:
    """A kit that passes everything is indistinguishable from no kit."""

    def test_missing_tracker_id_is_caught(self) -> None:
        class _Nameless(_Wrapper):
            def capabilities(self) -> TrackerCapabilities:
                inner = self._inner.capabilities()
                return TrackerCapabilities(
                    tracker_id="", version=inner.version, max_objects=inner.max_objects
                )

        report = TRACKER_KIT.run(_Nameless())
        assert not report.passed
        assert "shape/declares_capabilities" in failed_checks(report)

    def test_drifting_capabilities_are_caught(self) -> None:
        report = TRACKER_KIT.run(_DriftingCapabilities())
        assert not report.passed
        assert "shape/capabilities_are_stable" in failed_checks(report)

    def test_accepting_an_out_of_order_frame_is_caught(self) -> None:
        report = TRACKER_KIT.run(_AcceptsOutOfOrder())
        assert not report.passed
        assert "semantics/out_of_order_is_rejected" in failed_checks(report)

    def test_reused_ids_are_caught(self) -> None:
        report = TRACKER_KIT.run(_ReusesIds())
        assert not report.passed
        assert "semantics/ids_are_unique_within_epoch" in failed_checks(report)

    def test_a_prediction_sold_as_a_measurement_is_caught(self) -> None:
        report = TRACKER_KIT.run(_MislabelsPredictions())
        assert not report.passed
        assert "semantics/coasting_is_marked_predicted" in failed_checks(report)

    def test_termination_without_a_reason_is_caught(self) -> None:
        report = TRACKER_KIT.run(_TerminatesWithoutReason())
        assert not report.passed
        assert "semantics/termination_carries_a_reason" in failed_checks(report)

    def test_wrong_confidence_semantics_are_caught(self) -> None:
        report = TRACKER_KIT.run(_WrongConfidenceSemantics())
        assert not report.passed
        assert "semantics/association_confidence_semantics" in failed_checks(report)

    def test_a_reset_that_does_not_advance_the_epoch_is_caught(self) -> None:
        report = TRACKER_KIT.run(_ResetDoesNotAdvanceEpoch())
        assert not report.passed
        assert "semantics/reset_mints_a_new_epoch" in failed_checks(report)

    def test_cross_camera_leakage_is_caught(self) -> None:
        report = TRACKER_KIT.run(_LeaksAcrossCameras())
        assert not report.passed
        assert "semantics/no_cross_camera_state" in failed_checks(report)

    def test_rejecting_empty_frames_is_caught(self) -> None:
        report = TRACKER_KIT.run(_RejectsEmptyFrames())
        assert not report.passed
        assert "shape/empty_frame_is_handled" in failed_checks(report)

    def test_fabricated_tracks_are_caught(self) -> None:
        report = TRACKER_KIT.run(_FabricatesTracks())
        assert not report.passed
        assert "failure/no_fabricated_tracks" in failed_checks(report)

    def test_an_overstated_capacity_is_caught(self) -> None:
        report = TRACKER_KIT.run(_OverstatesCapacity())
        assert not report.passed
        assert (
            "resource/track_count_stays_within_declared_maximum" in failed_checks(report)
        )

    def test_a_tracker_that_never_terminates_is_caught(self) -> None:
        report = TRACKER_KIT.run(_NeverTerminates())
        assert not report.passed


class TestFailureReporting:
    def test_one_failure_does_not_abort_the_kit(self) -> None:
        """The full picture is the point; stopping at the first failure hides it."""
        report = TRACKER_KIT.run(_ReusesIds())
        assert len(report.executed) == len(TRACKER_KIT.checks)

    def test_a_failure_names_the_obligation_it_breaks(self) -> None:
        report = TRACKER_KIT.run(_AcceptsOutOfOrder())
        assert any("T1" in failure for failure in report.failures)

    def test_a_failure_summary_says_fail(self) -> None:
        assert "FAIL" in TRACKER_KIT.run(_ReusesIds()).summary()


class TestDeterminismCheck:
    def test_two_identical_runs_produce_identical_tracks(self) -> None:
        """Invariant V13. Needs a factory, not an instance — it compares two
        independent runs from a clean state."""
        from vision_os.conformance import DETERMINISM_CHECK

        assert DETERMINISM_CHECK.execute(lambda: build("sort")) is None

    @pytest.mark.parametrize("kind", ["iou", "sort", "bytetrack"])
    def test_every_shipped_tracker_is_deterministic(self, kind: str) -> None:
        from vision_os.conformance import DETERMINISM_CHECK

        assert DETERMINISM_CHECK.execute(lambda: build(kind)) is None

    def test_every_shipped_tracker_declares_determinism_honestly(self) -> None:
        for kind in ("iou", "sort", "bytetrack"):
            assert build(kind).capabilities().deterministic
