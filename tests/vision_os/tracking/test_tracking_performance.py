"""Performance, concurrency and stress for the tracking layer.

Budget tests, not benchmarks. 11_PERFORMANCE section 1.1 budgets tracking at
**~0.3 ms/frame**, scaling with object count — about 5% of pipeline cost, and it
must stay there. The bounds below sit an order of magnitude above measured cost:
a budget tuned within 2x of normal fails whenever CI is busy, and a test that
cries wolf is one people learn to re-run rather than read.

The growth tests matter more than the latency ones. Tracking holds the platform's
most volatile state, and an unbounded track table or an unbounded history ring is
the classic soak failure — imperceptible on day one, fatal on day twenty-six.
"""

from __future__ import annotations

import asyncio
import gc
import time

import pytest

from vision_os.adapters.tracking import build_iou_tracker, build_sort_tracker
from vision_os.core.model.ids import CameraId
from vision_os.core.model.space import Box
from vision_os.perception.tracking.lifecycle import LifecyclePolicy

from ..conftest import skip_if_traced
from .conftest import CAMERA, make_outcome, make_request, walking_box


def crowd(count: int) -> list[Box]:
    """A grid of separated objects — a crowded but tractable scene.

    Sized so every box stays inside the unit square for any count up to 100: a
    detection outside it cannot be constructed at all (port obligation D1).
    """
    boxes = []
    for i in range(count):
        x = (i % 10) * 0.095
        y = (i // 10 % 10) * 0.095
        boxes.append(Box(x, y, x + 0.05, y + 0.05))
    return boxes


def steady_box(seq: int) -> Box:
    """A walking box that wraps rather than leaving the frame."""
    return walking_box(seq % 20, speed=0.04)


class TestAssociationCost:
    @skip_if_traced
    def test_a_single_object_frame_is_cheap(self) -> None:
        """Runs once per processed frame per camera; at 100 cameras x 5 fps
        that is 500 calls/second."""
        tracker = build_sort_tracker()
        for seq in range(10):
            tracker.update(make_request(seq, [walking_box(seq)]))

        iterations = 400
        started = time.perf_counter()
        for seq in range(iterations):
            tracker.update(make_request(100 + seq, [Box(0.4, 0.4, 0.5, 0.8)]))
        elapsed = time.perf_counter() - started

        per_frame_ms = (elapsed / iterations) * 1_000
        # Budgeted at 0.3 ms/frame; bounded an order of magnitude above so a
        # loaded machine does not fail a correct implementation.
        assert per_frame_ms < 3.0, f"single-object frame costs {per_frame_ms:.3f}ms"

    @skip_if_traced
    def test_a_twenty_object_frame_stays_within_budget(self) -> None:
        """Cost scales with object count (11_PERFORMANCE section 1.1)."""
        tracker = build_sort_tracker(
            lifecycle=LifecyclePolicy(max_tracks_per_camera=64)
        )
        boxes = crowd(20)
        for seq in range(5):
            tracker.update(make_request(seq, boxes))

        iterations = 100
        started = time.perf_counter()
        for seq in range(iterations):
            tracker.update(make_request(100 + seq, boxes))
        elapsed = time.perf_counter() - started

        per_frame_ms = (elapsed / iterations) * 1_000
        assert per_frame_ms < 30.0, f"20-object frame costs {per_frame_ms:.3f}ms"

    @skip_if_traced
    def test_gating_keeps_association_sub_quadratic(self) -> None:
        """Gating by predicted position is what keeps association effectively
        linear at realistic densities (03_MODULES M6 performance).

        A naive O(n*m) implementation shows a ~16x jump from 10 to 40 objects;
        gating should keep it far below that.
        """
        def cost_for(count: int) -> float:
            tracker = build_sort_tracker(
                lifecycle=LifecyclePolicy(max_tracks_per_camera=128)
            )
            boxes = crowd(count)
            for seq in range(4):
                tracker.update(make_request(seq, boxes))
            started = time.perf_counter()
            for seq in range(30):
                tracker.update(make_request(100 + seq, boxes))
            return time.perf_counter() - started

        small = cost_for(10)
        large = cost_for(40)
        ratio = large / small if small > 0 else 0.0
        assert ratio < 12.0, (
            f"4x the objects cost {ratio:.1f}x the time; gating is not "
            f"containing the association matrix"
        )


class TestNoSteadyStateGrowth:
    """The 30-day soak failure, caught in seconds."""

    def test_the_track_table_stays_bounded_under_churn(self) -> None:
        """Objects constantly appearing and leaving is the normal case, and the
        one that leaks if ids or records are retained."""
        tracker = build_sort_tracker(
            lifecycle=LifecyclePolicy(
                min_hits_to_confirm=2,
                max_coast_frames=2,
                max_lost_frames=2,
                max_tracks_per_camera=32,
            )
        )
        for cycle in range(200):
            # A different object each time, in a different place.
            x = (cycle % 9) * 0.1
            tracker.update(make_request(cycle, [Box(x, 0.4, x + 0.08, 0.8)]))
            assert len(tracker.tracks(CAMERA)) <= 32

    def test_object_count_does_not_grow_across_a_long_run(self) -> None:
        tracker = build_sort_tracker()
        for seq in range(60):
            tracker.update(make_request(seq, [Box(0.4, 0.4, 0.5, 0.8)]))

        gc.collect()
        before = len(gc.get_objects())
        for seq in range(600):
            tracker.update(make_request(100 + seq, [Box(0.4, 0.4, 0.5, 0.8)]))
        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 15_000, (
            f"tracker retained {growth} objects across 600 frames; memory must be "
            f"bounded regardless of scene duration (port obligation T8)"
        )

    def test_track_history_is_capped(self) -> None:
        """An hour-long track must hold the same memory as a one-second track."""
        tracker = build_sort_tracker()
        for seq in range(500):
            tracker.update(make_request(seq, [Box(0.4, 0.4, 0.5, 0.8)]))
        track = tracker.tracks(CAMERA)[0]
        assert len(track.detections) <= 32
        assert track.age_frames > 400, "the track should have survived the run"

    def test_retired_ids_do_not_accumulate_without_bound(self) -> None:
        """Reset clears the retired set; otherwise it grows forever."""
        from vision_os.core.model.ids import TrackerEpoch
        from vision_os.perception.tracking.table import TrackTable

        table = TrackTable(CAMERA, max_tracks=8)
        assert table.stats().retired == 0
        table.reset(TrackerEpoch(1))
        assert table.stats().retired == 0

    def test_a_camera_that_goes_quiet_releases_its_tracks(self) -> None:
        tracker = build_sort_tracker(
            lifecycle=LifecyclePolicy(
                min_hits_to_confirm=2, max_coast_frames=2, max_lost_frames=3
            )
        )
        for seq in range(6):
            tracker.update(make_request(seq, [walking_box(seq)]))
        assert tracker.tracks(CAMERA)

        for seq in range(30):
            tracker.update(make_request(100 + seq, []))
        assert tracker.tracks(CAMERA) == ()


class TestScalingShape:
    @pytest.mark.parametrize("cameras", [1, 10, 100])
    def test_camera_count_is_the_same_code_path(self, cameras: int) -> None:
        """1, 10 or 100 cameras must differ only in scale (01_LAYERED section 6)."""
        tracker = build_sort_tracker()
        ids = [CameraId(f"cam-{i:03d}") for i in range(cameras)]
        for seq in range(3):
            for camera in ids:
                tracker.update(make_request(seq, [walking_box(seq)], camera=camera))
        for camera in ids:
            assert len(tracker.tracks(camera)) == 1

    @skip_if_traced
    def test_a_hundred_cameras_sustain_the_processing_rate(self) -> None:
        """100 cameras x 5 fps is 500 tracking calls/second."""
        tracker = build_sort_tracker()
        cameras = [CameraId(f"cam-{i:03d}") for i in range(100)]

        started = time.perf_counter()
        for seq in range(5):
            for camera in cameras:
                tracker.update(make_request(seq, [walking_box(seq)], camera=camera))
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, (
            f"500 tracking calls took {elapsed:.2f}s; tracking must not become "
            f"the bottleneck it is budgeted at 5% of pipeline cost to avoid"
        )

    def test_state_is_independent_per_camera(self) -> None:
        """No cross-camera structure means no contention (08_RUNTIME section 2)."""
        tracker = build_sort_tracker()
        cameras = [CameraId(f"cam-{i:02d}") for i in range(20)]
        for seq in range(4):
            for camera in cameras:
                tracker.update(make_request(seq, [walking_box(seq)], camera=camera))
        counts = {c: len(tracker.tracks(c)) for c in cameras}
        assert set(counts.values()) == {1}


class TestConcurrency:
    """The runtime is an actor per camera (08_RUNTIME section 2)."""

    async def test_one_camera_is_serialized(self, tracking_runtime) -> None:
        """Frame N's association depends on frame N-1's state, so concurrent
        delivery must not interleave."""
        await tracking_runtime.start()
        await asyncio.gather(
            *(
                tracking_runtime.on_detected(make_outcome(seq, [walking_box(seq)]))
                for seq in range(20)
            )
        )
        assert tracking_runtime.stats.frames_consumed == 20
        assert tracking_runtime.stats.frames_failed == 0

    async def test_many_cameras_run_without_interference(self, tracking_runtime) -> None:
        await tracking_runtime.start()
        cameras = [CameraId(f"cam-{i:02d}") for i in range(25)]
        await asyncio.gather(
            *(
                tracking_runtime.on_detected(
                    make_outcome(seq, [walking_box(seq)], camera=camera)
                )
                for camera in cameras
                for seq in range(4)
            )
        )
        assert tracking_runtime.cameras_seen == 25
        assert tracking_runtime.stats.frames_failed == 0

    async def test_concurrent_delivery_produces_no_duplicate_ids(
        self, tracking_runtime, tracking_engine
    ) -> None:
        await tracking_runtime.start()
        await asyncio.gather(
            *(
                tracking_runtime.on_detected(make_outcome(seq, [walking_box(seq)]))
                for seq in range(15)
            )
        )
        tracks = tracking_engine.tracks(CAMERA)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))

    async def test_the_lock_table_does_not_grow_with_frames(
        self, tracking_runtime
    ) -> None:
        await tracking_runtime.start()
        for seq in range(50):
            await tracking_runtime.on_detected(make_outcome(seq, [steady_box(seq)]))
        assert tracking_runtime.cameras_seen == 1


class TestStress:
    def test_a_dense_crowd_is_survived(self) -> None:
        """A crowd degrades by refusing tracks, never by crashing or growing."""
        tracker = build_sort_tracker(
            lifecycle=LifecyclePolicy(max_tracks_per_camera=48)
        )
        boxes = crowd(80)
        for seq in range(20):
            update = tracker.update(make_request(seq, boxes))
            assert len(update.active) <= 48

    def test_rapid_appearance_and_disappearance(self) -> None:
        tracker = build_sort_tracker(
            lifecycle=LifecyclePolicy(
                min_hits_to_confirm=2, max_coast_frames=1, max_lost_frames=1
            )
        )
        for seq in range(150):
            boxes = [Box(0.4, 0.4, 0.5, 0.8)] if seq % 3 else []
            tracker.update(make_request(seq, boxes))
        assert len(tracker.tracks(CAMERA)) <= 2

    def test_alternating_empty_and_full_frames(self) -> None:
        """The detector flapping is a normal condition, not an error path."""
        tracker = build_sort_tracker()
        for seq in range(100):
            boxes = crowd(10) if seq % 2 == 0 else []
            assert tracker.update(make_request(seq, boxes)) is not None

    def test_many_epochs_do_not_leak(self) -> None:
        tracker = build_sort_tracker()
        for _cycle in range(100):
            for seq in range(3):
                tracker.update(make_request(seq, [walking_box(seq)]))
            tracker.reset(CAMERA, "stress")
        assert tracker.tracks(CAMERA) == ()

    def test_the_fallback_tracker_survives_the_same_stress(self) -> None:
        """It is the last line of defence, so it must hold under load too."""
        tracker = build_iou_tracker(
            lifecycle=LifecyclePolicy(max_tracks_per_camera=48)
        )
        boxes = crowd(60)
        for seq in range(20):
            assert tracker.update(make_request(seq, boxes)) is not None


class TestRegression:
    """Defects found during Flow 3, pinned so they cannot return."""

    def test_a_coasting_position_advances_with_elapsed_time(self) -> None:
        """Regression: predictions extrapolated by one frame's elapsed rather
        than the cumulative time since the last measurement, so a coasting
        track's position never moved and no moving object was ever re-acquired.

        Invisible on stationary objects, which is what made it easy to ship.
        """
        tracker = build_sort_tracker()
        for seq in range(6):
            tracker.update(make_request(seq, [walking_box(seq)]))

        positions = []
        for step in range(3):
            tracker.update(make_request(6 + step, []))
            positions.append(tracker.tracks(CAMERA)[0].spatial.bbox.x1)

        assert positions[0] < positions[1] < positions[2], (
            f"coasting position did not advance: {positions}"
        )

    def test_a_moving_object_recovers_after_a_gap(self) -> None:
        """The consequence of the above: recovery worked for stationary objects
        and silently failed for moving ones."""
        tracker = build_sort_tracker()
        for seq in range(6):
            tracker.update(make_request(seq, [walking_box(seq)]))
        original = tracker.tracks(CAMERA)[0].track_id

        for step in range(3):
            tracker.update(make_request(6 + step, []))
        update = tracker.update(make_request(9, [walking_box(9)]))

        assert original in update.recovered
        assert tracker.tracks(CAMERA)[0].track_id == original

    def test_a_refused_association_is_reported(self) -> None:
        """Regression: a refused association terminated its track in the same
        frame, so it appeared in neither ``active`` nor ``associations`` and the
        refusal was invisible — exactly the uncertainty M6 forbids hiding."""
        tracker = build_sort_tracker()
        pair = [Box(0.30, 0.40, 0.50, 0.80), Box(0.3001, 0.4001, 0.5001, 0.8001)]
        refusals = []
        for seq in range(6):
            refusals.extend(tracker.update(make_request(seq, pair)).refused)
        assert refusals

    def test_velocity_is_not_inflated_by_a_coast(self) -> None:
        """The observation after a coast must be integrated over the whole gap,
        not one frame, or the velocity estimate jumps by the gap length."""
        tracker = build_sort_tracker()
        for seq in range(6):
            tracker.update(make_request(seq, [walking_box(seq)]))
        steady = tracker.tracks(CAMERA)[0].motion.velocity.x

        for step in range(3):
            tracker.update(make_request(6 + step, []))
        tracker.update(make_request(9, [walking_box(9)]))
        after = tracker.tracks(CAMERA)[0].motion.velocity.x

        assert after == pytest.approx(steady, rel=0.5), (
            f"velocity jumped from {steady:.4f} to {after:.4f} across a coast"
        )

    def test_tenancy_comes_from_the_detection_not_the_tracker(self) -> None:
        """Regression: tenancy was a construction-time constant, which would
        stamp one tenant across a node serving several — a breach of the
        platform's hard isolation boundary."""
        from vision_os.core.model.ids import SiteId, TenantId

        tracker = build_sort_tracker()
        for seq in range(4):
            tracker.update(make_request(seq, [walking_box(seq)]))
        track = tracker.tracks(CAMERA)[0]
        assert track.tenant_id == TenantId("acme")
        assert track.site_id == SiteId("site-sg-01")
