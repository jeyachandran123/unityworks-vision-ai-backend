"""Performance, concurrency, stress and regression for the Object Registry.

Budget tests, not benchmarks. ``11_PERFORMANCE`` section 1.1 budgets the registry
at **~0.1 ms/frame**, scaling with object count — relative cost 0.5 against
detection's 15. Bounds below sit an order of magnitude above measured cost: a
budget tuned within 2x of normal fails whenever CI is busy, and a test that cries
wolf is one people learn to re-run rather than read.

The growth tests matter more than the latency ones. Section M7 says it plainly:
*"Unbounded history here is the most likely long-run memory leak in the entire
platform"* and *"a runaway registry is a memory leak with a face"*.
"""

from __future__ import annotations

import asyncio
import gc
import time

import pytest

from vision_os.core.model.ids import CameraId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.model.visual_object import LifecycleState
from vision_os.perception.registry import LifecyclePolicy, ObjectRegistry

from ..conftest import skip_if_traced
from .conftest import (
    CAMERA,
    SITE,
    TENANT,
    age,
    at,
    coast,
    drive,
    make_track,
    make_update,
    walking,
)


def crowd(count: int, seq: int = 0):
    """A grid of separated tracks — crowded but tractable, all in-frame."""
    tracks = []
    for i in range(count):
        x = (i % 10) * 0.095
        y = (i // 10 % 10) * 0.095
        tracks.append(
            make_track(local=i, box=Box(x, y, x + 0.05, y + 0.05), seq=seq)
        )
    return tracks


def build(clock, bus, metrics, config, provenance, **policy):
    return ObjectRegistry(
        clock=clock,
        bus=bus,
        metrics=metrics,
        config=config,
        tenant_id=TENANT,
        site_id=SITE,
        provenance=provenance,
        lifecycle=LifecyclePolicy(**policy) if policy else None,
    )


class TestIngestCost:
    @skip_if_traced
    def test_a_single_object_frame_is_cheap(self, registry) -> None:
        """Runs once per processed frame per camera."""
        drive(registry, 10)

        iterations = 300
        started = time.perf_counter()
        for step in range(iterations):
            registry.ingest(
                CAMERA,
                make_update([make_track(seq=100 + step)], seq=100 + step),
            )
        elapsed = time.perf_counter() - started

        per_frame_ms = (elapsed / iterations) * 1_000
        # Budgeted at 0.1 ms/frame; bounded an order of magnitude above so a
        # loaded machine does not fail a correct implementation.
        assert per_frame_ms < 2.0, f"single-object frame costs {per_frame_ms:.3f}ms"

    @skip_if_traced
    def test_a_twenty_object_frame_stays_within_budget(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Cost scales with object count (11_PERFORMANCE section 1.1)."""
        registry = build(
            clock, bus, metrics, registry_config, registry_provenance,
            max_objects_per_camera=64,
        )
        for seq in range(4):
            registry.ingest(CAMERA, make_update(crowd(20, seq), seq=seq))

        iterations = 100
        started = time.perf_counter()
        for step in range(iterations):
            registry.ingest(
                CAMERA, make_update(crowd(20, 100 + step), seq=100 + step)
            )
        elapsed = time.perf_counter() - started

        per_frame_ms = (elapsed / iterations) * 1_000
        assert per_frame_ms < 20.0, f"20-object frame costs {per_frame_ms:.3f}ms"

    @skip_if_traced
    def test_region_membership_is_not_naive(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Section M7: *"must not be naive at 100 objects x 20 regions"*.

        The spatial index rejects most pairs on four float comparisons, so
        twenty regions must cost far less than twenty times one region.
        """
        from .conftest import make_region

        def cost_for(regions: int) -> float:
            registry = build(
                clock, bus, metrics, registry_config, registry_provenance,
                max_objects_per_camera=64,
            )
            registry.set_regions(
                CAMERA,
                tuple(
                    make_region(
                        f"R{i}",
                        box=(
                            (i % 5) * 0.19,
                            (i // 5) * 0.24,
                            (i % 5) * 0.19 + 0.18,
                            (i // 5) * 0.24 + 0.23,
                        ),
                    )
                    for i in range(regions)
                ),
            )
            for seq in range(3):
                registry.ingest(CAMERA, make_update(crowd(20, seq), seq=seq))
            started = time.perf_counter()
            for step in range(30):
                registry.ingest(
                    CAMERA, make_update(crowd(20, 100 + step), seq=100 + step)
                )
            return time.perf_counter() - started

        one = cost_for(1)
        twenty = cost_for(20)
        ratio = twenty / one if one > 0 else 0.0
        assert ratio < 10.0, (
            f"20 regions cost {ratio:.1f}x one region; the spatial index is not "
            f"containing the polygon tests"
        )


class TestBoundedMemory:
    """The 30-day soak failure, caught in seconds."""

    def test_the_population_never_exceeds_the_cap(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = build(
            clock, bus, metrics, registry_config, registry_provenance,
            min_observations_to_confirm=3, max_objects_per_camera=16,
        )
        for seq in range(60):
            registry.ingest(CAMERA, make_update(crowd(30, seq), seq=seq))
            assert len(registry.objects(CAMERA)) <= 16

    def test_spatial_history_stays_bounded(self, registry) -> None:
        """An hour-long object holds the same memory as a one-second object."""
        drive(registry, 400)
        obj = registry.active(CAMERA)[0]
        assert len(obj.spatial_history) <= 64
        assert obj.observation_count > 300, "the object should have survived the run"

    def test_class_history_stays_bounded(self, registry) -> None:
        drive(registry, 400)
        assert len(registry.active(CAMERA)[0].class_history) <= 32

    def test_object_count_does_not_grow_across_a_long_run(self, registry) -> None:
        drive(registry, 50)

        gc.collect()
        before = len(gc.get_objects())
        for step in range(500):
            registry.ingest(
                CAMERA, make_update([make_track(seq=100 + step)], seq=100 + step)
            )
        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 20_000, (
            f"registry retained {growth} objects across 500 frames; memory must be "
            f"bounded regardless of scene duration (section M7)"
        )

    def test_expired_objects_leave_the_population(self, registry) -> None:
        drive(registry, 6)
        coast(registry, 1, start=10)
        age(registry, 500)
        assert registry.objects(CAMERA) == ()

    def test_sustained_churn_does_not_accumulate(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Objects constantly appearing and leaving is the normal case, and the
        one that leaks if records are retained."""
        registry = build(
            clock, bus, metrics, registry_config, registry_provenance,
            min_observations_to_confirm=2,
            provisional_horizon=Duration.from_millis(400),
            occlusion_horizon=Duration.from_millis(400),
            dormant_horizon=Duration.from_millis(800),
            retention_horizon=Duration.from_millis(1_200),
            max_objects_per_camera=64,
        )
        for cycle in range(40):
            seq = cycle * 5
            x = (cycle % 9) * 0.1
            registry.ingest(
                CAMERA,
                make_update(
                    [make_track(local=cycle, box=Box(x, 0.4, x + 0.08, 0.8), seq=seq)],
                    seq=seq,
                ),
            )
            registry.expire_stale(at(seq))
        assert len(registry.objects(CAMERA)) <= 64


class TestScalingShape:
    @pytest.mark.parametrize("cameras", [1, 10, 100])
    def test_camera_count_is_the_same_code_path(
        self, cameras: int, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = build(clock, bus, metrics, registry_config, registry_provenance)
        ids = [CameraId(f"cam-{i:03d}") for i in range(cameras)]
        for seq in range(3):
            for camera in ids:
                registry.ingest(
                    camera,
                    make_update(
                        [make_track(box=walking(seq), seq=seq, camera=camera)],
                        seq=seq,
                        camera=camera,
                    ),
                )
        for camera in ids:
            assert len(registry.active(camera)) == 1

    @skip_if_traced
    def test_a_hundred_cameras_sustain_the_processing_rate(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """100 cameras x 5 fps is 500 registry calls per second."""
        registry = build(clock, bus, metrics, registry_config, registry_provenance)
        cameras = [CameraId(f"cam-{i:03d}") for i in range(100)]

        started = time.perf_counter()
        for seq in range(5):
            for camera in cameras:
                registry.ingest(
                    camera,
                    make_update(
                        [make_track(box=walking(seq), seq=seq, camera=camera)],
                        seq=seq,
                        camera=camera,
                    ),
                )
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, (
            f"500 registry calls took {elapsed:.2f}s; the registry is budgeted at "
            f"0.5 relative cost against detection's 15 and must stay there"
        )

    def test_partitions_are_independent(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = build(clock, bus, metrics, registry_config, registry_provenance)
        cameras = [CameraId(f"cam-{i:02d}") for i in range(20)]
        for seq in range(4):
            for camera in cameras:
                registry.ingest(
                    camera,
                    make_update(
                        [make_track(box=walking(seq), seq=seq, camera=camera)],
                        seq=seq,
                        camera=camera,
                    ),
                )
        counts = {c: len(registry.active(c)) for c in cameras}
        assert set(counts.values()) == {1}


class TestConcurrency:
    """One writer per partition (07_STATE section 4.1, 08_RUNTIME section 2)."""

    async def test_one_camera_is_serialized(self, registry_runtime) -> None:
        await registry_runtime.start()
        await asyncio.gather(
            *(
                registry_runtime.on_tracked(
                    CAMERA, make_update([make_track(seq=seq)], seq=seq)
                )
                for seq in range(30)
            )
        )
        assert registry_runtime.stats.frames_consumed == 30
        assert registry_runtime.stats.frames_failed == 0

    async def test_concurrent_ingestion_produces_no_duplicate_objects(
        self, registry_runtime, registry
    ) -> None:
        await registry_runtime.start()
        await asyncio.gather(
            *(
                registry_runtime.on_tracked(
                    CAMERA, make_update([make_track(seq=seq)], seq=seq)
                )
                for seq in range(20)
            )
        )
        objects = registry.objects(CAMERA)
        ids = [o.object_id for o in objects]
        assert len(ids) == len(set(ids))

    async def test_many_cameras_do_not_contend(self, registry_runtime) -> None:
        await registry_runtime.start()
        cameras = [CameraId(f"cam-{i:02d}") for i in range(25)]
        await asyncio.gather(
            *(
                registry_runtime.on_tracked(
                    camera,
                    make_update(
                        [make_track(box=walking(seq), seq=seq, camera=camera)],
                        seq=seq,
                        camera=camera,
                    ),
                )
                for camera in cameras
                for seq in range(4)
            )
        )
        assert registry_runtime.cameras_seen == 25
        assert registry_runtime.stats.frames_failed == 0

    async def test_the_lock_table_does_not_grow_with_frames(
        self, registry_runtime
    ) -> None:
        await registry_runtime.start()
        for seq in range(50):
            await registry_runtime.on_tracked(
                CAMERA, make_update([make_track(seq=seq)], seq=seq)
            )
        assert registry_runtime.cameras_seen == 1


class TestStress:
    def test_a_dense_crowd_is_survived(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = build(
            clock, bus, metrics, registry_config, registry_provenance,
            max_objects_per_camera=48,
        )
        for seq in range(25):
            result = registry.ingest(CAMERA, make_update(crowd(80, seq), seq=seq))
            assert not result.failed
            assert len(registry.objects(CAMERA)) <= 48

    def test_rapid_appearance_and_disappearance(self, registry) -> None:
        for seq in range(120):
            tracks = [make_track(seq=seq)] if seq % 3 else []
            assert not registry.ingest(CAMERA, make_update(tracks, seq=seq)).failed

    def test_alternating_empty_and_full_frames(self, registry) -> None:
        for seq in range(80):
            tracks = crowd(10, seq) if seq % 2 == 0 else []
            assert registry.ingest(CAMERA, make_update(tracks, seq=seq)) is not None

    def test_repeated_merges_do_not_corrupt_resolution(self, registry) -> None:
        for seq in range(5):
            registry.ingest(CAMERA, make_update(crowd(8, seq), seq=seq))
        objects = list(registry.active(CAMERA))

        survivor = objects[-1].object_id
        for obj in objects[:-1]:
            survivor = registry.merge(obj.object_id, survivor)

        for obj in objects[:-1]:
            assert registry.resolve(obj.object_id).object_id == survivor

    def test_many_expiry_sweeps_are_safe(self, registry) -> None:
        drive(registry, 10)
        for step in range(200):
            registry.expire_stale(at(step))
        assert len(registry.objects(CAMERA)) <= 1


class TestRegression:
    """Defects found during Flow 4, pinned so they cannot return."""

    def test_object_time_never_regresses(self, registry) -> None:
        """Regression: an empty frame fell back to the injected clock, which
        could be *earlier* than ``last_confirmed`` — producing an object whose
        measurement was newer than its most recent update.

        The model refused to be constructed, which is how it was found; the fix
        is that a camera's capture time is monotonic and an empty frame does not
        advance it.
        """
        drive(registry, 6)
        before = registry.objects(CAMERA)[0]
        registry.ingest(CAMERA, make_update([], seq=100))
        after = registry.objects(CAMERA)[0]
        assert after.last_seen.ns >= before.last_seen.ns
        assert after.last_confirmed.ns <= after.last_seen.ns

    def test_a_vanished_track_releases_its_binding(self, registry) -> None:
        """Regression: an object whose track disappeared kept its binding open,
        so the binder — which only considers *unbound* candidates — could never
        re-bind it. The object was permanently unreachable for re-entry.
        """
        drive(registry, 6)
        assert registry.objects(CAMERA)[0].bound_track is not None
        registry.ingest(CAMERA, make_update([], seq=100))
        assert registry.objects(CAMERA)[0].bound_track is None

    def test_a_large_time_jump_reaches_the_right_terminal_state(
        self, registry
    ) -> None:
        """Regression: one sweep advanced one lifecycle edge, so a long gap left
        an object 'occluded' when it should have expired. The result depended on
        how often the sweep ran rather than on elapsed time.
        """
        drive(registry, 6)
        coast(registry, 1, start=10)
        removed = registry.expire_stale(at(1_000))
        assert removed, "a twenty-minute gap must expire the object outright"
        assert registry.objects(CAMERA) == ()

    def test_dwell_is_measured_in_capture_time(self, registry, bus) -> None:
        """Regression: object timestamps came from the registry's own clock, so
        every duration measured the platform rather than the world (V11).
        """
        from vision_os.kernel.events import RegionTransition

        from .conftest import make_region

        registry.set_regions(CAMERA, (make_region(),))
        subscription = bus.subscribe([RegionTransition])
        for seq in range(6):
            registry.ingest(
                CAMERA,
                make_update([make_track(box=Box(0.45, 0.5, 0.55, 0.7), seq=seq)], seq=seq),
            )
        registry.ingest(
            CAMERA,
            make_update([make_track(box=Box(0.02, 0.02, 0.08, 0.10), seq=10)], seq=10),
        )
        exits = [e for e in subscription.drain() if not e.entered]
        assert exits
        # Five frames at 200 ms of capture time, not however long the test ran.
        assert exits[0].dwell_ms == pytest.approx(2_000, abs=1)

    def test_provenance_carries_the_real_config_revision(self, registry) -> None:
        """Regression: provenance was hardcoded to a placeholder revision, which
        makes a result unreproducible six months later (V4).
        """
        drive(registry, 4)
        assert registry.active(CAMERA)[0].provenance.config_revision == "test"

    def test_a_confirmed_object_is_never_shed(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Withdrawing an assertion under memory pressure would make the
        platform's claims a function of its load."""
        registry = build(
            clock, bus, metrics, registry_config, registry_provenance,
            min_observations_to_confirm=1, max_objects_per_camera=4,
        )
        for seq in range(6):
            registry.ingest(CAMERA, make_update(crowd(4, seq), seq=seq))
        confirmed = [
            o for o in registry.objects(CAMERA) if o.lifecycle is LifecycleState.ACTIVE
        ]
        ids = {o.object_id for o in confirmed}

        for seq in range(6, 12):
            registry.ingest(CAMERA, make_update(crowd(8, seq), seq=seq))

        surviving = {o.object_id for o in registry.objects(CAMERA)}
        assert ids <= surviving, "a confirmed object was shed under pressure"
