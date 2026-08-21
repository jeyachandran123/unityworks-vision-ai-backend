"""Determinism and concurrency for the Crop Manager.

Invariant V13 says a replay reproduces the run. For M8 that means something
specific and checkable: **the same objects, demands and pixels produce the same
requests, the same skips, and the same crop ids** — because a crop id is a
content hash, and a hash that moved would mean the pixels moved.

The concurrency half tests what §M8's Thread Safety section actually promises:

* trigger state is per-camera single-writer, so two cameras never interleave;
* the budget is a **shared counter** with a short critical section, so a hard
  ceiling holds under contention;
* no camera ever waits on another — the shared thing is a number, not a barrier.
"""

from __future__ import annotations

import threading

from vision_os.core.model.crop import SkipReason
from vision_os.core.model.timebase import Duration, Instant
from vision_os.perception.cropping import (
    CropDeduplicationCache,
    PriorityQueue,
    UnderstandingBudget,
)

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    OTHER_TENANT,
    TENANT,
    other_sharp_frame,
    frame_context,
    make_demand,
    make_object,
    sharp_frame,
)


class TestDeterministicEvaluation:
    def test_the_same_inputs_produce_the_same_decisions(self, manager) -> None:
        manager.register_demand(make_demand())
        objects = [make_object(object_id=f"obj-{i}") for i in range(10)]

        first = manager.evaluate(objects, frame_context())
        manager.forget_camera(CAMERA)
        second = manager.evaluate(objects, frame_context())

        assert [(r.object_id, r.trigger_reason) for r in first.requests] == [
            (r.object_id, r.trigger_reason) for r in second.requests
        ]
        assert [(s.object_id, s.reason) for s in first.skipped] == [
            (s.object_id, s.reason) for s in second.skipped
        ]

    def test_requests_come_back_in_a_stable_order(self, manager) -> None:
        """Shedding order depends on it; an unstable sort changes what is dropped."""
        manager.register_demand(make_demand())
        objects = [make_object(object_id=f"obj-{i:03d}") for i in range(20)]

        orders = []
        for _ in range(3):
            manager.forget_camera(CAMERA)
            result = manager.evaluate(objects, frame_context())
            orders.append([r.object_id for r in result.requests])
        assert orders[0] == orders[1] == orders[2]

    def test_the_crop_id_is_stable_across_runs(self, manager) -> None:
        """A content hash that moved would mean the pixels moved."""
        manager.register_demand(make_demand())
        frame = frame_context()
        pixels = sharp_frame()

        ids = []
        for _ in range(3):
            manager.forget_camera(CAMERA)
            request = manager.evaluate([make_object()], frame).requests[0]
            ids.append(manager.extract(request, pixels=pixels, frame=frame).crop_id)
        assert len(set(ids)) == 1

    def test_different_pixels_hash_differently(self, manager) -> None:
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        sharp = manager.extract(request, pixels=sharp_frame(), frame=frame)
        other = manager.extract(request, pixels=other_sharp_frame(), frame=frame)
        assert sharp.crop_id != other.crop_id

    def test_the_transform_is_reproducible(self, manager) -> None:
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        first = manager.extract(request, pixels=sharp_frame(), frame=frame)
        second = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert first.transform == second.transform

    def test_grades_are_reproducible(self, manager) -> None:
        """A gate rejection six months old must be reproducible from a replay."""
        manager.register_demand(make_demand())
        frame = frame_context()
        request = manager.evaluate([make_object()], frame).requests[0]
        first = manager.extract(request, pixels=sharp_frame(), frame=frame)
        second = manager.extract(request, pixels=sharp_frame(), frame=frame)
        assert first.quality == second.quality

    def test_skip_attribution_is_reproducible(self, manager) -> None:
        objects = [make_object(object_id=f"obj-{i}") for i in range(8)]
        runs = []
        for _ in range(3):
            manager.forget_camera(CAMERA)
            result = manager.evaluate(objects, frame_context())
            runs.append(result.skips_by_reason())
        assert runs[0] == runs[1] == runs[2]
        assert runs[0] == {SkipReason.NO_DEMAND: 8}


class TestSharedBudgetUnderContention:
    def test_the_ceiling_holds_across_threads(self) -> None:
        """The whole reason the budget is shared rather than partitioned.

        A per-camera cap cannot stop 100 cameras each staying under their own
        limit while collectively exhausting one GPU.
        """
        budget = UnderstandingBudget(
            ceiling_per_hour=3_600.0,
            window=Duration.from_millis(3_600_000),
            now=Instant(0),
        )
        granted: list[int] = []
        lock = threading.Lock()

        def spend() -> None:
            local = 0
            for _ in range(500):
                if budget.try_spend(Instant(0)):
                    local += 1
            with lock:
                granted.append(local)

        threads = [threading.Thread(target=spend) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        total = sum(granted)
        assert total == 3_600, (
            f"{total} calls were granted against a 3600 ceiling; a budget that "
            f"can be exceeded under load is not a budget"
        )

    def test_no_thread_starves(self) -> None:
        """A shared counter, not a barrier: nobody waits on anybody."""
        budget = UnderstandingBudget(
            ceiling_per_hour=3_600.0,
            window=Duration.from_millis(3_600_000),
            now=Instant(0),
        )
        results: dict[int, int] = {}
        lock = threading.Lock()

        def spend(index: int) -> None:
            local = sum(1 for _ in range(200) if budget.try_spend(Instant(0)))
            with lock:
                results[index] = local

        threads = [threading.Thread(target=spend, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 6, "every thread completed"
        assert all(count > 0 for count in results.values()), (
            f"a thread got nothing: {results}; the critical section is a counter "
            f"increment and must not serialize progress"
        )

    def test_refunds_are_thread_safe(self) -> None:
        budget = UnderstandingBudget(
            ceiling_per_hour=3_600.0,
            window=Duration.from_millis(3_600_000),
            now=Instant(0),
        )

        def churn() -> None:
            for _ in range(200):
                if budget.try_spend(Instant(0)):
                    budget.refund()

        threads = [threading.Thread(target=churn) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert budget.credit <= 3_600.0, "refunds must not manufacture credit"
        assert budget.status(Instant(0)).spent_in_window >= 0


class TestConcurrentCacheAccess:
    def test_the_cache_stays_bounded_under_contention(self) -> None:
        from vision_os.core.model.ids import CropId

        cache = CropDeduplicationCache(capacity=64)

        def fill(offset: int) -> None:
            for index in range(500):
                cache.put(TENANT, f"hash-{offset}-{index}", CropId("c"))

        threads = [threading.Thread(target=fill, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(cache) == 64, (
            f"cache grew to {len(cache)} under contention; an unbounded dedup "
            f"cache is a memory leak that looks like a hit-rate improvement"
        )

    def test_tenant_isolation_holds_under_contention(self) -> None:
        from vision_os.core.model.ids import CropId

        cache = CropDeduplicationCache(capacity=256)

        def fill(tenant) -> None:
            for index in range(200):
                cache.put(tenant, f"shared-{index}", CropId(f"{tenant}-{index}"))

        threads = [
            threading.Thread(target=fill, args=(tenant,))
            for tenant in (TENANT, OTHER_TENANT)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        mine = cache.get(TENANT, "shared-0")
        theirs = cache.get(OTHER_TENANT, "shared-0")
        assert mine != theirs, "tenant keys must never collide, contention or not"


class TestPerCameraPartitioning:
    def test_two_cameras_keep_separate_trigger_state(self, manager) -> None:
        manager.register_demand(make_demand())
        manager.evaluate([make_object()], frame_context())
        manager.evaluate(
            [make_object(camera=OTHER_CAMERA)],
            frame_context(camera=OTHER_CAMERA),
        )
        assert manager.trigger_state.cameras == (CAMERA, OTHER_CAMERA)
        assert manager.trigger_state.partition(CAMERA).tracked_objects == 1
        assert manager.trigger_state.partition(OTHER_CAMERA).tracked_objects == 1

    def test_forgetting_one_camera_leaves_the_other(self, manager) -> None:
        manager.register_demand(make_demand())
        manager.evaluate([make_object()], frame_context())
        manager.evaluate(
            [make_object(camera=OTHER_CAMERA)], frame_context(camera=OTHER_CAMERA)
        )
        manager.forget_camera(CAMERA)
        assert manager.trigger_state.cameras == (OTHER_CAMERA,)

    def test_no_cross_camera_synchronization_exists(self) -> None:
        """§M8: per-camera single-writer, and the store holds no lock.

        A lock in the trigger-state store would suggest cross-camera access is
        expected. It is not — the runtime serializes per camera, and the only
        genuinely shared thing in M8 is the budget counter.
        """
        from vision_os.perception.cropping import TriggerStateStore

        assert not any(
            "lock" in slot.lower() for slot in TriggerStateStore.__slots__
        ), (
            "the trigger-state store must not hold a lock; the partition is the "
            "camera and the runtime is its single writer"
        )


class TestPriorityDeterminism:
    def test_shedding_order_is_reproducible(self) -> None:
        class _Request:
            def __init__(self, name: str, priority_class: str) -> None:
                self.name = name
                self.priority_class = priority_class

        orders = []
        for _ in range(3):
            queue = PriorityQueue(("urgent", "standard", "background"))
            requests = [
                _Request(f"r{i}", ("urgent", "standard", "background")[i % 3])
                for i in range(12)
            ]
            orders.append([r.name for r in queue.order(requests)])
        assert orders[0] == orders[1] == orders[2]

    def test_equal_priorities_keep_arrival_order(self) -> None:
        class _Request:
            def __init__(self, name: str) -> None:
                self.name = name
                self.priority_class = "standard"

        queue = PriorityQueue(("standard",))
        requests = [_Request(f"r{i:02d}") for i in range(20)]
        assert [r.name for r in queue.order(requests)] == [r.name for r in requests]


class TestVirtualClockIndependence:
    def test_evaluation_does_not_depend_on_wall_time(self, manager, clock) -> None:
        """Attention decisions come from capture time, never from the wall.

        A trigger that fired because the platform was slow would make the
        evidence a measurement of the platform rather than of the world (V11).
        """
        manager.register_demand(make_demand())
        objects = [make_object(object_id=f"obj-{i}") for i in range(5)]

        first = manager.evaluate(objects, frame_context())
        clock.advance(Duration.from_millis(5_000))
        manager.forget_camera(CAMERA)
        second = manager.evaluate(objects, frame_context())

        assert [r.object_id for r in first.requests] == [
            r.object_id for r in second.requests
        ]
        assert len(first.skipped) == len(second.skipped)
