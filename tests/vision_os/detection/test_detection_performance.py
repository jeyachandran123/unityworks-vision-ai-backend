"""Performance and stability characteristics of the detection layer.

Budget tests, not benchmarks. They assert the properties the architecture depends
on — batching actually amortizes, nothing on a hot path grows without bound, and
detection failure never escalates under sustained load — with thresholds loose
enough for a shared CI machine but tight enough to catch a regression of the kind
that turns 100 cameras into 10.
"""

from __future__ import annotations

import asyncio
import gc
import time

import pytest

from vision_os.adapters.detection import ReferenceDetector, ScriptedDetection
from vision_os.core.errors import DetectionFailedError
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import CameraId, ClassId, FrameRef, FrameSeq, StreamEpoch
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.ports.detection import DetectionRequest, FrameView
from vision_os.kernel.models import DeviceBroker
from vision_os.perception.detection import (
    BatchKey,
    DetectionScheduler,
    DeviceWorker,
)

from ..conftest import skip_if_traced

WIDTH, HEIGHT = 64, 32
FRAME_BYTES = WIDTH * HEIGHT * 3
DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)
GIGABYTE = 1024**3


def _view(seq: int, camera: str = "cam-01") -> FrameView:
    return FrameView(
        frame_ref=FrameRef(CameraId(camera), StreamEpoch(1), FrameSeq(seq)),
        dimensions=DIMENSIONS,
        pixels=memoryview(bytearray(FRAME_BYTES)).toreadonly(),
    )


def _key() -> BatchKey:
    return BatchKey(
        model_id="perf",
        model_version="1.0.0",
        precision="fp32",
        inference_width=640,
        inference_height=640,
    )


@pytest.fixture
def perf_detector(clock) -> ReferenceDetector:
    return ReferenceDetector(
        clock=clock,
        producible_classes=(ClassId("person"),),
        script=(ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
    )


class TestAdapterHotPath:
    @skip_if_traced
    def test_translation_cost_is_bounded(self, perf_detector) -> None:
        """Runs once per detection per frame; at 100 cameras that is thousands/s."""
        frames = [_view(i) for i in range(8)]
        request = DetectionRequest(min_confidence=0.25, max_detections=100)

        for _ in range(50):
            perf_detector.detect(frames, request)

        iterations = 400
        started = time.perf_counter()
        for _ in range(iterations):
            perf_detector.detect(frames, request)
        elapsed = time.perf_counter() - started

        per_frame_us = (elapsed / (iterations * len(frames))) * 1_000_000
        assert per_frame_us < 1_500.0, (
            f"adapter translation costs {per_frame_us:.1f}us/frame; the platform "
            f"budget assumes this is negligible beside inference"
        )

    @skip_if_traced
    def test_letterbox_arithmetic_is_cheap(self) -> None:
        """Called per detection; a slow inverse taxes every frame."""
        from vision_os.adapters.detection.letterbox import (
            compute_transform,
            invert_to_normalized,
        )

        transform = compute_transform(
            source_width=1920, source_height=1080, target_width=640, target_height=640
        )
        iterations = 200_000
        started = time.perf_counter()
        for _ in range(iterations):
            invert_to_normalized(transform, 64.0, 64.0, 192.0, 320.0)
        elapsed = time.perf_counter() - started

        per_call_us = (elapsed / iterations) * 1_000_000
        # Typically ~1us. Bounded far above so a busy machine does not fail it.
        assert per_call_us < 80.0, f"letterbox inverse costs {per_call_us:.2f}us"


class TestBatchingAmortizes:
    async def test_a_batch_costs_less_per_frame_than_singles(
        self, clock, perf_detector
    ) -> None:
        """The whole economic case for a shared inference tier.

        A fixed per-call overhead is amortized across the batch; if this ever
        stops being true, cross-camera batching has stopped paying for itself.
        """
        calls = {"count": 0}

        class CountingDetector(ReferenceDetector):
            def detect(self, frames, request):
                calls["count"] += 1
                return super().detect(frames, request)

        detector = CountingDetector(
            clock=clock,
            producible_classes=(ClassId("person"),),
            script=(ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
        )
        worker = DeviceWorker(clock=clock, detector=detector, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=8,
            max_wait=Duration.from_millis(2), queue_capacity=64,
            inference_timeout=Duration.from_millis(2_000),
        )
        key = _key()
        await asyncio.gather(
            *(
                scheduler.submit(
                    key=key, frame_ref=_view(i, f"cam-{i:02d}").frame_ref,
                    camera_id=CameraId(f"cam-{i:02d}"),
                    view=_view(i, f"cam-{i:02d}"), request=DetectionRequest(),
                )
                for i in range(16)
            )
        )
        assert calls["count"] < 16, (
            f"16 frames took {calls['count']} inference calls; batching is not "
            f"amortizing and a shared GPU tier buys nothing"
        )
        await scheduler.close()

    @skip_if_traced
    async def test_scheduler_sustains_a_hundred_camera_rate(
        self, clock, perf_detector
    ) -> None:
        """100 cameras at 5 fps is ~500 detections/second."""
        worker = DeviceWorker(clock=clock, detector=perf_detector, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=16,
            max_wait=Duration(0), queue_capacity=256,
            inference_timeout=Duration.from_millis(5_000),
        )
        key = _key()

        # Five waves of 100, not 500 at once: cameras submit at their cadence, and
        # firing everything simultaneously would exceed the bounded queue and
        # (correctly) shed — measuring the shed path rather than throughput.
        started = time.perf_counter()
        for wave in range(5):
            await asyncio.gather(
                *(
                    scheduler.submit(
                        key=key,
                        frame_ref=_view(wave * 100 + i, f"cam-{i:03d}").frame_ref,
                        camera_id=CameraId(f"cam-{i:03d}"),
                        view=_view(wave * 100 + i, f"cam-{i:03d}"),
                        request=DetectionRequest(),
                    )
                    for i in range(100)
                )
            )
        elapsed = time.perf_counter() - started

        assert elapsed < 30.0, (
            f"500 detections took {elapsed:.2f}s; the scheduler must not become "
            f"the bottleneck it exists to prevent"
        )
        await scheduler.close()


class TestNoSteadyStateGrowth:
    """Detection runs continuously for months; a per-frame leak is fatal by day 26."""

    async def test_scheduler_queue_returns_to_empty(self, clock, perf_detector) -> None:
        worker = DeviceWorker(clock=clock, detector=perf_detector, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=4,
            max_wait=Duration(0), queue_capacity=64,
            inference_timeout=Duration.from_millis(2_000),
        )
        key = _key()
        for cycle in range(50):
            await asyncio.gather(
                *(
                    scheduler.submit(
                        key=key, frame_ref=_view(cycle * 4 + i).frame_ref,
                        camera_id=CameraId("cam-01"), view=_view(cycle * 4 + i),
                        request=DetectionRequest(),
                    )
                    for i in range(4)
                )
            )
        assert scheduler.depth == 0, "the queue leaked entries across 200 frames"
        await scheduler.close()

    async def test_failed_batches_do_not_leak_queue_slots(self, clock) -> None:
        """A failure path that leaks is a leak that only shows up in a bad week."""

        async def failing(key, batch):
            raise RuntimeError("device fault")

        scheduler = DetectionScheduler(
            clock=clock, executor=failing, max_batch_size=2,
            max_wait=Duration(0), queue_capacity=8,
            inference_timeout=Duration.from_millis(500),
        )
        key = _key()
        for i in range(60):
            with pytest.raises(RuntimeError):
                await scheduler.submit(
                    key=key, frame_ref=_view(i).frame_ref, camera_id=CameraId("cam-01"),
                    view=_view(i), request=DetectionRequest(),
                )
        assert scheduler.depth == 0
        await scheduler.close()

    def test_adapter_retains_nothing_across_calls(self, perf_detector) -> None:
        frames = [_view(i) for i in range(4)]
        request = DetectionRequest(min_confidence=0.25)
        for _ in range(100):
            perf_detector.detect(frames, request)

        gc.collect()
        before = len(gc.get_objects())
        for _ in range(600):
            perf_detector.detect(frames, request)
        gc.collect()
        growth = len(gc.get_objects()) - before

        assert growth < 20_000, (
            f"adapter retained {growth} objects across 600 batches"
        )

    def test_device_broker_returns_to_baseline(self, cpu_devices, gpu_devices) -> None:
        broker = DeviceBroker((cpu_devices, gpu_devices), allow_cpu_fallback=True)
        baseline = broker.report().total_reserved_bytes
        for _ in range(500):
            reservation = broker.reserve(owner="churn", bytes_required=GIGABYTE)
            broker.release(reservation)
        assert broker.report().total_reserved_bytes == baseline

    @skip_if_traced
    def test_taxonomy_lookup_is_allocation_light(self, taxonomy) -> None:
        """Called once per detection per frame."""
        iterations = 100_000
        started = time.perf_counter()
        for _ in range(iterations):
            taxonomy.has(ClassId("person"))
            taxonomy.is_a(ClassId("vehicle.forklift"), ClassId("vehicle"))
        elapsed = time.perf_counter() - started

        per_call_us = (elapsed / iterations) * 1_000_000
        assert per_call_us < 80.0, f"taxonomy lookup costs {per_call_us:.2f}us"


class TestStressAndDegradation:
    async def test_sustained_failure_never_escalates(self, clock) -> None:
        """A detector failing every call must not take anything else down."""

        class AlwaysFails(ReferenceDetector):
            def detect(self, frames, request):
                raise DetectionFailedError("permanent fault")

        worker = DeviceWorker(
            clock=clock,
            detector=AlwaysFails(clock=clock, producible_classes=(ClassId("person"),)),
            device_id="cpu",
        )
        for _ in range(200):
            with pytest.raises(DetectionFailedError):
                await worker.execute([_view(0)], DetectionRequest())

        assert worker.stats.failures == 200
        assert worker.stats.batches == 0, "a failed batch is never counted as work done"

    async def test_mixed_success_and_failure_is_stable(self, clock) -> None:
        """The realistic case: an intermittently faulty device."""
        state = {"calls": 0}

        class Flaky(ReferenceDetector):
            def detect(self, frames, request):
                state["calls"] += 1
                if state["calls"] % 3 == 0:
                    raise DetectionFailedError("intermittent fault")
                return super().detect(frames, request)

        worker = DeviceWorker(
            clock=clock,
            detector=Flaky(
                clock=clock,
                producible_classes=(ClassId("person"),),
                script=(
                    ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),
                ),
            ),
            device_id="cpu",
        )
        succeeded = failed = 0
        for _ in range(90):
            try:
                await worker.execute([_view(0)], DetectionRequest())
                succeeded += 1
            except DetectionFailedError:
                failed += 1

        assert succeeded == 60
        assert failed == 30
        assert worker.stats.batches == 60

    async def test_concurrent_submissions_are_all_answered(
        self, clock, perf_detector
    ) -> None:
        """Nothing may be dropped without an answer — a silent loss is worse."""
        worker = DeviceWorker(clock=clock, detector=perf_detector, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=8,
            max_wait=Duration(0), queue_capacity=512,
            inference_timeout=Duration.from_millis(5_000),
        )
        key = _key()
        results = await asyncio.gather(
            *(
                scheduler.submit(
                    key=key, frame_ref=_view(i, f"cam-{i % 20:02d}").frame_ref,
                    camera_id=CameraId(f"cam-{i % 20:02d}"),
                    view=_view(i, f"cam-{i % 20:02d}"), request=DetectionRequest(),
                )
                for i in range(200)
            ),
            return_exceptions=True,
        )
        assert len(results) == 200
        assert not any(isinstance(r, Exception) for r in results)
        await scheduler.close()

    async def test_close_fails_waiters_explicitly(self, clock) -> None:
        """A submitter left awaiting a future that never resolves is the worst
        shutdown outcome; an error is better than a hang."""
        release = asyncio.Event()

        async def blocking(key, batch):
            await release.wait()
            return []

        scheduler = DetectionScheduler(
            clock=clock, executor=blocking, max_batch_size=1,
            max_wait=Duration(0), queue_capacity=8,
            inference_timeout=Duration.from_millis(5_000),
        )
        key = _key()
        first = asyncio.create_task(
            scheduler.submit(
                key=key, frame_ref=_view(0).frame_ref, camera_id=CameraId("cam-01"),
                view=_view(0), request=DetectionRequest(),
            )
        )
        pending = asyncio.create_task(
            scheduler.submit(
                key=key, frame_ref=_view(1).frame_ref, camera_id=CameraId("cam-01"),
                view=_view(1), request=DetectionRequest(),
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)

        await scheduler.close()
        with pytest.raises(DetectionFailedError, match="closed"):
            await pending

        release.set()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.parametrize("cameras", [1, 10, 100])
async def test_scales_across_camera_counts(clock, perf_detector, cameras: int) -> None:
    """1, 10 or 100 cameras must be the same code path at different scales."""
    worker = DeviceWorker(clock=clock, detector=perf_detector, device_id="cpu")

    async def execute(key, batch):
        return await worker.execute([i.view for i in batch], batch[0].request)

    scheduler = DetectionScheduler(
        clock=clock, executor=execute, max_batch_size=16,
        max_wait=Duration(0), queue_capacity=512,
        inference_timeout=Duration.from_millis(5_000),
    )
    key = _key()
    results = await asyncio.gather(
        *(
            scheduler.submit(
                key=key, frame_ref=_view(i, f"cam-{i:03d}").frame_ref,
                camera_id=CameraId(f"cam-{i:03d}"),
                view=_view(i, f"cam-{i:03d}"), request=DetectionRequest(),
            )
            for i in range(cameras)
        )
    )
    assert len(results) == cameras
    await scheduler.close()
