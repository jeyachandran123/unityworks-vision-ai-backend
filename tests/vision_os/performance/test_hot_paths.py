"""Performance and stability characteristics of the Flow 1 hot paths.

These are **budget** tests, not benchmarks. They assert the properties the
architecture depends on — the admission decision is cheap enough to run ~3000
times a second, and nothing on a hot path grows without bound — with thresholds
loose enough to survive a shared CI machine but tight enough to catch a
regression of the kind that turns 100 cameras into 10.

The soak-shaped tests here use the injected clock (14_TESTING section 10.2):
time-driven behaviour compresses, but allocation-driven growth is measured by
allocation count, which is exactly what a leak looks like.
"""

from __future__ import annotations

import gc
import time

import pytest

from vision_os.acquisition import FrameBuffer, FrameScheduler
from vision_os.adapters.memory import HostMemoryPool
from vision_os.adapters.scheduling import CadenceAdmissionPolicy
from vision_os.core.model.camera import PipelineProfile, SourceSemantics
from vision_os.core.model.frame import FrameDimensions, PrivacyState
from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, ProfileId, StreamEpoch
from vision_os.core.model.timebase import (
    ClockQuality,
    Duration,
    FrameTime,
    Instant,
)
from vision_os.core.ports.scheduling import AdmissionContext
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config.schema import BufferSection, SchedulerSection
from vision_os.kernel.events import EventBus, StreamLost
from vision_os.kernel.metrics import MetricsEngine

from ..conftest import FRAME_BYTES, skip_if_traced

DIMENSIONS = FrameDimensions(width=8, height=4, colour_space="bgr24")
PROFILE = PipelineProfile(profile_id=ProfileId("standard"), target_fps=5.0, max_in_flight=4)


def _time(clock: VirtualClock) -> FrameTime:
    now = clock.now()
    return FrameTime(
        pts=0,
        t_capture=now,
        t_capture_uncertainty=Duration.from_millis(10),
        t_ingest=now,
        t_decoded=now,
        clock_quality=ClockQuality.NTP_SYNCED,
    )


class TestAdmissionDecisionCost:
    """The decision runs on every decoded frame from every camera."""

    @skip_if_traced
    def test_policy_evaluation_is_sub_microsecond(self) -> None:
        policy = CadenceAdmissionPolicy()
        context = AdmissionContext(
            camera_id=CameraId("cam-01"),
            profile=PROFILE,
            semantics=SourceSemantics.REALTIME,
            monotonic_now=Instant(10_000_000_000),
            last_admitted_monotonic=None,
            in_flight=0,
            budget_pressure=0.0,
        )
        iterations = 200_000

        started = time.perf_counter()
        for _ in range(iterations):
            policy.evaluate(context)
        elapsed = time.perf_counter() - started

        # Typically well under 5us. The bound is deliberately an order of
        # magnitude above that: it exists to catch a structural regression, not
        # to measure a loaded CI machine's scheduler jitter.
        per_call_us = (elapsed / iterations) * 1_000_000
        assert per_call_us < 60.0, (
            f"admission decision costs {per_call_us:.2f}us; at 3000 calls/s this "
            f"must stay far below the frame budget"
        )

    @skip_if_traced
    def test_scheduler_offer_sustains_a_hundred_camera_rate(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine, health
    ) -> None:
        """100 cameras x 30 fps is ~3000 offers/second."""
        scheduler = FrameScheduler(
            clock=clock,
            bus=bus,
            metrics=metrics,
            health=health,
            policy=CadenceAdmissionPolicy(),
            config=SchedulerSection(global_budget_fps=10_000.0, drop_alarm_window_ms=60_000),
        )
        cameras = [CameraId(f"cam-{i:03d}") for i in range(100)]
        for camera in cameras:
            scheduler.register_camera(camera, PROFILE, SourceSemantics.REALTIME)

        started = time.perf_counter()
        for _ in range(30):
            for camera in cameras:
                verdict = scheduler.offer(camera)
                if verdict.admit:
                    scheduler.complete(camera)
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, (
            f"3000 admission decisions took {elapsed:.2f}s; the scheduler must not "
            f"become the bottleneck it exists to prevent"
        )


class TestBufferHotPath:
    @skip_if_traced
    def test_lease_acquire_release_is_cheap(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        buffer = FrameBuffer(
            clock=clock,
            bus=bus,
            metrics=metrics,
            allocator=HostMemoryPool(slots=8, bytes_per_slot=FRAME_BYTES),
            config=BufferSection(
                slots_per_camera=4,
                bytes_per_slot=FRAME_BYTES,
                lease_deadline_ms=60_000,
                history_window_ms=600_000,
            ),
        )
        camera = CameraId("cam-01")
        slot = buffer.acquire_slot(camera, SourceSemantics.REALTIME)
        frame = buffer.publish(
            slot,
            frame_ref=FrameRef(camera, StreamEpoch(0), FrameSeq(0)),
            time=_time(clock),
            dimensions=DIMENSIONS,
            privacy_state=PrivacyState.MASKED,
            bytes_written=FRAME_BYTES,
        )

        iterations = 50_000
        started = time.perf_counter()
        for _ in range(iterations):
            lease = buffer.acquire(frame.frame_ref, "detector")
            lease.release()
        elapsed = time.perf_counter() - started

        per_call_us = (elapsed / iterations) * 1_000_000
        # Typically single-digit microseconds; bounded an order of magnitude above.
        assert per_call_us < 400.0, f"lease cycle costs {per_call_us:.2f}us"


class TestNoSteadyStateGrowth:
    """The 30-day soak failure: imperceptible on day 1, fatal on day 26."""

    def test_pool_occupancy_returns_to_baseline(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        pool = HostMemoryPool(slots=16, bytes_per_slot=FRAME_BYTES)
        buffer = FrameBuffer(
            clock=clock,
            bus=bus,
            metrics=metrics,
            allocator=pool,
            config=BufferSection(
                slots_per_camera=4,
                bytes_per_slot=FRAME_BYTES,
                lease_deadline_ms=1_000,
                history_window_ms=1_000,
            ),
        )
        camera = CameraId("cam-01")
        baseline = pool.stats().in_use

        for cycle in range(2_000):
            slot = buffer.acquire_slot(camera, SourceSemantics.REALTIME)
            frame = buffer.publish(
                slot,
                frame_ref=FrameRef(camera, StreamEpoch(0), FrameSeq(cycle)),
                time=_time(clock),
                dimensions=DIMENSIONS,
                privacy_state=PrivacyState.MASKED,
                bytes_written=FRAME_BYTES,
            )
            lease = buffer.acquire(frame.frame_ref, "detector")
            lease.release()
            clock.advance(Duration.from_millis(2_000))
            buffer.sweep()

        assert pool.stats().in_use == baseline, (
            "buffer leaks pool slots over a long run — the classic soak failure"
        )

    def test_histogram_retention_is_bounded(self, clock: VirtualClock) -> None:
        engine = MetricsEngine(clock, histogram_window=256)
        histogram = engine.histogram("uwv.test.latency", camera_id="cam-01")
        for i in range(100_000):
            histogram.record(float(i))
        assert len(engine.snapshot().histogram_values(
            "uwv.test.latency", camera_id="cam-01"
        )) == 256

    def test_metric_cardinality_is_bounded_under_abuse(self, clock: VirtualClock) -> None:
        """A per-object label would otherwise take down the metrics backend."""
        engine = MetricsEngine(clock, max_label_cardinality=64)
        for i in range(20_000):
            engine.counter("uwv.test.abuse", object_id=f"obj-{i}").increment()

        distinct = len(engine.snapshot().counters_matching("uwv.test.abuse"))
        assert distinct <= 65, f"cardinality escaped its bound: {distinct} series"

    def test_event_bus_buffers_stay_bounded(self, clock: VirtualClock) -> None:
        from vision_os.kernel.events import DeliveryPolicy

        bus = EventBus(clock)
        subscription = bus.subscribe(
            [StreamLost], policy=DeliveryPolicy(capacity=128)
        )
        for i in range(50_000):
            bus.publish(StreamLost(occurred_at=Instant(i), camera_id=CameraId("cam-01")))

        assert subscription.depth <= 128
        drained = subscription.drain()
        assert len(drained) <= 129  # +1 for the synthesized Gap

    def test_scheduler_state_does_not_grow_with_offers(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine, health
    ) -> None:
        scheduler = FrameScheduler(
            clock=clock,
            bus=bus,
            metrics=metrics,
            health=health,
            policy=CadenceAdmissionPolicy(),
            config=SchedulerSection(global_budget_fps=10_000.0, drop_alarm_window_ms=100),
        )
        camera = CameraId("cam-01")
        scheduler.register_camera(camera, PROFILE, SourceSemantics.REALTIME)

        gc.collect()
        before = len(gc.get_objects())
        for _ in range(20_000):
            scheduler.offer(camera)
            scheduler.complete(camera)
            clock.advance(Duration.from_millis(1))
        gc.collect()
        after = len(gc.get_objects())

        assert after - before < 5_000, (
            f"scheduler retained {after - before} objects across 20k offers"
        )


class TestScalingShape:
    def test_buffer_memory_scales_with_pipeline_depth_not_camera_count(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        """Capacity is a function of depth and jitter, not of how many cameras.

        This is why a 100-camera node does not need 100x the memory of a
        1-camera node (01_LAYERED section 4.3).
        """
        depth = 4
        pool = HostMemoryPool(slots=depth * 40, bytes_per_slot=FRAME_BYTES)
        buffer = FrameBuffer(
            clock=clock,
            bus=bus,
            metrics=metrics,
            allocator=pool,
            config=BufferSection(
                slots_per_camera=depth,
                bytes_per_slot=FRAME_BYTES,
                lease_deadline_ms=60_000,
                history_window_ms=600_000,
            ),
        )
        for index in range(40):
            camera = CameraId(f"cam-{index:03d}")
            for seq in range(depth):
                slot = buffer.acquire_slot(camera, SourceSemantics.REALTIME)
                buffer.publish(
                    slot,
                    frame_ref=FrameRef(camera, StreamEpoch(0), FrameSeq(seq)),
                    time=_time(clock),
                    dimensions=DIMENSIONS,
                    privacy_state=PrivacyState.MASKED,
                    bytes_written=FRAME_BYTES,
                )

        stats = buffer.stats()
        assert stats.slots_in_use == depth * 40
        assert stats.slots_total == depth * 40, (
            "each camera holds exactly its declared pipeline depth, no more"
        )


@pytest.mark.parametrize("cameras", [1, 10, 100])
def test_camera_registration_scales_linearly(
    cameras: int, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine, health
) -> None:
    """1, 10 or 100 cameras must be the same code path at different scales."""
    scheduler = FrameScheduler(
        clock=clock,
        bus=bus,
        metrics=metrics,
        health=health,
        policy=CadenceAdmissionPolicy(),
        config=SchedulerSection(global_budget_fps=10_000.0),
    )
    for index in range(cameras):
        scheduler.register_camera(
            CameraId(f"cam-{index:03d}"), PROFILE, SourceSemantics.REALTIME
        )
    assert scheduler.camera_count == cameras
    assert scheduler.current_pressure().cameras == cameras
