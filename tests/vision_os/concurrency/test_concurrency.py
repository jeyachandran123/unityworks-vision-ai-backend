"""Concurrency tests — the hazards the design claims to make impossible.

08_RUNTIME section 9 lists the failure modes this concurrency model exists to
prevent. Each one below is exercised rather than assumed, because a race that
reproduces once a fortnight will first appear in a customer's 400-camera
deployment, not in CI.
"""

from __future__ import annotations

import asyncio
import threading

from vision_os.acquisition import FrameBuffer
from vision_os.adapters.memory import HostMemoryPool
from vision_os.core.errors import PoolExhaustedError
from vision_os.core.model.camera import SourceSemantics
from vision_os.core.model.frame import FrameDimensions, PrivacyState
from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
from vision_os.core.model.timebase import (
    ClockQuality,
    Duration,
    FrameTime,
    Instant,
)
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config.schema import BufferSection
from vision_os.kernel.events import DeliveryPolicy, EventBus, Gap, StreamLost
from vision_os.kernel.metrics import MetricsEngine

from ..conftest import FRAME_BYTES

DIMENSIONS = FrameDimensions(width=8, height=4, colour_space="bgr24")


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


def _wide_buffer(clock: VirtualClock, bus: EventBus, metrics: MetricsEngine) -> FrameBuffer:
    return FrameBuffer(
        clock=clock,
        bus=bus,
        metrics=metrics,
        allocator=HostMemoryPool(slots=512, bytes_per_slot=FRAME_BYTES),
        config=BufferSection(
            slots_per_camera=64,
            bytes_per_slot=FRAME_BYTES,
            lease_deadline_ms=60_000,
            history_window_ms=600_000,
        ),
    )


def _publish(buffer: FrameBuffer, clock: VirtualClock, camera: CameraId, seq: int):
    slot = buffer.acquire_slot(camera, SourceSemantics.REALTIME)
    slot.memory()[:FRAME_BYTES] = bytes([seq % 251]) * FRAME_BYTES
    return buffer.publish(
        slot,
        frame_ref=FrameRef(camera, StreamEpoch(0), FrameSeq(seq)),
        time=_time(clock),
        dimensions=DIMENSIONS,
        privacy_state=PrivacyState.MASKED,
        bytes_written=FRAME_BYTES,
    )


class TestBufferUnderConcurrentAccess:
    def test_concurrent_leases_on_one_frame_are_consistent(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        """Published frames are immutable, so readers need no lock at all."""
        buffer = _wide_buffer(clock, bus, metrics)
        camera = CameraId("cam-01")
        frame = _publish(buffer, clock, camera, 0)

        errors: list[BaseException] = []
        payloads: list[bytes] = []
        barrier = threading.Barrier(8)

        def reader() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(200):
                    lease = buffer.acquire(frame.frame_ref, "reader")
                    payloads.append(bytes(lease.pixels()[:4]))
                    lease.release()
            except BaseException as exc:  # noqa: BLE001 - surfaced by assertion
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors, errors
        assert len(set(payloads)) == 1, "readers must never observe torn pixels"
        assert buffer.stats().leases_active == 0, "every lease must be accounted for"

    def test_concurrent_publish_across_cameras_has_no_contention_errors(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        """Per-camera rings: each camera writes only to its own ring."""
        buffer = _wide_buffer(clock, bus, metrics)
        errors: list[BaseException] = []
        published: list[FrameRef] = []
        lock = threading.Lock()

        def writer(index: int) -> None:
            camera = CameraId(f"cam-{index:02d}")
            try:
                for seq in range(50):
                    frame = _publish(buffer, clock, camera, seq)
                    with lock:
                        published.append(frame.frame_ref)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not errors, errors
        assert len(published) == 400
        assert len(set(published)) == 400, "FrameRefs must be globally unique"

    def test_concurrent_acquire_and_evict_never_corrupts(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        """Eviction under concurrent readers must never free a leased frame."""
        buffer = FrameBuffer(
            clock=clock,
            bus=bus,
            metrics=metrics,
            allocator=HostMemoryPool(slots=64, bytes_per_slot=FRAME_BYTES),
            config=BufferSection(
                slots_per_camera=4,
                bytes_per_slot=FRAME_BYTES,
                lease_deadline_ms=60_000,
                history_window_ms=600_000,
            ),
        )
        camera = CameraId("cam-01")
        errors: list[BaseException] = []
        stop = threading.Event()

        def churn() -> None:
            seq = 0
            try:
                while not stop.is_set() and seq < 300:
                    _publish(buffer, clock, camera, seq)
                    seq += 1
            except PoolExhaustedError:
                pass
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                while not stop.is_set():
                    for seq in range(300):
                        lease = buffer.try_acquire(
                            FrameRef(camera, StreamEpoch(0), FrameSeq(seq)), "reader"
                        )
                        if lease is None:
                            continue
                        # Holding a lease guarantees the frame stays readable.
                        assert len(lease.pixels()) == FRAME_BYTES
                        lease.release()
                    break
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=churn), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        stop.set()

        assert not errors, errors

    def test_no_deadlock_between_publish_sweep_and_acquire(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        """The design has no lock hierarchy, so there is no lock-ordering deadlock."""
        buffer = _wide_buffer(clock, bus, metrics)
        camera = CameraId("cam-01")
        errors: list[BaseException] = []
        done = threading.Event()

        def publisher() -> None:
            try:
                for seq in range(200):
                    _publish(buffer, clock, camera, seq)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def sweeper() -> None:
            try:
                while not done.is_set():
                    buffer.sweep()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def acquirer() -> None:
            try:
                for seq in range(200):
                    lease = buffer.try_acquire(
                        FrameRef(camera, StreamEpoch(0), FrameSeq(seq)), "reader"
                    )
                    if lease is not None:
                        lease.release()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=sweeper),
            threading.Thread(target=acquirer),
        ]
        for thread in threads:
            thread.start()
        threads[0].join(timeout=20)
        threads[2].join(timeout=20)
        done.set()
        threads[1].join(timeout=20)

        assert not any(thread.is_alive() for thread in threads), "possible deadlock"
        assert not errors, errors


class TestEventBusConcurrency:
    def test_concurrent_publish_is_safe_and_counted(self, clock: VirtualClock) -> None:
        bus = EventBus(clock)
        subscription = bus.subscribe(
            [StreamLost], policy=DeliveryPolicy(capacity=100_000)
        )
        errors: list[BaseException] = []

        def publisher(index: int) -> None:
            try:
                for i in range(200):
                    bus.publish(
                        StreamLost(
                            occurred_at=Instant(i),
                            partition_key=f"cam-{index}",
                            camera_id=CameraId(f"cam-{index}"),
                        )
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=publisher, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert not errors, errors
        assert bus.stats().published == 1200
        assert len(subscription.drain()) == 1200

    def test_slow_subscriber_never_stalls_a_publisher(self, clock: VirtualClock) -> None:
        """One slow consumer must not stall the platform (08_RUNTIME section 5.2)."""
        bus = EventBus(clock)
        slow = bus.subscribe([StreamLost], policy=DeliveryPolicy(capacity=4))
        fast = bus.subscribe([StreamLost], policy=DeliveryPolicy(capacity=10_000))

        for i in range(1_000):
            bus.publish(StreamLost(occurred_at=Instant(i), camera_id=CameraId("cam-01")))

        fast_events = [e for e in fast.drain() if isinstance(e, StreamLost)]
        assert len(fast_events) == 1_000, "the healthy subscriber is unaffected"

        slow_events = slow.drain()
        assert any(isinstance(e, Gap) for e in slow_events), "drops must be announced"
        assert len(slow_events) <= 6


class TestMetricsConcurrency:
    def test_concurrent_recording_is_lossless(self, clock: VirtualClock) -> None:
        """Recording happens on every hot path; it must never contend or lose counts."""
        engine = MetricsEngine(clock)
        errors: list[BaseException] = []

        def recorder() -> None:
            try:
                counter = engine.counter("uwv.test.concurrent", camera_id="cam-01")
                for _ in range(2_000):
                    counter.increment()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=recorder) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not errors, errors
        assert engine.snapshot().counter_value(
            "uwv.test.concurrent", camera_id="cam-01"
        ) == 16_000


class TestActorOrdering:
    async def test_per_camera_frame_order_is_preserved(
        self, sources, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Out-of-order frames corrupt tracking silently, so ordering is a hard rule."""
        from ..conftest import make_bindings, make_camera, make_frames
        from ..integration.test_acquisition_flow import FrameCollector

        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(make_frames(12), clock=clock, dimensions=dimensions),
            collector,
        )
        await collector.wait(12)

        seqs = [f.frame_ref.frame_seq for f in collector.frames]
        assert seqs == sorted(seqs) == list(range(12))
        await sources.close_all()

    async def test_many_cameras_run_concurrently_without_interference(
        self, sources, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        from ..conftest import make_bindings, make_camera, make_frames
        from ..integration.test_acquisition_flow import FrameCollector

        collectors = {}
        for index in range(12):
            camera_id = CameraId(f"cam-{index:02d}")
            collector = FrameCollector()
            collectors[camera_id] = collector
            sources.open(
                make_camera(camera_id),
                make_bindings(make_frames(6), clock=clock, dimensions=dimensions),
                collector,
            )

        await asyncio.gather(*(c.wait(6) for c in collectors.values()))

        for camera_id, collector in collectors.items():
            assert len(collector.frames) == 6
            assert all(f.frame_ref.camera_id == camera_id for f in collector.frames)
            assert [f.frame_ref.frame_seq for f in collector.frames] == [0, 1, 2, 3, 4, 5]
        await sources.close_all()
