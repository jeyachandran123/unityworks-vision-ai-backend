"""M2 Video Source Manager and the end-to-end Flow 1 acquisition path.

These are the tests that only a wired pipeline can catch: epoch advance across
reconnect, the fail-closed privacy path, decode-error ladders, actor isolation
between cameras, and deterministic replay.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.acquisition import (
    ActorState,
    FrameBuffer,
    FrameScheduler,
    InMemoryEpochStore,
    VideoSourceManager,
)
from vision_os.adapters.acquisition import (
    FailingMask,
    InMemoryRawSource,
    JsonFileEpochStore,
    PassthroughDecoder,
    PtsClockSync,
    StaticZoneMask,
    UnknownClockSync,
    WallclockHintClockSync,
)
from vision_os.core.model.camera import SourceSemantics
from vision_os.core.model.frame import Frame, FrameDimensions, PrivacyState
from vision_os.core.model.health import ObservabilityReason
from vision_os.core.model.ids import CameraId, PrivacyPolicyId, StreamEpoch
from vision_os.core.model.space import Point, Polygon
from vision_os.core.model.timebase import ClockQuality, Duration, Instant
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.events import EventBus
from vision_os.kernel.health import HealthMonitor
from vision_os.kernel.metrics import MetricName, MetricsEngine

from ..conftest import CAMERA, FRAME_BYTES, make_bindings, make_camera, make_frames


class FrameCollector:
    """A frame sink that records what the acquisition layer produced."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.done = asyncio.Event()
        self.expect = 0

    async def __call__(self, frame: Frame) -> None:
        self.frames.append(frame)
        if self.expect and len(self.frames) >= self.expect:
            self.done.set()

    async def wait(self, count: int, *, timeout: float = 2.0) -> None:
        self.expect = count
        if len(self.frames) >= count:
            return
        await asyncio.wait_for(self.done.wait(), timeout=timeout)


async def _drain(sources: VideoSourceManager, collector: FrameCollector, count: int) -> None:
    await collector.wait(count)


async def _pump(clock: VirtualClock, predicate, *, steps: int = 60, step_ms: int = 50) -> None:
    """Advance virtual time until ``predicate`` holds or the budget is spent.

    Reconnect backoff sleeps on the injected clock, so a test that never
    advances virtual time would wait forever — which is precisely the property
    that makes the platform's timing testable at all (invariant V13).
    """
    for _ in range(steps):
        if predicate():
            return
        clock.advance(Duration.from_millis(step_ms))
        for _ in range(20):
            await asyncio.sleep(0)


class TestFramePublication:
    async def test_frames_are_published_with_monotonic_sequence(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        camera = make_camera()
        collector = FrameCollector()
        sources.open(camera, make_bindings(make_frames(5), clock=clock, dimensions=dimensions), collector)
        await _drain(sources, collector, 5)

        seqs = [f.frame_ref.frame_seq for f in collector.frames]
        assert seqs == [0, 1, 2, 3, 4]
        assert all(f.frame_ref.camera_id == CAMERA for f in collector.frames)
        await sources.close_all()

    async def test_published_frames_carry_mandatory_uncertainty(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        camera = make_camera()
        collector = FrameCollector()
        sources.open(camera, make_bindings(make_frames(3), clock=clock, dimensions=dimensions), collector)
        await _drain(sources, collector, 3)

        for frame in collector.frames:
            assert frame.time.t_capture_uncertainty.ns > 0
            assert isinstance(frame.time.clock_quality, ClockQuality)
        await sources.close_all()

    async def test_frame_pixels_match_the_source_payload(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frames = make_frames(2)
        collector = FrameCollector()
        sources.open(
            make_camera(), make_bindings(frames, clock=clock, dimensions=dimensions), collector
        )
        await _drain(sources, collector, 2)

        published = bytes(collector.frames[0].pixels.readonly_view())
        assert published == frames[0].payload
        await sources.close_all()


class TestEpochDiscipline:
    async def test_reconnect_advances_the_epoch(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        bus: EventBus,
    ) -> None:
        """The bug class the epoch exists to prevent (02_VOM §4.1).

        Without it, frame 100 before and after a reconnect compare equal while
        describing different instants.
        """
        subscription = bus.subscribe(["stream.epoch_advanced"])
        source = InMemoryRawSource(
            make_frames(2),
            clock=clock,
            semantics=SourceSemantics.REALTIME,
            loop=True,
            fail_after=2,
        )
        camera = make_camera(semantics=SourceSemantics.REALTIME)
        collector = FrameCollector()
        sources.open(
            camera,
            make_bindings([], clock=clock, dimensions=dimensions, source=source),
            collector,
        )
        await _pump(clock, lambda: len(collector.frames) >= 4)

        epochs = {f.frame_ref.stream_epoch for f in collector.frames}
        assert len(epochs) >= 2, "each reconnect must mint a new epoch"
        assert subscription.drain()

        refs = [f.frame_ref for f in collector.frames]
        assert len(set(refs)) == len(refs), "FrameRefs must be unique across reconnects"

        seqs_by_epoch: dict[int, list[int]] = {}
        for frame in collector.frames:
            seqs_by_epoch.setdefault(frame.frame_ref.stream_epoch, []).append(
                frame.frame_ref.frame_seq
            )
        for seqs in seqs_by_epoch.values():
            assert seqs == sorted(seqs), "sequence must stay monotonic within an epoch"
        await sources.close_all()

    async def test_decoder_and_clocksync_reset_on_epoch_advance(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """A reconnected stream must not carry reference frames across the break."""
        decoder = PassthroughDecoder(dimensions=dimensions)
        source = InMemoryRawSource(
            make_frames(2),
            clock=clock,
            semantics=SourceSemantics.REALTIME,
            loop=True,
            fail_after=2,
        )
        collector = FrameCollector()
        sources.open(
            make_camera(semantics=SourceSemantics.REALTIME),
            make_bindings(
                [], clock=clock, dimensions=dimensions, source=source, decoder=decoder
            ),
            collector,
        )
        await _pump(clock, lambda: decoder.reset_calls >= 2)
        assert decoder.reset_calls >= 2
        await sources.close_all()

    def test_epoch_store_survives_restart(self, tmp_path) -> None:
        """Without persistence a restart can reuse an epoch."""
        path = tmp_path / "epochs.json"
        first = JsonFileEpochStore(path)
        first.record_epoch(CAMERA, StreamEpoch(7))

        second = JsonFileEpochStore(path)
        assert second.last_epoch(CAMERA) == 7

    def test_epoch_store_never_regresses(self) -> None:
        store = InMemoryEpochStore()
        store.record_epoch(CAMERA, StreamEpoch(5))
        store.record_epoch(CAMERA, StreamEpoch(2))
        assert store.last_epoch(CAMERA) == 5

    def test_corrupt_epoch_store_does_not_prevent_startup(self, tmp_path) -> None:
        path = tmp_path / "epochs.json"
        path.write_text("{ not json", encoding="utf-8")
        assert JsonFileEpochStore(path).last_epoch(CAMERA) == -1


class TestPrivacyFailsClosed:
    async def test_mask_failure_drops_the_frame_and_blinds_the_camera(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        health: HealthMonitor,
        bus: EventBus,
        metrics: MetricsEngine,
    ) -> None:
        """The platform's only fail-closed path (12_SECURITY §2.1).

        A masking failure that proceeds is a compliance incident regardless of
        intent, so the frame is dropped rather than degraded.
        """
        subscription = bus.subscribe(["privacy.mask_failed"])
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(3), clock=clock, dimensions=dimensions, privacy=FailingMask()
            ),
            collector,
        )
        for _ in range(50):
            await asyncio.sleep(0)

        assert collector.frames == [], "no frame may be emitted when masking fails"
        assert subscription.drain()
        assert metrics.snapshot().counter_value(
            MetricName.MASK_FAILURES, camera_id=str(CAMERA)
        ) >= 1

        # Assert on the recorded gap rather than the live state: the stream has
        # since ended and moved to DRAINING, but the blind interval is history
        # and must remain in the record (invariant V8).
        reasons = {gap.reason for gap in health.coverage_gaps(CAMERA)}
        assert ObservabilityReason.PRIVACY_MASK_FAILED in reasons
        await sources.close_all()

    async def test_static_zone_mask_blanks_pixels_before_publication(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """No component ever sees unmasked pixels."""
        mask = StaticZoneMask(
            policy_id=PrivacyPolicyId("neighbour-window"),
            zones=(Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))),),
        )
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(make_frames(1), clock=clock, dimensions=dimensions, privacy=mask),
            collector,
        )
        await _drain(sources, collector, 1)

        frame = collector.frames[0]
        assert frame.privacy_state is PrivacyState.MASKED
        assert set(bytes(frame.pixels.readonly_view())) == {0}
        await sources.close_all()

    async def test_no_mask_policy_reports_unmasked_permitted(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Distinguishing "no policy" from "masked" is auditable, not cosmetic."""
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(make_frames(1), clock=clock, dimensions=dimensions),
            collector,
        )
        await _drain(sources, collector, 1)
        assert collector.frames[0].privacy_state is PrivacyState.UNMASKED_PERMITTED
        await sources.close_all()


class TestDecodeFailures:
    async def test_decode_error_drops_one_frame_and_continues(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        metrics: MetricsEngine,
    ) -> None:
        decoder = PassthroughDecoder(dimensions=dimensions, fail_every=2)
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(6), clock=clock, dimensions=dimensions, decoder=decoder
            ),
            collector,
        )
        await _drain(sources, collector, 3)

        assert len(collector.frames) == 3, "half the packets failed to decode"
        assert metrics.snapshot().counter_value(
            MetricName.DECODE_ERRORS, camera_id=str(CAMERA)
        ) == 3
        await sources.close_all()

    async def test_poison_payload_is_quarantined_without_stopping_the_stream(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """The poison failure class: one bad input must not end a stream."""
        frames = make_frames(4)
        poison = frames[1].payload
        decoder = PassthroughDecoder(
            dimensions=dimensions, poison_payloads=frozenset({poison})
        )
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(frames, clock=clock, dimensions=dimensions, decoder=decoder),
            collector,
        )
        await _drain(sources, collector, 3)
        assert len(collector.frames) == 3
        await sources.close_all()

    async def test_sustained_decode_errors_degrade_observability(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        bus: EventBus,
    ) -> None:
        """Degradation is published, never silent (10_RELIABILITY §4.5).

        A decode-failing camera is ``DEGRADED`` rather than ``BLIND`` — it is
        still connected and still trying — so no coverage *gap* opens. The
        honesty lives in the published transition and its ``effective_rate``.
        """
        subscription = bus.subscribe(["health.coverage_changed"])
        decoder = PassthroughDecoder(dimensions=dimensions, fail_every=1)
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(10), clock=clock, dimensions=dimensions, decoder=decoder
            ),
            collector,
        )
        for _ in range(80):
            await asyncio.sleep(0)

        assert collector.frames == [], "every packet failed to decode"
        transitions = subscription.drain()
        decode_failing = [
            e for e in transitions if getattr(e, "reason", "") == "decode_failing"
        ]
        assert decode_failing, "sustained decode failure must be published"
        assert decode_failing[0].effective_rate == 0.0
        await sources.close_all()


class TestConnectionFailures:
    async def test_connect_failure_backs_off_and_recovers(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        bus: EventBus,
        metrics: MetricsEngine,
    ) -> None:
        subscription = bus.subscribe(["stream.lost"])
        source = InMemoryRawSource(
            make_frames(2), clock=clock, semantics=SourceSemantics.ARCHIVAL, fail_on_open=2
        )
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings([], clock=clock, dimensions=dimensions, source=source),
            collector,
        )

        for _ in range(10):
            clock.advance(Duration.from_millis(200))
            for _ in range(20):
                await asyncio.sleep(0)
            if len(collector.frames) >= 2:
                break

        assert source.open_calls >= 3, "must retry after transient connect failures"
        assert len(collector.frames) == 2
        assert subscription.drain()
        assert metrics.snapshot().counter_value(
            MetricName.CONNECT_FAILURES, camera_id=str(CAMERA)
        ) == 2
        await sources.close_all()

    async def test_camera_is_blind_while_disconnected(
        self,
        sources: VideoSourceManager,
        clock: VirtualClock,
        dimensions: FrameDimensions,
        health: HealthMonitor,
    ) -> None:
        """Absence of frames is published as blindness, never as silence (V8)."""
        source = InMemoryRawSource(
            make_frames(1), clock=clock, semantics=SourceSemantics.ARCHIVAL, fail_on_open=5
        )
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings([], clock=clock, dimensions=dimensions, source=source),
            collector,
        )
        for _ in range(30):
            await asyncio.sleep(0)

        state = health.observability(CAMERA)
        assert not state.observing
        assert state.reason is ObservabilityReason.STREAM_DISCONNECTED
        gaps = health.coverage_gaps(CAMERA)
        assert any(not g.closed for g in gaps), "an open coverage gap must be recorded"
        await sources.close_all()


class TestActorIsolation:
    async def test_a_failing_camera_never_affects_another(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Fault containment as a structural consequence of the actor model."""
        healthy_camera = make_camera(CameraId("cam-healthy"))
        broken_camera = make_camera(CameraId("cam-broken"))

        healthy = FrameCollector()
        broken = FrameCollector()

        sources.open(
            healthy_camera,
            make_bindings(make_frames(4), clock=clock, dimensions=dimensions),
            healthy,
        )
        sources.open(
            broken_camera,
            make_bindings(
                make_frames(4), clock=clock, dimensions=dimensions, privacy=FailingMask()
            ),
            broken,
        )

        await healthy.wait(4)
        assert len(healthy.frames) == 4
        assert broken.frames == []
        await sources.close_all()

    async def test_each_camera_has_its_own_sequence_space(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        first = FrameCollector()
        second = FrameCollector()
        sources.open(
            make_camera(CameraId("cam-a")),
            make_bindings(make_frames(3), clock=clock, dimensions=dimensions),
            first,
        )
        sources.open(
            make_camera(CameraId("cam-b")),
            make_bindings(make_frames(3), clock=clock, dimensions=dimensions),
            second,
        )
        await first.wait(3)
        await second.wait(3)

        assert [f.frame_ref.frame_seq for f in first.frames] == [0, 1, 2]
        assert [f.frame_ref.frame_seq for f in second.frames] == [0, 1, 2]
        await sources.close_all()


class TestClockSyncAdapters:
    async def test_wallclock_hint_raises_clock_quality(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frames = make_frames(2)
        for spec in frames:
            spec.wallclock_hint = Instant(5_000_000_000)

        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                frames,
                clock=clock,
                dimensions=dimensions,
                clock_sync=WallclockHintClockSync(),
            ),
            collector,
        )
        await _drain(sources, collector, 2)
        assert collector.frames[0].time.clock_quality is ClockQuality.RTCP_DERIVED
        assert collector.frames[0].time.t_capture == Instant(5_000_000_000)
        await sources.close_all()

    async def test_missing_hint_falls_back_honestly(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Falls back rather than silently reusing a stale offset."""
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(1),
                clock=clock,
                dimensions=dimensions,
                clock_sync=WallclockHintClockSync(),
            ),
            collector,
        )
        await _drain(sources, collector, 1)
        assert collector.frames[0].time.clock_quality is ClockQuality.ESTIMATED
        await sources.close_all()

    async def test_unknown_quality_is_marked_unfusable(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(1),
                clock=clock,
                dimensions=dimensions,
                clock_sync=UnknownClockSync(),
            ),
            collector,
        )
        await _drain(sources, collector, 1)
        assert not collector.frames[0].time.clock_quality.fusable
        await sources.close_all()

    async def test_pts_clock_sync_is_deterministic(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Deterministic capture time is what makes replay reproducible (V13)."""
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(
                make_frames(3), clock=clock, dimensions=dimensions, clock_sync=PtsClockSync()
            ),
            collector,
        )
        await _drain(sources, collector, 3)
        captures = [f.time.t_capture.ns for f in collector.frames]
        assert captures == [0, 40_000_000, 80_000_000]
        await sources.close_all()


class TestDeterministicReplay:
    async def test_identical_input_yields_identical_output(
        self,
        clock: VirtualClock,
        bus: EventBus,
        metrics: MetricsEngine,
        health_config,
        buffer_config,
        source_config,
        dimensions: FrameDimensions,
    ) -> None:
        """Invariant V13 — the basis of every regression test above L3."""

        async def run() -> list[tuple[int, int, bytes]]:
            local_clock = VirtualClock()
            local_bus = EventBus(local_clock)
            local_metrics = MetricsEngine(local_clock)
            local_health = HealthMonitor(
                clock=local_clock,
                bus=local_bus,
                metrics=local_metrics,
                config=health_config,
            )
            from vision_os.adapters.memory import HostMemoryPool

            local_buffer = FrameBuffer(
                clock=local_clock,
                bus=local_bus,
                metrics=local_metrics,
                allocator=HostMemoryPool(slots=64, bytes_per_slot=FRAME_BYTES),
                config=buffer_config,
            )
            local_sources = VideoSourceManager(
                clock=local_clock,
                bus=local_bus,
                metrics=local_metrics,
                health=local_health,
                buffer=local_buffer,
                config=source_config,
            )
            collector = FrameCollector()
            local_sources.open(
                make_camera(),
                make_bindings(
                    make_frames(5),
                    clock=local_clock,
                    dimensions=dimensions,
                    clock_sync=PtsClockSync(),
                ),
                collector,
            )
            await collector.wait(5)
            await local_sources.close_all()
            return [
                (f.frame_ref.frame_seq, f.time.t_capture.ns, bytes(f.pixels.readonly_view()))
                for f in collector.frames
            ]

        assert await run() == await run()


class TestSourceStatus:
    async def test_status_is_reported_per_camera(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        collector = FrameCollector()
        sources.open(
            make_camera(),
            make_bindings(make_frames(2), clock=clock, dimensions=dimensions),
            collector,
        )
        await _drain(sources, collector, 2)

        status = sources.status(CAMERA)
        assert status.stats.frames_published == 2
        assert status.state in (ActorState.STREAMING, ActorState.STOPPED)
        assert sources.open_count == 1
        await sources.close_all()

    async def test_unknown_camera_status_is_typed(
        self, sources: VideoSourceManager
    ) -> None:
        from vision_os.core.errors import NotFoundError

        with pytest.raises(NotFoundError):
            sources.status(CameraId("ghost"))

    async def test_close_all_drains_every_actor(
        self, sources: VideoSourceManager, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        for index in range(3):
            sources.open(
                make_camera(CameraId(f"cam-{index}")),
                make_bindings(make_frames(2), clock=clock, dimensions=dimensions),
                FrameCollector(),
            )
        assert sources.open_count == 3
        await sources.close_all()
        assert sources.open_count == 0


class TestSchedulerIntegration:
    async def test_admitted_frames_flow_from_source_to_scheduler(
        self,
        sources: VideoSourceManager,
        scheduler: FrameScheduler,
        clock: VirtualClock,
        dimensions: FrameDimensions,
    ) -> None:
        """The complete Flow 1 path: source -> decode -> mask -> buffer -> admission."""
        camera = make_camera(target_fps=1000.0)
        scheduler.register_camera(
            camera.camera_id, camera.pipeline_profile, camera.source_semantics
        )
        admitted: list[bool] = []

        async def sink(frame: Frame) -> None:
            verdict = scheduler.offer(camera.camera_id)
            admitted.append(verdict.admit)
            scheduler.complete(camera.camera_id)

        sources.open(camera, make_bindings(make_frames(5), clock=clock, dimensions=dimensions), sink)
        for _ in range(60):
            await asyncio.sleep(0)
            if len(admitted) >= 5:
                break

        assert len(admitted) == 5
        assert any(admitted)
        await sources.close_all()
