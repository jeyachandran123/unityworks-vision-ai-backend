"""M19 Event Bus, M21 Metrics Engine, M20 Health Monitor, and the clocks.

The properties defended here are the ones whose absence is silent: gap markers on
overflow, cardinality bounds, "silence is never health", and the injected clock
that makes everything else reproducible.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.observability import (
    FailingEventTransport,
    InMemoryMetricsExporter,
    OpenMetricsTextExporter,
    RecordingEventTransport,
)
from vision_os.core.model.health import (
    ComponentHealth,
    HealthState,
    ObservabilityReason,
)
from vision_os.core.model.ids import CameraId, ModuleId
from vision_os.core.model.timebase import Duration, Instant
from vision_os.kernel.clock import ScaledClock, SystemClock, VirtualClock
from vision_os.kernel.config.schema import HealthSection
from vision_os.kernel.events import (
    CameraChanged,
    DeliveryPolicy,
    EventBus,
    Gap,
    OverflowPolicy,
    StreamLost,
)
from vision_os.kernel.health import HealthMonitor
from vision_os.kernel.metrics import MetricName, MetricsEngine

# --- clocks ---------------------------------------------------------------- #


class TestClocks:
    def test_virtual_clock_advances_only_by_control(self) -> None:
        """The prerequisite for invariant V13."""
        clock = VirtualClock()
        start = clock.now()
        assert clock.now() == start
        clock.advance(Duration.from_millis(500))
        assert clock.now().since(start).millis == 500

    def test_virtual_clock_refuses_to_move_backwards(self) -> None:
        clock = VirtualClock()
        with pytest.raises(ValueError, match="backwards"):
            clock.advance(Duration.from_millis(-1))

    async def test_virtual_clock_wakes_sleepers_in_deadline_order(self) -> None:
        clock = VirtualClock()
        woken: list[str] = []

        async def sleeper(name: str, ms: int) -> None:
            await clock.sleep(Duration.from_millis(ms))
            woken.append(name)

        import asyncio

        tasks = [
            asyncio.create_task(sleeper("late", 300)),
            asyncio.create_task(sleeper("early", 100)),
            asyncio.create_task(sleeper("mid", 200)),
        ]
        await asyncio.sleep(0)
        clock.advance(Duration.from_millis(500))
        await asyncio.gather(*tasks)
        assert woken == ["early", "mid", "late"]

    def test_system_clock_monotonic_is_independent_of_wall_clock(self) -> None:
        clock = SystemClock()
        assert clock.monotonic().ns > 0
        assert not clock.is_virtual

    def test_scaled_clock_compresses_time(self) -> None:
        clock = ScaledClock(factor=1000.0)
        assert not clock.is_virtual
        assert clock.monotonic().ns >= 0

    def test_scaled_clock_rejects_non_positive_factor(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ScaledClock(factor=0.0)


# --- event bus -------------------------------------------------------------- #


class TestEventBus:
    def test_delivers_to_matching_subscribers_only(self, bus: EventBus) -> None:
        stream = bus.subscribe([StreamLost])
        camera = bus.subscribe([CameraChanged])

        bus.publish(
            StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01"), reason="test")
        )
        assert len(stream.drain()) == 1
        assert len(camera.drain()) == 0

    def test_subscribe_to_all_receives_everything(self, bus: EventBus) -> None:
        every = bus.subscribe()
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        bus.publish(CameraChanged(occurred_at=Instant(2), camera_id=CameraId("cam-01")))
        assert len(every.drain()) == 2

    def test_unregistered_event_type_is_rejected(self, bus: EventBus) -> None:
        """The bus stays typed rather than becoming an untyped message soup."""
        with pytest.raises(ValueError, match="unregistered event types"):
            bus.subscribe(["totally.invented.event"])

    def test_overflow_emits_an_explicit_gap_never_silence(self, bus: EventBus) -> None:
        """Invariant V8 applied to delivery: a subscriber is never silently skipped."""
        subscription = bus.subscribe(
            [StreamLost], policy=DeliveryPolicy(capacity=2, overflow=OverflowPolicy.DROP_OLDEST)
        )
        for i in range(5):
            bus.publish(
                StreamLost(occurred_at=Instant(i), camera_id=CameraId("cam-01"), reason=str(i))
            )

        delivered = subscription.drain()
        gaps = [e for e in delivered if isinstance(e, Gap)]
        assert gaps, "overflow must produce an explicit Gap marker"
        assert gaps[0].dropped >= 1
        assert gaps[0].reason == "subscriber_overflow"

    def test_conflate_keeps_latest_per_partition(self, bus: EventBus) -> None:
        subscription = bus.subscribe(
            [StreamLost],
            policy=DeliveryPolicy(capacity=4, overflow=OverflowPolicy.CONFLATE),
        )
        for i in range(10):
            bus.publish(
                StreamLost(
                    occurred_at=Instant(i),
                    partition_key="cam-01",
                    camera_id=CameraId("cam-01"),
                    reason=str(i),
                )
            )
        events = [e for e in subscription.drain() if isinstance(e, StreamLost)]
        assert len(events) == 1
        assert events[0].reason == "9"

    def test_predicate_filters_content(self, bus: EventBus) -> None:
        subscription = bus.subscribe(
            [StreamLost], predicate=lambda e: getattr(e, "camera_id", "") == "cam-02"
        )
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        bus.publish(StreamLost(occurred_at=Instant(2), camera_id=CameraId("cam-02")))
        assert len(subscription.drain()) == 1

    def test_a_raising_predicate_does_not_break_the_bus(self, bus: EventBus) -> None:
        def bad(_event) -> bool:
            raise RuntimeError("bad filter")

        bus.subscribe([StreamLost], predicate=bad)
        healthy = bus.subscribe([StreamLost])
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        assert len(healthy.drain()) == 1

    def test_transport_failure_never_reaches_the_publisher(
        self, clock: VirtualClock
    ) -> None:
        """A transport outage degrades remote visibility, never local perception."""
        bus = EventBus(clock, transport=FailingEventTransport())
        subscription = bus.subscribe([StreamLost])
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        assert len(subscription.drain()) == 1
        assert bus.stats().transport_failures == 1

    def test_events_are_forwarded_to_the_transport(
        self, bus: EventBus, transport: RecordingEventTransport
    ) -> None:
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        assert transport.events("stream.lost")

    def test_unsubscribe_stops_delivery(self, bus: EventBus) -> None:
        subscription = bus.subscribe([StreamLost])
        bus.unsubscribe(subscription)
        bus.publish(StreamLost(occurred_at=Instant(1), camera_id=CameraId("cam-01")))
        assert subscription.drain() == []


# --- metrics ----------------------------------------------------------------- #


class TestMetricsEngine:
    def test_counters_gauges_and_histograms(self, metrics: MetricsEngine) -> None:
        metrics.counter("uwv.test.count", camera_id="cam-01").increment(3)
        metrics.gauge("uwv.test.level", camera_id="cam-01").set(0.75)
        metrics.histogram("uwv.test.latency", camera_id="cam-01").record(12.5)

        snapshot = metrics.snapshot()
        assert snapshot.counter_value("uwv.test.count", camera_id="cam-01") == 3
        assert snapshot.gauge_value("uwv.test.level", camera_id="cam-01") == 0.75
        assert snapshot.histogram_values("uwv.test.latency", camera_id="cam-01") == (12.5,)

    def test_cardinality_is_bounded_and_offenders_collapse(
        self, clock: VirtualClock
    ) -> None:
        """Unbounded labels take down the metrics backend, then the platform."""
        engine = MetricsEngine(clock, max_label_cardinality=10)
        for i in range(200):
            engine.counter("uwv.test.perobject", object_id=f"obj-{i}").increment()

        assert engine.cardinality_violations > 0
        snapshot = engine.snapshot()
        collapsed = snapshot.counters.get(("uwv.test.perobject", (("other", "other"),)))
        assert collapsed is not None and collapsed > 1

    def test_histogram_window_is_bounded(self, clock: VirtualClock) -> None:
        """Unbounded sample retention is a slow memory leak."""
        engine = MetricsEngine(clock, histogram_window=50)
        for i in range(500):
            engine.histogram("uwv.test.h").record(float(i))
        assert len(engine.snapshot().histogram_values("uwv.test.h")) == 50

    def test_timer_records_elapsed_time(self, clock: VirtualClock) -> None:
        engine = MetricsEngine(clock)
        with engine.timer("uwv.test.timed"):
            clock.advance(Duration.from_millis(25))
        samples = engine.snapshot().histogram_values("uwv.test.timed")
        assert samples and samples[0] == pytest.approx(25.0, abs=0.5)

    def test_export_failure_is_absorbed(self, clock: VirtualClock) -> None:
        class Boom:
            exporter_id = "boom"

            def export(self, snapshot) -> None:
                raise RuntimeError("exporter down")

        engine = MetricsEngine(clock, exporter=Boom())
        engine.counter("uwv.test.x").increment()
        engine.export()  # must not raise
        assert engine.snapshot().counter_value("vision_os.metrics.export_failures") == 1

    def test_exporter_receives_snapshot(
        self, metrics: MetricsEngine, metrics_exporter: InMemoryMetricsExporter
    ) -> None:
        metrics.counter(MetricName.FRAMES_RECEIVED, camera_id="cam-01").increment()
        metrics.export()
        assert metrics_exporter.exports == 1
        assert metrics_exporter.last is not None

    def test_openmetrics_rendering(self, clock: VirtualClock) -> None:
        exporter = OpenMetricsTextExporter()
        engine = MetricsEngine(clock, exporter=exporter)
        engine.counter("uwv.frames.received", camera_id="cam-01").increment(5)
        engine.export()
        assert 'uwv_frames_received{camera_id="cam-01"} 5' in exporter.text()


# --- health ------------------------------------------------------------------- #


class TestHealthMonitor:
    def test_silence_is_never_health(self, health: HealthMonitor, clock: VirtualClock) -> None:
        """The single most important default in the module."""
        component = ModuleId("detector")
        health.report(
            ComponentHealth(
                component_id=component, state=HealthState.HEALTHY, reported_at=clock.now()
            )
        )
        assert health.component_health(component).state is HealthState.HEALTHY

        clock.advance(Duration.from_millis(5_000))
        stale = health.component_health(component)
        assert stale.state is HealthState.FAILED
        assert "no report" in stale.detail

    def test_never_reported_component_is_failed(self, health: HealthMonitor) -> None:
        assert health.component_health(ModuleId("ghost")).state is HealthState.FAILED

    def test_blind_is_distinct_from_healthy(self, health: HealthMonitor) -> None:
        """A streaming camera pointed at a parked truck is healthy and useless."""
        camera = CameraId("cam-01")
        health.register_camera(camera)
        health.set_observability(
            camera, HealthState.BLIND, ObservabilityReason.SCENE_OBSCURED, effective_rate=0.0
        )
        state = health.observability(camera)
        assert state.status is HealthState.BLIND
        assert not state.observing
        assert state.reason is ObservabilityReason.SCENE_OBSCURED

    def test_coverage_gap_opens_and_closes(self, health: HealthMonitor) -> None:
        camera = CameraId("cam-01")
        health.register_camera(camera)
        health.set_observability(
            camera, HealthState.BLIND, ObservabilityReason.STREAM_DISCONNECTED
        )
        open_gaps = [g for g in health.coverage_gaps(camera) if not g.closed]
        assert len(open_gaps) == 1

        health.set_observability(camera, HealthState.HEALTHY, ObservabilityReason.NORMAL)
        closed = [g for g in health.coverage_gaps(camera) if g.closed]
        assert len(closed) == 1
        assert closed[0].reason is ObservabilityReason.STREAM_DISCONNECTED

    def test_observability_change_publishes_an_event(
        self, health: HealthMonitor, bus: EventBus
    ) -> None:
        subscription = bus.subscribe(["health.coverage_changed"])
        camera = CameraId("cam-01")
        health.register_camera(camera)
        health.set_observability(
            camera, HealthState.DEGRADED, ObservabilityReason.SCHEDULER_SHEDDING
        )
        assert subscription.drain()

    def test_frozen_frames_raise_suspicion_not_a_verdict(
        self, health: HealthMonitor, bus: EventBus
    ) -> None:
        """A false positive that blinds a working camera is itself an outage."""
        subscription = bus.subscribe(["health.silent_failure_suspected"])
        camera = CameraId("cam-01")
        health.register_camera(camera)
        health.set_observability(camera, HealthState.HEALTHY)

        for _ in range(5):
            health.observe_frame_digest(camera, 0xDEADBEEF)

        assert subscription.drain(), "frozen frames must raise a suspicion"
        assert health.observability(camera).status is HealthState.HEALTHY, (
            "suspicion must not automatically blind the camera"
        )

    def test_changing_content_does_not_raise_suspicion(
        self, health: HealthMonitor, bus: EventBus
    ) -> None:
        subscription = bus.subscribe(["health.silent_failure_suspected"])
        camera = CameraId("cam-01")
        health.register_camera(camera)
        for digest in range(10):
            health.observe_frame_digest(camera, digest)
        assert not subscription.drain()

    def test_site_health_aggregates(self, health: HealthMonitor) -> None:
        for index, state in enumerate(
            [HealthState.HEALTHY, HealthState.DEGRADED, HealthState.BLIND]
        ):
            camera = CameraId(f"cam-{index}")
            health.register_camera(camera)
            health.set_observability(camera, state)

        site = health.site_health()
        assert site.total_cameras == 3
        assert site.observing == 2  # healthy + degraded
        assert site.blind == 1
        assert site.observable_fraction == pytest.approx(2 / 3)

    def test_liveness_stays_true_even_with_failed_components(
        self, health: HealthMonitor, clock: VirtualClock
    ) -> None:
        """Health is observational, never load-bearing."""
        health.report(
            ComponentHealth(
                component_id=ModuleId("x"), state=HealthState.FAILED, reported_at=clock.now()
            )
        )
        alive, impaired = health.liveness()
        assert alive
        assert "x" in impaired

    def test_hysteresis_suppresses_flapping(
        self, clock: VirtualClock, bus: EventBus, metrics: MetricsEngine
    ) -> None:
        monitor = HealthMonitor(
            clock=clock,
            bus=bus,
            metrics=metrics,
            config=HealthSection(hysteresis_samples=3, report_timeout_ms=60_000),
        )
        component = ModuleId("flappy")
        monitor.report(
            ComponentHealth(component, HealthState.HEALTHY, clock.now())
        )
        monitor.report(ComponentHealth(component, HealthState.DEGRADED, clock.now()))
        assert monitor.component_health(component).state is HealthState.HEALTHY

        monitor.report(ComponentHealth(component, HealthState.DEGRADED, clock.now()))
        monitor.report(ComponentHealth(component, HealthState.DEGRADED, clock.now()))
        assert monitor.component_health(component).state is HealthState.DEGRADED
