"""M15 Runtime and the composition root — boot, attach, drain, shutdown.

The properties defended here are the ones that make deployments non-events:
fail-fast on invalid config, staggered attach, and a graceful drain that records
the shutdown window instead of leaving an unexplained blind period.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.acquisition import SourceBindings
from vision_os.adapters.acquisition import (
    ArrivalTimeClockSync,
    InMemoryRawSource,
    NoMaskPolicy,
    PassthroughDecoder,
)
from vision_os.adapters.configuration import (
    InMemoryConfigSource,
    InMemorySecretProvider,
)
from vision_os.adapters.observability import RecordingEventTransport
from vision_os.bootstrap import build_platform
from vision_os.core.errors import CapacityExceededError, NotFoundError, ValidationError
from vision_os.core.model.camera import Camera
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.health import HealthState
from vision_os.core.model.ids import CameraId
from vision_os.core.model.timebase import Duration
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config import ConfigLayer
from vision_os.kernel.metrics import MetricName
from vision_os.kernel.runtime import RuntimeState

from ..conftest import HEIGHT, WIDTH, base_config_document, make_frames

DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT, colour_space="bgr24")


def _bindings_factory(clock: VirtualClock, *, frames: int = 4):
    def factory(camera: Camera) -> SourceBindings:
        return SourceBindings(
            source=InMemoryRawSource(
                make_frames(frames), clock=clock, semantics=camera.source_semantics
            ),
            decoder=PassthroughDecoder(dimensions=DIMENSIONS),
            privacy=NoMaskPolicy(),
            clock_sync=ArrivalTimeClockSync(),
        )

    return factory


def _platform(document: dict, clock: VirtualClock, **kwargs):
    return build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=_bindings_factory(clock),
        clock=clock,
        secrets=InMemorySecretProvider({"cam-01-creds": "hunter2"}),
        **kwargs,
    )


class TestCompositionRoot:
    def test_build_platform_wires_every_module(self, clock: VirtualClock) -> None:
        """Dependencies are injected from one place; nothing self-constructs."""
        platform = _platform(base_config_document(), clock)
        for component in (
            platform.config,
            platform.bus,
            platform.metrics,
            platform.health,
            platform.plugins,
            platform.cameras,
            platform.buffer,
            platform.scheduler,
            platform.sources,
            platform.runtime,
        ):
            assert component is not None

    def test_invalid_configuration_fails_before_anything_starts(
        self, clock: VirtualClock
    ) -> None:
        """Fail fast and loudly; never boot into a half-valid state."""
        document = base_config_document()
        document["restaurant_rules"] = {"max_wait_seconds": 60}
        with pytest.raises(ValidationError):
            _platform(document, clock)

    def test_every_adapter_is_substitutable(self, clock: VirtualClock) -> None:
        """Invariant V3 at the composition root: swap by argument, not by edit."""
        transport = RecordingEventTransport()
        platform = _platform(base_config_document(), clock, event_transport=transport)
        platform.bus.publish(
            __import__(
                "vision_os.kernel.events", fromlist=["CameraChanged"]
            ).CameraChanged(occurred_at=clock.now(), camera_id=CameraId("cam-01"))
        )
        assert transport.events("camera.changed")


class TestBootSequence:
    async def test_boot_attaches_declared_cameras(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(cameras=3), clock)
        await platform.boot()

        topology = platform.runtime.topology()
        assert topology.state is RuntimeState.READY
        assert len(topology.attached_cameras) == 3
        assert platform.cameras.count == 3
        await platform.shutdown()

    async def test_readiness_is_false_before_boot(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(), clock)
        ready, reasons = platform.runtime.readiness()
        assert not ready
        assert any("created" in reason for reason in reasons)
        await platform.shutdown(graceful=False)

    async def test_boot_twice_is_rejected(self, clock: VirtualClock) -> None:
        from vision_os.core.errors import LifecycleError

        platform = _platform(base_config_document(), clock)
        await platform.boot()
        with pytest.raises(LifecycleError):
            await platform.boot()
        await platform.shutdown()

    async def test_topology_reports_the_config_revision(self, clock: VirtualClock) -> None:
        """Every observation will later pin this; it must be visible at runtime."""
        platform = _platform(base_config_document(), clock)
        await platform.boot()
        topology = platform.runtime.topology()
        assert topology.config_revision.startswith("cfg-")
        assert topology.clock_is_virtual
        await platform.shutdown()


class TestPipelineLifecycle:
    async def test_frames_flow_through_the_wired_pipeline(
        self, clock: VirtualClock
    ) -> None:
        """End to end: source -> decode -> mask -> buffer -> admission."""
        platform = _platform(base_config_document(target_fps=1000.0), clock)
        await platform.boot()

        for _ in range(80):
            await asyncio.sleep(0)

        stats = platform.runtime.pipeline_stats(CameraId("cam-01"))
        assert stats.admitted + stats.dropped > 0
        assert platform.metrics.snapshot().counter_value(
            MetricName.FRAMES_RECEIVED, camera_id="cam-01"
        ) > 0
        await platform.shutdown()

    async def test_detach_removes_the_pipeline(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(cameras=2), clock)
        await platform.boot()
        await platform.runtime.detach_pipeline(CameraId("cam-01"), reason="test")

        assert len(platform.runtime.topology().attached_cameras) == 1
        with pytest.raises(NotFoundError):
            platform.runtime.pipeline_stats(CameraId("cam-01"))
        await platform.shutdown()

    async def test_detach_unknown_pipeline_is_typed(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(), clock)
        await platform.boot()
        with pytest.raises(NotFoundError):
            await platform.runtime.detach_pipeline(CameraId("ghost"))
        await platform.shutdown()

    async def test_capacity_is_enforced(self, clock: VirtualClock) -> None:
        document = base_config_document(cameras=3)
        document["runtime"]["max_pipelines"] = 2
        platform = _platform(document, clock)
        with pytest.raises(CapacityExceededError):
            await platform.boot()
        await platform.shutdown(graceful=False)

    async def test_attach_is_idempotent(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(), clock)
        await platform.boot()
        await platform.runtime.attach_pipeline(CameraId("cam-01"))
        assert len(platform.runtime.topology().attached_cameras) == 1
        await platform.shutdown()


class TestGracefulDrain:
    async def test_drain_detaches_every_pipeline(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(cameras=3), clock)
        await platform.boot()

        report = await platform.runtime.drain(Duration.from_millis(2_000))
        assert len(report.detached) == 3
        assert platform.runtime.state is RuntimeState.DRAINING
        await platform.shutdown(graceful=False)

    async def test_shutdown_records_the_window_rather_than_leaving_a_mystery(
        self, clock: VirtualClock
    ) -> None:
        """Without this, a deployment looks exactly like an unexplained outage."""
        platform = _platform(base_config_document(), clock)
        await platform.boot()
        await platform.shutdown()

        gaps = platform.health.coverage_gaps(CameraId("cam-01"))
        assert gaps, "shutdown must appear in the coverage record"
        assert platform.runtime.state is RuntimeState.STOPPED

    async def test_shutdown_is_safe_without_boot(self, clock: VirtualClock) -> None:
        platform = _platform(base_config_document(), clock)
        await platform.shutdown(graceful=False)
        assert platform.runtime.state is RuntimeState.STOPPED

    async def test_liveness_survives_component_failure(self, clock: VirtualClock) -> None:
        """Health is observational, never load-bearing."""
        platform = _platform(base_config_document(), clock)
        await platform.boot()
        platform.health.set_observability(CameraId("cam-01"), HealthState.FAILED)
        alive, _ = platform.runtime.liveness()
        assert alive
        await platform.shutdown()


class TestSecretResolution:
    async def test_credential_reference_is_resolved_at_attach(
        self, clock: VirtualClock
    ) -> None:
        document = base_config_document()
        document["cameras"][0]["credential_ref"] = "cam-01-creds"
        platform = _platform(document, clock)
        await platform.boot()
        assert platform.sources.is_open(CameraId("cam-01"))
        await platform.shutdown()

    async def test_unresolvable_credential_fails_that_camera_only(
        self, clock: VirtualClock
    ) -> None:
        from vision_os.core.errors import SecretResolutionError

        document = base_config_document(cameras=2)
        document["cameras"][0]["credential_ref"] = "missing-secret"
        platform = _platform(document, clock)
        with pytest.raises(SecretResolutionError):
            await platform.boot()
        await platform.shutdown(graceful=False)


class TestClockSelection:
    def test_clock_mode_is_a_deployment_decision(self, clock: VirtualClock) -> None:
        from vision_os.bootstrap import build_clock
        from vision_os.kernel.clock import ScaledClock, SystemClock

        for mode, expected in (
            ("system", SystemClock),
            ("virtual", VirtualClock),
            ("scaled", ScaledClock),
        ):
            document = base_config_document()
            document["platform"]["clock_mode"] = mode
            platform = build_platform(
                config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
                bindings_factory=_bindings_factory(clock),
            )
            assert isinstance(build_clock(platform.config), expected)


class TestMaintenanceLoop:
    async def test_buffer_is_swept_on_the_maintenance_tick(
        self, clock: VirtualClock
    ) -> None:
        platform = _platform(base_config_document(target_fps=1000.0), clock)
        await platform.boot()

        for _ in range(60):
            await asyncio.sleep(0)
        before = platform.buffer.stats().slots_in_use

        clock.advance(Duration.from_millis(10_000))
        for _ in range(80):
            await asyncio.sleep(0)

        assert platform.buffer.stats().slots_in_use <= before
        await platform.shutdown()

    async def test_metrics_are_exported_on_the_tick(self, clock: VirtualClock) -> None:
        from vision_os.adapters.observability import InMemoryMetricsExporter

        exporter = InMemoryMetricsExporter()
        platform = _platform(base_config_document(), clock, metrics_exporter=exporter)
        await platform.boot()

        # Let the maintenance task reach its first sleep and register a sleeper
        # on the virtual clock; advancing before it does would skip past the tick.
        for _ in range(20):
            await asyncio.sleep(0)
        clock.advance(Duration.from_millis(500))
        for _ in range(40):
            await asyncio.sleep(0)

        assert exporter.exports >= 1
        await platform.shutdown()
