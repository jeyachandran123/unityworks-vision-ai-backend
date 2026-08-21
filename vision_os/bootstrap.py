"""Composition root helper - assemble a Flow 1 platform from configuration.

This is the *only* place in the codebase where concrete adapters are chosen and
wired to ports. Every module receives its collaborators through its constructor
(01_LAYERED section 8.1); nothing reaches out to find a dependency, and no module knows
which adapter it was given.

The three consequences are the point:

1. Every module is testable in isolation with fakes, without a GPU, camera, or
   network.
2. Every module is replaceable in place, because nothing downstream knows what
   it was given.
3. The clock is injectable, which is the single prerequisite for deterministic
   replay (invariant V13).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .acquisition import (
    CameraManager,
    EpochStore,
    FrameBuffer,
    FrameScheduler,
    SourceBindings,
    VideoSourceManager,
)
from .adapters.memory import HostMemoryPool
from .adapters.observability import NullEventTransport
from .adapters.scheduling import CadenceAdmissionPolicy, NullChangeDetector
from .conformance import ConformanceRegistry, platform_registry
from .core.model.camera import Camera
from .core.ports.acquisition import ClockSyncPort, DecoderPort, PrivacyMaskPort, SourcePort
from .core.ports.clock import Clock
from .core.ports.configuration import ConfigSourcePort, SecretProviderPort
from .core.ports.observability import EventTransportPort, MetricsExportPort
from .core.ports.pipeline import AdmittedFrameConsumer
from .core.ports.scheduling import AdmissionPolicyPort, ChangeDetectorPort
from .kernel.clock import ScaledClock, SystemClock, VirtualClock
from .kernel.config import ConfigLayer, ConfigurationManager
from .kernel.config.schema import ClockMode
from .kernel.events import EventBus
from .kernel.health import HealthMonitor
from .kernel.metrics import MetricsEngine
from .kernel.plugins import PluginManager, SignatureVerifier
from .kernel.runtime import VisionRuntime

BindingsFactory = Callable[[Camera], SourceBindings]


@dataclass(slots=True)
class VisionPlatform:
    """Every constructed collaborator, exposed for operation and test.

    Returned rather than hidden so that a test can drive any single module
    directly - the practical expression of "every module is independently
    testable".
    """

    clock: Clock
    config: ConfigurationManager
    bus: EventBus
    metrics: MetricsEngine
    health: HealthMonitor
    plugins: PluginManager
    conformance: ConformanceRegistry
    cameras: CameraManager
    buffer: FrameBuffer
    scheduler: FrameScheduler
    sources: VideoSourceManager
    runtime: VisionRuntime

    async def boot(self) -> None:
        await self.runtime.boot()

    async def shutdown(self, *, graceful: bool = True) -> Any:
        return await self.runtime.shutdown(graceful=graceful)


def build_clock(config: ConfigurationManager) -> Clock:
    """Select the clock declared by configuration.

    Deterministic mode requires a virtual clock; ``ClockMode`` makes that an
    explicit deployment decision rather than an implicit test-only behaviour.
    """
    platform = config.platform()
    if platform.clock_mode is ClockMode.VIRTUAL:
        return VirtualClock()
    if platform.clock_mode is ClockMode.SCALED:
        return ScaledClock(factor=platform.clock_scale_factor)
    return SystemClock()


def build_platform(
    *,
    config_sources: dict[ConfigLayer, ConfigSourcePort],
    bindings_factory: BindingsFactory,
    clock: Clock | None = None,
    secrets: SecretProviderPort | None = None,
    allocator: Any | None = None,
    admission_policy: AdmissionPolicyPort | None = None,
    change_detector: ChangeDetectorPort | None = None,
    event_transport: EventTransportPort | None = None,
    metrics_exporter: MetricsExportPort | None = None,
    conformance: ConformanceRegistry | None = None,
    require_signatures: bool = False,
    defaults: dict[str, Any] | None = None,
    admitted_frame_consumer: AdmittedFrameConsumer | None = None,
) -> VisionPlatform:
    """Construct a Flow 1 platform.

    Every adapter is an argument with a dependency-free default, so a caller may
    substitute any one of them without touching the platform (invariant V3).

    ``admitted_frame_consumer`` is the documented extension point at which a
    later flow resumes the admitted-frame path. Omitted, the platform behaves
    exactly as Flow 1 did: an admitted frame is counted and released.
    """
    configuration = ConfigurationManager(
        clock=SystemClock(),
        sources=config_sources,
        secrets=secrets,
        defaults=defaults,
    )
    configuration.load()

    resolved_clock = clock or build_clock(configuration)
    # Rebuild configuration against the selected clock so that override expiry
    # and revision history run on the same timeline as everything else.
    configuration = ConfigurationManager(
        clock=resolved_clock,
        sources=config_sources,
        secrets=secrets,
        defaults=defaults,
    )
    configuration.load()

    effective = configuration.effective()

    bus = EventBus(resolved_clock, transport=event_transport or NullEventTransport())
    metrics = MetricsEngine(
        resolved_clock,
        exporter=metrics_exporter,
        max_label_cardinality=effective.metrics.max_label_cardinality,
        histogram_window=effective.metrics.histogram_window,
    )
    health = HealthMonitor(
        clock=resolved_clock, bus=bus, metrics=metrics, config=effective.health
    )
    plugins = PluginManager(
        clock=resolved_clock,
        bus=bus,
        metrics=metrics,
        conformance=conformance or platform_registry(),
        verifier=SignatureVerifier(),
        require_signatures=require_signatures,
    )

    pool = allocator or HostMemoryPool(
        slots=max(
            1,
            int(
                effective.buffer.slots_per_camera
                * max(1, len(effective.cameras))
                * effective.buffer.jitter_factor
            ),
        ),
        bytes_per_slot=effective.buffer.bytes_per_slot,
    )

    cameras = CameraManager(clock=resolved_clock, bus=bus)
    buffer = FrameBuffer(
        clock=resolved_clock,
        bus=bus,
        metrics=metrics,
        allocator=pool,
        config=effective.buffer,
    )
    scheduler = FrameScheduler(
        clock=resolved_clock,
        bus=bus,
        metrics=metrics,
        health=health,
        policy=admission_policy or CadenceAdmissionPolicy(),
        config=effective.scheduler,
        change_detector=change_detector or NullChangeDetector(),
    )
    sources = VideoSourceManager(
        clock=resolved_clock,
        bus=bus,
        metrics=metrics,
        health=health,
        buffer=buffer,
        config=effective.source,
    )
    runtime = VisionRuntime(
        clock=resolved_clock,
        config=configuration,
        bus=bus,
        metrics=metrics,
        health=health,
        plugins=plugins,
        camera_manager=cameras,
        buffer=buffer,
        scheduler=scheduler,
        sources=sources,
        bindings_factory=bindings_factory,
        admitted_frame_consumer=admitted_frame_consumer,
    )

    return VisionPlatform(
        clock=resolved_clock,
        config=configuration,
        bus=bus,
        metrics=metrics,
        health=health,
        plugins=plugins,
        conformance=conformance or platform_registry(),
        cameras=cameras,
        buffer=buffer,
        scheduler=scheduler,
        sources=sources,
        runtime=runtime,
    )


__all__ = [
    "BindingsFactory",
    "ClockSyncPort",
    "DecoderPort",
    "EpochStore",
    "PrivacyMaskPort",
    "SourcePort",
    "VisionPlatform",
    "build_clock",
    "build_platform",
]
