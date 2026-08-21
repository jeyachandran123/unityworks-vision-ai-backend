"""Shared fixtures for the UnityWorks Vision OS Flow 1 suite.

Every fixture here builds real modules wired to dependency-free reference
adapters. Nothing is mocked at a module boundary: the platform's own ports are
the seam, so tests exercise production code paths rather than stand-ins.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence

import pytest

from vision_os.acquisition import (
    CameraManager,
    FrameBuffer,
    FrameScheduler,
    SourceBindings,
    VideoSourceManager,
)
from vision_os.adapters.acquisition import (
    ArrivalTimeClockSync,
    InMemoryRawSource,
    NoMaskPolicy,
    PassthroughDecoder,
    RawFrameSpec,
)
from vision_os.adapters.configuration import InMemoryConfigSource, InMemorySecretProvider
from vision_os.adapters.memory import HostMemoryPool
from vision_os.adapters.observability import (
    InMemoryMetricsExporter,
    RecordingEventTransport,
)
from vision_os.adapters.scheduling import CadenceAdmissionPolicy
from vision_os.conformance import flow1_registry
from vision_os.core.model.camera import (
    Camera,
    NativeProfile,
    PipelineProfile,
    SourceSemantics,
    SourceSpec,
)
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import CameraId, ProfileId, SiteId, TenantId
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config import ConfigLayer, ConfigurationManager
from vision_os.kernel.config.schema import (
    BufferSection,
    HealthSection,
    SchedulerSection,
    SourceSection,
)
from vision_os.kernel.events import EventBus
from vision_os.kernel.health import HealthMonitor
from vision_os.kernel.metrics import MetricsEngine
from vision_os.kernel.plugins import PluginManager

WIDTH = 8
HEIGHT = 4
CHANNELS = 3
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS

TENANT = TenantId("acme")
SITE = SiteId("site-sg-01")
CAMERA = CameraId("cam-01")


#: Timing budgets are meaningless under a trace function.
#:
#: Coverage instrumentation inflates per-call cost by an order of magnitude, so a
#: latency assertion measured under it tests the profiler, not the platform.
#: Growth and boundedness assertions stay enabled — those are still valid.
UNDER_TRACING = sys.gettrace() is not None

skip_if_traced = pytest.mark.skipif(
    UNDER_TRACING,
    reason="timing budget is not measurable under coverage instrumentation",
)


def pytest_configure(config: pytest.Config) -> None:
    """Silence one deprecation caused by the Atlas root conftest, not by this suite.

    ``tests/conftest.py`` defines a session-scoped ``event_loop`` for the Atlas
    application's database fixtures. Vision OS shadows it (below) because it has
    no session-scoped async state, and pytest-asyncio warns on any redefinition.
    """
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The event_loop fixture provided by pytest-asyncio has been "
        "redefined:DeprecationWarning",
    )


@pytest.fixture
def event_loop():
    """Function-scoped loop, shadowing the Atlas root conftest's session-scoped one.

    Vision OS has no database, no FastAPI app, and no session-scoped async state;
    inheriting a session loop couples this suite to another platform's fixtures
    for no benefit and breaks introspection under pytest-asyncio 0.24.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def clock() -> VirtualClock:
    """A virtual clock. Deterministic by construction (invariant V13)."""
    return VirtualClock()


@pytest.fixture
def transport() -> RecordingEventTransport:
    return RecordingEventTransport()


@pytest.fixture
def bus(clock: VirtualClock, transport: RecordingEventTransport) -> EventBus:
    return EventBus(clock, transport=transport)


@pytest.fixture
def metrics_exporter() -> InMemoryMetricsExporter:
    return InMemoryMetricsExporter()


@pytest.fixture
def metrics(clock: VirtualClock, metrics_exporter: InMemoryMetricsExporter) -> MetricsEngine:
    return MetricsEngine(clock, exporter=metrics_exporter)


@pytest.fixture
def health_config() -> HealthSection:
    return HealthSection(
        report_timeout_ms=1_000,
        aggregation_interval_ms=100,
        frozen_frame_threshold=3,
        hysteresis_samples=1,
    )


@pytest.fixture
def health(
    clock: VirtualClock, bus: EventBus, metrics: MetricsEngine, health_config: HealthSection
) -> HealthMonitor:
    return HealthMonitor(clock=clock, bus=bus, metrics=metrics, config=health_config)


@pytest.fixture
def dimensions() -> FrameDimensions:
    return FrameDimensions(width=WIDTH, height=HEIGHT, colour_space="bgr24")


@pytest.fixture
def pool() -> HostMemoryPool:
    """Frames here are 96 bytes, so a generous pool costs nothing.

    Sized so multi-camera tests exercise concurrency rather than incidentally
    testing pool exhaustion; tests that mean to exercise exhaustion build their
    own small pool.
    """
    return HostMemoryPool(slots=512, bytes_per_slot=FRAME_BYTES)


@pytest.fixture
def buffer_config() -> BufferSection:
    """A ring deep enough that flow tests are not incidentally backpressure tests.

    Buffer capacity is a function of pipeline depth, not camera count. Tests that
    mean to exercise backpressure construct their own narrow ring.
    """
    return BufferSection(
        slots_per_camera=16,
        bytes_per_slot=FRAME_BYTES,
        lease_deadline_ms=1_000,
        history_window_ms=5_000,
    )


@pytest.fixture
def narrow_buffer_config() -> BufferSection:
    """A deliberately shallow ring, for eviction and backpressure tests."""
    return BufferSection(
        slots_per_camera=3,
        bytes_per_slot=FRAME_BYTES,
        lease_deadline_ms=1_000,
        history_window_ms=5_000,
    )


@pytest.fixture
def buffer(
    clock: VirtualClock,
    bus: EventBus,
    metrics: MetricsEngine,
    pool: HostMemoryPool,
    buffer_config: BufferSection,
) -> FrameBuffer:
    return FrameBuffer(
        clock=clock, bus=bus, metrics=metrics, allocator=pool, config=buffer_config
    )


@pytest.fixture
def scheduler_config() -> SchedulerSection:
    return SchedulerSection(
        global_budget_fps=1000.0,
        sustained_drop_threshold=0.5,
        drop_alarm_window_ms=1_000,
        duplicate_suppression=False,
    )


@pytest.fixture
def scheduler(
    clock: VirtualClock,
    bus: EventBus,
    metrics: MetricsEngine,
    health: HealthMonitor,
    scheduler_config: SchedulerSection,
) -> FrameScheduler:
    return FrameScheduler(
        clock=clock,
        bus=bus,
        metrics=metrics,
        health=health,
        policy=CadenceAdmissionPolicy(),
        config=scheduler_config,
    )


@pytest.fixture
def source_config() -> SourceSection:
    return SourceSection(
        reconnect_backoff_initial_ms=10,
        reconnect_backoff_max_ms=100,
        reconnect_backoff_jitter=0.0,
        stall_watchdog_ms=1_000,
        max_consecutive_decode_errors=3,
    )


@pytest.fixture
def sources(
    clock: VirtualClock,
    bus: EventBus,
    metrics: MetricsEngine,
    health: HealthMonitor,
    buffer: FrameBuffer,
    source_config: SourceSection,
) -> VideoSourceManager:
    return VideoSourceManager(
        clock=clock,
        bus=bus,
        metrics=metrics,
        health=health,
        buffer=buffer,
        config=source_config,
    )


@pytest.fixture
def camera_manager(clock: VirtualClock, bus: EventBus) -> CameraManager:
    return CameraManager(clock=clock, bus=bus)


@pytest.fixture
def plugins(clock: VirtualClock, bus: EventBus, metrics: MetricsEngine) -> PluginManager:
    return PluginManager(
        clock=clock, bus=bus, metrics=metrics, conformance=flow1_registry()
    )


def make_camera(
    camera_id: CameraId = CAMERA,
    *,
    semantics: SourceSemantics = SourceSemantics.ARCHIVAL,
    target_fps: float = 5.0,
    max_in_flight: int = 4,
) -> Camera:
    return Camera(
        camera_id=camera_id,
        tenant_id=TENANT,
        site_id=SITE,
        source_spec=SourceSpec(uri=f"mem://{camera_id}", transport="memory"),
        source_semantics=semantics,
        native_profile=NativeProfile(width=WIDTH, height=HEIGHT, fps=25.0, codec="raw_bgr24"),
        pipeline_profile=PipelineProfile(
            profile_id=ProfileId("standard"),
            target_fps=target_fps,
            max_in_flight=max_in_flight,
            inference_width=640,
            inference_height=640,
        ),
    )


@pytest.fixture
def camera() -> Camera:
    return make_camera()


def make_frames(count: int, *, distinct: bool = True) -> list[RawFrameSpec]:
    """Build a scripted raw-frame sequence.

    ``distinct=False`` produces identical payloads, which is how frozen-camera
    and duplicate-suppression behaviour is exercised.
    """
    frames = []
    for index in range(count):
        fill = (index % 250) + 1 if distinct else 7
        frames.append(
            RawFrameSpec(
                payload=bytes([fill]) * FRAME_BYTES,
                width=WIDTH,
                height=HEIGHT,
                pts=index * 40,
            )
        )
    return frames


def make_bindings(
    frames: Sequence[RawFrameSpec],
    *,
    clock: VirtualClock,
    dimensions: FrameDimensions,
    semantics: SourceSemantics = SourceSemantics.ARCHIVAL,
    privacy=None,
    decoder=None,
    source=None,
    clock_sync=None,
) -> SourceBindings:
    return SourceBindings(
        source=source
        or InMemoryRawSource(frames, clock=clock, semantics=semantics),
        decoder=decoder or PassthroughDecoder(dimensions=dimensions),
        privacy=privacy or NoMaskPolicy(),
        clock_sync=clock_sync or ArrivalTimeClockSync(),
    )


def base_config_document(cameras: int = 1, *, target_fps: float = 5.0) -> dict:
    """A minimal valid configuration document for the closed schema."""
    return {
        "platform": {"deployment_profile": "embedded", "clock_mode": "virtual"},
        "buffer": {
            "slots_per_camera": 3,
            "bytes_per_slot": FRAME_BYTES,
            "lease_deadline_ms": 1000,
            "history_window_ms": 5000,
        },
        "scheduler": {"global_budget_fps": 1000.0, "drop_alarm_window_ms": 1000},
        "source": {
            "reconnect_backoff_initial_ms": 10,
            "reconnect_backoff_max_ms": 100,
            "stall_watchdog_ms": 1000,
        },
        "health": {"aggregation_interval_ms": 100, "report_timeout_ms": 1000},
        "runtime": {"attach_stagger_ms": 0, "drain_timeout_ms": 1000},
        "profiles": [
            {
                "profile_id": "standard",
                "target_fps": target_fps,
                "max_in_flight": 4,
                "inference_width": 640,
                "inference_height": 640,
            }
        ],
        "regions": [
            {
                "region_id": "Z3",
                "label": "Z3",
                "vertices": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "frame_of_reference": "normalized",
            }
        ],
        "cameras": [
            {
                "camera_id": f"cam-{i:02d}",
                "tenant_id": str(TENANT),
                "site_id": str(SITE),
                "uri": f"mem://cam-{i:02d}",
                "transport": "memory",
                "source_semantics": "archival",
                "profile_id": "standard",
                "width": WIDTH,
                "height": HEIGHT,
                "fps": 25.0,
                "codec": "raw_bgr24",
                "region_ids": ["Z3"],
            }
            for i in range(1, cameras + 1)
        ],
    }


@pytest.fixture
def config(clock: VirtualClock) -> ConfigurationManager:
    manager = ConfigurationManager(
        clock=clock,
        sources={ConfigLayer.SITE: InMemoryConfigSource(base_config_document())},
        secrets=InMemorySecretProvider({"cam-secret": "hunter2"}),
    )
    manager.load()
    return manager
