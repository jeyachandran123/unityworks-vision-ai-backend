"""M15 Runtime — make the platform exist and keep it running. Perform no perception.

The **composition root**: the one place the dependency graph is constructed and
injected. No other module constructs a dependency (01_LAYERED §8.1), which is
what makes every module testable in isolation and replaceable in place.

Boot order follows 08_RUNTIME §7.1. Three steps matter more than they look:

* Conformance runs **before a single frame is processed**, so a mis-built adapter
  is rejected at boot rather than in production data.
* The kernel is ready before pipelines attach, so health and metrics observe the
  attach sequence itself.
* Attach is **staggered**: a hundred cameras connecting at once is a
  self-inflicted thundering herd that can make boot itself fail.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field

from ...core.errors import CapacityExceededError, LifecycleError, NotFoundError
from ...core.model.camera import Camera
from ...core.model.frame import Frame
from ...core.model.health import ComponentHealth, HealthState, ObservabilityReason
from ...core.model.ids import CameraId, ModuleId
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...core.ports.pipeline import AdmittedFrameConsumer
from ...core.ports.scheduling import AdmissionVerdict, Fidelity
from ..config import ConfigurationManager
from ..events import EventBus, PipelineAttached, PipelineDetached
from ..health import HealthMonitor
from ..metrics import MetricName, MetricsEngine
from ..plugins import PluginManager

_RUNTIME_ID = ModuleId("runtime")

#: Used when an admission policy admits without naming a fidelity. Full
#: resolution is the honest default: a consumer must never silently receive
#: degraded input believing it was primary.
_DEFAULT_FIDELITY = Fidelity(inference_width=640, inference_height=640)


class RuntimeState(enum.Enum):
    CREATED = "created"
    BOOTING = "booting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeTopology:
    state: RuntimeState
    attached_cameras: tuple[CameraId, ...]
    max_pipelines: int
    config_revision: str
    clock_is_virtual: bool


@dataclass(slots=True)
class DrainReport:
    detached: tuple[CameraId, ...] = ()
    frames_in_flight_at_start: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False


@dataclass(slots=True)
class PipelineStats:
    """A camera's logical flow. Not a thread — an identity plus its state.

    Flow 1 wires acquisition only: source actor → scheduler admission. Later
    flows extend the admitted-frame path; nothing here anticipates them.
    """

    camera: Camera
    admitted: int = 0
    dropped: int = 0
    restarts: int = 0
    last_verdict: AdmissionVerdict | None = None
    recent_drop_reasons: dict[str, int] = field(default_factory=dict)


class VisionRuntime:
    """Owns process lifecycle, pipeline placement, and the composition root.

    Collaborators are injected rather than constructed here, so the runtime can
    be exercised in tests with fakes and so a deployment can substitute an
    adapter without the runtime knowing (invariant V3).
    """

    def __init__(
        self,
        *,
        clock: Clock,
        config: ConfigurationManager,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        plugins: PluginManager,
        camera_manager,  # CameraManager
        buffer,  # FrameBuffer
        scheduler,  # FrameScheduler
        sources,  # VideoSourceManager
        bindings_factory,  # Callable[[Camera], SourceBindings]
        admitted_frame_consumer: AdmittedFrameConsumer | None = None,
    ) -> None:
        self._clock = clock
        self._config = config
        self._bus = bus
        self._metrics = metrics
        self._health = health
        self._plugins = plugins
        self._cameras = camera_manager
        self._buffer = buffer
        self._scheduler = scheduler
        self._sources = sources
        self._bindings_factory = bindings_factory
        # The single documented extension point at which a later flow resumes the
        # admitted-frame path. ``None`` is the Flow 1 behaviour: an admitted frame
        # is counted and released. The runtime holds a protocol and never learns
        # what implements it.
        self._admitted_consumer = admitted_frame_consumer

        self._state = RuntimeState.CREATED
        self._pipelines: dict[CameraId, PipelineStats] = {}
        self._maintenance: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    # --- lifecycle ----------------------------------------------------------- #

    async def boot(self) -> None:
        """Construct and start the platform. Fails fast and loudly."""
        if self._state is not RuntimeState.CREATED:
            raise LifecycleError(f"cannot boot from state {self._state.value}")
        self._state = RuntimeState.BOOTING
        self._report_health(HealthState.STARTING, "booting")

        effective = self._config.effective()
        self._cameras.load_declarations(
            cameras=effective.cameras,
            profiles=effective.profiles,
            regions=effective.regions,
        )

        self._maintenance = asyncio.create_task(self._maintenance_loop(), name="uwv-maintenance")

        for camera in self._cameras.list():
            await self.attach_pipeline(camera.camera_id)
            stagger = effective.runtime.attach_stagger_ms
            if stagger > 0:
                await self._clock.sleep(Duration.from_millis(stagger))

        self._state = RuntimeState.READY
        self._report_health(HealthState.HEALTHY, "ready")

    async def attach_pipeline(self, camera_id: CameraId) -> None:
        """Start a camera's logical flow."""
        if camera_id in self._pipelines:
            return
        runtime_config = self._config.runtime()
        if len(self._pipelines) >= runtime_config.max_pipelines:
            raise CapacityExceededError(
                f"cannot attach '{camera_id}': at capacity "
                f"({runtime_config.max_pipelines} pipelines)",
                camera_id=str(camera_id),
            )

        camera = self._cameras.get(camera_id)
        pipeline = PipelineStats(camera=camera)
        self._pipelines[camera_id] = pipeline

        self._scheduler.register_camera(
            camera_id, camera.pipeline_profile, camera.source_semantics
        )
        credential = self._config.resolve_secret(camera.source_spec.credential_ref)
        self._sources.open(
            camera,
            self._bindings_factory(camera),
            self._make_sink(camera_id),
            credential=credential,
        )

        self._metrics.counter(MetricName.PIPELINES_ATTACHED).increment()
        self._bus.publish(
            PipelineAttached(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
            )
        )

    async def detach_pipeline(
        self, camera_id: CameraId, *, reason: str = "requested", timeout: Duration | None = None
    ) -> None:
        pipeline = self._pipelines.pop(camera_id, None)
        if pipeline is None:
            raise NotFoundError(f"no attached pipeline for '{camera_id}'")
        await self._sources.close(camera_id, timeout)
        self._scheduler.forget_camera(camera_id)
        self._health.forget_camera(camera_id)
        self._bus.publish(
            PipelineDetached(
                occurred_at=self._clock.now(),
                partition_key=str(camera_id),
                camera_id=camera_id,
                reason=reason,
            )
        )

    def _make_sink(self, camera_id: CameraId):
        """Offer each published frame for admission, then continue the pipeline.

        With no consumer attached this is exactly the Flow 1 behaviour: an
        admitted frame is counted and released. With one attached, the frame
        *reference* is handed on and the consumer takes its own lease — so the
        payload stays control-plane sized and an eviction between admission and
        consumption degrades cleanly (invariant V12).

        ``complete`` runs after the consumer returns, which is what makes
        ``max_in_flight`` a real bound on work in the pipeline rather than a
        bound on frames the scheduler has merely looked at.
        """

        async def sink(frame: Frame) -> None:
            pipeline = self._pipelines.get(camera_id)
            if pipeline is None:
                return
            verdict = self._scheduler.offer(
                camera_id,
                view=frame.pixels.readonly_view(),
                dimensions=frame.dimensions,
            )
            pipeline.last_verdict = verdict
            if not verdict.admit:
                pipeline.dropped += 1
                reason = verdict.reason.value if verdict.reason else "unattributed"
                pipeline.recent_drop_reasons[reason] = (
                    pipeline.recent_drop_reasons.get(reason, 0) + 1
                )
                return

            pipeline.admitted += 1
            self._health.observe_frame_digest(camera_id, _digest(frame))
            try:
                if self._admitted_consumer is not None:
                    await self._admitted_consumer.on_admitted(
                        frame.frame_ref,
                        verdict.fidelity or _DEFAULT_FIDELITY,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a later flow never stops acquisition
                # The consumer contract says it must not raise. If one does, the
                # source actor still survives: acquisition is the platform's
                # floor and no downstream stage may take it down (invariant V9).
                self._metrics.counter(
                    MetricName.PIPELINE_CONSUMER_FAILURES, camera_id=str(camera_id)
                ).increment()
            finally:
                self._scheduler.complete(camera_id)

        return sink

    # --- maintenance ---------------------------------------------------------- #

    async def _maintenance_loop(self) -> None:
        """Buffer sweep, metric export, health aggregation.

        Runs on the injected clock, so a virtual clock makes this deterministic
        and a scaled clock compresses a 30-day soak into hours.
        """
        interval = Duration.from_millis(self._config.health().aggregation_interval_ms)
        while not self._stopping.is_set():
            try:
                await self._clock.sleep(interval)
                self._buffer.sweep()
                self._metrics.export()
                self._report_health(
                    HealthState.HEALTHY if self._state is RuntimeState.READY
                    else HealthState.STARTING,
                    self._state.value,
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001, S112 - maintenance never kills the platform
                continue

    # --- shutdown -------------------------------------------------------------- #

    async def drain(self, timeout: Duration | None = None) -> DrainReport:
        """Stop admission, finish in flight, and record the shutdown window.

        Emitting an explicit shutdown record matters: without it a deployment
        looks in the record exactly like an unexplained blind period.
        """
        self._state = RuntimeState.DRAINING
        self._report_health(HealthState.DRAINING, "draining")
        started = self._clock.monotonic()
        budget = timeout or Duration.from_millis(self._config.runtime().drain_timeout_ms)

        camera_ids = tuple(self._pipelines)
        for camera_id in camera_ids:
            self._health.set_observability(
                camera_id,
                HealthState.DRAINING,
                ObservabilityReason.DRAINING,
                effective_rate=0.0,
                detail="planned shutdown",
            )

        await self._sources.close_all(budget)
        for camera_id in camera_ids:
            self._pipelines.pop(camera_id, None)
            self._scheduler.forget_camera(camera_id)

        elapsed = (self._clock.monotonic().ns - started.ns) / 1_000_000
        return DrainReport(
            detached=camera_ids,
            duration_ms=elapsed,
            timed_out=elapsed > budget.millis,
        )

    async def shutdown(self, *, graceful: bool = True) -> DrainReport:
        """Stop the platform.

        An immediate shutdown skips the drain but still stops every actor: a
        process that exits with source tasks still running leaks work that
        outlives the runtime that owns it.
        """
        if graceful:
            report = await self.drain()
        else:
            report = DrainReport(detached=tuple(self._pipelines))
            await self._sources.close_all(Duration.from_millis(0))
            self._pipelines.clear()

        self._stopping.set()
        if self._maintenance is not None:
            self._maintenance.cancel()
            try:
                await self._maintenance
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                pass
            self._maintenance = None
        self._buffer.close()
        self._bus.close()
        self._state = RuntimeState.STOPPED
        return report

    # --- probes ----------------------------------------------------------------- #

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        if self._state is not RuntimeState.READY:
            return (False, (f"runtime:{self._state.value}",))
        return self._health.readiness()

    def liveness(self) -> tuple[bool, tuple[str, ...]]:
        return self._health.liveness()

    def topology(self) -> RuntimeTopology:
        return RuntimeTopology(
            state=self._state,
            attached_cameras=tuple(self._pipelines),
            max_pipelines=self._config.runtime().max_pipelines,
            config_revision=str(self._config.revision()),
            clock_is_virtual=self._clock.is_virtual,
        )

    @property
    def state(self) -> RuntimeState:
        return self._state

    def pipeline_stats(self, camera_id: CameraId) -> PipelineStats:
        pipeline = self._pipelines.get(camera_id)
        if pipeline is None:
            raise NotFoundError(f"no attached pipeline for '{camera_id}'")
        return pipeline

    def _report_health(self, state: HealthState, detail: str) -> None:
        self._health.report(
            ComponentHealth(
                component_id=_RUNTIME_ID,
                state=state,
                reported_at=self._clock.now(),
                detail=detail,
            )
        )


def _digest(frame: Frame) -> int:
    """A cheap content digest for frozen-frame detection.

    Samples rather than hashing the whole frame: at 3000 frames a second a full
    hash would cost more than the detection is worth.
    """
    view = frame.pixels.readonly_view()
    length = len(view)
    if length == 0:
        return 0
    step = max(1, length // 64)
    return hash(bytes(view[::step]))
