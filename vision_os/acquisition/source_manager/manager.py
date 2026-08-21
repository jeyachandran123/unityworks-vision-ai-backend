"""M2 Video Source Manager — produce trustworthy frames; interpret nothing.

Turns a source specification into a reliable, correctly-identified, correctly-
timestamped, privacy-masked stream of decoded frames — and keeps it that way
across the network failures that define real deployments.

Actor isolation is the load-bearing property: one actor per source, each owning
its own socket, decoder session, and frame numbering, so **a failing camera never
affects another camera**.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...core.errors import NotFoundError
from ...core.model.camera import Camera
from ...core.model.ids import CameraId
from ...core.model.timebase import Duration
from ...core.ports.acquisition import ClockSyncPort, DecoderPort, PrivacyMaskPort, SourcePort
from ...core.ports.clock import Clock
from ...kernel.config.schema import SourceSection
from ...kernel.events import EventBus
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricsEngine
from ..buffer import FrameBuffer
from .actor import ActorState, FrameSink, SourceActor, StreamStats
from .epoch import EpochAllocator, EpochStore, InMemoryEpochStore


@dataclass(frozen=True, slots=True)
class SourceBindings:
    """The four adapters a source actor needs, injected together.

    Grouping them makes it explicit that a camera's acquisition behaviour is
    fully determined by adapter selection — the concrete mechanism behind
    "RTSP today, WebRTC tomorrow, drone streams later" (invariant V3).
    """

    source: SourcePort
    decoder: DecoderPort
    privacy: PrivacyMaskPort
    clock_sync: ClockSyncPort


@dataclass(frozen=True, slots=True)
class SourceStatus:
    camera_id: CameraId
    state: ActorState
    stats: StreamStats


class VideoSourceManager:
    """Supervises one actor per open source."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        buffer: FrameBuffer,
        config: SourceSection,
        epoch_store: EpochStore | None = None,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._health = health
        self._buffer = buffer
        self._config = config
        self._epochs = EpochAllocator(epoch_store or InMemoryEpochStore())
        self._actors: dict[CameraId, SourceActor] = {}

    # --- lifecycle ---------------------------------------------------------- #

    def open(
        self,
        camera: Camera,
        bindings: SourceBindings,
        on_frame: FrameSink,
        *,
        credential: str | None = None,
    ) -> SourceActor:
        """Start an actor for this camera. Idempotent per camera."""
        existing = self._actors.get(camera.camera_id)
        if existing is not None:
            return existing

        self._buffer.register_camera(camera.camera_id)
        self._health.register_camera(camera.camera_id)

        actor = SourceActor(
            camera=camera,
            clock=self._clock,
            bus=self._bus,
            metrics=self._metrics,
            health=self._health,
            buffer=self._buffer,
            source=bindings.source,
            decoder=bindings.decoder,
            privacy=bindings.privacy,
            clock_sync=bindings.clock_sync,
            epochs=self._epochs,
            config=self._config,
            credential=credential,
            on_frame=on_frame,
        )
        self._actors[camera.camera_id] = actor
        actor.start()
        return actor

    async def close(self, camera_id: CameraId, timeout: Duration | None = None) -> None:
        actor = self._actors.pop(camera_id, None)
        if actor is None:
            return
        await actor.stop(timeout)
        self._buffer.forget_camera(camera_id)

    async def close_all(self, timeout: Duration | None = None) -> None:
        """Drain every actor concurrently; shutdown must be bounded."""
        camera_ids = list(self._actors)
        await asyncio.gather(
            *(self.close(camera_id, timeout) for camera_id in camera_ids),
            return_exceptions=True,
        )

    # --- introspection -------------------------------------------------------- #

    def status(self, camera_id: CameraId) -> SourceStatus:
        actor = self._actors.get(camera_id)
        if actor is None:
            raise NotFoundError(
                f"no open source for camera '{camera_id}'", camera_id=str(camera_id)
            )
        return SourceStatus(camera_id=camera_id, state=actor.state, stats=actor.stats)

    def statuses(self) -> tuple[SourceStatus, ...]:
        return tuple(
            SourceStatus(camera_id=cid, state=actor.state, stats=actor.stats)
            for cid, actor in self._actors.items()
        )

    def is_open(self, camera_id: CameraId) -> bool:
        return camera_id in self._actors

    @property
    def open_count(self) -> int:
        return len(self._actors)
