"""The live runtime: sessions, their lifecycle, and who may see them.

### Nothing starts by itself

Importing this module opens no socket. Importing the application opens no socket.
Opening DevTools opens no socket. A camera session starts when
`LiveRuntime.start_configured()` is called from the application lifespan **and**
`FEATURE_LIVE_CCTV` is on **and** a camera is configured.

That is three deliberate acts. A backend that dials a DVR because somebody ran
the test suite is a backend that will one day dial a customer's DVR from a
developer's laptop.

### Scoping

Every read is filtered by tenant, then by the caller's camera scope. A user who
is not granted a camera receives nothing about it — not its state, not its URI,
not the fact that it exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.configuration.settings import Settings
from app.errors import ConfigurationInvalidError, NotFoundError
from app.vision.frames import LiveFrame
from app.vision.secrets import EnvironmentSecretProvider, SecretProvider
from app.vision.session import SessionSpec, VisionSession, session_for
from app.vision.sources.base import FrameSource, SourceKind
from app.vision.sources.replay import ReplayFrameSource, SyntheticFrameSource
from app.vision.sources.rtsp import LiveRtspSource, ReconnectPolicy, RtspCameraConfig


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    """What the runtime is doing. Safe for any authorised caller."""

    enabled: bool
    reason: str
    active_sessions: int
    streaming_sessions: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "active_sessions": self.active_sessions,
            "streaming_sessions": self.streaming_sessions,
            # The system-wide answer to "is anything actually live". Derived from
            # sessions, never set by hand.
            "streaming": self.streaming_sessions > 0,
        }


class LiveRuntime:
    """Owns every session for the process."""

    def __init__(
        self,
        settings: Settings,
        *,
        secrets: SecretProvider | None = None,
        on_frame=None,
    ) -> None:
        self._settings = settings
        self._secrets = secrets or EnvironmentSecretProvider()
        self._sessions: dict[str, VisionSession] = {}
        self._lock = asyncio.Lock()
        #: Set by the composition root so a frame reaches Vision OS. Absent, the
        #: runtime still proves source, session and backpressure behaviour —
        #: which is what the development path exercises today.
        self._on_frame = on_frame

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def sessions(self) -> tuple[VisionSession, ...]:
        return tuple(self._sessions.values())

    def summary(self) -> RuntimeSummary:
        active = [s for s in self._sessions.values() if s.state.is_active]
        return RuntimeSummary(
            enabled=self._settings.feature_live_cctv,
            reason=(
                ""
                if self._settings.feature_live_cctv
                else "FEATURE_LIVE_CCTV is off; no camera session will start"
            ),
            active_sessions=len(active),
            streaming_sessions=sum(1 for s in active if s.streaming),
        )

    def visible(self, *, tenant_id: str, camera_ids: tuple[str, ...] | None) -> list[VisionSession]:
        """Sessions this caller may see.

        `camera_ids is None` means the caller holds a tenant-wide grant. An
        **empty tuple means none** — the same three-state discipline the
        authorization model uses, and for the same reason: an empty list must
        never be read as a wildcard.
        """
        allowed = []
        for session in self._sessions.values():
            if session.spec.tenant_id != tenant_id:
                continue
            if camera_ids is not None and session.camera_id not in camera_ids:
                continue
            allowed.append(session)
        return allowed

    def get(self, session_id: str, *, tenant_id: str) -> VisionSession:
        session = self._sessions.get(session_id)
        # Tenant mismatch is reported as "not found", not "forbidden": telling a
        # caller that a session exists in another tenant is itself a disclosure.
        if session is None or session.spec.tenant_id != tenant_id:
            raise NotFoundError("no such session")
        return session

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start_configured(self) -> int:
        """Start the sessions configuration asks for. Returns how many started.

        Called from the application lifespan and nowhere else.
        """
        if not self._settings.feature_live_cctv:
            logger.info("live CCTV disabled (FEATURE_LIVE_CCTV=false); no sessions started")
            return 0

        cameras = self._configured_cameras()
        if not cameras:
            # Not an error. A deployment with no camera configured is a valid
            # deployment, and it says so rather than failing to boot.
            logger.info("live CCTV enabled but no camera is configured")
            return 0

        started = 0
        for config in cameras:
            try:
                await self.start_live(config)
                started += 1
            except Exception as exc:  # noqa: BLE001 - one camera, not the process
                logger.error(
                    "camera {} failed to start: {}: {}",
                    config.camera_id,
                    type(exc).__name__,
                    exc,
                )
        return started

    async def start_from_records(self, configs: Sequence[RtspCameraConfig]) -> int:
        """Start the cameras the **database** says are enabled.

        The durable replacement for `start_configured()`. The rule is unchanged
        and the source of truth moved: a camera row that is not `enabled` opens
        no socket. What changed is that the decision now survives a restart and
        is auditable, instead of living in an environment variable nobody can
        show you the history of.

        One camera failing to start does not stop the others — sixteen kitchens
        must not go dark because one DVR channel is unplugged.
        """
        if not self._settings.feature_live_cctv:
            logger.info(
                "live CCTV disabled (FEATURE_LIVE_CCTV=false); {} enabled camera " "row(s) ignored",
                len(configs),
            )
            return 0

        started = 0
        for config in configs:
            try:
                await self.start_live(config)
                started += 1
            except Exception as exc:  # noqa: BLE001 - one camera, not the process
                logger.error(
                    "camera {} failed to start: {}: {}",
                    config.camera_id,
                    type(exc).__name__,
                    exc,
                )
        return started

    async def start_live(self, config: RtspCameraConfig) -> VisionSession:
        """Start one live camera."""
        source = LiveRtspSource(
            config,
            secrets=self._secrets,
            reconnect=ReconnectPolicy(
                initial_ms=self._settings.cctv_reconnect_initial_ms,
                max_ms=self._settings.cctv_reconnect_max_ms,
                max_attempts=self._settings.cctv_reconnect_max_attempts,
            ),
        )
        return await self._start(
            source,
            SessionSpec(
                camera_id=config.camera_id,
                tenant_id=self._settings.default_tenant_id,
                queue_capacity=self._settings.cctv_queue_capacity,
                analysis_fps=config.analysis_fps,
            ),
        )

    async def start_replay(
        self,
        *,
        camera_id: str,
        path: str,
        tenant_id: str,
        loop: bool = False,
        analysis_fps: float | None = None,
    ) -> VisionSession:
        """Start a replay session. **Labelled REPLAY everywhere it appears.**"""
        source = ReplayFrameSource(camera_id=camera_id, path=path, loop=loop)
        return await self._start(
            source,
            SessionSpec(
                camera_id=camera_id,
                tenant_id=tenant_id,
                queue_capacity=self._settings.cctv_queue_capacity,
                analysis_fps=analysis_fps or self._settings.cctv_analysis_fps,
            ),
        )

    async def start_synthetic(
        self,
        *,
        camera_id: str,
        tenant_id: str,
        fps: float = 25.0,
        count: int | None = None,
        analysis_fps: float | None = None,
        queue_capacity: int | None = None,
        interval_override_s: float | None = None,
    ) -> VisionSession:
        """A generated continuous source. Tests and development only, and REPLAY."""
        source = SyntheticFrameSource(
            camera_id=camera_id,
            fps=fps,
            count=count,
            interval_override_s=interval_override_s,
        )
        return await self._start(
            source,
            SessionSpec(
                camera_id=camera_id,
                tenant_id=tenant_id,
                queue_capacity=queue_capacity or self._settings.cctv_queue_capacity,
                analysis_fps=analysis_fps or self._settings.cctv_analysis_fps,
            ),
        )

    async def _start(self, source: FrameSource, spec: SessionSpec) -> VisionSession:
        async with self._lock:
            # One session per camera. A second would open a second connection to
            # the same DVR channel and double the cost for the same pictures.
            for existing in self._sessions.values():
                if existing.camera_id == spec.camera_id and existing.state.is_active:
                    raise ConfigurationInvalidError(
                        f"camera '{spec.camera_id}' already has an active session",
                        details={"session_id": existing.session_id},
                    )

            handler = self._handler_for(spec)
            session = session_for(source, spec, handler=handler)
            self._sessions[session.session_id] = session

        await session.start()
        return session

    def _handler_for(self, spec: SessionSpec):
        if self._on_frame is None:
            return None

        on_frame = self._on_frame

        async def _handle(frame: LiveFrame) -> None:
            await on_frame(spec, frame)

        return _handle

    async def stop(self, session_id: str, *, tenant_id: str) -> None:
        session = self.get(session_id, tenant_id=tenant_id)
        await session.stop()
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def stop_all(self) -> None:
        """Shutdown. No orphan task, queue or source survives this."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        # Concurrently: a dozen cameras each waiting out a reconnect backoff
        # would otherwise make shutdown take minutes.
        await asyncio.gather(*(s.stop() for s in sessions), return_exceptions=True)
        logger.info("live runtime stopped {} session(s)", len(sessions))

    # ── configuration ────────────────────────────────────────────────────────

    def _configured_cameras(self) -> list[RtspCameraConfig]:
        """Cameras named by configuration.

        **An empty `CCTV_CHANNELS` selects nothing.** The DVR has 16 channels and
        must not become 16 pipelines because nobody said otherwise; cost follows
        configuration, not hardware.
        """
        settings = self._settings
        if not settings.cctv_host or not settings.cctv_channels.strip():
            return []

        cameras = []
        for raw in settings.cctv_channels.split(","):
            token = raw.strip()
            if not token:
                continue
            try:
                channel = int(token)
            except ValueError as exc:
                # Raised, not skipped. A silently dropped channel is a kitchen
                # nobody is watching.
                raise ConfigurationInvalidError(
                    f"CCTV_CHANNELS contains '{token}', which is not a number"
                ) from exc

            cameras.append(
                RtspCameraConfig(
                    camera_id=f"cam-{channel:02d}",
                    host=settings.cctv_host,
                    port=settings.cctv_rtsp_port,
                    channel=channel,
                    stream_type=settings.cctv_stream_type,
                    username=settings.cctv_username,
                    credential_ref=settings.cctv_credential_ref,
                    analysis_fps=settings.cctv_analysis_fps,
                )
            )
        return cameras

    def describe_cameras(self) -> list[dict[str, Any]]:
        """Configured cameras, with **redacted** URIs. Never a credential."""
        return [
            {
                "camera_id": config.camera_id,
                "uri": config.redacted_uri(),
                "channel": config.channel,
                "stream_type": config.stream_type,
                "analysis_fps": config.analysis_fps,
                "credential_configured": bool(config.credential_ref),
            }
            for config in self._configured_cameras()
        ]


__all__ = ["LiveRuntime", "RuntimeSummary", "SourceKind"]
