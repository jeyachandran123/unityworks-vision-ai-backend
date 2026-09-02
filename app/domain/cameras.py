"""Camera configuration, from the database.

Replaces `CCTV_CHANNELS`. The rule it enforces is unchanged and now durable: the
DVR has 16 channels, and **only rows that are `enabled` create a session**. A
disabled camera opens no socket, decodes nothing and reaches no model. Cost
follows configuration, not hardware.

### The credential is still a reference

`Camera.credential_ref` holds `env:CCTV_PASSWORD`, never a password. Moving
camera config into the database changed where the *pointer* lives; it did not
change what the row is allowed to contain. A database dump must not be a
credential dump.

### Frame metadata

`FrameService` records that a frame existed and what it contributed to — not the
frame. §12: this is traceability, not a video recorder. The crops that were
deliberately retained live in `evidence_records`; everything else is gone by
design, which is a smaller privacy surface and the whole point.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Camera, CameraZoneAssignment, FrameRecord
from app.domain.zone_attribution import record_assignment
from app.errors import ConfigurationInvalidError, ConflictError, NotFoundError, ValidationError

if TYPE_CHECKING:
    from app.vision.sources.rtsp import RtspCameraConfig

VALID_STREAM_TYPES = {"main", "sub"}


class CameraService:
    """Camera configuration. Tenant-scoped at every entry point."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        organization_id: str,
        restaurant_id: str,
        camera_key: str,
        name: str,
        channel: int,
        host: str = "",
        rtsp_port: int = 554,
        stream_type: str = "sub",
        username: str = "",
        credential_ref: str = "",
        analysis_fps: float = 4.0,
        purpose: str = "",
        zone_id: str | None = None,
        enabled: bool = False,
        assigned_by: str = "",
    ) -> Camera:
        """Register a camera. **Disabled by default.**

        Creating a camera and switching it on are two acts. A row that started
        enabled would mean adding a camera silently begins processing video of
        people — which should always be a deliberate second decision.
        """
        _validate(
            camera_key=camera_key,
            channel=channel,
            stream_type=stream_type,
            analysis_fps=analysis_fps,
            credential_ref=credential_ref,
        )

        existing = await self._by_key(organization_id, camera_key)
        if existing is not None:
            raise ConflictError(f"camera '{camera_key}' already exists in this organization")

        camera = Camera(
            organization_id=organization_id,
            restaurant_id=restaurant_id,
            zone_id=zone_id,
            camera_key=camera_key,
            name=name,
            purpose=purpose,
            host=host,
            rtsp_port=rtsp_port,
            channel=channel,
            stream_type=stream_type,
            username=username,
            credential_ref=credential_ref,
            analysis_fps=analysis_fps,
            enabled=enabled,
        )
        self._session.add(camera)

        # Open the first zone interval. Recorded here rather than left to the
        # caller because "where was this camera" must be answerable for every
        # camera the moment it exists — a camera whose zone was never written
        # down produces observations nobody can place, and a route that forgot
        # this call would create one silently.
        await record_assignment(
            self._session,
            organization_id=organization_id,
            camera_key=camera_key,
            zone_id=zone_id,
            restaurant_id=restaurant_id,
            assigned_by=assigned_by,
        )
        return camera

    async def update(
        self, *, organization_id: str, camera_key: str, assigned_by: str = "", **changes: Any
    ) -> Camera:
        camera = await self.get(organization_id=organization_id, camera_key=camera_key)

        allowed = {
            "name",
            "purpose",
            "host",
            "rtsp_port",
            "channel",
            "stream_type",
            "username",
            "credential_ref",
            "analysis_fps",
            "zone_id",
            "enabled",
        }
        zone_before = camera.zone_id
        for field, value in changes.items():
            if field not in allowed or value is None:
                continue
            setattr(camera, field, value)

        # A zone change closes the interval in force and opens a new one. The
        # old row keeps its zone forever, so every reading this camera produced
        # before the move stays attributed to where it actually happened.
        if camera.zone_id != zone_before:
            await record_assignment(
                self._session,
                organization_id=organization_id,
                camera_key=camera.camera_key,
                zone_id=camera.zone_id,
                restaurant_id=camera.restaurant_id,
                assigned_by=assigned_by,
            )

        _validate(
            camera_key=camera.camera_key,
            channel=camera.channel,
            stream_type=camera.stream_type,
            analysis_fps=camera.analysis_fps,
            credential_ref=camera.credential_ref,
        )
        camera.updated_at = datetime.now(UTC)
        return camera

    async def set_enabled(self, *, organization_id: str, camera_key: str, enabled: bool) -> Camera:
        camera = await self.get(organization_id=organization_id, camera_key=camera_key)
        camera.enabled = enabled
        camera.updated_at = datetime.now(UTC)
        return camera

    async def retire(
        self,
        *,
        organization_id: str,
        camera_key: str,
        retired_by: str = "",
        observation_log: Any = None,
        durable_log: bool = False,
    ) -> int:
        """Delete a camera **and** destroy its observation partition, or do neither.

        Returns the number of observations removed.

        ### The gap this closes

        Retention enumerates observation-log partitions from this table, because
        a partition read from the store instead could not be attributed to a
        tenant. The consequence is that a deleted camera row would orphan its
        partition: the sweep would stop visiting it, and its observations —
        records about identifiable staff at work — would sit on disk past their
        retention date with nothing left to clean them up.

        The fix belongs here rather than in the sweep. A sweep that went looking
        for orphaned directories would have to guess which of them were once
        cameras and which tenant each belonged to, and a guess is exactly what
        the roster-based enumeration exists to avoid.

        ### Neither, rather than one

        If the partition cannot be purged, **the camera is not deleted**. That
        ordering is the whole safety property: a deployment that binds a durable
        log but runs this request in a process with no synthesis assembled
        cannot reach the log, and deleting the row there would create precisely
        the orphan this method exists to prevent. It refuses instead, and says
        why.

        The purge is total — `truncate(partition, now)` removes every record
        before this instant, which is all of them. That is a deliberate choice
        over letting them age out: a camera that no longer exists has no
        retention schedule to age out *on*, and no configuration a later
        operator could consult to find out what the schedule had been.

        ### What is deliberately kept

        `camera_zone_assignments` rows survive, with the open interval closed.
        They are the historical attribution for incidents, evidence and frames
        that still exist and still name this `camera_key`; deleting them would
        erase where those past events happened, which is the exact failure the
        assignment history was built to prevent. A camera is configuration; where
        it was is history.
        """
        camera = await self.get(organization_id=organization_id, camera_key=camera_key)

        if durable_log and observation_log is None:
            raise ConfigurationInvalidError(
                "refusing to delete a camera: this deployment keeps a durable "
                "observation log and this process cannot reach it, so the "
                "camera's observations would outlive their retention with "
                "nothing left to sweep them",
                details={"camera_key": camera_key},
            )

        removed = 0
        if observation_log is not None:
            from vision_os.core.model.ids import CameraId
            from vision_os.core.model.timebase import Instant

            now = datetime.now(UTC)
            # Everything before this instant, which is everything. `truncate` is
            # the only shortening operation P20 offers and its own contract says
            # it exists "for retention alone"; this is a retention act.
            removed = int(
                observation_log.truncate(
                    CameraId(camera_key),
                    Instant(int(now.timestamp() * 1_000_000_000)),
                )
            )

        # Close the interval in force. The row is never deleted and never
        # rewritten — a past observation still resolves to the zone it was
        # actually observed in.
        open_interval = (
            await self._session.execute(
                select(CameraZoneAssignment)
                .where(
                    CameraZoneAssignment.organization_id == organization_id,
                    CameraZoneAssignment.camera_key == camera_key,
                    CameraZoneAssignment.effective_to.is_(None),
                )
                .order_by(CameraZoneAssignment.effective_from.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if open_interval is not None:
            open_interval.effective_to = datetime.now(UTC)

        await self._session.delete(camera)
        return removed

    async def get(self, *, organization_id: str, camera_key: str) -> Camera:
        camera = await self._by_key(organization_id, camera_key)
        if camera is None:
            raise NotFoundError("no such camera")
        return camera

    async def list(
        self,
        *,
        organization_id: str,
        camera_keys: tuple[str, ...] | None = None,
        enabled_only: bool = False,
    ) -> list[Camera]:
        """Cameras this caller may see.

        `camera_keys is None` is a tenant-wide grant; an **empty tuple is none**.
        """
        if camera_keys is not None and len(camera_keys) == 0:
            return []

        statement = select(Camera).where(Camera.organization_id == organization_id)
        if camera_keys is not None:
            statement = statement.where(Camera.camera_key.in_(camera_keys))
        if enabled_only:
            statement = statement.where(Camera.enabled.is_(True))

        result = await self._session.execute(statement.order_by(Camera.camera_key))
        return list(result.scalars().all())

    async def enabled_for_runtime(self, *, organization_id: str) -> list[Camera]:
        """What the live runtime should start. **Enabled rows only.**

        The one query that decides whether a DVR channel becomes a pipeline.
        """
        return await self.list(organization_id=organization_id, enabled_only=True)

    async def _by_key(self, organization_id: str, camera_key: str) -> Camera | None:
        result = await self._session.execute(
            select(Camera).where(
                Camera.organization_id == organization_id,
                Camera.camera_key == camera_key,
            )
        )
        return result.scalar_one_or_none()


def _validate(
    *,
    camera_key: str,
    channel: int,
    stream_type: str,
    analysis_fps: float,
    credential_ref: str,
) -> None:
    if not camera_key.strip():
        raise ValidationError("a camera must have a key; identity is never inferred")
    if channel < 1:
        raise ValidationError("channel numbering starts at 1")
    if stream_type not in VALID_STREAM_TYPES:
        raise ValidationError(f"stream_type must be one of {sorted(VALID_STREAM_TYPES)}")
    if analysis_fps <= 0:
        raise ValidationError("analysis_fps must be positive")

    if credential_ref and not any(
        credential_ref.startswith(scheme) for scheme in ("env:", "file:", "literal:")
    ):
        # A bare value here is a password in the database. Refused at the
        # boundary rather than discovered in a backup.
        raise ValidationError(
            "credential_ref must be a reference (env:, file: or literal:), never "
            "a password; the secret provider resolves it at connect time",
            details={"hint": "env:CCTV_PASSWORD"},
        )


def to_wire(camera: Camera) -> dict[str, Any]:
    """A camera for the API.

    Carries `credential_ref` — which is a *pointer*, not a secret — but never a
    username-and-password URL, and never a resolved value. `credential_configured`
    answers "is this camera able to authenticate" without revealing anything.
    """
    return {
        "camera_key": camera.camera_key,
        "name": camera.name,
        "purpose": camera.purpose,
        "restaurant_id": camera.restaurant_id,
        "zone_id": camera.zone_id,
        "channel": camera.channel,
        "stream_type": camera.stream_type,
        "host": camera.host,
        "rtsp_port": camera.rtsp_port,
        "username": camera.username,
        "credential_ref": camera.credential_ref,
        "credential_configured": bool(camera.credential_ref),
        "analysis_fps": camera.analysis_fps,
        "enabled": camera.enabled,
        # Redacted, and still diagnosable. Never the dialling URL.
        "uri": _redacted_uri(camera),
        "created_at": _iso(camera.created_at),
        "updated_at": _iso(camera.updated_at),
    }


def _redacted_uri(camera: Camera) -> str:
    if not camera.host:
        return ""
    subtype = 0 if camera.stream_type == "main" else 1
    credential = "***:***@" if (camera.username or camera.credential_ref) else ""
    return (
        f"rtsp://{credential}{camera.host}:{camera.rtsp_port}"
        f"/cam/realmonitor?channel={camera.channel}&subtype={subtype}"
    )


class FrameService:
    """Frame **metadata**. No pixels, ever."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        organization_id: str,
        camera_key: str,
        sequence: int,
        epoch: int,
        captured_at: datetime,
        received_at: datetime,
        width: int = 0,
        height: int = 0,
        source_kind: str = "replay",
        frame_ref: str = "",
        observation_count: int = 0,
    ) -> FrameRecord:
        record = FrameRecord(
            organization_id=organization_id,
            camera_key=camera_key,
            sequence=sequence,
            epoch=epoch,
            frame_ref=frame_ref or f"{camera_key}:{epoch}:{sequence}",
            captured_at=captured_at,
            received_at=received_at,
            width=width,
            height=height,
            observation_count=observation_count,
            # `live` or `replay`, carried through verbatim. A replay frame is
            # never presented as live.
            source_kind=source_kind,
        )
        self._session.add(record)
        return record

    async def list(
        self,
        *,
        organization_id: str,
        camera_key: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[FrameRecord]:
        statement = select(FrameRecord).where(
            FrameRecord.organization_id == organization_id,
            FrameRecord.camera_key == camera_key,
        )
        if since:
            statement = statement.where(FrameRecord.captured_at >= since)
        if until:
            statement = statement.where(FrameRecord.captured_at <= until)

        statement = statement.order_by(FrameRecord.captured_at.desc()).limit(
            min(max(limit, 1), 500)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())


def frame_to_wire(record: FrameRecord) -> dict[str, Any]:
    return {
        "frame_ref": record.frame_ref,
        "camera_key": record.camera_key,
        "sequence": record.sequence,
        "epoch": record.epoch,
        "captured_at": _iso(record.captured_at),
        "received_at": _iso(record.received_at),
        "width": record.width,
        "height": record.height,
        "observation_count": record.observation_count,
        "source_kind": record.source_kind,
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def to_rtsp_config(camera: Camera) -> RtspCameraConfig:
    """A durable camera row, as live-runtime configuration.

    The one translation between the persisted record and the thing that opens a
    socket. `credential_ref` crosses unchanged and unresolved — the secret
    provider resolves it at connect time, inside the source, and the resolved
    value never returns here.
    """
    from app.vision.sources.rtsp import RtspCameraConfig

    return RtspCameraConfig(
        camera_id=camera.camera_key,
        host=camera.host,
        channel=camera.channel,
        port=camera.rtsp_port,
        stream_type=camera.stream_type,
        username=camera.username,
        credential_ref=camera.credential_ref,
        analysis_fps=camera.analysis_fps,
        enabled=camera.enabled,
    )


__all__ = [
    "CameraService",
    "FrameService",
    "frame_to_wire",
    "to_rtsp_config",
    "to_wire",
]
