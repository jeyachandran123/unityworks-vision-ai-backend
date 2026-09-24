"""Camera configuration, from the database.

Replaces `CCTV_CHANNELS`. The rule it enforces is unchanged and now durable: the
DVR has 16 channels, and **only rows that are `enabled` create a session**. A
disabled camera opens no socket, decodes nothing and reaches no model. Cost
follows configuration, not hardware.

### A camera is a channel of a recorder

How to *reach* a camera — address, port, account, password, and the URL
convention of its vendor — belongs to its `Recorder`, because it is the same for
every channel on that box. A camera row holds what genuinely differs: which
channel, which stream, where it is, and how it is analysed. See
`app/domain/recorders.py`.

The credential is still a reference, and still never a password. It lives on
the recorder now.

### Frame metadata

`FrameService` records that a frame existed and what it contributed to — not the
frame. §12: this is traceability, not a video recorder. The crops that were
deliberately retained live in `evidence_records`; everything else is gone by
design, which is a smaller privacy surface and the whole point.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.models import Camera, CameraZoneAssignment, FrameRecord, Recorder
from app.domain.runtime_identity import for_camera, runtime_camera_id, validate_camera_key
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
        camera_key: str,
        name: str,
        channel: int,
        recorder_id: str,
        stream_type: str = "sub",
        analysis_fps: float = 4.0,
        purpose: str = "",
        zone_id: str = "",
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
        )

        existing = await self._by_key(organization_id, camera_key)
        if existing is not None:
            raise ConflictError(f"camera '{camera_key}' already exists in this organization")

        camera = Camera(
            organization_id=organization_id,
            zone_id=zone_id,
            recorder_id=recorder_id,
            camera_key=camera_key,
            name=name,
            purpose=purpose,
            channel=channel,
            stream_type=stream_type,
            analysis_fps=analysis_fps,
            enabled=enabled,
        )
        self._session.add(camera)
        # Loaded now, so the camera returned from here can be turned into a
        # runtime config or a wire shape without a second round trip.
        await self._session.flush()
        await self._session.refresh(camera, attribute_names=["recorder"])

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
            assigned_by=assigned_by,
        )
        return camera

    async def update(
        self, *, organization_id: str, camera_key: str, assigned_by: str = "", **changes: Any
    ) -> Camera:
        camera = await self.get(organization_id=organization_id, camera_key=camera_key)

        # How to reach the camera is the recorder's. A camera can be moved to
        # another recorder, but never handed an address of its own.
        allowed = {
            "name",
            "purpose",
            "recorder_id",
            "channel",
            "stream_type",
            "analysis_fps",
            "zone_id",
            "enabled",
            "analysis_enabled",
        }
        # A truthy string is the dangerous case: `"false"` is truthy in Python,
        # so an untyped payload could switch analysis ON while the operator
        # believed they had switched it off. Rejected rather than coerced —
        # guessing what a client meant about whether people are analysed is not
        # a decision this layer may make.
        #
        # Scoped to this flag deliberately. `enabled` has the same untyped
        # exposure and is left exactly as it is: widening the check would change
        # an existing API's behaviour in a stage scoped to `analysis_enabled`.
        analysis_flag = changes.get("analysis_enabled")
        if analysis_flag is not None and not isinstance(analysis_flag, bool):
            raise ValidationError(
                f"analysis_enabled must be true or false, got " f"{type(analysis_flag).__name__}"
            )

        zone_before = camera.zone_id
        recorder_before = camera.recorder_id
        for field, value in changes.items():
            if field not in allowed or value is None:
                continue
            setattr(camera, field, value)
        if camera.recorder_id != recorder_before:
            await self._session.flush()
            await self._session.refresh(camera, attribute_names=["recorder"])

        # A zone change closes the interval in force and opens a new one. The
        # old row keeps its zone forever, so every reading this camera produced
        # before the move stays attributed to where it actually happened.
        if camera.zone_id != zone_before:
            await record_assignment(
                self._session,
                organization_id=organization_id,
                camera_key=camera.camera_key,
                zone_id=camera.zone_id,
                assigned_by=assigned_by,
            )

        _validate(
            camera_key=camera.camera_key,
            channel=camera.channel,
            stream_type=camera.stream_type,
            analysis_fps=camera.analysis_fps,
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
            # On a worker thread, and awaited, so it still happens before the
            # row is deleted. `truncate` takes the log's lock, which the
            # analysis pipeline and whole-file readers also hold; under load it
            # waits. Waiting on the API loop froze every other request with it
            # — 10.9 s measured on 2026-09-24, over a minute once.
            removed = int(
                await asyncio.to_thread(
                    observation_log.truncate,
                    # The tenant-qualified partition, never the bare key. A bare
                    # key here would truncate whichever organization's partition
                    # happened to be named that.
                    CameraId(runtime_camera_id(organization_id, camera_key)),
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

    async def next_key(self, *, organization_id: str) -> str:
        """The first free `cam-NN` in this organization.

        Proposed by the server because the key is what the pipeline partitions
        on: it names observation files on disk and appears in every runtime
        identity. Asking a person to invent one is asking them to collide with
        a camera they cannot see.
        """
        used = set(
            (
                await self._session.execute(
                    select(Camera.camera_key).where(Camera.organization_id == organization_id)
                )
            )
            .scalars()
            .all()
        )
        number = 1
        while f"cam-{number:02d}" in used:
            number += 1
        return f"cam-{number:02d}"

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
        recorder_id: str | None = None,
    ) -> list[Camera]:
        """Cameras this caller may see.

        `camera_keys is None` is a tenant-wide grant; an **empty tuple is none**.
        """
        if camera_keys is not None and len(camera_keys) == 0:
            return []

        statement = (
            select(Camera)
            .where(Camera.organization_id == organization_id)
            .options(selectinload(Camera.recorder))
        )
        if camera_keys is not None:
            statement = statement.where(Camera.camera_key.in_(camera_keys))
        if enabled_only:
            statement = statement.where(Camera.enabled.is_(True))
        if recorder_id is not None:
            statement = statement.where(Camera.recorder_id == recorder_id)

        result = await self._session.execute(statement.order_by(Camera.camera_key))
        return list(result.scalars().all())

    async def enabled_for_runtime(self, *, organization_id: str) -> list[Camera]:
        """What the live runtime should start. **Enabled rows on active recorders.**

        The one query that decides whether a DVR channel becomes a pipeline. A
        deactivated recorder keeps every camera configured on it, and starts
        none of them.
        """
        return [
            camera
            for camera in await self.list(organization_id=organization_id, enabled_only=True)
            if is_dialable(camera)
        ]

    async def _by_key(self, organization_id: str, camera_key: str) -> Camera | None:
        result = await self._session.execute(
            select(Camera)
            .where(
                Camera.organization_id == organization_id,
                Camera.camera_key == camera_key,
            )
            .options(selectinload(Camera.recorder))
        )
        return result.scalar_one_or_none()


def _validate(
    *,
    camera_key: str,
    channel: int,
    stream_type: str,
    analysis_fps: float,
) -> None:
    # Charset, not merely non-emptiness. The key is one half of the camera's
    # runtime identity, and `app.domain.runtime_identity` depends on this
    # charset to guarantee that identity parses back unambiguously.
    validate_camera_key(camera_key)
    if channel < 1:
        raise ValidationError("channel numbering starts at 1")
    if stream_type not in VALID_STREAM_TYPES:
        raise ValidationError(f"stream_type must be one of {sorted(VALID_STREAM_TYPES)}")
    if analysis_fps <= 0:
        raise ValidationError("analysis_fps must be positive")


def to_wire(camera: Camera) -> dict[str, Any]:
    """A camera for the API.

    Names its recorder rather than repeating the recorder's address: how to
    reach the camera is the recorder's, and a camera page that showed an address
    would invite editing it in the one place it cannot be changed.
    `credential_configured` answers "can this camera authenticate" from the
    recorder, and never carries the credential or its reference.
    """
    from app.domain.recorders import brand_label, credential_configured, credential_scheme

    recorder = camera.recorder
    return {
        "camera_key": camera.camera_key,
        # Globally unique. The key alone is not, and a client that keys its own
        # state on the bare key inherits the collision this exists to close.
        "runtime_id": for_camera(camera),
        "name": camera.name,
        "purpose": camera.purpose,
        "zone_id": camera.zone_id,
        "channel": camera.channel,
        "stream_type": camera.stream_type,
        "recorder_id": camera.recorder_id,
        "recorder_name": recorder.name if recorder is not None else "",
        "recorder_brand": brand_label(recorder) if recorder is not None else "",
        "recorder_active": bool(recorder is not None and recorder.is_active),
        "credential_configured": bool(recorder is not None and credential_configured(recorder)),
        "credential_scheme": credential_scheme(recorder.credential_ref) if recorder else "",
        "analysis_fps": camera.analysis_fps,
        "enabled": camera.enabled,
        # Two decisions, not one. `enabled` is "this camera streams"; this is
        # "this camera is analysed". A client that shows only the first cannot
        # explain why a visibly live camera raises nothing.
        "analysis_enabled": camera.analysis_enabled,
        # Redacted, and still diagnosable. Never the dialling URL.
        "uri": _redacted_uri(camera),
        "created_at": _iso(camera.created_at),
        "updated_at": _iso(camera.updated_at),
    }


def is_dialable(camera: Camera) -> bool:
    """Whether the runtime may open this camera at all.

    Its recorder exists, is active, and has an address. A camera failing any of
    these is kept exactly as configured and simply not started — which is what
    "deactivate the recorder" means.
    """
    recorder = getattr(camera, "recorder", None)
    return bool(recorder is not None and recorder.is_active and recorder.host)


def _redacted_uri(camera: Camera) -> str:
    """The URL this camera dials, with the account and password replaced.

    Built by the same code that builds the real one, so it cannot drift from
    what is actually dialled — the old version hardcoded the Dahua path and
    would have shown a Hikvision camera a URL it never used.
    """
    recorder = getattr(camera, "recorder", None)
    if recorder is None or not recorder.host:
        return ""
    try:
        return to_rtsp_config(camera).redacted_uri()
    except Exception:  # noqa: BLE001 - a display string must never break a listing
        return ""


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


def to_rtsp_config(camera: Camera, *, recorder: Recorder | None = None) -> RtspCameraConfig:
    """A durable camera row, as live-runtime configuration.

    The one translation between the persisted record and the thing that opens a
    socket. The address, account and credential reference come from the
    camera's recorder, and the path and stream numbering from the recorder's
    brand — so a Hikvision camera and a Dahua camera with identical settings
    each dial what their own device expects.

    `credential_ref` crosses unchanged and unresolved: the secret provider
    resolves it at connect time, inside the source, and the resolved value never
    returns here.

    `recorder` may be passed explicitly for a caller holding one that is not
    attached to the camera object; otherwise the camera's own is used.
    """
    from app.domain.recorders import brand_of
    from app.vision.sources.rtsp import RtspCameraConfig

    box = recorder if recorder is not None else camera.recorder
    if box is None:
        raise ValueError(f"camera '{camera.camera_key}' has no recorder, so it cannot be dialled")
    brand = brand_of(box)

    return RtspCameraConfig(
        # The runtime identity, not the tenant-scoped key: this id names the
        # session in every process-wide registry it reaches, and two tenants
        # may legitimately both call a camera `cam-01`.
        camera_id=for_camera(camera),
        host=box.host,
        channel=camera.channel,
        port=box.rtsp_port,
        stream_type=camera.stream_type,
        username=box.username,
        credential_ref=box.credential_ref,
        analysis_fps=camera.analysis_fps,
        enabled=camera.enabled,
        path_template=brand.path_template,
        stream_values=(brand.main, brand.sub),
    )


__all__ = [
    "CameraService",
    "FrameService",
    "frame_to_wire",
    "is_dialable",
    "to_rtsp_config",
    "to_wire",
]
