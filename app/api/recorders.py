"""Recorders — adding, testing and running the DVRs and NVRs cameras plug into.

Gated on the camera permissions: `VIEW_CAMERAS` to read, `MANAGE_CAMERAS` to
change. A recorder is camera infrastructure, and a new permission nobody is
granted would make the whole feature invisible until somebody noticed.

### The password only goes in

`POST /recorders` and `PUT /recorders/{id}/credential` accept a password and
seal it before it reaches the database. No response on any route here carries
it back — not the value, not the ciphertext, not a masked copy. Every response
is `Cache-Control: no-store`, because each describes how to reach a camera.

### One save for the whole wizard

`POST /recorders` accepts the cameras to create with it, and creates the lot in
one transaction. A recorder saved with half its cameras — because the fourth
one's channel was taken — would be a partial configuration nobody chose, and
the person would be left to work out which half.

### Every value is the organization's own

Address, domain, port, account, password and brand arrive in the request and
are stored on this organization's recorder. Nothing on these routes reads a
deployment-wide CCTV setting, and no request can name a credential reference —
only a password, which is sealed.

### Tests are real, and limited

`POST /recorders/test` (before saving) and `POST /recorders/{id}/test` (after)
open the stream and decode a picture, through the same code the runtime dials
with. Each is audited, because it uses a credential, and throttled per
recorder, because an unthrottled button is a way to lock a DVR account.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request, Response

from app.api.dependencies import CurrentAccess, DbSession, live_of, requires, settings_of
from app.api.product import scope_cameras
from app.authorization.model import AccessDecision, Permission
from app.domain import cameras as camera_domain
from app.domain import recorders as recorder_domain
from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.recorder_brands import CUSTOM, BrandError, custom_brand
from app.domain.recorder_probe import ProbeThrottle, check_connection
from app.domain.runtime_identity import for_camera
from app.errors import RateLimitedError, ValidationError

router = APIRouter(prefix="/api/v1", tags=["recorders"])

#: Shared by both test routes, per process. See `ProbeThrottle`.
_THROTTLE = ProbeThrottle(interval_s=3.0)

_NO_STORE = "no-store, private"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _roles(access: AccessDecision) -> tuple[str, ...]:
    return tuple(sorted(r.value for r in access.roles))


def _store(request: Request) -> Any:
    return getattr(request.app.state, "recorder_credentials", None)


def _provider(request: Request) -> Any:
    """The process-wide credential provider — the one the runtime dials with."""
    return request.app.state.credential_provider


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = _NO_STORE


async def _audit(
    session,
    request: Request,
    access: AccessDecision,
    action: AuditAction,
    recorder_id: str,
    detail: dict[str, Any] | None = None,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
) -> None:
    await AuditTrail(session).record(
        action=action,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="recorder",
        resource_id=recorder_id,
        request_id=_request_id(request),
        outcome=outcome,
        detail=detail,
    )


def _connection_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """The editable connection fields a client may send. Never a credential reference."""
    return {field: payload[field] for field in recorder_domain.EDITABLE if field in payload}


def _via(payload: dict[str, Any]) -> str | None:
    """Which of the recorder's addresses a test should use, when the caller says."""
    via = payload.get("via")
    if via in (None, ""):
        return None
    if via not in recorder_domain.ADDRESS_CHOICES:
        raise ValidationError("choose whether to test the IP address or the domain")
    return str(via)


def _password(payload: dict[str, Any], *, required: bool) -> str | None:
    value = payload.get("password")
    if value is None or value == "":
        if required:
            raise ValidationError("enter the recorder's password")
        return None
    if not isinstance(value, str):
        raise ValidationError("the password must be text")
    return value


async def _recorder_wire(session, access: AccessDecision, recorder) -> dict[str, Any]:
    counts = await recorder_domain.RecorderService(session).camera_counts(
        organization_id=access.tenant_id
    )
    total, on = counts.get(recorder.id, (0, 0))
    return recorder_domain.to_wire(recorder, camera_count=total, cameras_on=on)


# ── reading ──────────────────────────────────────────────────────────────────


@router.get("/recorder-brands", dependencies=[Depends(requires(Permission.VIEW_CAMERAS))])
async def list_recorder_brands() -> dict[str, Any]:
    """The brands the dropdown offers, plus the custom escape hatch.

    Labels and default ports only. A person choosing a brand is answering "what
    does it say on the box"; the URL it implies is the server's business.
    """
    return {"brands": recorder_domain.brands_to_wire(), "custom": CUSTOM}


@router.get("/recorders", dependencies=[Depends(requires(Permission.VIEW_CAMERAS))])
async def list_recorders(
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    rows, total = await service.list(organization_id=access.tenant_id, limit=limit, offset=offset)
    counts = await service.camera_counts(organization_id=access.tenant_id)
    return {
        "recorders": [
            recorder_domain.to_wire(
                row,
                camera_count=counts.get(row.id, (0, 0))[0],
                cameras_on=counts.get(row.id, (0, 0))[1],
            )
            for row in rows
        ],
        "count": len(rows),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/recorders/{recorder_id}", dependencies=[Depends(requires(Permission.VIEW_CAMERAS))])
async def get_recorder(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """One recorder, with the cameras on it and whether each is running."""
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    recorder = await service.get(organization_id=access.tenant_id, recorder_id=recorder_id)
    cameras = await service.cameras(organization_id=access.tenant_id, recorder_id=recorder.id)
    # Narrowed to the caller's camera reach, exactly as `/cameras` is: somebody
    # granted two cameras sees those two here, not every channel on the box.
    reach = scope_cameras(access)
    if reach is not None:
        cameras = [camera for camera in cameras if camera.camera_key in reach]
    body = await _recorder_wire(session, access, recorder)
    # Whether a session is actually open for each camera right now, read from
    # the runtime rather than inferred from `enabled` — "switched on" and
    # "streaming" differ exactly when something is wrong, which is when
    # somebody is looking.
    running = {
        live.camera_id
        for live in live_of(request).visible(tenant_id=access.tenant_id, camera_ids=None)
    }
    body["cameras"] = [
        {**camera_domain.to_wire(camera), "running": for_camera(camera) in running}
        for camera in cameras
    ]
    return body


# ── changing ─────────────────────────────────────────────────────────────────


@router.post("/recorders", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))])
async def create_recorder(
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Add a recorder — and, optionally, its cameras — in one transaction.

    `cameras` is a list of `{channel, name, zone_id, stream_type?, purpose?}`.
    Each camera is created **switched off**, as every camera is: adding one and
    beginning to process video of people are two decisions.
    """
    _no_store(response)
    settings = settings_of(request)
    service = recorder_domain.RecorderService(session)

    recorder = await service.create(
        organization_id=access.tenant_id,
        settings=settings,
        password=_password(payload, required=True),
        **_connection_fields(payload),
    )

    requested = payload.get("cameras") or []
    if not isinstance(requested, list):
        raise ValidationError("'cameras' must be a list")
    created = await _create_cameras(session, request, access, recorder, requested)

    recorder_domain.sync_credential(_store(request), recorder)
    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_CREATED,
        recorder.id,
        detail={
            "name": recorder.name,
            "brand": recorder.brand,
            "address": recorder.address,
            "connect_via": recorder.connect_via,
            "credential_scheme": recorder_domain.credential_scheme(recorder.credential_ref),
            "cameras": [camera.camera_key for camera in created],
        },
    )
    body = recorder_domain.to_wire(recorder, camera_count=len(created), cameras_on=0)
    body["cameras"] = [camera_domain.to_wire(camera) for camera in created]
    return body


async def _create_cameras(session, request, access, recorder, requested: list) -> list:
    """Create the wizard's cameras, refusing the whole save on any problem."""
    from app.api.product import _placement

    cameras = camera_domain.CameraService(session)
    seen: set[int] = set()
    created = []
    for position, item in enumerate(requested, start=1):
        if not isinstance(item, dict):
            raise ValidationError(f"camera {position} is not valid")
        try:
            channel = int(item.get("channel"))
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"camera {position}: enter a channel number") from exc
        if channel in seen:
            raise ValidationError(
                f"channel {channel} is listed twice; each camera is a different channel"
            )
        seen.add(channel)
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValidationError(f"camera on channel {channel}: give it a name")

        zone_id = await _placement(session, access.tenant_id, item.get("zone_id"))
        camera = await cameras.create(
            organization_id=access.tenant_id,
            camera_key=await cameras.next_key(organization_id=access.tenant_id),
            name=name,
            channel=channel,
            recorder_id=recorder.id,
            stream_type=str(item.get("stream_type") or "sub"),
            wall_fps=item.get("wall_fps", camera_domain.DEFAULT_WALL_FPS),
            purpose=str(item.get("purpose") or ""),
            zone_id=zone_id,
            enabled=False,
            assigned_by=access.subject,
        )
        await AuditTrail(session).record(
            action=AuditAction.CAMERA_CREATED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="camera",
            resource_id=camera.camera_key,
            request_id=_request_id(request),
            detail={"recorder_id": recorder.id, "channel": camera.channel},
        )
        created.append(camera)
    return created


@router.patch(
    "/recorders/{recorder_id}", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))]
)
async def update_recorder(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Change how a recorder is reached. The password is a separate route.

    Cameras already running keep their current connection until restarted —
    the page says so, rather than this route silently reconnecting a
    kitchen's worth of cameras mid-shift.
    """
    _no_store(response)
    if "password" in payload:
        raise ValidationError(
            "the password is changed on its own, with PUT /recorders/{id}/credential"
        )
    service = recorder_domain.RecorderService(session)
    recorder, changed = await service.update(
        organization_id=access.tenant_id,
        recorder_id=recorder_id,
        **_connection_fields(payload),
    )
    if changed:
        await _audit(
            session,
            request,
            access,
            AuditAction.RECORDER_UPDATED,
            recorder.id,
            detail={"fields": changed},
        )
    return await _recorder_wire(session, access, recorder)


@router.put(
    "/recorders/{recorder_id}/credential",
    dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))],
)
async def set_recorder_credential(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Replace the recorder's password. The old one is overwritten, never read."""
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    recorder = await service.set_credential(
        organization_id=access.tenant_id,
        recorder_id=recorder_id,
        password=_password(payload, required=True) or "",
        settings=settings_of(request),
    )
    recorder_domain.sync_credential(_store(request), recorder)
    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_CREDENTIAL_SET,
        recorder.id,
        detail={"credential_scheme": recorder_domain.credential_scheme(recorder.credential_ref)},
    )
    return await _recorder_wire(session, access, recorder)


@router.post(
    "/recorders/{recorder_id}/deactivate",
    dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))],
)
async def deactivate_recorder(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Stop every camera on this recorder, and start none until it is reactivated.

    Nothing is deleted. Each camera keeps its configuration and its own
    switched-on flag, so reactivating brings back exactly what was running.
    """
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    recorder = await service.set_active(
        organization_id=access.tenant_id, recorder_id=recorder_id, active=False
    )
    cameras = await service.cameras(organization_id=access.tenant_id, recorder_id=recorder.id)
    runtime_ids = [for_camera(camera) for camera in cameras]

    live = live_of(request)
    stopped = 0
    for runtime_id in runtime_ids:
        if await live.stop_camera(runtime_id, tenant_id=access.tenant_id):
            stopped += 1
    wall = getattr(request.app.state, "wall", None)
    if wall is not None:
        await wall.stop_cameras(runtime_ids)

    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_DEACTIVATED,
        recorder.id,
        detail={"cameras": len(cameras), "sessions_stopped": stopped},
    )
    body = await _recorder_wire(session, access, recorder)
    body["sessions_stopped"] = stopped
    return body


@router.post(
    "/recorders/{recorder_id}/activate",
    dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))],
)
async def activate_recorder(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Allow this recorder's cameras to run again, and restart the switched-on ones."""
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    recorder = await service.set_active(
        organization_id=access.tenant_id, recorder_id=recorder_id, active=True
    )
    recorder_domain.sync_credential(_store(request), recorder)
    cameras = await service.cameras(organization_id=access.tenant_id, recorder_id=recorder.id)

    started = 0
    settings = settings_of(request)
    if settings.feature_live_cctv:
        live = live_of(request)
        for camera in cameras:
            if not (camera.enabled and camera.analysis_enabled):
                continue
            try:
                await live.start_one(
                    camera_domain.to_rtsp_config(camera), tenant_id=access.tenant_id
                )
                started += 1
            except Exception:  # noqa: BLE001 - one camera, not the recorder
                continue
    wall = getattr(request.app.state, "wall", None)
    if wall is not None and settings.feature_camera_wall:
        await wall.start_cameras([camera for camera in cameras if camera.enabled])

    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_ACTIVATED,
        recorder.id,
        detail={"cameras": len(cameras), "sessions_started": started},
    )
    body = await _recorder_wire(session, access, recorder)
    body["sessions_started"] = started
    return body


@router.delete(
    "/recorders/{recorder_id}", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))]
)
async def delete_recorder(
    recorder_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Delete a recorder that holds no cameras. Refused, with the count, otherwise."""
    service = recorder_domain.RecorderService(session)
    recorder = await service.get(organization_id=access.tenant_id, recorder_id=recorder_id)
    name = recorder.name
    await service.delete(organization_id=access.tenant_id, recorder_id=recorder_id)
    store = _store(request)
    if store is not None:
        store.put(recorder_id, None)
    await _audit(
        session, request, access, AuditAction.RECORDER_DELETED, recorder_id, detail={"name": name}
    )
    return {"recorder_id": recorder_id, "deleted": True}


# ── testing ──────────────────────────────────────────────────────────────────


def _channel(payload: dict[str, Any]) -> int:
    try:
        channel = int(payload.get("channel", 1))
    except (TypeError, ValueError) as exc:
        raise ValidationError("the channel must be a number") from exc
    if channel < 1:
        raise ValidationError("channel numbering starts at 1")
    return channel


def _throttle(key: str) -> None:
    wait = _THROTTLE.check(key)
    if wait:
        raise RateLimitedError(
            f"This recorder was tested a moment ago. Try again in {wait:g} seconds.",
            details={"retry_after_s": wait},
        )


@router.post("/recorders/test", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))])
async def test_new_recorder(
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Test a recorder that has not been saved yet, with the password just typed.

    Nothing is stored. The password is used for this one connection and then
    dropped; it is never logged, echoed or written anywhere.

    `via` (`ip_address` or `hostname`) tests that one of the two addresses;
    without it, the one the recorder would connect with.
    """
    _no_store(response)
    fields = _connection_fields(payload)
    fields.setdefault("name", "Connection test")
    fields.setdefault("rtsp_port", 554)
    via = _via(payload)
    if via and not fields.get("connect_via"):
        # Testing one address of two is a choice of address for this test.
        fields["connect_via"] = via
    # Reuses the same validation a save would apply, so a test cannot pass for a
    # configuration the save would then refuse.
    clean = recorder_domain.clean_connection(**{k: fields.get(k) for k in recorder_domain.EDITABLE})
    address = clean[via] if via else recorder_domain.dial_address(clean)
    if not address:
        raise ValidationError(
            "enter the domain to test it"
            if via == "hostname"
            else "enter the IP address to test it"
        )
    password = _password(payload, required=True) or ""
    channel = _channel(payload)
    _throttle(f"{access.tenant_id}:new:{address}:{clean['rtsp_port']}")

    result = await check_connection(
        host=address,
        port=clean["rtsp_port"],
        username=clean["username"],
        password=password,
        brand=_brand_from(clean),
        channel=channel,
        stream_type=str(payload.get("stream_type") or "sub"),
    )
    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_TESTED,
        "(unsaved)",
        detail={"address": address, "channel": channel, "outcome": result.outcome},
        outcome=AuditOutcome.SUCCESS if result.ok else AuditOutcome.FAILED,
    )
    return result.as_dict()


@router.post(
    "/recorders/{recorder_id}/test",
    dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))],
)
async def test_saved_recorder(
    recorder_id: str,
    request: Request,
    response: Response,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(default_factory=dict)],
) -> dict[str, Any]:
    """Test a saved recorder with its stored password, on one channel.

    The password is resolved through the same provider the runtime uses, so this
    answers the question "will a camera on this recorder start" and not a
    neighbouring one. `via` tests the recorder's other address instead of the
    one it connects with — the check to run before switching.
    """
    _no_store(response)
    service = recorder_domain.RecorderService(session)
    recorder = await service.get(organization_id=access.tenant_id, recorder_id=recorder_id)
    channel = _channel(payload)
    address = recorder_domain.address_for(recorder, _via(payload))
    _throttle(f"{access.tenant_id}:{recorder.id}")

    store = _store(request)
    recorder_domain.sync_credential(store, recorder)
    try:
        password = _provider(request).resolve(recorder.credential_ref)
    except Exception:  # noqa: BLE001 - reported as an outcome, never as its text
        await _audit(
            session,
            request,
            access,
            AuditAction.RECORDER_TESTED,
            recorder.id,
            detail={"address": address, "channel": channel, "outcome": "credential_unavailable"},
            outcome=AuditOutcome.FAILED,
        )
        return {
            "ok": False,
            "outcome": "credential_unavailable",
            "message": "This server cannot read the recorder's saved password.",
            "hint": "Set the password again. If that fails, whoever runs the server needs to "
            "check its encryption key (RECORDER_SECRET_KEY).",
            "address": address,
            "port": recorder.rtsp_port,
            "channel": channel,
            "elapsed_ms": 0,
            "resolution": None,
        }

    try:
        result = await check_connection(
            host=address,
            port=recorder.rtsp_port,
            username=recorder.username,
            password=password,
            brand=recorder_domain.brand_of(recorder),
            channel=channel,
            stream_type=str(payload.get("stream_type") or "sub"),
        )
    finally:
        del password

    await _audit(
        session,
        request,
        access,
        AuditAction.RECORDER_TESTED,
        recorder.id,
        detail={"address": address, "channel": channel, "outcome": result.outcome},
        outcome=AuditOutcome.SUCCESS if result.ok else AuditOutcome.FAILED,
    )
    return result.as_dict()


def _brand_from(clean: dict[str, Any]):
    if clean["brand"] == CUSTOM:
        try:
            return custom_brand(
                path_template=clean["path_template"],
                main=int(clean["stream_main"]),
                sub=int(clean["stream_sub"]),
            )
        except BrandError as exc:  # pragma: no cover - `clean_connection` already checked
            raise ValidationError(str(exc)) from exc
    from app.domain.recorder_brands import brand_for

    return brand_for(clean["brand"])
