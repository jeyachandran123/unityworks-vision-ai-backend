"""The camera-wall API: list, stream, detail.

### Why a stream ticket exists

A wall tile is an `<img src="...">`. The browser issues that request itself and
there is no way to attach an `Authorization` header to it. The options were:

* **Put the bearer token in the URL** — it would land in browser history, in
  every proxy log and in the `Referer` header. A fifteen-minute credential in a
  log file is a fifteen-minute credential for whoever reads the log.
* **Rely on the refresh cookie** — it is scoped to `/api/v1/auth` on purpose,
  and widening that scope to cover media would undo a Phase 2 decision made for
  good reasons.
* **Mint a ticket.** An authenticated caller exchanges its token for a short,
  single-camera, single-tenant ticket that can do exactly one thing: read that
  camera's pictures.

The third is what this does. A leaked ticket is worth one camera for sixty
seconds, and it grants nothing else — not the API, not another camera, not
evidence.

### What never reaches the browser

The DVR username, the DVR password, the credential reference and the RTSP URL —
redacted or otherwise. The browser learns a camera id, a channel number and a
state. It could not connect to the DVR with everything this API will tell it.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.authorization.model import Permission
from app.domain.cameras import CameraService
from app.errors import AuthenticationError, NotFoundError
from app.vision.wall import DEFAULT_DETAIL_FPS, DEFAULT_WALL_FPS

router = APIRouter(prefix="/api/v1/wall", tags=["camera-wall"])

#: How long a stream ticket is worth anything. Long enough to open sixteen
#: tiles, short enough that a leaked one is nearly spent.
TICKET_TTL_S = 60
#: Multipart boundary for `multipart/x-mixed-replace`.
BOUNDARY = "uwvframe"


# ── tickets ──────────────────────────────────────────────────────────────────


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]


def mint_ticket(secret: str, *, tenant_id: str, camera_id: str, subject: str) -> str:
    """A ticket for **one camera**, for one tenant, for a minute.

    Carries no permission beyond reading this camera's pictures, so it cannot be
    replayed against the API. Signed rather than stored: a stateless ticket
    survives a restart and needs no cleanup, and the TTL bounds the damage.
    """
    expires = int(time.time()) + TICKET_TTL_S
    payload = f"{tenant_id}:{camera_id}:{subject}:{expires}"
    return f"{expires}.{_sign(secret, payload)}"


def verify_ticket(
    secret: str, ticket: str, *, tenant_id: str, camera_id: str, subject: str
) -> bool:
    try:
        raw_expiry, signature = ticket.split(".", 1)
        expires = int(raw_expiry)
    except (ValueError, AttributeError):
        return False
    if expires < time.time():
        return False
    payload = f"{tenant_id}:{camera_id}:{subject}:{expires}"
    # Constant-time: a timing oracle on a signature is a signature.
    return hmac.compare_digest(signature, _sign(secret, payload))


# ── listing ──────────────────────────────────────────────────────────────────


def _visible_keys(access) -> tuple[str, ...] | None:
    from app.authorization.model import ScopeBreadth

    if access.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return None
    return access.cameras.camera_ids


@router.get("/cameras", dependencies=[Depends(requires(Permission.VIEW_LIVE))])
async def list_wall_cameras(
    request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """Every camera this caller may watch, with its live state.

    **No filtering by usefulness.** A camera that is disabled, offline or
    pointed at a store cupboard is still listed, because a monitoring wall whose
    operator cannot see that channel 7 is dark is worse than no wall. The only
    thing that removes a camera here is authorization.
    """
    wall = request.app.state.wall
    rows = await CameraService(session).list(
        organization_id=access.tenant_id, camera_keys=_visible_keys(access)
    )

    cameras = []
    for camera in rows:
        stream = wall.get(camera.camera_key)
        if stream is not None:
            entry = stream.to_wire()
        else:
            # Configured but not streaming — reported as itself, not omitted.
            entry = {
                "camera_id": camera.camera_key,
                "name": camera.name,
                "channel": camera.channel,
                "stream_type": camera.stream_type,
                "enabled": camera.enabled,
                "state": "disabled" if not camera.enabled else "offline",
                "width": 0,
                "height": 0,
                "viewers": 0,
                "reconnects": 0,
                "frames_decoded": 0,
                "seconds_since_frame": None,
                "first_frame_latency_s": None,
                "last_error": "",
            }
        entry["purpose"] = camera.purpose
        cameras.append(entry)

    cameras.sort(key=lambda c: c["channel"])
    return {
        "cameras": cameras,
        "total": len(cameras),
        "live": sum(1 for c in cameras if c["state"] == "live"),
        "wall": wall.summary(),
        "default_wall_fps": DEFAULT_WALL_FPS,
        "default_detail_fps": DEFAULT_DETAIL_FPS,
    }


@router.post("/cameras/{camera_id}/ticket", dependencies=[Depends(requires(Permission.VIEW_LIVE))])
async def issue_ticket(
    camera_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(default_factory=dict)],
) -> dict[str, Any]:
    """Exchange an authenticated session for a one-camera viewing ticket."""
    allowed = _visible_keys(access)
    if allowed is not None and camera_id not in allowed:
        # Not `Forbidden`: confirming the camera exists is itself a disclosure.
        raise NotFoundError("no such camera")

    # Proves the camera is real and in this tenant before minting anything.
    await CameraService(session).get(
        organization_id=access.tenant_id, camera_key=camera_id
    )

    settings = settings_of(request)
    ticket = mint_ticket(
        settings.secret_key.get_secret_value(),
        tenant_id=access.tenant_id,
        camera_id=camera_id,
        subject=access.subject,
    )
    return {
        "camera_id": camera_id,
        "ticket": ticket,
        "expires_in": TICKET_TTL_S,
        # The path the browser should put in `<img src>`. Built here so the
        # frontend never assembles a media URL by hand.
        "stream_path": f"/api/v1/wall/cameras/{camera_id}/stream.mjpg",
    }


# ── streaming ────────────────────────────────────────────────────────────────


@router.get("/cameras/{camera_id}/stream.mjpg")
async def stream_camera(
    camera_id: str,
    request: Request,
    ticket: Annotated[str, Query()],
    tenant: Annotated[str, Query()],
    subject: Annotated[str, Query()],
    fps: Annotated[float, Query(ge=0.5, le=25.0)] = DEFAULT_WALL_FPS,
) -> StreamingResponse:
    """`multipart/x-mixed-replace` JPEG. Authorized by ticket, not by header.

    Deliberately **not** behind `requires(...)`: an `<img>` tag cannot send an
    Authorization header, so this route authorizes on the signed ticket the
    caller obtained from an authenticated request. The ticket names the tenant,
    the camera and the subject, and all three are checked here.
    """
    settings = settings_of(request)
    if not verify_ticket(
        settings.secret_key.get_secret_value(),
        ticket,
        tenant_id=tenant,
        camera_id=camera_id,
        subject=subject,
    ):
        raise AuthenticationError("this stream ticket is not valid for this camera")

    wall = request.app.state.wall
    stream = wall.get(camera_id)
    if stream is None:
        raise NotFoundError("this camera is not streaming")

    # Tenancy is settled by the ticket, not by the stream's copy of the camera.
    #
    # The ticket was minted only after a tenant-scoped database read proved the
    # camera exists in that tenant, and it is HMAC-signed over exactly
    # (tenant, camera, subject). Re-checking against `stream.camera` would add
    # nothing to that and did add a real failure: the wall holds ORM rows loaded
    # at start-up, so a camera moved between tenants left every stream returning
    # 403 until the process was restarted — a stale cache silently acting as an
    # authorization input.

    interval = 1.0 / max(fps, 0.5)

    async def frames():
        stream.stats.viewers += 1
        seq = 0
        try:
            while True:
                if await request.is_disconnected():
                    break
                # Blocking wait moved off the event loop: sixteen viewers must
                # not each hold it while waiting for a camera.
                seq, jpeg = await asyncio.to_thread(stream.latest, seq, 5.0)
                if jpeg is None:
                    continue
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                await asyncio.sleep(interval)
        finally:
            stream.stats.viewers = max(0, stream.stats.viewers - 1)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            # Live imagery of identifiable people is never cached.
            "Cache-Control": "no-store, private",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cameras/{camera_id}", dependencies=[Depends(requires(Permission.VIEW_LIVE))])
async def camera_detail(
    camera_id: str, request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """One camera, for the detail view. Carries no credential and no URL."""
    allowed = _visible_keys(access)
    if allowed is not None and camera_id not in allowed:
        raise NotFoundError("no such camera")

    camera = await CameraService(session).get(
        organization_id=access.tenant_id, camera_key=camera_id
    )
    stream = request.app.state.wall.get(camera_id)
    detail = stream.to_wire() if stream is not None else {
        "camera_id": camera.camera_key,
        "name": camera.name,
        "channel": camera.channel,
        "stream_type": camera.stream_type,
        "enabled": camera.enabled,
        "state": "disabled" if not camera.enabled else "offline",
    }
    detail["purpose"] = camera.purpose
    detail["restaurant_id"] = camera.restaurant_id
    return detail


__all__ = ["TICKET_TTL_S", "mint_ticket", "router", "verify_ticket"]
