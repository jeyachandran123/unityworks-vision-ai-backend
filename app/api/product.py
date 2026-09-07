"""The product API over durable state.

Five groups: cameras, incidents, evidence, frames, audit. Every route
authenticates, checks its own permission, and scopes to the caller's tenant and
cameras before it reads anything.

### Scoping is constructed, never filtered afterwards

Every query is built already narrowed. There is no moment at which another
tenant's rows exist in memory to leak — the same discipline the platform applies
to `Scope`, for the same reason.

`camera_keys is None` means a tenant-wide grant; an **empty tuple means none**
and matches nothing. An empty list must never read as a wildcard.

### Permissions are not implied by one another

`VIEW_EVIDENCE` is not implied by `VIEW_OBSERVATIONS` — one reads that a person
was there, the other looks at their picture. `DELETE_EVIDENCE` is not implied by
`VIEW_EVIDENCE` — one is looking, the other destroys a record that may be needed
to defend a finding. Each route names the exact permission it needs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request, Response

from app.api.administration import _restaurant_in_tenant
from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.authorization.model import AccessDecision, Permission, ScopeBreadth
from app.domain import cameras as camera_domain
from app.domain import evidence as evidence_domain
from app.domain import incidents as incident_domain
from app.domain import observations as observation_fold
from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.audit import to_wire as audit_to_wire
from app.errors import AppError, EvidenceForbiddenError, ValidationError

router = APIRouter(prefix="/api/v1", tags=["product"])


def scope_cameras(access: AccessDecision) -> tuple[str, ...] | None:
    """The caller's camera reach. `None` is tenant-wide, `()` is none."""
    if access.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return None
    return access.cameras.camera_ids


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _roles(access: AccessDecision) -> tuple[str, ...]:
    return tuple(sorted(r.value for r in access.roles))


# ── Cameras ──────────────────────────────────────────────────────────────────


@router.get("/cameras", dependencies=[Depends(requires(Permission.VIEW_CAMERAS))])
async def list_cameras(access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    service = camera_domain.CameraService(session)
    found = await service.list(organization_id=access.tenant_id, camera_keys=scope_cameras(access))
    return {
        "cameras": [camera_domain.to_wire(c) for c in found],
        "enabled": sum(1 for c in found if c.enabled),
        # Stated so an operator can see that 14 of 16 channels are deliberately
        # not being processed, rather than wondering why they are missing.
        "total": len(found),
    }


async def _placement(
    session, organization_id: str, restaurant_id: str, zone_id: Any
) -> str | None:
    """Validate where a camera is being put, and refuse anything else.

    Two checks, and the second is the one that was missing.

    **The site belongs to this organization.** Already enforced; a camera could
    not be attached to another customer's restaurant by naming its id.

    **The zone belongs to that site.** Not enforced anywhere until now. The
    route accepted `zone_id` from the request body and passed it to the domain
    service unread, so a caller could attach a camera to any zone in the
    database — including another organization's — and every observation it
    produced from then on would be attributed there. The join through
    `Restaurant` is what makes this checkable at all: `zones` carries no
    organization column of its own, so the parent is the only thing that can
    establish tenancy.

    A frontend that only offers zones from the selected site is a convenience.
    This is the boundary.
    """
    from sqlalchemy import select

    from app.domain.models import Restaurant, Zone

    await _restaurant_in_tenant(session, organization_id, restaurant_id)

    key = str(zone_id or "").strip()
    if not key:
        # A camera with no zone is legitimate: it has been placed at a site but
        # not yet in a room. `record_assignment` opens an interval with a null
        # zone, so "where was this camera" still has an answer.
        return None

    found = (
        await session.execute(
            select(Zone)
            .join(Restaurant, Restaurant.id == Zone.restaurant_id)
            .where(
                Zone.id == key,
                Zone.restaurant_id == restaurant_id,
                Restaurant.organization_id == organization_id,
            )
        )
    ).scalar_one_or_none()
    if found is None:
        # Deliberately does not distinguish "no such zone" from "that zone is
        # somewhere else": both answers would confirm something about a row the
        # caller may not see.
        raise ValidationError(
            "that zone is not part of the selected site",
            details={"zone_id": key, "restaurant_id": restaurant_id},
        )
    return key


@router.post("/cameras", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))])
async def create_camera(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Register a camera. Created **disabled** — enabling is a separate act."""
    service = camera_domain.CameraService(session)
    audit = AuditTrail(session)

    restaurant_id = str(payload.get("restaurant_id", ""))
    zone_id = await _placement(session, access.tenant_id, restaurant_id, payload.get("zone_id"))

    host = str(payload.get("host", "") or "").strip()
    if not host:
        # Required, because the runtime filters on it: `_start_cameras_from_
        # database` and `_start_camera_wall` both skip rows with no host. A
        # camera created without one is accepted, listed, and permanently
        # inert — it never connects and nothing ever says why. Refusing here
        # is the difference between a validation error and a camera that
        # silently does not exist.
        raise ValidationError(
            "'host' is required: a camera with no address cannot be dialled, "
            "and the runtime skips rows without one, so it would be created "
            "and then never connect"
        )

    camera = await service.create(
        organization_id=access.tenant_id,
        restaurant_id=restaurant_id,
        camera_key=str(payload.get("camera_key", "")).strip(),
        name=str(payload.get("name", "")),
        channel=int(payload.get("channel", 1)),
        host=host,
        rtsp_port=int(payload.get("rtsp_port", 554)),
        stream_type=str(payload.get("stream_type", "sub")),
        username=str(payload.get("username", "")),
        credential_ref=str(payload.get("credential_ref", "")),
        analysis_fps=float(payload.get("analysis_fps", 4.0)),
        purpose=str(payload.get("purpose", "")),
        zone_id=zone_id,
        enabled=False,
        # Recorded on the zone interval this creates. An attribution nobody can
        # trace is worth less than one that is wrong and known to be.
        assigned_by=access.subject,
    )
    await audit.record(
        action=AuditAction.CAMERA_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="camera",
        resource_id=camera.camera_key,
        request_id=_request_id(request),
        # The scheme, not the reference. `credential_ref` is now scrubbed by
        # name anyway (`app.domain.audit._FORBIDDEN_KEYS`), so recording it
        # would store `***` and lose the one useful fact — that this camera
        # points at an environment variable rather than a file.
        detail={
            "channel": camera.channel,
            "credential_scheme": camera_domain._credential_scheme(camera.credential_ref),
        },
    )
    return camera_domain.to_wire(camera)


@router.patch("/cameras/{camera_key}", dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))])
async def update_camera(
    camera_key: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    service = camera_domain.CameraService(session)
    audit = AuditTrail(session)

    existing = await service.get(organization_id=access.tenant_id, camera_key=camera_key)
    was_enabled = existing.enabled

    # Placement is re-validated on every update that touches it, not only on
    # create. `update` passes `**payload` into the domain service, so before
    # this a camera could be moved into another organization's zone by a PATCH
    # even though it could not be created in one.
    if "restaurant_id" in payload or "zone_id" in payload:
        restaurant_id = str(payload.get("restaurant_id") or existing.restaurant_id)
        payload = dict(payload)
        payload["restaurant_id"] = restaurant_id
        payload["zone_id"] = await _placement(
            session,
            access.tenant_id,
            restaurant_id,
            payload.get("zone_id", existing.zone_id),
        )

    camera = await service.update(
        organization_id=access.tenant_id,
        camera_key=camera_key,
        assigned_by=access.subject,
        **payload,
    )

    # Enabling a camera starts processing video of people. It gets its own audit
    # action rather than hiding inside a generic "updated".
    if payload.get("enabled") is not None and camera.enabled != was_enabled:
        action = AuditAction.CAMERA_ENABLED if camera.enabled else AuditAction.CAMERA_DISABLED
        await audit.record(
            action=action,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="camera",
            resource_id=camera.camera_key,
            request_id=_request_id(request),
        )
    else:
        await audit.record(
            action=AuditAction.CAMERA_UPDATED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="camera",
            resource_id=camera.camera_key,
            request_id=_request_id(request),
            detail={"fields": sorted(payload.keys())},
        )

    return camera_domain.to_wire(camera)


@router.post(
    "/cameras/test-connection",
    dependencies=[Depends(requires(Permission.MANAGE_CAMERAS))],
)
async def test_camera_connection(
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(default_factory=dict)],
) -> dict[str, Any]:
    """Is anything listening at this address?

    Two ways to ask, and the first is the one to prefer:

    * `{"camera_key": "cam-01"}` — the address comes from the database, is
      already tenant-scoped, and the caller cannot choose it. No part of the
      request influences where the socket goes.
    * `{"host": ..., "rtsp_port": ...}` — for the onboarding wizard, which
      needs to check an address *before* the row exists. Gated on
      `MANAGE_CAMERAS` and refused for reserved ranges; see
      `app.domain.connectivity` for why that is sufficient.

    Never returns a credential, and never resolves one: the test is a TCP
    connect, so there is no authentication step for a secret to be needed by.
    """
    from app.domain import connectivity

    camera_key = str(payload.get("camera_key", "") or "").strip()
    if camera_key:
        camera = await camera_domain.CameraService(session).get(
            organization_id=access.tenant_id, camera_key=camera_key
        )
        host, port = camera.host, camera.rtsp_port
        if not host:
            raise ValidationError(
                "this camera has no address, so there is nothing to test. A "
                "camera without a host is skipped by the runtime and will "
                "never connect."
            )
    else:
        host = str(payload.get("host", "") or "").strip()
        port = int(payload.get("rtsp_port", 554))

    result = await connectivity.probe(host, port)
    body = result.as_dict()
    # Echoed so a wizard can show what it tested. The host is something the
    # caller either supplied or is already entitled to read.
    body["host"] = host
    body["rtsp_port"] = port
    return body


@router.delete(
    "/cameras/{camera_key}",
    dependencies=[Depends(requires(Permission.RETIRE_CAMERAS))],
)
async def delete_camera(
    camera_key: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Remove a camera and destroy its observation partition.

    ### Its own permission, not `MANAGE_CAMERAS`

    Gated on `RETIRE_CAMERAS`. Everything this docstring goes on to say about
    destroying records is the argument for the separation: an account that may
    add a camera and correct its name has not thereby been trusted to close an
    observation partition, and no amount of renaming undoes this.

    ### Two audit rows, always

    Deleting a camera is a configuration change *and* a destruction of records
    about people at work, so it writes both `camera.deleted` and
    `observation.truncated`. A deletion that left only the first would be a
    record of the change with no record of the data it destroyed — and the
    observation sweep's own discipline is that every truncation is provable.

    ### The refusal is the point

    If this process cannot reach a durable observation log, the camera is not
    deleted and the caller is told why. Deleting it there would orphan the
    partition: retention enumerates partitions from the camera table, so a
    deleted row means nothing ever sweeps those observations again.
    """
    settings = settings_of(request)
    service = camera_domain.CameraService(session)
    audit = AuditTrail(session)

    # Reached through the composition rather than rebuilt: a second
    # `FileObservationLog` over the same directory would be a second writer to
    # an append-only store, and the purge must reach the log the pipeline is
    # actually appending to.
    from app.main import _observation_log_of

    log = _observation_log_of(request.app)
    durable = settings.observation_log == "file"

    try:
        removed = await service.retire(
            organization_id=access.tenant_id,
            camera_key=camera_key,
            retired_by=access.subject,
            observation_log=log,
            durable_log=durable,
        )
    except AppError as exc:
        # A refused deletion is still an attempt to destroy a camera's records,
        # and it is recorded with the same weight as a success — the pattern
        # evidence retrieval already follows.
        await audit.record(
            action=AuditAction.CAMERA_DELETED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="camera",
            resource_id=camera_key,
            outcome=AuditOutcome.DENIED,
            request_id=_request_id(request),
            detail={"reason": type(exc).__name__},
        )
        await session.commit()
        raise

    await audit.record(
        action=AuditAction.CAMERA_DELETED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="camera",
        resource_id=camera_key,
        request_id=_request_id(request),
        detail={"observations_removed": removed, "durable_log": durable},
    )
    # The same row the retention sweep writes, for the same reason: a deletion
    # of observations that leaves no trace of having happened cannot be shown to
    # have happened.
    await audit.record(
        action=AuditAction.OBSERVATIONS_TRUNCATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="observation",
        resource_id=camera_key,
        request_id=_request_id(request),
        detail={"reason": "camera deleted", "removed": removed},
    )

    return {
        "deleted": camera_key,
        "observations_removed": removed,
        # Stated rather than implied: the zone history survives on purpose,
        # because incidents and evidence still name this camera_key and still
        # need to say where they happened.
        "zone_history_retained": True,
    }


# ── Incidents ────────────────────────────────────────────────────────────────


@router.get("/incidents", dependencies=[Depends(requires(Permission.VIEW_INCIDENTS))])
async def list_incidents(
    access: CurrentAccess,
    session: DbSession,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    service = incident_domain.IncidentService(session)
    found = await service.list(
        organization_id=access.tenant_id,
        status=status,
        camera_keys=scope_cameras(access),
        limit=limit,
    )
    return {
        "incidents": [incident_domain.to_wire(i) for i in found],
        "count": len(found),
    }


@router.get(
    "/incidents/{incident_id}",
    dependencies=[Depends(requires(Permission.VIEW_INCIDENTS))],
)
async def get_incident(
    incident_id: str, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    service = incident_domain.IncidentService(session)
    incident = await service.get(organization_id=access.tenant_id, incident_id=incident_id)
    return incident_domain.to_wire(incident)


@router.post(
    "/incidents/{incident_id}/acknowledge",
    dependencies=[Depends(requires(Permission.ACKNOWLEDGE_INCIDENTS))],
)
async def acknowledge_incident(
    incident_id: str, request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    service = incident_domain.IncidentService(session)
    audit = AuditTrail(session)

    incident = await service.acknowledge(
        organization_id=access.tenant_id, incident_id=incident_id, actor=access.subject
    )
    await audit.record(
        action=AuditAction.INCIDENT_ACKNOWLEDGED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="incident",
        resource_id=incident.id,
        request_id=_request_id(request),
    )
    return incident_domain.to_wire(incident)


@router.post(
    "/incidents/{incident_id}/resolve",
    dependencies=[Depends(requires(Permission.RESOLVE_INCIDENTS))],
)
async def resolve_incident(
    incident_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(default_factory=dict)],
) -> dict[str, Any]:
    """Close an incident by explicit operator action.

    `kind` is fixed to `operator` here: this route *is* the operator action. An
    observation-driven resolution comes from the pipeline, not from an HTTP
    request, and letting a client claim `kind=observation` would let the UI
    assert something only the platform can know.
    """
    service = incident_domain.IncidentService(session)
    audit = AuditTrail(session)

    incident = await service.resolve(
        organization_id=access.tenant_id,
        incident_id=incident_id,
        actor=access.subject,
        kind="operator",
        note=str(payload.get("note", "")),
    )
    await audit.record(
        action=AuditAction.INCIDENT_RESOLVED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="incident",
        resource_id=incident.id,
        request_id=_request_id(request),
        detail={"kind": "operator"},
    )
    return incident_domain.to_wire(incident)


# ── Evidence ─────────────────────────────────────────────────────────────────


@router.get(
    "/evidence/{evidence_ref}",
    dependencies=[Depends(requires(Permission.VIEW_EVIDENCE))],
)
async def get_evidence_metadata(
    evidence_ref: str, access: CurrentAccess, session: DbSession, request: Request
) -> dict[str, Any]:
    """Evidence **metadata**. No bytes, no path, no storage credential."""
    store = evidence_domain.EvidenceStore(session, root=settings_of(request).evidence_path)
    record = await store.metadata(organization_id=access.tenant_id, evidence_ref=evidence_ref)
    return evidence_domain.to_wire(record)


@router.get(
    "/evidence/{evidence_ref}/image",
    dependencies=[Depends(requires(Permission.VIEW_EVIDENCE))],
)
async def get_evidence_image(
    evidence_ref: str, request: Request, access: CurrentAccess, session: DbSession
) -> Response:
    """The imagery itself. **Every call leaves an audit row.**

    Two gates beyond the permission: `ALLOW_EVIDENCE` is a deployment decision
    and defaults off, and the record's own lifecycle refuses expired or deleted
    evidence even when the permission is held.
    """
    from app.infrastructure.observability import EVIDENCE_ACCESS

    settings = settings_of(request)
    audit = AuditTrail(session)

    if not settings.allow_evidence:
        EVIDENCE_ACCESS.labels("deployment_disabled").inc()
        await audit.record(
            action=AuditAction.EVIDENCE_DENIED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="evidence",
            resource_id=evidence_ref,
            outcome=AuditOutcome.DENIED,
            request_id=_request_id(request),
            detail={"reason": "ALLOW_EVIDENCE is false"},
        )
        # Committed *before* raising. The session rolls back on the way out, and
        # a refused attempt to view CCTV imagery is precisely the row an
        # investigation needs — losing it because the request failed would be
        # exactly backwards.
        await session.commit()
        raise EvidenceForbiddenError(
            "evidence retrieval is disabled for this deployment",
            details={"setting": "ALLOW_EVIDENCE"},
        )

    store = evidence_domain.EvidenceStore(session, root=settings.evidence_path)
    try:
        record, payload = await store.fetch(
            organization_id=access.tenant_id, evidence_ref=evidence_ref
        )
    except AppError as exc:
        # Expired, deleted, missing or corrupt. A refusal is still an attempt to
        # view imagery of an identifiable person, and it is recorded with the
        # same weight as a success.
        EVIDENCE_ACCESS.labels("refused").inc()
        await audit.record(
            action=AuditAction.EVIDENCE_DENIED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="evidence",
            resource_id=evidence_ref,
            outcome=AuditOutcome.DENIED,
            request_id=_request_id(request),
            detail={"reason": type(exc).__name__},
        )
        await session.commit()
        raise

    EVIDENCE_ACCESS.labels("served").inc()
    await audit.record(
        action=AuditAction.EVIDENCE_READ,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="evidence",
        resource_id=evidence_ref,
        request_id=_request_id(request),
        # Size and camera, never the bytes (§24).
        detail={"camera_key": record.camera_key, "size_bytes": record.size_bytes},
    )

    return Response(
        content=payload,
        media_type=record.media_type,
        headers={
            # Never cached. A cached frame of an identifiable person outlives
            # the retention policy that governs it.
            "Cache-Control": "no-store, private",
        },
    )


@router.delete(
    "/evidence/{evidence_ref}",
    dependencies=[Depends(requires(Permission.DELETE_EVIDENCE))],
)
async def delete_evidence(
    evidence_ref: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(default_factory=dict)],
) -> dict[str, Any]:
    """Erase the bytes, keep the tombstone.

    A reason is required. Deletion without one cannot be evidenced afterwards,
    and an erasure request that cannot be evidenced has not been answered.
    """
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValidationError("a deletion reason is required")

    store = evidence_domain.EvidenceStore(session, root=settings_of(request).evidence_path)
    audit = AuditTrail(session)

    record = await store.delete(
        organization_id=access.tenant_id,
        evidence_ref=evidence_ref,
        actor=access.subject,
        reason=reason,
    )
    await audit.record(
        action=AuditAction.EVIDENCE_DELETED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="evidence",
        resource_id=evidence_ref,
        request_id=_request_id(request),
        detail={"reason": reason},
    )
    return evidence_domain.to_wire(record)


# ── Frames ───────────────────────────────────────────────────────────────────


@router.get(
    "/cameras/{camera_key}/frames",
    dependencies=[Depends(requires(Permission.VIEW_OBSERVATIONS))],
)
async def list_frames(
    camera_key: str,
    access: CurrentAccess,
    session: DbSession,
    since: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Frame metadata for one camera. **Metadata only** — no pixels."""
    allowed = scope_cameras(access)
    if allowed is not None and camera_key not in allowed:
        # Reported as empty rather than forbidden: the caller is not granted this
        # camera, and confirming it exists would be a disclosure.
        return {"camera_key": camera_key, "frames": [], "count": 0}

    service = camera_domain.FrameService(session)
    found = await service.list(
        organization_id=access.tenant_id,
        camera_key=camera_key,
        since=_parse_time(since),
        limit=limit,
    )
    return {
        "camera_key": camera_key,
        "frames": [camera_domain.frame_to_wire(f) for f in found],
        "count": len(found),
    }


# ── Audit ────────────────────────────────────────────────────────────────────


@router.get("/audit", dependencies=[Depends(requires(Permission.VIEW_AUDIT))])
async def list_audit(
    access: CurrentAccess,
    session: DbSession,
    actor: Annotated[str | None, Query()] = None,
    resource_type: Annotated[str | None, Query()] = None,
    resource_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """The audit trail, tenant-scoped.

    Its own permission. Not implied by administration — knowing who looked at
    imagery of a named employee is its own kind of access.
    """
    trail = AuditTrail(session)
    events = await trail.query(
        organization_id=access.tenant_id,
        actor=actor,
        resource_type=resource_type,
        resource_id=resource_id,
        limit=limit,
    )
    return {"events": [audit_to_wire(e) for e in events], "count": len(events)}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"'{value}' is not an ISO-8601 timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ── Observations ─────────────────────────────────────────────────────────────
#
# The product's read of what the cameras actually saw, and the first non-DevTools
# surface for it.
#
# **The application stores none of this.** Every value is read from Vision OS's
# own observation log through its Observation API; there is no application table
# holding a perception result, and `app/domain/models.py` explains at length why
# there must never be one. This route is a projection for a screen, not a second
# source of truth.
#
# Two gates, both real. `VIEW_OBSERVATIONS` is checked here, on the route, before
# anything is read. The platform then checks `READ_OBSERVATIONS` against the
# principal's tenant and narrows the scope itself. Neither substitutes for the
# other, and the camera scope handed to the platform is built from the caller's
# grant rather than from anything in the request.


def _observation_window(
    since: datetime | None, until: datetime | None
) -> tuple[datetime, datetime]:
    """The query window, defaulted to the last 24 hours and validated.

    Bounded here as well as by the platform: `query_observations` raises
    `WindowTooLargeError` past its own policy limit, and a route that let a
    caller ask for a decade would turn that into a 500 rather than a clear answer.
    """
    end = until or datetime.now(UTC)
    start = since or (end - timedelta(hours=24))
    if end < start:
        raise ValidationError("'since' must not be later than 'until'")
    return start, end


@router.get(
    "/observations",
    dependencies=[Depends(requires(Permission.VIEW_OBSERVATIONS))],
)
async def list_observations(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    since: Annotated[str | None, Query()] = None,
    until: Annotated[str | None, Query()] = None,
    camera_key: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    """PPE observation history for the caller's cameras, grouped by subject.

    The fold itself lives in `app/domain/observations.py` — one implementation,
    shared with the reporting engine, because it is the single place where
    `not_visible` could be collapsed into `none` and two copies would mean two
    places to get that wrong.

    ### Why grouped by subject rather than returned as a flat stream

    The log holds one observation per attribute per capture. A screen asking
    "who was observed, and what were they wearing" wants those assembled back
    into a subject, with the **most recent** value for each attribute — which is
    what `latest` means here and why each attribute carries its own timestamp
    rather than the group's.

    ### Availability is not emptiness

    A platform that is not assembled returns `available: false` with a reason,
    never an empty list. "No subject was observed" and "nothing was watching"
    are different facts, and a hygiene screen that renders them identically is
    the failure this product is built to avoid.
    """
    from app.api.dependencies import vision_of

    start, end = _observation_window(_parse_time(since), _parse_time(until))
    cameras = scope_cameras(access)

    # An explicit empty grant matches nothing. It must never read as a wildcard,
    # and it is not the same answer as "the platform is down".
    if cameras is not None and not cameras:
        return {
            "available": True,
            "reason": "",
            "subjects": [],
            "count": 0,
            "cameras_queried": [],
            "window": {"since": start.isoformat(), "until": end.isoformat()},
            "window_fully_observable": True,
        }

    vision = vision_of(request)
    composition = getattr(vision, "composition", None)
    exposure = getattr(composition, "exposure", None) if composition else None
    if exposure is None or getattr(exposure, "api", None) is None:
        return {
            "available": False,
            # The platform's own words when it has them, so an operator is told
            # which of the many ways this can be unassembled actually happened.
            "reason": getattr(vision, "reason", "")
            or "Vision OS is not assembled in this process, so no observation can be read.",
            "subjects": [],
            "count": 0,
            "cameras_queried": [],
            "window": {"since": start.isoformat(), "until": end.isoformat()},
            "window_fully_observable": False,
        }

    # A tenant-wide grant still needs a concrete camera list for `Scope`. Read
    # from the durable camera table rather than from live sessions: a camera
    # that is configured but not currently streaming still has history worth
    # returning, and reading live sessions would silently hide it.
    if cameras is None:
        registered = await camera_domain.CameraService(session).list(
            organization_id=access.tenant_id, camera_keys=None
        )
        cameras = tuple(c.camera_key for c in registered)
    if camera_key:
        if camera_key not in cameras:
            raise ValidationError(f"camera '{camera_key}' is not within your access")
        cameras = (camera_key,)

    subjects, page_count, fully_observable = observation_fold.query_observations(
        exposure.api, access, cameras, start, end, limit
    )
    await _attribute_zones(session, access.tenant_id, cameras, subjects)

    trail = AuditTrail(session)
    await trail.record(
        action=AuditAction.OBSERVATIONS_READ,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="observation",
        # A window and a count, never an attribute value. An audit row that
        # copied the observations would become a second, unretained store of
        # exactly the record it exists to govern.
        resource_id=",".join(cameras)[:255],
        request_id=_request_id(request),
        detail={
            "since": start.isoformat(),
            "until": end.isoformat(),
            "cameras": len(cameras),
            "subjects": len(subjects),
            "observations": page_count,
        },
    )

    return {
        "available": True,
        "reason": "",
        "subjects": subjects,
        "count": len(subjects),
        "observation_count": page_count,
        "cameras_queried": list(cameras),
        "window": {"since": start.isoformat(), "until": end.isoformat()},
        "window_fully_observable": fully_observable,
    }


async def _attribute_zones(
    session: Any,
    organization_id: str,
    cameras: tuple[str, ...],
    subjects: list[dict[str, Any]],
) -> None:
    """Attach the zone each subject was observed in, **as it was then**.

    ### Why this is not `join cameras on camera_key`

    `cameras.zone_id` says where a camera is *now*. Reading it onto a past
    observation would mean that moving a camera from the prep line to the wash
    station silently relocates every reading it has ever produced — a quarter of
    prep-line history rewritten by one dropdown, with nothing in the record
    showing it happened. That is the same class of error `finding_snapshot`
    exists to prevent, and it is why `camera_zone_assignments` records the
    mapping as closed intervals instead.

    ### `zone_recorded: false` is a real answer

    Intervals begin the first time a camera's zone was written after that table
    existed. An observation older than a camera's first interval has no recorded
    zone, and this reports that rather than inferring one from today's mapping —
    inferring would commit the exact error being avoided. Nothing is backfilled.

    Resolved at `last_seen`, which is the instant the row's attribute values are
    current as of, so the zone shown and the readings shown describe the same
    moment.
    """
    if not subjects:
        return

    from app.domain.zone_attribution import ZoneHistory

    history = await ZoneHistory.load(
        session, organization_id=organization_id, camera_keys=cameras
    )
    for subject in subjects:
        attribution = history.resolve_ns(
            str(subject.get("camera_key", "")), subject.get("last_seen")
        )
        subject["zone_id"] = attribution.zone_id if attribution else None
        subject["zone_name"] = attribution.zone_name if attribution else ""
        # Distinguishes "recorded as belonging to no zone" from "nobody wrote it
        # down". Both render as no zone; only one of them is a gap somebody can
        # close, and a client that could not tell them apart would report the
        # second as the first.
        subject["zone_recorded"] = attribution is not None


__all__ = ["router", "scope_cameras"]
