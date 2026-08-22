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

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request, Response

from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.authorization.model import AccessDecision, Permission, ScopeBreadth
from app.domain import cameras as camera_domain
from app.domain import evidence as evidence_domain
from app.domain import incidents as incident_domain
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

    camera = await service.create(
        organization_id=access.tenant_id,
        restaurant_id=str(payload.get("restaurant_id", "")),
        camera_key=str(payload.get("camera_key", "")).strip(),
        name=str(payload.get("name", "")),
        channel=int(payload.get("channel", 1)),
        host=str(payload.get("host", "")),
        rtsp_port=int(payload.get("rtsp_port", 554)),
        stream_type=str(payload.get("stream_type", "sub")),
        username=str(payload.get("username", "")),
        credential_ref=str(payload.get("credential_ref", "")),
        analysis_fps=float(payload.get("analysis_fps", 4.0)),
        purpose=str(payload.get("purpose", "")),
        zone_id=payload.get("zone_id"),
        enabled=False,
    )
    await audit.record(
        action=AuditAction.CAMERA_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="camera",
        resource_id=camera.camera_key,
        request_id=_request_id(request),
        # `credential_ref` is a pointer and safe to record; the scrubber would
        # remove a value even if one were passed by mistake.
        detail={"channel": camera.channel, "credential_ref": camera.credential_ref},
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

    was_enabled = (
        await service.get(organization_id=access.tenant_id, camera_key=camera_key)
    ).enabled

    camera = await service.update(
        organization_id=access.tenant_id, camera_key=camera_key, **payload
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


__all__ = ["router", "scope_cameras"]
