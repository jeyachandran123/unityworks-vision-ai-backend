"""The reporting API. Three routes, and every one of them writes an audit row.

### Why generating a report is an auditable act

Evidence retrieval is audited because looking at imagery of a named employee is
a thing somebody did. A report is the same act at a different resolution: it
assembles what the cameras observed, which incidents were raised about whom, and
which zone each happened in, into one document somebody can keep. Aggregation
does not make that harmless — a hygiene report for one zone over one week is a
statement about a small, identifiable shift team.

So this module follows `get_evidence_image` exactly: the permission is checked
on the route, a **refusal is audited with the same weight as a success**, the
audit row is committed before the error propagates, and the response is
`no-store`.

### Two permissions, because two acts

`VIEW_REPORTS` runs a report and returns JSON for the screen. `EXPORT_REPORTS`
produces a file. They are separate because an exported file leaves this system:
no retention sweep here reaches it, and it outlives every policy this
application enforces.

On top of both, a report type requires the permission for **every source it
reads** — see `app/reporting/catalogue.py`. Reporting must not be the way an
account reads data it is otherwise refused.

### Bounded, not backgrounded

Report generation is synchronous and bounded rather than queued: the window is
capped at 366 days, each source is capped at `row_limit` rows and reports
`truncated` when it hits it, and rendering — which is CPU-bound in reportlab and
openpyxl — runs in a worker thread via `asyncio.to_thread` so it never blocks the
event loop. An unbounded background job with no visibility would be the wrong
trade here: the caller would lose the coverage panel, and a report nobody can
see the progress of is a report nobody trusts.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.authorization.model import AccessDecision, Permission, ScopeBreadth
from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.models import Restaurant
from app.errors import (
    AppError,
    AuthorizationError,
    CapabilityNotConfiguredError,
    NotFoundError,
    ValidationError,
)
from app.reporting import catalogue
from app.reporting.model import (
    DEFAULT_ROW_LIMIT,
    MAX_WINDOW_DAYS,
    ExportFormat,
    Granularity,
    ReportData,
    ReportRequest,
)
from app.reporting.periods import parse_instant, resolve_timezone, resolve_window
from app.reporting.render import format_available, render

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _roles(access: AccessDecision) -> tuple[str, ...]:
    return tuple(sorted(r.value for r in access.roles))


def _scope_cameras(access: AccessDecision) -> tuple[str, ...] | None:
    """`None` is tenant-wide, `()` is none. Never inferred from an empty list."""
    if access.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return None
    return access.cameras.camera_ids


async def _site_timezone(session, organization_id: str, restaurant_id: str) -> tuple[str, str]:
    """The site's own zone, and its name. UTC when no site was named.

    Period boundaries are a local question: "September" for a Singapore kitchen
    begins at 16:00 UTC on 31 August, and a monthly report computed on UTC
    boundaries misattributes eight hours of every month.
    """
    if not restaurant_id:
        return "UTC", ""
    from sqlalchemy import select

    row = (
        await session.execute(
            select(Restaurant.timezone, Restaurant.name).where(
                Restaurant.id == restaurant_id,
                # Tenancy from the session, never from the request body — asking
                # for another organisation's site is a 404, not a 403, because
                # "it exists but is not yours" is itself a disclosure.
                Restaurant.organization_id == organization_id,
            )
        )
    ).first()
    if row is None:
        raise NotFoundError(f"no restaurant '{restaurant_id}'")
    return (row[0] or "UTC"), (row[1] or "")


async def _prepare(
    request: Request,
    access: AccessDecision,
    session,
    *,
    report_id: str,
    since: str | None,
    until: str | None,
    granularity: str,
    restaurant_id: str,
) -> tuple[catalogue.ReportType, ReportRequest, Any]:
    """Resolve and validate everything before a single row is read."""
    report = catalogue.report_for(report_id)
    if report is None:
        raise NotFoundError(f"no report type '{report_id}'")

    try:
        grain = Granularity(granularity)
    except ValueError as exc:
        raise ValidationError(
            f"'{granularity}' is not a granularity",
            details={"supported": [g.value for g in Granularity]},
        ) from exc
    if grain not in report.granularities:
        raise ValidationError(
            f"'{report.id}' does not support {grain.value} granularity",
            details={"supported": [g.value for g in report.granularities]},
        )

    tz_name, _site = await _site_timezone(session, access.tenant_id, restaurant_id)
    zone = resolve_timezone(tz_name)
    start, end = resolve_window(
        since=parse_instant(since), until=parse_instant(until), zone=zone, granularity=grain
    )

    return (
        report,
        ReportRequest(
            report_id=report.id,
            since=start,
            until=end,
            granularity=grain,
            timezone=zone.name,
            timezone_resolved=zone.resolved,
            organization_id=access.tenant_id,
            camera_keys=_scope_cameras(access),
            restaurant_id=restaurant_id,
            row_limit=DEFAULT_ROW_LIMIT,
        ),
        zone,
    )


async def _authorize(
    request: Request,
    access: AccessDecision,
    session,
    report: catalogue.ReportType,
    *,
    action: AuditAction,
) -> None:
    """Every permission the report names, or a refusal that is itself recorded."""
    if catalogue.permitted(report, access):
        return

    missing = [p.value for p in report.permissions if not access.has(p)]
    await AuditTrail(session).record(
        action=AuditAction.REPORT_DENIED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="report",
        resource_id=report.id,
        outcome=AuditOutcome.DENIED,
        request_id=_request_id(request),
        detail={"missing": missing, "attempted": action.value},
    )
    # Committed before raising. The session rolls back on the way out, and a
    # refused attempt to assemble a report about staff is precisely the row an
    # investigation needs — losing it because the request failed would be
    # exactly backwards. Same reasoning as a refused evidence retrieval.
    await session.commit()

    raise AuthorizationError(
        "this account does not hold every permission this report requires",
        details={"report": report.id, "missing": missing},
    )


async def _collect(
    request: Request, access: AccessDecision, session, report, report_request, zone
) -> ReportData:
    from app.api.dependencies import vision_of

    settings = settings_of(request)
    vision = vision_of(request)
    composition = getattr(vision, "composition", None)
    exposure = getattr(composition, "exposure", None) if composition else None
    exposure_api = getattr(exposure, "api", None) if exposure else None

    context = catalogue.CollectContext(
        session=session,
        request=report_request,
        zone=zone,
        access=access,
        exposure_api=exposure_api,
        vision_reason=getattr(vision, "reason", ""),
        observation_retention_days=settings.observation_retention_days,
        incident_retention_days=settings.incident_retention_days,
    )
    return await catalogue.build(report, context)


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/types", dependencies=[Depends(requires(Permission.VIEW_REPORTS))])
async def list_report_types(access: CurrentAccess) -> dict[str, Any]:
    """The catalogue, with what this caller may actually run.

    Every report is listed, including the ones this account cannot run and the
    seven modules that have no data source. A menu that silently omitted them
    would leave an operator unable to tell "this report does not exist" from
    "you may not run it" from "that module is not connected yet" — three
    different answers, and the whole point of this phase is that they stay
    three.
    """
    exports = {
        fmt.value: {"available": available, "reason": reason}
        for fmt, available, reason in (
            (fmt, *format_available(fmt)) for fmt in ExportFormat
        )
    }
    return {
        "reports": [
            report.as_dict(permitted=catalogue.permitted(report, access))
            for report in catalogue.CATALOGUE
        ],
        "count": len(catalogue.CATALOGUE),
        "can_export": access.has(Permission.EXPORT_REPORTS),
        # Which formats this deployment can actually produce. A UI that offered
        # PDF against a deployment without reportlab would be offering a button
        # that fails.
        "formats": exports,
        "max_window_days": MAX_WINDOW_DAYS,
    }


@router.get("/{report_id}", dependencies=[Depends(requires(Permission.VIEW_REPORTS))])
async def generate_report(
    report_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    since: Annotated[str | None, Query()] = None,
    until: Annotated[str | None, Query()] = None,
    granularity: Annotated[str, Query()] = "total",
    restaurant_id: Annotated[str, Query()] = "",
) -> dict[str, Any]:
    """Run a report and return it as JSON. **Audited.**"""
    report, report_request, zone = await _prepare(
        request,
        access,
        session,
        report_id=report_id,
        since=since,
        until=until,
        granularity=granularity,
        restaurant_id=restaurant_id,
    )
    await _authorize(request, access, session, report, action=AuditAction.REPORT_GENERATED)

    data = await _collect(request, access, session, report, report_request, zone)

    await AuditTrail(session).record(
        action=AuditAction.REPORT_GENERATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="report",
        resource_id=report.id,
        request_id=_request_id(request),
        # The window and the shape of the answer, never its contents. An audit
        # row that copied the figures would become a second, unretained store of
        # exactly the record it exists to govern.
        detail={
            "since": report_request.since.isoformat(),
            "until": report_request.until.isoformat(),
            "granularity": report_request.granularity.value,
            "restaurant_id": restaurant_id,
            "complete": data.coverage.complete,
            "sources": [s.source for s in data.coverage.sources],
        },
    )

    return data.as_dict()


@router.get(
    "/{report_id}/export",
    dependencies=[Depends(requires(Permission.EXPORT_REPORTS))],
)
async def export_report(
    report_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    fmt: Annotated[str, Query(alias="format")] = "pdf",
    since: Annotated[str | None, Query()] = None,
    until: Annotated[str | None, Query()] = None,
    granularity: Annotated[str, Query()] = "total",
    restaurant_id: Annotated[str, Query()] = "",
) -> Response:
    """Produce a file. **Audited separately from generation.**

    `GET` rather than `POST` so the browser's own download path can carry the
    Authorization header through `authorizedFetch`, matching how evidence
    imagery is already retrieved. The response is `no-store`, so a report about
    a named shift team is not left in a shared cache.
    """
    try:
        export_format = ExportFormat(fmt.lower())
    except ValueError as exc:
        raise ValidationError(
            f"'{fmt}' is not an export format",
            details={"supported": [f.value for f in ExportFormat]},
        ) from exc

    available, reason = format_available(export_format)
    if not available:
        raise CapabilityNotConfiguredError(reason, details={"format": export_format.value})

    report, report_request, zone = await _prepare(
        request,
        access,
        session,
        report_id=report_id,
        since=since,
        until=until,
        granularity=granularity,
        restaurant_id=restaurant_id,
    )
    await _authorize(request, access, session, report, action=AuditAction.REPORT_EXPORTED)

    try:
        data = await _collect(request, access, session, report, report_request, zone)
        # CPU-bound: reportlab lays out a document and openpyxl serialises a
        # workbook, both synchronously. Off the event loop, bounded by the row
        # cap above rather than by a queue nobody can see.
        rendered = await asyncio.to_thread(render, data, export_format)
    except AppError as exc:
        await AuditTrail(session).record(
            action=AuditAction.REPORT_EXPORTED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles(access),
            resource_type="report",
            resource_id=report.id,
            outcome=AuditOutcome.FAILED,
            request_id=_request_id(request),
            detail={"format": export_format.value, "error": type(exc).__name__},
        )
        await session.commit()
        raise

    await AuditTrail(session).record(
        action=AuditAction.REPORT_EXPORTED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles(access),
        resource_type="report",
        resource_id=report.id,
        request_id=_request_id(request),
        # Format, size and window. Never the contents — the file is the record,
        # and the trail says a copy was taken rather than keeping a second one.
        detail={
            "format": export_format.value,
            "size_bytes": len(rendered.content),
            "since": report_request.since.isoformat(),
            "until": report_request.until.isoformat(),
            "complete": data.coverage.complete,
        },
    )

    return Response(
        content=rendered.content,
        media_type=rendered.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{rendered.filename}"',
            # Never cached. A report naming a small shift team outlives the
            # retention policies that govern the data it was built from, and a
            # cached copy outlives even that.
            "Cache-Control": "no-store, private",
        },
    )


__all__ = ["router"]
