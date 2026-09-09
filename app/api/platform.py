"""The platform operator API — organizations, and their lifecycle.

Every route here is gated on `current_operator`, which resolves a
`PlatformOperator` and refuses anything else. There is no `requires(...)` in
this module and no `CurrentAccess`, because a tenant `Permission` cannot
authorize a cross-tenant act — that is the whole reason the operator is a
separate principal (`app.authorization.platform`).

### The boundary, stated as a rule

A route in this module never reads a `Permission`. A route outside it never
sees a `PlatformOperator`. Nothing translates between them. An organization
administrator cannot reach these routes by acquiring any role, because no role
produces the type they require.

`POST /organizations/{id}/enter` is the one deliberate crossing, and it does
not weaken the rule — it *is* the rule made explicit. It reads no permission
either; it mints a separate, audited, read-only tenant session and hands it
back, so that reaching a customer's data is a recorded act with its own
credential rather than something an operator token could do quietly.

### Lifecycle is not a UI state

`status` reaches authentication, authorization and the camera runtime:

* `ACTIVE`     — unchanged.
* `SUSPENDED`  — login and reads continue; writes are refused
                 (`resolver._suspend`); cameras stop.
* `ARCHIVED`   — login refused; cameras stop.

The camera consequences happen here, synchronously, in the same request that
changes the status. A lifecycle that only took effect at the next process
restart would mean an archived customer's cameras kept running — and kept
costing money and kept recording people — for as long as the process happened
to live.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request, Response
from loguru import logger
from sqlalchemy import func, select

from app.api.dependencies import CurrentOperator, DbSession, auth_of, settings_of
from app.auth.cookies import set_refresh_cookie
from app.authorization.model import OrganizationStatus
from app.authorization.platform import PlatformOperator, entry_decision
from app.domain.audit import AuditAction, AuditTrail
from app.domain.models import Camera, Restaurant
from app.domain.runtime_identity import validate_organization_id
from app.errors import ConflictError, NotFoundError, ValidationError
from app.users.models import Organization, User

router = APIRouter(prefix="/api/v1/platform", tags=["platform"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _slugify(name: str) -> str:
    kept = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    slug = "".join(kept)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:128]


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def organization_to_wire(
    organization: Organization,
    *,
    site_count: int = 0,
    camera_count: int = 0,
    user_count: int = 0,
    running_cameras: int = 0,
) -> dict[str, Any]:
    """One organization, with the counts an operator console actually reads.

    Counts rather than nested collections: a list of customers wants to know
    one has four sites, not to carry all four on every row.
    """
    status = str(organization.status or "").strip().lower()
    return {
        "id": organization.id,
        "name": organization.name,
        "slug": organization.slug,
        "status": status,
        "status_changed_at": _iso(organization.status_changed_at),
        "status_reason": organization.status_reason or "",
        "created_at": _iso(organization.created_at),
        "site_count": site_count,
        "camera_count": camera_count,
        "user_count": user_count,
        # What the runtime is *actually* doing, as distinct from what the
        # configuration says it should. An operator looking at a suspended
        # customer needs to see that their cameras have in fact stopped, not
        # merely that the status field says they ought to have.
        "running_cameras": running_cameras,
    }


async def _counts(session: DbSession) -> dict[str, dict[str, int]]:
    """Site, camera and user counts for every organization, in three queries."""
    out: dict[str, dict[str, int]] = {}

    async def tally(statement, key: str) -> None:
        for organization_id, count in (await session.execute(statement)).all():
            out.setdefault(organization_id, {})[key] = int(count)

    await tally(
        select(Restaurant.organization_id, func.count()).group_by(Restaurant.organization_id),
        "site_count",
    )
    await tally(
        select(Camera.organization_id, func.count()).group_by(Camera.organization_id),
        "camera_count",
    )
    await tally(
        select(User.organization_id, func.count()).group_by(User.organization_id),
        "user_count",
    )
    return out


def _running(request: Request, organization_id: str) -> int:
    """How many of this organization's cameras this process is streaming.

    Read from the wall's own registry rather than from the camera table, so it
    reports reality. The registry is keyed on runtime ids, whose tenant half is
    exactly what makes this countable per organization at all.
    """
    from app.domain.runtime_identity import SEPARATOR

    wall = getattr(request.app.state, "wall", None)
    if wall is None:
        return 0
    prefix = f"{organization_id}{SEPARATOR}"
    return sum(1 for key in wall.streams if key.startswith(prefix))


# ── organizations ────────────────────────────────────────────────────────────


@router.get("/organizations")
async def list_organizations(
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    q: Annotated[str | None, Query(max_length=200)] = None,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Every organization on the platform.

    Paginated and searchable from the start rather than when it becomes a
    problem: this list grows with the customer base, and a console that fetches
    all of them works perfectly until the day it does not.
    """
    statement = select(Organization)
    if q:
        needle = f"%{q.strip().lower()}%"
        statement = statement.where(
            func.lower(Organization.name).like(needle)
            | func.lower(Organization.slug).like(needle)
        )
    if status:
        statement = statement.where(Organization.status == status.strip().lower())

    total = int(
        (
            await session.execute(
                select(func.count()).select_from(statement.subquery())
            )
        ).scalar_one()
    )
    rows = (
        (
            await session.execute(
                statement.order_by(Organization.name).limit(limit).offset(offset)
            )
        )
        .scalars()
        .all()
    )

    counts = await _counts(session)
    return {
        "organizations": [
            organization_to_wire(
                organization,
                running_cameras=_running(request, organization.id),
                **counts.get(organization.id, {}),
            )
            for organization in rows
        ],
        "count": len(rows),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/organizations/{organization_id}")
async def get_organization(
    organization_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
) -> dict[str, Any]:
    organization = await _organization(session, organization_id)
    counts = await _counts(session)
    return organization_to_wire(
        organization,
        running_cameras=_running(request, organization.id),
        **counts.get(organization.id, {}),
    )


@router.post("/organizations")
async def create_organization(
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Create a tenant.

    The id is the slug, not a UUID, and that is deliberate. It becomes the
    tenant half of every camera's runtime identity
    (`app.domain.runtime_identity`), it appears in observation partition
    filenames, and it is what an engineer reads in a log line at three in the
    morning. `org-acme:cam-01` says something; a hex id does not.

    It is therefore also validated against that module's charset here, at the
    only place an organization id is ever minted — so an id that would break
    the runtime identity round trip cannot enter the database at all.
    """
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise ValidationError("'name' is required")

    slug = _slugify(str(payload.get("slug", "") or "") or name)
    if not slug:
        raise ValidationError("'name' must contain at least one alphanumeric character")

    # `org-` prefixed so an organization id is recognisable on sight wherever
    # it turns up composed into something else.
    organization_id = validate_organization_id(f"org-{slug}")

    existing = (
        await session.execute(
            select(Organization).where(
                (Organization.id == organization_id) | (Organization.slug == slug)
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError(f"an organization with slug '{slug}' already exists")

    organization = Organization(
        id=organization_id,
        name=name,
        slug=slug,
        # Every organization starts ACTIVE. There is no "provisioning" state,
        # because there is nothing asynchronous to wait for: a tenant with no
        # sites and no cameras is already coherent.
        status=OrganizationStatus.ACTIVE.value,
        is_active=True,
    )
    session.add(organization)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_CREATED,
        organization_id=organization.id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="organization",
        resource_id=organization.id,
        request_id=_request_id(request),
        detail={"name": name, "slug": slug},
    )
    return organization_to_wire(organization)


@router.patch("/organizations/{organization_id}")
async def update_organization(
    organization_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Rename an organization. Not its slug, and not its status.

    `slug` is immutable for the same reason a restaurant's is, only more so:
    it is the organization id, it is half of every camera's runtime identity,
    and it names observation partitions on disk. Changing it would orphan
    every one of them.

    Status has its own route below, because a lifecycle change is a different
    kind of act from a rename and must not be reachable by a client that meant
    to correct a typo.
    """
    organization = await _organization(session, organization_id)

    changed: list[str] = []
    if "name" in payload:
        name = str(payload["name"] or "").strip()
        if not name:
            raise ValidationError("'name' must not be empty")
        organization.name = name
        changed.append("name")

    if not changed:
        return await get_organization(organization_id, request, operator, session)

    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_UPDATED,
        organization_id=organization.id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="organization",
        resource_id=organization.id,
        request_id=_request_id(request),
        detail={"fields": sorted(changed)},
    )
    return await get_organization(organization_id, request, operator, session)


@router.put("/organizations/{organization_id}/status")
async def set_organization_status(
    organization_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Move an organization through its lifecycle, and make the runtime obey.

    ### The reason is required for the two narrowing states

    Suspension and archival stop a paying customer's product working. "Why"
    is the first question that produces, and requiring it at the moment of
    the act is the only time anyone actually knows the answer.

    ### The runtime consequence is synchronous

    Cameras are stopped in this request, not at the next restart. An archived
    organization whose cameras kept running would keep costing money and keep
    recording identifiable people after the deployment had formally ended —
    which is a data-protection failure, not a billing one.

    Restoring to ACTIVE deliberately does **not** restart cameras here. The
    bootstrap supervisor owns starting sessions and knows how to do it safely;
    duplicating that inside a status route would be a second, subtly different
    implementation of the most failure-prone operation in the application.
    The response says so rather than leaving the operator to wonder.
    """
    organization = await _organization(session, organization_id)

    try:
        target = OrganizationStatus(str(payload.get("status", "")).strip().lower())
    except ValueError as exc:
        raise ValidationError(
            "'status' must be 'active', 'suspended' or 'archived'",
            details={"status": payload.get("status")},
        ) from exc

    reason = str(payload.get("reason", "") or "").strip()
    if target is not OrganizationStatus.ACTIVE and not reason:
        raise ValidationError(
            f"'reason' is required to move an organization to {target.value}: it "
            f"stops the customer's product working, and the explanation is only "
            f"reliably known at the moment of the act"
        )

    previous = str(organization.status or "").strip().lower()
    if previous == target.value:
        return await get_organization(organization_id, request, operator, session)

    organization.status = target.value
    organization.status_changed_at = datetime.now(UTC)
    organization.status_reason = reason
    # `is_active` predates `status`, and `AuthService.authenticate` still reads
    # it as "may anyone here log in". So it tracks ARCHIVED, not ACTIVE: a
    # SUSPENDED organization keeps working for reads, which is the whole
    # difference between the two states, and clearing this flag on suspension
    # would lock every one of its users out and collapse them back into the
    # single boolean `OrganizationStatus` exists to replace.
    organization.is_active = target is not OrganizationStatus.ARCHIVED

    stopped = 0
    if target is not OrganizationStatus.ACTIVE:
        stopped = await _stop_runtime(request, organization.id)

    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_STATUS_CHANGED,
        organization_id=organization.id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="organization",
        resource_id=organization.id,
        request_id=_request_id(request),
        detail={
            "from": previous,
            "to": target.value,
            "reason": reason,
            "cameras_stopped": stopped,
        },
    )

    body = await get_organization(organization_id, request, operator, session)
    body["cameras_stopped"] = stopped
    if target is OrganizationStatus.ACTIVE:
        body["note"] = (
            "Cameras are not restarted by this request. The bootstrap "
            "supervisor starts sessions and will pick them up; restart the "
            "API process to bring them back immediately."
        )
    return body


async def _stop_runtime(request: Request, organization_id: str) -> int:
    """Stop this organization's camera streams. Never raises.

    A lifecycle change that failed because a camera thread misbehaved would
    leave the organization in a state neither the operator nor the database
    agrees with. The status change is the durable fact; the runtime is brought
    into line with it and any failure is reported rather than propagated.
    """
    wall = getattr(request.app.state, "wall", None)
    if wall is None:
        return 0
    try:
        return int(await wall.stop_organization(organization_id))
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "could not stop cameras for organization {}: {}: {}",
            organization_id,
            type(exc).__name__,
            exc,
        )
        return 0


async def _organization(session: DbSession, organization_id: str) -> Organization:
    found = (
        await session.execute(select(Organization).where(Organization.id == organization_id))
    ).scalar_one_or_none()
    if found is None:
        raise NotFoundError(f"no organization '{organization_id}'")
    return found


@router.post("/organizations/{organization_id}/enter")
async def enter_organization(
    organization_id: str,
    request: Request,
    response: Response,
    operator: CurrentOperator,
    session: DbSession,
) -> dict[str, Any]:
    """Take a read-only session inside one organization.

    ### This is a widening, and it is meant to look like one

    Everything else in this module operates on organizations *as objects* —
    their existence, their names, their lifecycle. None of it reads a customer's
    data, and `PlatformOperator` carries nothing that could. This route is the
    single deliberate exception, and the shape of it is the argument:

    * it is a **POST**, because it is an act rather than a view;
    * it **writes an audit row before it returns**, filed against the customer,
      so the record exists in the tenant whose data is about to be read;
    * it mints a **separate token** carrying `act: platform_operator`, so every
      subsequent request is identifiable as part of this entry rather than
      indistinguishable from the customer's own staff;
    * the reach it grants is **stated, not resolved** — see
      `app.authorization.platform.OPERATOR_ENTRY_PERMISSIONS` — and contains no
      write, no evidence, no patron identity and no audit read.

    The property the previous design had, and which is kept: an operator cannot
    read a tenant's data *through the operator door*. `GET /platform/...` still
    returns counts and never contents. What has changed is that there is now a
    second door, it is locked differently, and going through it is recorded.

    ### Archived organizations are refused

    Login is refused inside an archived organization, and entry is a login by
    another name. An operator who needs to look at an archived customer's data
    is describing a restore, which is a decision rather than a click.
    """
    organization = await _organization(session, organization_id)

    if not organization.is_active or (
        str(organization.status or "").strip().lower() == OrganizationStatus.ARCHIVED.value
    ):
        raise ValidationError(
            "an archived organization cannot be entered: nobody may sign in to "
            "one, and platform authority does not exempt this session from that",
            details={"organization_id": organization_id, "status": organization.status},
        )

    decision = entry_decision(operator, organization.id)
    issued = auth_of(request).issue(decision)

    await AuditTrail(session).record(
        action=AuditAction.PLATFORM_OPERATOR_ENTERED,
        organization_id=organization.id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="organization",
        resource_id=organization.id,
        request_id=_request_id(request),
        detail={
            "home_organization_id": operator.home_organization_id,
            # The reach, written into the row rather than left to be inferred
            # from the code that was deployed at the time. A permission set that
            # changes later must not silently rewrite what an old entry meant.
            "permissions": sorted(p.value for p in decision.permissions),
            "read_only": True,
        },
    )

    set_refresh_cookie(response, issued.refresh_token, settings_of(request))

    return {
        "access_token": issued.access_token,
        "token_type": "bearer",
        "expires_at": issued.expires_at.isoformat(),
        "organization": organization_to_wire(
            organization,
            running_cameras=_running(request, organization.id),
            **(await _counts(session)).get(organization.id, {}),
        ),
        "acting_as": decision.acting_as,
        "read_only": True,
        "permissions": sorted(p.value for p in decision.permissions),
    }


@router.get("/me")
async def whoami(operator: CurrentOperator) -> dict[str, Any]:
    """Who this operator is. No tenant, because an operator has none.

    Exists so the frontend can decide whether to show the platform surface at
    all without probing a real route and reading the 403.
    """
    assert isinstance(operator, PlatformOperator)
    return {
        "subject": operator.subject,
        "display_name": operator.display_name,
        "is_platform_operator": True,
    }


__all__ = ["organization_to_wire", "router"]
