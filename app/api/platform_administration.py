"""The platform control plane — everything above one organization.

A second module on the same router prefix and the same principal as
`app.api.platform`, split from it because the two answer different questions.
`platform.py` owns an organization's *existence and lifecycle*: create it,
rename it, suspend it, enter it. This module owns administering the platform
*across* organizations: who exists, who may enter what, what an operator is, and
what the estate looks like in aggregate.

Since 2026-09-30 it also sets **what a person may do** in each organization —
the access matrix (`GET/PUT /people/{id}/access`, `POST /people`). That is not
a widening: the Platform Admin can already enter any organization with every
permission and do the same from inside it. The matrix writes the same role and
override rows an Organization Admin writes, through the same guarded modules,
and files every change in the organization it changes.

Every route here is gated on `CurrentOperator`, reads no `Permission`, and is
refused outright for any tenant principal. That is the same boundary
`platform.py` documents, and the reason this is a separate file rather than a
separate authorization model.

### What this module deliberately does not do

**It never returns a customer's operational content.** Counts, names,
configuration and membership — never an incident, never an observation, never a
frame, never an evidence record. An operator who needs to see those enters the
organization explicitly through `POST /platform/organizations/{id}/enter`, which
is audited and grants full reach inside that one organization, and reads them
through the ordinary tenant routes with the separate token it issues. Adding a cross-tenant read here would make that entry pointless
and would be the silent surveillance reach the architecture exists to prevent.

**It does not grant platform-operator status.** Listing operators is a read;
minting one stays a command-line act (`scripts/manage.py grant-operator`),
deliberately, for the reason `app.authorization.platform` sets out at length.
Exposing a grant route here would let one operator create another, which makes
the privilege self-propagating.

**It does not edit role definitions.** `GET /roles` reports
`ROLE_PERMISSIONS` as the read-only, code-defined policy it actually is. There
is no table behind it and no write path, and a UI that implied otherwise would
be lying about what the server enforces.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentOperator, DbSession, settings_of
from app.auth.passwords import hash_password
from app.authorization.assignments import (
    AppliedAccess,
    OrganizationAccess,
    access_audit_detail,
    apply_organization_access,
    load_for_access,
    organization_access_to_wire,
    parse_access_items,
)
from app.authorization.model import (
    ROLE_PERMISSIONS,
    OrganizationStatus,
    Permission,
    Role,
)
from app.authorization.platform import PlatformOperator
from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.models import AuditEvent, Camera, Zone
from app.errors import ConflictError, NotFoundError, ScopeError, ValidationError
from app.users.models import (
    AccessGrant,
    Organization,
    OrganizationMembership,
    PlatformOperatorGrant,
    RoleAssignment,
    User,
)

router = APIRouter(prefix="/api/v1/platform", tags=["platform"])


#: The audit actions this console reports as "platform activity".
#:
#: An explicit tuple rather than a prefix match on `organization.`, for the same
#: reason `OPERATOR_ENTRY_PERMISSIONS` is a list: a rule that depends on how an
#: action was spelled is not a rule about what it does, and the next action
#: added would join this feed or not depending on its name.
PLATFORM_ACTIONS: tuple[str, ...] = (
    AuditAction.ORGANIZATION_CREATED.value,
    AuditAction.ORGANIZATION_UPDATED.value,
    AuditAction.ORGANIZATION_STATUS_CHANGED.value,
    AuditAction.PLATFORM_OPERATOR_ENTERED.value,
    AuditAction.ORGANIZATION_MEMBER_ADDED.value,
    AuditAction.ORGANIZATION_MEMBER_REMOVED.value,
    AuditAction.ORGANIZATION_ADMIN_CREATED.value,
    AuditAction.ACCESS_SET.value,
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _iso(value: datetime | None) -> str | None:
    """A timestamp with its offset stated, so a browser in any zone reads it right."""
    return _aware(value).isoformat() if value else None


# ── overview ─────────────────────────────────────────────────────────────────

#: How many days the activity chart covers. A fortnight is long enough to set a
#: week beside the week before it, and short enough that one busy onboarding
#: still reads as a spike rather than as the baseline.
ACTIVITY_DAYS = 14

#: The stream state that means pictures are arriving. Every other state the wall
#: reports — connecting, reconnecting, offline, error, disabled — is a camera
#: the wall holds and nobody can currently see through.
_LIVE = "live"


@router.get("/overview")
async def overview(
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    utc_offset: Annotated[
        int,
        Query(
            ge=-840,
            le=840,
            description=(
                "The reader's offset from UTC, in minutes. Decides where one day "
                "of the activity chart ends and the next begins."
            ),
        ),
    ] = 0,
) -> dict[str, Any]:
    """The platform at a glance.

    ### Every figure here is counted, none is scored

    There is no health percentage, no compliance rate and no trend. Each number
    below is a `COUNT(*)` over a table that exists, and each one answers a
    question that genuinely spans organizations. A figure that could only be
    computed by inventing a weighting would be a figure nobody could act on, and
    the first person to act on it would be acting on our arithmetic rather than
    on their estate.

    Deliberately absent: incident counts, violation rates, compliance scores.
    Those are organization judgements — "that is a hygiene violation" is an
    opinion a consumer forms, and aggregating opinions across unrelated
    customers produces a number with no referent. They belong on a Command
    Center, which is one organization's own reading of its own kitchen.

    ### What a Platform Admin opens this page to find out

    Which customer needs them. So beside the totals it reports every
    organization on one line (`fleet`), and names the three ways a customer goes
    quietly wrong without anybody inside it raising a ticket: onboarding that
    never finished, nobody left who can administer it, and cameras that are
    configured and switched on with not one of them live.
    """
    now = datetime.now(UTC)
    organizations = (await session.execute(select(Organization))).scalars().all()
    names = {organization.id: organization.name for organization in organizations}

    by_status: dict[str, int] = {status.value: 0 for status in OrganizationStatus}
    for organization in organizations:
        key = _status_of(organization)
        by_status[key] = by_status.get(key, 0) + 1

    zones = await _group_count(session, Zone.organization_id)
    cameras = await _group_count(session, Camera.organization_id)
    switched_on = await _group_count(session, Camera.organization_id, Camera.enabled.is_(True))
    members = await _group_count(session, OrganizationMembership.organization_id)
    admins = await _admin_counts(session)
    last_sign_in = await _last_sign_in(session)
    wall = _wall_states(request)

    platform_states: dict[str, int] = {}
    for states in wall.values():
        for state, count in states.items():
            platform_states[state] = platform_states.get(state, 0) + count

    async def users_where(*conditions) -> int:
        statement = select(func.count()).select_from(User)
        if conditions:
            statement = statement.where(*conditions)
        return int((await session.execute(statement)).scalar_one())

    users_total = await users_where()
    users_active = await users_where(User.is_active.is_(True))
    never_signed_in = await users_where(User.last_login_at.is_(None))

    # People who work for more than one customer. The number that says whether
    # the multi-organization model is being used at all, and the population every
    # cross-tenant question is really about.
    multi_org = int(
        (
            await session.execute(
                select(func.count()).select_from(
                    select(OrganizationMembership.user_id)
                    .group_by(OrganizationMembership.user_id)
                    .having(func.count() > 1)
                    .subquery()
                )
            )
        ).scalar_one()
    )

    operators = int(
        (
            await session.execute(select(func.count()).select_from(PlatformOperatorGrant))
        ).scalar_one()
    )

    # An account with a password and nowhere to use it. Usually the residue of
    # a removed membership; occasionally somebody who was never finished. The
    # Platform Admin belongs to no organization by design and is not counted.
    without_membership = await users_where(
        ~User.id.in_(select(OrganizationMembership.user_id)),
        ~User.id.in_(select(PlatformOperatorGrant.user_id)),
    )

    fleet = [
        {
            "id": organization.id,
            "name": organization.name,
            "slug": organization.slug,
            "status": _status_of(organization),
            "created_at": _iso(organization.created_at),
            "zones": zones.get(organization.id, 0),
            "cameras": cameras.get(organization.id, 0),
            "cameras_enabled": switched_on.get(organization.id, 0),
            "cameras_live": wall.get(organization.id, {}).get(_LIVE, 0),
            "stream_states": wall.get(organization.id, {}),
            "members": members.get(organization.id, 0),
            "admins": admins.get(organization.id, 0),
            "last_sign_in_at": _iso(last_sign_in.get(organization.id)),
        }
        for organization in sorted(organizations, key=lambda o: (o.name or "").lower())
    ]

    def active(row: dict[str, Any]) -> bool:
        return row["status"] == OrganizationStatus.ACTIVE.value

    # Onboarding that stalled. An organization with no zones or no cameras is
    # not broken — it is unfinished, and nobody is currently told about it.
    stalled = [
        {
            "id": row["id"],
            "name": row["name"],
            "zone_count": row["zones"],
            "camera_count": row["cameras"],
        }
        for row in fleet
        if active(row) and (row["zones"] == 0 or row["cameras"] == 0)
    ]

    activity, refused_7d = await _activity(session, now=now, utc_offset=utc_offset)

    return {
        "generated_at": now.isoformat(),
        "organizations": {
            "total": len(organizations),
            "active": by_status.get(OrganizationStatus.ACTIVE.value, 0),
            "suspended": by_status.get(OrganizationStatus.SUSPENDED.value, 0),
            "archived": by_status.get(OrganizationStatus.ARCHIVED.value, 0),
            "new_30d": sum(
                1
                for organization in organizations
                if organization.created_at is not None
                and _aware(organization.created_at) >= now - timedelta(days=30)
            ),
        },
        "estate": {
            "zones": sum(zones.values()),
            "cameras": sum(cameras.values()),
            "cameras_enabled": sum(switched_on.values()),
            # Every stream the wall holds, whatever its state. Kept for the
            # callers that already read it; `cameras_live` is the number that
            # means pictures are arriving.
            "cameras_running": _running_total(request),
            "cameras_live": platform_states.get(_LIVE, 0),
            # Read from the wall registry rather than the camera table, so it
            # reports reality rather than intent. This process only.
            "stream_states": platform_states,
        },
        "people": {
            "users": users_total,
            "active_users": users_active,
            "disabled_users": users_total - users_active,
            "multi_organization_users": multi_org,
            "never_signed_in": never_signed_in,
            "signed_in_24h": await users_where(User.last_login_at >= now - timedelta(hours=24)),
            "signed_in_7d": await users_where(User.last_login_at >= now - timedelta(days=7)),
            "signed_in_30d": await users_where(User.last_login_at >= now - timedelta(days=30)),
            "without_membership": without_membership,
            "platform_operators": operators,
        },
        "fleet": fleet,
        "attention": {
            "organizations_needing_setup": stalled,
            # Nobody left who can let anybody in, take anybody out, or change
            # what they may do. The customer cannot fix this from the inside.
            "organizations_without_admin": [
                {"id": row["id"], "name": row["name"]}
                for row in fleet
                if active(row) and row["admins"] == 0
            ],
            # Switched on, and not one of them live: the failure that produces
            # the angriest call, because the customer believes they are watched.
            "organizations_not_live": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "cameras_enabled": row["cameras_enabled"],
                }
                for row in fleet
                if active(row) and row["cameras_enabled"] > 0 and row["cameras_live"] == 0
            ],
            "refused_changes_7d": refused_7d,
        },
        "activity": activity,
        "recent_activity": await _recent_activity(session, names),
    }


def _status_of(organization: Organization) -> str:
    return str(organization.status or "").strip().lower()


def _aware(value: datetime) -> datetime:
    """A stored timestamp, in UTC.

    PostgreSQL hands back an aware value; SQLite drops the zone on the way in
    and hands back a naive one. Every write here is UTC, so naive means UTC.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def _group_count(session: DbSession, column, *where) -> dict[str, int]:
    statement = select(column, func.count()).group_by(column)
    if where:
        statement = statement.where(*where)
    rows = await session.execute(statement)
    return {organization_id: int(count) for organization_id, count in rows.all()}


async def _admin_counts(session: DbSession) -> dict[str, int]:
    """Organization Admins per organization who can actually act as one.

    Three conditions, each one a way a role row outlives its holder's reach: the
    role is held there, the person is still a member there (removing a
    membership leaves the roles behind), and the account is still active.
    """
    rows = await session.execute(
        select(RoleAssignment.organization_id, func.count(func.distinct(RoleAssignment.user_id)))
        .join(User, User.id == RoleAssignment.user_id)
        .join(
            OrganizationMembership,
            (OrganizationMembership.user_id == RoleAssignment.user_id)
            & (OrganizationMembership.organization_id == RoleAssignment.organization_id),
        )
        .where(RoleAssignment.role == Role.ORG_ADMIN.value, User.is_active.is_(True))
        .group_by(RoleAssignment.organization_id)
    )
    return {organization_id: int(count) for organization_id, count in rows.all()}


async def _last_sign_in(session: DbSession) -> dict[str, datetime]:
    """The most recent sign-in by any member, per organization.

    Read from the account, not from the organization's own audit trail: a
    sign-in time is the platform's fact about a person, and the trail is the
    customer's record of what happened inside.
    """
    rows = await session.execute(
        select(OrganizationMembership.organization_id, func.max(User.last_login_at))
        .join(User, User.id == OrganizationMembership.user_id)
        .group_by(OrganizationMembership.organization_id)
    )
    return {organization_id: at for organization_id, at in rows.all() if at is not None}


def _running_total(request: Request) -> int:
    wall = getattr(request.app.state, "wall", None)
    if wall is None:
        return 0
    return len(getattr(wall, "streams", ()) or ())


def _wall_states(request: Request) -> dict[str, dict[str, int]]:
    """Every stream the wall holds, by organization and then by state.

    The state is the stream's own, derived from frame arrival: `live` means a
    frame arrived within the last few seconds, and nothing else does. The
    organization is the runtime id's tenant half — the same split that lets the
    wall stop one customer's cameras without touching another's.
    """
    from app.domain.runtime_identity import SEPARATOR

    wall = getattr(request.app.state, "wall", None)
    if wall is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    for key, stream in (getattr(wall, "streams", None) or {}).items():
        organization_id = str(key).split(SEPARATOR, 1)[0]
        state = str(getattr(stream, "state", "") or "unknown")
        states = out.setdefault(organization_id, {})
        states[state] = states.get(state, 0) + 1
    return out


async def _activity(
    session: DbSession, *, now: datetime, utc_offset: int
) -> tuple[dict[str, Any], int]:
    """Platform acts per day for the last `ACTIVITY_DAYS`, in the reader's days.

    A day is cut at the reader's midnight, not at UTC's: an act at 23:30 UTC is
    tomorrow morning in Singapore, and a chart that filed it under yesterday
    would disagree with the clock on the reader's own wall.

    Returns the chart and, separately, how many changes were refused in the last
    seven days — counted on the same rows so the two can never disagree.
    """
    shift = timedelta(minutes=utc_offset)
    today = (now + shift).date()
    first = today - timedelta(days=ACTIVITY_DAYS - 1)
    since = datetime(first.year, first.month, first.day, tzinfo=UTC) - shift

    rows = (
        await session.execute(
            select(AuditEvent.occurred_at, AuditEvent.outcome).where(
                AuditEvent.action.in_(PLATFORM_ACTIONS), AuditEvent.occurred_at >= since
            )
        )
    ).all()

    totals = {first + timedelta(days=offset): [0, 0] for offset in range(ACTIVITY_DAYS)}
    week_ago = now - timedelta(days=7)
    refused_7d = 0
    for occurred_at, outcome in rows:
        at = _aware(occurred_at)
        bucket = totals.get((at + shift).date())
        if bucket is None:
            continue
        bucket[0] += 1
        if outcome == AuditOutcome.DENIED.value:
            bucket[1] += 1
            if at >= week_ago:
                refused_7d += 1

    days = [
        {"date": day.isoformat(), "total": total, "refused": refused}
        for day, (total, refused) in sorted(totals.items())
    ]
    return (
        {
            "utc_offset": utc_offset,
            "days": days,
            "last_7d": sum(day["total"] for day in days[-7:]),
            "previous_7d": sum(day["total"] for day in days[:-7]),
        },
        refused_7d,
    )


async def _recent_activity(
    session: DbSession, names: dict[str, str], limit: int = 30
) -> list[dict[str, Any]]:
    """The last few platform-level acts, across every organization.

    Read from the same `audit_events` table every other trail in the product
    uses — there is no second log. Filtered to `PLATFORM_ACTIONS` so that a
    customer's own operational audit rows (evidence reads, incident changes) do
    not leak into a cross-tenant console: those are a tenant's record, readable
    with `VIEW_AUDIT` inside that tenant, and an operator does not hold it.

    ### Named, summarised, and never the raw detail

    Each row names the organization and, when the act was about a person, that
    person — an id is not something a reader can act on. What changed is
    summarised into a few whitelisted facts (`_change`). The stored detail
    itself never leaves: it is the writer's record, its shape is free-form, and
    a wire format that forwarded it would forward whatever the next writer put
    there.
    """
    rows = (
        (
            await session.execute(
                select(AuditEvent)
                .where(AuditEvent.action.in_(PLATFORM_ACTIONS))
                .order_by(AuditEvent.occurred_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    user_ids = {row.resource_id for row in rows if row.resource_type == "user" and row.resource_id}
    emails: dict[str, str] = {}
    if user_ids:
        emails = dict(
            (await session.execute(select(User.id, User.email).where(User.id.in_(user_ids)))).all()
        )

    out: list[dict[str, Any]] = []
    for row in rows:
        detail = _detail(row.detail)
        subject: str | None = None
        if row.resource_type == "user":
            remembered = detail.get("email")
            subject = emails.get(row.resource_id) or (
                remembered if isinstance(remembered, str) and remembered else None
            )
        out.append(
            {
                "id": row.id,
                "action": row.action,
                "organization_id": row.organization_id,
                "organization_name": names.get(row.organization_id, ""),
                "actor": row.actor,
                "resource_id": row.resource_id,
                "subject": subject,
                "outcome": row.outcome,
                "change": _change(row.action, row.outcome, detail),
                "occurred_at": _iso(row.occurred_at),
            }
        )
    return out


def _detail(raw: str) -> dict[str, Any]:
    """A stored detail, or nothing. An unreadable one is survived, not trusted."""
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _change(action: str, outcome: str, detail: dict[str, Any]) -> dict[str, Any] | None:
    """What an act changed, as the few facts a reader needs. Whitelisted."""

    def text(key: str) -> str | None:
        value = detail.get(key)
        return value[:200] if isinstance(value, str) and value else None

    change: dict[str, Any] = {}
    denied = outcome == AuditOutcome.DENIED.value

    if action == AuditAction.ORGANIZATION_STATUS_CHANGED.value:
        for key in ("from", "to", "reason"):
            if text(key):
                change[key] = text(key)
    elif action == AuditAction.ACCESS_SET.value and not denied:
        if "role" in detail:
            change["role"] = text("role")
        for key in ("added", "removed"):
            if isinstance(detail.get(key), list):
                change[key] = len(detail[key])
    elif action == AuditAction.ORGANIZATION_UPDATED.value:
        fields = detail.get("fields")
        if isinstance(fields, list):
            change["fields"] = [str(field) for field in fields]

    if denied and text("reason"):
        change["reason"] = text("reason")
    return change or None


# ── people ───────────────────────────────────────────────────────────────────


@router.get("/people")
async def list_people(
    operator: CurrentOperator,
    session: DbSession,
    q: Annotated[str | None, Query(max_length=200)] = None,
    organization_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Everyone on the platform, and which organizations they may enter.

    ### Home organization and memberships are reported separately, always

    `User.organization_id` is where the account *lives* — it owns email
    uniqueness, it is where a failed login is filed, and it is not by itself
    permission to enter anywhere. `memberships` is what they may actually enter.
    For almost every account today the two agree, and the moment they stop
    agreeing is exactly when somebody needs to see both. Collapsing them into
    one "organization" field would destroy the distinction this whole layer
    exists to express.
    """
    statement = select(User)
    if q:
        needle = f"%{q.strip().lower()}%"
        statement = statement.where(
            func.lower(User.email).like(needle) | func.lower(User.display_name).like(needle)
        )
    if organization_id:
        statement = statement.where(
            User.id.in_(
                select(OrganizationMembership.user_id).where(
                    OrganizationMembership.organization_id == organization_id
                )
            )
        )

    total = int(
        (await session.execute(select(func.count()).select_from(statement.subquery()))).scalar_one()
    )
    users = (
        (
            await session.execute(
                statement.options(
                    selectinload(User.memberships), selectinload(User.role_assignments)
                )
                .order_by(User.email)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )

    names = await _organization_names(session)
    operators = await _operator_user_ids(session)

    return {
        "people": [_person_to_wire(user, names, operators) for user in users],
        "count": len(users),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/people/{user_id}")
async def get_person(user_id: str, operator: CurrentOperator, session: DbSession) -> dict[str, Any]:
    user = await _user(session, user_id)
    names = await _organization_names(session)
    operators = await _operator_user_ids(session)
    return _person_to_wire(user, names, operators)


def _person_to_wire(user: User, names: dict[str, str], operators: set[str]) -> dict[str, Any]:
    roles_by_org: dict[str, list[str]] = {}
    for assignment in user.role_assignments or ():
        roles_by_org.setdefault(assignment.organization_id, []).append(assignment.role)

    memberships = [
        {
            "organization_id": membership.organization_id,
            "organization_name": names.get(membership.organization_id, membership.organization_id),
            "roles": sorted(roles_by_org.get(membership.organization_id, [])),
            "is_home": user.organization_id is not None
            and membership.organization_id == user.organization_id,
            "granted_at": _iso(membership.granted_at),
            "granted_by": membership.granted_by or "",
        }
        for membership in sorted(
            user.memberships or (), key=lambda m: names.get(m.organization_id, m.organization_id)
        )
    ]

    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "is_active": user.is_active,
        #: Where the account was created, or `None` for the Platform Admin, who
        #: belongs to no organization. Never conflated with what it may enter.
        "home_organization_id": user.organization_id,
        "home_organization_name": (
            names.get(user.organization_id, user.organization_id) if user.organization_id else ""
        ),
        "memberships": memberships,
        "organization_count": len(memberships),
        "last_login_at": _iso(user.last_login_at),
        "is_platform_operator": user.id in operators,
        #: True when the home organization is not among the memberships — an
        #: account filed somewhere it can no longer enter. Reported rather than
        #: hidden: it is a legitimate state, and it is also what a mistaken
        #: membership revocation looks like.
        "home_membership_missing": user.organization_id is not None
        and all(
            membership.organization_id != user.organization_id
            for membership in (user.memberships or ())
        ),
    }


# ── organization membership ──────────────────────────────────────────────────


@router.get("/organizations/{organization_id}/members")
async def list_members(
    organization_id: str, operator: CurrentOperator, session: DbSession
) -> dict[str, Any]:
    """Who may enter this organization, and what they hold once inside."""
    await _organization(session, organization_id)

    users = (
        (
            await session.execute(
                select(User)
                .where(
                    User.id.in_(
                        select(OrganizationMembership.user_id).where(
                            OrganizationMembership.organization_id == organization_id
                        )
                    )
                )
                .options(selectinload(User.memberships), selectinload(User.role_assignments))
                .order_by(User.email)
            )
        )
        .scalars()
        .all()
    )

    names = await _organization_names(session)
    operators = await _operator_user_ids(session)
    return {
        "organization_id": organization_id,
        "members": [_person_to_wire(user, names, operators) for user in users],
        "count": len(users),
    }


@router.post("/organizations/{organization_id}/members")
async def add_member(
    organization_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Admit an existing account to an organization.

    ### Membership is the entry ticket, and only the entry ticket

    This writes one `organization_memberships` row and nothing else. It grants
    no role, no camera scope and no permission — which means the person can now
    sign into this organization and, until somebody grants them a role *there*,
    see nothing. That is the correct default and it is deliberate: admitting
    somebody and deciding what they may do are two decisions, made by different
    people at different times, and collapsing them is how an account acquires
    authority nobody consciously granted.

    Roles inside the organization are granted through the organization's own
    user administration, by somebody who holds `MANAGE_USERS` there.

    ### Except the one role the platform gives: Organization Admin

    `"role": "org_admin"` also writes the role and an every-camera grant in the
    same transaction. That is how one Organization Admin comes to hold several
    organizations, and an admin admitted with nothing would be an admin who
    signs in and sees nothing. No other role is accepted here: the rest are
    staffed from inside the organization, by its own administrators.

    ### It admits, it does not create

    Creating a brand-new Organization Admin is `POST .../people`.
    """
    organization = await _organization(session, organization_id)
    user_id = str(payload.get("user_id", "") or "").strip()
    if not user_id:
        raise ValidationError("'user_id' is required")
    role = str(payload.get("role", "") or "").strip().lower()
    if role not in ("", Role.ORG_ADMIN.value):
        raise ValidationError(
            "only 'org_admin' can be given from the platform; other roles are "
            "granted inside the organization by its administrators",
            details={"role": role},
        )

    user = await _user(session, user_id)

    if str(organization.status or "").strip().lower() == OrganizationStatus.ARCHIVED.value:
        raise ValidationError(
            "an archived organization cannot take new members: nobody may sign in to one",
            details={"organization_id": organization_id},
        )

    existing = next((m for m in user.memberships if m.organization_id == organization_id), None)
    if existing is not None:
        raise ConflictError(
            f"{user.email} is already a member of {organization_id}",
            details={"organization_id": organization_id, "user_id": user_id},
        )

    session.add(
        OrganizationMembership(
            user_id=user.id,
            organization_id=organization_id,
            granted_by=operator.subject,
        )
    )
    if role:
        for row in _org_admin_rows(user.id, organization_id, granted_by=operator.subject):
            session.add(row)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_MEMBER_ADDED,
        # Filed against the organization gaining a member, because that is the
        # trail somebody auditing *that* customer's access will read.
        organization_id=organization_id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={
            "email": user.email,
            "home_organization_id": user.organization_id,
            "role": role or None,
        },
    )

    await session.refresh(user, attribute_names=["memberships", "role_assignments"])
    names = await _organization_names(session)
    operators = await _operator_user_ids(session)
    return _person_to_wire(user, names, operators)


def _org_admin_rows(user_id: str, organization_id: str, *, granted_by: str) -> list:
    """What makes somebody an Organization Admin *in* one organization.

    The role, and every camera. The membership is written by the caller,
    because admitting a person and making them an admin are the same act here
    but not everywhere.
    """
    return [
        RoleAssignment(
            user_id=user_id,
            organization_id=organization_id,
            role=Role.ORG_ADMIN.value,
            granted_by=granted_by,
        ),
        AccessGrant(
            user_id=user_id,
            organization_id=organization_id,
            camera_breadth="all_in_tenant",
            camera_ids="",
            site_ids="",
        ),
    ]


@router.post("/organizations/{organization_id}/people")
async def create_organization_admin(
    organization_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Create an Organization Admin: account, membership, role and cameras.

    ### One step, or nothing

    The Platform Admin creates an organization and then the person who runs it.
    Doing that as three separate acts — create an account, admit it, grant it a
    role — is how an organization ends up with somebody who can sign in and do
    nothing, or with nobody at all. So all of it happens in this request's one
    transaction, and the password is checked before any row exists: a refusal
    leaves nothing behind.

    ### Only Organization Admins

    Everyone else is staffed from inside the organization by its own admin,
    who knows the kitchen. The platform creates the first person, and that
    person creates the rest.

    ### A duplicate email is a plain conflict

    The caller is a Platform Admin, who can already list every person on the
    deployment (`GET /platform/people`), so naming the collision conceals
    nothing and tells him the next step: give the existing account this
    organization through `POST .../members` with `role: org_admin`.
    """
    organization = await _organization(session, organization_id)
    if str(organization.status or "").strip().lower() == OrganizationStatus.ARCHIVED.value:
        raise ValidationError(
            "an archived organization cannot take a new administrator: nobody may sign in to one",
            details={"organization_id": organization_id},
        )

    email = str(payload.get("email", "") or "").strip().lower()
    if not email or "@" not in email:
        raise ValidationError("'email' must be a valid address")
    display_name = str(payload.get("display_name", "") or "").strip() or email.split("@")[0]

    # Hashed before anything is written, so a short password creates nothing.
    password_hash = hash_password(
        str(payload.get("password", "") or ""),
        min_length=settings_of(request).password_min_length,
    )

    existing = (
        await session.execute(select(User.id).where(User.email == email))
    ).scalar_one_or_none()
    if existing is not None:
        # Audited with the same weight as a success, and committed before the
        # error propagates: the request rolls back on the way out.
        await AuditTrail(session).record(
            action=AuditAction.ORGANIZATION_ADMIN_CREATED,
            organization_id=organization.id,
            actor=operator.subject,
            actor_roles=("platform_operator",),
            outcome=AuditOutcome.DENIED,
            resource_type="user",
            resource_id=existing,
            request_id=_request_id(request),
            detail={"email": email, "reason": "email_in_use"},
        )
        await session.commit()
        raise ConflictError(
            f"{email} already has an account. To make it an Organization Admin here, "
            "add it as a member of this organization with the Organization Admin role.",
            details={"email": email, "user_id": existing},
        )

    user = User(
        organization_id=organization.id,
        email=email,
        display_name=display_name,
        password_hash=password_hash,
        is_active=True,
    )
    session.add(user)
    await session.flush()

    session.add(
        OrganizationMembership(
            user_id=user.id, organization_id=organization.id, granted_by=operator.subject
        )
    )
    for row in _org_admin_rows(user.id, organization.id, granted_by=operator.subject):
        session.add(row)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_ADMIN_CREATED,
        organization_id=organization.id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={"email": email, "display_name": display_name, "role": Role.ORG_ADMIN.value},
    )

    await session.refresh(user, attribute_names=["memberships", "role_assignments"])
    names = await _organization_names(session)
    operators = await _operator_user_ids(session)
    return _person_to_wire(user, names, operators)


@router.delete("/organizations/{organization_id}/members/{user_id}")
async def remove_member(
    organization_id: str,
    user_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
) -> dict[str, Any]:
    """Revoke somebody's access to an organization.

    Takes effect on their **next request**, not at their next login: the
    membership is re-read from the database on every call
    (`app.auth.service.user_for_claims`), so a token already naming this
    organization stops working immediately.

    ### The role rows are deliberately left behind

    Removing a membership does not delete that person's roles in the
    organization. They cannot enter, so the roles grant nothing — and if the
    revocation was a mistake, re-admitting them restores exactly what they had.
    Deleting the roles here would make an undo silently lossy, and "restore
    their access" would quietly mean "restore their access to nothing".

    ### Two consequences are reported rather than refused

    Removing somebody's *home* organization membership leaves the account filed
    where it can no longer sign in, and removing their *last* membership leaves
    it unable to sign in anywhere. Both are legitimate — offboarding looks
    exactly like this — so neither is blocked, and both are named in the
    response so the console can say so plainly.
    """
    await _organization(session, organization_id)
    user = await _user(session, user_id)

    membership = next((m for m in user.memberships if m.organization_id == organization_id), None)
    if membership is None:
        raise NotFoundError(f"{user.email} is not a member of {organization_id}")

    was_home = user.organization_id == organization_id
    remaining = [m for m in user.memberships if m.organization_id != organization_id]

    await session.delete(membership)
    await session.flush()

    await AuditTrail(session).record(
        action=AuditAction.ORGANIZATION_MEMBER_REMOVED,
        organization_id=organization_id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={
            "email": user.email,
            "was_home_organization": was_home,
            "organizations_remaining": len(remaining),
        },
    )

    return {
        "organization_id": organization_id,
        "user_id": user.id,
        "email": user.email,
        "removed": True,
        "was_home_organization": was_home,
        "organizations_remaining": len(remaining),
        #: The account can no longer sign in anywhere. Not an error — this is
        #: what offboarding looks like — but the console must be able to say it.
        "left_without_access": not remaining,
    }


# ── access ───────────────────────────────────────────────────────────────────
#
# The access matrix. Membership used to be the only thing this console
# administered; roles were granted inside each organization. Since 2026-09-22
# the Platform Admin can enter any organization with every permission and do
# exactly this from inside it, so setting access from here grants nobody reach
# they could not already be given — it is a second door onto the same
# authority, and every write through it is audited into the organization it
# changes, not into the platform's own trail.
#
# Enforcement is untouched: the rows written are the rows an Organization Admin
# writes from `/admin/users/:id`, read by `decide()` on every request.


@router.get("/people/{user_id}/access")
async def person_access(
    user_id: str, operator: CurrentOperator, session: DbSession
) -> dict[str, Any]:
    """What this person may do in every organization on the platform.

    Every organization is listed, member or not, because the matrix offers every
    one of them. For an organization the person is not in, everything is empty
    and `camera_scope` says `none` — roles left behind by a removed membership
    grant nothing and are not shown as if they did.

    `permissions` is what the matrix shows, `(role ∪ GRANTs) − REVOKEs`.
    `effective` is what `decide()` gives on the next request; the two differ
    only in a suspended organization, whose writes are withheld until it is
    restored. Both are reported so the console never re-derives either.
    """
    return await _access_to_wire(session, await _access_user(session, user_id))


async def _access_user(session: DbSession, user_id: str) -> User:
    found = await load_for_access(session, user_id)
    if found is None:
        raise NotFoundError(f"no user '{user_id}'")
    return found


async def _access_to_wire(session: DbSession, user: User) -> dict[str, Any]:
    organizations = (
        (await session.execute(select(Organization).order_by(Organization.name))).scalars().all()
    )
    operators = await _operator_user_ids(session)
    member_of = {m.organization_id for m in (user.memberships or ())}
    return {
        "user_id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "is_platform_operator": user.id in operators,
        "home_organization_id": user.organization_id,
        "organizations": [
            organization_access_to_wire(user, organization, is_member=organization.id in member_of)
            for organization in organizations
        ],
    }


@router.put("/people/{user_id}/access")
async def set_person_access(
    user_id: str,
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Set what this person may do, in one or more organizations, at once.

    Each item states an organization's whole access: a template role (or none),
    the permissions ticked, and optionally the camera breadth. The server writes
    the role and the exceptions that make `decide()` answer exactly those
    permissions (`app.authorization.assignments`).

    ### All or nothing

    Every item is parsed and every organization checked — it exists, it is not
    archived — before the first row is written, and the request is one
    transaction. One bad organization refuses the lot.

    ### Who may not be a target

    Yourself (403): an operator who could set their own access could give
    themselves anything, in any organization. And another Platform Admin
    (422): a Platform Admin belongs to no organization, and giving one
    memberships would make them a tenant principal as well.

    Only the organizations named are touched. An organization the person is not
    in is joined only if its item asks for something — a role, a permission or
    every camera — and that admission is audited as such.
    """
    user = await _access_user(session, user_id)
    items = parse_access_items(payload.get("organizations"))
    # Organizations first: a refusal below is audited *in* them, and an audit
    # row needs an organization that exists.
    await _require_open_organizations(session, items)
    await _refuse_unassignable_target(session, request, operator, user, items)

    actor = await _operator_user(session, operator)
    for item in items:
        target = await _access_user(session, user_id)
        applied = await apply_organization_access(
            session, actor=actor, target=target, access=item, granted_by=operator.subject
        )
        if applied is not None:
            await _audit_access(session, request, operator, target, applied)

    return await _access_to_wire(session, await _access_user(session, user_id))


@router.post("/people")
async def create_person(
    request: Request,
    operator: CurrentOperator,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Create a person and give them their access, in one step.

    The access matrix's first screen: account details, a home organization, and
    the same per-organization items `PUT .../access` takes. Everything is
    checked — the email, the password's floor, every organization — before any
    row exists, and it all happens in this request's one transaction, so a
    refusal leaves nothing behind.

    The home organization is always joined, even with nothing ticked there: it
    is where the account lives, and an account filed somewhere it cannot enter
    is the state `home_membership_missing` exists to report, not one to create.

    A duplicate email is a plain 409 naming the existing account. The caller is
    a Platform Admin who can already list everyone, and the next step is to edit
    that person's access rather than create a second one.
    """
    email = str(payload.get("email", "") or "").strip().lower()
    if not email or "@" not in email:
        raise ValidationError("'email' must be a valid address")
    display_name = str(payload.get("display_name", "") or "").strip() or email.split("@")[0]

    home_id = str(payload.get("home_organization_id", "") or "").strip()
    if not home_id:
        raise ValidationError("'home_organization_id' is required")
    items = parse_access_items(payload.get("organizations", []))
    home_item = OrganizationAccess(organization_id=home_id, role=None, permissions=frozenset())
    await _require_open_organizations(session, [home_item, *items])

    password_hash = hash_password(
        str(payload.get("password", "") or ""),
        min_length=settings_of(request).password_min_length,
    )

    existing = (
        await session.execute(select(User.id).where(User.email == email))
    ).scalar_one_or_none()
    if existing is not None:
        # Audited with the same weight as a success, and committed before the
        # error propagates: the request rolls back on the way out.
        await AuditTrail(session).record(
            action=AuditAction.USER_CREATED,
            organization_id=home_id,
            actor=operator.subject,
            actor_roles=("platform_operator",),
            outcome=AuditOutcome.DENIED,
            resource_type="user",
            resource_id=existing,
            request_id=_request_id(request),
            detail={"email": email, "reason": "email_in_use", "via": "access_matrix"},
        )
        await session.commit()
        raise ConflictError(
            f"{email} already has an account. Open that person and edit their access instead.",
            details={"email": email, "user_id": existing},
        )

    user = User(
        organization_id=home_id,
        email=email,
        display_name=display_name,
        password_hash=password_hash,
        is_active=True,
    )
    session.add(user)
    await session.flush()
    session.add(
        OrganizationMembership(
            user_id=user.id, organization_id=home_id, granted_by=operator.subject
        )
    )
    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.USER_CREATED,
        organization_id=home_id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={"email": email, "display_name": display_name, "via": "access_matrix"},
    )

    actor = await _operator_user(session, operator)
    for item in items:
        target = await _access_user(session, user.id)
        applied = await apply_organization_access(
            session, actor=actor, target=target, access=item, granted_by=operator.subject
        )
        if applied is not None:
            await _audit_access(session, request, operator, target, applied)

    return await _access_to_wire(session, await _access_user(session, user.id))


async def _refuse_unassignable_target(
    session: DbSession,
    request: Request,
    operator: PlatformOperator,
    user: User,
    items: list[OrganizationAccess],
) -> None:
    """Refuse yourself and other Platform Admins — audited, then raised.

    A refusal is recorded with the same weight as a success, in each
    organization the request aimed at, and committed before the error
    propagates: the request rolls back on the way out, and the record that
    somebody tried must not go with it.
    """
    if user.id == operator.user_id:
        reason, error = "self", ScopeError("you may not change your own access")
    elif user.id in await _operator_user_ids(session):
        reason, error = "platform_operator", ValidationError(
            "a Platform Admin belongs to no organization; platform authority is not "
            "set from the access matrix",
            details={"user_id": user.id},
        )
    else:
        return

    trail = AuditTrail(session)
    for item in items:
        await trail.record(
            action=AuditAction.ACCESS_SET,
            organization_id=item.organization_id,
            actor=operator.subject,
            actor_roles=("platform_operator",),
            outcome=AuditOutcome.DENIED,
            resource_type="user",
            resource_id=user.id,
            request_id=_request_id(request),
            detail={"email": user.email, "reason": reason},
        )
    await session.commit()
    raise error


async def _require_open_organizations(session: DbSession, items: list[OrganizationAccess]) -> None:
    """Every named organization exists (404) and is not archived (422)."""
    for item in items:
        organization = await _organization(session, item.organization_id)
        if str(organization.status or "").strip().lower() == OrganizationStatus.ARCHIVED.value:
            raise ValidationError(
                "an archived organization cannot be given access: nobody may sign in to one",
                details={"organization_id": organization.id},
            )


async def _operator_user(session: DbSession, operator: PlatformOperator) -> User:
    """The operator's own row — the `actor` the override and camera guards
    compare against, and whose id is recorded as `granted_by` on overrides."""
    return (await session.execute(select(User).where(User.id == operator.user_id))).scalar_one()


async def _audit_access(
    session: DbSession,
    request: Request,
    operator: PlatformOperator,
    user: User,
    applied: AppliedAccess,
) -> None:
    """Filed in the organization that changed — its own trail is where somebody
    reviewing that customer's access will look."""
    trail = AuditTrail(session)
    if applied.admitted:
        await trail.record(
            action=AuditAction.ORGANIZATION_MEMBER_ADDED,
            organization_id=applied.organization_id,
            actor=operator.subject,
            actor_roles=("platform_operator",),
            resource_type="user",
            resource_id=user.id,
            request_id=_request_id(request),
            detail={
                "email": user.email,
                "home_organization_id": user.organization_id,
                "via": "access_matrix",
            },
        )
    await trail.record(
        action=AuditAction.ACCESS_SET,
        organization_id=applied.organization_id,
        actor=operator.subject,
        actor_roles=("platform_operator",),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail=access_audit_detail(user, applied),
    )


# ── operators ────────────────────────────────────────────────────────────────


@router.get("/operators")
async def list_operators(operator: CurrentOperator, session: DbSession) -> dict[str, Any]:
    """Who holds platform authority, since when, and why.

    Read-only, and that is a deliberate boundary rather than an unfinished
    feature. Granting platform authority stays a command-line act — an operator
    who could mint another operator would make the privilege self-propagating,
    and the one control on a grant that reaches every customer's data is that
    making one is awkward and leaves a human trail outside this application.

    Visibility is the half that genuinely belongs in a console: "who can reach
    all of our customers" is a question somebody should be able to answer
    without database access, even when the answer is not editable here.
    """
    rows = (
        await session.execute(
            select(PlatformOperatorGrant, User)
            .join(User, User.id == PlatformOperatorGrant.user_id)
            .order_by(User.email)
        )
    ).all()

    names = await _organization_names(session)
    return {
        "operators": [
            {
                "user_id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "is_active": user.is_active,
                "home_organization_id": user.organization_id,
                "home_organization_name": (
                    names.get(user.organization_id, user.organization_id)
                    if user.organization_id
                    else ""
                ),
                "granted_at": _iso(grant.granted_at),
                "granted_by": grant.granted_by or "",
                "reason": grant.reason or "",
                "last_login_at": _iso(user.last_login_at),
            }
            for grant, user in rows
        ],
        "count": len(rows),
        #: Stated in the payload so the console does not have to hardcode the
        #: policy, and so changing it is a server change rather than a UI one.
        "grant_is_manageable_here": False,
        "how_to_grant": "scripts/manage.py grant-operator --email ... --reason ...",
    }


# ── role policy ──────────────────────────────────────────────────────────────


@router.get("/roles")
async def role_policy(operator: CurrentOperator) -> dict[str, Any]:
    """What each role means, as the server actually enforces it.

    ### This is a read, and it will stay a read until the model changes

    `ROLE_PERMISSIONS` is a Python constant and `Role` is a closed enum
    (`app.authorization.model`). There is no table behind either, no write path,
    and no runtime by which anybody — platform operator included — can change
    what a role means. `editable: false` says so in the payload rather than
    leaving a console to discover it by getting a 405.

    The mechanism that *does* exist for making two holders of the same role
    differ is `permission_overrides`, per user and per organization, applied
    through that organization's own user administration. It is reported here as
    the answer to the question this endpoint will otherwise be asked.
    """
    return {
        "roles": [
            {
                "role": role.value,
                "permissions": sorted(p.value for p in ROLE_PERMISSIONS.get(role, frozenset())),
                "permission_count": len(ROLE_PERMISSIONS.get(role, frozenset())),
                "is_platform_role": role.is_platform_role,
            }
            for role in Role
        ],
        "permissions": sorted(p.value for p in Permission),
        #: Role definitions are code, not data. A console must not offer to edit
        #: them, because the server would not enforce the edit.
        "editable": False,
        "customization": {
            "mechanism": "permission_overrides",
            "scope": "per user, per organization",
            "where": "the organization's own user administration",
            "note": (
                "Two people holding the same role are made to differ by granting "
                "or revoking individual permissions on the person, inside the "
                "organization it applies to — not by redefining the role."
            ),
        },
    }


# ── shared lookups ───────────────────────────────────────────────────────────


async def _organization_names(session: DbSession) -> dict[str, str]:
    rows = await session.execute(select(Organization.id, Organization.name))
    return dict(rows.all())


async def _operator_user_ids(session: DbSession) -> set[str]:
    rows = await session.execute(select(PlatformOperatorGrant.user_id))
    return {user_id for (user_id,) in rows.all()}


async def _organization(session: DbSession, organization_id: str) -> Organization:
    found = (
        await session.execute(select(Organization).where(Organization.id == organization_id))
    ).scalar_one_or_none()
    if found is None:
        raise NotFoundError(f"no organization '{organization_id}'")
    return found


async def _user(session: DbSession, user_id: str) -> User:
    found = (
        await session.execute(
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.memberships), selectinload(User.role_assignments))
        )
    ).scalar_one_or_none()
    if found is None:
        raise NotFoundError(f"no user '{user_id}'")
    return found


__all__ = ["PLATFORM_ACTIONS", "router"]
