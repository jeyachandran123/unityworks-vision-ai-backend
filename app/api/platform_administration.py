"""The platform control plane — everything above one organization.

A second module on the same router prefix and the same principal as
`app.api.platform`, split from it because the two answer different questions.
`platform.py` owns an organization's *existence and lifecycle*: create it,
rename it, suspend it, enter it. This module owns administering the platform
*across* organizations: who exists, who may enter what, what an operator is, and
what the estate looks like in aggregate.

Every route here is gated on `CurrentOperator`, reads no `Permission`, and is
refused outright for any tenant principal. That is the same boundary
`platform.py` documents, and the reason this is a separate file rather than a
separate authorization model.

### What this module deliberately does not do

**It never returns a customer's operational content.** Counts, names,
configuration and membership — never an incident, never an observation, never a
frame, never an evidence record. An operator who needs to see those enters the
organization explicitly through `POST /platform/organizations/{id}/enter`, which
is audited and read-only, and reads them through the ordinary tenant routes with
a tenant token. Adding a cross-tenant read here would make that entry pointless
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

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentOperator, DbSession
from app.authorization.model import (
    ROLE_PERMISSIONS,
    OrganizationStatus,
    Permission,
    Role,
)
from app.domain.audit import AuditAction, AuditTrail
from app.domain.models import AuditEvent, Camera, Restaurant
from app.errors import ConflictError, NotFoundError, ValidationError
from app.users.models import (
    Organization,
    OrganizationMembership,
    PlatformOperatorGrant,
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
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# ── overview ─────────────────────────────────────────────────────────────────


@router.get("/overview")
async def overview(
    request: Request, operator: CurrentOperator, session: DbSession
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
    """
    organizations = (await session.execute(select(Organization))).scalars().all()

    by_status: dict[str, int] = {status.value: 0 for status in OrganizationStatus}
    for organization in organizations:
        key = str(organization.status or "").strip().lower()
        by_status[key] = by_status.get(key, 0) + 1

    sites = await _group_count(session, Restaurant.organization_id)
    cameras = await _group_count(session, Camera.organization_id)

    users_total = int((await session.execute(select(func.count()).select_from(User))).scalar_one())
    users_active = int(
        (
            await session.execute(
                select(func.count()).select_from(User).where(User.is_active.is_(True))
            )
        ).scalar_one()
    )
    never_signed_in = int(
        (
            await session.execute(
                select(func.count()).select_from(User).where(User.last_login_at.is_(None))
            )
        ).scalar_one()
    )

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

    # Onboarding that stalled. An organization with no sites or no cameras is
    # not broken — it is unfinished, and nobody is currently told about it.
    stalled = [
        {
            "id": organization.id,
            "name": organization.name,
            "site_count": sites.get(organization.id, 0),
            "camera_count": cameras.get(organization.id, 0),
        }
        for organization in organizations
        if str(organization.status or "").strip().lower() == OrganizationStatus.ACTIVE.value
        and (sites.get(organization.id, 0) == 0 or cameras.get(organization.id, 0) == 0)
    ]

    return {
        "organizations": {
            "total": len(organizations),
            "active": by_status.get(OrganizationStatus.ACTIVE.value, 0),
            "suspended": by_status.get(OrganizationStatus.SUSPENDED.value, 0),
            "archived": by_status.get(OrganizationStatus.ARCHIVED.value, 0),
        },
        "estate": {
            "sites": sum(sites.values()),
            "cameras": sum(cameras.values()),
            # What the runtime is actually streaming, platform-wide. Read from
            # the wall registry rather than the camera table, so it reports
            # reality rather than intent.
            "cameras_running": _running_total(request),
        },
        "people": {
            "users": users_total,
            "active_users": users_active,
            "multi_organization_users": multi_org,
            "never_signed_in": never_signed_in,
            "platform_operators": operators,
        },
        "attention": {"organizations_needing_setup": stalled},
        "recent_activity": await _recent_activity(session),
    }


async def _group_count(session: DbSession, column) -> dict[str, int]:
    rows = await session.execute(select(column, func.count()).group_by(column))
    return {organization_id: int(count) for organization_id, count in rows.all()}


def _running_total(request: Request) -> int:
    wall = getattr(request.app.state, "wall", None)
    if wall is None:
        return 0
    return len(getattr(wall, "streams", ()) or ())


async def _recent_activity(session: DbSession, limit: int = 20) -> list[dict[str, Any]]:
    """The last few platform-level acts, across every organization.

    Read from the same `audit_events` table every other trail in the product
    uses — there is no second log. Filtered to `PLATFORM_ACTIONS` so that a
    customer's own operational audit rows (evidence reads, incident changes) do
    not leak into a cross-tenant console: those are a tenant's record, readable
    with `VIEW_AUDIT` inside that tenant, and an operator does not hold it.
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
    return [
        {
            "id": row.id,
            "action": row.action,
            "organization_id": row.organization_id,
            "actor": row.actor,
            "resource_id": row.resource_id,
            "outcome": row.outcome,
            "occurred_at": _iso(row.occurred_at),
        }
        for row in rows
    ]


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
            "is_home": membership.organization_id == user.organization_id,
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
        #: Where the account lives. Never conflated with what it may enter.
        "home_organization_id": user.organization_id,
        "home_organization_name": names.get(user.organization_id, user.organization_id),
        "memberships": memberships,
        "organization_count": len(memberships),
        "last_login_at": _iso(user.last_login_at),
        "is_platform_operator": user.id in operators,
        #: True when the home organization is not among the memberships — an
        #: account filed somewhere it can no longer enter. Reported rather than
        #: hidden: it is a legitimate state, and it is also what a mistaken
        #: membership revocation looks like.
        "home_membership_missing": all(
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

    ### It admits, it does not create

    There is no user creation here. An operator who could mint accounts into any
    customer could mint one for themselves, and the audited read-only entry
    route would become an unnecessary formality.
    """
    organization = await _organization(session, organization_id)
    user_id = str(payload.get("user_id", "") or "").strip()
    if not user_id:
        raise ValidationError("'user_id' is required")

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
        detail={"email": user.email, "home_organization_id": user.organization_id},
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
                "home_organization_name": names.get(user.organization_id, user.organization_id),
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
