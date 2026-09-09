"""User management, role assignment, and permission-override administration.

Stage 5 of the multi-organization identity plan (see
`docs/architecture/MULTI_ORGANIZATION_IDENTITY_USER_MANAGEMENT_PERMISSION_OVERRIDE_IMPLEMENTATION.md`
and the frozen architecture document). Stages 1-4 built the domain model —
`PermissionOverride`, `Organization.status`, and the `(role permissions ∪
GRANTs) − REVOKEs` composition at `app.authorization.resolver.decide()` — and
left it reachable only from Python. This module is the HTTP layer on top of
that already-tested domain logic. It does not reimplement any of it:
role/permission mutation always goes through `RoleAssignment` rows or
`app.authorization.overrides.set_permission_override` /
`clear_permission_override`, exactly as `app/api/administration.py` calls into
`Restaurant`/`Zone` rather than re-deriving their invariants.

### Tenant scope

Every query here is narrowed to `access.tenant_id` — never to an id supplied
by the caller — the same discipline `app/api/administration.py` documents. A
user id that resolves to another organisation is a 404, not a 403: existence
across a tenant boundary is itself a disclosure.

### Credentials

Nothing in this module ever returns a password or a password hash. Creating a
user accepts an optional plaintext password in the request body (present only
in that one request, never logged, never echoed back) or, if omitted,
generates one and returns it exactly once in the creation response — the same
"printed once, deliberately" convention `scripts/manage.py reset-password
--generate` already uses. There is no invitation or SSO flow here; this is the
smallest mechanism that reuses what `app/auth` already has.

### Anti-privilege-escalation

Three rules are enforced here, all traceable to the caller's own
request-scoped `AccessDecision` — never a cached or stale value, since
`current_access` rebuilds it from the database on every request:

1. **No self-modification.** An actor may not change their own roles or
   permission overrides, and may not deactivate their own account. For
   permission overrides this is enforced a second time, structurally, inside
   `app.authorization.overrides._guard()` — this module's checks are not a
   substitute for that one, they run first so the error is uniform.
2. **A grantor may only give out what they hold.** GRANTing a `Permission`
   requires the actor to already hold it (`access.has(permission)`).
   Assigning a `Role` requires that *every permission the role would carry*
   is already something the actor holds
   (`permissions_for({role}) <= access.permissions`) — checking the actor's
   own `Role` membership instead (e.g. "must already hold `org_admin` to
   assign `org_admin`") would have blocked the role assignment API's
   primary real-world use: an `org_admin` (whose own `RoleAssignment` rows
   name only `org_admin`) staffing `restaurant_manager`/`kitchen_supervisor`/
   etc. accounts is exactly the case this API exists for, and every
   permission those roles carry is already a subset of what `org_admin`
   itself holds. The same subset check still refuses an `org_admin`
   assigning `super_admin` or `developer`, both of which carry permissions
   (`access_devtools`, `view_model_evaluation`, …) `org_admin` does not have
   — which is the actual escalation this rule exists to close. Removing a
   role or REVOKing a permission carries no such requirement — both only
   ever narrow the target's access, and narrowing what someone else can do
   is not an escalation in the direction this rule exists to block.
3. **`MANAGE_USERS` gates the whole surface.** Every route below depends on
   it.

**Deliberately left unimplemented, not decided here:** whether a
`MANAGE_USERS` holder who does not themselves hold a role should be permitted
to *remove* that role from someone else, or to REVOKE a permission from
someone else, when the target holds it via a role the actor cannot reach.
Concretely: today `MANAGE_USERS` is held by exactly `Role.SUPER_ADMIN` and
`Role.ORG_ADMIN` (`app/authorization/model.py`, `ROLE_PERMISSIONS`), and
`ORG_ADMIN` does not hold every permission `SUPER_ADMIN` does — so an
`org_admin` reaching this API can remove a `super_admin`'s role, or REVOKE a
permission from a `super_admin` target, using only the rules above, and rule 2
does not stop it because removal/REVOKE is a narrowing act rather than a
granting one. Whether that should be blocked by a role hierarchy is exactly
the ambiguity the frozen architecture document's Decision Table calls
"REQUIRES OWNER DECISION" for the REVOKE asymmetry, and inventing a hierarchy
rule here would be creating security policy rather than implementing an
approved one. Left open; see the Stage 5 report, §12.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import CurrentAccess, DbSession, requires, settings_of
from app.auth.passwords import hash_password
from app.authorization.camera_scope import (
    grant_to_wire,
    parse_camera_scope_request,
    require_grantable_scope,
    resolve_camera_keys,
    set_camera_scope,
)
from app.authorization.model import (
    AccessDecision,
    OverrideState,
    Permission,
    Role,
    ScopeBreadth,
    permissions_for,
)
from app.authorization.overrides import clear_permission_override, set_permission_override
from app.authorization.resolver import decide, parse_overrides
from app.domain.audit import AuditAction, AuditTrail
from app.errors import ConflictError, NotFoundError, ScopeError, ValidationError
from app.users.models import OrganizationMembership, RoleAssignment, User

#: Reads are `VIEW_USERS`; writes are `MANAGE_USERS`. Declared per route rather
#: than once on the router, because gating the whole surface on `MANAGE_USERS`
#: made the roster unreadable to anyone who could not also change it — which is
#: not how any other domain in this application works, and which also meant a
#: SUSPENDED organization could not read its own user list (suspension drops
#: `MANAGE_*`, so it took the reads with it).
router = APIRouter(prefix="/api/v1/admin/users", tags=["user-administration"])

_READS = [Depends(requires(Permission.VIEW_USERS))]
_WRITES = [Depends(requires(Permission.MANAGE_USERS))]


# ── shared helpers ───────────────────────────────────────────────────────────


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _roles_tuple(access: AccessDecision) -> tuple[str, ...]:
    return tuple(sorted(r.value for r in access.roles))


async def _user_in_tenant(session: AsyncSession, tenant_id: str, user_id: str) -> User:
    """Load a user, eagerly, already narrowed to the caller's tenant.

    A mismatch is a `NotFoundError`, not a `ScopeError` — the same "it exists
    but is not yours" avoidance `app/api/administration.py:_restaurant_in_tenant`
    uses, mirrored here rather than reinvented.
    """
    found = (
        await session.execute(
            select(User)
            .where(User.id == user_id, User.organization_id == tenant_id)
            .options(
                selectinload(User.role_assignments),
                selectinload(User.access_grants),
                selectinload(User.permission_overrides),
                selectinload(User.organization),
                # `decide()` reads the memberships to find which organization's
                # status applies. Lazily loading a relationship inside an async
                # request raises `MissingGreenlet` rather than doing the IO, so
                # every path that reaches `decide()` has to bring them along.
                selectinload(User.memberships).selectinload(
                    OrganizationMembership.organization
                ),
            )
        )
    ).scalar_one_or_none()
    if found is None:
        raise NotFoundError(f"no user '{user_id}'")
    return found


async def _actor(session: AsyncSession, access: AccessDecision) -> User:
    """The caller's own row, loaded fresh in this request's session.

    Needed because `set_permission_override`/`clear_permission_override` take
    `User` objects (for `.id`, `.organization_id`, and `granted_by`), not an
    `AccessDecision`. `access.subject` is the caller's email — unique within
    their own tenant — so this cannot resolve to another organisation's user.
    """
    found = (
        await session.execute(
            select(User).where(
                User.email == access.subject, User.organization_id == access.tenant_id
            )
        )
    ).scalar_one_or_none()
    if found is None:  # pragma: no cover - the caller authenticated as this user
        raise NotFoundError("the acting account could not be reloaded")
    return found


def _is_self(access: AccessDecision, target: User) -> bool:
    return target.email == access.subject


def _role_from(value: Any) -> Role:
    try:
        return Role(str(value).strip().lower())
    except ValueError as exc:
        raise ValidationError(f"'{value}' is not a known role") from exc


def _permission_from(value: Any) -> Permission:
    try:
        return Permission(str(value).strip().lower())
    except ValueError as exc:
        raise ValidationError(f"'{value}' is not a known permission") from exc


def _state_from(value: Any) -> OverrideState:
    try:
        return OverrideState(str(value).strip().lower())
    except ValueError as exc:
        raise ValidationError(
            f"'{value}' is not a known override state; use 'grant' or 'revoke'"
        ) from exc


def _require_grantable_role(access: AccessDecision, role: Role) -> None:
    """Refuse assigning a role that would hand the target a permission the
    actor does not themselves hold. See rule 2 in the module docstring."""
    role_permissions = permissions_for(frozenset({role}))
    if not role_permissions <= access.permissions:
        raise ScopeError(
            f"cannot assign role '{role.value}': it carries a permission you do not hold",
            details={
                "role": role.value,
                "missing": sorted(p.value for p in role_permissions - access.permissions),
            },
        )


def _user_to_wire(user: User) -> dict[str, Any]:
    """A user record. No `password_hash`, ever — the same discipline
    `app/api/administration.py:list_users` already applies."""
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "is_active": bool(user.is_active),
        "roles": sorted(a.role for a in user.role_assignments),
        # Reported on every user, because "which cameras can this account
        # reach" is half of what access means here and it was previously
        # invisible to every administration screen.
        "camera_scope": grant_to_wire(_grant_of(user)),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def _grant_of(user: User):
    """The user's single `AccessGrant`, or `None`.

    `None` is not a missing value to paper over: it is the state that means no
    camera access, and `grant_to_wire` says so explicitly rather than omitting
    the field.
    """
    grants = list(user.access_grants or ())
    return grants[0] if grants else None


async def _permission_rows(target: User, organization_id: str) -> list[dict[str, Any]]:
    """Every `Permission`, its stored state, and its computed effective result.

    One authoritative response rather than something the caller reconstructs
    from separate role and override reads — `decide()` is the single existing
    chokepoint that already composes roles, overrides, and organization
    lifecycle together, so "effective" here is exactly what the same user
    would be given on their next authenticated request, not a re-derivation
    that could drift from it.
    """
    granted, revoked = parse_overrides(
        [
            override
            for override in (target.permission_overrides or ())
            if override.organization_id == organization_id
        ]
    )
    decision = decide(target, organization_id=organization_id)
    # Reuses the exact `permissions_for(roles)` call `decide()` itself makes
    # internally (via `effective_permissions`) — not a re-derivation, just the
    # same role->permission union, called again here so the response can show
    # "does the role alone grant this" separately from the override and the
    # composed effective result.
    role_permissions = permissions_for(decision.roles)
    rows: list[dict[str, Any]] = []
    for permission in Permission:
        if permission in granted:
            state = OverrideState.GRANT.value
        elif permission in revoked:
            state = OverrideState.REVOKE.value
        else:
            state = "inherit"
        rows.append(
            {
                "permission": permission.value,
                "state": state,
                "role_grants": permission in role_permissions,
                "effective": decision.has(permission),
            }
        )
    return rows


# ── users ────────────────────────────────────────────────────────────────────


@router.get("", dependencies=_READS)
async def list_users(
    access: CurrentAccess,
    session: DbSession,
    q: Annotated[str | None, Query(max_length=200)] = None,
    role: Annotated[str | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Users in the caller's organisation, searchable and filterable.

    The role filter is a subquery rather than a join, deliberately: a user may
    hold several roles, and joining would return them once per matching role
    and make `total` wrong in a way that only shows up for exactly the accounts
    an administrator is most likely to be looking at.
    """
    statement = select(User).where(User.organization_id == access.tenant_id)
    if q:
        needle = f"%{q.strip().lower()}%"
        statement = statement.where(
            func.lower(User.email).like(needle) | func.lower(User.display_name).like(needle)
        )
    if is_active is not None:
        statement = statement.where(User.is_active.is_(is_active))
    if role:
        # Validated rather than passed through: an unknown role name should be
        # a clear 422, not a silently empty list that reads as "nobody holds
        # this".
        wanted = _role_from(role)
        statement = statement.where(
            User.id.in_(
                select(RoleAssignment.user_id).where(RoleAssignment.role == wanted.value)
            )
        )

    total = int(
        (await session.execute(select(func.count()).select_from(statement.subquery()))).scalar_one()
    )
    users = (
        (
            await session.execute(
                statement.options(
                    selectinload(User.role_assignments),
                    selectinload(User.access_grants),
                    selectinload(User.memberships),
                )
                .order_by(User.email)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "users": [_user_to_wire(u) for u in users],
        "count": len(users),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{user_id}", dependencies=_READS)
async def get_user(user_id: str, access: CurrentAccess, session: DbSession) -> dict[str, Any]:
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    return _user_to_wire(user)


@router.post("", dependencies=_WRITES)
async def create_user(
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Create a user in the caller's organisation.

    Duplicate email within the organisation is a clean `ConflictError` (409),
    never a 500 — the unique constraint is `(organization_id, email)`
    (`app/users/models.py:uq_users_org_email`), so this checks the same pair
    before insert rather than letting the database raise.

    Every role in `roles` must be grantable by the actor (rule 2 in the module
    docstring, `_require_grantable_role`); an unknown role name is a
    `ValidationError` (422).

    ### `camera_scope` is required, and that is the fix for a real bug

    This route used to create a `User` and its `RoleAssignment` rows and stop.
    No `AccessGrant` was written, and a missing grant reads as
    `CameraScope.none()` — correctly, since that is the only safe reading of an
    absent row. The result was an account that logged in, held every permission
    its role carried, and could reach no cameras at all. Nothing reported it.
    `admin1@unityworks.local` is in exactly that state in production today.

    The fix is not a default. Both candidate defaults are wrong: `none`
    silently recreates the unusable account, and `all_in_tenant` silently
    creates an over-privileged one. So the caller states which of the three
    scopes they mean, and `none` remains a legitimate answer — for an account
    that will be scoped later, or one that never needs video.
    """
    email = str(payload.get("email", "")).strip().lower()
    if not email or "@" not in email:
        raise ValidationError("'email' must be a valid address")

    display_name = str(payload.get("display_name", "") or "").strip() or email.split("@")[0]
    role_values = payload.get("roles", [])
    if not isinstance(role_values, list):
        raise ValidationError("'roles' must be a list")
    roles = [_role_from(r) for r in role_values]

    for role in roles:
        _require_grantable_role(access, role)

    existing = (
        await session.execute(
            select(User).where(User.organization_id == access.tenant_id, User.email == email)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError(f"a user with email '{email}' already exists in this organization")

    scope = parse_camera_scope_request(payload)
    require_grantable_scope(access, scope)
    if scope.breadth is ScopeBreadth.LISTED:
        # Checked before the user exists, so a typo cannot leave a half-created
        # account behind.
        await resolve_camera_keys(
            session, organization_id=access.tenant_id, camera_keys=scope.camera_ids
        )

    settings = settings_of(request)
    provided_password = payload.get("password")
    generated_password: str | None = None
    if provided_password:
        password = str(provided_password)
    else:
        generated_password = secrets.token_urlsafe(18)
        password = generated_password

    user = User(
        organization_id=access.tenant_id,
        email=email,
        display_name=display_name,
        password_hash=hash_password(password, min_length=settings.password_min_length),
        is_active=True,
    )
    session.add(user)
    await session.flush()

    # The membership *before* the roles, and both before anything reads them.
    # A user with role rows and no membership is an account that cannot sign
    # into the organization those roles are for — the exact shape of bug the
    # entry-ticket rule exists to make impossible, so it must not be creatable
    # by the one route that makes users.
    session.add(
        OrganizationMembership(
            user_id=user.id,
            organization_id=access.tenant_id,
            granted_by=access.subject,
        )
    )

    for role in roles:
        session.add(
            RoleAssignment(
                user_id=user.id,
                organization_id=access.tenant_id,
                role=role.value,
                granted_by=access.subject,
            )
        )

    actor = await _actor(session, access)
    await set_camera_scope(session, actor=actor, target=user, scope=scope)

    await session.flush()
    await session.refresh(user, attribute_names=["role_assignments", "access_grants"])

    await AuditTrail(session).record(
        action=AuditAction.USER_CREATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={
            "email": email,
            "roles": sorted(r.value for r in roles),
            "camera_breadth": scope.breadth.value,
            "camera_count": len(scope.camera_ids),
        },
    )

    wire = _user_to_wire(user)
    if generated_password is not None:
        # Returned exactly once, in this response, and stored nowhere. The
        # same "printed once" convention `scripts/manage.py reset-password
        # --generate` already uses for the same reason: a generated password
        # nobody sees is a locked account.
        wire["generated_password"] = generated_password
    return wire


@router.patch("/{user_id}", dependencies=_WRITES)
async def update_user(
    user_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Update the allowed mutable fields. Today, only `display_name`.

    Email, password, and activation state each have their own, more specific
    path — email identity and credentials are not casually PATCHable fields,
    and activation is audited as its own event (`activate`/`deactivate`
    below), not folded into a generic update.
    """
    user = await _user_in_tenant(session, access.tenant_id, user_id)

    changed: list[str] = []
    if "display_name" in payload:
        name = str(payload["display_name"] or "").strip()
        if not name:
            raise ValidationError("'display_name' must not be empty")
        user.display_name = name
        changed.append("display_name")

    if not changed:
        return _user_to_wire(user)

    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.USER_UPDATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
        detail={"fields": sorted(changed)},
    )
    return _user_to_wire(user)


@router.post("/{user_id}/activate", dependencies=_WRITES)
async def activate_user(
    user_id: str, request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    if user.is_active:
        return _user_to_wire(user)

    user.is_active = True
    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.USER_ACTIVATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
    )
    return _user_to_wire(user)


@router.post("/{user_id}/deactivate", dependencies=_WRITES)
async def deactivate_user(
    user_id: str, request: Request, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """Deactivate an account. `decide()` already treats an inactive user as
    having no roles and no cameras (`app/authorization/resolver.py`), and
    `AuthService.authenticate`/`decision_for_claims` already refuse an
    inactive user at login and on every subsequent request — this route only
    flips the flag those chokepoints already read.

    An actor may not deactivate their own account: unlike a permission
    override or a role, there is no "undo" available to a user who has just
    locked themselves out, and if they were the organisation's only
    `MANAGE_USERS` holder nobody else could undo it either.
    """
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    if _is_self(access, user):
        raise ScopeError("you may not deactivate your own account")
    if not user.is_active:
        return _user_to_wire(user)

    user.is_active = False
    await session.flush()
    await AuditTrail(session).record(
        action=AuditAction.USER_DEACTIVATED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=user.id,
        request_id=_request_id(request),
    )
    return _user_to_wire(user)


# ── role assignment ──────────────────────────────────────────────────────────


@router.post("/{user_id}/roles", dependencies=_WRITES)
async def assign_role(
    user_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Assign an existing `Role` to a user. No custom roles.

    Idempotent: assigning a role the user already holds is a no-op, matching
    `uq_role_assignment`'s own shape rather than raising a conflict for a
    request that already describes the desired end state.
    """
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    role = _role_from(payload.get("role"))

    if _is_self(access, user):
        raise ScopeError("you may not change your own roles")
    _require_grantable_role(access, role)

    already = any(
        a.role == role.value and a.organization_id == access.tenant_id
        for a in user.role_assignments
    )
    if not already:
        session.add(
            RoleAssignment(
                user_id=user.id,
                organization_id=access.tenant_id,
                role=role.value,
                granted_by=access.subject,
            )
        )
        await session.flush()
        await session.refresh(user, attribute_names=["role_assignments"])

        await AuditTrail(session).record(
            action=AuditAction.ROLE_ASSIGNED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles_tuple(access),
            resource_type="user",
            resource_id=user.id,
            request_id=_request_id(request),
            detail={"role": role.value},
        )
    return _user_to_wire(user)


@router.delete("/{user_id}/roles/{role_value}", dependencies=_WRITES)
async def remove_role(
    user_id: str,
    role_value: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Remove a role from a user.

    No requirement that the actor hold the role being removed: removing a
    role only narrows the target's access, which is not the direction rule 2
    in the module docstring exists to block. Removing a role the user does
    not hold is a no-op, mirroring `clear_permission_override`'s own
    idempotent shape.
    """
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    role = _role_from(role_value)

    if _is_self(access, user):
        raise ScopeError("you may not change your own roles")

    row = next((a for a in user.role_assignments if a.role == role.value), None)
    if row is not None:
        await session.delete(row)
        await session.flush()
        await session.refresh(user, attribute_names=["role_assignments"])

        await AuditTrail(session).record(
            action=AuditAction.ROLE_REMOVED,
            organization_id=access.tenant_id,
            actor=access.subject,
            actor_roles=_roles_tuple(access),
            resource_type="user",
            resource_id=user.id,
            request_id=_request_id(request),
            detail={"role": role.value},
        )
    return _user_to_wire(user)


# ── permission overrides ─────────────────────────────────────────────────────


@router.get("/{user_id}/permissions", dependencies=_READS)
async def list_permission_overrides(
    user_id: str, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """Every permission, its stored state (inherit/grant/revoke), and its
    computed effective result — one response, not something the caller
    reconstructs from separate role and override reads."""
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    return {
        "user_id": user.id,
        "permissions": await _permission_rows(user, access.tenant_id),
    }


@router.put("/{user_id}/permissions/{permission_value}", dependencies=_WRITES)
async def set_override(
    user_id: str,
    permission_value: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Set a GRANT or REVOKE override for one (user, permission) pair.

    Calls straight into `app.authorization.overrides.set_permission_override`
    — no reimplementation of INHERIT/GRANT/REVOKE semantics here. Self-
    modification and cross-tenant reach are refused there (structurally, and
    a second time by that function's own `_guard`); the one check that
    belongs at this layer and not that one is rule 2 (a grantor may only
    GRANT a permission they themselves currently hold), because it needs the
    caller's own request-scoped `AccessDecision`, which does not exist below
    the HTTP layer.
    """
    target = await _user_in_tenant(session, access.tenant_id, user_id)
    permission = _permission_from(permission_value)
    state = _state_from(payload.get("state"))

    if state is OverrideState.GRANT and not access.has(permission):
        raise ScopeError(
            f"cannot grant '{permission.value}': you do not hold it yourself",
            details={"permission": permission.value},
        )

    actor = await _actor(session, access)
    try:
        await set_permission_override(
            session, actor=actor, target=target, permission=permission, state=state
        )
    except ScopeError:
        # Self-modification or (structurally unreachable here, since `target`
        # is already tenant-scoped) cross-tenant — surfaced as-is; both are
        # already `AuthorizationError` subclasses with the right HTTP status.
        raise

    await session.flush()
    await session.refresh(target, attribute_names=["permission_overrides"])

    await AuditTrail(session).record(
        action=(
            AuditAction.PERMISSION_GRANTED
            if state is OverrideState.GRANT
            else AuditAction.PERMISSION_REVOKED
        ),
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=target.id,
        request_id=_request_id(request),
        detail={"permission": permission.value},
    )

    decision = decide(target, organization_id=access.tenant_id)
    return {
        "permission": permission.value,
        "state": state.value,
        "effective": decision.has(permission),
    }


@router.delete("/{user_id}/permissions/{permission_value}", dependencies=_WRITES)
async def reset_override(
    user_id: str,
    permission_value: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
) -> dict[str, Any]:
    """Reset an override back to INHERIT by deleting its row."""
    target = await _user_in_tenant(session, access.tenant_id, user_id)
    permission = _permission_from(permission_value)

    actor = await _actor(session, access)
    await clear_permission_override(session, actor=actor, target=target, permission=permission)

    await session.flush()
    await session.refresh(target, attribute_names=["permission_overrides"])

    await AuditTrail(session).record(
        action=AuditAction.PERMISSION_RESET,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=target.id,
        request_id=_request_id(request),
        detail={"permission": permission.value},
    )

    decision = decide(target, organization_id=access.tenant_id)
    return {
        "permission": permission.value,
        "state": "inherit",
        "effective": decision.has(permission),
    }


# ── camera access scope ─────────────────────────────────────────────


@router.get("/{user_id}/camera-scope", dependencies=_READS)
async def get_camera_scope(
    user_id: str, access: CurrentAccess, session: DbSession
) -> dict[str, Any]:
    """Which cameras this account reaches.

    Its own route rather than a field alone, so an administration screen can
    refresh camera access without re-reading the whole user — and so the
    absent-grant case has one authoritative answer instead of each caller
    inventing its own reading of a missing row.
    """
    user = await _user_in_tenant(session, access.tenant_id, user_id)
    return {"user_id": user.id, "camera_scope": grant_to_wire(_grant_of(user))}


@router.put("/{user_id}/camera-scope", dependencies=_WRITES)
async def set_camera_scope_route(
    user_id: str,
    request: Request,
    access: CurrentAccess,
    session: DbSession,
    payload: Annotated[dict, Body(...)],
) -> dict[str, Any]:
    """Replace this account's camera access.

    Goes straight through `app.authorization.camera_scope.set_camera_scope`,
    which owns the self-modification and cross-tenant refusals, exactly as the
    permission-override routes go through `overrides.set_permission_override`.
    The one check that belongs here and not there is the anti-escalation rule,
    because it needs the caller's own request-scoped `AccessDecision`.
    """
    target = await _user_in_tenant(session, access.tenant_id, user_id)
    scope = parse_camera_scope_request(payload)

    require_grantable_scope(access, scope)
    if scope.breadth is ScopeBreadth.LISTED:
        await resolve_camera_keys(
            session, organization_id=access.tenant_id, camera_keys=scope.camera_ids
        )

    actor = await _actor(session, access)
    await set_camera_scope(session, actor=actor, target=target, scope=scope)

    await session.flush()
    await session.refresh(target, attribute_names=["access_grants"])

    await AuditTrail(session).record(
        action=AuditAction.CAMERA_SCOPE_CHANGED,
        organization_id=access.tenant_id,
        actor=access.subject,
        actor_roles=_roles_tuple(access),
        resource_type="user",
        resource_id=target.id,
        request_id=_request_id(request),
        detail={
            "breadth": scope.breadth.value,
            # The count, not the list. Which cameras an account may watch is
            # operational detail; how wide the grant is, is the security fact
            # an audit reader needs at a glance.
            "camera_count": len(scope.camera_ids),
        },
    )
    return {"user_id": target.id, "camera_scope": grant_to_wire(_grant_of(target))}


__all__ = ["router"]
