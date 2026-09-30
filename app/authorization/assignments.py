"""Setting somebody's whole access in one organization at once.

The platform console's access matrix (`/platform/people/{id}/access`) states
access the way a person thinks about it: *these* permissions, in *this*
organization. The model stores it differently — one role, plus per-permission
exceptions to it (`app.authorization.model.OverrideState`) — and `decide()`
composes the two on every request as `(role permissions ∪ GRANTs) − REVOKEs`.

This module is the translation between the two, and it has one job: whatever it
writes, `decide()` must answer with exactly the stated set. The role is a
starting template and nothing more; `plan_access` computes the exceptions that
make up the difference, so the role chosen never changes what the person can do
— only how it is recorded, and how it reads on the organization's own user page.

### It writes through the guarded modules, never around them

Overrides go through `overrides.replace_permission_overrides` and camera reach
through `camera_scope.set_camera_scope`, which hold the no-self-modification and
must-be-a-member guards. Membership and role rows are written here, exactly as
`app/api/platform_administration.py` and `user_administration.py` write them.

### Enforcement is untouched

Nothing here is read on a request. The rows it writes are the same rows an
Organization Admin writes from `/admin/users/:id`, and every route keeps gating
on `requires(Permission.X)` through `decide()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.authorization.camera_scope import (
    grant_to_wire,
    require_grantable_scope,
    set_camera_scope,
)
from app.authorization.model import (
    AccessDecision,
    CameraScope,
    OrganizationStatus,
    Permission,
    Role,
    ScopeBreadth,
    effective_permissions,
    permissions_for,
)
from app.authorization.overrides import replace_permission_overrides
from app.authorization.resolver import decide, parse_overrides, parse_roles
from app.errors import ScopeError, ValidationError
from app.users.models import Organization, OrganizationMembership, RoleAssignment, User

#: The two camera breadths the platform door accepts. `listed` names specific
#: cameras, which only the organization's own administration can see to choose
#: — so only that door passes `allow_listed=True`.
_SETTABLE_BREADTHS = frozenset({ScopeBreadth.NONE, ScopeBreadth.ALL_IN_TENANT})


@dataclass(frozen=True, slots=True)
class AccessPlan:
    """The rows that make `decide()` answer a stated set."""

    role: Role | None
    granted: frozenset[Permission]
    revoked: frozenset[Permission]


def plan_access(role: Role | None, permissions: frozenset[Permission]) -> AccessPlan:
    """The role and exceptions that make `decide()` answer exactly `permissions`.

    `(role ∪ granted) − revoked == permissions` for every input, because
    `granted` is what the role lacks and `revoked` is what it has too much of.
    """
    role_permissions = permissions_for(frozenset({role})) if role is not None else frozenset()
    return AccessPlan(
        role=role,
        granted=permissions - role_permissions,
        revoked=role_permissions - permissions,
    )


@dataclass(frozen=True, slots=True)
class OrganizationAccess:
    """One organization's worth of an access request, already parsed."""

    organization_id: str
    role: Role | None
    permissions: frozenset[Permission]
    #: `None` leaves camera reach as it is — which is how a list of specific
    #: cameras, set inside the organization, survives a change made here.
    camera_breadth: ScopeBreadth | None = None
    #: The named cameras, for `LISTED` only — and only ever from inside the
    #: organization, which is the one place that can see its cameras to pick.
    camera_keys: tuple[str, ...] = ()

    @property
    def camera_scope(self) -> CameraScope | None:
        """The stated camera reach as a scope, or `None` when not stated."""
        if self.camera_breadth is None:
            return None
        if self.camera_breadth is ScopeBreadth.LISTED:
            return CameraScope.listed(self.camera_keys)
        return CameraScope(breadth=self.camera_breadth)

    @property
    def touches(self) -> bool:
        """Whether this asks for anything at all. An untouched organization the
        person is not in is left alone rather than joined with nothing."""
        return (
            self.role is not None
            or bool(self.permissions)
            or self.camera_breadth is ScopeBreadth.ALL_IN_TENANT
        )


@dataclass(frozen=True, slots=True)
class AppliedAccess:
    """What one organization's write did, for the audit row."""

    organization_id: str
    admitted: bool
    before: frozenset[Permission]
    after: frozenset[Permission]
    plan: AccessPlan
    camera_breadth: ScopeBreadth | None


def parse_organization_access(raw: Any, *, allow_listed: bool = False) -> OrganizationAccess:
    """Read one `{organization_id, role, permissions, camera_breadth}` item.

    Every unknown value is a refusal, never a drop: a matrix that silently
    ignored a permission it could not read would save something other than what
    the operator looked at and approved.

    `allow_listed` admits `camera_breadth: "listed"` with `camera_keys` — the
    organization's own door passes it; the platform's does not.
    """
    if not isinstance(raw, dict):
        raise ValidationError("each organization entry must be an object")

    organization_id = str(raw.get("organization_id", "") or "").strip()
    if not organization_id:
        raise ValidationError("'organization_id' is required")

    role: Role | None = None
    role_value = raw.get("role")
    if role_value not in (None, ""):
        try:
            role = Role(str(role_value).strip().lower())
        except ValueError as exc:
            raise ValidationError(
                f"'{role_value}' is not a known role", details={"role": role_value}
            ) from exc

    values = raw.get("permissions", [])
    if not isinstance(values, list):
        raise ValidationError("'permissions' must be a list")
    permissions: set[Permission] = set()
    for value in values:
        try:
            permissions.add(Permission(str(value).strip().lower()))
        except ValueError as exc:
            raise ValidationError(
                f"'{value}' is not a known permission", details={"permission": value}
            ) from exc

    breadth: ScopeBreadth | None = None
    keys: tuple[str, ...] = ()
    if raw.get("camera_breadth") is not None:
        try:
            breadth = ScopeBreadth(str(raw["camera_breadth"]).strip().lower())
        except ValueError:
            breadth = None
        settable = _SETTABLE_BREADTHS | ({ScopeBreadth.LISTED} if allow_listed else set())
        if breadth not in settable:
            raise ValidationError(
                (
                    "'camera_breadth' must be 'none', 'listed' or 'all_in_tenant'"
                    if allow_listed
                    else "'camera_breadth' must be 'none' or 'all_in_tenant' here; specific "
                    "cameras are chosen inside the organization"
                ),
                details={"camera_breadth": raw["camera_breadth"]},
            )
        if breadth is ScopeBreadth.LISTED:
            values = raw.get("camera_keys", [])
            if not isinstance(values, list):
                raise ValidationError("'camera_keys' must be a list")
            keys = tuple(dict.fromkeys(str(k).strip() for k in values if str(k).strip()))
            if not keys:
                # An empty list is ambiguous — and the platform reads an empty
                # camera tuple as *every* camera. Say `none` when it is none.
                raise ValidationError(
                    "'camera_breadth' is 'listed' but no cameras were named; say "
                    "'none' when the answer is none"
                )

    return OrganizationAccess(
        organization_id=organization_id,
        role=role,
        permissions=frozenset(permissions),
        camera_breadth=breadth,
        camera_keys=keys,
    )


def parse_access_items(raw: Any) -> list[OrganizationAccess]:
    """Read the `organizations` list. An organization named twice is refused:
    the second entry would silently overwrite the first."""
    if not isinstance(raw, list):
        raise ValidationError("'organizations' must be a list")
    items = [parse_organization_access(item) for item in raw]
    seen: set[str] = set()
    for item in items:
        if item.organization_id in seen:
            raise ValidationError(
                "an organization may appear only once",
                details={"organization_id": item.organization_id},
            )
        seen.add(item.organization_id)
    return items


def _roles_in(user: User, organization_id: str) -> frozenset[Role]:
    return parse_roles(
        [a.role for a in (user.role_assignments or ()) if a.organization_id == organization_id]
    )


def intended_permissions(user: User, organization_id: str) -> frozenset[Permission]:
    """`(role permissions ∪ GRANTs) − REVOKEs` in one organization, before suspension.

    What the matrix shows. Distinct from `decide()`, which also narrows a
    suspended organization: a Manage tick stored there is still the person's
    access, it just does not take effect until the organization is restored.
    """
    granted, revoked = parse_overrides(
        [o for o in (user.permission_overrides or ()) if o.organization_id == organization_id]
    )
    return effective_permissions(_roles_in(user, organization_id), granted=granted, revoked=revoked)


def template_role(user: User, organization_id: str) -> Role | None:
    """The role the matrix starts from: the first held, in `Role` declaration order."""
    held = _roles_in(user, organization_id)
    return next((role for role in Role if role in held), None)


async def load_for_access(session: AsyncSession, user_id: str) -> User | None:
    """A user with every relationship `decide()` and this module read.

    `populate_existing` because a caller that has just written rows through this
    session needs the collections re-read, not the copies loaded before the write.
    """
    return (
        await session.execute(
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.role_assignments),
                selectinload(User.permission_overrides),
                selectinload(User.access_grants),
                selectinload(User.organization),
                selectinload(User.memberships).selectinload(OrganizationMembership.organization),
            )
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def apply_organization_access(
    session: AsyncSession,
    *,
    actor: User,
    target: User,
    access: OrganizationAccess,
    granted_by: str,
) -> AppliedAccess | None:
    """Make `target`'s access in one organization exactly `access`.

    Returns `None`, having written nothing, for an organization the target is
    not in and that the request does not touch. Otherwise, in order:
    membership (when missing), roles (exactly the template), overrides (exactly
    the difference), camera breadth (when stated).

    The caller has already checked that the organization exists and is not
    archived; this function does not re-read it.
    """
    organization_id = access.organization_id
    is_member = any(m.organization_id == organization_id for m in (target.memberships or ()))
    if not is_member and not access.touches:
        return None

    before = intended_permissions(target, organization_id) if is_member else frozenset()

    camera_scope = access.camera_scope
    if camera_scope is None and not is_member:
        # `remove_member` keeps a person's rows so that an undo is lossless —
        # an every-camera grant included. The matrix shows a non-member as "no
        # cameras", so admitting one without a stated breadth must mean that,
        # never whatever a removed membership left behind: deny by default.
        camera_scope = CameraScope.none()

    if not is_member:
        session.add(
            OrganizationMembership(
                user_id=target.id, organization_id=organization_id, granted_by=granted_by
            )
        )
        # The override and camera guards read membership from the table.
        await session.flush()

    plan = plan_access(access.role, access.permissions)

    keep = plan.role.value if plan.role is not None else None
    current = [a for a in (target.role_assignments or ()) if a.organization_id == organization_id]
    for assignment in current:
        if assignment.role != keep:
            await session.delete(assignment)
    if keep is not None and all(a.role != keep for a in current):
        session.add(
            RoleAssignment(
                user_id=target.id,
                organization_id=organization_id,
                role=keep,
                granted_by=granted_by,
            )
        )
    await session.flush()

    await replace_permission_overrides(
        session,
        actor=actor,
        target=target,
        organization_id=organization_id,
        granted=plan.granted,
        revoked=plan.revoked,
    )

    if camera_scope is not None:
        await set_camera_scope(
            session,
            actor=actor,
            target=target,
            scope=camera_scope,
            organization_id=organization_id,
        )
    await session.flush()

    return AppliedAccess(
        organization_id=organization_id,
        admitted=not is_member,
        before=before,
        after=access.permissions,
        plan=plan,
        camera_breadth=camera_scope.breadth if camera_scope is not None else None,
    )


def refuse_escalation(
    actor: AccessDecision, access: OrganizationAccess, *, target: User | None
) -> None:
    """Refuse a grid that would give more than the actor holds.

    The organization's own door, not the platform's: a Platform Admin holds
    every permission by entry, while somebody inside an organization who may
    administer people can hold much less. This is `user_administration`'s rule
    2 — a grantor may only give out what they hold — applied to a whole grid:

    * a permission newly ticked must be one the actor holds;
    * a template role newly recorded must carry nothing the actor lacks — the
      role outlives the ticks, and clearing the exceptions later would hand
      over all of it;
    * camera reach may not exceed the actor's own (`require_grantable_scope`).

    Removing is never restricted. Narrowing somebody is not an escalation in
    the direction this rule exists to block.

    `target` is `None` for somebody being created, who holds nothing yet.
    """
    organization_id = access.organization_id
    before = intended_permissions(target, organization_id) if target is not None else frozenset()
    beyond = (access.permissions - before) - actor.permissions
    if beyond:
        raise ScopeError(
            "you cannot give a permission you do not hold yourself",
            details={"missing": sorted(p.value for p in beyond)},
        )

    held_roles = _roles_in(target, organization_id) if target is not None else frozenset()
    if access.role is not None and access.role not in held_roles:
        missing = permissions_for(frozenset({access.role})) - actor.permissions
        if missing:
            raise ScopeError(
                f"cannot record role '{access.role.value}': it carries a permission "
                "you do not hold",
                details={"role": access.role.value, "missing": sorted(p.value for p in missing)},
            )

    scope = access.camera_scope
    if scope is not None:
        require_grantable_scope(actor, scope)


def organization_access_to_wire(
    user: User, organization: Organization, *, is_member: bool
) -> dict[str, Any]:
    """One organization's worth of a person's access, as the matrix reads it.

    Shared by the platform console (every organization) and the organization's
    own User & Access (just that one), so the two grids read the same shape.
    `permissions` is `(role ∪ GRANTs) − REVOKEs`; `effective` is what `decide()`
    gives on the next request — they differ only in a suspended organization.
    """
    base = {
        "organization_id": organization.id,
        "organization_name": organization.name,
        "status": str(organization.status or OrganizationStatus.ACTIVE.value).strip().lower(),
        "is_member": is_member,
        "is_home": user.organization_id == organization.id,
    }
    if not is_member:
        # Roles left behind by a removed membership grant nothing, and are not
        # shown as if they did.
        return {
            **base,
            "role": None,
            "roles": [],
            "permissions": [],
            "effective": [],
            "camera_scope": grant_to_wire(None),
        }
    template = template_role(user, organization.id)
    grant = next(
        (g for g in (user.access_grants or ()) if g.organization_id == organization.id), None
    )
    return {
        **base,
        "role": template.value if template is not None else None,
        "roles": sorted(
            a.role for a in (user.role_assignments or ()) if a.organization_id == organization.id
        ),
        "permissions": sorted(p.value for p in intended_permissions(user, organization.id)),
        "effective": sorted(
            p.value for p in decide(user, organization_id=organization.id).permissions
        ),
        "camera_scope": grant_to_wire(grant),
    }


def access_audit_detail(user: User, applied: AppliedAccess) -> dict[str, Any]:
    """The `user.access_set` row's detail: what the change made them able to do."""
    return {
        "email": user.email,
        "role": applied.plan.role.value if applied.plan.role is not None else None,
        "added": sorted(p.value for p in applied.after - applied.before),
        "removed": sorted(p.value for p in applied.before - applied.after),
        "granted": sorted(p.value for p in applied.plan.granted),
        "revoked": sorted(p.value for p in applied.plan.revoked),
        "camera_breadth": (
            applied.camera_breadth.value if applied.camera_breadth is not None else None
        ),
        "admitted": applied.admitted,
    }


__all__ = [
    "AccessPlan",
    "AppliedAccess",
    "OrganizationAccess",
    "access_audit_detail",
    "apply_organization_access",
    "intended_permissions",
    "load_for_access",
    "organization_access_to_wire",
    "parse_access_items",
    "parse_organization_access",
    "plan_access",
    "refuse_escalation",
    "template_role",
]
