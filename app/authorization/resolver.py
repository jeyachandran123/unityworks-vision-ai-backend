"""Turning a stored user into an ``AccessDecision``.

This is the only place the database's representation of access becomes the
in-memory one. Two rules hold here and are tested:

**Unknown values deny.** A ``role`` string that no longer maps to a ``Role``, or
a ``camera_breadth`` that is not a ``ScopeBreadth``, yields *less* access, never
more. A row written by a newer version of the application must not grant
something an older one cannot reason about.

**A missing grant is no access.** Not tenant-wide access, not an empty list
passed onward — no access, expressed by ``ScopeBreadth.NONE``, which
``AccessDecision.to_grant()`` then refuses to convert into a Vision OS grant at
all.

**Access is resolved *in* an organization, never merely *for* a user.**
``decide()`` takes the active organization and reads only the rows that name it.
Somebody who is ``org_admin`` at one customer and an auditor at another gets
exactly one of those two answers per request — chosen by the tenant on the
token, and by nothing the caller sent.
"""

from __future__ import annotations

from app.authorization.model import (
    AccessDecision,
    CameraScope,
    OrganizationStatus,
    OverrideState,
    Permission,
    Role,
    ScopeBreadth,
    effective_permissions,
)
from app.users.models import (
    AccessGrant,
    OrganizationMembership,
    PermissionOverride,
    User,
)


def parse_roles(values: list[str] | tuple[str, ...]) -> frozenset[Role]:
    """Map stored role strings to ``Role``. Unrecognised values are dropped.

    Dropped rather than raising: one stale row should narrow that user's access,
    not break every request in the process.
    """
    roles: set[Role] = set()
    for value in values:
        try:
            roles.add(Role(str(value).strip().lower()))
        except ValueError:
            continue
    return frozenset(roles)


def parse_camera_scope(grant: AccessGrant | None) -> CameraScope:
    """Read a stored grant into an explicit three-state camera scope."""
    if grant is None:
        return CameraScope.none()

    try:
        breadth = ScopeBreadth(str(grant.camera_breadth).strip().lower())
    except ValueError:
        # An unreadable breadth is the most dangerous field in the schema to
        # guess at, because one of the guesses is "every camera".
        return CameraScope.none()

    if breadth is ScopeBreadth.ALL_IN_TENANT:
        return CameraScope.all_in_tenant()
    if breadth is ScopeBreadth.NONE:
        return CameraScope.none()

    ids = _split(grant.camera_ids)
    if not ids:
        # Breadth says "listed" and nothing is listed. The row is inconsistent,
        # and the safe reading of an inconsistent grant is none.
        return CameraScope.none()
    return CameraScope.listed(ids)


def parse_overrides(
    rows: list[PermissionOverride] | tuple[PermissionOverride, ...],
) -> tuple[frozenset[Permission], frozenset[Permission]]:
    """Read stored override rows into explicit GRANT and REVOKE sets.

    Unrecognised permission or state values are dropped rather than raising —
    the same "unknown values deny" discipline as `parse_roles`: a row an older
    or newer version of the application cannot read must never silently widen
    access, so it is treated as if it were not there (neither granted nor
    revoked) rather than guessed at.
    """
    granted: set[Permission] = set()
    revoked: set[Permission] = set()
    for row in rows:
        try:
            permission = Permission(str(row.permission).strip().lower())
        except ValueError:
            continue
        try:
            state = OverrideState(str(row.state).strip().lower())
        except ValueError:
            continue
        if state is OverrideState.GRANT:
            granted.add(permission)
        else:
            revoked.add(permission)
    return frozenset(granted), frozenset(revoked)


def parse_organization_status(value: str | None) -> OrganizationStatus:
    """Read a stored organization status, denying toward the narrower reading.

    A missing value reads as ``ACTIVE`` — the column's own default, and the
    behavior every row had before this column existed. An unreadable value is
    different: it is not "no opinion", it is a row this version cannot make
    sense of, and the safe reading of that is the narrower of the two
    non-default states rather than the widest one — the same "unknown values
    deny" rule `parse_camera_scope` applies to a bad ``camera_breadth``.
    """
    if value is None:
        return OrganizationStatus.ACTIVE
    try:
        return OrganizationStatus(str(value).strip().lower())
    except ValueError:
        return OrganizationStatus.SUSPENDED


#: What a SUSPENDED organization may no longer do. Stated as a set, not
#: inferred from a name.
#:
#: This was a `manage_` prefix filter, which happened to be right for every
#: permission that existed when it was written and is a trap for every one
#: added since. `RETIRE_CAMERAS` is the proof: retiring a camera destroys an
#: observation partition irreversibly, it is the single most destructive act in
#: the product, and it sailed through a suspension because its name does not
#: begin with `manage_`. A rule that depends on how a permission was spelled is
#: not a rule about what it does.
#:
#: `DELETE_EVIDENCE` is here for the same reason. Nothing about a billing hold
#: should make it easier to erase records than to write them.
#:
#: Deliberately absent: every read, and the incident work queue
#: (`ACKNOWLEDGE_INCIDENTS`, `RESOLVE_INCIDENTS`). Suspension is a commercial
#: state; the incidents are real food-safety findings about a kitchen that is
#: still operating, and refusing to let anyone close one would leave a genuine
#: violation open because an invoice is late. That is a deliberate reading and
#: it is written down here so that changing it is a decision rather than an
#: edit.
SUSPENDED_FORBIDDEN: frozenset[Permission] = frozenset(
    {
        Permission.MANAGE_ORGANIZATION,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SITES,
        Permission.MANAGE_ZONES,
        Permission.MANAGE_CAMERAS,
        Permission.RETIRE_CAMERAS,
        Permission.MANAGE_TABLE_OCCUPANCY,
        Permission.MANAGE_CUTTING_BOARD,
        Permission.MANAGE_POS_INTEGRATION,
        Permission.MANAGE_PATRON_ID,
        Permission.DELETE_EVIDENCE,
        Permission.REGISTER_DEMAND,
    }
)


def _suspend(permissions: frozenset[Permission]) -> frozenset[Permission]:
    """The write surfaces a SUSPENDED organization loses, while login and reads
    continue working. See `SUSPENDED_FORBIDDEN`."""
    return permissions - SUSPENDED_FORBIDDEN


def membership_for(user: User, organization_id: str) -> OrganizationMembership | None:
    """This user's membership in one organization, or ``None``.

    The question every organization-scoped decision starts from. It reads the
    loaded relationship rather than issuing its own query, so that a caller
    which already holds the user cannot accidentally answer it against a
    different snapshot of the database than the one it is deciding on.
    """
    for membership in user.memberships or ():
        if membership.organization_id == organization_id:
            return membership
    return None


def accessible_organization_ids(user: User) -> tuple[str, ...]:
    """Every organization this user may enter. Sorted, so it is comparable.

    Derived from memberships alone. ``User.organization_id`` is deliberately not
    added as a special case: the migration writes an explicit membership for it,
    which means "may enter" has exactly one source — and a home organization
    whose membership was deliberately revoked stays revoked.
    """
    return tuple(sorted({m.organization_id for m in (user.memberships or ())}))


def _organization_status(user: User, organization_id: str) -> OrganizationStatus:
    """The *active* organization's status, not the home organization's.

    Reading ``user.organization.status`` here would suspend the wrong customer
    in both directions: a user whose home organization is suspended would keep
    full write access everywhere else they are a member, and a user working
    inside a suspended organization would keep theirs because their own account
    lives somewhere healthy.
    """
    membership = membership_for(user, organization_id)
    if membership is not None and membership.organization is not None:
        return parse_organization_status(membership.organization.status)
    if user.organization is not None and user.organization_id == organization_id:
        return parse_organization_status(user.organization.status)
    # No organization row to read. The narrower reading, for the same reason
    # `parse_organization_status` prefers SUSPENDED to a guess.
    return OrganizationStatus.SUSPENDED


def decide(user: User, *, organization_id: str, grant: AccessGrant | None = None) -> AccessDecision:
    """Build the request-scoped access decision for an authenticated user, in one
    organization.

    ``organization_id`` is required rather than defaulted to the user's home
    organization. A default here would be the single line that reintroduces the
    bug this change exists to prevent: a caller that forgot to say which
    organization it meant would silently get the home one, which for a
    multi-organization user is a different customer's data than the token asked
    for.

    Only rows naming that organization are read — roles, overrides and the
    camera grant alike. Holding ``org_admin`` somewhere else contributes nothing
    here.

    The grant may be passed explicitly (when the caller has already loaded it) or
    read from the relationship. Both paths agree.

    Effective permissions are ``(role permissions ∪ GRANTs) − REVOKEs``,
    recomputed here from the freshly loaded rows on every call — the same
    "rebuilt, never cached" guarantee `decision_for_claims` already gives role
    changes now extends to override changes, with no separate cache to
    invalidate. A SUSPENDED organization narrows the result further, after
    overrides are applied, so a REVOKE and a suspension compose rather than
    one hiding the other.
    """
    if not organization_id:
        raise ValueError("an access decision must name the organization it is being made in")

    if not user.is_active:
        # An inactive user reaches nothing. Represented as a real decision with
        # no roles and no cameras rather than as an exception, so that callers
        # handle it through the same deny path as everything else.
        return AccessDecision(
            subject=user.email,
            tenant_id=organization_id,
            roles=frozenset(),
            cameras=CameraScope.none(),
            display_name=user.display_name,
        )

    if membership_for(user, organization_id) is None:
        # A backstop, not the control. `decision_for_claims` refuses a token
        # naming an organization the user is not a member of before reaching
        # here, and refusing *there* is what makes a revoked membership take
        # effect on the very next request. This exists so that a future caller
        # which skips that door still cannot manufacture reach out of nothing.
        return AccessDecision(
            subject=user.email,
            tenant_id=organization_id,
            roles=frozenset(),
            cameras=CameraScope.none(),
            display_name=user.display_name,
            permissions=frozenset(),
        )

    roles = parse_roles(
        [
            assignment.role
            for assignment in (user.role_assignments or ())
            if assignment.organization_id == organization_id
        ]
    )
    granted, revoked = parse_overrides(
        [
            override
            for override in (user.permission_overrides or ())
            if override.organization_id == organization_id
        ]
    )
    permissions = effective_permissions(roles, granted=granted, revoked=revoked)

    if _organization_status(user, organization_id) is OrganizationStatus.SUSPENDED:
        permissions = _suspend(permissions)

    effective = grant
    if effective is None:
        grants = [
            candidate
            for candidate in (user.access_grants or ())
            if candidate.organization_id == organization_id
        ]
        effective = grants[0] if grants else None

    return AccessDecision(
        subject=user.email,
        tenant_id=organization_id,
        roles=roles,
        cameras=parse_camera_scope(effective),
        site_ids=_split(effective.site_ids) if effective is not None else (),
        display_name=user.display_name,
        permissions=permissions,
    )


def _split(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


__all__ = [
    "SUSPENDED_FORBIDDEN",
    "accessible_organization_ids",
    "decide",
    "membership_for",
    "parse_camera_scope",
    "parse_organization_status",
    "parse_overrides",
    "parse_roles",
]
