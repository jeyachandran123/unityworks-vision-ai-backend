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
from app.users.models import AccessGrant, PermissionOverride, User


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


def decide(user: User, *, grant: AccessGrant | None = None) -> AccessDecision:
    """Build the request-scoped access decision for an authenticated user.

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
    if not user.is_active:
        # An inactive user reaches nothing. Represented as a real decision with
        # no roles and no cameras rather than as an exception, so that callers
        # handle it through the same deny path as everything else.
        return AccessDecision(
            subject=user.email,
            tenant_id=user.organization_id,
            roles=frozenset(),
            cameras=CameraScope.none(),
            display_name=user.display_name,
        )

    roles = parse_roles([a.role for a in (user.role_assignments or ())])
    granted, revoked = parse_overrides(list(user.permission_overrides or ()))
    permissions = effective_permissions(roles, granted=granted, revoked=revoked)

    org_status = parse_organization_status(
        user.organization.status if user.organization is not None else None
    )
    if org_status is OrganizationStatus.SUSPENDED:
        permissions = _suspend(permissions)

    effective = grant
    if effective is None:
        grants = list(user.access_grants or ())
        effective = grants[0] if grants else None

    return AccessDecision(
        subject=user.email,
        tenant_id=user.organization_id,
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
    "decide",
    "parse_camera_scope",
    "parse_organization_status",
    "parse_overrides",
    "parse_roles",
]
