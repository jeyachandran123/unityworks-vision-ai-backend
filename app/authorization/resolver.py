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
    Role,
    ScopeBreadth,
)
from app.users.models import AccessGrant, User


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


def decide(user: User, *, grant: AccessGrant | None = None) -> AccessDecision:
    """Build the request-scoped access decision for an authenticated user.

    The grant may be passed explicitly (when the caller has already loaded it) or
    read from the relationship. Both paths agree.
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

    effective = grant
    if effective is None:
        grants = list(user.access_grants or ())
        effective = grants[0] if grants else None

    return AccessDecision(
        subject=user.email,
        tenant_id=user.organization_id,
        roles=parse_roles([a.role for a in (user.role_assignments or ())]),
        cameras=parse_camera_scope(effective),
        site_ids=_split(effective.site_ids) if effective is not None else (),
        display_name=user.display_name,
    )


def _split(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


__all__ = ["decide", "parse_camera_scope", "parse_roles"]
