"""Writing to ``AccessGrant`` — the one place a user's camera reach is set.

The sibling of `app.authorization.overrides`, and it exists for the same
reason: the row is small, the consequences of getting it wrong are not, and
the invariants belong next to the write rather than in whichever route
happens to make it.

### The bug this was written to close

`POST /admin/users` created a `User` and its `RoleAssignment` rows and then
stopped. No `AccessGrant` was ever written, and `resolver.parse_camera_scope`
reads a missing grant as `CameraScope.none()` — correctly, because that is the
safe reading of an absent row. The result was an account that authenticated,
held every permission its role carried, and could reach **no cameras at all**.
Not an error anywhere: a working login onto an empty product.

That failure is in production today. `admin1@unityworks.local` has role
assignments and no grant row.

So camera scope is now part of creating a user rather than something a second
call might remember to do, and this module is what both paths write through.

### Deny by default is preserved, not weakened

The fix is *not* to default new users to tenant-wide access. `ScopeBreadth`
exists precisely because "no cameras" and "every camera" must never be
confusable, and quietly widening every new account would be the more dangerous
half of that confusion. The creation API requires the caller to say which of
the three states they mean, and `NONE` remains a legitimate, explicitly
chosen answer — for an account that will be scoped later, or one that never
needs video at all.

### Anti-escalation

Two rules, both narrower than they look:

1. **No self-modification.** Consistent with `overrides._guard`, and for the
   same reason: an actor who can widen their own camera reach can widen it to
   everything.
2. **An actor may not grant reach they do not have.** A `LISTED` actor cannot
   confer `ALL_IN_TENANT`, and cannot list a camera outside their own list.
   `ALL_IN_TENANT` actors may confer anything within the tenant. This mirrors
   the permission rule in `app/api/user_administration.py` ("a grantor may
   only give out what they hold") applied to the other half of access.

Narrowing is not restricted, exactly as REVOKE is not: reducing someone's
reach is not an escalation in the direction these rules exist to block.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.authorization.model import AccessDecision, CameraScope, ScopeBreadth
from app.domain.models import Camera
from app.errors import ScopeError, ValidationError
from app.users.models import AccessGrant, User


def parse_camera_scope_request(payload: dict) -> CameraScope:
    """Read a client's stated camera scope. Never inferred from what is absent.

    A missing `camera_scope` is a `ValidationError` rather than a default,
    because both plausible defaults are wrong: `NONE` silently creates the
    unusable account this module exists to prevent, and `ALL_IN_TENANT`
    silently creates an over-privileged one. The caller says which.
    """
    raw = payload.get("camera_scope")
    if not isinstance(raw, dict):
        raise ValidationError(
            "'camera_scope' is required and must be an object with a 'breadth' "
            "of 'none', 'listed' or 'all_in_tenant'; a user created without one "
            "can authenticate but reach no cameras, which is the failure this "
            "field exists to make impossible to cause by omission",
            details={"hint": {"breadth": "listed", "camera_keys": ["cam-01"]}},
        )

    try:
        breadth = ScopeBreadth(str(raw.get("breadth", "")).strip().lower())
    except ValueError as exc:
        raise ValidationError(
            "'camera_scope.breadth' must be 'none', 'listed' or 'all_in_tenant'",
            details={"breadth": raw.get("breadth")},
        ) from exc

    if breadth is not ScopeBreadth.LISTED:
        return CameraScope(breadth=breadth)

    keys = raw.get("camera_keys", [])
    if not isinstance(keys, list):
        raise ValidationError("'camera_scope.camera_keys' must be a list")
    cleaned = tuple(dict.fromkeys(str(k).strip() for k in keys if str(k).strip()))
    if not cleaned:
        raise ValidationError(
            "'camera_scope.breadth' is 'listed' but no cameras were listed; "
            "an empty list is ambiguous, so say 'none' when the answer is none"
        )
    return CameraScope.listed(cleaned)


async def resolve_camera_keys(
    session: AsyncSession, *, organization_id: str, camera_keys: tuple[str, ...]
) -> tuple[str, ...]:
    """Confirm every named camera exists **in this organization**.

    Refusing an unknown key rather than dropping it: a grant that silently
    ignores a typo hands out access the administrator did not review, and one
    that silently ignores another tenant's camera key hides the fact that they
    tried to name it.
    """
    found = set(
        (
            await session.execute(
                select(Camera.camera_key).where(
                    Camera.organization_id == organization_id,
                    Camera.camera_key.in_(camera_keys),
                )
            )
        )
        .scalars()
        .all()
    )
    missing = [k for k in camera_keys if k not in found]
    if missing:
        raise ValidationError(
            "these cameras do not exist in this organization",
            details={"camera_keys": sorted(missing)},
        )
    return camera_keys


def require_grantable_scope(actor: AccessDecision, scope: CameraScope) -> None:
    """Rule 2 — an actor may not confer camera reach they do not hold."""
    if scope.breadth is ScopeBreadth.NONE:
        return  # Narrowing to nothing is always allowed.
    if actor.cameras.breadth is ScopeBreadth.ALL_IN_TENANT:
        return
    if actor.cameras.breadth is ScopeBreadth.NONE:
        raise ScopeError(
            "you cannot grant camera access: you hold none yourself",
            details={"required": "camera access"},
        )

    if scope.breadth is ScopeBreadth.ALL_IN_TENANT:
        raise ScopeError(
            "you cannot grant access to every camera in the organization: your "
            "own access is limited to a named list",
            details={"required": "all_in_tenant"},
        )

    beyond = sorted(set(scope.camera_ids) - set(actor.cameras.camera_ids))
    if beyond:
        raise ScopeError(
            "you cannot grant access to cameras you cannot reach yourself",
            details={"camera_keys": beyond},
        )


async def set_camera_scope(
    session: AsyncSession,
    *,
    actor: User,
    target: User,
    scope: CameraScope,
    site_ids: tuple[str, ...] | None = None,
    organization_id: str = "",
) -> AccessGrant:
    """Create or replace the target's `AccessGrant` row in one organization.

    One row per user *per organization* (`uq_access_grant_user`), so this
    updates in place rather than accumulating history — the audit trail is
    where the history of a change lives, not a pile of superseded grant rows
    whose precedence nobody would be able to state.

    `organization_id` defaults to the target's home organization, which is the
    only organization any caller currently administers a user in, and which is
    what the row meant before the column existed. A caller that means a
    different one has to say so — a camera id is unique only within an
    organization, so a grant written into the wrong one would name whatever
    cameras happened to share those keys.
    """
    organization = organization_id or target.organization_id
    if actor.id == target.id:
        raise ScopeError(
            "you may not change your own camera access; an actor who can widen "
            "their own reach can widen it to everything"
        )
    if actor.organization_id != target.organization_id:
        raise ScopeError("a camera grant may not cross an organization boundary")

    grant = (
        await session.execute(
            select(AccessGrant).where(
                AccessGrant.user_id == target.id,
                AccessGrant.organization_id == organization,
            )
        )
    ).scalar_one_or_none()
    if grant is None:
        grant = AccessGrant(user_id=target.id, organization_id=organization)
        session.add(grant)

    grant.camera_breadth = scope.breadth.value
    # Meaningful only for LISTED, and cleared otherwise so a stale list can
    # never sit behind a wildcard where a later reader might find it.
    grant.camera_ids = ",".join(scope.camera_ids)
    if site_ids is not None:
        grant.site_ids = ",".join(site_ids)
    return grant


def grant_to_wire(grant: AccessGrant | None) -> dict:
    """A camera grant for the API. The absent row is reported as what it means."""
    if grant is None:
        return {"breadth": ScopeBreadth.NONE.value, "camera_keys": [], "site_ids": []}
    return {
        "breadth": str(grant.camera_breadth or "").strip().lower(),
        "camera_keys": [k for k in (grant.camera_ids or "").split(",") if k.strip()],
        "site_ids": [s for s in (grant.site_ids or "").split(",") if s.strip()],
    }


__all__ = [
    "grant_to_wire",
    "parse_camera_scope_request",
    "require_grantable_scope",
    "resolve_camera_keys",
    "set_camera_scope",
]
