"""Writing to ``PermissionOverride`` — the one place a row is created or changed.

Not an API route. Stage 5/6 build the `MANAGE_USERS`-gated HTTP surface that
calls this; today it is reachable from Python only (tests, and any internal
service code), which is exactly the scope this pass is asked to cover.

Two structural guards live here rather than being left to a caller to
remember:

**No self-modification.** A user may not grant or revoke a permission for
themselves — `actor.id == target.id` is refused unconditionally, regardless of
what the actor already holds. Without it, any user who could reach this
function at all could hand themselves anything.

**Cross-tenant modification is structurally impossible, not merely checked.**
An override keys off `user_id`, and `user_id` already resolves to exactly one
`organization_id` through the `users` table — so before this function can even
compare tenants, `target` had to be loaded from *some* row, and that row
already names one organization. The `organization_id` equality check below is
belt-and-suspenders: it exists so a caller that loads `actor` and `target` from
two different queries without itself enforcing tenancy still gets refused here,
rather than relying on every future caller to have gotten that right.

What this module deliberately does **not** do: decide whether the actor is
*permitted* to grant or revoke the specific permission in question (the
"grantor can only grant what they hold" rule the architecture freeze calls the
asymmetric anti-escalation rule for REVOKE). That rule reads as `MANAGE_USERS`
in the target's tenant plus, for GRANT, holding the permission being granted —
and enforcing it meaningfully needs the admin API's own request-scoped
`AccessDecision`, which does not exist yet. Building half of it here would be
worse than not building it: a partial check that looks complete invites a
caller to skip the real one once the API arrives. That is Stage 5/6 work and is
deliberately left to it.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.authorization.model import OverrideState, Permission
from app.errors import ScopeError
from app.users.models import PermissionOverride, User


async def set_permission_override(
    session: AsyncSession,
    *,
    actor: User,
    target: User,
    permission: Permission,
    state: OverrideState,
) -> PermissionOverride:
    """Create or update one (user, permission) override row.

    Idempotent: setting the same state twice updates `granted_at`/`granted_by`
    on the existing row rather than creating a duplicate, which is what the
    `uq_permission_override_user_permission` constraint would refuse anyway.
    """
    _guard(actor, target)

    # The target's home organization: the only one an administrator currently
    # administers a user in, and what the row meant before the column existed.
    organization = target.organization_id

    existing = await session.execute(
        select(PermissionOverride).where(
            PermissionOverride.user_id == target.id,
            PermissionOverride.organization_id == organization,
            PermissionOverride.permission == permission.value,
        )
    )
    row = existing.scalar_one_or_none()
    if row is None:
        row = PermissionOverride(
            user_id=target.id,
            organization_id=organization,
            permission=permission.value,
            state=state.value,
            granted_by=actor.id,
        )
        session.add(row)
    else:
        row.state = state.value
        row.granted_by = actor.id
    return row


async def clear_permission_override(
    session: AsyncSession,
    *,
    actor: User,
    target: User,
    permission: Permission,
) -> None:
    """Remove an override, reverting the permission to INHERIT (role behavior).

    Removing a row that does not exist is a no-op: the end state — no override
    — is what the caller asked for either way.
    """
    _guard(actor, target)

    existing = await session.execute(
        select(PermissionOverride).where(
            PermissionOverride.user_id == target.id,
            PermissionOverride.permission == permission.value,
        )
    )
    row = existing.scalar_one_or_none()
    if row is not None:
        await session.delete(row)


def _guard(actor: User, target: User) -> None:
    if actor.id == target.id:
        raise ScopeError(
            "a user may not grant or revoke a permission for themselves",
            details={"user_id": actor.id},
        )
    if actor.organization_id != target.organization_id:
        raise ScopeError(
            "cross-tenant permission overrides are not permitted",
            details={
                "actor_organization_id": actor.organization_id,
                "target_organization_id": target.organization_id,
            },
        )


__all__ = ["clear_permission_override", "set_permission_override"]
