"""The platform operator — the one principal that is not a tenant principal.

### Why this is a separate type and not a role

`AccessDecision` is structurally tenant-scoped. `tenant_id` is mandatory, it is
validated in `__post_init__`, and every scope and grant built from it carries
it. That is not incidental — it is the property that makes cross-tenant
leakage impossible to write rather than merely discouraged.

Managing organizations is the one job that has to reach across that boundary,
and there are two ways to give it that reach:

1. Widen `super_admin` until it means "every tenant". This is the tempting one
   and it is wrong. `super_admin` is a *tenant* role — `ROLE_PERMISSIONS` says
   so, `AccessDecision` builds it inside one organization, and every route
   gated on one of its permissions assumes a tenant is in scope. Redefining it
   would silently convert every existing `super_admin` account in every
   customer into a cross-customer superuser, and it would make `tenant_id`
   advisory in a type whose whole value is that it is not.

2. Make the operator a different kind of principal, with a different type,
   resolved by a different dependency, that no role can produce.

This module is the second. `PlatformOperator` has no `tenant_id` because it
genuinely has none, and it carries no `Permission` set because tenant
permissions do not mean anything at this altitude. An `AccessDecision` can
never become one, and a `PlatformOperator` can never be passed to a route
expecting an `AccessDecision` — the type system refuses both directions.

### Becoming one

A row in `platform_operator_grants`, and nothing else. Not a role, not a flag
on `User`, not a setting. It is deliberately not reachable from the
administration API at all: an organization admin who could mint platform
operators would be a platform operator, and the tenant boundary would be a
formality. Granting it is an out-of-band act performed with
`scripts/manage.py grant-operator`, which is exactly as awkward as it should
be for a privilege that reaches every customer's data.

The migration that creates the table deliberately seeds **nobody**. A
privilege that arrives switched on for whoever happened to be in the database
is a privilege nobody decided to give.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import ScopeError
from app.users.models import PlatformOperatorGrant, User


@dataclass(frozen=True, slots=True)
class PlatformOperator:
    """Someone who administers organizations rather than working inside one.

    Deliberately minimal. It answers "who is this" and "may they act", and it
    carries nothing that could be mistaken for tenant authority — no
    `tenant_id`, no `Permission` set, no `CameraScope`. An operator manages the
    *existence and lifecycle* of organizations; reading a customer's incidents
    or watching their cameras is not part of the job and is not reachable from
    this type.
    """

    subject: str
    user_id: str
    display_name: str = ""
    #: The organization this operator's own account lives in. Recorded for the
    #: audit trail only, and never used as a scope: an operator's home tenant
    #: confers nothing and restricts nothing.
    home_organization_id: str = ""

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a platform operator must name a subject")


async def resolve_operator(session: AsyncSession, user: User) -> PlatformOperator:
    """Turn an authenticated user into an operator, or refuse.

    Refuses an inactive account for the same reason `decide()` gives an
    inactive user no roles: deactivation must take effect immediately, and an
    operator grant that outlived it would be the most valuable stale
    credential in the system.
    """
    if not user.is_active:
        raise ScopeError("this account is not active")

    granted = (
        await session.execute(
            select(PlatformOperatorGrant.id).where(PlatformOperatorGrant.user_id == user.id)
        )
    ).scalar_one_or_none()
    if granted is None:
        raise ScopeError(
            "this account is not a platform operator; managing organizations is "
            "not something a role inside an organization can confer",
            details={"required": "platform_operator"},
        )

    return PlatformOperator(
        subject=user.email,
        user_id=user.id,
        display_name=user.display_name,
        home_organization_id=user.organization_id,
    )


__all__ = ["PlatformOperator", "resolve_operator"]
