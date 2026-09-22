"""The platform operator — the one principal that is not a tenant principal.

### Why this is a separate type and not a role

`AccessDecision` is structurally tenant-scoped. `tenant_id` is mandatory, it is
validated in `__post_init__`, and every scope and grant built from it carries
it. That is not incidental — it is the property that makes cross-tenant
leakage impossible to write rather than merely discouraged.

Managing organizations is the one job that has to reach across that boundary,
and there are two ways to give it that reach:

1. Widen `org_admin`, the top tenant role, until it means "every tenant".
   This is the tempting one and it is wrong. `org_admin` is a *tenant* role —
   `ROLE_PERMISSIONS` says so, `AccessDecision` builds it inside one
   organization, and every route gated on one of its permissions assumes a
   tenant is in scope. Redefining it would silently convert every customer's
   Organization Admin into a cross-customer superuser, and it would make
   `tenant_id` advisory in a type whose whole value is that it is not.

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

from app.auth.tokens import PLATFORM_ACT
from app.authorization.model import AccessDecision, CameraScope, Permission
from app.authorization.resolver import SUSPENDED_FORBIDDEN
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
    #: Empty: a Platform Admin belongs to no organization. Kept as a field so
    #: an account that predates that rule is still reported truthfully in the
    #: audit trail; never used as a scope either way.
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
        home_organization_id=user.organization_id or "",
    )


#: The claim that marks a token minted by `POST /platform/organizations/{id}/enter`.
#:
#: Its presence is what tells `decision_for_claims` that the tenant on this
#: token was reached by an audited platform entry rather than by a membership,
#: and that the reach to build is `OPERATOR_ENTRY_PERMISSIONS` rather than
#: whatever roles the account happens to hold in that organization. A token
#: without it is an ordinary tenant token and is resolved the ordinary way.
ACTING_AS_PLATFORM_OPERATOR = PLATFORM_ACT


#: What a platform operator may do inside an organization they have entered:
#: **everything.**
#:
#: The Platform Admin owns every organization on the deployment and can do
#: anything inside any of them (product-owner decision, 2026-09-22). This set
#: used to be a hand-written read-only list that withheld evidence, patron
#: identity, the audit trail, exports and every write. It is now the complete
#: set, stated as such so that a permission added later reaches an entered
#: Platform Admin without anybody having to remember to list it.
#:
#: ### What still makes entry different from membership
#:
#: * it is a separate, deliberate act (`POST .../enter`), not an ambient reach;
#: * it writes an audit row into the entered organization *before* it returns,
#:   so the customer's own trail shows that the platform came in;
#: * its token carries `act: platform_operator`, so every request made inside
#:   is attributable to the entry rather than to the customer's own staff;
#: * suspension still applies — see `entry_decision`.
OPERATOR_ENTRY_PERMISSIONS: frozenset[Permission] = frozenset(Permission)


def entry_decision(
    operator: PlatformOperator, organization_id: str, *, suspended: bool = False
) -> AccessDecision:
    """The tenant-scoped reach of an operator who has entered an organization.

    An `AccessDecision`, because every route in the application already speaks
    that language and inventing a second authorization type for this would be
    the parallel tenant-context mechanism the architecture exists to avoid. It
    is built here rather than by the resolver because it is not resolved from
    anything stored about the user *in that organization* — there is nothing
    stored, which is the entire point.

    **Suspension narrows it**, exactly as it narrows a member's session:
    `SUSPENDED_FORBIDDEN` is subtracted when the organization is suspended. The
    entry never went through `decide()`, which is where members get that
    narrowing, and while entry was read-only it did not need to. With every
    permission in reach it does — a suspended customer is one whose writes are
    stopped, and platform authority is not an exemption from that inside the
    customer. The suspension itself is lifted from the console.

    **No roles.** `roles=frozenset()` and `permissions` stated explicitly, so
    the decision cannot be mistaken for one produced by a role and cannot pick
    up a permission by having a role redefined under it later.

    **Every camera, no sites.** `all_in_tenant` because a console showing an
    operator an empty wall would be indistinguishable from a customer whose
    cameras are all down, which is precisely the question entry is for. No site
    ids for the same reason a tenant-wide camera scope needs none.
    """
    return AccessDecision(
        subject=operator.subject,
        tenant_id=organization_id,
        roles=frozenset(),
        cameras=CameraScope.all_in_tenant(),
        display_name=operator.display_name,
        permissions=(
            OPERATOR_ENTRY_PERMISSIONS - SUSPENDED_FORBIDDEN
            if suspended
            else OPERATOR_ENTRY_PERMISSIONS
        ),
        acting_as=ACTING_AS_PLATFORM_OPERATOR,
    )


__all__ = [
    "ACTING_AS_PLATFORM_OPERATOR",
    "OPERATOR_ENTRY_PERMISSIONS",
    "PlatformOperator",
    "entry_decision",
    "resolve_operator",
]
