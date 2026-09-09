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

from app.authorization.model import AccessDecision, CameraScope, Permission
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


#: The claim that marks a token minted by `POST /platform/organizations/{id}/enter`.
#:
#: Its presence is what tells `decision_for_claims` that the tenant on this
#: token was reached by an audited platform entry rather than by a membership,
#: and that the reach to build is `OPERATOR_ENTRY_PERMISSIONS` rather than
#: whatever roles the account happens to hold in that organization. A token
#: without it is an ordinary tenant token and is resolved the ordinary way.
ACTING_AS_PLATFORM_OPERATOR = "platform_operator"


#: What a platform operator may read inside an organization they have entered.
#:
#: ### Why this is a list and not a rule
#:
#: The obvious implementation is `p.value.startswith("view_")`. It is wrong for
#: the reason `resolver.SUSPENDED_FORBIDDEN` records at length: a rule that
#: depends on how a permission was *spelled* is not a rule about what it does.
#: `EXPORT_REPORTS` does not begin with `view_` and is excluded here anyway;
#: `VIEW_EVIDENCE` does begin with it and is excluded too. Spelling would have
#: got both wrong, in opposite directions.
#:
#: ### What is deliberately absent
#:
#: * **`VIEW_EVIDENCE`, `VIEW_PATRON_ID`** — retained imagery of identifiable
#:   people, and the biometric surface. Support work does not require looking
#:   at a named employee's face, and an operator grant reaches every customer.
#: * **`VIEW_AUDIT`** — who looked at imagery of whom. A privacy record in its
#:   own right, and not an administrative by-product.
#: * **`EXPORT_REPORTS`** — a copy that leaves the system and outlives every
#:   retention policy this application enforces. Reading a report on screen is
#:   a read; taking one away is not.
#: * **`REGISTER_DEMAND`** — spends the customer's money and causes computation.
#: * **`ACCESS_DEVTOOLS`** — a separate engineering surface with its own gate.
#: * **every `MANAGE_*`, `RETIRE_CAMERAS`, `DELETE_EVIDENCE`, and the incident
#:   queue** — entry is read-only, and the write path must refuse rather than
#:   merely be hidden.
#:
#: ### What is present, and the tension in it
#:
#: `VIEW_LIVE` is here. It is the least comfortable entry in the set, because
#: live imagery of people at work is not obviously less sensitive than the
#: retained kind that `VIEW_EVIDENCE` gates. It is included because an operator
#: who cannot see whether a customer's cameras are actually producing pictures
#: cannot answer the support question that entry exists for. Recording the
#: tension here so that removing it later is a decision someone makes on
#: purpose rather than a gap somebody notices.
OPERATOR_ENTRY_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.VIEW_USERS,
        Permission.VIEW_SITES,
        Permission.VIEW_ZONES,
        Permission.VIEW_LIVE,
        Permission.VIEW_OBSERVATIONS,
        Permission.VIEW_CAMERA_HEALTH,
        Permission.VIEW_CAMERAS,
        Permission.VIEW_INCIDENTS,
        Permission.VIEW_REPORTS,
        Permission.VIEW_MODEL_EVALUATION,
        Permission.VIEW_PEOPLE_COUNT,
        Permission.VIEW_DEMOGRAPHY,
        Permission.VIEW_TABLE_OCCUPANCY,
        Permission.VIEW_CUTTING_BOARD,
        Permission.VIEW_MEAL_DETECTION,
        Permission.VIEW_POS_INTEGRATION,
    }
)


def entry_decision(operator: PlatformOperator, organization_id: str) -> AccessDecision:
    """The tenant-scoped reach of an operator who has entered an organization.

    An `AccessDecision`, because every route in the application already speaks
    that language and inventing a second authorization type for this would be
    the parallel tenant-context mechanism the architecture exists to avoid. It
    is built here rather than by the resolver because it is not resolved from
    anything stored about the user *in that organization* — there is nothing
    stored, which is the entire point.

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
        permissions=OPERATOR_ENTRY_PERMISSIONS,
        acting_as=ACTING_AS_PLATFORM_OPERATOR,
    )


__all__ = [
    "ACTING_AS_PLATFORM_OPERATOR",
    "OPERATOR_ENTRY_PERMISSIONS",
    "PlatformOperator",
    "entry_decision",
    "resolve_operator",
]
