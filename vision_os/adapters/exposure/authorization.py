"""P31 adapters — `StaticAuthorizer` and `DenyAll`.

12_SECURITY §5.2's permission model is `(action, resource_scope, conditions)`.
`StaticAuthorizer` implements the first two from configuration; conditions —
time windows, purpose declarations, attribute restrictions — are where a real
deployment's policy engine earns its place, and the port is the seam for it.

**Both adapters narrow rather than filter.** §4.2 forbids post-filtering, so
`authorize` returns the scope the principal may query and the API queries *that*.
A grant for two cameras out of fifty returns a scope naming two cameras, and no
query the API constructs can reach the other forty-eight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ...core.model.api import Action, Principal, Scope
from ...core.model.ids import CameraId, TenantId
from ...core.ports.exposure import AuthorizationDecision


@dataclass(frozen=True, slots=True)
class Grant:
    """What one principal may do.

    ``cameras`` empty means *every camera in the tenant* — the common case for a
    site-wide operator. It does **not** mean "no cameras": an empty grant would be
    indistinguishable from a misconfigured one, so a principal with no access is
    expressed by having no grant at all.
    """

    subject: str
    tenant_id: TenantId
    actions: frozenset[Action] = field(default_factory=frozenset)
    cameras: tuple[CameraId, ...] = ()
    regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a grant must name a subject")
        if not self.tenant_id:
            raise ValueError(
                "a grant must name a tenant; a tenantless grant would be a "
                "cross-tenant grant by omission"
            )

    def permits(self, action: Action) -> bool:
        return action in self.actions

    def covers(self, camera_id: CameraId) -> bool:
        return not self.cameras or camera_id in self.cameras


class StaticAuthorizer:
    """``authz.static`` — grants declared up front.

    For an embedded deployment, a test, or a site whose access model genuinely is
    a short list. Honest about its limits: it evaluates no conditions and consults
    no external policy, and its id says ``static`` so an operator reading metrics
    knows which model is in force.
    """

    __slots__ = ("_grants",)

    def __init__(self, grants: Sequence[Grant] = ()) -> None:
        self._grants: dict[str, Grant] = {g.subject: g for g in grants}

    @property
    def authorizer_id(self) -> str:
        return "authz.static"

    def authorize(
        self, principal: Principal, action: Action, scope: Scope
    ) -> AuthorizationDecision:
        """Decide, and narrow (Z1).

        The tenant check comes **first and unconditionally** (Z2): no grant, no
        role and no configuration can permit a principal of one tenant to read
        another's data. 12_SECURITY §4.1 makes the isolation boundary absolute,
        and an authorization model able to cross it would make it advisory.
        """
        if principal.tenant_id != scope.tenant_id:
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason=(
                    f"principal belongs to tenant '{principal.tenant_id}' and "
                    f"requested '{scope.tenant_id}'; the isolation boundary is "
                    f"not configurable (12_SECURITY §4.1)"
                ),
            )

        grant = self._grants.get(principal.subject)
        if grant is None:
            # Z5 — fail closed. An unknown principal is denied, never defaulted
            # to a permissive role.
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason=f"no grant for principal '{principal.subject}'",
            )

        if grant.tenant_id != principal.tenant_id:
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason="the grant belongs to a different tenant",
            )

        if not grant.permits(action):
            # Z3 in practice: read_observations does not imply read_evidence,
            # because each action is a separate member of the grant's set.
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason=(
                    f"principal '{principal.subject}' has no '{action.value}' "
                    f"permission"
                ),
            )

        return AuthorizationDecision(granted=True, scope=self._narrow(grant, scope))

    def _narrow(self, grant: Grant, scope: Scope) -> Scope:
        """Intersect the request with the grant.

        The result is what the API will query. Nothing outside it is ever fetched,
        so there is no moment at which out-of-scope data exists in memory (§4.2).
        """
        if grant.cameras and scope.camera_ids:
            cameras = tuple(c for c in scope.camera_ids if c in grant.cameras)
        elif grant.cameras:
            cameras = grant.cameras
        else:
            cameras = scope.camera_ids

        return Scope(
            tenant_id=scope.tenant_id,
            site_ids=scope.site_ids,
            camera_ids=cameras,
            region_ids=scope.region_ids,
        )

    def visible_cameras(
        self, principal: Principal, scope: Scope
    ) -> Sequence[CameraId]:
        grant = self._grants.get(principal.subject)
        if grant is None or grant.tenant_id != principal.tenant_id:
            return ()
        if not grant.cameras:
            return scope.camera_ids
        if not scope.camera_ids:
            return grant.cameras
        return tuple(c for c in scope.camera_ids if grant.covers(c))

    def add(self, grant: Grant) -> None:
        self._grants[grant.subject] = grant

    def __len__(self) -> int:
        return len(self._grants)


class DenyAll:
    """``authz.deny_all`` — the honest default.

    Bound when a deployment has not configured authorization. Refusing everything
    is safe; the alternative — permitting everything until somebody remembers to
    configure a policy — has exactly one failure mode and it is a breach.
    """

    __slots__ = ()

    @property
    def authorizer_id(self) -> str:
        return "authz.deny_all"

    def authorize(
        self, principal: Principal, action: Action, scope: Scope
    ) -> AuthorizationDecision:
        return AuthorizationDecision(
            granted=False,
            scope=Scope(tenant_id=principal.tenant_id),
            reason=(
                "no authorization model is configured; this deployment denies by "
                "default rather than permitting until somebody configures a policy"
            ),
        )

    def visible_cameras(
        self, principal: Principal, scope: Scope
    ) -> Sequence[CameraId]:
        return ()


class AllowReadsWithinTenant:
    """``authz.tenant_reads`` — every read action, within the principal's tenant.

    For a single-tenant embedded deployment where the process boundary *is* the
    authorization boundary. Deliberately does **not** grant `read_evidence` or
    `register_demand`: 12_SECURITY §5.3 separates imagery access and demand
    registration from ordinary reads, and a convenience adapter that quietly
    included them would erase the separation the architecture is most explicit
    about.
    """

    __slots__ = ()

    _READS = frozenset(
        {
            Action.READ_STATE,
            Action.READ_OBSERVATIONS,
            Action.READ_COVERAGE,
            Action.READ_CAPABILITY,
            Action.SUBSCRIBE,
        }
    )

    @property
    def authorizer_id(self) -> str:
        return "authz.tenant_reads"

    def authorize(
        self, principal: Principal, action: Action, scope: Scope
    ) -> AuthorizationDecision:
        if principal.tenant_id != scope.tenant_id:
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason="cross-tenant access is never permitted",
            )
        if action not in self._READS:
            return AuthorizationDecision(
                granted=False,
                scope=Scope(tenant_id=principal.tenant_id),
                reason=(
                    f"'{action.value}' requires an explicit grant; this adapter "
                    f"covers ordinary reads only (12_SECURITY §5.3)"
                ),
            )
        return AuthorizationDecision(granted=True, scope=scope)

    def visible_cameras(
        self, principal: Principal, scope: Scope
    ) -> Sequence[CameraId]:
        return scope.camera_ids


#: Selectable by name from configuration. Closed, like every factory table in the
#: platform: an unknown name is refused rather than defaulted, because defaulting
#: an authorization model is how a deployment ends up more permissive than
#: anybody intended.
AUTHORIZER_FACTORIES: Mapping[str, object] = {
    "authz.static": StaticAuthorizer,
    "authz.deny_all": DenyAll,
    "authz.tenant_reads": AllowReadsWithinTenant,
}


def read_only_grant(
    subject: str, tenant_id: TenantId, *, cameras: Sequence[CameraId] = ()
) -> Grant:
    """A grant for a consumer that reads facts but never imagery.

    The shape §5.3 calls the common case: *"Most consumers need the first and
    must never have the second."*
    """
    return Grant(
        subject=subject,
        tenant_id=tenant_id,
        actions=frozenset(
            {
                Action.READ_STATE,
                Action.READ_OBSERVATIONS,
                Action.READ_COVERAGE,
                Action.READ_CAPABILITY,
                Action.SUBSCRIBE,
            }
        ),
        cameras=tuple(cameras),
    )


def full_grant(
    subject: str, tenant_id: TenantId, *, cameras: Sequence[CameraId] = ()
) -> Grant:
    """Every action, including evidence and demand registration.

    For an operator or an integration test. Named ``full`` rather than ``admin``
    because 12_SECURITY §5.3 keeps administration — config, models, plugins —
    *"wholly separate"* from consumer actions, and no consumer grant, however
    broad, reaches them.
    """
    return Grant(
        subject=subject,
        tenant_id=tenant_id,
        actions=frozenset(Action),
        cameras=tuple(cameras),
    )
