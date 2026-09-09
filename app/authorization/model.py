"""The authorization model.

This module holds the single most security-critical decision in the migration:
**how an application user becomes a Vision OS principal**.

### Deny by default, and never by accident

Vision OS's ``Grant`` treats an empty ``cameras`` tuple as *every camera in the
tenant*, documented in its own adapter:

    "``cameras`` empty means *every camera in the tenant* — the common case for a
     site-wide operator. It does **not** mean 'no cameras': an empty grant would
     be indistinguishable from a misconfigured one, so a principal with no
     access is expressed by having no grant at all."

That is a coherent rule inside the platform and a loaded gun at the application
boundary, because the natural application-side value for "this user has no camera
access yet" is an empty list. Phase 0 flagged it as the single most dangerous
line in the migration.

So this module never hands Vision OS an empty tuple by inference. Access is a
three-state decision made explicitly here:

    NONE          → no grant is issued at all; every call is refused before it
                    reaches the platform
    ALL_IN_TENANT → an explicit, deliberate widening; the caller must say so
    LISTED        → exactly the named cameras

``CameraScope`` cannot be constructed in a state where those are confusable, and
``AccessDecision.to_grant()`` raises rather than guessing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.errors import ScopeError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vision_os.adapters.exposure.authorization import Grant
    from vision_os.core.model.api import Principal, Scope


class Role(enum.Enum):
    """Product roles. A closed set — a role that is not here does not exist.

    Deliberately not the reference backend's ``admin | developer | viewer``: that
    is a coding-assistant model, and mapping it onto a compliance product would
    put "viewer" in charge of deciding who may look at CCTV footage of a named
    employee.
    """

    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    RESTAURANT_MANAGER = "restaurant_manager"
    KITCHEN_SUPERVISOR = "kitchen_supervisor"
    HYGIENE_OFFICER = "hygiene_officer"
    AUDITOR = "auditor"
    DEVELOPER = "developer"

    @property
    def is_platform_role(self) -> bool:
        """Whether this role may reach engineering surfaces (DevTools)."""
        return self in (Role.SUPER_ADMIN, Role.DEVELOPER)


class Permission(enum.Enum):
    """Application-level permissions.

    Distinct from Vision OS ``Action``. These govern *product* surfaces; Actions
    govern the platform. Both are enforced, and neither substitutes for the
    other — hiding a route is not closing it.

    Phase 1 declares only the permissions the foundation needs. Product
    permissions (incidents, reports, notifications) arrive with the features
    they guard, so that no permission exists without something to protect.
    """

    # identity and administration
    #: Organisation *settings and lifecycle* — the organisation's own name,
    #: timezone policy and status. Deliberately **not** a blanket write
    #: permission over the physical estate: sites and zones have their own keys
    #: below. It held that job historically, which made "may edit one zone's
    #: name" and "may reconfigure the whole organisation" the same grant.
    MANAGE_ORGANIZATION = "manage_organization"
    MANAGE_USERS = "manage_users"
    #: The user roster, and nothing else. It historically also gated reading
    #: sites and zones, so holding it meant "may read every site" as a side
    #: effect of "may see who works here" — two unrelated questions.
    VIEW_USERS = "view_users"

    # the physical estate: sites and the zones inside them
    #
    # Read and manage are separate keys per domain because that is the
    # distinction the product actually needs to express: two managers on the
    # same role where one may edit the estate and the other may only look at
    # it. Neither is implied by the other, and neither is implied by
    # MANAGE_ORGANIZATION.

    VIEW_SITES = "view_sites"
    MANAGE_SITES = "manage_sites"
    VIEW_ZONES = "view_zones"
    MANAGE_ZONES = "manage_zones"

    # observation surfaces
    VIEW_LIVE = "view_live"
    VIEW_OBSERVATIONS = "view_observations"
    #: Separate from VIEW_OBSERVATIONS on purpose. Never implied by it.
    VIEW_EVIDENCE = "view_evidence"
    VIEW_CAMERA_HEALTH = "view_camera_health"

    # cameras
    MANAGE_CAMERAS = "manage_cameras"
    VIEW_CAMERAS = "view_cameras"
    #: Retiring a camera is not a heavier edit. It closes an observation
    #: partition — the historical record a finding may later need to be
    #: defended with — and it cannot be undone by renaming the camera back.
    #: This is the same separation the enum already draws between
    #: VIEW_EVIDENCE and DELETE_EVIDENCE, for the same reason.
    RETIRE_CAMERAS = "retire_cameras"

    # incidents — the work queue
    VIEW_INCIDENTS = "view_incidents"
    ACKNOWLEDGE_INCIDENTS = "acknowledge_incidents"
    RESOLVE_INCIDENTS = "resolve_incidents"

    #: Erasing evidence is NOT implied by being allowed to view it. One is
    #: looking; the other is destroying a record that may be needed to defend a
    #: finding — or to answer an erasure request.
    DELETE_EVIDENCE = "delete_evidence"

    #: The audit trail records who looked at imagery of identifiable people.
    #: Reading it is its own privilege, and it is not implied by administration.
    VIEW_AUDIT = "view_audit"

    # reporting
    #
    # Neither of these grants access to anything on its own. A report type
    # requires VIEW_REPORTS **and** the permission for every source it reads,
    # so an incident report still needs VIEW_INCIDENTS and an audit report
    # still needs VIEW_AUDIT. Without that rule reporting would be the
    # bypass — it is the one surface whose whole purpose is to assemble data
    # from everywhere at once.

    #: Run a report and read it on screen.
    VIEW_REPORTS = "view_reports"
    #: Take a copy away. Separate from VIEW_REPORTS deliberately: an exported
    #: file leaves this system entirely — it can be forwarded, no retention
    #: sweep here reaches it, and it outlives every policy this application
    #: enforces. Reading a figure and removing it are different acts.
    EXPORT_REPORTS = "export_reports"

    #: Model evaluation artifacts: how well the perception stack scores against
    #: human-annotated data. Its own permission and not implied by VIEW_REPORTS,
    #: for two reasons. It is engineering data rather than an operational
    #: figure — attribute agreement on a 43-subject split answers "should we
    #: ship this model", not "is the kitchen clean". And it is candid about the
    #: product's weaknesses in a way an operational report is not.
    VIEW_MODEL_EVALUATION = "view_model_evaluation"

    # ── product modules with no data source yet ──────────────────────────────
    #
    # Declared now because each guards a table that exists now. The rule this
    # module already follows — "no permission exists without something to
    # protect" — is satisfied by the schema, not by the route: `patron_tokens`
    # is a real table today, and the moment anything can write to it the gate
    # must already be there rather than being remembered at the same time.
    #
    # A MANAGE_* is declared only where there is genuinely configuration to
    # administer. Counting, demography and meal detection have none yet, so
    # they get a read permission and nothing more; their MANAGE arrives with
    # the configuration model it would protect.

    VIEW_PEOPLE_COUNT = "view_people_count"
    #: Its own permission, never implied by VIEW_PEOPLE_COUNT. Counting how many
    #: people passed is a footfall figure; inferring their age or gender is a
    #: different purpose in PDPA terms and needs its own lawful basis, so it
    #: also gets its own key.
    VIEW_DEMOGRAPHY = "view_demography"

    VIEW_TABLE_OCCUPANCY = "view_table_occupancy"
    #: The floor plan: which tables exist, where, and which camera watches them.
    MANAGE_TABLE_OCCUPANCY = "manage_table_occupancy"

    VIEW_CUTTING_BOARD = "view_cutting_board"
    #: The colour-to-ingredient policy. Changing it changes what counts as a
    #: violation, which is the same kind of act as changing a rule set — and it
    #: is why this is separate from viewing the readings.
    MANAGE_CUTTING_BOARD = "manage_cutting_board"

    VIEW_MEAL_DETECTION = "view_meal_detection"

    #: Reading which pseudonymous patron tokens exist. Not the imagery, not a
    #: name — the platform holds neither — but still the most sensitive read in
    #: the product, and never implied by any other permission.
    VIEW_PATRON_ID = "view_patron_id"
    #: Configuring biometric re-identification. Held by SUPER_ADMIN alone, and
    #: even then the write path refuses until the legal gate is satisfied: this
    #: permission decides who may *ask*, not whether the answer is yes.
    MANAGE_PATRON_ID = "manage_patron_id"

    VIEW_POS_INTEGRATION = "view_pos_integration"
    #: Points a connector at a URL and a credential reference. A POS credential
    #: reaches sales and often payment data, so this is the most consequential
    #: configuration permission in the application.
    MANAGE_POS_INTEGRATION = "manage_pos_integration"

    # engineering
    ACCESS_DEVTOOLS = "access_devtools"
    #: Registering a demand spends money and causes computation. Not a read.
    REGISTER_DEMAND = "register_demand"


#: Role → permissions. Explicit, exhaustive, and flat: no inheritance, because
#: an inherited permission is one nobody remembers granting.
#:
#: Two assignments are load-bearing and easy to get wrong:
#:
#:   KITCHEN_SUPERVISOR has no VIEW_EVIDENCE. It is the role most likely to be a
#:   shared screen on a kitchen wall, and imagery of a named employee should not
#:   be visible to whoever walks past it.
#:
#:   AUDITOR has no VIEW_LIVE. An auditor reviews the record; live monitoring is
#:   an operational act with a different purpose and a different lawful basis.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    #: Everything **except** biometric re-identification.
    #:
    #: `frozenset(Permission)` would hand `MANAGE_PATRON_ID` to this role by
    #: construction, and it would be inert only because
    #: `app/domain/patron.require_writable` refuses unconditionally. A
    #: permission that is harmless solely because of an unrelated guard is a
    #: trap for whoever relaxes that guard later without knowing it was doing
    #: silent work — so the exclusion is stated here, where somebody granting
    #: it has to write the line themselves.
    #:
    #: Until a DPIA and a named DPO sign-off exist, the correct holder of this
    #: permission is **nobody**, and that is what this says.
    Role.SUPER_ADMIN: frozenset(Permission) - {Permission.MANAGE_PATRON_ID},
    Role.ORG_ADMIN: frozenset(
        {
            Permission.MANAGE_ORGANIZATION,
            Permission.MANAGE_USERS,
            Permission.VIEW_USERS,
            Permission.VIEW_LIVE,
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_CAMERA_HEALTH,
            Permission.REGISTER_DEMAND,
            Permission.MANAGE_CAMERAS,
            Permission.VIEW_CAMERAS,
            # Retirement, unlike the rest of camera administration, is held
            # here and by SUPER_ADMIN alone. A restaurant manager who may add
            # and rename cameras should not thereby be able to close an
            # observation partition.
            Permission.RETIRE_CAMERAS,
            # The estate. These are new keys, and this role held their reach
            # before they existed — it gated sites and zones through
            # MANAGE_ORGANIZATION and VIEW_USERS. Listing them here preserves
            # exactly what this role could already do.
            Permission.VIEW_SITES,
            Permission.MANAGE_SITES,
            Permission.VIEW_ZONES,
            Permission.MANAGE_ZONES,
            Permission.VIEW_INCIDENTS,
            Permission.ACKNOWLEDGE_INCIDENTS,
            Permission.RESOLVE_INCIDENTS,
            Permission.DELETE_EVIDENCE,
            Permission.VIEW_AUDIT,
            # The new modules. An org admin reads all of them and configures the
            # ones that are operational configuration.
            Permission.VIEW_PEOPLE_COUNT,
            Permission.VIEW_DEMOGRAPHY,
            Permission.VIEW_TABLE_OCCUPANCY,
            Permission.MANAGE_TABLE_OCCUPANCY,
            Permission.VIEW_CUTTING_BOARD,
            Permission.MANAGE_CUTTING_BOARD,
            Permission.VIEW_MEAL_DETECTION,
            Permission.VIEW_POS_INTEGRATION,
            Permission.MANAGE_POS_INTEGRATION,
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
            # No VIEW_MODEL_EVALUATION.
            #
            # It was held here on the reasoning that an organisation
            # administrator answers for what the system claims, so they may see
            # how well it scores. That reasoning is sound about accountability
            # and wrong about this permission. What the dashboard actually shows
            # is per-attribute agreement on a 43-subject annotated split,
            # per-state confusion matrices, prompt-variant comparisons and a
            # dataset's own written admission that it cannot measure detection
            # recall. Those answer "should we ship this model" — an engineering
            # question — and an administrator scanning for an operational figure
            # is the wrong reader for them.
            #
            # The accountability the reasoning was reaching for is served by
            # VIEW_REPORTS, which this role holds: reports carry coverage,
            # completeness and the ruleset version every figure was computed
            # under, which is what "answering for what the system claims"
            # actually needs.
            #
            # Holders are now SUPER_ADMIN and DEVELOPER.
            #
            # Reads that patron identification exists and is blocked. Does NOT
            # hold MANAGE_PATRON_ID: an organisation administrator is the wrong
            # altitude for a decision that needs a DPIA behind it, and the
            # separation means turning it on is visibly not routine.
            Permission.VIEW_PATRON_ID,
        }
    ),
    Role.RESTAURANT_MANAGER: frozenset(
        {
            Permission.VIEW_USERS,
            Permission.VIEW_LIVE,
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_CAMERA_HEALTH,
            Permission.VIEW_CAMERAS,
            # Read the estate, do not change it. This is exactly what the role
            # could do before these keys existed — it held VIEW_USERS, which
            # gated site and zone reads — and deliberately no more. A manager
            # who should be able to edit sites gets MANAGE_SITES as a
            # per-user GRANT override; that difference between two managers on
            # one role is the whole point of the override engine.
            Permission.VIEW_SITES,
            Permission.VIEW_ZONES,
            Permission.VIEW_INCIDENTS,
            Permission.ACKNOWLEDGE_INCIDENTS,
            Permission.RESOLVE_INCIDENTS,
            # Operational reads for the site they run.
            Permission.VIEW_PEOPLE_COUNT,
            Permission.VIEW_TABLE_OCCUPANCY,
            Permission.VIEW_CUTTING_BOARD,
            Permission.VIEW_MEAL_DETECTION,
            # No VIEW_DEMOGRAPHY. Age and gender inference is a marketing
            # purpose with its own lawful basis, not something a site manager
            # inherits by running the site.
            # No VIEW_PATRON_ID, and no POS configuration.
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
        }
    ),
    Role.KITCHEN_SUPERVISOR: frozenset(
        {
            Permission.VIEW_LIVE,
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_CAMERA_HEALTH,
            Permission.VIEW_CAMERAS,
            Permission.VIEW_INCIDENTS,
            # May acknowledge — "I have seen this" — but not resolve. Closing a
            # violation is a judgement a supervisor owns.
            Permission.ACKNOWLEDGE_INCIDENTS,
            # Board compliance is this role's actual job, and it is the one new
            # module that belongs on a kitchen screen. Nothing else here is:
            # the role has no VIEW_EVIDENCE for the same reason — the screen is
            # shared, and whoever walks past it sees what is on it.
            Permission.VIEW_CUTTING_BOARD,
            # May read a report on screen; may not take a copy away. The
            # kitchen screen is shared, and an exported file is not.
            Permission.VIEW_REPORTS,
        }
    ),
    Role.HYGIENE_OFFICER: frozenset(
        {
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_CAMERA_HEALTH,
            Permission.VIEW_CAMERAS,
            Permission.VIEW_INCIDENTS,
            Permission.RESOLVE_INCIDENTS,
            # Board colour coding is food safety, which is the whole remit of
            # this role — including authority over the policy itself.
            Permission.VIEW_CUTTING_BOARD,
            Permission.MANAGE_CUTTING_BOARD,
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
        }
    ),
    Role.AUDITOR: frozenset(
        {
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_INCIDENTS,
            # The whole point of the role: read the record, including who looked
            # at what. Reads nothing live and changes nothing.
            Permission.VIEW_AUDIT,
            # An auditor reads the food-safety record, which now includes board
            # usage. Deliberately not the commercial modules: footfall,
            # demography, dish detection and POS are business analytics, and an
            # auditor's lawful basis for reading the compliance record does not
            # extend to a company's sales.
            Permission.VIEW_CUTTING_BOARD,
            # Producing the evidence pack is the job. Exporting is part of it:
            # an audit finding that cannot leave the building is not a finding.
            Permission.VIEW_REPORTS,
            Permission.EXPORT_REPORTS,
        }
    ),
    Role.DEVELOPER: frozenset(
        {
            Permission.VIEW_LIVE,
            Permission.VIEW_OBSERVATIONS,
            Permission.VIEW_EVIDENCE,
            Permission.VIEW_CAMERA_HEALTH,
            Permission.ACCESS_DEVTOOLS,
            Permission.REGISTER_DEMAND,
            Permission.VIEW_CAMERAS,
            Permission.VIEW_INCIDENTS,
            # The role the evaluation dashboard is actually for.
            Permission.VIEW_MODEL_EVALUATION,
        }
    ),
}


def permissions_for(roles: frozenset[Role]) -> frozenset[Permission]:
    """Union of the permissions of every held role. Empty for no roles."""
    granted: set[Permission] = set()
    for role in roles:
        granted |= ROLE_PERMISSIONS.get(role, frozenset())
    return frozenset(granted)


class OverrideState(enum.Enum):
    """A per-user, per-permission exception to what their roles would give them.

    Three logical states, two of which are ever stored: no row means INHERIT —
    the role's own answer stands, and that is deliberately not a row at all, so
    that "this user has no exceptions" is the empty case rather than a table
    full of no-op rows. A stored row is always ``GRANT`` (add a permission the
    role doesn't carry) or ``REVOKE`` (remove one it does).

    This is the mechanism `Role` being a closed, code-defined enum requires: a
    "read-only manager" cannot be expressed by inventing a narrower role without
    causing role explosion, so the difference is carried per user instead. Pure
    union (grants only) was considered and rejected — it cannot express REVOKE,
    and REVOKE is exactly the case a closed role set cannot solve any other way.
    """

    GRANT = "grant"
    REVOKE = "revoke"


class OrganizationStatus(enum.Enum):
    """The tenant's own lifecycle state, independent of any one user's.

    Three states, never a bare boolean, for the same reason `Role` is a closed
    set and `ScopeBreadth` is three-valued rather than two: "suspended" and
    "archived" are different acts with different remedies, and collapsing them
    into a single `is_active` flag would make a billing hold and a permanent
    shutdown look identical to every caller downstream.
    """

    #: Unchanged behavior — every existing organization is here today.
    ACTIVE = "active"
    #: Login and reads continue; `MANAGE_*` writes are refused. Reversible.
    SUSPENDED = "suspended"
    #: Login and all API access refused. Data is retained, nothing is deleted.
    ARCHIVED = "archived"


def effective_permissions(
    roles: frozenset[Role],
    *,
    granted: frozenset[Permission] = frozenset(),
    revoked: frozenset[Permission] = frozenset(),
) -> frozenset[Permission]:
    """``(role permissions ∪ explicit GRANTs) − explicit REVOKEs``.

    REVOKE is applied last and always wins over both the role and any GRANT, so
    a revoked permission cannot be reinstated by also holding a role that
    carries it — the override is the more specific statement about this one
    user and takes precedence for exactly that reason.
    """
    return (permissions_for(roles) | granted) - revoked


class ScopeBreadth(enum.Enum):
    """How wide a camera or site grant reaches. Three states, never two.

    ``NONE`` and ``ALL_IN_TENANT`` are the two that an empty collection would
    otherwise conflate, which is exactly the failure this enum exists to make
    impossible.
    """

    NONE = "none"
    LISTED = "listed"
    ALL_IN_TENANT = "all_in_tenant"


@dataclass(frozen=True, slots=True)
class CameraScope:
    """Which cameras a principal reaches, stated rather than inferred."""

    breadth: ScopeBreadth
    camera_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.breadth is ScopeBreadth.LISTED and not self.camera_ids:
            raise ValueError(
                "ScopeBreadth.LISTED requires at least one camera; an empty list "
                "is ambiguous, so say ScopeBreadth.NONE when the answer is none"
            )
        if self.breadth is not ScopeBreadth.LISTED and self.camera_ids:
            raise ValueError(
                f"camera_ids must be empty for breadth {self.breadth.value}; "
                "listing cameras alongside a wildcard hides which one is in force"
            )

    @property
    def grants_anything(self) -> bool:
        return self.breadth is not ScopeBreadth.NONE

    @classmethod
    def none(cls) -> CameraScope:
        return cls(breadth=ScopeBreadth.NONE)

    @classmethod
    def listed(cls, camera_ids: tuple[str, ...]) -> CameraScope:
        return cls(breadth=ScopeBreadth.LISTED, camera_ids=tuple(camera_ids))

    @classmethod
    def all_in_tenant(cls) -> CameraScope:
        """Every camera in the tenant. Deliberate, never a default."""
        return cls(breadth=ScopeBreadth.ALL_IN_TENANT)


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """Everything the application knows about one principal's reach.

    Built once per request from the authenticated session, and never from
    request input — the leak that ``Scope``'s design exists to prevent is
    reintroduced the moment a client can name its own tenant.
    """

    subject: str
    tenant_id: str
    roles: frozenset[Role]
    cameras: CameraScope
    site_ids: tuple[str, ...] = ()
    display_name: str = ""
    #: Non-empty only for a session that reached this tenant by an audited
    #: platform-operator entry rather than by a membership
    #: (`app.authorization.platform.ACTING_AS_PLATFORM_OPERATOR`). It grants
    #: nothing and restricts nothing by itself — `permissions` already carries
    #: the whole of the reach — but the application has to be able to *say* that
    #: this is not an ordinary organization session, in the audit trail and on
    #: the operator's screen. Defaults to empty, so every existing construction
    #: keeps meaning exactly what it meant.
    acting_as: str = ""
    #: ``None`` means "not stated; derive from roles" — the historical, still
    #: most common path. An explicit ``frozenset()`` is different from that and
    #: must survive as-is: a user whose every role permission has been REVOKEd
    #: has *zero* effective permissions, and re-deriving from roles here would
    #: silently undo the revocation this dataclass exists to carry.
    permissions: frozenset[Permission] | None = None

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("an access decision must name a subject")
        if not self.tenant_id:
            raise ValueError(
                "an access decision must name a tenant; tenancy is part of "
                "identity rather than a filter applied afterwards"
            )
        if self.permissions is None:
            object.__setattr__(self, "permissions", permissions_for(self.roles))

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    def require(self, permission: Permission) -> None:
        if not self.has(permission):
            raise ScopeError(
                f"this account does not hold '{permission.value}'",
                details={"required": permission.value},
            )

    # ── the translation into Vision OS ───────────────────────────────────────

    def to_principal(self) -> Principal:
        """The platform-side identity. Constructed at the edge and never below it."""
        from vision_os.core.model.api import Principal

        return Principal(
            subject=self.subject,
            tenant_id=self._tenant(),
            scopes=tuple(sorted(p.value for p in self.permissions)),
            display_name=self.display_name,
        )

    def to_scope(self) -> Scope:
        """The platform-side scope.

        Raises rather than returning a tenant-wide scope when the decision grants
        no cameras. A caller that reaches here with ``ScopeBreadth.NONE`` has a
        bug, and the safe failure is loud.
        """
        from vision_os.core.model.api import Scope
        from vision_os.core.model.ids import CameraId, SiteId

        if not self.cameras.grants_anything:
            raise ScopeError(
                "this account is granted no cameras",
                details={"subject": self.subject},
            )

        return Scope(
            tenant_id=self._tenant(),
            site_ids=tuple(SiteId(s) for s in self.site_ids),
            camera_ids=self._camera_ids(),
        )

    def to_grant(self) -> Grant:
        """The platform-side grant.

        The empty-tuple hazard is handled here and nowhere else:

        * ``NONE`` raises — there is no such thing as a grant for nobody, and
          issuing one with an empty camera tuple would silently mean *all*.
        * ``ALL_IN_TENANT`` passes ``()`` **deliberately**, which is the
          platform's documented wildcard.
        * ``LISTED`` passes exactly the named cameras.
        """
        from vision_os.adapters.exposure.authorization import Grant
        from vision_os.core.model.ids import CameraId

        if not self.cameras.grants_anything:
            raise ScopeError(
                "refusing to build a Vision OS grant for a principal with no "
                "camera access: an empty camera tuple means ALL cameras to the "
                "platform, so the safe representation of 'none' is no grant",
                details={"subject": self.subject},
            )

        cameras: tuple[CameraId, ...] = ()
        if self.cameras.breadth is ScopeBreadth.LISTED:
            cameras = self._camera_ids()

        return Grant(
            subject=self.subject,
            tenant_id=self._tenant(),
            actions=self._actions(),
            cameras=cameras,
        )

    def _camera_ids(self):
        """The stored camera keys, as the ids the platform actually knows them by.

        `AccessGrant.camera_ids` holds bare `camera_key` values, which is right
        for a tenant-scoped table. The platform is not tenant-scoped in its
        partitioning — an observation is filed under the `CameraId` it was
        published with, and the pipeline publishes runtime ids
        (`app.domain.runtime_identity`). Translating here is what keeps the two
        halves agreeing: a grant that named bare keys would match nothing
        written since, and a grant naming another tenant's key would be a
        cross-tenant read waiting for a collision.
        """
        from vision_os.core.model.ids import CameraId

        from app.domain.runtime_identity import runtime_camera_id

        return tuple(
            CameraId(runtime_camera_id(self.tenant_id, key))
            for key in self.cameras.camera_ids
        )

    def _tenant(self):
        from vision_os.core.model.ids import TenantId

        return TenantId(self.tenant_id)

    def _actions(self) -> frozenset:
        """Product permissions → platform actions.

        The two evidence-adjacent rules are preserved exactly as the platform
        states them: ``READ_EVIDENCE`` is not implied by ``READ_OBSERVATIONS``,
        and ``REGISTER_DEMAND`` is not a read.
        """
        from vision_os.core.model.api import Action

        actions: set[Action] = set()

        if self.has(Permission.VIEW_OBSERVATIONS):
            actions |= {
                Action.READ_STATE,
                Action.READ_OBSERVATIONS,
                Action.READ_CAPABILITY,
                Action.READ_COVERAGE,
            }
        if self.has(Permission.VIEW_LIVE):
            actions.add(Action.SUBSCRIBE)
        if self.has(Permission.VIEW_CAMERA_HEALTH):
            actions.add(Action.READ_COVERAGE)
        if self.has(Permission.VIEW_EVIDENCE):
            actions.add(Action.READ_EVIDENCE)
        if self.has(Permission.REGISTER_DEMAND):
            actions.add(Action.REGISTER_DEMAND)

        return frozenset(actions)


__all__ = [
    "AccessDecision",
    "CameraScope",
    "OrganizationStatus",
    "OverrideState",
    "Permission",
    "ROLE_PERMISSIONS",
    "Role",
    "ScopeBreadth",
    "effective_permissions",
    "permissions_for",
]
