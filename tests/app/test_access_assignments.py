"""Setting somebody's whole access in one organization at once — the domain half.

The platform console's access matrix states access as a set of permissions per
organization. These tests pin the one property everything else depends on: what
is written makes `decide()` answer exactly that set, no more and no less.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.authorization.assignments import (
    OrganizationAccess,
    apply_organization_access,
    load_for_access,
    parse_access_items,
    plan_access,
)
from app.authorization.model import ROLE_PERMISSIONS, Permission, Role, ScopeBreadth
from app.authorization.overrides import replace_permission_overrides
from app.authorization.resolver import decide
from app.domain.audit import AuditAction
from app.errors import ScopeError, ValidationError
from app.users.models import (
    AccessGrant,
    Organization,
    OrganizationMembership,
    PermissionOverride,
    RoleAssignment,
    User,
)
from tests.app.conftest import make_user

ACTOR = "user-actor@example.com"
TARGET = "user-target@example.com"


async def _people(app, *, target_roles: tuple[str, ...] = ("restaurant_manager",)) -> None:
    """Org A with an administrator and a member to act on; org B, which the
    member is not in. The member starts with one stray override and a
    listed camera grant, so a test can see both being replaced or kept."""
    async with app.state.database.session_scope() as session:
        org, actor = make_user(
            org_id="org-a",
            email="actor@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        _, target = make_user(
            org_id="org-a",
            email="target@example.com",
            roles=target_roles,
            camera_breadth="listed",
            camera_ids="cam-01",
        )
        session.add_all([org, actor, target, Organization(id="org-b", name="B", slug="b")])
        await session.flush()
        session.add(
            PermissionOverride(
                user_id=TARGET, organization_id="org-a", permission="view_audit", state="grant"
            )
        )


async def _override_rows(session, user_id: str) -> set[tuple[str, str]]:
    rows = await session.execute(
        select(PermissionOverride.permission, PermissionOverride.state).where(
            PermissionOverride.user_id == user_id
        )
    )
    return set(rows.all())


def test_access_set_is_an_audit_action():
    assert AuditAction.ACCESS_SET.value == "user.access_set"


async def test_replacing_overrides_leaves_exactly_the_stated_rows(app):
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        target = await session.get(User, TARGET)
        await replace_permission_overrides(
            session,
            actor=actor,
            target=target,
            organization_id="org-a",
            granted=frozenset({Permission.MANAGE_ZONES}),
            revoked=frozenset({Permission.VIEW_EVIDENCE}),
        )
        await session.flush()
        # The stray `view_audit` grant is gone; only what was stated remains.
        assert await _override_rows(session, TARGET) == {
            ("manage_zones", "grant"),
            ("view_evidence", "revoke"),
        }


async def test_replacing_overrides_can_rewrite_the_same_permission(app):
    """Review focus 2: delete-then-insert must not collide on the unique key."""
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        target = await session.get(User, TARGET)
        await replace_permission_overrides(
            session,
            actor=actor,
            target=target,
            organization_id="org-a",
            granted=frozenset(),
            revoked=frozenset({Permission.VIEW_AUDIT}),
        )
        await session.flush()
        assert await _override_rows(session, TARGET) == {("view_audit", "revoke")}


async def test_replacing_overrides_records_who_granted_them(app):
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        target = await session.get(User, TARGET)
        await replace_permission_overrides(
            session,
            actor=actor,
            target=target,
            organization_id="org-a",
            granted=frozenset({Permission.MANAGE_ZONES}),
            revoked=frozenset(),
        )
        await session.flush()
        row = (
            await session.execute(
                select(PermissionOverride).where(PermissionOverride.user_id == TARGET)
            )
        ).scalar_one()
        assert row.granted_by == ACTOR


async def test_replacing_overrides_refuses_self(app):
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        with pytest.raises(ScopeError):
            await replace_permission_overrides(
                session,
                actor=actor,
                target=actor,
                organization_id="org-a",
                granted=frozenset({Permission.VIEW_AUDIT}),
                revoked=frozenset(),
            )


async def test_replacing_overrides_refuses_an_organization_the_target_is_not_in(app):
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        target = await session.get(User, TARGET)
        with pytest.raises(ScopeError):
            await replace_permission_overrides(
                session,
                actor=actor,
                target=target,
                organization_id="org-b",
                granted=frozenset({Permission.VIEW_AUDIT}),
                revoked=frozenset(),
            )


async def test_a_permission_both_granted_and_revoked_is_refused(app):
    await _people(app)
    async with app.state.database.session_scope() as session:
        actor = await session.get(User, ACTOR)
        target = await session.get(User, TARGET)
        with pytest.raises(ValidationError):
            await replace_permission_overrides(
                session,
                actor=actor,
                target=target,
                organization_id="org-a",
                granted=frozenset({Permission.VIEW_AUDIT}),
                revoked=frozenset({Permission.VIEW_AUDIT}),
            )


# ── plan_access: pure ────────────────────────────────────────────────────────

MANAGER = ROLE_PERMISSIONS[Role.RESTAURANT_MANAGER]


@pytest.mark.parametrize(
    ("role", "ticks", "granted", "revoked"),
    [
        (
            Role.RESTAURANT_MANAGER,
            MANAGER | {Permission.MANAGE_ZONES},
            {Permission.MANAGE_ZONES},
            set(),
        ),
        (
            Role.RESTAURANT_MANAGER,
            MANAGER - {Permission.VIEW_EVIDENCE},
            set(),
            {Permission.VIEW_EVIDENCE},
        ),
        (None, frozenset({Permission.VIEW_LIVE}), {Permission.VIEW_LIVE}, set()),
        (
            Role.ORG_ADMIN,
            frozenset(Permission) - {Permission.VIEW_AUDIT},
            set(),
            {Permission.VIEW_AUDIT},
        ),
    ],
)
def test_plan_access_is_the_difference_from_the_role(role, ticks, granted, revoked):
    plan = plan_access(role, frozenset(ticks))
    assert plan.role is role
    assert plan.granted == granted
    assert plan.revoked == revoked


# ── parsing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "not a list",
        [{"organization_id": "", "permissions": []}],
        [{"organization_id": "org-a", "role": "chef", "permissions": []}],
        [{"organization_id": "org-a", "permissions": ["fly"]}],
        [{"organization_id": "org-a", "permissions": "view_live"}],
        [{"organization_id": "org-a", "permissions": [], "camera_breadth": "listed"}],
        [
            {"organization_id": "org-a", "permissions": []},
            {"organization_id": "org-a", "permissions": []},
        ],
    ],
)
def test_malformed_access_is_refused(raw):
    with pytest.raises(ValidationError):
        parse_access_items(raw)


def test_well_formed_access_parses():
    [item] = parse_access_items(
        [
            {
                "organization_id": "org-a",
                "role": "AUDITOR",
                "permissions": ["view_live", "view_live"],
                "camera_breadth": "all_in_tenant",
            }
        ]
    )
    assert item == OrganizationAccess(
        organization_id="org-a",
        role=Role.AUDITOR,
        permissions=frozenset({Permission.VIEW_LIVE}),
        camera_breadth=ScopeBreadth.ALL_IN_TENANT,
    )


# ── apply_organization_access ────────────────────────────────────────────────


async def _apply(app, access: OrganizationAccess):
    async with app.state.database.session_scope() as session:
        actor = await load_for_access(session, ACTOR)
        target = await load_for_access(session, TARGET)
        return await apply_organization_access(
            session, actor=actor, target=target, access=access, granted_by="op@example.com"
        )


async def _decision(app, organization_id: str):
    async with app.state.database.session_scope() as session:
        return decide(await load_for_access(session, TARGET), organization_id=organization_id)


@pytest.mark.parametrize(
    ("role", "ticks"),
    [
        (
            Role.RESTAURANT_MANAGER,
            (MANAGER | {Permission.MANAGE_ZONES}) - {Permission.VIEW_EVIDENCE},
        ),
        (None, frozenset({Permission.VIEW_LIVE})),
        (Role.ORG_ADMIN, frozenset(Permission) - {Permission.VIEW_AUDIT}),
        (Role.AUDITOR, frozenset()),
    ],
)
async def test_decide_answers_exactly_the_ticks(app, role, ticks):
    await _people(app)
    await _apply(app, OrganizationAccess("org-a", role, frozenset(ticks)))
    assert (await _decision(app, "org-a")).permissions == frozenset(ticks)


async def test_a_non_member_is_admitted_with_the_stated_access(app):
    await _people(app)
    applied = await _apply(
        app,
        OrganizationAccess(
            "org-b", Role.AUDITOR, ROLE_PERMISSIONS[Role.AUDITOR], ScopeBreadth.ALL_IN_TENANT
        ),
    )
    assert applied.admitted is True
    assert applied.before == frozenset()
    assert applied.after == ROLE_PERMISSIONS[Role.AUDITOR]

    decision = await _decision(app, "org-b")
    assert decision.permissions == ROLE_PERMISSIONS[Role.AUDITOR]
    assert decision.cameras.breadth is ScopeBreadth.ALL_IN_TENANT
    async with app.state.database.session_scope() as session:
        membership = (
            await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == TARGET,
                    OrganizationMembership.organization_id == "org-b",
                )
            )
        ).scalar_one()
        assert membership.granted_by == "op@example.com"


async def test_an_untouched_organization_is_not_joined(app):
    """Review focus 3: nothing asked for, nothing written, nobody admitted."""
    await _people(app)
    applied = await _apply(app, OrganizationAccess("org-b", None, frozenset(), ScopeBreadth.NONE))
    assert applied is None
    async with app.state.database.session_scope() as session:
        joined = (
            await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == TARGET,
                    OrganizationMembership.organization_id == "org-b",
                )
            )
        ).all()
        assert joined == []


async def test_two_roles_are_normalised_to_the_template_without_changing_access(app):
    """Review focus 1: several roles in, one role out, access exactly the ticks."""
    await _people(app, target_roles=("restaurant_manager", "auditor"))
    ticks = ROLE_PERMISSIONS[Role.AUDITOR] | {Permission.VIEW_LIVE}
    await _apply(app, OrganizationAccess("org-a", Role.AUDITOR, ticks))

    async with app.state.database.session_scope() as session:
        roles = (
            (
                await session.execute(
                    select(RoleAssignment.role).where(
                        RoleAssignment.user_id == TARGET,
                        RoleAssignment.organization_id == "org-a",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert roles == ["auditor"]
    assert (await _decision(app, "org-a")).permissions == ticks


async def test_all_cameras_is_written_and_omitted_breadth_keeps_the_grant(app):
    await _people(app)
    await _apply(app, OrganizationAccess("org-a", Role.RESTAURANT_MANAGER, MANAGER))
    # Omitted: the listed grant survives.
    async with app.state.database.session_scope() as session:
        grant = (
            await session.execute(select(AccessGrant).where(AccessGrant.user_id == TARGET))
        ).scalar_one()
        assert (grant.camera_breadth, grant.camera_ids) == ("listed", "cam-01")

    await _apply(
        app,
        OrganizationAccess("org-a", Role.RESTAURANT_MANAGER, MANAGER, ScopeBreadth.ALL_IN_TENANT),
    )
    assert (await _decision(app, "org-a")).cameras.breadth is ScopeBreadth.ALL_IN_TENANT


async def test_applied_access_reports_before_and_after(app):
    await _people(app)
    applied = await _apply(
        app, OrganizationAccess("org-a", Role.RESTAURANT_MANAGER, MANAGER - {Permission.VIEW_LIVE})
    )
    # Before: the manager role plus the stray `view_audit` grant from `_people`.
    assert applied.before == MANAGER | {Permission.VIEW_AUDIT}
    assert applied.after == MANAGER - {Permission.VIEW_LIVE}
    assert applied.admitted is False
