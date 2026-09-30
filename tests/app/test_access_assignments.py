"""Setting somebody's whole access in one organization at once — the domain half.

The platform console's access matrix states access as a set of permissions per
organization. These tests pin the one property everything else depends on: what
is written makes `decide()` answer exactly that set, no more and no less.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.authorization.model import Permission
from app.authorization.overrides import replace_permission_overrides
from app.domain.audit import AuditAction
from app.errors import ScopeError, ValidationError
from app.users.models import Organization, PermissionOverride, User
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
