"""Permission overrides (INHERIT/GRANT/REVOKE) and organization lifecycle gating.

Three things are under test:

* `effective_permissions` / `decide()` — the (role ∪ GRANT) − REVOKE composition,
  computed fresh from the database on every call, with no cache to go stale.
* `app.authorization.overrides` — the domain-layer write path, and its two
  structural guards: no self-modification, no cross-tenant reach.
* `Organization.status` — ACTIVE unchanged, SUSPENDED narrows to reads,
  ARCHIVED refuses everything, and an organization predating the column reads
  as ACTIVE.
"""

from __future__ import annotations

import pytest

from app.authorization.model import (
    AccessDecision,
    CameraScope,
    OrganizationStatus,
    OverrideState,
    Permission,
    Role,
    effective_permissions,
    permissions_for,
)
from app.authorization.overrides import clear_permission_override, set_permission_override
from app.authorization.resolver import decide, parse_organization_status, parse_overrides
from app.errors import ScopeError
from app.users.models import Organization, PermissionOverride, User

from .conftest import bearer, login, make_user

# ── effective_permissions: the pure composition ─────────────────────────────


class TestEffectivePermissions:
    def test_inherit_when_no_override(self) -> None:
        """No GRANT, no REVOKE: the role's own answer stands."""
        roles = frozenset({Role.KITCHEN_SUPERVISOR})
        assert effective_permissions(roles) == permissions_for(roles)

    def test_grant_adds_a_permission_the_role_does_not_carry(self) -> None:
        roles = frozenset({Role.KITCHEN_SUPERVISOR})
        assert Permission.VIEW_EVIDENCE not in permissions_for(roles)
        result = effective_permissions(roles, granted=frozenset({Permission.VIEW_EVIDENCE}))
        assert Permission.VIEW_EVIDENCE in result

    def test_revoke_removes_a_permission_the_role_does_carry(self) -> None:
        roles = frozenset({Role.RESTAURANT_MANAGER})
        assert Permission.VIEW_LIVE in permissions_for(roles)
        result = effective_permissions(roles, revoked=frozenset({Permission.VIEW_LIVE}))
        assert Permission.VIEW_LIVE not in result

    def test_revoke_wins_over_role_and_grant_together(self) -> None:
        """A REVOKE is the more specific statement and always wins."""
        roles = frozenset({Role.RESTAURANT_MANAGER})
        result = effective_permissions(
            roles,
            granted=frozenset({Permission.VIEW_LIVE}),
            revoked=frozenset({Permission.VIEW_LIVE}),
        )
        assert Permission.VIEW_LIVE not in result

    def test_multiple_roles_and_multiple_overrides_compose(self) -> None:
        roles = frozenset({Role.KITCHEN_SUPERVISOR, Role.HYGIENE_OFFICER})
        base = permissions_for(roles)
        assert Permission.VIEW_PATRON_ID not in base
        assert Permission.RESOLVE_INCIDENTS in base  # from HYGIENE_OFFICER

        result = effective_permissions(
            roles,
            granted=frozenset({Permission.VIEW_PATRON_ID}),
            revoked=frozenset({Permission.RESOLVE_INCIDENTS}),
        )
        assert Permission.VIEW_PATRON_ID in result
        assert Permission.RESOLVE_INCIDENTS not in result
        # Everything else about the union is untouched.
        assert Permission.VIEW_CUTTING_BOARD in result

    def test_revoking_every_permission_a_role_holds_leaves_none(self) -> None:
        """The edge `AccessDecision` must represent without silently reverting
        to the role's own answer: zero effective permissions is a real,
        distinct state from 'not stated'."""
        roles = frozenset({Role.KITCHEN_SUPERVISOR})
        result = effective_permissions(roles, revoked=permissions_for(roles))
        assert result == frozenset()

        decision = AccessDecision(
            subject="a@b.c",
            tenant_id="org-1",
            roles=roles,
            cameras=CameraScope.none(),
            permissions=result,
        )
        assert decision.permissions == frozenset()
        assert not decision.has(Permission.VIEW_OBSERVATIONS)


class TestParseOverrides:
    def test_unknown_permission_is_dropped(self) -> None:
        rows = [PermissionOverride(user_id="u1", permission="not_a_real_permission", state="grant")]
        granted, revoked = parse_overrides(rows)
        assert granted == frozenset()
        assert revoked == frozenset()

    def test_unknown_state_is_dropped(self) -> None:
        rows = [PermissionOverride(user_id="u1", permission="view_live", state="maybe")]
        granted, revoked = parse_overrides(rows)
        assert granted == frozenset()
        assert revoked == frozenset()

    def test_grant_and_revoke_sort_into_their_own_sets(self) -> None:
        rows = [
            PermissionOverride(user_id="u1", permission="view_evidence", state="grant"),
            PermissionOverride(user_id="u1", permission="view_live", state="revoke"),
        ]
        granted, revoked = parse_overrides(rows)
        assert granted == frozenset({Permission.VIEW_EVIDENCE})
        assert revoked == frozenset({Permission.VIEW_LIVE})


class TestParseOrganizationStatus:
    def test_missing_reads_as_active(self) -> None:
        assert parse_organization_status(None) is OrganizationStatus.ACTIVE

    def test_known_values_round_trip(self) -> None:
        assert parse_organization_status("active") is OrganizationStatus.ACTIVE
        assert parse_organization_status("SUSPENDED") is OrganizationStatus.SUSPENDED
        assert parse_organization_status(" archived ") is OrganizationStatus.ARCHIVED

    def test_an_unreadable_value_denies_toward_the_narrower_reading(self) -> None:
        """Unlike a missing value, a value this version cannot parse is not
        'no opinion' — it must not silently behave as ACTIVE."""
        assert parse_organization_status("something-new") is OrganizationStatus.SUSPENDED


# ── decide(): the resolver-level integration ─────────────────────────────────


class TestDecideWithOverrides:
    def test_no_override_row_means_role_behavior_wins(self) -> None:
        _, user = make_user(roles=("kitchen_supervisor",))
        user.is_active = True
        decision = decide(user)
        assert decision.permissions == permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))

    def test_a_grant_row_widens_the_decision(self) -> None:
        _, user = make_user(roles=("kitchen_supervisor",))
        user.is_active = True
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="view_evidence", state="grant")
        ]
        decision = decide(user)
        assert decision.has(Permission.VIEW_EVIDENCE)

    def test_a_revoke_row_narrows_the_decision(self) -> None:
        _, user = make_user(roles=("restaurant_manager",))
        user.is_active = True
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="view_live", state="revoke")
        ]
        decision = decide(user)
        assert not decision.has(Permission.VIEW_LIVE)

    def test_removing_the_override_reverts_to_role_behavior(self) -> None:
        _, user = make_user(roles=("restaurant_manager",))
        user.is_active = True
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="view_live", state="revoke")
        ]
        assert not decide(user).has(Permission.VIEW_LIVE)

        user.permission_overrides = []
        assert decide(user).has(Permission.VIEW_LIVE)

    def test_an_inactive_user_reaches_nothing_regardless_of_overrides(self) -> None:
        _, user = make_user(roles=("kitchen_supervisor",))
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="access_devtools", state="grant")
        ]
        user.is_active = False
        decision = decide(user)
        assert decision.permissions == frozenset()

    def test_inherit_revoke_inherit_cycle_never_reads_stale_state(self) -> None:
        """The full read-modify-read-modify-read cycle, each `decide()` call
        rebuilding a brand new `AccessDecision` from the row state at that
        moment, with no cache anywhere to go stale.

        Step 1 — INHERIT (no row): the role's own answer stands, allowed.
        Step 2 — REVOKE: denied, even though the role still carries it.
        Step 3 — back to INHERIT (row removed): allowed again, proving the
        first REVOKE left no residue once its row is gone.
        """
        _, user = make_user(roles=("restaurant_manager",))
        user.is_active = True
        assert Permission.VIEW_LIVE in permissions_for(frozenset({Role.RESTAURANT_MANAGER}))

        # Step 1: INHERIT.
        user.permission_overrides = []
        first = decide(user)
        assert first.has(Permission.VIEW_LIVE)

        # Step 2: REVOKE.
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="view_live", state="revoke")
        ]
        second = decide(user)
        assert not second.has(Permission.VIEW_LIVE)
        # The first decision object is untouched by the second call.
        assert first.has(Permission.VIEW_LIVE)

        # Step 3: back to INHERIT.
        user.permission_overrides = []
        third = decide(user)
        assert third.has(Permission.VIEW_LIVE)
        # The second (denying) decision object is likewise untouched.
        assert not second.has(Permission.VIEW_LIVE)

    def test_revoke_wins_regardless_of_how_many_roles_grant_it(self) -> None:
        """Three roles that each independently carry `VIEW_INCIDENTS`; a
        REVOKE on that permission must still win over every one of them."""
        roles = frozenset(
            {Role.RESTAURANT_MANAGER, Role.KITCHEN_SUPERVISOR, Role.HYGIENE_OFFICER}
        )
        for role in roles:
            assert Permission.VIEW_INCIDENTS in permissions_for(frozenset({role}))

        _, user = make_user(roles=tuple(r.value for r in roles))
        user.is_active = True
        user.permission_overrides = [
            PermissionOverride(user_id=user.id, permission="view_incidents", state="revoke")
        ]
        decision = decide(user)
        assert not decision.has(Permission.VIEW_INCIDENTS)
        # Unrelated permissions the roles carry are untouched.
        assert decision.has(Permission.VIEW_CAMERA_HEALTH)


# ── organization lifecycle: ACTIVE / SUSPENDED / ARCHIVED ───────────────────


class TestOrganizationLifecycleAtDecide:
    def test_active_organization_is_unaffected(self) -> None:
        org, user = make_user(roles=("restaurant_manager",))
        user.is_active = True
        org.status = "active"
        user.organization = org
        decision = decide(user)
        assert decision.permissions == permissions_for(frozenset({Role.RESTAURANT_MANAGER}))

    def test_suspended_organization_strips_manage_permissions(self) -> None:
        org, user = make_user(roles=("org_admin",))
        user.is_active = True
        org.status = "suspended"
        user.organization = org
        decision = decide(user)
        assert Permission.MANAGE_CAMERAS not in decision.permissions
        assert Permission.MANAGE_ORGANIZATION not in decision.permissions
        # Reads survive.
        assert decision.has(Permission.VIEW_CAMERAS)
        assert decision.has(Permission.VIEW_OBSERVATIONS)

    def test_a_missing_organization_status_column_value_reads_as_active(self) -> None:
        """An organization row that predates this column has no `.status`
        override applied in the fixture; the parser must still land on ACTIVE."""
        org, user = make_user(roles=("restaurant_manager",))
        # Simulate the pre-migration row shape: attribute present (the ORM
        # default already wrote 'active'), never left unset in practice, but
        # the parser is what actually guarantees the safe reading.
        assert parse_organization_status(org.status) is OrganizationStatus.ACTIVE


class TestOrganizationLifecycleOverHttp:
    @pytest.fixture
    async def org_and_client(self, seeded, client):
        return seeded, client

    async def test_suspended_org_allows_login_and_reads_but_blocks_writes(
        self, seeded, client
    ) -> None:
        database = seeded.state.database
        async with database.session_scope() as session:
            org = await session.get(Organization, "org-test")
            org.status = "suspended"

        headers = await bearer(client, "manager@example.com")
        me = await client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200

        status_resp = await client.get("/api/v1/status", headers=headers)
        assert status_resp.status_code == 200

        # restaurant_manager holds no MANAGE_* permission to begin with, so
        # exercise the org_admin account instead, where the effect is visible.
        async with database.session_scope() as session:
            _, admin = make_user(
                email="admin2@example.com",
                roles=("org_admin",),
                camera_breadth="all_in_tenant",
                camera_ids="",
            )
            session.add(admin)

        admin_headers = await bearer(client, "admin2@example.com")
        me2 = (await client.get("/api/v1/auth/me", headers=admin_headers)).json()
        assert "manage_organization" not in me2["permissions"]
        assert "manage_cameras" not in me2["permissions"]
        assert "view_cameras" in me2["permissions"]

    async def test_archived_org_blocks_login(self, seeded, client) -> None:
        database = seeded.state.database
        async with database.session_scope() as session:
            org = await session.get(Organization, "org-test")
            org.status = "archived"

        response = await login(client, "manager@example.com")
        assert response.status_code == 401
        assert response.json()["code"] == "INVALID_CREDENTIALS"

    async def test_archived_org_blocks_an_already_issued_token(self, seeded, client) -> None:
        headers = await bearer(client, "manager@example.com")

        database = seeded.state.database
        async with database.session_scope() as session:
            org = await session.get(Organization, "org-test")
            org.status = "archived"

        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 401

    async def test_active_org_login_and_access_are_unchanged(self, seeded, client) -> None:
        headers = await bearer(client, "manager@example.com")
        response = await client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

    async def test_an_override_takes_effect_on_the_very_next_request(self, seeded, client) -> None:
        """No cache to invalidate: a GRANT written mid-session is visible on
        the next call using the same still-valid access token."""
        headers = await bearer(client, "manager@example.com")
        before = (await client.get("/api/v1/auth/me", headers=headers)).json()
        assert "view_audit" not in before["permissions"]

        database = seeded.state.database
        async with database.session_scope() as session:
            user = await session.get(User, "user-manager@example.com")
            session.add(PermissionOverride(user_id=user.id, permission="view_audit", state="grant"))

        after = (await client.get("/api/v1/auth/me", headers=headers)).json()
        assert "view_audit" in after["permissions"]


# ── app.authorization.overrides: the domain write path ───────────────────────


class TestSetPermissionOverride:
    @pytest.fixture
    async def two_users_same_org(self, app):
        database = app.state.database
        async with database.session_scope() as session:
            org, actor = make_user(email="actor@example.com", roles=("org_admin",))
            session.add(org)
            session.add(actor)
            _, target = make_user(email="target@example.com", roles=("kitchen_supervisor",))
            session.add(target)
        return app

    async def test_a_grant_is_visible_through_decide(self, two_users_same_org) -> None:
        database = two_users_same_org.state.database
        async with database.session_scope() as session:
            actor = await session.get(User, "user-actor@example.com")
            target = await session.get(User, "user-target@example.com")
            await set_permission_override(
                session,
                actor=actor,
                target=target,
                permission=Permission.VIEW_EVIDENCE,
                state=OverrideState.GRANT,
            )

        async with database.session_scope() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            result = await session.execute(
                select(User)
                .where(User.id == "user-target@example.com")
                .options(
                    selectinload(User.role_assignments),
                    selectinload(User.access_grants),
                    selectinload(User.permission_overrides),
                    selectinload(User.organization),
                )
            )
            target = result.scalar_one()
            assert decide(target).has(Permission.VIEW_EVIDENCE)

    async def test_clearing_an_override_removes_it(self, two_users_same_org) -> None:
        database = two_users_same_org.state.database
        async with database.session_scope() as session:
            actor = await session.get(User, "user-actor@example.com")
            target = await session.get(User, "user-target@example.com")
            await set_permission_override(
                session,
                actor=actor,
                target=target,
                permission=Permission.VIEW_EVIDENCE,
                state=OverrideState.GRANT,
            )

        async with database.session_scope() as session:
            actor = await session.get(User, "user-actor@example.com")
            target = await session.get(User, "user-target@example.com")
            await clear_permission_override(
                session, actor=actor, target=target, permission=Permission.VIEW_EVIDENCE
            )

        async with database.session_scope() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            result = await session.execute(
                select(User)
                .where(User.id == "user-target@example.com")
                .options(
                    selectinload(User.role_assignments),
                    selectinload(User.access_grants),
                    selectinload(User.permission_overrides),
                    selectinload(User.organization),
                )
            )
            target = result.scalar_one()
            assert not decide(target).has(Permission.VIEW_EVIDENCE)

    async def test_self_modification_is_refused(self, two_users_same_org) -> None:
        database = two_users_same_org.state.database
        async with database.session_scope() as session:
            actor = await session.get(User, "user-actor@example.com")
            with pytest.raises(ScopeError, match="themselves"):
                await set_permission_override(
                    session,
                    actor=actor,
                    target=actor,
                    permission=Permission.MANAGE_USERS,
                    state=OverrideState.GRANT,
                )

    async def test_cross_tenant_modification_is_refused(self, seeded) -> None:
        database = seeded.state.database
        async with database.session_scope() as session:
            from sqlalchemy import select

            actor = (
                await session.execute(select(User).where(User.email == "outsider@example.com"))
            ).scalar_one()
            target = (
                await session.execute(select(User).where(User.email == "manager@example.com"))
            ).scalar_one()
            assert actor.organization_id != target.organization_id
            with pytest.raises(ScopeError, match="cross-tenant"):
                await set_permission_override(
                    session,
                    actor=actor,
                    target=target,
                    permission=Permission.VIEW_EVIDENCE,
                    state=OverrideState.GRANT,
                )
