"""The user management, role assignment, and permission-override admin API.

Stage 5: the HTTP layer over Stage 1-4's already-tested domain model
(`app.authorization.overrides`, `app.authorization.resolver.decide`). These
tests exercise the route layer only — INHERIT/GRANT/REVOKE composition itself
is covered by `tests/app/test_permission_overrides.py` and is not re-tested
here beyond confirming the HTTP surface reaches the same chokepoint.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.users.models import Organization

from .conftest import bearer, login, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/admin/users"


@pytest.fixture
async def admin(seeded):
    """An `org_admin` inside org-test. The seeded admin belongs to org-other."""
    database = seeded.state.database
    async with database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


# ── router-level gate ────────────────────────────────────────────────────────


async def test_manage_users_gates_the_whole_router(client: AsyncClient, seeded) -> None:
    """`kitchen_supervisor` holds no `MANAGE_USERS`."""
    headers = await bearer(client, "supervisor@example.com")
    response = await client.get(BASE, headers=headers)
    assert response.status_code == 403


async def test_restaurant_manager_also_lacks_manage_users(client: AsyncClient, seeded) -> None:
    """The roster is readable; changing it is not.

    Reads are gated on `VIEW_USERS` and writes on `MANAGE_USERS`, so a
    restaurant manager — who holds the first and not the second — can see who
    works here and can change nothing about them. Gating the reads on
    `MANAGE_USERS` too had made the roster invisible to everyone who could not
    also rewrite it, which is not how any other domain in this application
    works.
    """
    headers = await bearer(client, "manager@example.com")

    assert (await client.get(BASE, headers=headers)).status_code == 200

    created = await client.post(
        BASE,
        json={
            "email": "new@example.com",
            "roles": [],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert created.status_code == 403, "a read permission became a write"


# ── list / get, tenant scope ─────────────────────────────────────────────────


async def test_admin_lists_users_scoped_to_own_tenant(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.get(BASE, headers=headers)
    assert response.status_code == 200
    emails = {u["email"] for u in response.json()["users"]}
    assert "manager@example.com" in emails
    assert "admin@example.com" in emails
    # org-other's user must never appear.
    assert "outsider@example.com" not in emails
    for user in response.json()["users"]:
        assert "password_hash" not in user
        assert "password" not in user


async def test_get_one_user(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.get(f"{BASE}/user-manager@example.com", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "manager@example.com"
    assert body["roles"] == ["restaurant_manager"]
    assert "password_hash" not in body


async def test_getting_another_organizations_user_is_404_not_403(
    client: AsyncClient, admin
) -> None:
    """Existence across a tenant boundary is itself a disclosure."""
    headers = await bearer(client, "admin@example.com")
    response = await client.get(f"{BASE}/user-outsider@example.com", headers=headers)
    assert response.status_code == 404


async def test_assigning_a_role_to_a_cross_tenant_user_is_404_not_403(
    client: AsyncClient, admin
) -> None:
    """`outsider@example.com` belongs to `org-other`; `admin` belongs to
    `org-test`. Role assignment must resolve the target through the same
    tenant-scoped lookup every other route uses, refusing with 404 rather
    than disclosing the account exists in another organization."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        f"{BASE}/user-outsider@example.com/roles",
        json={"role": "restaurant_manager"},
        headers=headers,
    )
    assert response.status_code == 404


# ── create ────────────────────────────────────────────────────────────────────


async def test_admin_creates_a_user_with_a_role_it_holds_permissions_for(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "newmanager@example.com",
            "display_name": "New Manager",
            "roles": ["restaurant_manager"],
            "camera_scope": {"breadth": "all_in_tenant"},
            "password": "a-perfectly-fine-password",
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["email"] == "newmanager@example.com"
    assert body["roles"] == ["restaurant_manager"]
    assert body["is_active"] is True
    assert "generated_password" not in body
    assert "password" not in body
    assert "password_hash" not in body

    # The new account can actually authenticate with the password given.
    login_resp = await login(client, "newmanager@example.com", "a-perfectly-fine-password")
    assert login_resp.status_code == 200


async def test_creating_a_user_without_a_password_generates_and_returns_one_once(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "generated@example.com",
            "roles": [],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text
    generated = response.json()["generated_password"]
    assert generated

    login_resp = await login(client, "generated@example.com", generated)
    assert login_resp.status_code == 200


async def test_duplicate_email_within_org_is_a_clean_conflict(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "manager@example.com",
            "roles": [],
            "camera_scope": {"breadth": "none"},
            "password": "irrelevant-password-1",
        },
        headers=headers,
    )
    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"


async def test_create_user_rejects_an_unknown_role(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "x@example.com",
            "roles": ["galaxy_emperor"],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert response.status_code == 422


async def test_org_admin_cannot_create_a_super_admin(client: AsyncClient, admin) -> None:
    """`super_admin` carries permissions (`access_devtools`, ...) `org_admin`
    does not hold — the escalation rule 2 in the module docstring exists to
    close, checked via permission subset rather than role identity."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "wannabe@example.com",
            "roles": ["super_admin"],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert response.status_code == 403


# ── update / activate / deactivate ──────────────────────────────────────────


async def test_update_display_name_is_audited(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.patch(
        f"{BASE}/user-manager@example.com",
        json={"display_name": "Renamed Manager"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["display_name"] == "Renamed Manager"

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "user.updated" in actions


async def test_deactivate_then_reactivate_blocks_and_restores_login(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")

    deactivated = await client.post(f"{BASE}/user-manager@example.com/deactivate", headers=headers)
    assert deactivated.status_code == 200
    assert deactivated.json()["is_active"] is False

    login_fail = await login(client, "manager@example.com")
    assert login_fail.status_code == 401
    assert login_fail.json()["code"] == "INVALID_CREDENTIALS"

    reactivated = await client.post(f"{BASE}/user-manager@example.com/activate", headers=headers)
    assert reactivated.status_code == 200
    assert reactivated.json()["is_active"] is True

    login_ok = await login(client, "manager@example.com")
    assert login_ok.status_code == 200

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "user.deactivated" in actions
    assert "user.activated" in actions


async def test_a_deactivated_users_existing_token_is_refused_immediately(
    client: AsyncClient, admin
) -> None:
    manager_headers = await bearer(client, "manager@example.com")
    admin_headers = await bearer(client, "admin@example.com")

    await client.post(f"{BASE}/user-manager@example.com/deactivate", headers=admin_headers)

    response = await client.get("/api/v1/auth/me", headers=manager_headers)
    assert response.status_code == 401


async def test_an_admin_may_not_deactivate_their_own_account(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(f"{BASE}/user-admin@example.com/deactivate", headers=headers)
    assert response.status_code == 403


# ── role assignment ──────────────────────────────────────────────────────────


async def test_assign_and_remove_a_role(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")

    assigned = await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "kitchen_supervisor"},
        headers=headers,
    )
    # Already held by the seeded fixture; still succeeds, idempotently.
    assert assigned.status_code == 200

    added = await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "restaurant_manager"},
        headers=headers,
    )
    assert added.status_code == 200
    assert set(added.json()["roles"]) == {"kitchen_supervisor", "restaurant_manager"}

    removed = await client.delete(
        f"{BASE}/user-supervisor@example.com/roles/restaurant_manager", headers=headers
    )
    assert removed.status_code == 200
    assert removed.json()["roles"] == ["kitchen_supervisor"]

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "role.assigned" in actions
    assert "role.removed" in actions


async def test_removing_a_role_the_user_does_not_hold_is_a_no_op(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.delete(
        f"{BASE}/user-supervisor@example.com/roles/hygiene_officer", headers=headers
    )
    assert response.status_code == 200
    assert response.json()["roles"] == ["kitchen_supervisor"]


async def test_cannot_assign_a_role_carrying_a_permission_the_actor_lacks(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "developer"},
        headers=headers,
    )
    assert response.status_code == 403


async def test_org_admin_may_assign_a_role_it_does_not_itself_hold(
    client: AsyncClient, admin
) -> None:
    """The primary real-world case: `org_admin`'s own `RoleAssignment` rows
    name only `org_admin`, yet every permission `restaurant_manager` carries
    is already a subset of what `org_admin` holds."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "restaurant_manager"},
        headers=headers,
    )
    assert response.status_code == 200
    assert "restaurant_manager" in response.json()["roles"]


async def test_cannot_assign_an_unknown_role(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "not_a_role"},
        headers=headers,
    )
    assert response.status_code == 422


async def test_an_actor_may_not_change_their_own_roles(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        f"{BASE}/user-admin@example.com/roles",
        json={"role": "org_admin"},
        headers=headers,
    )
    assert response.status_code == 403


async def test_a_grant_override_survives_unrelated_role_removal(
    client: AsyncClient, admin
) -> None:
    """`PermissionOverride` and `RoleAssignment` are independent tables keyed
    only off `user_id` (`app/users/models.py`) — no foreign key ties one to
    the other. Removing every role a user holds must not silently delete a
    GRANT override: `kitchen_supervisor` does not carry `view_evidence`, so
    the GRANT is the only thing giving it, and it must still be in force
    after the role that never granted it in the first place is removed."""
    headers = await bearer(client, "admin@example.com")
    await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/view_evidence",
        json={"state": "grant"},
        headers=headers,
    )
    removed = await client.delete(
        f"{BASE}/user-supervisor@example.com/roles/kitchen_supervisor", headers=headers
    )
    assert removed.status_code == 200
    assert removed.json()["roles"] == []

    rows = (
        await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    ).json()["permissions"]
    row = next(r for r in rows if r["permission"] == "view_evidence")
    assert row["state"] == "grant"
    assert row["effective"] is True


async def test_a_revoke_override_survives_role_removal(client: AsyncClient, admin) -> None:
    """Give `supervisor` a second role (`restaurant_manager`) that also
    carries `view_incidents`, REVOKE `view_incidents`, then remove
    `kitchen_supervisor` (one of the two roles that carried it). The REVOKE
    row must still exist and still win — proving it was never tied to the
    specific role that happened to be held when it was written."""
    headers = await bearer(client, "admin@example.com")
    await client.post(
        f"{BASE}/user-supervisor@example.com/roles",
        json={"role": "restaurant_manager"},
        headers=headers,
    )
    await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/view_incidents",
        json={"state": "revoke"},
        headers=headers,
    )
    removed = await client.delete(
        f"{BASE}/user-supervisor@example.com/roles/kitchen_supervisor", headers=headers
    )
    assert removed.status_code == 200
    assert removed.json()["roles"] == ["restaurant_manager"]

    rows = (
        await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    ).json()["permissions"]
    row = next(r for r in rows if r["permission"] == "view_incidents")
    # restaurant_manager alone still carries view_incidents...
    assert row["role_grants"] is True
    # ...but the surviving REVOKE row still wins.
    assert row["state"] == "revoke"
    assert row["effective"] is False


# ── permission overrides ─────────────────────────────────────────────────────


async def test_listing_permissions_shows_inherit_by_default(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    assert response.status_code == 200
    rows = {r["permission"]: r for r in response.json()["permissions"]}
    assert rows["view_live"]["state"] == "inherit"
    assert rows["view_live"]["effective"] is True
    assert rows["view_evidence"]["state"] == "inherit"
    assert rows["view_evidence"]["effective"] is False


async def test_listing_permissions_shows_role_grants(client: AsyncClient, admin) -> None:
    """`role_grants` reports whether the role alone (ignoring any override)
    carries the permission — additive to Stage 5's `state`/`effective`
    fields, computed from the same `permissions_for(roles)` union
    `decide()` itself calls internally, not re-derived independently."""
    headers = await bearer(client, "admin@example.com")
    response = await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    assert response.status_code == 200
    rows = {r["permission"]: r for r in response.json()["permissions"]}

    # role_grants=true + INHERIT -> effective true
    assert rows["view_live"]["role_grants"] is True
    assert rows["view_live"]["state"] == "inherit"
    assert rows["view_live"]["effective"] is True

    # role_grants=false + INHERIT -> effective false
    assert rows["view_evidence"]["role_grants"] is False
    assert rows["view_evidence"]["state"] == "inherit"
    assert rows["view_evidence"]["effective"] is False


async def test_role_grants_true_plus_revoke_still_loses_to_revoke(
    client: AsyncClient, admin
) -> None:
    """`restaurant_manager` holds `view_live` (role_grants=true); REVOKE still
    wins over the role, per Stage 1-4's own composition rule."""
    headers = await bearer(client, "admin@example.com")
    await client.put(
        f"{BASE}/user-manager@example.com/permissions/view_live",
        json={"state": "revoke"},
        headers=headers,
    )
    rows = (
        await client.get(f"{BASE}/user-manager@example.com/permissions", headers=headers)
    ).json()["permissions"]
    row = next(r for r in rows if r["permission"] == "view_live")
    assert row["role_grants"] is True
    assert row["state"] == "revoke"
    assert row["effective"] is False


async def test_role_grants_false_plus_grant_widens_to_effective_true(
    client: AsyncClient, admin
) -> None:
    """`kitchen_supervisor` does not hold `view_evidence` (role_grants=false);
    a GRANT override still widens effective access to true."""
    headers = await bearer(client, "admin@example.com")
    await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/view_evidence",
        json={"state": "grant"},
        headers=headers,
    )
    rows = (
        await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    ).json()["permissions"]
    row = next(r for r in rows if r["permission"] == "view_evidence")
    assert row["role_grants"] is False
    assert row["state"] == "grant"
    assert row["effective"] is True


async def test_grant_widens_and_is_audited(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/view_evidence",
        json={"state": "grant"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "permission": "view_evidence",
        "state": "grant",
        "effective": True,
    }

    rows = (
        await client.get(f"{BASE}/user-supervisor@example.com/permissions", headers=headers)
    ).json()["permissions"]
    row = next(r for r in rows if r["permission"] == "view_evidence")
    assert row["state"] == "grant"
    assert row["effective"] is True

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "permission.granted" in actions


async def test_revoke_narrows_and_wins_over_role(client: AsyncClient, admin) -> None:
    """`restaurant_manager` holds `view_live`; a REVOKE removes it even
    though the role still grants it — REVOKE wins."""
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-manager@example.com/permissions/view_live",
        json={"state": "revoke"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["effective"] is False

    me_headers = await bearer(client, "manager@example.com")
    me = (await client.get("/api/v1/auth/me", headers=me_headers)).json()
    assert "view_live" not in me["permissions"]


async def test_reset_restores_role_behavior(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    await client.put(
        f"{BASE}/user-manager@example.com/permissions/view_live",
        json={"state": "revoke"},
        headers=headers,
    )
    reset = await client.delete(
        f"{BASE}/user-manager@example.com/permissions/view_live", headers=headers
    )
    assert reset.status_code == 200
    assert reset.json() == {"permission": "view_live", "state": "inherit", "effective": True}

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "permission.reset" in actions


async def test_grant_requires_the_actor_to_hold_the_permission(client: AsyncClient, admin) -> None:
    """`org_admin` does not hold `access_devtools`; it may not GRANT it to
    anyone else, even though it holds `MANAGE_USERS`."""
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/access_devtools",
        json={"state": "grant"},
        headers=headers,
    )
    assert response.status_code == 403


async def test_revoke_does_not_require_the_actor_to_hold_the_permission(
    client: AsyncClient, admin
) -> None:
    """The asymmetric rule: REVOKE only narrows, so no "must hold" check
    applies. `org_admin` does not hold `resolve_incidents`... actually it
    does; use an org_admin-held permission on a target whose role grants a
    permission org_admin itself lacks to prove the asymmetry meaningfully:
    revoking `view_model_evaluation` (a `developer`-only permission org_admin
    never holds) from a developer must still succeed."""
    database = admin.state.database
    async with database.session_scope() as session:
        _, dev = make_user(
            email="dev2@example.com",
            roles=("developer",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(dev)

    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-dev2@example.com/permissions/view_model_evaluation",
        json={"state": "revoke"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["effective"] is False


async def test_self_escalation_via_override_is_blocked(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-admin@example.com/permissions/manage_organization",
        json={"state": "grant"},
        headers=headers,
    )
    assert response.status_code == 403


async def test_cross_tenant_target_is_404_not_403_and_does_not_leak_existence(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-outsider@example.com/permissions/view_live",
        json={"state": "grant"},
        headers=headers,
    )
    assert response.status_code == 404


async def test_invalid_permission_value_is_a_clean_validation_error(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/not_a_real_permission",
        json={"state": "grant"},
        headers=headers,
    )
    assert response.status_code == 422


async def test_invalid_override_state_is_a_clean_validation_error(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.put(
        f"{BASE}/user-supervisor@example.com/permissions/view_live",
        json={"state": "maybe"},
        headers=headers,
    )
    assert response.status_code == 422


# ── organization lifecycle interaction ──────────────────────────────────────


async def test_suspended_organization_blocks_manage_users_writes(
    client: AsyncClient, admin
) -> None:
    """No bypass and no redundant special-case check: this route is gated by
    `MANAGE_USERS`, which the SUSPENDED-organization chokepoint in
    `app/authorization/resolver.py:decide()` already strips (it starts with
    `manage_`) — proven by exercising the route, not assumed."""
    database = admin.state.database
    async with database.session_scope() as session:
        org = await session.get(Organization, "org-test")
        org.status = "suspended"

    headers = await bearer(client, "admin@example.com")

    # Reads survive a suspension. That is the whole shape of SUSPENDED: login
    # and reads continue, writes are refused. `VIEW_USERS` is not a `manage_*`
    # permission, so the chokepoint leaves it alone — and an organization on a
    # billing hold can still see its own staff list, which it could not when
    # `MANAGE_USERS` gated the reads too.
    listed = await client.get(BASE, headers=headers)
    assert listed.status_code == 200

    created = await client.post(
        BASE,
        json={
            "email": "x2@example.com",
            "roles": [],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert created.status_code == 403


async def test_archived_organization_blocks_everything_including_this_router(
    client: AsyncClient, admin
) -> None:
    database = admin.state.database
    async with database.session_scope() as session:
        org = await session.get(Organization, "org-test")
        org.status = "archived"

    headers_login = await login(client, "admin@example.com")
    assert headers_login.status_code == 401


# ── multiple users, same role, separate identities ──────────────────────────


async def test_multiple_users_with_the_same_role_stay_independent(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")

    ids: dict[str, str] = {}
    for email in ("staffer1@example.com", "staffer2@example.com"):
        response = await client.post(
            BASE,
            json={
                "email": email,
                "roles": ["restaurant_manager"],
                "camera_scope": {"breadth": "all_in_tenant"},
                "password": "distinct-password-value-1",
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        ids[email] = response.json()["id"]

    # Distinct identities: two different users, same role, different ids.
    assert ids["staffer1@example.com"] != ids["staffer2@example.com"]

    # Independent credentials: staffer1's password does not authenticate
    # staffer2, even though it was supplied identically at creation time.
    ok = await login(client, "staffer1@example.com", "distinct-password-value-1")
    assert ok.status_code == 200

    # Give one an override; confirm it does not leak onto the other.
    await client.put(
        f"{BASE}/{ids['staffer1@example.com']}/permissions/view_audit",
        json={"state": "grant"},
        headers=headers,
    )
    rows1 = (
        await client.get(f"{BASE}/{ids['staffer1@example.com']}/permissions", headers=headers)
    ).json()["permissions"]
    rows2 = (
        await client.get(f"{BASE}/{ids['staffer2@example.com']}/permissions", headers=headers)
    ).json()["permissions"]
    row1 = next(r for r in rows1 if r["permission"] == "view_audit")
    row2 = next(r for r in rows2 if r["permission"] == "view_audit")
    assert row1["state"] == "grant"
    assert row2["state"] == "inherit"

    # Deactivating one does not touch the other.
    await client.post(f"{BASE}/{ids['staffer1@example.com']}/deactivate", headers=headers)
    fail = await login(client, "staffer1@example.com", "distinct-password-value-1")
    assert fail.status_code == 401
    ok2 = await login(client, "staffer2@example.com", "distinct-password-value-1")
    assert ok2.status_code == 200

    # Distinct audit trails, both attributed to the admin who acted, naming
    # the distinct target.
    trail = (await client.get("/api/v1/audit", headers=headers)).json()["events"]
    created_ids = {e["resource_id"] for e in trail if e["action"] == "user.created"}
    assert ids["staffer1@example.com"] in created_ids
    assert ids["staffer2@example.com"] in created_ids


# ── camera access scope: the provisioning bug, and its administration ────────


async def test_a_new_user_receives_a_real_access_grant(
    client: AsyncClient, admin
) -> None:
    """The bug this route shipped with, asserted directly.

    `POST /admin/users` created the user and its roles and wrote no
    `AccessGrant`. A missing grant reads as `CameraScope.none()` — correctly,
    since that is the only safe reading of an absent row — so the account
    authenticated, held every permission its role carried, and could reach no
    cameras. Nothing anywhere reported it.
    """
    headers = await bearer(client, "admin@example.com")
    created = await client.post(
        BASE,
        json={
            "email": "granted@example.com",
            "roles": ["restaurant_manager"],
            "camera_scope": {"breadth": "all_in_tenant"},
            "password": "a-perfectly-fine-password",
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["camera_scope"]["breadth"] == "all_in_tenant"

    scope = await client.get(
        f"{BASE}/{created.json()['id']}/camera-scope", headers=headers
    )
    assert scope.status_code == 200
    assert scope.json()["camera_scope"]["breadth"] == "all_in_tenant", (
        "the account was created without a usable camera grant"
    )


async def test_creating_a_user_without_stating_a_camera_scope_is_refused(
    client: AsyncClient, admin
) -> None:
    """Neither default is safe, so there is no default.

    `none` silently recreates the unusable account; `all_in_tenant` silently
    creates an over-privileged one. The administrator says which.
    """
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={"email": "unscoped@example.com", "roles": []},
        headers=headers,
    )
    assert response.status_code == 422
    assert "camera_scope" in response.text


async def test_none_remains_a_legitimate_stated_answer(
    client: AsyncClient, admin
) -> None:
    """Requiring the field must not become "everyone gets cameras".

    An account that will be scoped later, or one that never needs video, is a
    real case and stays expressible — the point is that it is now *chosen*.
    """
    headers = await bearer(client, "admin@example.com")
    created = await client.post(
        BASE,
        json={
            "email": "deliberately-unscoped@example.com",
            "roles": [],
            "camera_scope": {"breadth": "none"},
            "password": "a-perfectly-fine-password",
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["camera_scope"]["breadth"] == "none"


async def test_a_listed_scope_refuses_a_camera_that_does_not_exist(
    client: AsyncClient, admin
) -> None:
    """A typo must not hand out an empty grant that reads as deliberate."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        BASE,
        json={
            "email": "typo@example.com",
            "roles": [],
            "camera_scope": {"breadth": "listed", "camera_keys": ["cam-does-not-exist"]},
            "password": "a-perfectly-fine-password",
        },
        headers=headers,
    )
    assert response.status_code == 422
    assert "cam-does-not-exist" in response.text


async def test_an_admin_may_not_change_their_own_camera_scope(
    client: AsyncClient, admin
) -> None:
    """Consistent with permission overrides, and for the same reason: an actor
    who can widen their own reach can widen it to everything."""
    headers = await bearer(client, "admin@example.com")
    listed = await client.get(BASE, headers=headers)
    me = next(u for u in listed.json()["users"] if u["email"] == "admin@example.com")

    response = await client.put(
        f"{BASE}/{me['id']}/camera-scope",
        json={"camera_scope": {"breadth": "all_in_tenant"}},
        headers=headers,
    )
    assert response.status_code == 403
