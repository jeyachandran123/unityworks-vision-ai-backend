"""The access matrix inside one organization — User & Access, from the inside.

The same grid the platform console offers, pinned to the caller's own
organization and gated on its own `VIEW_USERS` / `MANAGE_USERS`. The one thing
the platform version never had to answer is the one these tests care most
about: somebody inside an organization may only give out what they hold
themselves. Removing is never restricted; adding is.
"""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.authorization.model import ROLE_PERMISSIONS, Role
from app.domain.audit import AuditAction
from app.domain.models import AuditEvent, Camera, Zone
from app.users.models import PermissionOverride, User

from .conftest import bearer, make_user

USERS = "/api/v1/admin/users"
ROLES = "/api/v1/admin/roles"
MANAGER = sorted(p.value for p in ROLE_PERMISSIONS[Role.RESTAURANT_MANAGER])
SUPERVISOR = sorted(p.value for p in ROLE_PERMISSIONS[Role.KITCHEN_SUPERVISOR])
SUPERVISOR_ID = "user-supervisor@example.com"


@pytest.fixture
async def admin(seeded):
    """An `org_admin` inside org-test, with every camera, and one real camera
    (`cam-01`) so a listed grant can name something that exists."""
    async with seeded.state.database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
        session.add(Zone(id="zone-t", organization_id="org-test", name="Kitchen"))
        await session.flush()
        session.add(
            Camera(
                organization_id="org-test",
                zone_id="zone-t",
                recorder_id="rec-org-test",
                camera_key="cam-01",
                name="Pass",
            )
        )
    return seeded


async def manager_who_manages_users(client: AsyncClient) -> dict[str, str]:
    """A restaurant manager given `manage_users`: may administer people, holds
    less than an Organization Admin, and reaches only two listed cameras."""
    admin_headers = await bearer(client, "admin@example.com")
    granted = await client.put(
        f"{USERS}/user-manager@example.com/permissions/manage_users",
        json={"state": "grant"},
        headers=admin_headers,
    )
    assert granted.status_code == 200, granted.text
    return await bearer(client, "manager@example.com")


async def _put(client, headers, user_id=SUPERVISOR_ID, **body):
    return await client.put(f"{USERS}/{user_id}/access", headers=headers, json=body)


async def _overrides(app, user_id: str) -> set[tuple[str, str]]:
    async with app.state.database.session_scope() as session:
        rows = await session.execute(
            select(PermissionOverride.permission, PermissionOverride.state).where(
                PermissionOverride.user_id == user_id
            )
        )
        return set(rows.all())


# ── role policy ──────────────────────────────────────────────────────────────


async def test_roles_say_which_the_caller_may_grant(admin, client: AsyncClient):
    as_admin = (await client.get(ROLES, headers=await bearer(client, "admin@example.com"))).json()
    assert {r["role"] for r in as_admin["roles"]} == {r.value for r in Role}
    assert all(r["grantable"] for r in as_admin["roles"])

    manager = await manager_who_manages_users(client)
    by_role = {r["role"]: r for r in (await client.get(ROLES, headers=manager)).json()["roles"]}
    assert by_role["kitchen_supervisor"]["grantable"] is True
    assert by_role["restaurant_manager"]["grantable"] is True
    # Auditor carries VIEW_AUDIT and the manager does not hold it.
    assert by_role["auditor"]["grantable"] is False
    assert by_role["org_admin"]["grantable"] is False
    assert by_role["auditor"]["permissions"] == sorted(
        p.value for p in ROLE_PERMISSIONS[Role.AUDITOR]
    )


async def test_roles_need_view_users(admin, client: AsyncClient):
    headers = await bearer(client, "supervisor@example.com")
    assert (await client.get(ROLES, headers=headers)).status_code == 403


# ── reading one person's access ──────────────────────────────────────────────


async def test_access_reads_this_organization_only(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    body = (await client.get(f"{USERS}/user-manager@example.com/access", headers=headers)).json()
    assert body["organization_id"] == "org-test"
    assert body["is_member"] is True
    assert body["role"] == "restaurant_manager"
    assert body["permissions"] == MANAGER
    assert body["effective"] == MANAGER
    assert body["camera_scope"]["breadth"] == "listed"
    assert body["camera_scope"]["camera_keys"] == ["cam-01", "cam-02"]


async def test_another_organizations_person_is_not_found(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await client.get(f"{USERS}/user-outsider@example.com/access", headers=headers)
    assert response.status_code == 404


# ── setting it ───────────────────────────────────────────────────────────────


async def test_an_admin_sets_exactly_the_ticks(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    ticks = sorted((set(SUPERVISOR) - {"view_live"}) | {"view_zones"})
    response = await _put(client, headers, role="kitchen_supervisor", permissions=ticks)
    assert response.status_code == 200, response.text
    assert response.json()["permissions"] == ticks
    assert response.json()["effective"] == ticks
    assert await _overrides(admin, SUPERVISOR_ID) == {
        ("view_zones", "grant"),
        ("view_live", "revoke"),
    }

    async with admin.state.database.session_scope() as session:
        row = (
            await session.execute(
                select(AuditEvent).where(AuditEvent.action == AuditAction.ACCESS_SET.value)
            )
        ).scalar_one()
    assert (row.organization_id, row.actor, row.outcome) == (
        "org-test",
        "admin@example.com",
        "success",
    )
    detail = json.loads(row.detail)
    assert (detail["added"], detail["removed"]) == (["view_zones"], ["view_live"])


async def test_a_manager_cannot_tick_what_they_do_not_hold(admin, client: AsyncClient):
    manager = await manager_who_manages_users(client)
    response = await _put(
        client,
        manager,
        role="kitchen_supervisor",
        permissions=sorted(set(SUPERVISOR) | {"view_audit"}),
    )
    assert response.status_code == 403
    assert "view_audit" in response.text
    assert await _overrides(admin, SUPERVISOR_ID) == set()


async def test_a_manager_may_always_take_away(admin, client: AsyncClient):
    manager = await manager_who_manages_users(client)
    response = await _put(
        client,
        manager,
        role="kitchen_supervisor",
        permissions=sorted(set(SUPERVISOR) - {"view_live"}),
    )
    assert response.status_code == 200, response.text
    assert "view_live" not in response.json()["effective"]


async def test_a_manager_cannot_record_a_wider_role_as_the_template(admin, client: AsyncClient):
    """Even with every tick inside their own reach: the role outlives the ticks,
    and clearing the exceptions later would hand over the whole role."""
    manager = await manager_who_manages_users(client)
    response = await _put(client, manager, role="org_admin", permissions=SUPERVISOR)
    assert response.status_code == 403


async def test_a_manager_cannot_widen_cameras_beyond_their_own(admin, client: AsyncClient):
    manager = await manager_who_manages_users(client)
    response = await _put(
        client,
        manager,
        role="kitchen_supervisor",
        permissions=SUPERVISOR,
        camera_breadth="all_in_tenant",
    )
    assert response.status_code == 403


async def test_specific_cameras_can_be_given_here(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await _put(
        client,
        headers,
        role="kitchen_supervisor",
        permissions=SUPERVISOR,
        camera_breadth="listed",
        camera_keys=["cam-01"],
    )
    assert response.status_code == 200, response.text
    scope = response.json()["camera_scope"]
    assert (scope["breadth"], scope["camera_keys"]) == ("listed", ["cam-01"])


async def test_an_unknown_camera_is_refused(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await _put(
        client,
        headers,
        role="kitchen_supervisor",
        permissions=SUPERVISOR,
        camera_breadth="listed",
        camera_keys=["cam-nope"],
    )
    assert response.status_code == 422


async def test_nobody_edits_their_own_access_and_the_attempt_is_audited(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await _put(
        client, headers, user_id="user-admin@example.com", role=None, permissions=["view_live"]
    )
    assert response.status_code == 403
    async with admin.state.database.session_scope() as session:
        row = (
            await session.execute(
                select(AuditEvent).where(
                    AuditEvent.action == AuditAction.ACCESS_SET.value,
                    AuditEvent.outcome == "denied",
                )
            )
        ).scalar_one()
    assert (row.organization_id, json.loads(row.detail)["reason"]) == ("org-test", "self")


async def test_another_organizations_person_cannot_be_set(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await _put(
        client, headers, user_id="user-outsider@example.com", role=None, permissions=[]
    )
    assert response.status_code == 404


async def test_setting_access_needs_manage_users(admin, client: AsyncClient):
    headers = await bearer(client, "manager@example.com")
    response = await _put(client, headers, role="kitchen_supervisor", permissions=SUPERVISOR)
    assert response.status_code == 403


async def test_a_tick_here_is_a_door_there(admin, client: AsyncClient):
    """End to end: untick `view_zones` for the manager, and the zones route
    refuses them on the very next request."""
    headers = await bearer(client, "admin@example.com")
    manager = await bearer(client, "manager@example.com")
    assert (await client.get("/api/v1/zones", headers=manager)).status_code == 200

    await _put(
        client,
        headers,
        user_id="user-manager@example.com",
        role="restaurant_manager",
        permissions=sorted(set(MANAGER) - {"view_zones"}),
    )
    assert (await client.get("/api/v1/zones", headers=manager)).status_code == 403


# ── creating somebody with their ticks ───────────────────────────────────────


def _draft(**overrides):
    body = {
        "email": "new@example.com",
        "display_name": "New Person",
        "password": "a-perfectly-fine-password",
        "role": "kitchen_supervisor",
        "permissions": sorted(set(SUPERVISOR) - {"view_live"}),
        "camera_scope": {"breadth": "none"},
    }
    body.update(overrides)
    return body


async def _user_count(app) -> int:
    async with app.state.database.session_scope() as session:
        return len((await session.execute(select(User.id))).all())


async def test_a_person_is_created_with_exactly_their_ticks(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    response = await client.post(USERS, headers=headers, json=_draft())
    assert response.status_code == 200, response.text
    user_id = response.json()["id"]
    assert response.json()["roles"] == ["kitchen_supervisor"]

    access = (await client.get(f"{USERS}/{user_id}/access", headers=headers)).json()
    assert access["permissions"] == sorted(set(SUPERVISOR) - {"view_live"})


async def test_a_manager_cannot_create_somebody_wider_than_themselves(admin, client: AsyncClient):
    manager = await manager_who_manages_users(client)
    before = await _user_count(admin)
    response = await client.post(
        USERS,
        headers=manager,
        json=_draft(permissions=sorted(set(SUPERVISOR) | {"view_audit"})),
    )
    assert response.status_code == 403
    assert await _user_count(admin) == before


async def test_roles_and_permissions_together_are_refused(admin, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    before = await _user_count(admin)
    response = await client.post(USERS, headers=headers, json=_draft(roles=["auditor"]))
    assert response.status_code == 422
    assert await _user_count(admin) == before
