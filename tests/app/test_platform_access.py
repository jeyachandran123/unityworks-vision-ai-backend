"""The platform access matrix — reading and setting a person's access in every
organization, from the control plane.

Two families, as in `test_platform_administration.py`: the matrix does what it
says, and no tenant principal — not even an Organization Admin — can reach it.
The end-to-end cases matter most: they prove a tick here is a door opening on
an ordinary tenant route, because that is the whole point of the feature.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.authorization.model import ROLE_PERMISSIONS, Role
from app.domain.audit import AuditAction
from app.domain.models import AuditEvent
from app.users.models import (
    Organization,
    OrganizationMembership,
    PermissionOverride,
    PlatformOperatorGrant,
)
from tests.app.conftest import bearer, make_user

PLATFORM = "/api/v1/platform"
SOLO = "user-solo@example.com"
MANAGER = sorted(p.value for p in ROLE_PERMISSIONS[Role.RESTAURANT_MANAGER])


@pytest_asyncio.fixture
async def estate(app):
    """Acme (named "Test Org" by `make_user`), Borden, and archived Closed Co.

    `solo@` is a restaurant manager at Acme with one listed camera; the
    operator holds platform authority."""
    async with app.state.database.session_scope() as session:
        acme, operator = make_user(
            org_id="org-acme",
            email="operator@example.com",
            roles=("developer",),
            camera_breadth="none",
            camera_ids="",
        )
        session.add_all([acme, operator])
        session.add(Organization(id="org-borden", name="Borden Foods", slug="borden"))
        session.add(
            Organization(id="org-closed", name="Closed Co", slug="closed", status="archived")
        )
        _, solo = make_user(
            org_id="org-acme",
            email="solo@example.com",
            roles=("restaurant_manager",),
            camera_breadth="listed",
            camera_ids="cam-01",
        )
        session.add(solo)
        await session.flush()
        session.add(
            PlatformOperatorGrant(user_id=operator.id, reason="test fixture", granted_by="test")
        )
    return app


@pytest_asyncio.fixture
async def tenant_admin(estate, client: AsyncClient):
    """An Organization Admin: every tenant permission, and no platform reach."""
    async with estate.state.database.session_scope() as session:
        _, admin = make_user(
            org_id="org-acme",
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
    return await bearer(client, "admin@example.com")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"/people/{SOLO}/access"),
        ("PUT", f"/people/{SOLO}/access"),
    ],
)
async def test_no_tenant_role_reaches_the_access_matrix(
    tenant_admin, client: AsyncClient, method: str, path: str
):
    response = await client.request(
        method, f"{PLATFORM}{path}", headers=tenant_admin, json={"organizations": []}
    )
    assert response.status_code == 403, f"{method} {path} returned {response.status_code}"


async def test_access_lists_every_organization_by_name(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/people/{SOLO}/access", headers=headers)).json()

    assert body["user_id"] == SOLO
    assert body["home_organization_id"] == "org-acme"
    assert body["is_platform_operator"] is False
    assert [o["organization_name"] for o in body["organizations"]] == [
        "Borden Foods",
        "Closed Co",
        "Test Org",
    ]


async def test_access_reports_what_the_person_holds_where_they_are_a_member(
    estate, client: AsyncClient
):
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/people/{SOLO}/access", headers=headers)).json()
    acme = next(o for o in body["organizations"] if o["organization_id"] == "org-acme")

    assert acme["is_member"] is True
    assert acme["is_home"] is True
    assert acme["status"] == "active"
    assert acme["role"] == "restaurant_manager"
    assert acme["roles"] == ["restaurant_manager"]
    assert acme["permissions"] == MANAGER
    assert acme["effective"] == MANAGER
    assert acme["camera_scope"]["breadth"] == "listed"
    assert acme["camera_scope"]["camera_keys"] == ["cam-01"]


async def test_access_is_empty_where_the_person_is_not_a_member(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/people/{SOLO}/access", headers=headers)).json()
    closed = next(o for o in body["organizations"] if o["organization_id"] == "org-closed")

    assert closed == {
        "organization_id": "org-closed",
        "organization_name": "Closed Co",
        "status": "archived",
        "is_member": False,
        "is_home": False,
        "role": None,
        "roles": [],
        "permissions": [],
        "effective": [],
        "camera_scope": {"breadth": "none", "camera_keys": [], "site_ids": []},
    }


async def test_access_for_an_unknown_person_is_404(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    response = await client.get(f"{PLATFORM}/people/nobody/access", headers=headers)
    assert response.status_code == 404


async def _put(client: AsyncClient, headers, *items: dict, user_id: str = SOLO):
    return await client.put(
        f"{PLATFORM}/people/{user_id}/access", headers=headers, json={"organizations": list(items)}
    )


def _org(body: dict, organization_id: str) -> dict:
    return next(o for o in body["organizations"] if o["organization_id"] == organization_id)


async def test_saving_makes_access_exactly_the_ticks(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    ticks = sorted((set(MANAGER) | {"manage_zones"}) - {"view_evidence"})
    response = await _put(
        client,
        headers,
        {"organization_id": "org-acme", "role": "restaurant_manager", "permissions": ticks},
    )
    assert response.status_code == 200, response.text
    acme = _org(response.json(), "org-acme")
    assert acme["permissions"] == ticks
    assert acme["effective"] == ticks

    async with estate.state.database.session_scope() as session:
        overrides = set(
            (
                await session.execute(
                    select(PermissionOverride.permission, PermissionOverride.state).where(
                        PermissionOverride.user_id == SOLO
                    )
                )
            ).all()
        )
    assert overrides == {("manage_zones", "grant"), ("view_evidence", "revoke")}


async def test_saving_in_another_organization_admits_and_audits_there(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {
            "organization_id": "org-borden",
            "role": None,
            "permissions": ["view_live", "view_zones"],
            "camera_breadth": "all_in_tenant",
        },
    )
    assert response.status_code == 200, response.text
    borden = _org(response.json(), "org-borden")
    assert borden["is_member"] is True
    assert borden["permissions"] == ["view_live", "view_zones"]
    assert borden["camera_scope"]["breadth"] == "all_in_tenant"

    async with estate.state.database.session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(AuditEvent).where(AuditEvent.organization_id == "org-borden")
                )
            )
            .scalars()
            .all()
        )
    assert sorted(row.action for row in rows) == sorted(
        [AuditAction.ORGANIZATION_MEMBER_ADDED.value, AuditAction.ACCESS_SET.value]
    )
    access_row = next(row for row in rows if row.action == AuditAction.ACCESS_SET.value)
    assert access_row.actor == "operator@example.com"
    assert access_row.resource_id == SOLO


async def test_the_access_set_row_names_the_difference(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    await _put(
        client,
        headers,
        {
            "organization_id": "org-acme",
            "role": "restaurant_manager",
            "permissions": sorted((set(MANAGER) - {"view_live"}) | {"view_audit"}),
        },
    )
    async with estate.state.database.session_scope() as session:
        row = (
            await session.execute(
                select(AuditEvent).where(AuditEvent.action == AuditAction.ACCESS_SET.value)
            )
        ).scalar_one()
    detail = json.loads(row.detail)
    assert detail["added"] == ["view_audit"]
    assert detail["removed"] == ["view_live"]
    assert detail["role"] == "restaurant_manager"
    assert detail["admitted"] is False


async def test_one_archived_organization_refuses_the_whole_request(estate, client: AsyncClient):
    """Review focus 4: checked before the first write, so Borden is not joined."""
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {"organization_id": "org-borden", "role": None, "permissions": ["view_live"]},
        {"organization_id": "org-closed", "role": None, "permissions": ["view_live"]},
    )
    assert response.status_code == 422
    async with estate.state.database.session_scope() as session:
        joined = (
            await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == SOLO,
                    OrganizationMembership.organization_id == "org-borden",
                )
            )
        ).all()
    assert joined == []


@pytest.mark.parametrize(
    ("item", "status"),
    [
        ({"organization_id": "org-acme", "role": None, "permissions": ["fly"]}, 422),
        ({"organization_id": "org-acme", "role": "chef", "permissions": []}, 422),
        (
            {
                "organization_id": "org-acme",
                "role": None,
                "permissions": [],
                "camera_breadth": "listed",
            },
            422,
        ),
        ({"organization_id": "org-nowhere", "role": None, "permissions": ["view_live"]}, 404),
    ],
)
async def test_bad_items_are_refused(estate, client: AsyncClient, item, status):
    headers = await bearer(client, "operator@example.com")
    assert (await _put(client, headers, item)).status_code == status


async def test_an_operator_may_not_set_their_own_access(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {"organization_id": "org-borden", "role": None, "permissions": ["view_live"]},
        user_id="user-operator@example.com",
    )
    assert response.status_code == 403


async def test_another_platform_admin_is_not_given_organizations(estate, client: AsyncClient):
    async with estate.state.database.session_scope() as session:
        _, second = make_user(
            org_id="org-acme",
            email="second-op@example.com",
            roles=(),
            camera_breadth="none",
            camera_ids="",
        )
        session.add(second)
        await session.flush()
        session.add(PlatformOperatorGrant(user_id=second.id, reason="test", granted_by="test"))
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {"organization_id": "org-borden", "role": None, "permissions": ["view_live"]},
        user_id="user-second-op@example.com",
    )
    assert response.status_code == 422


async def test_omitting_camera_breadth_keeps_a_listed_grant(estate, client: AsyncClient):
    """Review focus 5."""
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {"organization_id": "org-acme", "role": "restaurant_manager", "permissions": MANAGER},
    )
    scope = _org(response.json(), "org-acme")["camera_scope"]
    assert (scope["breadth"], scope["camera_keys"]) == ("listed", ["cam-01"])


async def test_a_suspended_organization_stores_writes_it_does_not_yet_honour(
    estate, client: AsyncClient
):
    async with estate.state.database.session_scope() as session:
        borden = await session.get(Organization, "org-borden")
        borden.status = "suspended"
    headers = await bearer(client, "operator@example.com")
    response = await _put(
        client,
        headers,
        {
            "organization_id": "org-borden",
            "role": None,
            "permissions": ["manage_zones", "view_zones"],
        },
    )
    borden = _org(response.json(), "org-borden")
    assert borden["permissions"] == ["manage_zones", "view_zones"]
    assert borden["effective"] == ["view_zones"]


async def test_a_tick_is_a_door_on_an_ordinary_tenant_route(estate, client: AsyncClient):
    """The feature, end to end: untick `view_zones`, the zones route refuses
    solo@ on his very next request; tick it again, it answers."""
    operator = await bearer(client, "operator@example.com")
    solo = await bearer(client, "solo@example.com")
    assert (await client.get("/api/v1/zones", headers=solo)).status_code == 200

    without = sorted(set(MANAGER) - {"view_zones"})
    await _put(
        client,
        operator,
        {"organization_id": "org-acme", "role": "restaurant_manager", "permissions": without},
    )
    assert (await client.get("/api/v1/zones", headers=solo)).status_code == 403

    await _put(
        client,
        operator,
        {"organization_id": "org-acme", "role": "restaurant_manager", "permissions": MANAGER},
    )
    assert (await client.get("/api/v1/zones", headers=solo)).status_code == 200
