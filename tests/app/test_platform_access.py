"""The platform access matrix — reading and setting a person's access in every
organization, from the control plane.

Two families, as in `test_platform_administration.py`: the matrix does what it
says, and no tenant principal — not even an Organization Admin — can reach it.
The end-to-end cases matter most: they prove a tick here is a door opening on
an ordinary tenant route, because that is the whole point of the feature.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.authorization.model import ROLE_PERMISSIONS, Role
from app.users.models import Organization, PlatformOperatorGrant
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
