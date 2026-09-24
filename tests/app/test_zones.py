"""Zones: the estate's one placement level.

A zone belongs to an organization and holds cameras. The rules worth holding:
its tenancy comes from the session and never from the request, another
organization's zone is *not found* rather than forbidden, and a zone that holds
cameras deactivates rather than deleting — it is part of the record of what was
watched and where.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.models import Camera, Zone
from tests.app.conftest import bearer, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/zones"


@pytest_asyncio.fixture
async def estate(seeded):
    """Three zones in org-test, one of them holding a camera, and one in
    org-other that must never be visible from org-test."""
    async with seeded.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
        session.add(Zone(id="zone-kitchen", organization_id="org-test", name="Kitchen"))
        session.add(Zone(id="zone-dine", organization_id="org-test", name="Dine hall"))
        session.add(Zone(id="zone-billing", organization_id="org-test", name="Billing"))
        session.add(Zone(id="zone-theirs", organization_id="org-other", name="Their kitchen"))
        session.add(
            Camera(
                id="cam-row-1",
                organization_id="org-test",
                zone_id="zone-kitchen",
                camera_key="cam-01",
                name="Prep line",
                recorder_id="rec-org-test",
                channel=1,
            )
        )
    return seeded


async def test_zones_are_listed_with_their_camera_counts(estate, client: AsyncClient):
    headers = await bearer(client, "manager@example.com")
    body = (await client.get(BASE, headers=headers)).json()

    by_name = {z["name"]: z for z in body["zones"]}
    assert set(by_name) == {"Kitchen", "Dine hall", "Billing"}
    assert by_name["Kitchen"]["camera_count"] == 1
    assert by_name["Billing"]["camera_count"] == 0
    assert body["total"] == 3


async def test_the_list_is_paginated_and_says_how_many_there_are(estate, client: AsyncClient):
    """Paginated from the start. The version that fetches every row works right
    up until the day it does not."""
    headers = await bearer(client, "manager@example.com")
    body = (await client.get(f"{BASE}?limit=2&offset=0", headers=headers)).json()

    assert len(body["zones"]) == 2
    assert (body["count"], body["total"], body["limit"], body["offset"]) == (2, 3, 2, 0)

    second = (await client.get(f"{BASE}?limit=2&offset=2", headers=headers)).json()
    assert len(second["zones"]) == 1
    assert second["total"] == 3


async def test_another_organizations_zones_are_not_in_the_list(estate, client: AsyncClient):
    headers = await bearer(client, "manager@example.com")
    body = (await client.get(BASE, headers=headers)).json()
    assert "Their kitchen" not in {z["name"] for z in body["zones"]}


async def test_another_organizations_zone_is_not_found(estate, client: AsyncClient):
    """404, never 403: existence across a tenant boundary is a disclosure."""
    headers = await bearer(client, "admin@example.com")
    refused = await client.get(f"{BASE}/zone-theirs", headers=headers)
    assert refused.status_code == 404, refused.text


async def test_a_zone_is_created_against_the_session_not_the_request(estate, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    created = await client.post(
        BASE, json={"name": "Pantry", "organization_id": "org-other"}, headers=headers
    )

    assert created.status_code == 200, created.text
    assert created.json()["camera_count"] == 0

    async with estate.state.database.session_scope() as session:
        zone = (await session.execute(select(Zone).where(Zone.name == "Pantry"))).scalar_one()
    # The organization in the body is ignored: tenancy is never request input.
    assert zone.organization_id == "org-test"


async def test_two_zones_cannot_share_a_name(estate, client: AsyncClient):
    """Two rows called Kitchen cannot be told apart when placing a camera."""
    headers = await bearer(client, "admin@example.com")
    refused = await client.post(BASE, json={"name": "kitchen"}, headers=headers)
    assert refused.status_code == 409, refused.text


async def test_a_zone_can_be_renamed(estate, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    renamed = await client.patch(f"{BASE}/zone-dine", json={"name": "Dining hall"}, headers=headers)
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Dining hall"


async def test_an_empty_zone_can_be_deleted(estate, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    deleted = await client.delete(f"{BASE}/zone-billing", headers=headers)
    assert deleted.status_code == 200, deleted.text

    async with estate.state.database.session_scope() as session:
        assert (
            await session.execute(select(Zone).where(Zone.id == "zone-billing"))
        ).scalar_one_or_none() is None


async def test_a_zone_holding_cameras_is_not_deleted_and_the_refusal_says_why(
    estate, client: AsyncClient
):
    headers = await bearer(client, "admin@example.com")
    refused = await client.delete(f"{BASE}/zone-kitchen", headers=headers)

    assert refused.status_code == 409, refused.text
    message = refused.json()["message"]
    assert "1 camera" in message
    assert "Deactivating" in message

    async with estate.state.database.session_scope() as session:
        assert (
            await session.execute(select(Zone).where(Zone.id == "zone-kitchen"))
        ).scalar_one_or_none() is not None


async def test_a_zone_with_cameras_can_be_deactivated_instead(estate, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    stood_down = await client.patch(
        f"{BASE}/zone-kitchen", json={"is_active": False}, headers=headers
    )
    assert stood_down.status_code == 200, stood_down.text
    assert stood_down.json()["is_active"] is False
    # Its cameras are untouched: the record of what was watched survives.
    assert stood_down.json()["camera_count"] == 1


async def test_reading_zones_needs_view_zones(estate, client: AsyncClient):
    """`auditor` reads the record and holds no estate permissions."""
    headers = await bearer(client, "nocameras@example.com")
    allowed = await client.get(BASE, headers=headers)
    # A restaurant manager holds VIEW_ZONES; the refusal case is the write.
    assert allowed.status_code == 200, allowed.text

    refused = await client.post(BASE, json={"name": "Nope"}, headers=headers)
    assert refused.status_code == 403, refused.text


async def test_creating_a_zone_needs_manage_zones(estate, client: AsyncClient):
    headers = await bearer(client, "supervisor@example.com")
    refused = await client.post(BASE, json={"name": "Nope"}, headers=headers)
    assert refused.status_code == 403, refused.text
