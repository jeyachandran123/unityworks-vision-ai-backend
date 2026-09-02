"""Restaurants, zones and the user list.

The interesting assertions are the boundaries rather than the CRUD:

* reading structure and changing it are different permissions
* tenancy comes from the session, never from the request body
* another organisation's rows are invisible, and asking for one by id is a 404
  rather than a 403 — "it exists but is not yours" is itself a disclosure
* the user list carries no credential material, and says why it cannot write
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from .conftest import bearer, make_user

pytestmark = pytest.mark.asyncio


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


# ── restaurants ──────────────────────────────────────────────────────────────


async def test_listing_restaurants_is_empty_before_any_exist(
    client: AsyncClient, seeded
) -> None:
    headers = await bearer(client, "manager@example.com")
    response = await client.get("/api/v1/restaurants", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"restaurants": [], "count": 0}


async def test_a_manager_may_read_structure_but_not_change_it(
    client: AsyncClient, admin
) -> None:
    """Reading is `VIEW_USERS`; writing is `MANAGE_ORGANIZATION`. Not the same."""
    headers = await bearer(client, "manager@example.com")

    assert (await client.get("/api/v1/restaurants", headers=headers)).status_code == 200
    created = await client.post(
        "/api/v1/restaurants", json={"name": "Nowhere"}, headers=headers
    )
    assert created.status_code == 403


async def test_an_admin_creates_a_restaurant_and_it_is_audited(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")

    response = await client.post(
        "/api/v1/restaurants",
        json={"name": "Harbour Kitchen", "timezone": "Asia/Singapore"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "Harbour Kitchen"
    assert body["slug"] == "harbour-kitchen"
    assert body["timezone"] == "Asia/Singapore"
    assert body["zone_count"] == 0

    trail = await client.get("/api/v1/audit", headers=headers)
    actions = [e["action"] for e in trail.json()["events"]]
    assert "restaurant.created" in actions


async def test_a_restaurant_cannot_be_created_into_another_organisation(
    client: AsyncClient, admin
) -> None:
    """`organization_id` in the body is ignored; tenancy comes from the session."""
    headers = await bearer(client, "admin@example.com")
    await client.post(
        "/api/v1/restaurants",
        json={"name": "Smuggled", "organization_id": "org-other"},
        headers=headers,
    )

    outsider = await bearer(client, "outsider@example.com")
    theirs = await client.get("/api/v1/restaurants", headers=outsider)
    assert theirs.json()["count"] == 0, "the row must not have landed in org-other"

    ours = await client.get("/api/v1/restaurants", headers=headers)
    assert [r["name"] for r in ours.json()["restaurants"]] == ["Smuggled"]


async def test_a_nameless_restaurant_is_refused(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.post("/api/v1/restaurants", json={"name": "   "}, headers=headers)
    assert response.status_code == 422


async def test_updating_a_restaurant_records_which_fields_changed(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    created = (
        await client.post("/api/v1/restaurants", json={"name": "Old"}, headers=headers)
    ).json()

    updated = await client.patch(
        f"/api/v1/restaurants/{created['id']}",
        json={"name": "New", "is_active": False},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "New"
    assert updated.json()["is_active"] is False
    # The slug is the stable handle and deliberately does not follow the name.
    assert updated.json()["slug"] == "old"


async def test_another_organisations_restaurant_is_not_found(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    mine = (
        await client.post("/api/v1/restaurants", json={"name": "Mine"}, headers=headers)
    ).json()

    outsider = await bearer(client, "outsider@example.com")
    response = await client.patch(
        f"/api/v1/restaurants/{mine['id']}", json={"name": "Theirs"}, headers=outsider
    )
    assert response.status_code == 404, "existence must not be confirmed to a stranger"


# ── zones ────────────────────────────────────────────────────────────────────


async def test_a_zone_belongs_to_a_restaurant_in_the_callers_tenant(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    restaurant = (
        await client.post("/api/v1/restaurants", json={"name": "Site"}, headers=headers)
    ).json()

    created = await client.post(
        "/api/v1/zones",
        json={"restaurant_id": restaurant["id"], "name": "Prep line"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["name"] == "Prep line"

    listed = await client.get("/api/v1/zones", headers=headers)
    assert listed.json()["count"] == 1

    # And the parent now reports it.
    restaurants = await client.get("/api/v1/restaurants", headers=headers)
    assert restaurants.json()["restaurants"][0]["zone_count"] == 1


async def test_a_zone_cannot_be_attached_to_another_organisations_restaurant(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    mine = (
        await client.post("/api/v1/restaurants", json={"name": "Mine"}, headers=headers)
    ).json()

    outsider = await bearer(client, "outsider@example.com")
    response = await client.post(
        "/api/v1/zones",
        json={"restaurant_id": mine["id"], "name": "Trespass"},
        headers=outsider,
    )
    assert response.status_code == 404


async def test_zones_can_be_filtered_to_one_restaurant(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    first = (
        await client.post("/api/v1/restaurants", json={"name": "First"}, headers=headers)
    ).json()
    second = (
        await client.post("/api/v1/restaurants", json={"name": "Second"}, headers=headers)
    ).json()
    await client.post(
        "/api/v1/zones", json={"restaurant_id": first["id"], "name": "A"}, headers=headers
    )
    await client.post(
        "/api/v1/zones", json={"restaurant_id": second["id"], "name": "B"}, headers=headers
    )

    filtered = await client.get(
        "/api/v1/zones", params={"restaurant_id": first["id"]}, headers=headers
    )
    assert [z["name"] for z in filtered.json()["zones"]] == ["A"]


async def test_renaming_a_zone_is_audited(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    restaurant = (
        await client.post("/api/v1/restaurants", json={"name": "Site"}, headers=headers)
    ).json()
    zone = (
        await client.post(
            "/api/v1/zones",
            json={"restaurant_id": restaurant["id"], "name": "Old"},
            headers=headers,
        )
    ).json()

    renamed = await client.patch(
        f"/api/v1/zones/{zone['id']}", json={"name": "New"}, headers=headers
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "New"

    trail = await client.get("/api/v1/audit", headers=headers)
    assert "zone.updated" in [e["action"] for e in trail.json()["events"]]


# ── users ────────────────────────────────────────────────────────────────────


async def test_the_user_list_names_roles_and_never_a_credential(
    client: AsyncClient, admin
) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.get("/api/v1/users", headers=headers)

    assert response.status_code == 200
    body = response.json()
    listed = {u["email"]: u for u in body["users"]}

    assert "manager@example.com" in listed
    assert listed["manager@example.com"]["roles"] == ["restaurant_manager"]
    # The organisation boundary holds here as everywhere else.
    assert "outsider@example.com" not in listed

    # Checked against the user records rather than the whole payload: the
    # `write_unavailable_reason` legitimately contains the word "password",
    # and an assertion over the raw text would fail on the explanation while
    # proving nothing about the records it is supposed to be guarding.
    allowed = {
        "id",
        "email",
        "display_name",
        "is_active",
        "roles",
        "created_at",
        "last_login_at",
    }
    for user in body["users"]:
        assert set(user) == allowed, f"unexpected field on a user record: {set(user) - allowed}"


async def test_the_user_list_states_that_it_cannot_create_accounts(
    client: AsyncClient, admin
) -> None:
    """The capability travels with the payload, so one place decides it."""
    headers = await bearer(client, "admin@example.com")
    body = (await client.get("/api/v1/users", headers=headers)).json()

    assert body["write_available"] is False
    assert body["write_unavailable_reason"]


async def test_a_supervisor_may_not_read_the_user_list(client: AsyncClient, seeded) -> None:
    """`kitchen_supervisor` holds no `VIEW_USERS` — most likely a shared screen."""
    headers = await bearer(client, "supervisor@example.com")
    response = await client.get("/api/v1/users", headers=headers)
    assert response.status_code == 403
