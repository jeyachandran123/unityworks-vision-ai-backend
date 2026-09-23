"""Restaurants, zones and the user list.

The interesting assertions are the boundaries rather than the CRUD:

* reading structure and changing it are different permissions
* tenancy comes from the session, never from the request body
* another organization's rows are invisible, and asking for one by id is a 404
  rather than a 403 — "it exists but is not yours" is itself a disclosure
* the user list carries no credential material, and says why it cannot write
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.audit import AuditAction
from app.domain.models import AuditEvent

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


# ── the estate ───────────────────────────────────────────────────────────────
#
# Zones replaced sites on 2026-09-23, and their own behaviour — listing,
# creating, renaming, deleting only while empty — is covered by
# `tests/app/test_zones.py`. What stays here is the split this module exists
# for: reading the estate and changing it are different permissions.


async def test_a_manager_may_read_structure_but_not_change_it(client: AsyncClient, admin) -> None:
    """`view_zones` without `manage_zones` is a routine state, not an error."""
    headers = await bearer(client, "manager@example.com")

    assert (await client.get("/api/v1/zones", headers=headers)).status_code == 200
    refused = await client.post("/api/v1/zones", json={"name": "Nope"}, headers=headers)
    assert refused.status_code == 403, refused.text


async def test_renaming_a_zone_is_audited(client: AsyncClient, admin) -> None:
    """A rename is a change to how every past reading is labelled on screen,
    so it is recorded with what it was before."""
    headers = await bearer(client, "admin@example.com")
    created = await client.post("/api/v1/zones", json={"name": "Prep"}, headers=headers)
    assert created.status_code == 200, created.text
    zone_id = created.json()["id"]

    renamed = await client.patch(
        f"/api/v1/zones/{zone_id}", json={"name": "Prep line"}, headers=headers
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Prep line"

    async with admin.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.action, AuditEvent.detail).where(
                    AuditEvent.resource_id == zone_id
                )
            )
        ).all()
    actions = [row[0] for row in rows]
    assert AuditAction.ZONE_CREATED.value in actions
    assert AuditAction.ZONE_UPDATED.value in actions
    assert "Prep" in str(rows[-1][1])


# ── users ────────────────────────────────────────────────────────────────────


async def test_the_user_list_names_roles_and_never_a_credential(client: AsyncClient, admin) -> None:
    headers = await bearer(client, "admin@example.com")
    response = await client.get("/api/v1/users", headers=headers)

    assert response.status_code == 200
    body = response.json()
    listed = {u["email"]: u for u in body["users"]}

    assert "manager@example.com" in listed
    assert listed["manager@example.com"]["roles"] == ["restaurant_manager"]
    # The organization boundary holds here as everywhere else.
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
