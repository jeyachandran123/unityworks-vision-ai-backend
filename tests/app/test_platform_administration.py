"""The platform control plane — what it administers, and what it must not reach.

Two families of test here, and the second matters more than the first.

The first is ordinary: the console's reads return the right numbers and its
writes change the right rows.

The second is the boundary. This module added eight cross-organization
endpoints to a product whose entire multi-tenant guarantee is that a principal
names exactly one tenant. Every one of them is a place that guarantee could be
lost, so each is tested from the wrong side as well as the right one: a tenant
`super_admin` — the most powerful role there is — must be refused by all of
them, and none of them may return a customer's operational content.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.audit import AuditAction
from app.domain.models import AuditEvent, Restaurant
from app.users.models import Organization, OrganizationMembership, PlatformOperatorGrant
from tests.app.conftest import admit, bearer, make_user

pytestmark = pytest.mark.asyncio

PLATFORM = "/api/v1/platform"


@pytest_asyncio.fixture
async def estate(app):
    """Two customers, an operator, and one person who works for both."""
    database = app.state.database
    async with database.session_scope() as session:
        acme, operator = make_user(
            org_id="org-acme",
            email="operator@example.com",
            roles=("developer",),
            camera_breadth="none",
            camera_ids="",
        )
        session.add(acme)
        session.add(operator)

        session.add(Organization(id="org-borden", name="Borden Foods", slug="borden"))
        session.add(
            Restaurant(
                id="rest-acme-1",
                organization_id="org-acme",
                name="Acme Kitchen",
                slug="acme-kitchen",
            )
        )

        _, solo = make_user(
            org_id="org-acme",
            email="solo@example.com",
            roles=("restaurant_manager",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(solo)

        _, both = make_user(
            org_id="org-acme",
            email="both@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        admit(both, "org-borden", roles=("auditor",))
        session.add(both)

        await session.flush()
        session.add(
            PlatformOperatorGrant(user_id=operator.id, reason="test fixture", granted_by="test")
        )

    return app


@pytest_asyncio.fixture
async def tenant_super(estate, client: AsyncClient):
    """A tenant `super_admin` — every permission there is, and no platform reach."""
    async with estate.state.database.session_scope() as session:
        _, superuser = make_user(
            org_id="org-acme",
            email="super@example.com",
            roles=("super_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(superuser)
    return await bearer(client, "super@example.com")


# ── the boundary, tested from the wrong side ─────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/overview"),
        ("GET", "/people"),
        ("GET", "/people/user-solo@example.com"),
        ("GET", "/organizations/org-acme/members"),
        ("POST", "/organizations/org-acme/members"),
        ("DELETE", "/organizations/org-acme/members/user-solo@example.com"),
        ("GET", "/operators"),
        ("GET", "/roles"),
    ],
)
async def test_no_tenant_role_reaches_the_control_plane(
    tenant_super, client: AsyncClient, method: str, path: str
):
    """Every new endpoint, refused for the most powerful tenant role there is.

    Parametrised rather than written once per route on purpose: the failure mode
    this guards against is somebody adding a ninth endpoint and forgetting the
    dependency, and a table is the only shape of test that makes the omission
    visible when the list is updated.
    """
    response = await client.request(
        method, f"{PLATFORM}{path}", headers=tenant_super, json={"user_id": "x"}
    )
    assert response.status_code == 403, f"{method} {path} returned {response.status_code}"


async def test_the_console_never_returns_a_customers_operational_content(
    estate, client: AsyncClient
):
    """Counts, names, configuration and membership. Never an incident, an
    observation, a frame or an evidence record.

    An operator who needs those enters the organization explicitly, which is
    audited and read-only. If this console started returning them, that entry —
    and the record it leaves — would become a formality nobody had to use.
    """
    headers = await bearer(client, "operator@example.com")
    forbidden = {"incidents", "observations", "evidence", "frames", "cameras_list"}

    for path in ("/overview", "/people", "/organizations/org-acme/members", "/operators"):
        body = (await client.get(f"{PLATFORM}{path}", headers=headers)).json()
        assert not forbidden & set(body), f"{path} leaked operational content"


# ── overview ─────────────────────────────────────────────────────────────────


async def test_overview_counts_what_exists_and_scores_nothing(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/overview", headers=headers)).json()

    assert body["organizations"]["total"] == 2
    assert body["organizations"]["active"] == 2
    assert body["estate"]["sites"] == 1
    # operator@, solo@, both@ — the fixture's three.
    assert body["people"]["users"] == 3
    # `both@` is the only account with two memberships.
    assert body["people"]["multi_organization_users"] == 1
    assert body["people"]["platform_operators"] == 1

    # No fabricated health. If one of these ever appears, it was invented.
    assert "health_score" not in body
    assert "compliance_rate" not in body


async def test_overview_names_organizations_whose_setup_never_finished(estate, client: AsyncClient):
    """Borden has no sites and no cameras. Nothing else in the product says so."""
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/overview", headers=headers)).json()

    stalled = {o["id"] for o in body["attention"]["organizations_needing_setup"]}
    assert "org-borden" in stalled


# ── people ───────────────────────────────────────────────────────────────────


async def test_people_reports_home_organization_and_memberships_separately(
    estate, client: AsyncClient
):
    """The distinction the whole layer exists to express.

    `both@` lives in Acme and may enter both. Reporting one "organization" field
    would have to pick one of those facts and discard the other.
    """
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/people?q=both", headers=headers)).json()

    assert body["total"] == 1
    person = body["people"][0]
    assert person["home_organization_id"] == "org-acme"
    assert {m["organization_id"] for m in person["memberships"]} == {"org-acme", "org-borden"}
    assert person["organization_count"] == 2

    # And the roles are reported per organization, never merged.
    by_org = {m["organization_id"]: m["roles"] for m in person["memberships"]}
    assert by_org["org-acme"] == ["org_admin"]
    assert by_org["org-borden"] == ["auditor"]


async def test_people_can_be_filtered_to_one_organization(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    body = (
        await client.get(f"{PLATFORM}/people?organization_id=org-borden", headers=headers)
    ).json()
    assert {p["email"] for p in body["people"]} == {"both@example.com"}


# ── membership administration ────────────────────────────────────────────────


async def test_adding_a_member_writes_a_membership_and_nothing_else(estate, client: AsyncClient):
    """The entry ticket, and only the entry ticket.

    No role, no camera scope. The person may now sign in to Borden and, until
    somebody grants them a role there, see nothing — which is the correct
    default and the reason admitting and authorising are two acts.
    """
    headers = await bearer(client, "operator@example.com")

    added = await client.post(
        f"{PLATFORM}/organizations/org-borden/members",
        json={"user_id": "user-solo@example.com"},
        headers=headers,
    )
    assert added.status_code == 200, added.text

    person = added.json()
    assert {m["organization_id"] for m in person["memberships"]} == {"org-acme", "org-borden"}
    borden = next(m for m in person["memberships"] if m["organization_id"] == "org-borden")
    assert borden["roles"] == []
    assert borden["is_home"] is False

    async with estate.state.database.session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(OrganizationMembership).where(
                        OrganizationMembership.organization_id == "org-borden"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert {r.user_id for r in rows} == {"user-both@example.com", "user-solo@example.com"}
        granted = next(r for r in rows if r.user_id == "user-solo@example.com")
        assert granted.granted_by == "operator@example.com"


async def test_a_new_member_can_actually_sign_in_to_that_organization(estate, client: AsyncClient):
    """End to end, because the row existing is not the same as the door opening."""
    headers = await bearer(client, "operator@example.com")
    await client.post(
        f"{PLATFORM}/organizations/org-borden/members",
        json={"user_id": "user-solo@example.com"},
        headers=headers,
    )

    solo = await bearer(client, "solo@example.com")
    selected = await client.post("/api/v1/auth/organizations/org-borden/select", headers=solo)
    assert selected.status_code == 200, selected.text
    assert selected.json()["user"]["tenant_id"] == "org-borden"
    # Admitted, and holding nothing there yet.
    assert selected.json()["user"]["roles"] == []


async def test_removing_a_membership_closes_the_door_on_the_next_request(
    estate, client: AsyncClient
):
    """Not at the next login — the next request. Membership is re-read every
    time, so a token already naming the organization stops working at once."""
    headers = await bearer(client, "operator@example.com")
    both = await bearer(client, "both@example.com")

    selected = await client.post("/api/v1/auth/organizations/org-borden/select", headers=both)
    inside = {"Authorization": f"Bearer {selected.json()['access_token']}"}
    assert (await client.get("/api/v1/auth/me", headers=inside)).status_code == 200

    removed = await client.delete(
        f"{PLATFORM}/organizations/org-borden/members/user-both@example.com", headers=headers
    )
    assert removed.status_code == 200, removed.text
    assert removed.json()["left_without_access"] is False
    assert removed.json()["was_home_organization"] is False

    assert (await client.get("/api/v1/auth/me", headers=inside)).status_code == 401


async def test_removing_a_membership_leaves_the_roles_behind(estate, client: AsyncClient):
    """So that re-admitting somebody restores what they had, rather than
    restoring their access to nothing."""
    headers = await bearer(client, "operator@example.com")
    await client.delete(
        f"{PLATFORM}/organizations/org-borden/members/user-both@example.com", headers=headers
    )

    readmitted = await client.post(
        f"{PLATFORM}/organizations/org-borden/members",
        json={"user_id": "user-both@example.com"},
        headers=headers,
    )
    assert readmitted.status_code == 200
    borden = next(
        m for m in readmitted.json()["memberships"] if m["organization_id"] == "org-borden"
    )
    assert borden["roles"] == ["auditor"]


async def test_removing_the_last_membership_is_allowed_and_reported(estate, client: AsyncClient):
    """Offboarding looks exactly like this, so it is not refused — but the
    console has to be able to say the account can no longer sign in anywhere."""
    headers = await bearer(client, "operator@example.com")
    removed = await client.delete(
        f"{PLATFORM}/organizations/org-acme/members/user-solo@example.com", headers=headers
    )
    assert removed.status_code == 200, removed.text
    assert removed.json()["was_home_organization"] is True
    assert removed.json()["left_without_access"] is True

    # And the door really is shut.
    refused = await client.post(
        "/api/v1/auth/login",
        json={"email": "solo@example.com", "password": "correct-horse-battery"},
    )
    assert refused.status_code == 401


async def test_a_duplicate_membership_is_refused(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    again = await client.post(
        f"{PLATFORM}/organizations/org-acme/members",
        json={"user_id": "user-solo@example.com"},
        headers=headers,
    )
    assert again.status_code == 409, again.text


async def test_an_archived_organization_takes_no_new_members(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    await client.put(
        f"{PLATFORM}/organizations/org-borden/status",
        json={"status": "archived", "reason": "Contract ended."},
        headers=headers,
    )
    refused = await client.post(
        f"{PLATFORM}/organizations/org-borden/members",
        json={"user_id": "user-solo@example.com"},
        headers=headers,
    )
    assert refused.status_code == 422, refused.text


@pytest.mark.parametrize(
    ("action", "method", "path", "body"),
    [
        (
            AuditAction.ORGANIZATION_MEMBER_ADDED,
            "POST",
            "/organizations/org-borden/members",
            {"user_id": "user-solo@example.com"},
        ),
        (
            AuditAction.ORGANIZATION_MEMBER_REMOVED,
            "DELETE",
            "/organizations/org-borden/members/user-both@example.com",
            None,
        ),
    ],
)
async def test_membership_changes_are_audited_against_the_organization(
    estate, client: AsyncClient, action, method, path, body
):
    """Filed against the organization whose access changed, because that is the
    trail an access review of *that customer* reads."""
    headers = await bearer(client, "operator@example.com")
    response = await client.request(method, f"{PLATFORM}{path}", json=body, headers=headers)
    assert response.status_code == 200, response.text

    async with estate.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.organization_id, AuditEvent.actor, AuditEvent.detail).where(
                    AuditEvent.action == action.value
                )
            )
        ).all()

    assert len(rows) == 1
    organization_id, actor, raw = rows[0]
    assert organization_id == "org-borden"
    assert actor == "operator@example.com"
    detail = json.loads(raw) if isinstance(raw, str) else raw
    assert "email" in detail


# ── operators and role policy ────────────────────────────────────────────────


async def test_operators_are_visible_but_not_grantable_here(estate, client: AsyncClient):
    """Visibility is the half that belongs in a console. Granting stays a
    command-line act so the privilege cannot propagate itself."""
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/operators", headers=headers)).json()

    assert body["count"] == 1
    assert body["operators"][0]["email"] == "operator@example.com"
    assert body["operators"][0]["reason"] == "test fixture"
    assert body["grant_is_manageable_here"] is False

    # And there is genuinely no write route to find.
    assert (await client.post(f"{PLATFORM}/operators", json={}, headers=headers)).status_code == 405


async def test_role_policy_is_reported_as_read_only_because_it_is(estate, client: AsyncClient):
    """`ROLE_PERMISSIONS` is a Python constant. The payload says so, so that a
    console never offers an edit the server would not enforce."""
    headers = await bearer(client, "operator@example.com")
    body = (await client.get(f"{PLATFORM}/roles", headers=headers)).json()

    assert body["editable"] is False
    assert body["customization"]["mechanism"] == "permission_overrides"

    roles = {r["role"]: r for r in body["roles"]}
    assert "super_admin" in roles and "kitchen_supervisor" in roles
    # The one exclusion the model makes by hand, still made.
    assert "manage_patron_id" not in roles["super_admin"]["permissions"]
