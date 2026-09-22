"""Who belongs to an organization is decided by membership, not by home.

User administration used to mean "users whose home organization is this one".
That stopped being true the moment one person could hold several
organizations, and it was never true of the Platform Admin, who has no home at
all. These tests hold the replacement rule: inside an organization, its people
are its members, and what the screen shows about each of them is what they hold
*there*.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.auth.passwords import hash_password
from app.domain.audit import AuditAction
from app.domain.models import AuditEvent
from app.users.models import AccessGrant, PermissionOverride, PlatformOperatorGrant, User
from tests.app.conftest import admit, bearer, login, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/admin/users"
AUTH = "/api/v1/auth"


@pytest_asyncio.fixture
async def estate(seeded):
    """`seeded`, plus an Org Admin who holds org-test *and* org-other, and a
    Platform Admin who belongs to no organization."""
    async with seeded.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        admit(admin, "org-other", roles=("org_admin",))
        session.add(admin)

        owner = User(
            id="user-platform",
            organization_id=None,
            email="platform@example.com",
            display_name="Platform Admin",
            password_hash=hash_password("correct-horse-battery"),
        )
        session.add(owner)
        await session.flush()
        session.add(PlatformOperatorGrant(user_id=owner.id, reason="test", granted_by="test"))
    return seeded


async def _inside(client: AsyncClient, email: str, organization_id: str) -> dict[str, str]:
    """A session for `email`, moved into `organization_id`."""
    headers = await bearer(client, email)
    moved = await client.post(f"{AUTH}/organizations/{organization_id}/select", headers=headers)
    assert moved.status_code == 200, moved.text
    return {"Authorization": f"Bearer {moved.json()['access_token']}"}


async def _entered(client: AsyncClient, organization_id: str) -> dict[str, str]:
    headers = await bearer(client, "platform@example.com")
    entered = await client.post(
        f"/api/v1/platform/organizations/{organization_id}/enter", headers=headers
    )
    assert entered.status_code == 200, entered.text
    return {"Authorization": f"Bearer {entered.json()['access_token']}"}


async def test_an_entered_platform_admin_can_create_a_user(estate, client: AsyncClient):
    inside = await _entered(client, "org-test")
    created = await client.post(
        BASE,
        json={
            "email": "cook@gayathri.example",
            "display_name": "Cook",
            "password": "a-long-enough-password",
            "roles": ["kitchen_supervisor"],
            "camera_scope": {"breadth": "all_in_tenant"},
        },
        headers=inside,
    )
    assert created.status_code == 200, created.text
    assert created.json()["roles"] == ["kitchen_supervisor"]

    signed_in = await login(client, "cook@gayathri.example", "a-long-enough-password")
    assert signed_in.status_code == 200, signed_in.text
    assert signed_in.json()["user"]["tenant_id"] == "org-test"


async def test_an_entered_platform_admin_can_change_a_members_access(estate, client: AsyncClient):
    inside = await _entered(client, "org-test")
    granted = await client.put(
        f"{BASE}/user-manager@example.com/permissions/manage_sites",
        json={"state": "grant"},
        headers=inside,
    )
    assert granted.status_code == 200, granted.text
    scoped = await client.put(
        f"{BASE}/user-manager@example.com/camera-scope",
        json={"camera_scope": {"breadth": "all_in_tenant"}},
        headers=inside,
    )
    assert scoped.status_code == 200, scoped.text


async def test_an_org_admin_administers_his_second_organization(estate, client: AsyncClient):
    inside = await _inside(client, "admin@example.com", "org-other")

    listed = (await client.get(BASE, headers=inside)).json()
    emails = {u["email"] for u in listed["users"]}
    assert "outsider@example.com" in emails
    assert "manager@example.com" not in emails  # org-test only

    renamed = await client.patch(
        f"{BASE}/user-outsider@example.com",
        json={"display_name": "Borden Lead"},
        headers=inside,
    )
    assert renamed.status_code == 200, renamed.text


async def test_a_member_from_another_home_appears_with_the_roles_held_here(
    estate, client: AsyncClient
):
    """admin@ lives in org-test and is also an Org Admin in org-other."""
    headers = await bearer(client, "outsider@example.com")
    listed = (await client.get(BASE, headers=headers)).json()
    admin = next(u for u in listed["users"] if u["email"] == "admin@example.com")
    assert admin["roles"] == ["org_admin"]
    assert admin["camera_scope"]["breadth"] == "all_in_tenant"


async def test_the_role_filter_reads_only_this_organizations_roles(estate, client: AsyncClient):
    """manager@ is a restaurant manager in org-test, not in org-other."""
    inside = await _inside(client, "admin@example.com", "org-other")
    listed = (await client.get(BASE, params={"role": "restaurant_manager"}, headers=inside)).json()
    assert listed["users"] == []


async def test_access_changed_in_the_second_organization_stays_there(estate, client: AsyncClient):
    inside = await _inside(client, "admin@example.com", "org-other")
    granted = await client.put(
        f"{BASE}/user-outsider@example.com/permissions/manage_sites",
        json={"state": "grant"},
        headers=inside,
    )
    assert granted.status_code == 200, granted.text
    scoped = await client.put(
        f"{BASE}/user-outsider@example.com/camera-scope",
        json={"camera_scope": {"breadth": "none"}},
        headers=inside,
    )
    assert scoped.status_code == 200, scoped.text

    async with estate.state.database.session_scope() as session:
        override = (
            await session.execute(
                select(PermissionOverride).where(
                    PermissionOverride.user_id == "user-outsider@example.com"
                )
            )
        ).scalar_one()
        grant = (
            await session.execute(
                select(AccessGrant).where(AccessGrant.user_id == "user-outsider@example.com")
            )
        ).scalar_one()
    assert override.organization_id == "org-other"
    assert grant.organization_id == "org-other"


async def test_a_non_member_cannot_be_administered(estate, client: AsyncClient):
    """manager@ is not a member of org-other, so from inside it they do not exist."""
    inside = await _inside(client, "admin@example.com", "org-other")
    refused = await client.patch(
        f"{BASE}/user-manager@example.com", json={"display_name": "x"}, headers=inside
    )
    assert refused.status_code == 404, refused.text


async def test_an_email_held_elsewhere_is_refused_without_saying_where(estate, client: AsyncClient):
    """Email is unique across the deployment now, so a collision is possible
    with an account at another customer. The refusal must not confirm which."""
    headers = await _inside(client, "admin@example.com", "org-test")
    refused = await client.post(
        BASE,
        json={
            "email": "outsider@example.com",
            "password": "a-long-enough-password",
            "roles": [],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert refused.status_code == 409, refused.text
    message = refused.text.lower()
    assert "org-other" not in message
    assert "platform admin" in message

    async with estate.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.outcome, AuditEvent.detail).where(
                    AuditEvent.action == AuditAction.USER_CREATED.value,
                    AuditEvent.organization_id == "org-test",
                )
            )
        ).all()
    assert rows, "the refusal was not audited"
    outcome, raw = rows[-1]
    detail = json.loads(raw) if isinstance(raw, str) else raw
    assert outcome == "denied"
    assert detail["held_in"] == ["org-other"]


async def test_an_email_already_here_says_so_plainly(estate, client: AsyncClient):
    headers = await _inside(client, "admin@example.com", "org-test")
    refused = await client.post(
        BASE,
        json={
            "email": "manager@example.com",
            "password": "a-long-enough-password",
            "roles": [],
            "camera_scope": {"breadth": "none"},
        },
        headers=headers,
    )
    assert refused.status_code == 409, refused.text
    assert "already" in refused.json()["message"].lower()
