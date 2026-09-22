"""The Platform Admin's session: signed in, and inside no organization.

The Platform Admin belongs to no organization. Every other session in the
application names one, so his is a different shape: a token carrying
`act: platform_operator` and no tenant at all. It reaches the platform console
and the doors into organizations, and nothing tenant-scoped until he goes
through one of those doors.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.auth.passwords import hash_password
from app.domain.audit import PLATFORM_AUDIT_SCOPE, AuditAction
from app.domain.models import AuditEvent
from app.users.models import Organization, PlatformOperatorGrant, User
from tests.app.conftest import bearer, login, make_user

pytestmark = pytest.mark.asyncio

AUTH = "/api/v1/auth"
PLATFORM = "/api/v1/platform"


@pytest_asyncio.fixture
async def platform(app):
    """Two organizations and a Platform Admin who belongs to neither."""
    async with app.state.database.session_scope() as session:
        gayathri, admin = make_user(
            org_id="org-gayathri",
            email="admin@gayathri.example",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        gayathri.name = "Gayathri Restaurant"
        session.add(gayathri)
        session.add(admin)
        session.add(Organization(id="org-borden", name="Borden Foods", slug="borden"))

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
    return app


async def test_the_platform_admin_signs_in_without_an_organization(platform, client: AsyncClient):
    response = await login(client, "platform@example.com")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["is_platform_operator"] is True
    assert body["must_select"] is True
    assert body["organizations"] == []
    assert body["user"]["tenant_id"] == ""
    assert body["user"]["acting_as"] == "platform_operator"
    assert body["user"]["roles"] == []
    assert body["user"]["permissions"] == []


async def test_a_platform_session_reaches_the_console(platform, client: AsyncClient):
    headers = await bearer(client, "platform@example.com")

    listed = await client.get(f"{PLATFORM}/organizations", headers=headers)
    assert listed.status_code == 200, listed.text
    assert {o["id"] for o in listed.json()["organizations"]} == {"org-gayathri", "org-borden"}

    identity = (await client.get(f"{AUTH}/me", headers=headers)).json()
    assert identity["tenant_id"] == ""
    assert identity["acting_as"] == "platform_operator"

    mine = (await client.get(f"{AUTH}/organizations", headers=headers)).json()
    assert mine["organizations"] == []
    assert mine["is_platform_operator"] is True


async def test_a_platform_session_reaches_no_tenant_route(platform, client: AsyncClient):
    headers = await bearer(client, "platform@example.com")
    refused = await client.get("/api/v1/restaurants", headers=headers)
    assert refused.status_code == 403, refused.text


async def test_a_platform_session_refreshes_as_a_platform_session(platform, client: AsyncClient):
    await login(client, "platform@example.com")
    refreshed = await client.post(f"{AUTH}/refresh")
    assert refreshed.status_code == 200, refreshed.text
    user = refreshed.json()["user"]
    assert user["tenant_id"] == ""
    assert user["acting_as"] == "platform_operator"


async def test_a_platform_session_enters_any_organization(platform, client: AsyncClient):
    headers = await bearer(client, "platform@example.com")
    for organization_id in ("org-gayathri", "org-borden"):
        entered = await client.post(
            f"{PLATFORM}/organizations/{organization_id}/enter", headers=headers
        )
        assert entered.status_code == 200, entered.text
        inside = {"Authorization": f"Bearer {entered.json()['access_token']}"}
        identity = (await client.get(f"{AUTH}/me", headers=inside)).json()
        assert identity["tenant_id"] == organization_id


async def test_revoking_the_grant_ends_a_platform_session(platform, client: AsyncClient):
    headers = await bearer(client, "platform@example.com")
    async with platform.state.database.session_scope() as session:
        grant = (await session.execute(select(PlatformOperatorGrant))).scalar_one()
        await session.delete(grant)

    refused = await client.get(f"{PLATFORM}/organizations", headers=headers)
    assert refused.status_code == 401, refused.text


async def test_an_account_with_no_organization_and_no_grant_cannot_sign_in(
    platform, client: AsyncClient
):
    async with platform.state.database.session_scope() as session:
        session.add(
            User(
                id="user-nowhere",
                organization_id=None,
                email="nowhere@example.com",
                display_name="Nowhere",
                password_hash=hash_password("correct-horse-battery"),
            )
        )
    refused = await login(client, "nowhere@example.com")
    assert refused.status_code == 401, refused.text


async def test_platform_sign_in_and_out_are_audited_under_the_platform_scope(
    platform, client: AsyncClient
):
    await login(client, "platform@example.com")
    await client.post(f"{AUTH}/logout")

    async with platform.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.action, AuditEvent.organization_id).where(
                    AuditEvent.actor == "platform@example.com"
                )
            )
        ).all()
    assert (AuditAction.LOGIN.value, PLATFORM_AUDIT_SCOPE) in rows
    assert (AuditAction.LOGOUT.value, PLATFORM_AUDIT_SCOPE) in rows


async def test_the_platform_admin_is_listed_as_belonging_to_no_organization(
    platform, client: AsyncClient
):
    headers = await bearer(client, "platform@example.com")
    people = (await client.get(f"{PLATFORM}/people", headers=headers)).json()
    owner = next(p for p in people["people"] if p["email"] == "platform@example.com")
    assert owner["home_organization_id"] is None
    assert owner["home_organization_name"] == ""
    assert owner["memberships"] == []
    assert owner["is_platform_operator"] is True
    assert owner["home_membership_missing"] is False

    counted = (await client.get(f"{PLATFORM}/organizations", headers=headers)).json()
    gayathri = next(o for o in counted["organizations"] if o["id"] == "org-gayathri")
    assert gayathri["user_count"] == 1  # its admin; the Platform Admin is nobody's


def test_the_platform_audit_scope_cannot_be_an_organization_id():
    """Organization ids are minted as `org-<slug>` and nowhere else."""
    assert not PLATFORM_AUDIT_SCOPE.startswith("org-")


def test_a_token_needs_a_tenant_unless_it_is_a_platform_session(settings):
    from app.auth.tokens import TokenService

    tokens = TokenService(settings)
    with pytest.raises(ValueError):
        tokens.issue_access(subject="someone@example.com", tenant_id="")

    token, _ = tokens.issue_access(
        subject="platform@example.com", tenant_id="", acting_as="platform_operator"
    )
    from app.auth.tokens import TokenType

    claims = tokens.verify(token, expect=TokenType.ACCESS)
    assert claims.is_platform_session
    assert claims.tenant_id == ""
