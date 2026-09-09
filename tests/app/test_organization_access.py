"""The access flow between login and the organization application.

Three journeys and five refusals. Every test here crosses the seam that the
membership table introduced — a token names an organization, and the server
decides on every request whether that naming was allowed.

The property under test throughout, stated once:

    The organization is chosen by the *server*, carried by the *token*, and
    re-checked against the *database* on every request. Nothing a client sends
    contributes to it.

If that is not true, none of the rest matters: the whole multi-tenant boundary
of this application is `AccessDecision.tenant_id`, and this change is the first
time that value has been allowed to be more than one thing per account.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.authorization.model import Permission
from app.authorization.platform import OPERATOR_ENTRY_PERMISSIONS
from app.domain.audit import AuditAction
from app.domain.models import AuditEvent, Restaurant
from app.users.models import Organization, OrganizationMembership, PlatformOperatorGrant
from tests.app.conftest import admit, bearer, login, make_user

pytestmark = pytest.mark.asyncio

AUTH = "/api/v1/auth"
PLATFORM = "/api/v1/platform"


@pytest_asyncio.fixture
async def estate(app):
    """Three accounts that between them cover every journey.

    * `solo@` belongs to one organization. The single-organization
      administrator whose experience must not change at all.
    * `both@` belongs to two, with a *different* role in each — which is the
      fact the per-organization role rows exist to carry.
    * `operator@` belongs to one and administers the platform.
    """
    database = app.state.database
    async with database.session_scope() as session:
        acme, solo = make_user(
            org_id="org-acme",
            email="solo@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        acme.name = "Acme Catering"
        session.add(acme)
        session.add(solo)

        borden = Organization(id="org-borden", name="Borden Foods", slug="borden")
        session.add(borden)

        _, both = make_user(
            org_id="org-acme",
            email="both@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        # An administrator at Acme and only an auditor at Borden. Being trusted
        # by one customer is not being trusted by the other.
        admit(both, "org-borden", roles=("auditor",))
        session.add(both)

        _, operator = make_user(
            org_id="org-acme",
            email="operator@example.com",
            roles=("developer",),
            camera_breadth="none",
            camera_ids="",
        )
        session.add(operator)
        await session.flush()
        session.add(
            PlatformOperatorGrant(user_id=operator.id, reason="test fixture", granted_by="test")
        )

    return app


# ── Scenario 1: one organization, no chooser ─────────────────────────────────


async def test_a_single_organization_user_is_never_asked_to_choose(estate, client: AsyncClient):
    """The requirement stated as a refusal to add a step.

    The whole point of asking the server rather than counting a list on the
    client is that this answer is authoritative. A single-organization
    administrator signs in and is in their organization.
    """
    response = await login(client, "solo@example.com")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["must_select"] is False
    assert body["is_platform_operator"] is False
    assert [organization["id"] for organization in body["organizations"]] == ["org-acme"]
    assert body["user"]["tenant_id"] == "org-acme"
    # And they arrive with the role they hold *there*, not with a placeholder.
    assert "org_admin" in body["user"]["roles"]


async def test_a_single_organization_user_reaches_their_application_immediately(
    estate, client: AsyncClient
):
    headers = await bearer(client, "solo@example.com")
    assert (await client.get("/api/v1/restaurants", headers=headers)).status_code == 200


# ── Scenario 2: several organizations, and only the ones they hold ───────────


async def test_a_multi_organization_user_is_owed_a_choice(estate, client: AsyncClient):
    response = await login(client, "both@example.com")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["must_select"] is True
    # Their home organization, so the session is usable before they choose —
    # but the frontend is told a choice is owed and sends them to the chooser.
    assert body["user"]["tenant_id"] == "org-acme"


async def test_the_list_contains_only_organizations_the_account_belongs_to(
    estate, client: AsyncClient
):
    """`org-test` and every other organization in the database are absent, and
    that is the whole of the requirement: a member must never be shown a
    customer they have no membership in."""
    headers = await bearer(client, "both@example.com")
    listed = await client.get(f"{AUTH}/organizations", headers=headers)
    assert listed.status_code == 200

    body = listed.json()
    assert {organization["id"] for organization in body["organizations"]} == {
        "org-acme",
        "org-borden",
    }
    assert body["active"] == "org-acme"
    assert body["acting_as"] == ""
    assert body["is_platform_operator"] is False

    # Enough to tell one customer from another, and nothing that belongs to the
    # Command Center behind it.
    for organization in body["organizations"]:
        assert set(organization) == {
            "id",
            "name",
            "slug",
            "status",
            "site_count",
            "camera_count",
        }


async def test_selecting_an_organization_moves_the_session_into_it(estate, client: AsyncClient):
    """The token's tenant becomes the selected organization, and the roles
    become the ones held *there* — administrator at Acme, auditor at Borden."""
    headers = await bearer(client, "both@example.com")

    selected = await client.post(f"{AUTH}/organizations/org-borden/select", headers=headers)
    assert selected.status_code == 200, selected.text
    body = selected.json()

    assert body["user"]["tenant_id"] == "org-borden"
    assert body["user"]["roles"] == ["auditor"]
    assert "org_admin" not in body["user"]["roles"]

    # And the new token actually works against the new organization.
    moved = {"Authorization": f"Bearer {body['access_token']}"}
    assert (await client.get(f"{AUTH}/me", headers=moved)).json()["tenant_id"] == "org-borden"


async def test_a_role_in_one_organization_confers_nothing_in_another(estate, client: AsyncClient):
    """`org_admin` at Acme, `auditor` at Borden, and the write refused there.

    This is the test that would have failed with a single `role_assignments`
    row per user: a membership would have carried every role across with it,
    and the second organization would have inherited an administrator nobody
    appointed.
    """
    headers = await bearer(client, "both@example.com")
    selected = await client.post(f"{AUTH}/organizations/org-borden/select", headers=headers)
    borden = {"Authorization": f"Bearer {selected.json()['access_token']}"}

    created = await client.post("/api/v1/restaurants", json={"name": "New site"}, headers=borden)
    assert created.status_code == 403, created.text


# ── The refusals ─────────────────────────────────────────────────────────────


async def test_selecting_an_organization_without_membership_is_refused(estate, client: AsyncClient):
    headers = await bearer(client, "solo@example.com")
    refused = await client.post(f"{AUTH}/organizations/org-borden/select", headers=headers)
    assert refused.status_code == 403, refused.text


async def test_an_unknown_organization_is_refused_the_same_way_as_a_forbidden_one(
    estate, client: AsyncClient
):
    """Identical refusals, deliberately. A member of one customer must not be
    able to enumerate the others by watching which ids answer differently."""
    headers = await bearer(client, "solo@example.com")
    forbidden = await client.post(f"{AUTH}/organizations/org-borden/select", headers=headers)
    unknown = await client.post(f"{AUTH}/organizations/org-nope/select", headers=headers)

    assert forbidden.status_code == unknown.status_code == 403
    assert forbidden.json()["code"] == unknown.json()["code"] == "OUT_OF_SCOPE"
    # The same sentence, too. A different message would be the oracle a matching
    # status code was supposed to close.
    assert forbidden.json()["message"] == unknown.json()["message"]


async def test_a_token_for_one_organization_cannot_reach_another(estate, client: AsyncClient):
    """URL and id tampering, from the only angle that could work.

    There is no organization parameter on `/api/v1/restaurants` to tamper with —
    the tenant comes from the token — so the strongest available attack is to
    hold a legitimate token for one organization and try it against another's
    data. It reaches the first organization's data and nothing else, because
    that is the only organization the token names.
    """
    headers = await bearer(client, "both@example.com")

    async with estate.state.database.session_scope() as session:
        session.add(
            Restaurant(
                id="rest-borden",
                organization_id="org-borden",
                name="Borden Kitchen",
                slug="borden-kitchen",
            )
        )

    listed = await client.get("/api/v1/restaurants", headers=headers)
    assert listed.status_code == 200
    names = [restaurant["name"] for restaurant in listed.json()["restaurants"]]
    assert "Borden Kitchen" not in names


async def test_a_revoked_membership_ends_the_session_it_authorised(estate, client: AsyncClient):
    """The reason membership is checked per request rather than at issue time.

    A token minted while the membership existed keeps naming that organization
    after the row is gone. If the naming were trusted, revoking somebody's
    access would take up to fifteen minutes to bite; because it is checked, it
    bites on the next call.
    """
    headers = await bearer(client, "both@example.com")
    selected = await client.post(f"{AUTH}/organizations/org-borden/select", headers=headers)
    borden = {"Authorization": f"Bearer {selected.json()['access_token']}"}

    assert (await client.get(f"{AUTH}/me", headers=borden)).status_code == 200

    async with estate.state.database.session_scope() as session:
        membership = (
            await session.execute(
                select(OrganizationMembership).where(
                    OrganizationMembership.organization_id == "org-borden"
                )
            )
        ).scalar_one()
        await session.delete(membership)

    refused = await client.get(f"{AUTH}/me", headers=borden)
    assert refused.status_code == 401, refused.text


# ── Scenario 3: the platform operator ────────────────────────────────────────


async def test_an_operator_is_owed_the_chooser_even_with_one_membership(
    estate, client: AsyncClient
):
    """`must_select` is not `len(organizations) > 1`.

    An operator belongs to one organization — their own — and the customers
    they administer are not among their memberships. Deriving the routing
    decision from the length of the membership list would send them straight
    into their home tenant, which is the one place their job is not.
    """
    body = (await login(client, "operator@example.com")).json()
    assert body["organizations"] == [
        {
            "id": "org-acme",
            "name": body["organizations"][0]["name"],
            "slug": body["organizations"][0]["slug"],
            "status": "active",
            "site_count": 0,
            "camera_count": 0,
        }
    ]
    assert body["is_platform_operator"] is True
    assert body["must_select"] is True


async def test_entering_an_organization_grants_a_read_only_session(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")

    entered = await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)
    assert entered.status_code == 200, entered.text
    body = entered.json()
    assert body["acting_as"] == "platform_operator"
    assert body["read_only"] is True
    assert body["organization"]["id"] == "org-borden"

    inside = {"Authorization": f"Bearer {body['access_token']}"}
    identity = (await client.get(f"{AUTH}/me", headers=inside)).json()
    assert identity["tenant_id"] == "org-borden"
    assert identity["acting_as"] == "platform_operator"
    # No roles at all. The reach is stated, never resolved from rows the
    # account does not have in an organization it is not a member of.
    assert identity["roles"] == []

    # The read works.
    assert (await client.get("/api/v1/restaurants", headers=inside)).status_code == 200


async def test_an_entered_operator_may_not_write(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    entered = await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)
    inside = {"Authorization": f"Bearer {entered.json()['access_token']}"}

    refused = await client.post("/api/v1/restaurants", json={"name": "Nope"}, headers=inside)
    assert refused.status_code == 403, refused.text


def test_the_operator_read_set_excludes_the_sensitive_reads():
    """A unit assertion, because this set is the whole of the policy.

    Stated as a list rather than derived from a `view_` prefix, and asserted as
    a list here for the same reason: the exclusions are decisions, and a rule
    that happened to produce them today would silently stop excluding a
    permission added tomorrow.
    """
    for excluded in (
        Permission.VIEW_EVIDENCE,
        Permission.VIEW_PATRON_ID,
        Permission.VIEW_AUDIT,
        Permission.EXPORT_REPORTS,
        Permission.REGISTER_DEMAND,
        Permission.ACCESS_DEVTOOLS,
    ):
        assert excluded not in OPERATOR_ENTRY_PERMISSIONS

    # And no write of any kind.
    for permission in OPERATOR_ENTRY_PERMISSIONS:
        assert not permission.value.startswith("manage_")
        assert permission not in {
            Permission.RETIRE_CAMERAS,
            Permission.DELETE_EVIDENCE,
            Permission.ACKNOWLEDGE_INCIDENTS,
            Permission.RESOLVE_INCIDENTS,
        }


async def test_entering_an_organization_is_audited_against_that_organization(
    estate, client: AsyncClient
):
    """Filed against the customer, not against the operator's home tenant.

    A cross-customer read that left its only trace in the operator's own
    organization would be invisible to the customer it was about, which is the
    one party with a reason to look.
    """
    headers = await bearer(client, "operator@example.com")
    await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)

    async with estate.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.organization_id, AuditEvent.actor, AuditEvent.detail).where(
                    AuditEvent.action == AuditAction.PLATFORM_OPERATOR_ENTERED.value
                )
            )
        ).all()

    assert len(rows) == 1
    organization_id, actor, raw_detail = rows[0]
    assert organization_id == "org-borden"
    assert actor == "operator@example.com"

    # `detail` is stored as JSON text, so the trail keeps what was written
    # rather than what a later version of the code would produce.
    detail = json.loads(raw_detail) if isinstance(raw_detail, str) else raw_detail
    assert detail["read_only"] is True
    assert "view_evidence" not in detail["permissions"]
    assert "view_incidents" in detail["permissions"]


async def test_an_archived_organization_cannot_be_entered(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    archived = await client.put(
        f"{PLATFORM}/organizations/org-borden/status",
        json={"status": "archived", "reason": "Contract ended 2026-08-31."},
        headers=headers,
    )
    assert archived.status_code == 200, archived.text

    refused = await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)
    assert refused.status_code == 422, refused.text


async def test_revoking_the_grant_ends_an_entry_session(estate, client: AsyncClient):
    """The same "re-read it every request" rule the membership check follows.

    An entry token names an organization the account is deliberately not a
    member of, so the membership table cannot vouch for it. The operator grant
    does — and it is read from the database on every request rather than
    trusted from the claim, so taking the grant away ends every entry session
    in flight rather than the ones minted afterwards.
    """
    headers = await bearer(client, "operator@example.com")
    entered = await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)
    inside = {"Authorization": f"Bearer {entered.json()['access_token']}"}

    assert (await client.get("/api/v1/restaurants", headers=inside)).status_code == 200

    async with estate.state.database.session_scope() as session:
        grant = (await session.execute(select(PlatformOperatorGrant))).scalar_one()
        await session.delete(grant)

    refused = await client.get("/api/v1/restaurants", headers=inside)
    assert refused.status_code == 401, refused.text


async def test_a_tenant_role_still_cannot_enter_an_organization(estate, client: AsyncClient):
    """The boundary, asserted from the wrong side, now that a door exists.

    `super_admin` is the most powerful tenant role there is. It does not reach
    the entry endpoint, because that endpoint does not read permissions — it
    requires a principal no role can produce.
    """
    async with estate.state.database.session_scope() as session:
        _, superuser = make_user(
            org_id="org-acme",
            email="super@example.com",
            roles=("super_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(superuser)

    headers = await bearer(client, "super@example.com")
    refused = await client.post(f"{PLATFORM}/organizations/org-borden/enter", headers=headers)
    assert refused.status_code == 403, refused.text


async def test_a_user_row_alone_does_not_admit_anyone(estate, client: AsyncClient):
    """Membership is the entry ticket, and `User.organization_id` is not one.

    The account below is filed in `org-borden` and has a role there, and has no
    membership row. Under the previous schema that was a working login. It is
    now refused, which is what "nothing else alone grants organization entry"
    means in practice.
    """
    async with estate.state.database.session_scope() as session:
        _, stranded = make_user(
            org_id="org-borden",
            email="stranded@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        stranded.memberships = []
        session.add(stranded)

    refused = await login(client, "stranded@example.com")
    assert refused.status_code == 401, refused.text
