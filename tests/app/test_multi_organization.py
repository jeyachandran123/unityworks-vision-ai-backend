"""The three pillars, asserted end to end.

This file exists because the previous eight stages each tested their own layer
and the product still did not work. Every test here crosses a boundary on
purpose: a permission composed with a route, a role composed with an override,
an organization's lifecycle composed with the runtime.

The requirement that drove all of it, stated once:

    Manager A may view and edit sites.
    Manager B may view sites and may not edit them.
    Manager C may not see them at all.

Same base role. No new roles. If that is not expressible, nothing else in this
file matters.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.authorization.model import Permission, Role, permissions_for
from app.authorization.resolver import SUSPENDED_FORBIDDEN
from app.domain.runtime_identity import runtime_camera_id, split_runtime_camera_id
from app.errors import ValidationError
from app.users.models import Organization, PlatformOperatorGrant
from tests.app.conftest import bearer, make_user

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def admin(seeded):
    """An `org_admin` inside org-test. The seeded set has no administrator."""
    async with seeded.state.database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


ADMIN_BASE = "/api/v1/admin/users"
PLATFORM = "/api/v1/platform"


# ── Pillar 3: read and manage, expressed per feature ─────────────────────────


class TestPermissionVocabulary:
    """The vocabulary correction, checked at the level it actually matters."""

    def test_sites_and_zones_have_independent_read_and_manage(self):
        """The whole requirement, reduced to its smallest statement.

        Before this existed, reading sites required `VIEW_USERS` and writing
        them required `MANAGE_ORGANIZATION` — so "may see who works here" and
        "may read every site" were one grant, and "may rename a zone" and "may
        reconfigure the organisation" were another.
        """
        for permission in (
            Permission.VIEW_SITES,
            Permission.MANAGE_SITES,
            Permission.VIEW_ZONES,
            Permission.MANAGE_ZONES,
        ):
            assert isinstance(permission, Permission)

        assert Permission.VIEW_SITES is not Permission.MANAGE_SITES
        assert Permission.VIEW_ZONES is not Permission.MANAGE_ZONES

    def test_restaurant_manager_keeps_the_reach_it_had_before_the_split(self):
        """The backward-compatibility trap, asserted.

        `RESTAURANT_MANAGER` holds `VIEW_USERS`, which used to gate site and
        zone *reads*. Splitting the vocabulary silently removes that unless the
        baseline is set deliberately, and a manager who could see their own
        sites yesterday must be able to see them today.
        """
        held = permissions_for(frozenset({Role.RESTAURANT_MANAGER}))

        assert Permission.VIEW_SITES in held
        assert Permission.VIEW_ZONES in held
        # And no more than that. Gaining write authority it never had would be
        # a privilege expansion hiding inside a compatibility fix.
        assert Permission.MANAGE_SITES not in held
        assert Permission.MANAGE_ZONES not in held

    def test_kitchen_supervisor_does_not_gain_site_visibility(self):
        """The other half of the trap. This role has no `VIEW_USERS`, so it
        could not read sites before, and must not be able to now."""
        held = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))

        assert Permission.VIEW_SITES not in held
        assert Permission.VIEW_ZONES not in held

    def test_org_admin_administers_the_estate(self):
        held = permissions_for(frozenset({Role.ORG_ADMIN}))
        for permission in (
            Permission.VIEW_SITES,
            Permission.MANAGE_SITES,
            Permission.VIEW_ZONES,
            Permission.MANAGE_ZONES,
            Permission.RETIRE_CAMERAS,
        ):
            assert permission in held, permission

    def test_retiring_a_camera_is_not_implied_by_managing_one(self):
        """Retirement destroys an observation partition. A role that may add
        and rename cameras has not been trusted with that."""
        manager = permissions_for(frozenset({Role.RESTAURANT_MANAGER}))
        assert Permission.RETIRE_CAMERAS not in manager

    def test_super_admin_picks_up_new_permissions_by_construction(self):
        """Its baseline is `frozenset(Permission)` minus one, so a permission
        added later is held automatically. That is correct and intended, and
        this asserts it did in fact happen rather than assuming it."""
        held = permissions_for(frozenset({Role.SUPER_ADMIN}))
        assert Permission.MANAGE_SITES in held
        assert Permission.RETIRE_CAMERAS in held


class TestSuspendedForbidden:
    def test_no_manage_permission_escapes_a_suspension(self):
        """This replaced a `manage_` prefix filter, and the risk of an explicit
        set is that somebody adds a permission and forgets it. Every `manage_*`
        must still be covered, so the replacement can never be *weaker* than
        the rule it replaced."""
        missed = {
            p for p in Permission if p.value.startswith("manage_")
        } - SUSPENDED_FORBIDDEN
        assert not missed, f"these manage permissions survive a suspension: {missed}"

    def test_the_prefix_rule_would_have_missed_these(self):
        """Why the set exists at all. `retire_cameras` is the important one:
        the single most destructive act in the product, sailing through a
        suspension because of how it was spelled."""
        assert Permission.RETIRE_CAMERAS in SUSPENDED_FORBIDDEN
        assert Permission.DELETE_EVIDENCE in SUSPENDED_FORBIDDEN

    def test_reads_and_the_incident_queue_survive(self):
        """Suspension is commercial. The incidents are real food-safety
        findings about a kitchen that is still operating, and refusing to let
        anyone close one would leave a genuine violation open because an
        invoice is late."""
        for permission in (
            Permission.VIEW_SITES,
            Permission.VIEW_CAMERAS,
            Permission.VIEW_INCIDENTS,
            Permission.ACKNOWLEDGE_INCIDENTS,
            Permission.RESOLVE_INCIDENTS,
        ):
            assert permission not in SUSPENDED_FORBIDDEN, permission


# ── Pillar 3, at the route: Manager A / B / C ────────────────────────────────


@pytest_asyncio.fixture
async def estate(admin, client: AsyncClient):
    """One organisation, one site, and three managers on one role."""
    headers = await bearer(client, "admin@example.com")

    site = await client.post(
        "/api/v1/restaurants",
        json={"name": "Adyar", "timezone": "Asia/Kolkata"},
        headers=headers,
    )
    assert site.status_code == 200, site.text

    created: dict[str, str] = {}
    for who in ("a", "b", "c"):
        response = await client.post(
            ADMIN_BASE,
            json={
                "email": f"manager-{who}@example.com",
                "roles": ["restaurant_manager"],
                "camera_scope": {"breadth": "all_in_tenant"},
                # The password `tests.app.conftest.login` uses, so these three
                # can log in and the requirement can be checked at the route
                # rather than only against `permissions_for`.
                "password": "correct-horse-battery",
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        created[who] = response.json()["id"]

    # Manager A is granted the write. Manager B is the role's natural state and
    # needs no override at all. Manager C has the read revoked.
    assert (
        await client.put(
            f"{ADMIN_BASE}/{created['a']}/permissions/manage_sites",
            json={"state": "grant"},
            headers=headers,
        )
    ).status_code == 200
    assert (
        await client.put(
            f"{ADMIN_BASE}/{created['c']}/permissions/view_sites",
            json={"state": "revoke"},
            headers=headers,
        )
    ).status_code == 200

    return admin, site.json()["id"], created


async def test_manager_a_may_read_and_edit_sites(estate, client: AsyncClient):
    _, site_id, _ = estate
    headers = await bearer(client, "manager-a@example.com")

    assert (await client.get("/api/v1/restaurants", headers=headers)).status_code == 200
    edited = await client.patch(
        f"/api/v1/restaurants/{site_id}", json={"name": "Adyar Main"}, headers=headers
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["name"] == "Adyar Main"


async def test_manager_b_may_read_sites_and_not_edit_them(estate, client: AsyncClient):
    """The case that could not be expressed before. Manager B holds the same
    role as Manager A and has **no overrides at all** — read-only is what the
    corrected baseline already means."""
    _, site_id, _ = estate
    headers = await bearer(client, "manager-b@example.com")

    assert (await client.get("/api/v1/restaurants", headers=headers)).status_code == 200
    refused = await client.patch(
        f"/api/v1/restaurants/{site_id}", json={"name": "Not Allowed"}, headers=headers
    )
    assert refused.status_code == 403


async def test_manager_c_may_not_see_sites_at_all(estate, client: AsyncClient):
    _, site_id, _ = estate
    headers = await bearer(client, "manager-c@example.com")

    assert (await client.get("/api/v1/restaurants", headers=headers)).status_code == 403
    assert (
        await client.patch(
            f"/api/v1/restaurants/{site_id}", json={"name": "No"}, headers=headers
        )
    ).status_code == 403


async def test_all_three_hold_exactly_one_role(estate, client: AsyncClient):
    """No role explosion. Three access profiles, one role, three override rows
    between them — which is the argument for the override engine existing."""
    _, _, created = estate
    headers = await bearer(client, "admin@example.com")

    for user_id in created.values():
        user = await client.get(f"{ADMIN_BASE}/{user_id}", headers=headers)
        assert user.json()["roles"] == ["restaurant_manager"]


async def test_revoke_still_wins_over_the_role(estate, client: AsyncClient):
    """Manager C's role grants `view_sites`. The REVOKE must beat it, or the
    override engine does not mean anything."""
    _, _, created = estate
    headers = await bearer(client, "admin@example.com")

    rows = await client.get(f"{ADMIN_BASE}/{created['c']}/permissions", headers=headers)
    view_sites = next(
        r for r in rows.json()["permissions"] if r["permission"] == "view_sites"
    )
    assert view_sites["role_grants"] is True
    assert view_sites["state"] == "revoke"
    assert view_sites["effective"] is False


# ── Pillar 1: two organizations, one camera key ──────────────────────────────


class TestRuntimeIdentity:
    def test_two_organizations_may_own_the_same_camera_key(self):
        a = runtime_camera_id("org-a", "cam-01")
        b = runtime_camera_id("org-b", "cam-01")
        assert a != b

    def test_the_identity_round_trips(self):
        for organization, key in (
            ("org-a", "cam-01"),
            ("org-unityworks", "kitchen_line_1"),
            ("org-b", "a-b-c_d"),
        ):
            assert split_runtime_camera_id(runtime_camera_id(organization, key)) == (
                organization,
                key,
            )

    def test_an_organization_id_may_not_contain_an_underscore(self):
        """Load-bearing, not stylistic. `FileObservationLog` rewrites the
        separator to `_` in the partition filename, so an organization id
        containing one would reintroduce the collision one layer down: `a_` +
        `b` and `a` + `_b` would land in the same file."""
        with pytest.raises(ValidationError):
            runtime_camera_id("org_a", "cam-01")

    def test_a_bare_camera_key_is_not_a_runtime_id(self):
        """Refused rather than guessed at. Guessing its tenant is exactly the
        failure this identity exists to close."""
        with pytest.raises(ValidationError):
            split_runtime_camera_id("cam-01")

    def test_a_camera_key_may_not_contain_the_separator(self):
        with pytest.raises(ValidationError):
            runtime_camera_id("org-a", "cam:01")


# ── Pillar 1, at the route: the platform operator boundary ───────────────────


@pytest_asyncio.fixture
async def operator(admin, client: AsyncClient):
    """An account that is a platform operator, granted the only way there is."""
    async with admin.state.database.session_scope() as session:
        _, user = make_user(
            email="operator@example.com",
            roles=("org_admin",),
            camera_breadth="none",
            camera_ids="",
        )
        session.add(user)
        await session.flush()
        session.add(
            PlatformOperatorGrant(user_id=user.id, reason="test fixture", granted_by="test")
        )
    return admin


async def test_a_super_admin_is_not_a_platform_operator(client: AsyncClient, admin):
    """The boundary, asserted from the wrong side.

    `super_admin` is the most powerful *tenant* role and holds every permission
    there is. None of them reach the platform surface, because the platform
    surface does not read permissions — it requires a different principal type
    that no role can produce.
    """
    async with admin.state.database.session_scope() as session:
        _, user = make_user(
            email="tenant-super@example.com",
            roles=("super_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)

    headers = await bearer(client, "tenant-super@example.com")
    assert (await client.get(f"{PLATFORM}/organizations", headers=headers)).status_code == 403
    assert (
        await client.post(
            f"{PLATFORM}/organizations", json={"name": "Sneaky"}, headers=headers
        )
    ).status_code == 403


async def test_an_operator_creates_and_lists_organizations(
    operator, client: AsyncClient
):
    headers = await bearer(client, "operator@example.com")

    created = await client.post(
        f"{PLATFORM}/organizations", json={"name": "Customer B"}, headers=headers
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["id"] == "org-customer-b"
    assert body["status"] == "active"

    listed = await client.get(f"{PLATFORM}/organizations", headers=headers)
    assert listed.status_code == 200
    assert "org-customer-b" in {o["id"] for o in listed.json()["organizations"]}


async def test_an_organization_id_that_would_break_runtime_identity_is_refused(
    operator, client: AsyncClient
):
    """The id becomes the tenant half of every camera's runtime identity, so
    the charset is enforced at the only place an id is ever minted."""
    headers = await bearer(client, "operator@example.com")
    response = await client.post(
        f"{PLATFORM}/organizations", json={"name": "!!!"}, headers=headers
    )
    assert response.status_code == 422


async def test_suspending_an_organization_requires_a_reason(
    operator, client: AsyncClient
):
    """It stops a paying customer's product working, and "why" is only
    reliably known at the moment of the act."""
    headers = await bearer(client, "operator@example.com")
    response = await client.put(
        f"{PLATFORM}/organizations/org-test/status",
        json={"status": "suspended"},
        headers=headers,
    )
    assert response.status_code == 422
    assert "reason" in response.text


async def test_suspension_reaches_authorization(operator, client: AsyncClient):
    """Not a hidden button. The permission is gone from the decision itself."""
    headers = await bearer(client, "operator@example.com")
    suspended = await client.put(
        f"{PLATFORM}/organizations/org-test/status",
        json={"status": "suspended", "reason": "Invoice 41 unpaid for 60 days."},
        headers=headers,
    )
    assert suspended.status_code == 200, suspended.text
    assert suspended.json()["status_reason"].startswith("Invoice 41")

    admin = await bearer(client, "admin@example.com")
    # Reads continue.
    assert (await client.get("/api/v1/restaurants", headers=admin)).status_code == 200
    # Writes do not.
    assert (
        await client.post("/api/v1/restaurants", json={"name": "New"}, headers=admin)
    ).status_code == 403


async def test_archiving_an_organization_refuses_every_request(
    operator, client: AsyncClient
):
    """Stronger than suspension, and enforced at authentication rather than
    per permission: a token minted before the archival must stop working."""
    headers = await bearer(client, "operator@example.com")
    admin = await bearer(client, "admin@example.com")

    archived = await client.put(
        f"{PLATFORM}/organizations/org-test/status",
        json={"status": "archived", "reason": "Contract ended 2026-08-31."},
        headers=headers,
    )
    assert archived.status_code == 200, archived.text

    assert (await client.get("/api/v1/restaurants", headers=admin)).status_code == 401


async def test_an_operator_cannot_read_a_tenants_data(operator, client: AsyncClient):
    """An operator manages the existence and lifecycle of organizations. Being
    able to create a customer is not being able to watch their kitchen, and the
    `PlatformOperator` type carries nothing that could be mistaken for tenant
    authority."""
    headers = await bearer(client, "operator@example.com")
    # The operator's own account holds `org_admin` in org-test, so this route
    # answers for *that* tenant through the ordinary tenant door — never
    # through the operator one, and never for another organization.
    listed = await client.get(f"{PLATFORM}/organizations", headers=headers)
    assert listed.status_code == 200
    for organization in listed.json()["organizations"]:
        # Counts, not contents. There is no route here that returns another
        # organization's incidents, evidence, users or camera list.
        assert set(organization) >= {"site_count", "camera_count", "user_count"}
        assert "users" not in organization
        assert "cameras" not in organization


# ── Pillar 2: camera placement cannot cross a boundary ───────────────────────


async def test_a_camera_cannot_be_placed_in_another_sites_zone(
    client: AsyncClient, admin
):
    """The gap that let `zone_id` through unread.

    The route validated the restaurant and then passed `zone_id` from the
    request body straight to the domain service, so a camera could be attached
    to any zone in the database — and every observation it produced would be
    attributed there.
    """
    headers = await bearer(client, "admin@example.com")

    first = await client.post(
        "/api/v1/restaurants", json={"name": "Site One"}, headers=headers
    )
    second = await client.post(
        "/api/v1/restaurants", json={"name": "Site Two"}, headers=headers
    )
    assert first.status_code == 200 and second.status_code == 200

    elsewhere = await client.post(
        "/api/v1/zones",
        json={"restaurant_id": second.json()["id"], "name": "Somebody Else's Kitchen"},
        headers=headers,
    )
    assert elsewhere.status_code == 200, elsewhere.text

    refused = await client.post(
        "/api/v1/cameras",
        json={
            "restaurant_id": first.json()["id"],
            "zone_id": elsewhere.json()["id"],
            "camera_key": "cam-90",
            "name": "Misplaced",
            "channel": 90,
            "host": "10.0.0.5",
        },
        headers=headers,
    )
    assert refused.status_code == 422
    assert "zone" in refused.text.lower()


async def test_a_camera_cannot_be_created_without_an_address(
    client: AsyncClient, admin
):
    """A camera with no host is skipped by the runtime, so it would be created,
    listed, and permanently inert with nothing ever saying why."""
    headers = await bearer(client, "admin@example.com")
    site = await client.post(
        "/api/v1/restaurants", json={"name": "Hostless"}, headers=headers
    )

    response = await client.post(
        "/api/v1/cameras",
        json={
            "restaurant_id": site.json()["id"],
            "camera_key": "cam-91",
            "name": "No Address",
            "channel": 91,
        },
        headers=headers,
    )
    assert response.status_code == 422
    assert "host" in response.text


async def test_a_camera_never_returns_its_credential_reference(
    client: AsyncClient, admin
):
    """`literal:` made this field a channel that could carry the secret itself.
    The scheme is useful to an administrator; the reference is not, and it
    names something an attacker who reaches the process can go and read."""
    headers = await bearer(client, "admin@example.com")
    site = await client.post(
        "/api/v1/restaurants", json={"name": "Credentialed"}, headers=headers
    )

    created = await client.post(
        "/api/v1/cameras",
        json={
            "restaurant_id": site.json()["id"],
            "camera_key": "cam-92",
            "name": "Watched",
            "channel": 92,
            "host": "10.0.0.5",
            "credential_ref": "env:CCTV_PASSWORD",
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    body = created.json()

    assert "credential_ref" not in body
    assert body["credential_configured"] is True
    assert body["credential_scheme"] == "env"
    assert "CCTV_PASSWORD" not in created.text


async def test_a_literal_credential_can_no_longer_be_written(
    client: AsyncClient, admin
):
    headers = await bearer(client, "admin@example.com")
    site = await client.post(
        "/api/v1/restaurants", json={"name": "Literal"}, headers=headers
    )

    response = await client.post(
        "/api/v1/cameras",
        json={
            "restaurant_id": site.json()["id"],
            "camera_key": "cam-93",
            "name": "Plaintext",
            "channel": 93,
            "host": "10.0.0.5",
            "credential_ref": "literal:hunter2",
        },
        headers=headers,
    )
    assert response.status_code == 422
    assert "hunter2" not in response.text, "the refusal echoed the secret back"


async def test_a_connection_test_refuses_a_loopback_address(
    client: AsyncClient, admin
):
    """The test would otherwise be a port scanner pointed at this server."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        "/api/v1/cameras/test-connection",
        json={"host": "127.0.0.1", "rtsp_port": 554},
        headers=headers,
    )
    assert response.status_code == 422
    assert "loopback" in response.text.lower()


async def test_a_connection_test_refuses_link_local(client: AsyncClient, admin):
    """169.254.169.254 is where cloud instance metadata lives."""
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        "/api/v1/cameras/test-connection",
        json={"host": "169.254.169.254", "rtsp_port": 80},
        headers=headers,
    )
    assert response.status_code == 422


# ── Cross-tenant reads ───────────────────────────────────────────────────────


async def test_another_organizations_site_is_not_found(client: AsyncClient, admin):
    """404, not 403. Confirming a row exists but belongs to someone else is
    itself a disclosure."""
    headers = await bearer(client, "admin@example.com")
    outsider = await bearer(client, "outsider@example.com")

    site = await client.post(
        "/api/v1/restaurants", json={"name": "Ours"}, headers=headers
    )
    assert site.status_code == 200

    response = await client.patch(
        f"/api/v1/restaurants/{site.json()['id']}",
        json={"name": "Theirs"},
        headers=outsider,
    )
    assert response.status_code == 404


async def test_a_camera_scope_cannot_name_another_organizations_camera(
    client: AsyncClient, admin
):
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        ADMIN_BASE,
        json={
            "email": "scoped@example.com",
            "roles": [],
            "camera_scope": {"breadth": "listed", "camera_keys": ["cam-elsewhere"]},
            "password": "a-perfectly-fine-password",
        },
        headers=headers,
    )
    assert response.status_code == 422
