"""The platform overview — the control plane's dashboard, and what it counts.

The overview grew from a row of totals into the page a Platform Admin opens
first: every customer on one line, who can still administer each one, whether
its cameras are actually live, who has signed in lately, and what the platform
did this fortnight. Every figure is still a count the server ran. These tests pin
each new count to the rows it is made of, and pin the two things the growth must
not do: return a customer's operational content, or let a configured camera pass
for a live one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.audit import AuditAction, AuditOutcome
from app.domain.models import AuditEvent, Zone
from app.users.models import Organization, PlatformOperatorGrant, User
from tests.app.conftest import admit, bearer, make_user

pytestmark = pytest.mark.asyncio

OVERVIEW = "/api/v1/platform/overview"


@pytest_asyncio.fixture
async def estate(app):
    """Two customers, an operator, and one person who works for both.

    Acme has an Organization Admin (`both@`); Borden has only an auditor — the
    same person, holding a lesser role there — so nobody can administer Borden.
    """
    database = app.state.database
    async with database.session_scope() as session:
        acme, operator = make_user(
            org_id="org-acme",
            email="operator@example.com",
            roles=("developer",),
            camera_breadth="none",
            camera_ids="",
        )
        acme.name = "Acme Catering"
        session.add(acme)
        session.add(operator)

        session.add(Organization(id="org-borden", name="Borden Foods", slug="borden"))
        session.add(Zone(id="zone-acme-1", organization_id="org-acme", name="Acme Kitchen"))

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


async def _read(client: AsyncClient, **params) -> dict:
    headers = await bearer(client, "operator@example.com")
    response = await client.get(OVERVIEW, headers=headers, params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def _audit(app, **row) -> None:
    async with app.state.database.session_scope() as session:
        session.add(AuditEvent(**row))


async def _set(app, email: str, **fields) -> None:
    async with app.state.database.session_scope() as session:
        user = (await session.execute(select(User).where(User.email == email))).scalar_one()
        for key, value in fields.items():
            setattr(user, key, value)


# ── every customer on one line ───────────────────────────────────────────────


async def test_each_organization_is_reported_with_its_own_counts(estate, client: AsyncClient):
    body = await _read(client)
    fleet = {row["id"]: row for row in body["fleet"]}

    assert set(fleet) == {"org-acme", "org-borden"}
    acme = fleet["org-acme"]
    assert acme["name"] == "Acme Catering"
    assert acme["status"] == "active"
    assert acme["zones"] == 1
    assert acme["cameras"] == 0
    # operator@, solo@ and both@ are all members of Acme.
    assert acme["members"] == 3
    assert acme["admins"] == 1

    borden = fleet["org-borden"]
    assert borden["members"] == 1
    # `both@` is Borden's auditor, not its administrator.
    assert borden["admins"] == 0


async def test_the_fleet_reports_when_somebody_last_signed_in_there(estate, client: AsyncClient):
    when = datetime(2026, 9, 20, 8, 30, tzinfo=UTC)
    await _set(estate, "both@example.com", last_login_at=when)

    body = await _read(client)
    fleet = {row["id"]: row for row in body["fleet"]}

    # Borden's only member is `both@`, so Borden's latest sign-in is theirs —
    # stated with its offset, so a browser in another zone reads it correctly.
    assert fleet["org-borden"]["last_sign_in_at"] == when.isoformat()
    # Acme's operator signed in a moment ago to read this page.
    assert fleet["org-acme"]["last_sign_in_at"] > when.isoformat()


async def test_an_organization_nobody_can_administer_is_named(estate, client: AsyncClient):
    body = await _read(client)

    orphaned = {row["id"] for row in body["attention"]["organizations_without_admin"]}
    assert orphaned == {"org-borden"}


async def test_a_deactivated_admin_administers_nothing(estate, client: AsyncClient):
    """A disabled account cannot sign in, so it cannot be anybody's way back in."""
    await _set(estate, "both@example.com", is_active=False)

    body = await _read(client)

    orphaned = {row["id"] for row in body["attention"]["organizations_without_admin"]}
    assert "org-acme" in orphaned
    assert {row["id"]: row for row in body["fleet"]}["org-acme"]["admins"] == 0


# ── live is read from the runtime, never from configuration ──────────────────


async def test_live_cameras_are_counted_by_state_not_by_registration(estate, client: AsyncClient):
    """A stream on the wall is not a live one. Only a frame makes it live.

    `cameras_running` has always counted what the wall *holds*, which includes
    a switched-off camera and one that will not authenticate. The breakdown is
    what tells those apart, and `cameras_live` is the one number that means
    pictures are arriving.
    """
    estate.state.wall = SimpleNamespace(
        streams={
            "org-acme:cam-1": SimpleNamespace(state="live"),
            "org-acme:cam-2": SimpleNamespace(state="live"),
            "org-acme:cam-3": SimpleNamespace(state="error"),
            "org-borden:cam-1": SimpleNamespace(state="disabled"),
            "org-borden:cam-2": SimpleNamespace(state="reconnecting"),
        }
    )

    body = await _read(client)

    assert body["estate"]["cameras_running"] == 5
    assert body["estate"]["cameras_live"] == 2
    assert body["estate"]["stream_states"] == {
        "live": 2,
        "error": 1,
        "disabled": 1,
        "reconnecting": 1,
    }
    fleet = {row["id"]: row for row in body["fleet"]}
    assert fleet["org-acme"]["cameras_live"] == 2
    assert fleet["org-borden"]["cameras_live"] == 0


async def test_no_runtime_means_nothing_is_live_rather_than_unknown_counts(
    estate, client: AsyncClient
):
    estate.state.wall = None

    body = await _read(client)

    assert body["estate"]["cameras_live"] == 0
    assert body["estate"]["stream_states"] == {}


# ── people, by when they last signed in ──────────────────────────────────────


async def test_sign_ins_are_counted_by_recency(estate, client: AsyncClient):
    now = datetime.now(UTC)
    await _set(estate, "solo@example.com", last_login_at=now - timedelta(hours=3))
    await _set(estate, "both@example.com", last_login_at=now - timedelta(days=12))

    body = await _read(client)
    people = body["people"]

    # The operator signed in to read this; solo@ three hours ago; both@ twelve
    # days ago. Windows are cumulative: a sign-in today is also one this week.
    assert people["signed_in_24h"] == 2
    assert people["signed_in_7d"] == 2
    assert people["signed_in_30d"] == 3
    assert people["never_signed_in"] == 0


async def test_disabled_accounts_and_accounts_that_can_enter_nowhere_are_counted(
    estate, client: AsyncClient
):
    await _set(estate, "solo@example.com", is_active=False)
    async with estate.state.database.session_scope() as session:
        _, stray = make_user(org_id="org-acme", email="stray@example.com")
        stray.memberships = []
        stray.role_assignments = []
        stray.access_grants = []
        session.add(stray)

    body = await _read(client)

    assert body["people"]["disabled_users"] == 1
    # The Platform Admin belongs to no organization by design and is not a
    # stray; stray@ is an account with a password and nowhere to use it.
    assert body["people"]["without_membership"] == 1


# ── what the platform did, day by day ────────────────────────────────────────


async def test_activity_is_counted_per_day_from_platform_acts_only(estate, client: AsyncClient):
    now = datetime.now(UTC)
    for days_ago in (0, 0, 1, 9):
        await _audit(
            estate,
            organization_id="org-acme",
            actor="operator@example.com",
            action=AuditAction.ORGANIZATION_MEMBER_ADDED.value,
            resource_type="user",
            resource_id="user-solo@example.com",
            occurred_at=now - timedelta(days=days_ago),
        )
    # A customer's own operational record. It must not be counted here.
    await _audit(
        estate,
        organization_id="org-acme",
        actor="solo@example.com",
        action=AuditAction.EVIDENCE_READ.value,
        resource_type="evidence",
        resource_id="ev-1",
        occurred_at=now,
    )
    # Fifteen days back: outside the window entirely.
    await _audit(
        estate,
        organization_id="org-acme",
        actor="operator@example.com",
        action=AuditAction.ORGANIZATION_UPDATED.value,
        resource_type="organization",
        resource_id="org-acme",
        occurred_at=now - timedelta(days=15),
    )

    body = await _read(client)
    days = body["activity"]["days"]

    assert len(days) == 14
    assert days[-1]["date"] == now.date().isoformat()
    assert days[-1]["total"] == 2
    assert days[-2]["total"] == 1
    assert sum(day["total"] for day in days) == 4
    assert body["activity"]["last_7d"] == 3
    assert body["activity"]["previous_7d"] == 1


async def test_days_are_cut_in_the_readers_own_timezone(estate, client: AsyncClient):
    """23:30 UTC is already tomorrow in Singapore. A day is the reader's day."""
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    late = today - timedelta(minutes=30)  # 23:30 UTC yesterday
    await _audit(
        estate,
        organization_id="org-acme",
        actor="operator@example.com",
        action=AuditAction.ORGANIZATION_MEMBER_ADDED.value,
        resource_type="user",
        resource_id="user-solo@example.com",
        occurred_at=late,
    )

    utc = (await _read(client))["activity"]
    singapore = (await _read(client, utc_offset=480))["activity"]

    yesterday = (today - timedelta(days=1)).date().isoformat()
    assert {d["date"]: d["total"] for d in utc["days"]}[yesterday] == 1
    assert {d["date"]: d["total"] for d in singapore["days"]}[today.date().isoformat()] >= 1
    assert singapore["utc_offset"] == 480


async def test_an_impossible_offset_is_refused(estate, client: AsyncClient):
    headers = await bearer(client, "operator@example.com")
    response = await client.get(OVERVIEW, headers=headers, params={"utc_offset": 100_000})
    assert response.status_code == 422


async def test_refused_changes_this_week_are_counted(estate, client: AsyncClient):
    now = datetime.now(UTC)
    for days_ago, outcome in (
        (1, AuditOutcome.DENIED),
        (2, AuditOutcome.DENIED),
        (1, AuditOutcome.SUCCESS),
        (10, AuditOutcome.DENIED),
    ):
        await _audit(
            estate,
            organization_id="org-acme",
            actor="operator@example.com",
            action=AuditAction.ACCESS_SET.value,
            resource_type="user",
            resource_id="user-solo@example.com",
            outcome=outcome.value,
            occurred_at=now - timedelta(days=days_ago),
        )

    body = await _read(client)

    assert body["attention"]["refused_changes_7d"] == 2
    days = body["activity"]["days"]
    assert sum(day["refused"] for day in days) == 3


# ── the feed names people and places, and carries no raw detail ──────────────


async def test_recent_activity_names_the_organization_and_the_person(estate, client: AsyncClient):
    await _audit(
        estate,
        organization_id="org-borden",
        actor="operator@example.com",
        action=AuditAction.ACCESS_SET.value,
        resource_type="user",
        resource_id="user-both@example.com",
        detail=json.dumps(
            {
                "email": "both@example.com",
                "role": "auditor",
                "added": ["view_audit", "view_reports"],
                "removed": ["view_users"],
                "granted": [],
                "revoked": [],
            }
        ),
    )

    body = await _read(client)
    row = next(r for r in body["recent_activity"] if r["action"] == "user.access_set")

    assert row["organization_name"] == "Borden Foods"
    assert row["subject"] == "both@example.com"
    assert row["change"] == {"role": "auditor", "added": 2, "removed": 1}
    # The raw detail is a writer's record, not a wire format. Only the
    # summarised facts above leave the server.
    assert "detail" not in row


async def test_a_status_change_says_from_what_to_what(estate, client: AsyncClient):
    await _audit(
        estate,
        organization_id="org-borden",
        actor="operator@example.com",
        action=AuditAction.ORGANIZATION_STATUS_CHANGED.value,
        resource_type="organization",
        resource_id="org-borden",
        detail=json.dumps(
            {"from": "active", "to": "suspended", "reason": "unpaid", "cameras_stopped": 0}
        ),
    )

    body = await _read(client)
    row = next(r for r in body["recent_activity"] if r["action"] == "organization.status_changed")

    assert row["subject"] is None
    assert row["change"] == {"from": "active", "to": "suspended", "reason": "unpaid"}


async def test_a_refusal_carries_its_reason(estate, client: AsyncClient):
    await _audit(
        estate,
        organization_id="org-acme",
        actor="both@example.com",
        action=AuditAction.ACCESS_SET.value,
        resource_type="user",
        resource_id="user-both@example.com",
        outcome=AuditOutcome.DENIED.value,
        detail=json.dumps({"email": "both@example.com", "reason": "self"}),
    )

    body = await _read(client)
    row = next(r for r in body["recent_activity"] if r["outcome"] == "denied")

    assert row["change"] == {"reason": "self"}


async def test_an_unreadable_detail_is_survived_not_trusted(estate, client: AsyncClient):
    await _audit(
        estate,
        organization_id="org-acme",
        actor="operator@example.com",
        action=AuditAction.ORGANIZATION_MEMBER_ADDED.value,
        resource_type="user",
        resource_id="user-gone@example.com",
        detail="{not json",
    )

    body = await _read(client)
    row = next(r for r in body["recent_activity"] if r["resource_id"] == "user-gone@example.com")

    # Nobody by that id exists any more and the detail says nothing usable.
    assert row["subject"] is None
    assert row["change"] is None


async def test_the_overview_says_when_it_was_counted(estate, client: AsyncClient):
    body = await _read(client)

    stamped = datetime.fromisoformat(body["generated_at"])
    assert stamped.tzinfo is not None
    assert abs((datetime.now(UTC) - stamped).total_seconds()) < 60
