"""The reporting engine, and the properties that make a report worth signing.

The two that matter most, and would each be easy to lose in a refactor:

* **A partial period never presents as complete.** `test_a_period_that_has_not
  _ended_is_never_complete` and its neighbours are the whole reason
  `Coverage.complete` is a computed property rather than a field somebody sets.
* **Reporting is not a permission bypass.** A report requires the permission for
  every source it reads, so an account that may not read the audit trail may not
  read it through a report either.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from app.authorization.model import Permission, Role, permissions_for
from app.domain.models import AuditEvent, Camera, Incident, Restaurant, Zone
from app.reporting import catalogue
from app.reporting.model import (
    Column,
    Coverage,
    ExportFormat,
    Gap,
    Granularity,
    ReportData,
    Section,
    SourceCoverage,
)
from app.reporting.periods import (
    buckets,
    gaps_for_window,
    parse_instant,
    resolve_timezone,
    resolve_window,
)
from app.reporting.render import format_available, render

from .conftest import bearer, make_user


@pytest.fixture
async def admin(seeded):
    """An `org_admin` inside org-test, holding every reporting permission."""
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


@pytest.fixture
async def with_incidents(admin):
    """A site in Singapore, a zone, a camera and five incidents over five days."""
    now = datetime.now(UTC)
    database = admin.state.database
    async with database.session_scope() as session:
        session.add(
            Restaurant(
                id="rest-1",
                organization_id="org-test",
                name="Harbour Kitchen",
                slug="harbour",
                timezone="Asia/Singapore",
            )
        )
        session.add(Zone(id="zone-prep", restaurant_id="rest-1", name="Prep line"))
        session.add(
            Camera(
                organization_id="org-test",
                restaurant_id="rest-1",
                camera_key="cam-01",
                name="Prep camera",
                channel=1,
            )
        )
        for index in range(5):
            session.add(
                Incident(
                    id=f"inc-{index}",
                    organization_id="org-test",
                    restaurant_id="rest-1",
                    # Frozen zone attribution: two recorded, three not.
                    zone_id="zone-prep" if index < 2 else None,
                    camera_key="cam-01",
                    rule_id="hairnet-required" if index % 2 == 0 else "gloves-required",
                    severity="high" if index < 2 else "medium",
                    status="resolved" if index == 0 else "active",
                    created_at=now - timedelta(days=index),
                    observed_at=now - timedelta(days=index),
                )
            )
    return admin


# ── The partial-period rule ──────────────────────────────────────────────────


def test_a_period_that_has_not_ended_is_never_complete() -> None:
    """The single most important property in this module.

    A month in progress produces rows, and a report that rendered them without
    saying the month is unfinished invites a comparison with a finished one. The
    gap is computed from the window alone, so it holds even when every source
    answered perfectly.
    """
    now = datetime.now(UTC)
    found = gaps_for_window(since=now - timedelta(days=10), until=now + timedelta(days=2))

    assert [gap.kind for gap in found] == ["future"]
    assert "has not finished" in found[0].detail

    coverage = Coverage(
        since=now - timedelta(days=10),
        until=now + timedelta(days=2),
        timezone="UTC",
        timezone_resolved=True,
        granularity=Granularity.DAY,
        sources=(SourceCoverage("incidents", available=True, rows=42),),
        gaps=tuple(found),
    )
    # Forty-two rows and still incomplete. Rows are not coverage.
    assert coverage.complete is False


def test_a_window_reaching_past_retention_is_not_a_quiet_period() -> None:
    """Deleted-on-schedule and never-happened must not look the same."""
    now = datetime.now(UTC)
    found = gaps_for_window(
        since=now - timedelta(days=200),
        until=now - timedelta(days=1),
        retention_days=90,
        retention_subject="observations",
    )

    kinds = [gap.kind for gap in found]
    assert kinds == ["before_history"]
    assert "retention outcome" in found[0].detail


def test_an_unavailable_source_makes_the_whole_report_incomplete() -> None:
    """One source down is enough. `complete` reads every source, not the best one."""
    now = datetime.now(UTC)
    coverage = Coverage(
        since=now - timedelta(days=2),
        until=now - timedelta(days=1),
        timezone="UTC",
        timezone_resolved=True,
        granularity=Granularity.TOTAL,
        sources=(
            SourceCoverage("incidents", available=True, rows=10),
            SourceCoverage("observations", available=False, reason="not assembled"),
        ),
    )
    assert coverage.complete is False


def test_a_truncated_source_makes_the_report_incomplete() -> None:
    """A row cap is a coverage gap, not a rendering detail."""
    now = datetime.now(UTC)
    coverage = Coverage(
        since=now - timedelta(days=2),
        until=now - timedelta(days=1),
        timezone="UTC",
        timezone_resolved=True,
        granularity=Granularity.TOTAL,
        sources=(SourceCoverage("audit", available=True, rows=5000, truncated=True),),
    )
    assert coverage.complete is False


def test_a_finished_window_with_every_source_read_is_complete() -> None:
    """The positive case, so `complete` is not merely always false."""
    now = datetime.now(UTC)
    coverage = Coverage(
        since=now - timedelta(days=3),
        until=now - timedelta(days=1),
        timezone="UTC",
        timezone_resolved=True,
        granularity=Granularity.TOTAL,
        sources=(SourceCoverage("incidents", available=True, rows=3),),
        gaps=(),
    )
    assert coverage.complete is True


def test_a_report_cannot_be_built_without_coverage() -> None:
    """Structural, not conventional. Forgetting is a TypeError, not a bad page."""
    with pytest.raises(TypeError):
        ReportData(  # type: ignore[call-arg]
            report_id="x", title="X", subtitle="", sections=()
        )


# ── Periods and timezones ────────────────────────────────────────────────────


def test_month_boundaries_are_local_not_utc() -> None:
    """September for a Singapore kitchen begins at 16:00 UTC on 31 August.

    A monthly report computed on UTC boundaries silently attributes eight hours
    of every month to the wrong one.
    """
    zone = resolve_timezone("Asia/Singapore")
    assert zone.resolved is True

    since, until = resolve_window(
        since=datetime(2026, 9, 3, tzinfo=UTC),
        until=datetime(2026, 9, 20, tzinfo=UTC),
        zone=zone,
        granularity=Granularity.MONTH,
    )
    assert since == datetime(2026, 8, 31, 16, 0, tzinfo=UTC)
    assert until == datetime(2026, 9, 30, 16, 0, tzinfo=UTC)
    assert [label for _, _, label in buckets(since, until, Granularity.MONTH, zone)] == ["2026-09"]


def test_an_unresolvable_timezone_is_reported_rather_than_swallowed() -> None:
    """UTC fallback is fine. A silent UTC fallback is not.

    `zoneinfo` reads the host's tz database and a minimal container may have
    none. Boundaries then differ by up to a day, with a report that looks
    entirely confident — so the fact travels in the coverage.
    """
    zone = resolve_timezone("Mars/Olympus_Mons")
    assert zone.resolved is False
    assert zone.effective is UTC


def test_a_window_longer_than_the_cap_is_refused() -> None:
    """Bounded rather than backgrounded. A decade is a narrower question."""
    from app.errors import ValidationError

    with pytest.raises(ValidationError):
        resolve_window(
            since=datetime(2020, 1, 1, tzinfo=UTC),
            until=datetime(2026, 1, 1, tzinfo=UTC),
            zone=resolve_timezone("UTC"),
            granularity=Granularity.TOTAL,
        )


def test_a_naive_instant_is_read_as_utc_not_as_server_local() -> None:
    """Otherwise a report means something different on every host."""
    parsed = parse_instant("2026-09-01T08:00:00")
    assert parsed == datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    assert parse_instant("2026-09-01T08:00:00Z") == parsed


# ── The routes ───────────────────────────────────────────────────────────────


async def test_the_catalogue_lists_every_report_including_the_unrunnable(
    client: AsyncClient, admin
) -> None:
    """Twelve, and the seven unconnected modules are among them.

    Omitting a module from a catalogue that claims to cover the product would
    let its absence read as "nothing to report" — the exact confusion those
    modules' own pages exist to prevent.
    """
    headers = await bearer(client, "admin@example.com")
    body = (await client.get("/api/v1/reports/types", headers=headers)).json()

    assert body["count"] == 12
    kinds = [r["kind"] for r in body["reports"]]
    assert kinds.count("data") == 5
    assert kinds.count("capability") == 7
    assert body["can_export"] is True


async def test_an_incident_report_aggregates_real_rows(
    client: AsyncClient, with_incidents
) -> None:
    headers = await bearer(client, "admin@example.com")
    since = (datetime.now(UTC) - timedelta(days=7)).isoformat()

    body = (
        await client.get(
            "/api/v1/reports/incident_summary",
            params={"since": since, "granularity": "day", "restaurant_id": "rest-1"},
            headers=headers,
        )
    ).json()

    assert body["coverage"]["timezone"] == "Asia/Singapore"
    assert body["coverage"]["sources"][0] == {
        "source": "incidents",
        "available": True,
        "reason": "",
        "rows": 5,
        "truncated": False,
        "earliest": body["coverage"]["sources"][0]["earliest"],
    }

    by_severity = next(s for s in body["sections"] if s["key"] == "incidents_severity")
    assert {row["severity"]: row["count"] for row in by_severity["rows"]} == {
        "high": 2,
        "medium": 3,
    }


async def test_frozen_zone_attribution_is_read_as_stored(
    client: AsyncClient, with_incidents
) -> None:
    """The Phase 2 property, defended inside a report.

    Two incidents carry `zone_id = zone-prep`; three carry none. Moving the
    camera afterwards must not move any of them — a report that joined
    `cameras.zone_id` would re-attribute all five to wherever the camera sits
    when the report runs.
    """
    from app.domain.cameras import CameraService

    database = with_incidents.state.database
    async with database.session_scope() as session:
        session.add(Zone(id="zone-wash", restaurant_id="rest-1", name="Wash station"))
        await session.flush()
        await CameraService(session).update(
            organization_id="org-test", camera_key="cam-01", zone_id="zone-wash"
        )

    headers = await bearer(client, "admin@example.com")
    since = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    body = (
        await client.get(
            "/api/v1/reports/incident_summary",
            params={"since": since, "restaurant_id": "rest-1"},
            headers=headers,
        )
    ).json()

    by_zone = next(s for s in body["sections"] if s["key"] == "incidents_zone")
    tally = {row["zone"]: row["count"] for row in by_zone["rows"]}

    # Still the prep line, even though the camera is now in the wash station.
    assert tally == {"Prep line": 2, "Not recorded": 3}
    assert "Wash station" not in tally


async def test_an_empty_section_carries_a_sentence_rather_than_nothing(
    client: AsyncClient, admin
) -> None:
    """A header with no rows under it reads as a clean month. It must not."""
    headers = await bearer(client, "admin@example.com")
    since = (datetime.now(UTC) - timedelta(days=3)).isoformat()

    body = (
        await client.get(
            "/api/v1/reports/incident_summary", params={"since": since}, headers=headers
        )
    ).json()

    by_zone = next(s for s in body["sections"] if s["key"] == "incidents_zone")
    assert by_zone["rows"] == []
    assert by_zone["empty_note"]
    assert "nothing to break down" in by_zone["empty_note"]


async def test_an_unavailable_platform_is_not_reported_as_no_observations(
    client: AsyncClient, admin
) -> None:
    """The most dangerous document this product could produce, refused.

    Vision is not assembled in the test app, so the hygiene report must say the
    source could not be read — never render an empty PPE table, which would tell
    a manager their kitchen was clean when nothing was watching it.
    """
    headers = await bearer(client, "admin@example.com")
    body = (await client.get("/api/v1/reports/hygiene_observations", headers=headers)).json()

    source = body["coverage"]["sources"][0]
    assert source["source"] == "observations"
    assert source["available"] is False
    assert "not assembled" in source["reason"]
    assert body["coverage"]["complete"] is False
    # No PPE table at all, rather than an empty one.
    assert body["sections"] == []
    assert any(gap["kind"] == "source_unavailable" for gap in body["coverage"]["gaps"])


@pytest.mark.parametrize(
    "report_id,module",
    [
        ("module_people_counting", "people_counting"),
        ("module_demography", "demography"),
        ("module_table_occupancy", "table_occupancy"),
        ("module_cutting_board", "cutting_board"),
        ("module_meal_detection", "meal_detection"),
        ("module_pos_integration", "pos_integration"),
        ("module_patron_id", "patron_id"),
    ],
)
async def test_every_unconnected_module_reports_its_own_honest_state(
    client: AsyncClient, admin, report_id: str, module: str
) -> None:
    """All seven, each in its own words, from the Phase 2 shape unchanged."""
    headers = await bearer(client, "admin@example.com")
    body = (await client.get(f"/api/v1/reports/{report_id}", headers=headers)).json()

    assert body["capability_state"] in {"not_configured", "blocked"}
    assert body["capability_reason"]
    assert body["awaiting"], "an unconnected module must name what it awaits"
    assert body["coverage"]["complete"] is False

    source = body["coverage"]["sources"][0]
    assert source["source"] == module
    assert source["available"] is False


async def test_patron_id_reports_blocked_rather_than_not_configured(
    client: AsyncClient, admin
) -> None:
    """The distinction survives into the report, as it must."""
    headers = await bearer(client, "admin@example.com")
    patron = (await client.get("/api/v1/reports/module_patron_id", headers=headers)).json()
    counting = (
        await client.get("/api/v1/reports/module_people_counting", headers=headers)
    ).json()

    assert patron["capability_state"] == "blocked"
    assert counting["capability_state"] == "not_configured"


# ── Permissions ──────────────────────────────────────────────────────────────


async def test_a_report_requires_every_permission_its_sources_need(
    client: AsyncClient, admin
) -> None:
    """Reporting is not a permission bypass.

    A restaurant manager holds `view_reports` but not `view_audit`, so the audit
    report is refused — and reaching the same data through a report must be
    exactly as hard as reaching it directly.
    """
    headers = await bearer(client, "manager@example.com")

    assert (
        await client.get("/api/v1/reports/incident_summary", headers=headers)
    ).status_code == 200
    refused = await client.get("/api/v1/reports/audit_activity", headers=headers)
    assert refused.status_code == 403
    assert "view_audit" in refused.json()["details"]["missing"]


async def test_a_refused_report_is_audited_with_the_same_weight_as_a_success(
    client: AsyncClient, admin
) -> None:
    """The pattern evidence retrieval established, applied here.

    A refused attempt to assemble a report about staff is precisely the row an
    investigation needs, and losing it because the request failed would be
    exactly backwards.
    """
    headers = await bearer(client, "manager@example.com")
    assert (await client.get("/api/v1/reports/audit_activity", headers=headers)).status_code == 403

    database = admin.state.database
    async with database.session_scope() as session:
        from sqlalchemy import select

        actions = (
            (
                await session.execute(
                    select(AuditEvent.action, AuditEvent.outcome, AuditEvent.resource_id)
                )
            )
            .all()
        )

    assert ("report.denied", "denied", "audit_activity") in actions


async def test_export_is_a_separate_permission_from_reading(
    client: AsyncClient, seeded
) -> None:
    """A kitchen supervisor may read a report and may not take a copy away.

    The screen is shared; a downloaded file is not, and it outlives every
    retention policy this application enforces.
    """
    supervisor = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
    assert Permission.VIEW_REPORTS in supervisor
    assert Permission.EXPORT_REPORTS not in supervisor

    headers = await bearer(client, "supervisor@example.com")
    assert (
        await client.get("/api/v1/reports/incident_summary", headers=headers)
    ).status_code == 200
    assert (
        await client.get(
            "/api/v1/reports/incident_summary/export", params={"format": "csv"}, headers=headers
        )
    ).status_code == 403


async def test_a_report_is_scoped_to_the_caller_tenant(
    client: AsyncClient, with_incidents
) -> None:
    """Another organisation's incidents are invisible, not merely filtered out."""
    outsider = await bearer(client, "outsider@example.com")
    body = (
        await client.get(
            "/api/v1/reports/incident_summary",
            params={"since": (datetime.now(UTC) - timedelta(days=7)).isoformat()},
            headers=outsider,
        )
    ).json()

    assert body["coverage"]["sources"][0]["rows"] == 0


async def test_an_account_with_no_cameras_reads_no_incidents(
    client: AsyncClient, with_incidents
) -> None:
    """An empty camera grant matches nothing. It must never read as a wildcard."""
    headers = await bearer(client, "nocameras@example.com")
    body = (
        await client.get(
            "/api/v1/reports/incident_summary",
            params={"since": (datetime.now(UTC) - timedelta(days=7)).isoformat()},
            headers=headers,
        )
    ).json()

    assert body["coverage"]["sources"][0]["rows"] == 0


# ── Export ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", ["pdf", "xlsx", "csv", "json"])
async def test_export_produces_a_file_and_never_caches_it(
    client: AsyncClient, with_incidents, fmt: str
) -> None:
    """A report naming a small shift team must not sit in a shared cache."""
    headers = await bearer(client, "admin@example.com")
    response = await client.get(
        "/api/v1/reports/incident_summary/export",
        params={"format": fmt, "since": (datetime.now(UTC) - timedelta(days=7)).isoformat()},
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert len(response.content) > 200
    assert response.headers["cache-control"] == "no-store, private"
    assert fmt in response.headers["content-disposition"]


async def test_every_export_writes_an_audit_row(client: AsyncClient, with_incidents) -> None:
    """Generation and export are recorded separately, because they differ.

    'Who ran a report' and 'who took a copy away' are different questions, and
    the second is the one an investigation actually asks.
    """
    headers = await bearer(client, "admin@example.com")
    await client.get(
        "/api/v1/reports/incident_summary/export", params={"format": "csv"}, headers=headers
    )

    database = with_incidents.state.database
    async with database.session_scope() as session:
        from sqlalchemy import select

        rows = (
            (
                await session.execute(
                    select(AuditEvent.action, AuditEvent.detail).where(
                        AuditEvent.resource_type == "report"
                    )
                )
            )
            .all()
        )

    actions = [action for action, _ in rows]
    assert "report.exported" in actions
    # Format, size and window — never the figures. An audit row that copied the
    # contents would be a second, unretained store of the record it governs.
    exported = next(detail for action, detail in rows if action == "report.exported")
    assert "size_bytes" in exported
    assert "format" in exported


def test_an_export_carries_the_coverage_and_the_empty_notes() -> None:
    """Every format, and the incomplete banner in each.

    A spreadsheet a manager opens must not need a second document to tell them
    the period was unfinished.
    """
    now = datetime.now(UTC)
    data = ReportData(
        report_id="incident_summary",
        title="Incident summary",
        subtitle="What this covers.",
        coverage=Coverage(
            since=now - timedelta(days=5),
            until=now + timedelta(days=1),
            timezone="Asia/Singapore",
            timezone_resolved=True,
            granularity=Granularity.DAY,
            sources=(
                SourceCoverage("incidents", available=True, rows=2),
                SourceCoverage("observations", available=False, reason="not assembled"),
            ),
            gaps=(Gap("future", "This period has not finished."),),
        ),
        sections=(
            Section(
                key="empty",
                title="By zone",
                columns=(Column("zone", "Zone"), Column("count", "Incidents", numeric=True)),
                rows=(),
                empty_note="No incident was raised in this period.",
            ),
        ),
        generated_at=now,
    )

    text = render(data, ExportFormat.CSV).content.decode("utf-8-sig")
    assert "INCOMPLETE" in text
    assert "NOT AVAILABLE" in text
    assert "No incident was raised in this period." in text
    assert "This period has not finished." in text

    payload = render(data, ExportFormat.JSON).content.decode("utf-8")
    assert '"complete": false' in payload

    # The binary formats produce real files rather than raising.
    assert render(data, ExportFormat.XLSX).content[:2] == b"PK"
    assert render(data, ExportFormat.PDF).content[:5] == b"%PDF-"


def test_a_missing_export_library_is_a_named_gap_not_a_traceback(monkeypatch) -> None:
    """CSV and JSON are stdlib and always work; the rest say why they cannot."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "reportlab":
            raise ImportError("no reportlab in this deployment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    available, reason = format_available(ExportFormat.PDF)
    assert available is False
    assert "reports" in reason and "reportlab" in reason

    # The two that never need a dependency.
    assert format_available(ExportFormat.CSV)[0] is True
    assert format_available(ExportFormat.JSON)[0] is True


# ── The catalogue as an extension point ──────────────────────────────────────


def test_every_report_declares_its_sources_permissions() -> None:
    """The rule that keeps reporting from becoming a bypass, asserted.

    Every report holds `VIEW_REPORTS`, and every data report additionally names
    at least one source permission. A report that named only `VIEW_REPORTS`
    would grant its data to anyone who could reach the page.
    """
    for report in catalogue.CATALOGUE:
        assert Permission.VIEW_REPORTS in report.permissions, report.id
        assert len(report.permissions) >= 2, report.id


def test_a_capability_report_needs_no_collector() -> None:
    """Adding a real one later is implementing a collector, not a redesign."""
    for report in catalogue.MODULE_REPORTS:
        assert report.is_capability
        assert report.collectors == ()

    for report in catalogue.DATA_REPORTS:
        assert not report.is_capability
        assert report.collectors, report.id


def test_every_report_supports_the_total_granularity() -> None:
    """`total` is the one bucket every window has, so it is always offered."""
    for report in catalogue.CATALOGUE:
        assert Granularity.TOTAL in report.granularities, report.id


# ── The Phase 2 close-out: deleting a camera ─────────────────────────────────


async def test_deleting_a_camera_purges_its_observation_partition(
    client: AsyncClient, with_incidents, monkeypatch
) -> None:
    """The retention gap, closed at the delete rather than guessed at by a sweep.

    Retention enumerates partitions from the camera table, so a deleted row
    would orphan its observations — they would outlive their retention with
    nothing left to sweep them.
    """
    truncated: list[str] = []

    class RecordingLog:
        def truncate(self, partition, before) -> int:
            truncated.append(str(partition))
            return 7

    # The route reaches the bound log through `app.main._observation_log_of`,
    # so that is what a test with no synthesis assembled substitutes. Patching
    # the accessor rather than the runtime keeps the route's own wiring under
    # test instead of replacing it.
    monkeypatch.setattr("app.main._observation_log_of", lambda app: RecordingLog())

    headers = await bearer(client, "admin@example.com")
    response = await client.delete("/api/v1/cameras/cam-01", headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["observations_removed"] == 7
    assert truncated == ["cam-01"]
    # The zone history survives: incidents still name this camera and still need
    # to say where they happened.
    assert body["zone_history_retained"] is True


async def test_deleting_a_camera_is_audited_as_both_acts(
    client: AsyncClient, with_incidents, monkeypatch
) -> None:
    """A configuration change and a destruction of records. Two rows, always."""

    monkeypatch.setattr(
        "app.main._observation_log_of",
        lambda app: type("L", (), {"truncate": lambda self, p, b: 3})(),
    )

    app = with_incidents
    headers = await bearer(client, "admin@example.com")
    await client.delete("/api/v1/cameras/cam-01", headers=headers)

    async with app.state.database.session_scope() as session:
        from sqlalchemy import select

        actions = (
            (await session.execute(select(AuditEvent.action, AuditEvent.resource_id))).all()
        )

    assert ("camera.deleted", "cam-01") in actions
    assert ("observation.truncated", "cam-01") in actions


async def test_a_camera_is_not_deleted_when_its_partition_cannot_be_purged(
    client: AsyncClient, with_incidents
) -> None:
    """Neither, rather than one. A failed purge must not leave an orphan.

    The test app binds no synthesis, so a deployment configured for a durable
    log cannot reach it — and the deletion refuses instead of creating exactly
    the orphan it exists to prevent.
    """
    app = with_incidents
    app.state.settings.observation_log = "file"

    headers = await bearer(client, "admin@example.com")
    response = await client.delete("/api/v1/cameras/cam-01", headers=headers)

    assert response.status_code == 500
    assert "outlive their retention" in response.json()["message"]

    # And the camera is still there.
    async with app.state.database.session_scope() as session:
        from sqlalchemy import select

        remaining = (
            await session.execute(select(Camera.camera_key).where(Camera.camera_key == "cam-01"))
        ).scalar_one_or_none()
    assert remaining == "cam-01"


def test_manage_patron_id_is_held_by_nobody() -> None:
    """Excluded explicitly, not merely inert because an unrelated guard refuses.

    A permission that is harmless only because of a guard elsewhere is a trap
    for whoever relaxes that guard later without knowing it was doing silent
    work.
    """
    for role in Role:
        assert Permission.MANAGE_PATRON_ID not in permissions_for(frozenset({role})), role
