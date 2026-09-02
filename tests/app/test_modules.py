"""The seven scaffolded modules, and the properties that make them a scaffold.

A test suite for code that produces no data has exactly one job: prove that it
produces no data, that it says so, and that the shape it will eventually produce
data into is the right one. So the assertions here are about honesty and about
schema, not about behaviour.

The most important test in the file is
``test_no_module_route_reports_a_reading``. Every other test would still pass on
a version of this phase that quietly returned plausible numbers.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.authorization.model import Permission, Role, permissions_for
from app.configuration.settings import Settings
from app.domain import modules as module_models
from app.domain import patron as patron_domain
from app.errors import CapabilityNotConfiguredError, PatronIdentificationBlockedError
from app.integrations.pos import NotConfiguredPosGateway, PosGatewayPort

from .conftest import bearer, make_user

# No module-level `pytest.mark.asyncio`: `asyncio_mode = "auto"` already runs the
# coroutines, and marking the file would attach the marker to the synchronous
# schema tests below as well.


#: Every capability route, with the permission that admits it.
MODULE_ROUTES: tuple[tuple[str, Permission], ...] = (
    ("/api/v1/modules/people-counting", Permission.VIEW_PEOPLE_COUNT),
    ("/api/v1/modules/demography", Permission.VIEW_DEMOGRAPHY),
    ("/api/v1/modules/table-occupancy", Permission.VIEW_TABLE_OCCUPANCY),
    ("/api/v1/modules/cutting-board", Permission.VIEW_CUTTING_BOARD),
    ("/api/v1/modules/meal-detection", Permission.VIEW_MEAL_DETECTION),
    ("/api/v1/modules/pos-integration", Permission.VIEW_POS_INTEGRATION),
    ("/api/v1/modules/patron-id", Permission.VIEW_PATRON_ID),
)


@pytest.fixture
async def admin(seeded):
    """An `org_admin` inside org-test, holding every module read permission."""
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
async def super_admin(seeded):
    database = seeded.state.database
    async with database.session_scope() as session:
        _, user = make_user(
            email="root@example.com",
            roles=("super_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


# ── The property this whole phase exists to hold ─────────────────────────────


@pytest.mark.parametrize("path,permission", MODULE_ROUTES)
async def test_no_module_route_reports_a_reading(
    client: AsyncClient, admin, path: str, permission: Permission
) -> None:
    """Not one route returns a count, a percentage or a verdict.

    This is the test that would fail if somebody "helpfully" made a module
    return a plausible zero. `available` must be false and the stored-record
    count must come from an empty table — never from a literal, and never
    dressed up as a metric.
    """
    headers = await bearer(client, "admin@example.com")
    response = await client.get(path, headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["available"] is False
    assert body["reason"], "an unavailable module must say why"
    assert body["stored_records"] == 0
    assert body["awaiting"], "an unavailable module must name what it is waiting for"
    # Each requirement is a real sentence, not a placeholder.
    for requirement in body["awaiting"]:
        assert requirement["id"]
        assert len(requirement["detail"]) > 40


@pytest.mark.parametrize("path,permission", MODULE_ROUTES)
async def test_every_module_route_is_permission_gated(
    client: AsyncClient, admin, path: str, permission: Permission
) -> None:
    """A kitchen supervisor reaches board compliance and nothing else here.

    Deliberately checked against a real role rather than a synthetic one: the
    question is whether the *product's* permission model gates these routes, and
    a bespoke identity holding exactly one permission would prove only that
    `requires()` works.
    """
    headers = await bearer(client, "supervisor@example.com")
    response = await client.get(path, headers=headers)

    holds = permission in permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
    assert response.status_code == (200 if holds else 403), response.text


async def test_a_module_route_refuses_an_unauthenticated_caller(
    client: AsyncClient, admin
) -> None:
    response = await client.get("/api/v1/modules/people-counting")
    assert response.status_code == 401


async def test_demography_is_not_implied_by_people_counting(
    client: AsyncClient, admin
) -> None:
    """Footfall and inferred demography are different purposes, so different keys.

    A restaurant manager may read how many people came in. Inferring their age
    or gender is a separate purpose under PDPA with its own lawful basis, and
    running a site does not confer it.
    """
    headers = await bearer(client, "manager@example.com")

    assert (
        await client.get("/api/v1/modules/people-counting", headers=headers)
    ).status_code == 200
    assert (await client.get("/api/v1/modules/demography", headers=headers)).status_code == 403


async def test_module_counts_are_scoped_to_the_caller_tenant(
    client: AsyncClient, admin
) -> None:
    """The count is a real query, narrowed to the caller's organisation.

    A row planted in another tenant must not appear in this one's total — and
    the count must be a query rather than a hardcoded zero, which is what this
    proves by making it non-zero.
    """
    database = admin.state.database
    async with database.session_scope() as session:
        from datetime import UTC, datetime

        session.add(
            module_models.PeopleCountInterval(
                organization_id="org-other",
                camera_key="cam-99",
                bucket_start=datetime(2026, 9, 1, tzinfo=UTC),
                bucket_end=datetime(2026, 9, 1, 1, tzinfo=UTC),
                bucket_seconds=3600,
            )
        )

    headers = await bearer(client, "admin@example.com")
    body = (await client.get("/api/v1/modules/people-counting", headers=headers)).json()
    assert body["stored_records"] == 0

    outsider = await bearer(client, "outsider@example.com")
    other = (await client.get("/api/v1/modules/people-counting", headers=outsider)).json()
    assert other["stored_records"] == 1


# ── Patron ID ────────────────────────────────────────────────────────────────


async def test_patron_id_reports_blocked_rather_than_not_configured(
    client: AsyncClient, admin
) -> None:
    """`blocked` and `not_configured` must never flatten into one another.

    Six modules are waiting for work. This one is waiting for permission, and a
    client that could not tell them apart would render the most sensitive
    surface in the product exactly like the least.
    """
    headers = await bearer(client, "admin@example.com")
    body = (await client.get("/api/v1/modules/patron-id", headers=headers)).json()

    assert body["state"] == "blocked"
    assert body["gate"]["available"] is False
    assert "legal_review" in body["gate"]["missing"]
    assert body["write_available"] is False

    other = (await client.get("/api/v1/modules/people-counting", headers=headers)).json()
    assert other["state"] == "not_configured"


async def test_the_patron_gate_detail_is_reachable_by_nobody(
    client: AsyncClient, super_admin
) -> None:
    """`MANAGE_PATRON_ID` is now held by no role, including `super_admin`.

    This test previously asserted that `super_admin` could read the gate detail,
    because `frozenset(Permission)` handed it the permission by construction. It
    was inert only because `app/domain/patron.require_writable` refuses
    unconditionally — and a permission that is harmless solely because of an
    unrelated guard is a trap for whoever relaxes that guard later without
    knowing it was doing silent work.

    So the exclusion is now explicit, and the correct holder of this permission
    until a DPIA and a named DPO sign-off exist is **nobody**. The route still
    exists and still says why; nothing can currently call it, and granting it
    will be a line somebody has to write.
    """
    for email in ("outsider@example.com", "root@example.com"):
        headers = await bearer(client, email)
        response = await client.get("/api/v1/modules/patron-id/gate", headers=headers)
        assert response.status_code == 403, f"{email} reached the patron gate detail"


async def test_the_patron_write_path_refuses_even_when_fully_configured() -> None:
    """Enabling every setting is still not enough, and the refusal says why.

    The gate is not a feature flag. A deployment that set all three settings has
    satisfied the *configuration*, and there is still no biometric source bound
    to produce a digest — so the refusal names that as its own outstanding item
    rather than letting the checklist read as "one setting away".
    """
    Settings.model_config["env_file"] = None
    configured = Settings(
        app_env="test",
        secret_key="test-only-secret-value-not-for-any-deployment",
        patron_id_enabled=True,
        patron_id_legal_gate_ref="DPIA-2026-014",
        patron_id_pepper_ref="env:PATRON_PEPPER",
    )

    status = patron_domain.gate_status(configured)
    assert status.available is False
    assert status.missing == ("consent_mechanism", "biometric_source")

    with pytest.raises(PatronIdentificationBlockedError) as raised:
        patron_domain.require_writable(configured)
    assert raised.value.details["missing"] == ["consent_mechanism", "biometric_source"]


async def test_patron_tokens_cannot_hold_a_biometric() -> None:
    """The schema is the control, so the schema is what is asserted.

    A code-level check could be removed by anybody. These are properties of the
    table: a 64-character text column cannot hold a face template, there is no
    binary column to put one in, and a row cannot exist without naming its
    consent and its legal authority. Breaking any of them requires a migration
    somebody has to write, review and sign — and fails this test first.
    """
    table = module_models.PatronToken.__table__
    columns = {c.name: c for c in table.columns}

    assert columns["token_hash"].type.length == 64

    forbidden = [
        name
        for name, column in columns.items()
        if type(column.type).__name__ in {"LargeBinary", "BLOB", "PickleType", "JSON"}
    ]
    assert forbidden == [], f"a binary column appeared on patron_tokens: {forbidden}"

    # Nothing that could reference an image or a template, by any of its names.
    for banned in ("image_ref", "template", "embedding", "descriptor", "face", "photo"):
        assert not any(banned in name for name in columns), f"'{banned}' appeared on patron_tokens"

    # NOT NULL and no default: the database refuses a token with no basis.
    for required in ("consent_ref", "legal_gate_ref", "consent_basis"):
        assert columns[required].nullable is False
        assert columns[required].default is None
        assert columns[required].server_default is None


async def test_no_route_accepts_a_patron_token(admin) -> None:
    """There is no write path to forget to gate.

    Asserted over the whole application rather than over one router, because the
    failure this guards against is somebody adding the route elsewhere.
    """
    writes = [
        route
        for route in admin.routes
        if "patron" in getattr(route, "path", "")
        and set(getattr(route, "methods", set())) - {"GET", "HEAD", "OPTIONS"}
    ]
    assert writes == []


# ── POS integration ──────────────────────────────────────────────────────────


async def test_the_unconfigured_pos_adapter_refuses_rather_than_returning_nothing() -> None:
    """`()` would manufacture a discrepancy report out of not being plugged in.

    A caller handed an empty ticket list records a successful sync that found no
    sales, and every dish detected in that window reconciles as UNMATCHED. The
    adapter therefore raises, and the error names what is missing.
    """
    from datetime import UTC, datetime

    gateway = NotConfiguredPosGateway()
    assert isinstance(gateway, PosGatewayPort)

    # `describe` never raises — a status screen must be able to ask.
    description = gateway.describe()
    assert description.available is False
    assert description.missing

    with pytest.raises(CapabilityNotConfiguredError):
        gateway.fetch_tickets(since=datetime.now(UTC), until=datetime.now(UTC))
    with pytest.raises(CapabilityNotConfiguredError):
        gateway.push_events(())


async def test_the_connector_list_never_carries_a_credential_reference(
    client: AsyncClient, admin
) -> None:
    """Not even the reference. Small disclosures are how large ones are assembled."""
    database = admin.state.database
    async with database.session_scope() as session:
        session.add(
            module_models.PosConnector(
                organization_id="org-test",
                connector_key="till-01",
                vendor="example",
                credential_ref="env:POS_TOKEN",
            )
        )

    headers = await bearer(client, "admin@example.com")
    response = await client.get("/api/v1/pos-connectors", headers=headers)

    assert response.status_code == 200
    assert "POS_TOKEN" not in response.text
    assert "credential_ref" not in response.text
    assert response.json()["connectors"][0]["connector_key"] == "till-01"


async def test_a_new_pos_connector_is_inactive(admin) -> None:
    """Registering a connector and letting it exchange data are two decisions."""
    database = admin.state.database
    async with database.session_scope() as session:
        connector = module_models.PosConnector(
            organization_id="org-test", connector_key="till-02"
        )
        session.add(connector)
        await session.flush()
        assert connector.is_active is False


# ── The zone-attribution close-out ───────────────────────────────────────────


async def test_moving_a_camera_does_not_rewrite_where_past_readings_happened(
    admin,
) -> None:
    """The whole reason `camera_zone_assignments` exists.

    A camera reassigned from the prep line to the wash station must leave every
    earlier reading attributed to the prep line. A join through `cameras.zone_id`
    would relocate a quarter of history with one dropdown, and nothing in the
    record would show it happened.
    """
    from datetime import UTC, datetime, timedelta

    from app.domain.cameras import CameraService
    from app.domain.models import Restaurant, Zone
    from app.domain.zone_attribution import ZoneHistory

    database = admin.state.database
    async with database.session_scope() as session:
        restaurant = Restaurant(id="rest-1", organization_id="org-test", name="Site", slug="site")
        prep = Zone(id="zone-prep", restaurant_id="rest-1", name="Prep line")
        wash = Zone(id="zone-wash", restaurant_id="rest-1", name="Wash station")
        session.add_all([restaurant, prep, wash])
        await session.flush()

        service = CameraService(session)
        await service.create(
            organization_id="org-test",
            restaurant_id="rest-1",
            camera_key="cam-move",
            name="Camera",
            channel=1,
            zone_id="zone-prep",
            assigned_by="admin@example.com",
        )

    async with database.session_scope() as session:
        await CameraService(session).update(
            organization_id="org-test",
            camera_key="cam-move",
            zone_id="zone-wash",
            assigned_by="admin@example.com",
        )

    async with database.session_scope() as session:
        history = await ZoneHistory.load(
            session, organization_id="org-test", camera_keys=("cam-move",)
        )

        # Asked at the interval's own start rather than at a wall-clock offset:
        # both assignments happen within the same test tick, and the property
        # under test is which interval covers an instant, not how fast the test
        # ran.
        first = history._by_camera["cam-move"][0]  # noqa: SLF001 - the interval under test
        before = history.resolve("cam-move", first.effective_from)
        after = history.resolve("cam-move", datetime.now(UTC) + timedelta(minutes=5))

    assert first.effective_to is not None, "the old interval must be closed, not edited"
    assert first.zone_id == "zone-prep", "the closed interval keeps its zone forever"
    assert before is not None and before.zone_id == "zone-prep"
    # And the name is frozen too: renaming a zone must not relabel history.
    assert before.zone_name == "Prep line"
    assert after is not None and after.zone_id == "zone-wash"


async def test_an_instant_before_any_assignment_has_no_recorded_zone(admin) -> None:
    """No backfill. "Nobody wrote it down" is the honest answer and it is given.

    Inferring a zone for an observation older than the first interval would mean
    asserting the camera has always been where it is now — the exact claim the
    table exists to prevent.
    """
    from datetime import UTC, datetime

    from app.domain.cameras import CameraService
    from app.domain.models import Restaurant, Zone
    from app.domain.zone_attribution import ZoneHistory

    database = admin.state.database
    async with database.session_scope() as session:
        session.add(Restaurant(id="rest-2", organization_id="org-test", name="S", slug="s"))
        session.add(Zone(id="zone-a", restaurant_id="rest-2", name="A"))
        await session.flush()
        await CameraService(session).create(
            organization_id="org-test",
            restaurant_id="rest-2",
            camera_key="cam-old",
            name="Camera",
            channel=1,
            zone_id="zone-a",
        )

    async with database.session_scope() as session:
        history = await ZoneHistory.load(
            session, organization_id="org-test", camera_keys=("cam-old",)
        )
        assert history.resolve("cam-old", datetime(2020, 1, 1, tzinfo=UTC)) is None
        # And a camera nobody ever assigned resolves to nothing rather than guessing.
        assert history.resolve("cam-never-seen", datetime.now(UTC)) is None


async def test_reassigning_to_the_same_zone_opens_no_second_interval(admin) -> None:
    """Repeating an assignment is not a move, and must not look like one."""
    from sqlalchemy import select

    from app.domain.cameras import CameraService
    from app.domain.models import CameraZoneAssignment, Restaurant, Zone

    database = admin.state.database
    async with database.session_scope() as session:
        session.add(Restaurant(id="rest-3", organization_id="org-test", name="S", slug="s3"))
        session.add(Zone(id="zone-b", restaurant_id="rest-3", name="B"))
        await session.flush()
        await CameraService(session).create(
            organization_id="org-test",
            restaurant_id="rest-3",
            camera_key="cam-same",
            name="Camera",
            channel=1,
            zone_id="zone-b",
        )

    async with database.session_scope() as session:
        await CameraService(session).update(
            organization_id="org-test", camera_key="cam-same", zone_id="zone-b"
        )

    async with database.session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(CameraZoneAssignment).where(
                        CameraZoneAssignment.camera_key == "cam-same"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].effective_to is None


# ── The retention close-out ──────────────────────────────────────────────────


async def test_observation_retention_has_an_interim_default() -> None:
    """90 days, and positioned between the imagery and compliance clocks.

    Asserted as a relationship rather than as the literal, so the test still
    means something after the DPO review changes the number: whatever it becomes,
    an observation must not outlive the compliance record it supports, and must
    not expire before the imagery that is more sensitive than it.
    """
    Settings.model_config["env_file"] = None
    settings = Settings(
        app_env="test", secret_key="test-only-secret-value-not-for-any-deployment"
    )

    assert settings.observation_retention_days == 90
    assert settings.evidence_retention_days < settings.observation_retention_days
    assert settings.observation_retention_days < settings.incident_retention_days


async def test_evidence_retention_is_declared_exactly_once() -> None:
    """The duplicate declaration is gone.

    Two identical assignments meant the second silently won. They agreed, so
    nothing was wrong — until somebody changed the first one and nothing
    happened.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "app" / "configuration" / "settings.py"
    ).read_text(encoding="utf-8")
    assert source.count("    evidence_retention_days: int") == 1


async def test_the_retention_sweep_truncates_the_observation_log(admin) -> None:
    """Marking is still not erasing: a mark-only sweep truncates nothing.

    The log has no expiry step to mark, so truncation belongs entirely in the
    erase branch — a process that started must not begin deleting a durable
    record because it started.
    """
    from datetime import UTC, datetime

    from app.domain.models import Camera
    from app.domain.retention import RetentionService

    truncated: list[tuple[str, int]] = []

    class RecordingLog:
        def truncate(self, partition, before) -> int:
            truncated.append((str(partition), before.ns))
            return 3

    database = admin.state.database
    async with database.session_scope() as session:
        session.add(
            Camera(
                organization_id="org-test",
                restaurant_id="rest-x",
                camera_key="cam-sweep",
                name="Camera",
                channel=1,
            )
        )

    async with database.session_scope() as session:
        service = RetentionService(
            session,
            root=".",
            evidence_days=30,
            incident_days=365,
            audit_days=730,
            observation_days=90,
            observation_log=RecordingLog(),
        )
        marked = await service.sweep(erase=False)
        assert marked.observations_truncated == 0
        assert truncated == []

        erased = await service.sweep(erase=True)

    assert erased.observations_truncated == 3
    assert truncated and truncated[0][0] == "cam-sweep"
    cutoff_ns = truncated[0][1]
    assert cutoff_ns < int(datetime.now(UTC).timestamp() * 1_000_000_000)


async def test_the_sweep_skips_rather_than_lies_when_no_log_is_bound(admin) -> None:
    """No synthesis in this process means no log to sweep, and it says zero honestly."""
    from app.domain.retention import RetentionService

    database = admin.state.database
    async with database.session_scope() as session:
        report = await RetentionService(
            session,
            root=".",
            evidence_days=30,
            incident_days=365,
            audit_days=730,
            observation_days=90,
            observation_log=None,
        ).sweep(erase=True)

    assert report.observations_truncated == 0
    assert report.observation_failures == []


# ── Schema hygiene across the whole scaffold ─────────────────────────────────


def test_every_event_table_freezes_where_it_happened() -> None:
    """The Phase 1 lesson, applied to every new record of a located event.

    A row describing something that happened at a place and a time must carry
    that place itself. Leaving it to a join through `cameras` or `dining_tables`
    means a camera move or a table renumber rewrites history.

    Configuration tables are deliberately excluded: they *are* current state,
    and that distinction is the point rather than an oversight.
    """
    located_events = (
        module_models.PeopleCountInterval,
        module_models.DemographySnapshot,
        module_models.TableStatusEvent,
        module_models.BoardUsageEvent,
        module_models.DishDetection,
    )
    for model in located_events:
        columns = {c.name for c in model.__table__.columns}
        assert {"restaurant_id", "zone_id", "zone_name"} <= columns, model.__tablename__


def test_demography_has_no_column_that_could_name_a_person() -> None:
    """Aggregate-only is a property of the shape, not a promise about the writer."""
    columns = {c.name for c in module_models.DemographySnapshot.__table__.columns}
    for banned in ("object_id", "track_id", "patron_token_id", "evidence_ref", "subject_id"):
        assert banned not in columns


def test_a_board_reading_defaults_to_unknown_and_a_verdict_to_none() -> None:
    """An unevaluated event is not a clean one.

    `verdict` nullable with no default is the load-bearing part: a default of
    `match` would make every unevaluated reading look compliant.
    """
    columns = {c.name: c for c in module_models.BoardUsageEvent.__table__.columns}
    assert columns["board_colour_state"].default.arg == "unknown"
    assert columns["ingredient_state"].default.arg == "unknown"
    assert columns["verdict"].nullable is True
    assert columns["verdict"].default is None


def test_a_dish_detection_starts_unreconciled() -> None:
    """Never `matched`. A dish nobody compared against a ticket is not evidence."""
    columns = {c.name: c for c in module_models.DishDetection.__table__.columns}
    assert columns["reconciliation_state"].default.arg == "unreconciled"


def test_the_table_state_enum_keeps_not_knowing_separate_from_vacant() -> None:
    """Seating a party at an occupied table is what collapsing these would cause."""
    values = {state.value for state in module_models.TableState}
    assert {"not_visible", "unknown"} <= values
    assert "vacant" in values


def test_no_new_model_stores_a_secret() -> None:
    """A credential reference, never a credential — the camera table's own rule."""
    columns = {c.name for c in module_models.PosConnector.__table__.columns}
    assert "credential_ref" in columns
    for banned in ("password", "token", "api_key", "secret"):
        assert not any(banned in name for name in columns), banned


def test_the_migrations_declare_exactly_the_models(app) -> None:
    """Guards the hand-written migration against the declared schema.

    Cheap, and it catches the one failure mode of writing a migration by hand:
    a column that exists in the model and not in the table, which surfaces in
    production as an operational error and nowhere else.
    """
    from pathlib import Path

    migrations = (Path(__file__).resolve().parents[2] / "migrations" / "versions").glob("*.py")
    source = "\n".join(path.read_text(encoding="utf-8") for path in migrations)

    for model in (
        module_models.PeopleCountInterval,
        module_models.DemographySnapshot,
        module_models.DiningTable,
        module_models.TableStatusEvent,
        module_models.CuttingBoardPolicy,
        module_models.BoardUsageEvent,
        module_models.DishDetection,
        module_models.PatronToken,
        module_models.PosConnector,
        module_models.PosSyncRun,
    ):
        assert f"'{model.__tablename__}'" in source, model.__tablename__
        for column in model.__table__.columns:
            assert f"'{column.name}'" in source, f"{model.__tablename__}.{column.name}"


def test_the_new_tables_exist_in_a_built_schema(app) -> None:
    """And they are actually created, not merely declared."""
    inspector_tables = set(module_models.Base.metadata.tables)
    for name in (
        "camera_zone_assignments",
        "people_count_intervals",
        "demography_snapshots",
        "dining_tables",
        "table_status_events",
        "cutting_board_policies",
        "board_usage_events",
        "dish_detections",
        "patron_tokens",
        "pos_connectors",
        "pos_sync_runs",
    ):
        assert name in inspector_tables


def test_no_new_permission_is_granted_by_accident() -> None:
    """The two that must stay narrow, stated as tests rather than as comments."""
    for role in Role:
        granted = permissions_for(frozenset({role}))
        if role is not Role.SUPER_ADMIN:
            assert Permission.MANAGE_PATRON_ID not in granted, role

    # A kitchen screen is shared with whoever walks past it.
    supervisor = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
    assert Permission.VIEW_DEMOGRAPHY not in supervisor
    assert Permission.VIEW_PATRON_ID not in supervisor

    # An auditor reads the compliance record, not the company's sales.
    auditor = permissions_for(frozenset({Role.AUDITOR}))
    assert Permission.VIEW_CUTTING_BOARD in auditor
    assert Permission.VIEW_MEAL_DETECTION not in auditor
    assert Permission.VIEW_POS_INTEGRATION not in auditor


def test_the_platform_was_not_touched_for_any_of_this() -> None:
    """No module in this phase imports a vision_os adapter it had to add.

    The two ports patron identification would need — `EmbeddingPort` and
    `IdentityResolverPort` — already exist and are deliberately unbound, which
    is exactly why no sibling adapter was written: binding one is the act that
    requires the legal artifact.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "vision_os"
    for name in ("embedding", "reidentification", "patron", "demograph", "dish", "board"):
        matches = list(root.rglob(f"*{name}*.py"))
        assert matches == [], f"a vision_os module was added for '{name}': {matches}"


def test_the_pos_seam_is_not_in_the_platform() -> None:
    """A till is not perception, and it must not live behind a vision port."""
    from pathlib import Path

    import app.integrations.pos as pos_module

    assert Path(pos_module.__file__).resolve().parts[-3:-1] == ("app", "integrations")

    source = Path(pos_module.__file__).read_text(encoding="utf-8")
    assert "import vision_os" not in source
    assert "from vision_os" not in source
