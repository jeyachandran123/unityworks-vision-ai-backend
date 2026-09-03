"""`cameras.analysis_enabled` — streaming and analysis are two decisions.

### What these pin

One flag was doing two jobs. `enabled` started a camera's RTSP session *and*
enrolled it in perception, so a site could not watch a corridor on the wall
without also paying detection, tracking, cropping and the understander's call
budget for it — and that budget is a single global allowance the kitchens
already spend in full.

The column was added by migration `d4a1c8e37b52`. A later revert removed the
application half while leaving the migration and the database column in place,
so for a period the column existed in Postgres, was reported as drift by
`alembic check`, and was read by nothing: every enabled camera was analysed
regardless of what the row said. These tests exist so that gap cannot reopen
silently.

The last class is the one that matters. Everything before it could be satisfied
by a column nobody consults; `TestAnalysisScheduling` drives the real bootstrap
path that decides which cameras get a perception session, and asserts the flag
actually governs it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio

from app import main as app_main
from app.domain.cameras import CameraService
from app.domain.cameras import to_wire as camera_to_wire
from app.domain.models import Camera
from app.errors import ValidationError
from tests.app.conftest import bearer, make_user

ORG = "org-test"


@pytest_asyncio.fixture
async def session(app):
    async with app.state.database.session_scope() as active:
        yield active


@pytest_asyncio.fixture
async def admin_headers(seeded, client):
    """An `org_admin`; `developer` deliberately lacks `MANAGE_CAMERAS`."""
    async with seeded.state.database.session_scope() as active:
        _, admin = make_user(
            email="cam-admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        active.add(admin)
    return await bearer(client, "cam-admin@example.com")


async def _camera(session, key: str, *, channel: int, enabled: bool, analysed: bool | None):
    service = CameraService(session)
    camera = await service.create(
        organization_id=ORG,
        restaurant_id="rest-01",
        camera_key=key,
        name=f"Channel {channel}",
        channel=channel,
        host="10.0.0.5",
    )
    camera.enabled = enabled
    if analysed is not None:
        camera.analysis_enabled = analysed
    await session.flush()
    return camera


# ── schema and defaults ──────────────────────────────────────────────────────


class TestSchema:
    def test_the_column_is_mapped_not_merely_migrated(self):
        """The defect this file exists for: the column present in Postgres and
        absent from the ORM, so `alembic check` reported drift and the runtime
        read nothing."""
        assert "analysis_enabled" in Camera.__table__.columns

    def test_it_is_not_nullable_so_there_is_no_third_state(self):
        """`NULL` would be a third state every reader would have to guess at."""
        assert Camera.__table__.columns["analysis_enabled"].nullable is False

    def test_the_backfill_is_true_so_existing_rows_keep_their_behaviour(self):
        """A row that predates the column meant "analysed" — the migration and
        the model must agree, or a redeploy changes what a site is doing."""
        column = Camera.__table__.columns["analysis_enabled"]
        assert column.default.arg is True
        assert column.server_default is not None

    @pytest.mark.asyncio
    async def test_a_new_camera_is_analysable_but_not_yet_streaming(self, session):
        """Deterministic, and safe: the gate that stops a new camera processing
        video of people is `enabled`, which is still false."""
        camera = await _camera(session, "cam-51", channel=51, enabled=False, analysed=None)
        assert camera.analysis_enabled is True
        assert camera.enabled is False


# ── API ──────────────────────────────────────────────────────────────────────


class TestApi:
    @pytest.mark.asyncio
    async def test_the_wire_says_whether_a_camera_is_analysed(self, session):
        """A client showing only `enabled` cannot explain why a visibly live
        camera raises nothing."""
        camera = await _camera(session, "cam-52", channel=52, enabled=True, analysed=False)
        wire = camera_to_wire(camera)
        assert wire["analysis_enabled"] is False
        assert wire["enabled"] is True

    @pytest.mark.asyncio
    async def test_it_can_be_changed_by_an_operator(self, seeded, client, admin_headers):
        async with seeded.state.database.session_scope() as active:
            await _camera(active, "cam-53", channel=53, enabled=True, analysed=True)

        response = await client.patch(
            "/api/v1/cameras/cam-53",
            json={"analysis_enabled": False},
            headers=admin_headers,
        )
        assert response.status_code == 200
        assert response.json()["analysis_enabled"] is False
        # Narrowing analysis must not also take the camera off the wall.
        assert response.json()["enabled"] is True

    @pytest.mark.asyncio
    async def test_changing_it_requires_camera_authority(self, seeded, client):
        """`developer` holds DevTools access, which is not operator authority."""
        async with seeded.state.database.session_scope() as active:
            await _camera(active, "cam-54", channel=54, enabled=True, analysed=True)
            _, dev = make_user(
                email="dev-only@example.com",
                roles=("developer",),
                camera_breadth="all_in_tenant",
                camera_ids="",
            )
            active.add(dev)

        headers = await bearer(client, "dev-only@example.com")
        response = await client.patch(
            "/api/v1/cameras/cam-54",
            json={"analysis_enabled": False},
            headers=headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_a_non_boolean_is_refused_rather_than_coerced(self, session):
        """`"false"` is truthy in Python. Coercing it would switch analysis ON
        while the operator believed they had switched it off."""
        await _camera(session, "cam-55", channel=55, enabled=True, analysed=True)
        with pytest.raises(ValidationError):
            await CameraService(session).update(
                organization_id=ORG, camera_key="cam-55", analysis_enabled="false"
            )

    @pytest.mark.asyncio
    async def test_omitting_it_leaves_it_alone(self, session):
        """A client that PATCHes an unrelated field must not silently stop
        analysing the camera."""
        await _camera(session, "cam-56", channel=56, enabled=True, analysed=True)
        camera = await CameraService(session).update(
            organization_id=ORG, camera_key="cam-56", name="Renamed"
        )
        assert camera.analysis_enabled is True


# ── the boundary that actually schedules analysis ────────────────────────────


class _Live:
    """Stands in for `LiveRuntime`, recording what it was asked to analyse."""

    def __init__(self) -> None:
        self.started: list[str] = []

    async def start_from_records(self, configs) -> int:
        self.started = [str(c.camera_id) for c in configs]
        return len(self.started)


class TestAnalysisScheduling:
    """The real integration boundary: `_start_cameras_from_database` is the
    only thing that opens a perception session, and it is called from the
    application lifespan and nowhere else."""

    @pytest_asyncio.fixture
    async def estate(self, app):
        """Four enabled cameras; two of them analysed. The shape of a site with
        kitchens and corridors."""
        async with app.state.database.session_scope() as active:
            await _camera(active, "cam-61", channel=61, enabled=True, analysed=True)
            await _camera(active, "cam-62", channel=62, enabled=True, analysed=True)
            await _camera(active, "cam-63", channel=63, enabled=True, analysed=False)
            await _camera(active, "cam-64", channel=64, enabled=True, analysed=False)
        return app

    @pytest.mark.asyncio
    async def test_only_analysed_cameras_get_a_perception_session(self, estate):
        live = _Live()
        holder = SimpleNamespace(
            state=SimpleNamespace(
                settings=estate.state.settings,
                database=estate.state.database,
                live=live,
            )
        )
        holder.state.settings.default_tenant_id = ORG

        started = await app_main._start_cameras_from_database(holder)

        assert sorted(live.started) == ["cam-61", "cam-62"]
        assert started == 2

    @pytest.mark.asyncio
    async def test_a_camera_off_analysis_is_suppressed_not_merely_unlisted(
        self, estate
    ):
        """The corridors are enabled and streaming. They must reach the wall and
        not the model — the whole point of separating the two flags."""
        live = _Live()
        holder = SimpleNamespace(
            state=SimpleNamespace(
                settings=estate.state.settings,
                database=estate.state.database,
                live=live,
            )
        )
        holder.state.settings.default_tenant_id = ORG

        await app_main._start_cameras_from_database(holder)

        assert "cam-63" not in live.started
        assert "cam-64" not in live.started
        # …and they are still enabled rows, so the wall will start them.
        async with estate.state.database.session_scope() as active:
            rows = await CameraService(active).enabled_for_runtime(organization_id=ORG)
            assert {"cam-63", "cam-64"} <= {r.camera_key for r in rows}

    @pytest.mark.asyncio
    async def test_turning_analysis_off_removes_a_camera_from_the_next_start(
        self, estate
    ):
        """The durable decision governs. This is the contract the runtime
        actually offers: the row is read at start, so a change takes effect on
        the next start rather than on the running process."""
        async with estate.state.database.session_scope() as active:
            await CameraService(active).update(
                organization_id=ORG, camera_key="cam-61", analysis_enabled=False
            )

        live = _Live()
        holder = SimpleNamespace(
            state=SimpleNamespace(
                settings=estate.state.settings,
                database=estate.state.database,
                live=live,
            )
        )
        holder.state.settings.default_tenant_id = ORG

        await app_main._start_cameras_from_database(holder)

        assert live.started == ["cam-62"]

    @pytest.mark.asyncio
    async def test_no_analysed_camera_is_reported_as_zero_not_as_a_read_failure(
        self, app
    ):
        """`0` and `None` are different facts. A site that has narrowed analysis
        to nothing has not failed to read its roster, and the bootstrap
        supervisor must not retry forever as though it had."""
        async with app.state.database.session_scope() as active:
            await _camera(active, "cam-71", channel=71, enabled=True, analysed=False)

        live = _Live()
        holder = SimpleNamespace(
            state=SimpleNamespace(
                settings=app.state.settings, database=app.state.database, live=live
            )
        )
        holder.state.settings.default_tenant_id = ORG

        started = await app_main._start_cameras_from_database(holder)

        assert started == 0
        assert live.started == []
