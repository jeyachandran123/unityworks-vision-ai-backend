"""Each camera's Live Wall frame rate is its own setting, and editing works.

The wall played every camera at one rate fixed in code (`DEFAULT_WALL_FPS`), so
every "make it smoother" was a code change and a restart. The rate is now a
value on each camera: chosen when it is added, changed when it is edited, and
handed to the Live Wall with the camera.

Editing a running camera's channel, stream or recorder reconnects it at once —
a camera left dialling its old channel after an edit would show one view under
another view's name.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.models import AuditEvent, Camera, Zone
from app.domain.runtime_identity import runtime_camera_id
from tests.app.conftest import bearer, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/cameras"


@pytest_asyncio.fixture
async def estate(seeded):
    async with seeded.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
        session.add(Zone(id="zone-kitchen", organization_id="org-test", name="Kitchen"))
        session.add(Zone(id="zone-hall", organization_id="org-test", name="Hall"))
    return seeded


async def _admin(client: AsyncClient) -> dict[str, str]:
    return await bearer(client, "admin@example.com")


def _camera(**overrides) -> dict:
    body = {
        "camera_key": "cam-51",
        "name": "Serving line",
        "recorder_id": "rec-org-test",
        "channel": 5,
        "zone_id": "zone-kitchen",
    }
    body.update(overrides)
    return body


# ── chosen when added ────────────────────────────────────────────────────────


async def test_a_camera_is_added_with_its_own_frame_rate(estate, client: AsyncClient):
    response = await client.post(BASE, json=_camera(wall_fps=12), headers=await _admin(client))
    assert response.status_code == 200, response.text
    assert response.json()["wall_fps"] == 12


async def test_without_a_choice_it_plays_at_the_standard_rate(estate, client: AsyncClient):
    response = await client.post(BASE, json=_camera(), headers=await _admin(client))
    assert response.status_code == 200, response.text
    assert response.json()["wall_fps"] == 4


@pytest.mark.parametrize("value", [0, 0.5, 26, 100, "fast", True, None, [4]])
async def test_a_rate_outside_one_to_twenty_five_is_refused(estate, client: AsyncClient, value):
    response = await client.post(BASE, json=_camera(wall_fps=value), headers=await _admin(client))
    assert response.status_code == 422, (value, response.text)
    assert "frame rate" in response.json()["message"]


async def test_the_recorder_flow_sets_each_cameras_rate(estate, client: AsyncClient):
    response = await client.post(
        "/api/v1/recorders",
        json={
            "name": "Canteen DVR",
            "ip_address": "192.168.54.243",
            "rtsp_port": 554,
            "username": "admin",
            "password": "Canteen-Pass-1",
            "brand": "dahua",
            "cameras": [
                {"channel": 1, "name": "Kitchen", "zone_id": "zone-kitchen", "wall_fps": 15},
                {"channel": 2, "name": "Hall", "zone_id": "zone-hall"},
            ],
        },
        headers=await _admin(client),
    )
    assert response.status_code == 200, response.text
    assert [c["wall_fps"] for c in response.json()["cameras"]] == [15, 4]


# ── changed when edited ──────────────────────────────────────────────────────


async def test_editing_changes_the_rate_and_is_audited(estate, client: AsyncClient):
    headers = await _admin(client)
    await client.post(BASE, json=_camera(), headers=headers)

    response = await client.patch(f"{BASE}/cam-51", json={"wall_fps": 20}, headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["wall_fps"] == 20
    assert response.json()["reconnected"] is False

    async with estate.state.database.session_scope() as session:
        audit = (
            (
                await session.execute(
                    select(AuditEvent).where(
                        AuditEvent.action == "camera.updated", AuditEvent.resource_id == "cam-51"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert audit and "wall_fps" in audit[-1].detail


async def test_an_edit_with_a_bad_rate_changes_nothing(estate, client: AsyncClient):
    headers = await _admin(client)
    await client.post(BASE, json=_camera(wall_fps=8), headers=headers)

    response = await client.patch(
        f"{BASE}/cam-51", json={"wall_fps": 60, "name": "Renamed"}, headers=headers
    )
    assert response.status_code == 422
    listing = (await client.get(BASE, headers=headers)).json()["cameras"]
    mine = next(c for c in listing if c["camera_key"] == "cam-51")
    assert (mine["wall_fps"], mine["name"]) == (8, "Serving line")


# ── handed to the Live Wall ──────────────────────────────────────────────────


async def test_the_live_wall_receives_each_cameras_own_rate(estate, client: AsyncClient):
    headers = await _admin(client)
    await client.post(BASE, json=_camera(wall_fps=15), headers=headers)
    await client.post(
        BASE, json=_camera(camera_key="cam-52", channel=6, name="Hall"), headers=headers
    )

    wall = (await client.get("/api/v1/wall/cameras", headers=headers)).json()
    rates = {c["camera_id"]: c["wall_fps"] for c in wall["cameras"]}
    assert rates["cam-51"] == 15
    assert rates["cam-52"] == 4


# ── editing a running camera reconnects it ───────────────────────────────────


class _Calls:
    def __init__(self) -> None:
        self.calls: list[tuple] = []


@pytest_asyncio.fixture
async def running(estate, monkeypatch):
    """`cam-51` on, with the live runtime and the wall recording what they are asked."""
    app = estate
    monkeypatch.setattr(app.state.settings, "feature_live_cctv", True)
    monkeypatch.setattr(app.state.settings, "feature_camera_wall", True)
    async with app.state.database.session_scope() as session:
        session.add(
            Camera(
                id="row-51",
                organization_id="org-test",
                zone_id="zone-kitchen",
                recorder_id="rec-org-test",
                camera_key="cam-51",
                name="Serving line",
                channel=5,
                stream_type="sub",
                enabled=True,
            )
        )
    record = _Calls()
    live = app.state.live

    async def stop_camera(camera_id, *, tenant_id):
        record.calls.append(("live.stop", camera_id))
        return True

    async def start_one(config, *, tenant_id):
        record.calls.append(("live.start", config.camera_id, config.channel, config.stream_type))

    class _Wall:
        async def start_cameras(self, cameras):
            record.calls.append(("wall.start", [(c.camera_key, c.channel) for c in cameras]))
            return len(cameras)

        async def stop_cameras(self, runtime_ids):
            record.calls.append(("wall.stop", list(runtime_ids)))
            return len(runtime_ids)

        async def stop_all(self):
            return None

    monkeypatch.setattr(live, "stop_camera", stop_camera)
    monkeypatch.setattr(live, "start_one", start_one)
    monkeypatch.setattr(app.state, "wall", _Wall())
    return record


async def test_changing_a_running_cameras_channel_reconnects_it(running, client: AsyncClient):
    headers = await _admin(client)
    rid = runtime_camera_id("org-test", "cam-51")

    response = await client.patch(
        f"{BASE}/cam-51", json={"channel": 7, "stream_type": "main"}, headers=headers
    )

    assert response.status_code == 200, response.text
    assert response.json()["reconnected"] is True
    assert running.calls == [
        ("live.stop", rid),
        ("live.start", rid, 7, "main"),
        ("wall.stop", [rid]),
        ("wall.start", [("cam-51", 7)]),
    ]


async def test_renaming_or_a_new_rate_does_not_reconnect(running, client: AsyncClient):
    headers = await _admin(client)
    response = await client.patch(
        f"{BASE}/cam-51", json={"name": "Pass", "wall_fps": 12}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["reconnected"] is False
    assert running.calls == []


# ── the column itself ────────────────────────────────────────────────────────


def test_existing_cameras_keep_todays_rate_after_the_upgrade(tmp_path: Path) -> None:
    from .test_recorder_migration import _alembic

    database = tmp_path / "rate.db"
    assert _alembic(database, "upgrade", "e5a1c7d9f302").returncode == 0
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            INSERT INTO organizations (id, name, slug, is_active, timezone, status,
                                       status_reason, created_at)
            VALUES ('org-a', 'A', 'a', 1, 'UTC', 'active', '', CURRENT_TIMESTAMP);
            INSERT INTO zones (id, organization_id, name, is_active, created_at)
            VALUES ('z', 'org-a', 'Kitchen', 1, CURRENT_TIMESTAMP);
            INSERT INTO recorders (id, organization_id, name, ip_address, hostname, connect_via,
                                   rtsp_port, username, credential_ref, secret_key_id, brand,
                                   path_template, is_active, created_at, updated_at)
            VALUES ('r', 'org-a', 'DVR', '10.0.0.5', '', 'ip_address', 554, 'admin',
                    'recorder:r', 'k1', 'dahua', '', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            INSERT INTO cameras (id, organization_id, zone_id, recorder_id, camera_key, name,
                                 purpose, channel, stream_type, analysis_fps, enabled,
                                 analysis_enabled, created_at, updated_at)
            VALUES ('c', 'org-a', 'z', 'r', 'cam-01', 'Kitchen', '', 1, 'sub', 4.0, 1, 1,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """
        )
        connection.commit()
    finally:
        connection.close()

    result = _alembic(database, "upgrade", "head")
    assert result.returncode == 0, result.stdout + result.stderr
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT wall_fps FROM cameras").fetchall() == [(4.0,)]
    finally:
        connection.close()
