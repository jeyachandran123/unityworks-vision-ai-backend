"""Start watching puts the camera on the Live Wall; Stop takes it off.

The defect this closes: the wall opened its streams in exactly two places — at
boot, and when a recorder was activated. "Start watching" opened the analysis
session only. A camera added and started after the server came up was
analysed, streaming and healthy, while its Live Wall tile said OFFLINE with no
error, until somebody restarted the process. Found on 2026-09-24 with a camera
added through the product in a newly created organization.

The wall is replaced by a recorder of calls: what matters here is that the
routes ask it to open and close the right stream, for the right organization.
The wall's own streaming is covered in `test_camera_wall.py`.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.domain.models import Camera, Zone
from app.domain.runtime_identity import runtime_camera_id
from tests.app.conftest import bearer, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/cameras"
RUNTIME_ID = runtime_camera_id("org-test", "cam-41")


class _WallCalls:
    """Stands in for `CameraWall`, recording what the routes asked of it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def start_cameras(self, cameras) -> int:
        self.calls.append(
            ("start", [(c.camera_key, c.organization_id, c.enabled) for c in cameras])
        )
        return len(cameras)

    async def stop_cameras(self, runtime_ids) -> int:
        self.calls.append(("stop", list(runtime_ids)))
        return len(runtime_ids)

    async def stop_all(self) -> None:  # the lifespan's shutdown
        return None


@pytest_asyncio.fixture
async def estate(seeded, monkeypatch):
    app = seeded
    monkeypatch.setattr(app.state.settings, "feature_live_cctv", True)
    monkeypatch.setattr(app.state.settings, "feature_camera_wall", True)
    async with app.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
        session.add(Zone(id="zone-kitchen", organization_id="org-test", name="Kitchen"))
        session.add(
            Camera(
                id="cam-row-41",
                organization_id="org-test",
                zone_id="zone-kitchen",
                camera_key="cam-41",
                name="Kitchen center",
                recorder_id="rec-org-test",
                channel=13,
                stream_type="main",
                enabled=False,
            )
        )

    live = app.state.live

    async def _start_one(config, *, tenant_id: str):
        return await live.start_synthetic(
            camera_id=config.camera_id, tenant_id=tenant_id, fps=1.0, count=1
        )

    monkeypatch.setattr(live, "start_one", _start_one)
    wall = _WallCalls()
    monkeypatch.setattr(app.state, "wall", wall)
    yield app, wall
    await live.stop_all()


async def test_start_watching_opens_the_live_wall_stream(estate, client: AsyncClient):
    _, wall = estate
    headers = await bearer(client, "admin@example.com")

    started = await client.post(f"{BASE}/cam-41/start", headers=headers)
    assert started.status_code == 200, started.text
    assert started.json()["streaming"] is True

    # Any dark tile the wall held for it — a camera that was switched off at
    # boot has one — is replaced, then the stream opens, switched on.
    assert wall.calls == [
        ("stop", [RUNTIME_ID]),
        ("start", [("cam-41", "org-test", True)]),
    ]


async def test_stop_takes_it_off_the_live_wall(estate, client: AsyncClient):
    _, wall = estate
    headers = await bearer(client, "admin@example.com")
    await client.post(f"{BASE}/cam-41/start", headers=headers)
    wall.calls.clear()

    stopped = await client.post(f"{BASE}/cam-41/stop", headers=headers)
    assert stopped.status_code == 200, stopped.text
    assert wall.calls == [("stop", [RUNTIME_ID])]


async def test_a_deployment_without_the_wall_is_not_asked_to_open_one(
    estate, client: AsyncClient, monkeypatch
):
    app, wall = estate
    monkeypatch.setattr(app.state.settings, "feature_camera_wall", False)
    headers = await bearer(client, "admin@example.com")

    started = await client.post(f"{BASE}/cam-41/start", headers=headers)
    assert started.status_code == 200, started.text
    assert wall.calls == []
