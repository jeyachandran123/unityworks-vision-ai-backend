"""Starting and stopping a camera from the product.

The defect these close: `LiveRuntime.start_from_records` runs once, at boot,
from the application lifespan, and nothing else ever dialled a camera. A camera
added through the product was written to the database, listed everywhere,
reported enabled — and connected to nothing until somebody restarted the
process. A working camera and a dead one looked identical.

No socket is opened here. `start_one` is replaced with a synthetic session
carrying the same identity, so the route, the tenancy and the audit trail are
exercised for real while the DVR is not.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.audit import AuditAction
from app.domain.models import AuditEvent, Camera, Zone
from app.domain.runtime_identity import runtime_camera_id
from tests.app.conftest import bearer, make_recorder, make_user

pytestmark = pytest.mark.asyncio

BASE = "/api/v1/cameras"


@pytest_asyncio.fixture
async def estate(seeded, monkeypatch):
    """The seeded organization, plus a zone and two cameras — one addressable,
    one not.

    Live CCTV is switched on here: the suite defaults it off so that importing
    the application opens no socket, and these tests are about what happens
    when a deployment has it on.
    """
    app = seeded
    monkeypatch.setattr(app.state.settings, "feature_live_cctv", True)
    # These are about the analysis session. Starting a camera also opens its
    # Live Wall stream, which would dial a real socket from a test; that half
    # is covered, with the wall replaced, in `test_camera_wall_lifecycle.py`.
    monkeypatch.setattr(app.state.settings, "feature_camera_wall", False)
    async with app.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
        session.add(Zone(id="zone-kitchen", organization_id="org-test", name="Kitchen"))
        # The "not addressable" case lives on the recorder now: a box whose
        # address was never filled in, with a camera plugged into it.
        session.add(make_recorder("org-test", recorder_id="rec-no-address", host=""))
        session.add(
            Camera(
                id="cam-row-1",
                organization_id="org-test",
                zone_id="zone-kitchen",
                camera_key="cam-01",
                name="Prep line",
                recorder_id="rec-org-test",
                channel=1,
                stream_type="main",
                enabled=False,
            )
        )
        session.add(
            Camera(
                id="cam-row-2",
                organization_id="org-test",
                zone_id="zone-kitchen",
                camera_key="cam-02",
                name="No address",
                recorder_id="rec-no-address",
                channel=2,
                stream_type="main",
                enabled=False,
            )
        )
    return app


@pytest_asyncio.fixture
async def dialled(estate, monkeypatch):
    """`start_one` opens a synthetic session instead of an RTSP one."""
    live = estate.state.live
    calls: list[tuple[str, str]] = []

    async def _start_one(config, *, tenant_id: str):
        calls.append((config.camera_id, tenant_id))
        return await live.start_synthetic(
            camera_id=config.camera_id, tenant_id=tenant_id, fps=1.0, count=1
        )

    monkeypatch.setattr(live, "start_one", _start_one)
    yield calls
    await live.stop_all()


async def test_starting_a_camera_opens_a_session_for_its_organization(
    dialled, estate, client: AsyncClient
):
    headers = await bearer(client, "admin@example.com")
    started = await client.post(f"{BASE}/cam-01/start", headers=headers)

    assert started.status_code == 200, started.text
    assert started.json() == {
        "camera_key": "cam-01",
        "streaming": True,
        "available": True,
        "reason": "",
    }
    # Filed under the camera's own organization, not the deployment's default:
    # `visible()` filters on it, so a session under the wrong tenant is
    # invisible to the people who own the camera.
    assert dialled == [(runtime_camera_id("org-test", "cam-01"), "org-test")]
    assert estate.state.live.visible(tenant_id="org-test", camera_ids=None)


async def test_starting_records_the_decision_so_a_restart_keeps_it(
    dialled, estate, client: AsyncClient
):
    headers = await bearer(client, "admin@example.com")
    await client.post(f"{BASE}/cam-01/start", headers=headers)

    async with estate.state.database.session_scope() as session:
        camera = (
            await session.execute(select(Camera).where(Camera.camera_key == "cam-01"))
        ).scalar_one()
        assert camera.enabled is True


async def test_stopping_closes_the_session_and_clears_the_decision(
    dialled, estate, client: AsyncClient
):
    headers = await bearer(client, "admin@example.com")
    await client.post(f"{BASE}/cam-01/start", headers=headers)

    stopped = await client.post(f"{BASE}/cam-01/stop", headers=headers)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["streaming"] is False
    assert estate.state.live.visible(tenant_id="org-test", camera_ids=None) == []

    async with estate.state.database.session_scope() as session:
        camera = (
            await session.execute(select(Camera).where(Camera.camera_key == "cam-01"))
        ).scalar_one()
        assert camera.enabled is False


async def test_stopping_something_already_stopped_is_the_state_asked_for(
    dialled, client: AsyncClient
):
    headers = await bearer(client, "admin@example.com")
    stopped = await client.post(f"{BASE}/cam-01/stop", headers=headers)
    assert stopped.status_code == 200, stopped.text
    assert stopped.json()["streaming"] is False
    assert "was not streaming" in stopped.json()["reason"]


async def test_live_cctv_switched_off_says_so_rather_than_doing_nothing(
    dialled, estate, client: AsyncClient, monkeypatch
):
    """A dead button is the failure this answer exists to prevent."""
    monkeypatch.setattr(estate.state.settings, "feature_live_cctv", False)
    headers = await bearer(client, "admin@example.com")

    started = await client.post(f"{BASE}/cam-01/start", headers=headers)
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["available"] is False
    assert body["streaming"] is False
    assert "FEATURE_LIVE_CCTV" in body["reason"]
    assert dialled == []


async def test_a_camera_with_no_address_explains_itself(dialled, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    started = await client.post(f"{BASE}/cam-02/start", headers=headers)

    assert started.status_code == 200, started.text
    assert started.json()["available"] is False
    assert "no address" in started.json()["reason"]


async def test_starting_needs_manage_cameras(dialled, client: AsyncClient):
    """`kitchen_supervisor` may watch a camera and may not dial one."""
    headers = await bearer(client, "supervisor@example.com")
    refused = await client.post(f"{BASE}/cam-01/start", headers=headers)
    assert refused.status_code == 403, refused.text


async def test_another_organizations_camera_is_not_found(dialled, client: AsyncClient):
    """404, never 403: existence across a tenant boundary is a disclosure."""
    headers = await bearer(client, "outsider@example.com")
    refused = await client.post(f"{BASE}/cam-01/start", headers=headers)
    assert refused.status_code == 404, refused.text


async def test_starting_and_stopping_are_audited(dialled, estate, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    await client.post(f"{BASE}/cam-01/start", headers=headers)
    await client.post(f"{BASE}/cam-01/stop", headers=headers)

    async with estate.state.database.session_scope() as session:
        rows = (
            await session.execute(
                select(AuditEvent.action, AuditEvent.resource_id).where(
                    AuditEvent.organization_id == "org-test"
                )
            )
        ).all()
    assert (AuditAction.CAMERA_STARTED.value, "cam-01") in rows
    assert (AuditAction.CAMERA_STOPPED.value, "cam-01") in rows


async def test_the_next_key_does_not_collide(dialled, client: AsyncClient):
    headers = await bearer(client, "admin@example.com")
    proposed = await client.get(f"{BASE}/next-key", headers=headers)
    assert proposed.status_code == 200, proposed.text
    # cam-01 and cam-02 exist.
    assert proposed.json() == {"camera_key": "cam-03"}
