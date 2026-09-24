"""The connection test tells a person which thing is wrong.

Each outcome here is a different fix — the password, the channel, the address,
the port, the camera itself — so each must be told apart, and none may echo the
password or the decoder's own text back to the screen.

The network is replaced at the two seams the real test uses: the TCP probe
(`connectivity.probe`) and the frame reader. Everything between them — URL
construction, brand paths, error mapping — is the production code.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.domain import connectivity, recorder_probe
from app.domain.audit import AuditAction
from app.domain.connectivity import ConnectivityResult
from app.domain.models import AuditEvent
from app.domain.recorder_brands import load_brands
from app.domain.recorder_probe import check_connection
from app.vision.sources.rtsp import RtspAuthenticationError, RtspStreamNotFoundError

from .conftest import bearer, make_user

PASSWORD = "Dvr@Gayathri#2026"


@pytest.fixture
def reachable(monkeypatch):
    async def probe(host, port, **_):
        return ConnectivityResult(True, "reachable", f"A device answered on {host}:{port}.", 3)

    monkeypatch.setattr(connectivity, "probe", probe)


def _unreachable(monkeypatch, outcome: str) -> None:
    async def probe(host, port, **_):
        return ConnectivityResult(False, outcome, "no", 4000)

    monkeypatch.setattr(connectivity, "probe", probe)


async def _check(reader, **overrides):
    arguments = {
        "host": "192.168.1.20",
        "port": 554,
        "username": "admin",
        "password": PASSWORD,
        "brand": load_brands()["hikvision"],
        "channel": 2,
        "stream_type": "sub",
        "reader": reader,
    }
    arguments.update(overrides)
    return await check_connection(**arguments)


# ── outcomes ─────────────────────────────────────────────────────────────────


async def test_a_picture_arriving_is_a_success_and_says_its_size(reachable):
    seen: list[str] = []

    def reader(uri: str):
        seen.append(uri)
        return 1920, 1080

    result = await _check(reader)
    body = result.as_dict()

    assert body["ok"] is True
    assert body["message"] == "Recorder connected successfully."
    assert body["resolution"] == "1920 × 1080"
    # The URL the runtime would dial for this brand and channel — built by the
    # runtime's own config, so a passing test means a starting camera.
    assert seen == ["rtsp://admin:Dvr%40Gayathri%232026@192.168.1.20:554/Streaming/Channels/202"]


async def test_a_rejected_password_says_so(reachable):
    def reader(uri: str):
        raise RtspAuthenticationError("the DVR rejected these credentials")

    body = (await _check(reader)).as_dict()
    assert body["outcome"] == "authentication_failed"
    assert body["message"] == "The recorder rejected the username or password."


async def test_a_missing_channel_says_so(reachable):
    def reader(uri: str):
        raise RtspStreamNotFoundError("no stream at this path")

    body = (await _check(reader)).as_dict()
    assert body["outcome"] == "channel_not_found"
    assert "channel" in body["message"]


async def test_no_picture_in_time_is_its_own_answer(reachable):
    import time

    def reader(uri: str):
        time.sleep(0.5)
        return 1, 1

    body = (await _check(reader, frame_timeout_s=0.05)).as_dict()
    assert body["outcome"] == "stream_unavailable"


@pytest.mark.parametrize(
    ("tcp", "outcome"),
    [
        ("timeout", "timeout"),
        ("dns_timeout", "timeout"),
        ("dns_failed", "address_not_found"),
        ("refused", "refused"),
        ("unreachable", "unreachable"),
    ],
)
async def test_network_failures_are_told_apart_before_anything_is_dialled(
    monkeypatch, tcp: str, outcome: str
):
    _unreachable(monkeypatch, tcp)

    def reader(uri: str):  # pragma: no cover - must not be reached
        raise AssertionError("dialled a recorder that did not answer")

    body = (await _check(reader)).as_dict()
    assert body["outcome"] == outcome
    assert body["ok"] is False
    assert body["hint"]


async def test_an_unexpected_decoder_error_is_logged_scrubbed_and_not_shown(reachable, monkeypatch):
    """Decoders quote the URL they failed to open. That text goes to the log with
    the password removed, and never to the person."""
    logged: list[str] = []
    monkeypatch.setattr(
        recorder_probe.logger,
        "warning",
        lambda message, *args: logged.append(message.format(*args)),
    )

    def reader(uri: str):
        raise OSError(f"could not open {uri}: broken pipe")

    body = (await _check(reader)).as_dict()

    assert body["outcome"] == "stream_unavailable"
    assert "broken pipe" not in str(body)
    assert logged and "broken pipe" in logged[0]
    assert PASSWORD not in logged[0]
    assert "Dvr%40Gayathri" not in logged[0]


async def test_a_server_without_video_support_says_so(reachable):
    def reader(uri: str):
        raise RuntimeError("live RTSP requires a decoder; install the 'video' extra (PyAV)")

    body = (await _check(reader)).as_dict()
    assert body["outcome"] == "decoder_unavailable"


async def test_a_loopback_address_is_refused_outright():
    """A connection test is an outbound connection from the server. Loopback is
    the server itself, and never a camera."""
    from app.errors import ValidationError

    with pytest.raises(ValidationError):
        await _check(lambda uri: (1, 1), host="127.0.0.1")


# ── over HTTP ────────────────────────────────────────────────────────────────


@pytest.fixture
async def admin(seeded):
    async with seeded.state.database.session_scope() as session:
        _, user = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(user)
    return seeded


@pytest.fixture
def fake_reader(monkeypatch):
    """The production reader, replaced; returns whatever the test sets."""
    state = {"result": (1280, 720), "seen": []}

    def reader(uri: str):
        state["seen"].append(uri)
        result = state["result"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(recorder_probe, "_first_frame_with_pyav", reader)
    return state


async def test_an_unsaved_recorder_can_be_tested_with_the_password_just_typed(
    admin, client: AsyncClient, reachable, fake_reader
):
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        "/api/v1/recorders/test",
        json={
            "host": "192.168.1.20",
            "rtsp_port": 554,
            "username": "admin",
            "password": PASSWORD,
            "brand": "dahua",
            "channel": 4,
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert PASSWORD not in response.text
    assert fake_reader["seen"][0].endswith("/cam/realmonitor?channel=4&subtype=1")


async def test_a_saved_recorder_is_tested_with_its_stored_password(
    admin, client: AsyncClient, reachable, fake_reader
):
    headers = await bearer(client, "admin@example.com")
    created = (
        await client.post(
            "/api/v1/recorders",
            json={
                "name": "Gayathri DVR",
                "host": "192.168.1.20",
                "rtsp_port": 554,
                "username": "admin",
                "password": PASSWORD,
                "brand": "hikvision",
            },
            headers=headers,
        )
    ).json()

    fake_reader["result"] = RtspAuthenticationError("401")
    response = await client.post(
        f"/api/v1/recorders/{created['id']}/test", json={"channel": 1}, headers=headers
    )

    body = response.json()
    assert body["outcome"] == "authentication_failed"
    # The stored password, unsealed, was what the reader was handed.
    assert "Dvr%40Gayathri%232026" in fake_reader["seen"][0]
    assert PASSWORD not in response.text


async def test_testing_twice_at_once_is_throttled(
    admin, client: AsyncClient, reachable, fake_reader
):
    """An unthrottled test button is a way to lock the DVR account the cameras
    use."""
    headers = await bearer(client, "admin@example.com")
    body = {
        "host": "192.168.1.30",
        "rtsp_port": 554,
        "username": "admin",
        "password": PASSWORD,
        "brand": "dahua",
    }
    first = await client.post("/api/v1/recorders/test", json=body, headers=headers)
    second = await client.post("/api/v1/recorders/test", json=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "seconds" in second.json()["message"]


async def test_every_test_is_audited_with_its_outcome_and_no_password(
    admin, client: AsyncClient, reachable, fake_reader
):
    headers = await bearer(client, "admin@example.com")
    await client.post(
        "/api/v1/recorders/test",
        json={
            "host": "192.168.1.40",
            "username": "admin",
            "password": PASSWORD,
            "brand": "axis",
        },
        headers=headers,
    )
    async with admin.state.database.session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(AuditEvent).where(AuditEvent.action == AuditAction.RECORDER_TESTED.value)
                )
            )
            .scalars()
            .all()
        )
    assert rows
    assert "connected" in rows[-1].detail
    assert PASSWORD not in rows[-1].detail


async def test_a_camera_test_uses_its_recorder_and_channel(
    admin, client: AsyncClient, reachable, fake_reader
):
    """`POST /cameras/test-connection` with a camera key is the real test now,
    through the camera's own recorder."""
    from app.domain.models import Zone

    async with admin.state.database.session_scope() as session:
        session.add(Zone(id="zone-k", organization_id="org-test", name="Kitchen"))
    headers = await bearer(client, "admin@example.com")
    created = (
        await client.post(
            "/api/v1/recorders",
            json={
                "name": "Gayathri DVR",
                "host": "192.168.1.20",
                "username": "admin",
                "password": PASSWORD,
                "brand": "hikvision",
                "rtsp_port": 554,
                "cameras": [{"channel": 7, "name": "Store", "zone_id": "zone-k"}],
            },
            headers=headers,
        )
    ).json()

    response = await client.post(
        "/api/v1/cameras/test-connection",
        json={"camera_key": created["cameras"][0]["camera_key"]},
        headers=headers,
    )
    body = response.json()
    assert body["ok"] is True
    assert fake_reader["seen"][-1].endswith("/Streaming/Channels/702")
    assert PASSWORD not in response.text
