"""The connection test tells a person which thing is wrong.

Each outcome here is a different fix — the password, the channel, the address,
the port, the camera itself — so each must be told apart, and none may echo the
password or the decoder's own text back to the screen.

The network is replaced at the three seams the real test uses: the TCP probe
(`connectivity.probe`), the unauthenticated RTSP handshake and the frame
reader. Everything between them — URL construction, brand paths, error
mapping — is the production code. The handshake itself is also run for real,
against servers on this machine that answer as RTSP, as HTTP, or not at all.
"""

from __future__ import annotations

import asyncio

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
    """Something listens, and it speaks RTSP: the recorder is on the right port."""

    async def probe(host, port, **_):
        return ConnectivityResult(True, "reachable", f"A device answered on {host}:{port}.", 3)

    async def handshake(host, port, **_):
        return "rtsp"

    monkeypatch.setattr(connectivity, "probe", probe)
    monkeypatch.setattr(recorder_probe, "_rtsp_handshake", handshake)


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


# ── the wrong port ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("spoken", ["silent", "other"])
async def test_a_port_that_is_not_a_video_port_is_told_apart(reachable, monkeypatch, spoken):
    """The port a recorder's phone app shows is often not its RTSP port. A port
    that stays silent, or answers in another protocol, is said to be the wrong
    kind of port — and nothing is signed in to on it."""

    async def handshake(host, port, **_):
        return spoken

    monkeypatch.setattr(recorder_probe, "_rtsp_handshake", handshake)

    def reader(uri: str):  # pragma: no cover - must not be reached
        raise AssertionError("signed in on a port that is not a video port")

    body = (await _check(reader, port=5501)).as_dict()
    assert body["outcome"] == "not_rtsp"
    assert body["ok"] is False
    assert "port 5501" in body["message"]
    assert "554" in body["hint"]
    assert body["port"] == 5501
    assert PASSWORD not in str(body)


async def test_an_inconclusive_handshake_leaves_the_answer_to_the_decoder(reachable, monkeypatch):
    """A connection that closed without a word proves nothing either way."""

    async def handshake(host, port, **_):
        return "closed"

    monkeypatch.setattr(recorder_probe, "_rtsp_handshake", handshake)
    body = (await _check(lambda uri: (640, 480))).as_dict()
    assert body["outcome"] == "connected"


async def test_a_local_address_nobody_reached_says_why_that_is_likely(monkeypatch):
    """`192.168.x.x` only exists on the site's own network. When it cannot be
    reached, the likeliest reason is that this server is not on that network."""
    _unreachable(monkeypatch, "timeout")
    body = (await _check(lambda uri: (1, 1), host="192.168.54.243")).as_dict()
    assert body["outcome"] == "timeout"
    assert "192.168.54.243 is a local network address" in body["hint"]
    assert "domain" in body["hint"]
    assert body["address"] == "192.168.54.243"


async def test_a_public_address_nobody_reached_keeps_the_plain_hint(monkeypatch):
    _unreachable(monkeypatch, "timeout")
    body = (await _check(lambda uri: (1, 1), host="site.example.com")).as_dict()
    assert "local network" not in body["hint"]


@pytest.mark.parametrize(
    ("behaviour", "expected"), [("rtsp", "rtsp"), ("http", "other"), ("silent", "silent")]
)
async def test_the_handshake_is_real_and_sends_no_credentials(behaviour, expected):
    """Against real sockets. A 401 still counts as RTSP — the server answered
    in RTSP — and the request carries no username or password at all."""
    received: list[bytes] = []

    async def handle(reader, writer):
        received.append(await reader.read(256))
        if behaviour == "rtsp":
            writer.write(b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n\r\n")
        elif behaviour == "http":
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        else:
            await asyncio.sleep(2)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        spoken = await recorder_probe._rtsp_handshake("127.0.0.1", port, timeout_s=0.5)
    finally:
        server.close()
        await server.wait_closed()

    assert spoken == expected
    assert received and received[0].startswith(b"OPTIONS rtsp://127.0.0.1:")
    assert b"Authorization" not in received[0]
    assert b"@" not in received[0]


async def test_the_handshake_calls_a_refused_connection_inconclusive():
    server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    assert await recorder_probe._rtsp_handshake("127.0.0.1", port, timeout_s=0.5) == "closed"


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
            "ip_address": "192.168.1.20",
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
    # What was tested is said back — the address and port the person typed.
    assert (body["address"], body["port"]) == ("192.168.1.20", 554)


async def test_either_address_of_a_new_recorder_can_be_tested(
    admin, client: AsyncClient, reachable, fake_reader
):
    """A site with a local IP and a domain: the person can check each one
    before deciding which this server should use."""
    headers = await bearer(client, "admin@example.com")
    body = {
        "ip_address": "192.168.54.243",
        "hostname": "canteen.example.net",
        "connect_via": "ip_address",
        "rtsp_port": 554,
        "username": "admin",
        "password": PASSWORD,
        "brand": "hikvision",
        "channel": 1,
    }
    by_domain = await client.post(
        "/api/v1/recorders/test", json={**body, "via": "hostname"}, headers=headers
    )
    assert by_domain.status_code == 200, by_domain.text
    assert by_domain.json()["address"] == "canteen.example.net"
    assert "@canteen.example.net:554/" in fake_reader["seen"][-1]

    by_ip = await client.post(
        "/api/v1/recorders/test", json={**body, "channel": 2}, headers=headers
    )
    # The throttle is per address, so the second address is not held up by the first.
    assert by_ip.status_code == 200, by_ip.text
    assert by_ip.json()["address"] == "192.168.54.243"
    assert "@192.168.54.243:554/" in fake_reader["seen"][-1]


async def test_testing_an_address_that_is_not_filled_in_is_refused(
    admin, client: AsyncClient, reachable, fake_reader
):
    headers = await bearer(client, "admin@example.com")
    response = await client.post(
        "/api/v1/recorders/test",
        json={
            "ip_address": "192.168.1.20",
            "username": "admin",
            "password": PASSWORD,
            "brand": "dahua",
            "via": "hostname",
        },
        headers=headers,
    )
    assert response.status_code == 422
    assert not fake_reader["seen"]


async def test_a_saved_recorder_is_tested_with_its_stored_password(
    admin, client: AsyncClient, reachable, fake_reader
):
    headers = await bearer(client, "admin@example.com")
    created = (
        await client.post(
            "/api/v1/recorders",
            json={
                "name": "Gayathri DVR",
                "ip_address": "192.168.1.20",
                "hostname": "gayathri.example.net",
                "connect_via": "ip_address",
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
    # The stored password, unsealed, was what the reader was handed — at the
    # address the recorder connects with.
    assert "Dvr%40Gayathri%232026@192.168.1.20:554/" in fake_reader["seen"][0]
    assert PASSWORD not in response.text

    # Its other address, tested before switching to it.
    import time

    time.sleep(3.1)  # the per-recorder throttle
    fake_reader["result"] = (1920, 1080)
    other = await client.post(
        f"/api/v1/recorders/{created['id']}/test",
        json={"channel": 1, "via": "hostname"},
        headers=headers,
    )
    assert other.json()["outcome"] == "connected"
    assert other.json()["address"] == "gayathri.example.net"
    assert "@gayathri.example.net:554/" in fake_reader["seen"][-1]


async def test_testing_twice_at_once_is_throttled(
    admin, client: AsyncClient, reachable, fake_reader
):
    """An unthrottled test button is a way to lock the DVR account the cameras
    use."""
    headers = await bearer(client, "admin@example.com")
    body = {
        "ip_address": "192.168.1.30",
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
            "ip_address": "192.168.1.40",
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
    assert "192.168.1.40" in rows[-1].detail
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
                "ip_address": "192.168.1.20",
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
