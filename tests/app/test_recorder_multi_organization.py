"""One codebase, many organizations, each with its own CCTV.

The application defines the fields — address, domain, port, account, password,
brand, channel. Each organization's rows hold the values. Nothing about a
recorder comes from the deployment, and nothing about one organization's
recorder can reach another's.

Two organizations here have nothing in common:

    Organization A (org-test) — a canteen in India
        Canteen DVR     192.168.54.243 and canteen-a.example.net, dialled by
                        domain, port 5544, Hikvision, account admin-a
        Main Building   main-a.example.net only, port 554, Dahua, account viewer-a

    Organization B (org-other) — a site in Singapore
        Singapore DVR   10.20.30.40 only, port 8554, Dahua, account operator-b

Every test drives the HTTP API the screens use, and the last one follows a
recorder from being typed in to the URL the runtime actually dials.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import quote

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.domain import connectivity, recorder_probe
from app.domain.cameras import CameraService, to_rtsp_config
from app.domain.connectivity import ConnectivityResult
from app.domain.models import Camera, Recorder, Zone
from app.domain.recorder_secrets import SealedSecret, master_key, open_sealed
from app.domain.recorders import sync_credential
from app.vision.sources.rtsp import LiveRtspSource, ReconnectPolicy

from .conftest import bearer, make_user

BASE = "/api/v1/recorders"

CANTEEN = {
    "name": "Canteen DVR",
    "ip_address": "192.168.54.243",
    "hostname": "canteen-a.example.net",
    "connect_via": "hostname",
    "rtsp_port": 5544,
    "username": "admin-a",
    "password": "Pass-A#India-2026",
    "brand": "hikvision",
}
MAIN_BUILDING = {
    "name": "Main Building DVR",
    "hostname": "main-a.example.net",
    "rtsp_port": 554,
    "username": "viewer-a",
    "password": "Pass-A2/second-box",
    "brand": "dahua",
}
SINGAPORE = {
    "name": "Singapore Main DVR",
    "ip_address": "10.20.30.40",
    "rtsp_port": 8554,
    "username": "operator-b",
    "password": "Pass-B@Singapore!",
    "brand": "dahua",
}
PASSWORDS = (CANTEEN["password"], MAIN_BUILDING["password"], SINGAPORE["password"])
#: Nothing a recorder response may carry, whatever the route.
SECRET_KEYS = ("password", "secret_ciphertext", "secret_nonce", "secret_key_id", "credential_ref")


@pytest.fixture
async def estate(seeded):
    """An administrator and a zone in each organization."""
    async with seeded.state.database.session_scope() as session:
        for org, email, zone in (
            ("org-test", "admin-a@example.com", "zone-a"),
            ("org-other", "admin-b@example.com", "zone-b"),
        ):
            _, admin = make_user(
                org_id=org,
                email=email,
                roles=("org_admin",),
                camera_breadth="all_in_tenant",
                camera_ids="",
            )
            session.add(admin)
            session.add(Zone(id=zone, organization_id=org, name=f"Kitchen {org}"))
    return seeded


async def _headers(client: AsyncClient) -> tuple[dict, dict]:
    return await bearer(client, "admin-a@example.com"), await bearer(client, "admin-b@example.com")


async def _add(client: AsyncClient, headers: dict, body: dict, **extra) -> dict:
    response = await client.post(BASE, json={**body, **extra}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_nothing_secret(text: str) -> None:
    for password in PASSWORDS:
        assert password not in text
        assert quote(password, safe="") not in text
    parsed = json.loads(text)
    rendered_keys = json.dumps(parsed)
    for key in SECRET_KEYS:
        assert f'"{key}"' not in rendered_keys, key


async def _recorder_row(app, recorder_id: str) -> Recorder:
    async with app.state.database.session_scope() as session:
        return (
            await session.execute(select(Recorder).where(Recorder.id == recorder_id))
        ).scalar_one()


# ── each organization sees its own ───────────────────────────────────────────


async def test_each_organization_lists_only_its_own_recorders(estate, client: AsyncClient):
    a, b = await _headers(client)
    canteen = await _add(client, a, CANTEEN)
    main = await _add(client, a, MAIN_BUILDING)
    singapore = await _add(client, b, SINGAPORE)

    seen_by_a = {r["id"] for r in (await client.get(BASE, headers=a)).json()["recorders"]}
    seen_by_b = {r["id"] for r in (await client.get(BASE, headers=b)).json()["recorders"]}

    assert {canteen["id"], main["id"]} <= seen_by_a
    assert singapore["id"] not in seen_by_a
    assert singapore["id"] in seen_by_b
    assert not {canteen["id"], main["id"]} & seen_by_b


async def test_another_organizations_recorder_does_not_exist(estate, client: AsyncClient):
    """Not "forbidden": a refusal would confirm it exists."""
    a, b = await _headers(client)
    singapore = await _add(client, b, SINGAPORE)
    theirs = f"{BASE}/{singapore['id']}"

    attempts = [
        ("GET", theirs, None),
        ("PATCH", theirs, {"name": "Taken over"}),
        ("PUT", f"{theirs}/credential", {"password": "not-theirs-to-set"}),
        ("POST", f"{theirs}/test", {"channel": 1}),
        ("POST", f"{theirs}/deactivate", None),
        ("POST", f"{theirs}/activate", None),
        ("DELETE", theirs, None),
    ]
    for method, url, body in attempts:
        response = await client.request(method, url, json=body, headers=a)
        assert response.status_code == 404, (method, url, response.status_code)

    # Nor can a camera be put on it.
    camera = await client.post(
        "/api/v1/cameras",
        json={
            "camera_key": "cam-90",
            "name": "Borrowed",
            "recorder_id": singapore["id"],
            "channel": 1,
            "zone_id": "zone-a",
        },
        headers=a,
    )
    assert camera.status_code == 404

    # And B's recorder is exactly as B left it.
    row = await _recorder_row(estate, singapore["id"])
    assert (row.name, row.is_active, row.ip_address) == (SINGAPORE["name"], True, "10.20.30.40")


# ── values, not configuration ────────────────────────────────────────────────


async def test_every_value_is_the_organizations_own(estate, client: AsyncClient):
    a, b = await _headers(client)
    canteen = await _add(client, a, CANTEEN)
    singapore = await _add(client, b, SINGAPORE)

    assert (canteen["ip_address"], canteen["hostname"], canteen["connect_via"]) == (
        "192.168.54.243",
        "canteen-a.example.net",
        "hostname",
    )
    assert canteen["address"] == "canteen-a.example.net"
    assert (canteen["rtsp_port"], canteen["username"], canteen["brand"]) == (
        5544,
        "admin-a",
        "hikvision",
    )
    assert (singapore["address"], singapore["rtsp_port"], singapore["username"]) == (
        "10.20.30.40",
        8554,
        "operator-b",
    )
    assert singapore["hostname"] == "" and singapore["connect_via"] == "ip_address"


async def test_adding_a_recorder_creates_a_new_record_every_time(estate, client: AsyncClient):
    a, _ = await _headers(client)
    canteen = await _add(client, a, CANTEEN)
    main = await _add(client, a, MAIN_BUILDING)

    assert canteen["id"] != main["id"]
    canteen_row = await _recorder_row(estate, canteen["id"])
    main_row = await _recorder_row(estate, main["id"])
    assert canteen_row.organization_id == main_row.organization_id == "org-test"
    # Each keeps its own password, sealed on its own row.
    assert canteen_row.credential_ref == f"recorder:{canteen['id']}"
    assert main_row.credential_ref == f"recorder:{main['id']}"
    assert bytes(canteen_row.secret_ciphertext) != bytes(main_row.secret_ciphertext)


async def test_passwords_are_sealed_and_never_returned(estate, client: AsyncClient, settings):
    a, b = await _headers(client)
    canteen = await _add(client, a, CANTEEN)
    singapore = await _add(client, b, SINGAPORE)

    key = master_key(settings)
    for created, body in ((canteen, CANTEEN), (singapore, SINGAPORE)):
        row = await _recorder_row(estate, created["id"])
        assert body["password"].encode() not in bytes(row.secret_ciphertext)
        sealed = SealedSecret(
            bytes(row.secret_ciphertext), bytes(row.secret_nonce), row.secret_key_id
        )
        assert open_sealed(sealed, key=key) == body["password"]

    responses = [
        await client.post(BASE, json={**MAIN_BUILDING}, headers=a),
        await client.get(BASE, headers=a),
        await client.get(f"{BASE}/{canteen['id']}", headers=a),
        await client.patch(f"{BASE}/{canteen['id']}", json={"rtsp_port": 5545}, headers=a),
        await client.put(
            f"{BASE}/{canteen['id']}/credential", json={"password": CANTEEN["password"]}, headers=a
        ),
        await client.post(f"{BASE}/{canteen['id']}/deactivate", headers=a),
        await client.post(f"{BASE}/{canteen['id']}/activate", headers=a),
        await client.get(BASE, headers=b),
        await client.get(f"{BASE}/{singapore['id']}", headers=b),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        _assert_nothing_secret(response.text)


# ── choosing an existing recorder ────────────────────────────────────────────


async def test_a_camera_on_an_existing_recorder_reuses_its_stored_connection(
    estate, client: AsyncClient
):
    """Choosing a recorder means naming it — the address, port, account and
    password are not asked for again, and cannot be supplied instead."""
    a, _ = await _headers(client)
    canteen = await _add(client, a, CANTEEN)

    response = await client.post(
        "/api/v1/cameras",
        json={
            "camera_key": "cam-31",
            "name": "Serving line",
            "recorder_id": canteen["id"],
            "channel": 3,
            "stream_type": "sub",
            "zone_id": "zone-a",
            # A client trying to give the camera its own connection is ignored:
            # how to reach a camera is its recorder's.
            "host": "attacker.example",
            "rtsp_port": 1,
            "username": "someone-else",
        },
        headers=a,
    )
    assert response.status_code in (200, 201), response.text
    camera = response.json()
    assert camera["recorder_id"] == canteen["id"]
    assert camera["recorder_name"] == "Canteen DVR"
    for field in ("host", "ip_address", "hostname", "rtsp_port", "username", "password"):
        assert field not in camera

    async with estate.state.database.session_scope() as session:
        row = await CameraService(session).get(organization_id="org-test", camera_key="cam-31")
        config = to_rtsp_config(row)
    assert (config.host, config.port, config.username) == ("canteen-a.example.net", 5544, "admin-a")
    assert config.credential_ref == f"recorder:{canteen['id']}"
    # Hikvision's path, from the recorder's brand: channel 3, sub stream.
    assert config.path() == "/Streaming/Channels/302"


async def test_switching_which_address_is_dialled_moves_every_camera_on_it(
    estate, client: AsyncClient
):
    a, _ = await _headers(client)
    canteen = await _add(
        client, a, CANTEEN, cameras=[{"channel": 1, "name": "Kitchen", "zone_id": "zone-a"}]
    )
    key = canteen["cameras"][0]["camera_key"]

    switched = await client.patch(
        f"{BASE}/{canteen['id']}", json={"connect_via": "ip_address"}, headers=a
    )
    assert switched.status_code == 200, switched.text
    assert switched.json()["address"] == "192.168.54.243"
    # Both addresses are still held; only the choice moved.
    assert switched.json()["hostname"] == "canteen-a.example.net"

    async with estate.state.database.session_scope() as session:
        row = await CameraService(session).get(organization_id="org-test", camera_key=key)
        assert to_rtsp_config(row).host == "192.168.54.243"


# ── channels belong to their recorder ────────────────────────────────────────


async def test_channels_belong_to_their_recorder(estate, client: AsyncClient):
    a, b = await _headers(client)
    canteen = await _add(
        client, a, CANTEEN, cameras=[{"channel": 1, "name": "Kitchen", "zone_id": "zone-a"}]
    )
    main = await _add(
        client, a, MAIN_BUILDING, cameras=[{"channel": 1, "name": "Lobby", "zone_id": "zone-a"}]
    )
    singapore = await _add(
        client, b, SINGAPORE, cameras=[{"channel": 1, "name": "Prep", "zone_id": "zone-b"}]
    )

    # Channel 1 on three different boxes is three different cameras.
    for recorder, headers, name in (
        (canteen, a, "Kitchen"),
        (main, a, "Lobby"),
        (singapore, b, "Prep"),
    ):
        detail = (await client.get(f"{BASE}/{recorder['id']}", headers=headers)).json()
        assert [(c["name"], c["channel"]) for c in detail["cameras"]] == [(name, 1)]

    # The same channel twice on one box is refused, naming the camera using it.
    duplicate = await client.post(
        "/api/v1/cameras",
        json={
            "camera_key": "cam-77",
            "name": "Kitchen again",
            "recorder_id": canteen["id"],
            "channel": 1,
            "zone_id": "zone-a",
        },
        headers=a,
    )
    assert duplicate.status_code == 409
    assert "Kitchen" in duplicate.json()["message"]

    # Moving a camera onto a channel another camera already uses is refused too.
    lobby = (await client.get(f"{BASE}/{main['id']}", headers=a)).json()["cameras"][0]
    moved = await client.patch(
        f"/api/v1/cameras/{lobby['camera_key']}",
        json={"recorder_id": canteen["id"]},
        headers=a,
    )
    assert moved.status_code == 409


async def test_a_new_recorder_with_the_same_channel_listed_twice_saves_nothing(
    estate, client: AsyncClient
):
    a, _ = await _headers(client)
    response = await client.post(
        BASE,
        json={
            **CANTEEN,
            "cameras": [
                {"channel": 2, "name": "One", "zone_id": "zone-a"},
                {"channel": 2, "name": "Two", "zone_id": "zone-a"},
            ],
        },
        headers=a,
    )
    assert response.status_code == 422
    async with estate.state.database.session_scope() as session:
        assert (
            await session.execute(select(Recorder).where(Recorder.name == CANTEEN["name"]))
        ).scalar_one_or_none() is None


# ── the runtime ──────────────────────────────────────────────────────────────


async def test_the_runtime_dials_each_organizations_own_recorder(estate, client: AsyncClient):
    """From row to dial URL, through the same provider the runtime uses."""
    a, b = await _headers(client)
    canteen = await _add(
        client, a, CANTEEN, cameras=[{"channel": 4, "name": "Kitchen", "zone_id": "zone-a"}]
    )
    singapore = await _add(
        client, b, SINGAPORE, cameras=[{"channel": 4, "name": "Prep", "zone_id": "zone-b"}]
    )
    provider = estate.state.credential_provider
    store = estate.state.recorder_credentials

    dialled = {}
    async with estate.state.database.session_scope() as session:
        rows = (await session.execute(select(Camera))).scalars().all()
        for row in rows:
            if row.recorder_id not in (canteen["id"], singapore["id"]):
                continue
            sync_credential(store, row.recorder)
            config = to_rtsp_config(row)
            dialled[row.organization_id] = config.dial_uri(provider.resolve(config.credential_ref))

    a_url, b_url = dialled["org-test"], dialled["org-other"]
    assert a_url.startswith(
        f"rtsp://admin-a:{quote(CANTEEN['password'], safe='')}@canteen-a.example.net:5544/"
    )
    assert b_url.startswith(
        f"rtsp://operator-b:{quote(SINGAPORE['password'], safe='')}@10.20.30.40:8554/"
    )
    assert quote(SINGAPORE["password"], safe="") not in a_url
    assert quote(CANTEEN["password"], safe="") not in b_url


# ── the whole path ───────────────────────────────────────────────────────────


class _RecordingSource(LiveRtspSource):
    """The runtime's own RTSP source, with the network replaced at the opener.

    Everything before the socket is production code: resolving the recorder's
    sealed password, building the URL from the recorder's address, port,
    account and brand. The opener records that URL and reports a network
    failure, so no test ever reaches a real recorder.
    """

    dialled: list[str] = []

    def __init__(self, config, *, secrets, reconnect=None, opener=None) -> None:
        def record(uri: str):
            _RecordingSource.dialled.append(uri)
            raise OSError("no network in tests")

        super().__init__(
            config,
            secrets=secrets,
            reconnect=ReconnectPolicy(initial_ms=5, max_ms=10, max_attempts=1),
            opener=record,
        )


async def test_from_typing_a_recorder_in_to_the_runtime_dialling_it(
    estate, client: AsyncClient, monkeypatch
):
    """Create recorder → network details → sealed password → camera → runtime
    config → secret resolved → connection test → Start watching → the live
    session dials exactly this organization's recorder."""
    import app.vision.manager as manager_module

    seen_by_test: list[str] = []

    async def reachable(host, port, **_):
        return ConnectivityResult(True, "reachable", "", 2)

    async def speaks_rtsp(host, port, **_):
        return "rtsp"

    def one_frame(uri: str):
        seen_by_test.append(uri)
        return 1280, 720

    monkeypatch.setattr(connectivity, "probe", reachable)
    monkeypatch.setattr(recorder_probe, "_rtsp_handshake", speaks_rtsp)
    monkeypatch.setattr(recorder_probe, "_first_frame_with_pyav", one_frame)
    monkeypatch.setattr(manager_module, "LiveRtspSource", _RecordingSource)
    _RecordingSource.dialled = []
    monkeypatch.setattr(estate.state.settings, "feature_live_cctv", True)
    # The Live Wall half of Start watching would open a real socket; it is
    # covered with the wall replaced in `test_camera_wall_lifecycle.py`.
    monkeypatch.setattr(estate.state.settings, "feature_camera_wall", False)

    a, b = await _headers(client)
    singapore = await _add(client, b, SINGAPORE)

    # 1-4. The recorder, its addresses, its sealed password, and a camera.
    canteen = await _add(
        client,
        a,
        CANTEEN,
        cameras=[{"channel": 2, "name": "Kitchen", "zone_id": "zone-a", "stream_type": "main"}],
    )
    camera_key = canteen["cameras"][0]["camera_key"]
    row = await _recorder_row(estate, canteen["id"])
    assert row.credential_ref == f"recorder:{canteen['id']}"

    # 5-7. The saved recorder's own test: stored password, its own address.
    tested = await client.post(f"{BASE}/{canteen['id']}/test", json={"channel": 2}, headers=a)
    assert tested.json()["outcome"] == "connected", tested.text
    expected = f"rtsp://admin-a:{quote(CANTEEN['password'], safe='')}@canteen-a.example.net:5544/"
    # A recorder test asks for the light sub stream (…02) whatever the camera uses.
    assert seen_by_test[-1] == expected + "Streaming/Channels/202"
    _assert_nothing_secret(tested.text)

    # 8. Start watching: the live runtime opens a session and dials.
    try:
        started = await client.post(f"/api/v1/cameras/{camera_key}/start", headers=a)
        assert started.status_code == 200, started.text
        assert started.json()["streaming"] is True, started.json()
        for _ in range(300):
            if _RecordingSource.dialled:
                break
            await asyncio.sleep(0.01)
        assert _RecordingSource.dialled, "the runtime never dialled"
        # The camera itself dials its own stream choice: main (…01).
        assert _RecordingSource.dialled[0] == expected + "Streaming/Channels/201"

        # The session is this organization's: on its recorder page, and nowhere
        # in the other organization's.
        mine = (await client.get(f"{BASE}/{canteen['id']}", headers=a)).json()
        assert [c["running"] for c in mine["cameras"]] == [True]
        live = estate.state.live
        assert [s.camera_id for s in live.visible(tenant_id="org-test", camera_ids=None)] == [
            f"org-test:{camera_key}"
        ]
        assert live.visible(tenant_id="org-other", camera_ids=None) == []
        theirs = (await client.get(f"{BASE}/{singapore['id']}", headers=b)).json()
        assert theirs["cameras"] == []
    finally:
        await estate.state.live.stop_all()
