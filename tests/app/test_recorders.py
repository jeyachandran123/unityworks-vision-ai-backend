"""The recorder API, end to end over HTTP.

Organised by what a person does with it — add a recorder, see it, change it,
change its password, stop it, remove it — and then by the properties that must
hold whatever they do: the password never comes back, another organization's
recorder does not exist, and the audit trail records acts without secrets.
"""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.domain.audit import AuditAction
from app.domain.models import AuditEvent, Camera, Recorder, Zone
from app.domain.recorder_secrets import SealedSecret, master_key, open_sealed

from .conftest import bearer, make_user

BASE = "/api/v1/recorders"
PASSWORD = "Dvr@Gayathri#2026"


@pytest.fixture
async def estate(seeded):
    """An org admin who may manage cameras, and three zones to put them in."""
    async with seeded.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
        session.add(Zone(id="zone-kitchen", organization_id="org-test", name="Kitchen"))
        session.add(Zone(id="zone-hall", organization_id="org-test", name="Dine hall"))
        session.add(Zone(id="zone-theirs", organization_id="org-other", name="Their kitchen"))
    return seeded


def _recorder(**overrides) -> dict:
    body = {
        "name": "Gayathri DVR",
        "ip_address": "192.168.1.20",
        "rtsp_port": 554,
        "username": "admin",
        "password": PASSWORD,
        "brand": "hikvision",
    }
    body.update(overrides)
    return body


async def _admin(client: AsyncClient) -> dict[str, str]:
    return await bearer(client, "admin@example.com")


async def _create(client: AsyncClient, headers, **overrides) -> dict:
    response = await client.post(BASE, json=_recorder(**overrides), headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def _row(app, recorder_id: str) -> Recorder:
    async with app.state.database.session_scope() as session:
        return (
            await session.execute(select(Recorder).where(Recorder.id == recorder_id))
        ).scalar_one()


# ── adding a recorder ────────────────────────────────────────────────────────


async def test_adding_a_recorder_returns_it_without_its_password(estate, client: AsyncClient):
    body = await _create(client, await _admin(client))

    assert body["name"] == "Gayathri DVR"
    assert body["brand"] == "hikvision"
    assert body["brand_label"] == "Hikvision"
    assert body["ip_address"] == "192.168.1.20"
    assert body["hostname"] == ""
    assert body["connect_via"] == "ip_address"
    assert body["address"] == "192.168.1.20"
    assert "host" not in body
    assert body["credential_configured"] is True
    assert body["credential_scheme"] == "recorder"
    assert body["is_active"] is True
    assert body["camera_count"] == 0


async def test_the_password_is_stored_encrypted_and_opens_with_the_master_key(
    estate, client: AsyncClient, settings
):
    """Stored, not as plaintext, and recoverable only with the key from the
    environment — the two halves the design depends on."""
    body = await _create(client, await _admin(client))
    row = await _row(estate, body["id"])

    assert row.secret_ciphertext is not None and row.secret_nonce is not None
    assert PASSWORD.encode() not in bytes(row.secret_ciphertext)
    assert row.credential_ref == f"recorder:{row.id}"
    sealed = SealedSecret(bytes(row.secret_ciphertext), bytes(row.secret_nonce), row.secret_key_id)
    assert open_sealed(sealed, key=master_key(settings)) == PASSWORD


async def test_a_recorder_needs_a_password(estate, client: AsyncClient):
    response = await client.post(BASE, json=_recorder(password=""), headers=await _admin(client))
    assert response.status_code == 422
    assert "password" in response.json()["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ip_address", "rtsp://192.168.1.20"),
        ("ip_address", "192.168.1.20:554"),
        ("ip_address", "192.168.1.20/cam"),
        ("ip_address", "evil.example@192.168.1.20"),
        ("ip_address", "192.168.1.300"),
        ("hostname", "rtsp://dvr.example.com"),
        ("hostname", "dvr.example.com:554"),
        ("hostname", "dvr.example.com/cam"),
        ("hostname", "evil.example@dvr.example.com"),
        ("hostname", "dvr example"),
    ],
)
async def test_an_address_that_would_rewrite_the_url_is_refused(
    estate, client: AsyncClient, field: str, value: str
):
    """The address is placed inside `rtsp://user:pass@HOST:port/path`. An `@`
    would move the credentials onto a host of the caller's choosing."""
    body = _recorder(**{"ip_address": "", "hostname": "", field: value})
    response = await client.post(BASE, json=body, headers=await _admin(client))
    assert response.status_code == 422, value


async def test_a_recorder_needs_at_least_one_address(estate, client: AsyncClient):
    response = await client.post(
        BASE, json=_recorder(ip_address="", hostname=""), headers=await _admin(client)
    )
    assert response.status_code == 422
    assert "IP address, its domain, or both" in response.json()["message"]


async def test_an_unknown_brand_is_refused(estate, client: AsyncClient):
    response = await client.post(BASE, json=_recorder(brand="acme"), headers=await _admin(client))
    assert response.status_code == 422
    assert "brand" in response.json()["message"]


async def test_a_custom_brand_needs_a_path_with_a_channel(estate, client: AsyncClient):
    headers = await _admin(client)
    missing = await client.post(
        BASE,
        json=_recorder(brand="custom", path_template="/live", stream_main=0, stream_sub=1),
        headers=headers,
    )
    assert missing.status_code == 422
    assert "{channel}" in missing.json()["message"]

    fine = await client.post(
        BASE,
        json=_recorder(
            brand="custom",
            path_template="/live/ch{channel}/{subtype}",
            stream_main=0,
            stream_sub=1,
        ),
        headers=headers,
    )
    assert fine.status_code == 200, fine.text
    assert fine.json()["path_template"] == "/live/ch{channel}/{subtype}"


async def test_two_recorders_cannot_share_a_name(estate, client: AsyncClient):
    headers = await _admin(client)
    await _create(client, headers)
    again = await client.post(BASE, json=_recorder(), headers=headers)
    assert again.status_code == 409


async def test_a_server_without_a_master_key_refuses_clearly_and_saves_nothing(
    estate, client: AsyncClient
):
    """A missed deployment step, reported as one — not a 500, and not a
    recorder row left behind with no password."""
    estate.state.settings.recorder_secret_key = type(estate.state.settings.recorder_secret_key)("")
    response = await client.post(BASE, json=_recorder(), headers=await _admin(client))

    assert response.status_code == 503
    assert "RECORDER_SECRET_KEY" in response.json()["message"]
    async with estate.state.database.session_scope() as session:
        mine = select(Recorder).where(Recorder.name == "Gayathri DVR")
        assert (await session.execute(mine)).scalars().all() == []


# ── adding its cameras in the same save ──────────────────────────────────────


async def test_the_wizard_saves_a_recorder_and_its_cameras_together(estate, client: AsyncClient):
    body = await _create(
        client,
        await _admin(client),
        cameras=[
            {"channel": 1, "name": "Kitchen entrance", "zone_id": "zone-kitchen"},
            {"channel": 2, "name": "Prep area", "zone_id": "zone-kitchen", "stream_type": "main"},
            {"channel": 3, "name": "Hall", "zone_id": "zone-hall"},
        ],
    )

    assert body["camera_count"] == 3
    cameras = {c["channel"]: c for c in body["cameras"]}
    assert cameras[2]["stream_type"] == "main"
    assert all(c["recorder_id"] == body["id"] for c in cameras.values())
    # Created switched off, as every camera is.
    assert not any(c["enabled"] for c in cameras.values())
    # And the camera never carries an address of its own.
    assert "host" not in cameras[1]


async def test_one_bad_camera_saves_nothing_at_all(estate, client: AsyncClient):
    """A recorder saved with half its cameras would be a configuration nobody
    chose, and the person would have to work out which half."""
    response = await client.post(
        BASE,
        json=_recorder(
            cameras=[
                {"channel": 1, "name": "One", "zone_id": "zone-kitchen"},
                {"channel": 1, "name": "Also one", "zone_id": "zone-kitchen"},
            ]
        ),
        headers=await _admin(client),
    )

    assert response.status_code == 422
    assert "channel 1" in response.json()["message"]
    async with estate.state.database.session_scope() as session:
        mine = select(Recorder).where(Recorder.name == "Gayathri DVR")
        assert (await session.execute(mine)).scalars().all() == []
        assert (await session.execute(select(Camera))).scalars().all() == []


async def test_a_camera_cannot_be_put_in_another_organizations_zone(estate, client: AsyncClient):
    response = await client.post(
        BASE,
        json=_recorder(cameras=[{"channel": 1, "name": "Sneaky", "zone_id": "zone-theirs"}]),
        headers=await _admin(client),
    )
    assert response.status_code == 404


# ── seeing it ────────────────────────────────────────────────────────────────


async def test_the_list_counts_each_recorders_cameras(estate, client: AsyncClient):
    headers = await _admin(client)
    await _create(
        client,
        headers,
        cameras=[
            {"channel": 1, "name": "A", "zone_id": "zone-kitchen"},
            {"channel": 2, "name": "B", "zone_id": "zone-kitchen"},
        ],
    )
    await _create(client, headers, name="Spare NVR", brand="dahua")

    body = (await client.get(BASE, headers=headers)).json()
    counts = {r["name"]: r["camera_count"] for r in body["recorders"]}
    # The seeded recorder is here too, and holds none of these.
    assert counts["Gayathri DVR"] == 2
    assert counts["Spare NVR"] == 0
    assert body["total"] == len(body["recorders"])


async def test_one_recorder_lists_its_cameras(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(
        client, headers, cameras=[{"channel": 4, "name": "Store", "zone_id": "zone-hall"}]
    )
    body = (await client.get(f"{BASE}/{created['id']}", headers=headers)).json()

    assert [(c["channel"], c["name"], c["running"]) for c in body["cameras"]] == [
        (4, "Store", False)
    ]


async def test_a_manager_may_read_recorders_but_not_add_one(estate, client: AsyncClient):
    headers = await bearer(client, "manager@example.com")
    assert (await client.get(BASE, headers=headers)).status_code == 200
    assert (await client.post(BASE, json=_recorder(), headers=headers)).status_code == 403


async def test_an_account_without_camera_access_may_not_see_recorders(estate, client: AsyncClient):
    """Reading recorders follows `VIEW_CAMERAS` exactly. An auditor reviews the
    record and does not hold it, so recorders are not theirs to see either."""
    async with estate.state.database.session_scope() as session:
        _, auditor = make_user(email="auditor@example.com", roles=("auditor",))
        session.add(auditor)
    headers = await bearer(client, "auditor@example.com")
    assert (await client.get(BASE, headers=headers)).status_code == 403


async def test_a_recorder_shows_only_the_cameras_the_caller_may_see(estate, client: AsyncClient):
    """The seeded manager may see cam-01 and cam-02 only. The recorder page must
    not become a way round that."""
    admin = await _admin(client)
    created = await _create(
        client,
        admin,
        cameras=[
            {"channel": 1, "name": "One", "zone_id": "zone-kitchen"},
            {"channel": 2, "name": "Two", "zone_id": "zone-kitchen"},
            {"channel": 3, "name": "Three", "zone_id": "zone-kitchen"},
        ],
    )
    manager = await bearer(client, "manager@example.com")
    body = (await client.get(f"{BASE}/{created['id']}", headers=manager)).json()

    assert sorted(c["camera_key"] for c in body["cameras"]) == ["cam-01", "cam-02"]


async def test_another_organizations_recorder_does_not_exist(estate, client: AsyncClient):
    """404, not 403: a refusal would confirm it is there."""
    created = await _create(client, await _admin(client))
    outsider = await bearer(client, "outsider@example.com")

    for response in (
        await client.get(f"{BASE}/{created['id']}", headers=outsider),
        await client.patch(f"{BASE}/{created['id']}", json={"name": "Mine"}, headers=outsider),
        await client.put(
            f"{BASE}/{created['id']}/credential", json={"password": "x" * 12}, headers=outsider
        ),
        await client.post(f"{BASE}/{created['id']}/deactivate", headers=outsider),
        await client.delete(f"{BASE}/{created['id']}", headers=outsider),
    ):
        assert response.status_code == 404, response.text

    listing = (await client.get(BASE, headers=outsider)).json()
    assert created["id"] not in {r["id"] for r in listing["recorders"]}


# ── changing it ──────────────────────────────────────────────────────────────


async def test_editing_changes_how_it_is_reached(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(client, headers)

    response = await client.patch(
        f"{BASE}/{created['id']}",
        json={"ip_address": "192.168.1.21", "rtsp_port": 8554, "brand": "dahua"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["ip_address"], body["rtsp_port"], body["brand"]) == (
        "192.168.1.21",
        8554,
        "dahua",
    )
    # The password is untouched by an edit.
    assert body["credential_configured"] is True


async def test_an_edit_cannot_carry_a_password(estate, client: AsyncClient):
    """Changing the password is its own act with its own audit row, so "who
    changed it" is always answerable."""
    headers = await _admin(client)
    created = await _create(client, headers)
    response = await client.patch(
        f"{BASE}/{created['id']}", json={"password": "sneaky-change"}, headers=headers
    )
    assert response.status_code == 422


async def test_changing_the_password_replaces_it_without_reading_it(
    estate, client: AsyncClient, settings
):
    headers = await _admin(client)
    created = await _create(client, headers)
    before = bytes((await _row(estate, created["id"])).secret_ciphertext)

    response = await client.put(
        f"{BASE}/{created['id']}/credential", json={"password": "New-Pass-2027"}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert "New-Pass-2027" not in response.text

    row = await _row(estate, created["id"])
    assert bytes(row.secret_ciphertext) != before
    sealed = SealedSecret(bytes(row.secret_ciphertext), bytes(row.secret_nonce), row.secret_key_id)
    assert open_sealed(sealed, key=master_key(settings)) == "New-Pass-2027"


async def test_a_recorder_moved_onto_a_typed_password_leaves_the_environment(
    estate, client: AsyncClient
):
    """The seeded recorder starts on `env:CCTV_PASSWORD`, as a migrated one does.
    Setting a password through the product moves it to its own sealed one."""
    headers = await _admin(client)
    response = await client.put(
        f"{BASE}/rec-org-test/credential", json={"password": "Typed-Once-2026"}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json()["credential_scheme"] == "recorder"


# ── stopping it ──────────────────────────────────────────────────────────────


async def test_a_deactivated_recorder_starts_none_of_its_cameras(estate, client: AsyncClient):
    from app.domain.cameras import CameraService

    headers = await _admin(client)
    created = await _create(
        client, headers, cameras=[{"channel": 1, "name": "A", "zone_id": "zone-kitchen"}]
    )
    camera_key = created["cameras"][0]["camera_key"]
    enabled = await client.patch(
        f"/api/v1/cameras/{camera_key}", json={"enabled": True}, headers=headers
    )
    assert enabled.status_code == 200, enabled.text

    async def runnable() -> list[str]:
        async with estate.state.database.session_scope() as session:
            rows = await CameraService(session).enabled_for_runtime(organization_id="org-test")
            return [row.camera_key for row in rows]

    assert camera_key in await runnable()

    off = await client.post(f"{BASE}/{created['id']}/deactivate", headers=headers)
    assert off.status_code == 200, off.text
    assert off.json()["is_active"] is False
    assert camera_key not in await runnable()

    # Nothing about the camera was lost: it is still switched on, and comes back.
    detail = (await client.get("/api/v1/cameras", headers=headers)).json()
    mine = next(c for c in detail["cameras"] if c["camera_key"] == camera_key)
    assert mine["enabled"] is True
    assert mine["recorder_active"] is False

    on = await client.post(f"{BASE}/{created['id']}/activate", headers=headers)
    assert on.status_code == 200, on.text
    assert camera_key in await runnable()


async def test_starting_a_camera_on_a_deactivated_recorder_says_why(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(
        client, headers, cameras=[{"channel": 1, "name": "A", "zone_id": "zone-kitchen"}]
    )
    await client.post(f"{BASE}/{created['id']}/deactivate", headers=headers)
    estate.state.settings.feature_live_cctv = True

    response = await client.post(
        f"/api/v1/cameras/{created['cameras'][0]['camera_key']}/start", headers=headers
    )
    body = response.json()
    assert body["streaming"] is False
    assert "deactivated" in body["reason"]


# ── removing it ──────────────────────────────────────────────────────────────


async def test_a_recorder_with_cameras_is_not_deleted(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(
        client, headers, cameras=[{"channel": 1, "name": "A", "zone_id": "zone-kitchen"}]
    )
    response = await client.delete(f"{BASE}/{created['id']}", headers=headers)

    assert response.status_code == 409
    assert "1 camera" in response.json()["message"]
    assert "Deactivate" in response.json()["message"]


async def test_an_empty_recorder_is_deleted(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(client, headers)
    response = await client.delete(f"{BASE}/{created['id']}", headers=headers)

    assert response.status_code == 200
    assert (await client.get(f"{BASE}/{created['id']}", headers=headers)).status_code == 404


# ── properties that hold whatever happens ────────────────────────────────────


async def test_no_response_on_any_route_contains_the_password(estate, client: AsyncClient):
    """Asserted against every response body whole, not field by field, so a field
    added later cannot quietly start carrying it."""
    headers = await _admin(client)
    created_response = await client.post(
        BASE,
        json=_recorder(cameras=[{"channel": 1, "name": "A", "zone_id": "zone-kitchen"}]),
        headers=headers,
    )
    recorder_id = created_response.json()["id"]

    responses = [
        created_response,
        await client.get(BASE, headers=headers),
        await client.get(f"{BASE}/{recorder_id}", headers=headers),
        await client.patch(f"{BASE}/{recorder_id}", json={"name": "Renamed"}, headers=headers),
        await client.put(
            f"{BASE}/{recorder_id}/credential", json={"password": "Rotated#2027"}, headers=headers
        ),
        await client.post(f"{BASE}/{recorder_id}/deactivate", headers=headers),
        await client.post(f"{BASE}/{recorder_id}/activate", headers=headers),
        await client.get("/api/v1/cameras", headers=headers),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        for secret in (PASSWORD, "Rotated#2027"):
            assert secret not in response.text
        for field in ("secret_ciphertext", "secret_nonce", "credential_ref", "password"):
            assert f'"{field}"' not in response.text, (field, response.request.url)


async def test_recorder_responses_are_never_cached(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(client, headers)
    for response in (
        await client.get(BASE, headers=headers),
        await client.get(f"{BASE}/{created['id']}", headers=headers),
    ):
        assert "no-store" in response.headers.get("cache-control", "")


async def test_the_audit_trail_records_every_act_and_no_secret(estate, client: AsyncClient):
    headers = await _admin(client)
    created = await _create(client, headers)
    await client.put(
        f"{BASE}/{created['id']}/credential", json={"password": "Rotated#2027"}, headers=headers
    )
    await client.post(f"{BASE}/{created['id']}/deactivate", headers=headers)
    await client.post(f"{BASE}/{created['id']}/activate", headers=headers)

    async with estate.state.database.session_scope() as session:
        events = (
            (
                await session.execute(
                    select(AuditEvent).where(AuditEvent.resource_type == "recorder")
                )
            )
            .scalars()
            .all()
        )
    actions = {event.action for event in events}
    assert {
        AuditAction.RECORDER_CREATED.value,
        AuditAction.RECORDER_CREDENTIAL_SET.value,
        AuditAction.RECORDER_DEACTIVATED.value,
        AuditAction.RECORDER_ACTIVATED.value,
    } <= actions

    everything = json.dumps([event.detail for event in events], default=str)
    for secret in (PASSWORD, "Rotated#2027"):
        assert secret not in everything
