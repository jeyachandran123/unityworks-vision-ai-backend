"""From a password typed into the application to the URL the runtime dials.

The chain these tests walk is the whole point of the feature:

    recorder saved through the API  (password sealed in its row)
      → camera saved on that recorder
      → the credential store loads the sealed value
      → `to_rtsp_config` builds a config holding only `recorder:<id>`
      → the runtime's provider resolves it at connect time
      → `dial_uri` produces the address for *this* brand and channel

Nothing in the chain is mocked except the network: the last step hands the URL
to an injected opener, the same seam `LiveRtspSource` uses in production.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.domain.cameras import CameraService, is_dialable, to_rtsp_config
from app.domain.models import Zone
from app.domain.recorder_secrets import SealedSecret, seal
from app.domain.recorders import load_sealed
from app.vision.recorder_credentials import (
    RecorderCredentialStore,
    RecorderSecretProvider,
    recorder_reference,
)
from app.vision.secrets import EnvironmentSecretProvider, MissingSecretError
from app.vision.sources.rtsp import LiveRtspSource, RtspAuthenticationError

from .conftest import bearer, make_user

KEY = bytes(range(32))
PASSWORD = "Dvr@Gayathri#2026"


# ── the provider on its own ──────────────────────────────────────────────────


def _store_with(recorder_id: str, password: str, *, key: bytes = KEY) -> RecorderCredentialStore:
    store = RecorderCredentialStore()
    store._key = key  # noqa: SLF001 - a configured key, without a settings object
    store.put(recorder_id, seal(password, key=key))
    return store


def test_a_recorder_reference_opens_from_the_store() -> None:
    provider = RecorderSecretProvider(
        _store_with("r1", PASSWORD), fallback=EnvironmentSecretProvider({})
    )
    assert provider.resolve("recorder:r1") == PASSWORD


def test_an_environment_reference_still_resolves_exactly_as_before() -> None:
    """Existing deployments keep working with nothing typed in."""
    provider = RecorderSecretProvider(
        RecorderCredentialStore(),
        fallback=EnvironmentSecretProvider({"CCTV_PASSWORD": "from-the-env"}),
    )
    assert provider.resolve("env:CCTV_PASSWORD") == "from-the-env"


def test_an_unknown_recorder_is_a_missing_credential_not_an_empty_one() -> None:
    """An empty password builds a URL that authenticates as nobody, and the
    failure would name the socket rather than the cause."""
    provider = RecorderSecretProvider(
        _store_with("r1", PASSWORD), fallback=EnvironmentSecretProvider({})
    )
    with pytest.raises(MissingSecretError, match="r2"):
        provider.resolve("recorder:r2")


def test_a_wrong_master_key_is_a_missing_credential_on_that_camera_only() -> None:
    """Reported through the path the RTSP source already handles, so one
    recorder with an unreadable password stops only its own cameras."""
    store = _store_with("r1", PASSWORD)
    store._key = bytes(reversed(range(32)))  # noqa: SLF001
    provider = RecorderSecretProvider(store, fallback=EnvironmentSecretProvider({}))

    with pytest.raises(MissingSecretError, match="cannot be read"):
        provider.resolve("recorder:r1")


def test_the_error_never_carries_the_password() -> None:
    store = _store_with("r1", PASSWORD)
    store._key = bytes(reversed(range(32)))  # noqa: SLF001
    provider = RecorderSecretProvider(store, fallback=EnvironmentSecretProvider({}))

    with pytest.raises(MissingSecretError) as caught:
        provider.resolve("recorder:r1")
    assert PASSWORD not in str(caught.value)


def test_the_store_holds_ciphertext_between_connections() -> None:
    """What sits in memory while nothing is dialling is sealed. Plaintext exists
    only inside `resolve`."""
    store = _store_with("r1", PASSWORD)
    held = store._sealed["r1"]  # noqa: SLF001
    assert isinstance(held, SealedSecret)
    assert PASSWORD.encode() not in held.ciphertext
    assert PASSWORD not in repr(held)


def test_sealed_passwords_with_no_key_to_open_them_fail_loudly() -> None:
    from app.domain.recorder_secrets import MissingMasterKeyError

    store = RecorderCredentialStore()
    with pytest.raises(MissingMasterKeyError, match="2 recorder"):
        store.require_key_for(2)
    # And a deployment that holds none needs no key at all.
    store.require_key_for(0)


# ── brand → URL ──────────────────────────────────────────────────────────────


def _camera(recorder, *, channel=3, stream_type="main"):
    return SimpleNamespace(
        organization_id="org-g",
        camera_key="cam-03",
        channel=channel,
        stream_type=stream_type,
        analysis_fps=4.0,
        enabled=True,
        recorder=recorder,
    )


def _recorder(**overrides):
    values = {
        "id": "r1",
        "host": "192.168.1.20",
        "rtsp_port": 554,
        "username": "admin",
        "credential_ref": "recorder:r1",
        "brand": "hikvision",
        "path_template": "",
        "stream_main": None,
        "stream_sub": None,
        "is_active": True,
        "name": "Gayathri DVR",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("brand", "stream", "path"),
    [
        ("hikvision", "main", "/Streaming/Channels/301"),
        ("hikvision", "sub", "/Streaming/Channels/302"),
        ("dahua", "main", "/cam/realmonitor?channel=3&subtype=0"),
        ("dahua", "sub", "/cam/realmonitor?channel=3&subtype=1"),
    ],
)
def test_each_brand_builds_its_own_url(brand: str, stream: str, path: str) -> None:
    config = to_rtsp_config(_camera(_recorder(brand=brand), stream_type=stream))
    assert config.path() == path
    assert config.redacted_uri() == f"rtsp://***:***@192.168.1.20:554{path}"


def test_a_custom_recorder_uses_the_template_it_was_given() -> None:
    recorder = _recorder(
        brand="custom", path_template="/live/ch{channel}/s{subtype}", stream_main=0, stream_sub=1
    )
    assert to_rtsp_config(_camera(recorder, stream_type="sub")).path() == "/live/ch3/s1"


def test_the_config_carries_a_reference_and_never_a_password() -> None:
    config = to_rtsp_config(_camera(_recorder()))
    assert config.credential_ref == "recorder:r1"
    assert PASSWORD not in repr(config)


def test_a_camera_on_an_inactive_recorder_is_not_dialable() -> None:
    assert is_dialable(_camera(_recorder()))
    assert not is_dialable(_camera(_recorder(is_active=False)))
    assert not is_dialable(_camera(_recorder(host="")))


# ── the whole chain, through the API ─────────────────────────────────────────


@pytest.fixture
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
    return seeded


async def test_a_recorder_added_in_the_application_is_dialled_with_its_own_password(
    estate, client: AsyncClient
):
    """The acceptance path, minus the network. Nothing here edits code, a
    config file or a database row by hand."""
    headers = await bearer(client, "admin@example.com")
    created = await client.post(
        "/api/v1/recorders",
        json={
            "name": "Gayathri DVR",
            "host": "192.168.1.20",
            "rtsp_port": 554,
            "username": "admin",
            "password": PASSWORD,
            "brand": "hikvision",
            "cameras": [{"channel": 3, "name": "Hall", "zone_id": "zone-kitchen"}],
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    recorder_id = created.json()["id"]
    camera_key = created.json()["cameras"][0]["camera_key"]

    # The runtime's own provider, loaded the way boot loads it.
    provider = estate.state.credential_provider
    async with estate.state.database.session_scope() as session:
        estate.state.recorder_credentials.replace(await load_sealed(session))
        camera = await CameraService(session).get(organization_id="org-test", camera_key=camera_key)
        config = to_rtsp_config(camera)

    assert config.credential_ref == recorder_reference(recorder_id)

    dialled: list[str] = []

    def opener(uri: str):
        dialled.append(uri)
        raise RtspAuthenticationError("stop after dialling; there is no DVR here")

    source = LiveRtspSource(config, secrets=provider, opener=opener)
    frames = [frame async for frame in source.frames()]

    assert frames == []
    assert dialled == ["rtsp://admin:Dvr%40Gayathri%232026@192.168.1.20:554/Streaming/Channels/302"]
    # And the source says what happened without repeating the password.
    assert PASSWORD not in source.status.last_error
    assert "Dvr%40Gayathri" not in source.status.last_error


async def test_a_changed_password_is_used_without_a_restart(estate, client: AsyncClient):
    """The store is updated the moment the password changes, so the next
    connection uses it — nobody has to restart the server."""
    headers = await bearer(client, "admin@example.com")
    created = (
        await client.post(
            "/api/v1/recorders",
            json={
                "name": "Gayathri DVR",
                "host": "192.168.1.20",
                "rtsp_port": 554,
                "username": "admin",
                "password": "first-password",
                "brand": "dahua",
            },
            headers=headers,
        )
    ).json()
    provider = estate.state.credential_provider
    assert provider.resolve(f"recorder:{created['id']}") == "first-password"

    await client.put(
        f"/api/v1/recorders/{created['id']}/credential",
        json={"password": "second-password"},
        headers=headers,
    )
    assert provider.resolve(f"recorder:{created['id']}") == "second-password"
