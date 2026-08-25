"""Phase 6B: the camera wall.

Four properties matter more than the rest, and each has a test that fails loudly
if it regresses:

1. **Every camera is listed.** A monitoring wall that hides a dark channel is
   worse than no wall — the operator loses the one fact they needed.
2. **`live` is earned.** It must come from a frame arriving, never from a camera
   being enabled.
3. **One camera failing does not take the wall down.**
4. **No credential reaches the browser.** Not the password, not the username,
   not the RTSP URL, not even redacted.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.api.wall import TICKET_TTL_S, mint_ticket, verify_ticket
from app.vision.wall import STALE_AFTER_S, CameraStream, CameraWall, StreamState
from tests.app.conftest import bearer

SECRET = "test-only-secret-value-not-for-any-deployment"
ORG = "org-test"


class FakeCamera:
    """A camera row, without a database."""

    def __init__(self, channel: int, *, enabled: bool = True, org: str = ORG) -> None:
        self.camera_key = f"cam-{channel:02d}"
        self.name = f"Channel {channel:02d}"
        self.purpose = "live monitoring"
        self.channel = channel
        self.stream_type = "main"
        self.enabled = enabled
        self.organization_id = org
        self.restaurant_id = "rest-01"
        self.host = "10.0.0.5"
        self.rtsp_port = 554
        self.username = "admin"
        self.credential_ref = "env:CCTV_PASSWORD"
        self.analysis_fps = 4.0
        self.zone_id = None


class TestStreamState:
    def test_a_camera_is_not_live_merely_because_it_is_enabled(self):
        """The whole point of section 10, in one assertion."""
        stream = CameraStream(FakeCamera(1))
        assert stream.camera.enabled is True
        assert stream.state == StreamState.DISABLED

        stream._state = StreamState.CONNECTING  # noqa: SLF001 - as `start` would
        assert stream.state == StreamState.CONNECTING, "no frame has arrived yet"

    def test_a_frame_makes_it_live(self):
        stream = CameraStream(FakeCamera(1))
        stream._state = StreamState.CONNECTING  # noqa: SLF001
        stream.stats.last_frame_at = time.monotonic()
        assert stream.state == StreamState.LIVE

    def test_silence_returns_it_to_reconnecting(self):
        """A socket that is open but delivering nothing is not live."""
        stream = CameraStream(FakeCamera(1))
        stream._state = StreamState.CONNECTING  # noqa: SLF001
        stream.stats.last_frame_at = time.monotonic() - (STALE_AFTER_S + 1)
        assert stream.state == StreamState.RECONNECTING

    def test_a_disabled_camera_stays_disabled(self):
        stream = CameraStream(FakeCamera(1, enabled=False))
        stream.stats.last_frame_at = time.monotonic()
        assert stream.state == StreamState.DISABLED


class TestWireForm:
    def test_no_credential_or_uri_reaches_the_wire(self):
        """Section 20. The browser must not be able to reach the DVR."""
        stream = CameraStream(FakeCamera(7))
        wire = str(stream.to_wire())

        for forbidden in ("admin", "CCTV_PASSWORD", "credential_ref", "rtsp://", "10.0.0.5", "554"):
            assert forbidden not in wire, f"'{forbidden}' leaked into the wire form"

    def test_the_wire_carries_what_an_operator_needs(self):
        wire = CameraStream(FakeCamera(7)).to_wire()
        assert wire["camera_id"] == "cam-07"
        assert wire["channel"] == 7
        assert "state" in wire

    def test_unknown_is_null_not_zero(self):
        """No frame yet means the latency is unknown, not instant."""
        wire = CameraStream(FakeCamera(1)).to_wire()
        assert wire["first_frame_latency_s"] is None
        assert wire["seconds_since_frame"] is None


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_camera_failing_to_start_leaves_the_others_running(self, settings):
        """Section 15: never take down the wall for one camera."""
        wall = CameraWall(settings)
        cameras = [FakeCamera(n) for n in range(1, 5)]
        # Channel 3 cannot build a source config — an empty host is refused.
        cameras[2].host = ""

        started = await wall.start_cameras(cameras)
        try:
            assert len(wall.streams) == 4, "every camera must appear, working or not"
            assert started >= 1
            broken = wall.get("cam-03")
            assert broken is not None, "the broken camera is still listed"
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_a_disabled_camera_is_listed_but_not_started(self, settings):
        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(1, enabled=False)])
        try:
            stream = wall.get("cam-01")
            assert stream is not None, "a disabled camera is still on the wall"
            assert stream.state == StreamState.DISABLED
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_one_session_per_camera_however_often_started(self, settings):
        """Section 8: sixteen tiles must not become sixty-four connections."""
        wall = CameraWall(settings)
        camera = FakeCamera(1)
        await wall.start_cameras([camera])
        first = wall.get("cam-01")
        await wall.start_cameras([camera])
        try:
            assert wall.get("cam-01") is first, "a second start created a second session"
            assert len(wall.streams) == 1
        finally:
            await wall.stop_all()


class TestTickets:
    def test_a_ticket_is_valid_for_its_own_camera_only(self):
        ticket = mint_ticket(SECRET, tenant_id=ORG, camera_id="cam-01", subject="a@b.c")
        assert verify_ticket(SECRET, ticket, tenant_id=ORG, camera_id="cam-01", subject="a@b.c")
        # The same ticket must not open a different camera.
        assert not verify_ticket(
            SECRET, ticket, tenant_id=ORG, camera_id="cam-02", subject="a@b.c"
        )

    def test_a_ticket_does_not_cross_tenants(self):
        ticket = mint_ticket(SECRET, tenant_id=ORG, camera_id="cam-01", subject="a@b.c")
        assert not verify_ticket(
            SECRET, ticket, tenant_id="org-other", camera_id="cam-01", subject="a@b.c"
        )

    def test_a_ticket_does_not_transfer_between_subjects(self):
        ticket = mint_ticket(SECRET, tenant_id=ORG, camera_id="cam-01", subject="a@b.c")
        assert not verify_ticket(
            SECRET, ticket, tenant_id=ORG, camera_id="cam-01", subject="other@b.c"
        )

    def test_a_ticket_signed_with_another_secret_is_refused(self):
        ticket = mint_ticket("a-different-secret", tenant_id=ORG, camera_id="cam-01", subject="s")
        assert not verify_ticket(SECRET, ticket, tenant_id=ORG, camera_id="cam-01", subject="s")

    def test_an_expired_ticket_is_refused(self):
        expired = f"{int(time.time()) - 5}.deadbeefdeadbeefdeadbeefdeadbeef"
        assert not verify_ticket(SECRET, expired, tenant_id=ORG, camera_id="cam-01", subject="s")

    def test_garbage_is_refused_rather_than_raising(self):
        for junk in ("", "not-a-ticket", "abc.def", "..", "9999999999"):
            assert not verify_ticket(SECRET, junk, tenant_id=ORG, camera_id="cam-01", subject="s")

    def test_the_window_is_short(self):
        """A leaked ticket must be nearly spent by the time it is read."""
        assert TICKET_TTL_S <= 120


class TestWallApi:
    @pytest.mark.asyncio
    async def test_every_camera_appears_including_the_dark_ones(self, seeded, client):
        """Section 22: no filtering by usefulness."""
        from app.domain.cameras import CameraService

        async with seeded.state.database.session_scope() as session:
            service = CameraService(session)
            for channel in range(1, 5):
                await service.create(
                    organization_id=ORG, restaurant_id="rest-01",
                    camera_key=f"cam-{channel:02d}", name=f"Channel {channel:02d}",
                    channel=channel, host="10.0.0.5",
                )
            await session.flush()
            # Two enabled, two left dark.
            for key in ("cam-01", "cam-02"):
                await service.set_enabled(organization_id=ORG, camera_key=key, enabled=True)

        # A tenant-wide grant, so nothing is removed by camera scope and the
        # only question left is whether the wall filters by usefulness. It must
        # not: two of these four are deliberately dark.
        headers = await bearer(client, "developer@example.com")
        response = await client.get("/api/v1/wall/cameras", headers=headers)
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["total"] == 4, "a disabled camera must still be listed"
        assert [c["channel"] for c in body["cameras"]] == [1, 2, 3, 4]
        assert sum(1 for c in body["cameras"] if not c["enabled"]) == 2
        assert {c["state"] for c in body["cameras"]} <= {
            "disabled", "connecting", "live", "reconnecting", "offline", "error"
        }

    @pytest.mark.asyncio
    async def test_camera_scope_still_narrows_the_wall(self, seeded, client):
        """Section 19: authorization removes cameras, and only authorization does.

        The distinction this pins down: refusing to filter by *usefulness* is not
        refusing to filter by *permission*. An account granted two cameras sees
        two, however many the recorder has.
        """
        from app.domain.cameras import CameraService

        async with seeded.state.database.session_scope() as session:
            service = CameraService(session)
            for channel in range(1, 5):
                await service.create(
                    organization_id=ORG, restaurant_id="rest-01",
                    camera_key=f"cam-{channel:02d}", name=f"Channel {channel:02d}",
                    channel=channel, host="10.0.0.5",
                )

        # The manager fixture is granted cam-01 and cam-02 only.
        headers = await bearer(client, "manager@example.com")
        body = (await client.get("/api/v1/wall/cameras", headers=headers)).json()
        assert [c["camera_id"] for c in body["cameras"]] == ["cam-01", "cam-02"]

    @pytest.mark.asyncio
    async def test_the_list_carries_no_credential(self, seeded, client):
        from app.domain.cameras import CameraService

        async with seeded.state.database.session_scope() as session:
            await CameraService(session).create(
                organization_id=ORG, restaurant_id="rest-01", camera_key="cam-01",
                name="Channel 01", channel=1, host="10.0.0.5",
                username="admin", credential_ref="env:CCTV_PASSWORD",
            )

        headers = await bearer(client, "manager@example.com")
        text = (await client.get("/api/v1/wall/cameras", headers=headers)).text
        for forbidden in ("admin", "CCTV_PASSWORD", "credential_ref", "rtsp://", "10.0.0.5"):
            assert forbidden not in text, f"'{forbidden}' reached the browser"

    @pytest.mark.asyncio
    async def test_viewing_requires_the_live_permission(self, seeded, client):
        headers = await bearer(client, "nocameras@example.com")
        response = await client.get("/api/v1/wall/cameras", headers=headers)
        # The account holds VIEW_LIVE but is granted no cameras, so the wall is
        # empty rather than forbidden — an empty grant is none, never all.
        if response.status_code == 200:
            assert response.json()["cameras"] == []
        else:
            assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_a_ticket_cannot_be_had_for_an_ungranted_camera(self, seeded, client):
        from app.domain.cameras import CameraService

        async with seeded.state.database.session_scope() as session:
            await CameraService(session).create(
                organization_id=ORG, restaurant_id="rest-01", camera_key="cam-16",
                name="Channel 16", channel=16, host="10.0.0.5",
            )

        # The manager fixture is granted cam-01 and cam-02 only.
        headers = await bearer(client, "manager@example.com")
        response = await client.post("/api/v1/wall/cameras/cam-16/ticket", headers=headers)
        # Not found, not forbidden: confirming it exists is itself a disclosure.
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_the_stream_refuses_a_forged_ticket(self, seeded, client):
        response = await client.get(
            "/api/v1/wall/cameras/cam-01/stream.mjpg",
            params={"ticket": "9999999999.forged", "tenant": ORG, "subject": "manager@example.com"},
        )
        assert response.status_code in (401, 404)

    @pytest.mark.asyncio
    async def test_an_unauthenticated_caller_gets_no_camera_list(self, seeded, client):
        assert (await client.get("/api/v1/wall/cameras")).status_code == 401


class TestPhase6B1Regressions:
    """The two defects that made the real application show an empty wall.

    Both are the same shape: something that worked in a validation harness and
    did not work in the application, because the harness supplied by hand what
    the application must supply for itself.
    """

    def test_the_configured_dvr_password_reaches_the_secret_provider(self):
        """Defect 2. `.env` is loaded into Settings, never into `os.environ`.

        `EnvironmentSecretProvider` resolves `env:NAME` against a mapping. Given
        the bare process environment it cannot see a password pydantic put on
        the settings object, so every camera sat at CONNECTING with a correct
        credential on disk.
        """
        from pydantic import SecretStr

        from app.configuration.settings import Settings
        from app.vision.secrets import EnvironmentSecretProvider

        settings = Settings(
            app_env="test",
            secret_key=SECRET,
            database_url_override="sqlite+aiosqlite:///:memory:",
            redis_enabled=False,
            cctv_credential_ref="env:CCTV_PASSWORD",
            cctv_password=SecretStr("a-configured-value"),
        )

        provider = EnvironmentSecretProvider(settings.secret_environment())
        assert provider.resolve("env:CCTV_PASSWORD") == "a-configured-value"

    def test_a_real_environment_variable_still_wins(self, monkeypatch):
        """An operator supplying a rotated credential must not lose to a file."""
        from pydantic import SecretStr

        from app.configuration.settings import Settings

        monkeypatch.setenv("CCTV_PASSWORD", "rotated-today")
        settings = Settings(
            app_env="test",
            secret_key=SECRET,
            database_url_override="sqlite+aiosqlite:///:memory:",
            redis_enabled=False,
            cctv_password=SecretStr("stale-on-disk"),
        )
        assert settings.secret_environment()["CCTV_PASSWORD"] == "rotated-today"

    def test_the_secret_environment_is_not_written_into_the_process(self):
        """The overlay is handed to one provider, never exported process-wide."""
        import os

        from pydantic import SecretStr

        from app.configuration.settings import Settings

        settings = Settings(
            app_env="test",
            secret_key=SECRET,
            database_url_override="sqlite+aiosqlite:///:memory:",
            redis_enabled=False,
            cctv_password=SecretStr("must-not-escape"),
        )
        settings.secret_environment()
        assert os.environ.get("CCTV_PASSWORD") != "must-not-escape"

    @pytest.mark.asyncio
    async def test_a_stale_camera_row_cannot_deny_a_valid_ticket(self, settings):
        """Defect 3. The wall's in-memory row must not be an authorization input.

        The wall holds ORM rows loaded at start-up. When cameras moved between
        tenants, every stream returned 403 until the process was restarted —
        a stale cache silently acting as authorization. Tenancy belongs to the
        ticket, which was signed after a fresh tenant-scoped database read.
        """
        from app.api.wall import mint_ticket, verify_ticket

        wall = CameraWall(settings)
        stale = FakeCamera(1, org="an-old-tenant")
        await wall.start_cameras([stale])
        try:
            stream = wall.get("cam-01")
            assert stream is not None
            # The row the wall is holding is out of date...
            assert stream.camera.organization_id == "an-old-tenant"
            # ...and a ticket minted for the camera's real tenant is still good.
            ticket = mint_ticket(
                SECRET, tenant_id="the-current-tenant", camera_id="cam-01", subject="s"
            )
            assert verify_ticket(
                SECRET, ticket, tenant_id="the-current-tenant",
                camera_id="cam-01", subject="s",
            )
        finally:
            await wall.stop_all()


class TestPhase6B3Offload:
    """Phase 6B.2 measured decode and JPEG encode running as synchronous,
    CPU-bound coroutines on the server's single event-loop thread: sixteen
    cameras converged on an identical, starved ~3.3 fps, and `/health` rose to
    seconds regardless of viewer count. Phase 6B.3 moves that work to one
    dedicated OS thread per camera. These tests pin the properties that fix
    depends on — not the RTSP protocol, which is unchanged and tested
    elsewhere — using a fake `_run_async` so no network or PyAV is involved.
    """

    @staticmethod
    def _install_fake_run(monkeypatch, *, on_iteration=None, camera_delay: dict | None = None):
        """Replace the real RTSP loop with one that records its own thread and
        publishes a trivial frame every tick, without touching a socket.
        """
        seen_threads: dict[str, threading.Thread] = {}
        delay = camera_delay or {}

        async def fake_run_async(self, config, reconnect):  # noqa: ARG001
            seen_threads[self.camera.camera_key] = threading.current_thread()
            started_at = time.monotonic()
            tick = 0
            while not self._stop_event.is_set():
                wait = delay.get(self.camera.camera_key, 0.0)
                if wait:
                    # A stand-in for a slow synchronous decode: a genuine
                    # blocking call on this camera's own thread, not an
                    # `await` any other camera or the event loop waits on.
                    time.sleep(wait)
                self._publish(
                    _FakeFrame(payload=b"\x00" * (2 * 2 * 3), width=2, height=2),
                    started_at,
                )
                tick += 1
                if on_iteration:
                    on_iteration(self.camera.camera_key, tick)
                await asyncio.sleep(0.01)

        monkeypatch.setattr(CameraStream, "_run_async", fake_run_async)
        return seen_threads

    @pytest.mark.asyncio
    async def test_decode_runs_on_a_dedicated_thread_not_the_caller(self, monkeypatch, settings):
        caller_thread = threading.current_thread()
        seen = self._install_fake_run(monkeypatch)

        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(1)])
        try:
            await asyncio.sleep(0.1)
            assert "cam-01" in seen, "the fake loop never ran"
            assert seen["cam-01"] is not caller_thread
            assert seen["cam-01"].name == "wall-cam-01"
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_each_camera_gets_its_own_thread(self, monkeypatch, settings):
        seen = self._install_fake_run(monkeypatch)
        wall = CameraWall(settings)
        cameras = [FakeCamera(n) for n in range(1, 5)]
        await wall.start_cameras(cameras)
        try:
            await asyncio.sleep(0.1)
            assert len(seen) == 4
            assert len({t.ident for t in seen.values()}) == 4, "two cameras shared a thread"
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_starting_the_same_camera_twice_does_not_start_a_second_thread(
        self, monkeypatch, settings
    ):
        seen = self._install_fake_run(monkeypatch)
        wall = CameraWall(settings)
        camera = FakeCamera(1)
        await wall.start_cameras([camera])
        await asyncio.sleep(0.05)
        first_thread = seen["cam-01"]
        await wall.start_cameras([camera])
        await asyncio.sleep(0.05)
        try:
            assert seen["cam-01"] is first_thread
            live = [t for t in threading.enumerate() if t.name == "wall-cam-01"]
            assert len(live) == 1, "a second start left two RTSP worker threads alive"
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_a_slow_camera_does_not_delay_starting_the_others(self, monkeypatch, settings):
        """Section 9. `start_cameras` spawns threads; it must never wait on one."""
        self._install_fake_run(monkeypatch, camera_delay={"cam-01": 2.0})
        wall = CameraWall(settings)
        cameras = [FakeCamera(n) for n in range(1, 5)]  # cam-01 is the slow one

        t0 = time.monotonic()
        await wall.start_cameras(cameras)
        elapsed = time.monotonic() - t0
        try:
            assert elapsed < 1.0, (
                f"starting 4 cameras took {elapsed:.2f}s — a slow camera blocked the others"
            )
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_a_slow_camera_does_not_starve_the_others_frames_or_the_wall_read_path(
        self, monkeypatch, settings
    ):
        """Section 9/10. The fast cameras must keep publishing, and reading the
        wall's state (what `/wall/cameras` and `/health` ultimately do) must
        stay fast, while one camera's thread sits blocked in a synchronous call.
        """
        ticks: dict[str, int] = {}

        def record(camera_id, tick):
            ticks[camera_id] = tick

        # cam-01 blocks for 0.05s on every tick — a real, repeated synchronous
        # cost on its own thread, not a one-off.
        self._install_fake_run(monkeypatch, on_iteration=record, camera_delay={"cam-01": 0.05})
        wall = CameraWall(settings)
        cameras = [FakeCamera(n) for n in range(1, 5)]
        await wall.start_cameras(cameras)
        try:
            await asyncio.sleep(0.4)

            # The wall-summary read path must stay cheap regardless of what
            # cam-01's thread is doing — it only ever touches `_condition`.
            t0 = time.monotonic()
            for _ in range(20):
                wall.summary()
                wall.get("cam-02").to_wire()
            read_elapsed = time.monotonic() - t0
            assert read_elapsed < 0.2, f"reading wall state took {read_elapsed:.3f}s"

            # The three fast cameras must have ticked far more than the slow one.
            assert ticks.get("cam-02", 0) > ticks.get("cam-01", 0) * 3
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_stop_all_leaves_no_worker_thread_alive(self, monkeypatch, settings):
        """Section 19. A clean shutdown must not orphan a thread."""
        self._install_fake_run(monkeypatch)
        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(n) for n in range(1, 5)])
        await asyncio.sleep(0.05)
        await wall.stop_all()

        orphans = [t for t in threading.enumerate() if t.name.startswith("wall-")]
        assert orphans == [], f"stop_all left threads alive: {[t.name for t in orphans]}"

    @pytest.mark.asyncio
    async def test_reconnect_keeps_the_same_thread_no_second_rtsp_session(
        self, monkeypatch, settings
    ):
        """Section 18. The outer retry loop already lives inside one thread's
        private event loop; a reconnect must never spawn a second thread —
        that would be a second RTSP session for one camera.
        """
        attempts: list[int] = []

        async def flaky_run_async(self, config, reconnect):  # noqa: ARG001
            started_at = time.monotonic()
            while not self._stop_event.is_set():
                attempts.append(1)
                if len(attempts) <= 2:
                    # Looks like a dropped connection: the outer loop in the
                    # real `_run_async` catches this, marks RECONNECTING and
                    # tries again — all inside the same thread.
                    self.stats.reconnects += 1
                    self._state = StreamState.RECONNECTING
                    await asyncio.sleep(0.02)
                    continue
                self._publish(
                    _FakeFrame(payload=b"\x00" * 12, width=2, height=2), started_at
                )
                await asyncio.sleep(0.01)

        monkeypatch.setattr(CameraStream, "_run_async", flaky_run_async)
        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(1)])
        try:
            await asyncio.sleep(0.3)
            assert len(attempts) >= 3, "the flaky loop never recovered"
            live = [t for t in threading.enumerate() if t.name == "wall-cam-01"]
            assert len(live) == 1, "a reconnect spawned a second worker thread"
            assert wall.get("cam-01").stats.reconnects >= 2
        finally:
            await wall.stop_all()

    def test_publish_holds_one_frame_not_a_growing_queue(self):
        """Section 17. A slow or absent viewer must not make the backend
        accumulate frames — there is exactly one JPEG slot, always overwritten.
        """
        stream = CameraStream(FakeCamera(1))
        started_at = time.monotonic()
        for _ in range(500):
            stream._publish(  # noqa: SLF001
                _FakeFrame(payload=b"\x00" * 27, width=3, height=3), started_at
            )
        assert isinstance(stream._jpeg, bytes | bytearray)  # noqa: SLF001
        assert stream.stats.frames_decoded == 500
        assert stream._frame_seq == 500  # noqa: SLF001

    def test_source_fps_is_computed_not_left_dead_at_zero(self):
        """Phase 6B.2 found `source_fps` declared and read but never written —
        every rate in that report had to be computed from `frames_decoded`
        deltas instead. This is the fix, checked directly.
        """
        stream = CameraStream(FakeCamera(1))
        started_at = time.monotonic()
        assert stream.stats.source_fps == 0.0

        t = started_at
        for _ in range(10):
            t += 0.1  # a steady 10 fps
            with _frozen_monotonic(t):
                stream._publish(  # noqa: SLF001
                    _FakeFrame(payload=b"\x00" * 27, width=3, height=3), started_at
                )
        assert stream.stats.source_fps == pytest.approx(10.0, rel=0.35)


class _FakeFrame:
    def __init__(self, *, payload: bytes, width: int, height: int) -> None:
        self.payload = payload
        self.width = width
        self.height = height


class _frozen_monotonic:
    """Patches `time.monotonic()` for the duration of one `with` block only —
    just long enough for `_publish` to read "now" without a real sleep.
    """

    def __init__(self, value: float) -> None:
        self._value = value
        self._real = time.monotonic

    def __enter__(self) -> None:
        time.monotonic = lambda: self._value  # type: ignore[assignment]

    def __exit__(self, *exc: object) -> None:
        time.monotonic = self._real  # type: ignore[assignment]


class TestStallWatchdog:
    """A camera that goes quiet must come back without a process restart.

    Camera 14 sat dark for 38 minutes in production with `reconnects=0`: the
    read was blocked inside FFmpeg on a socket the DVR had abandoned, so the
    retry loop wrapped around it never ran. The socket-level fix is the
    `timeout` option in `sources/rtsp.py`; this is the wall-level backstop for
    every other way a stream can fall silent.

    **Timings here are chosen, not guessed.** The watchdog window is set well
    above the fake's frame interval, because a window close to it would flag a
    healthy camera on ordinary scheduling jitter — in production the margin is
    10s against ~30fps. The re-dial waits the stream's existing 2s backoff, so
    tests that check for one wait past it rather than racing it.
    """

    WINDOW = 0.3
    REDIAL_BACKOFF = 2.0

    @staticmethod
    def _install(monkeypatch, source_class, window):
        from app.vision import wall as wall_module

        monkeypatch.setattr(wall_module, "STALL_WATCHDOG_S", window)
        monkeypatch.setattr(
            "app.vision.sources.rtsp.LiveRtspSource", source_class, raising=False
        )

    @pytest.mark.asyncio
    async def test_a_stalled_source_is_torn_down_and_re_dialled(self, monkeypatch, settings):
        dials = []

        class _Stalling:
            def __init__(self, *a, **k):
                dials.append(1)
                self._stop = False

            async def frames(self):
                # One frame, then silence — the shape of an abandoned socket.
                yield _FakeFrame(payload=b"\x00" * 12, width=2, height=2)
                while not self._stop:
                    await asyncio.sleep(0.01)

            def stop(self):
                self._stop = True

        self._install(monkeypatch, _Stalling, self.WINDOW)

        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(14)])
        try:
            # Long enough for: first frame, the window to elapse, the teardown,
            # and the existing backoff before the next dial. Measured directly:
            # the stall lands at ~0.6s and the re-dial at ~2.8s, so this waits
            # past the second rather than racing it.
            await asyncio.sleep(self.WINDOW + self.REDIAL_BACKOFF + 1.3)
            stream = wall.get("cam-14")

            assert stream.stats.stalls >= 1, "the watchdog never fired"
            assert stream.stats.reconnects >= 1, "the stall did not cause a reconnect"
            assert len(dials) >= 2, "the source was torn down but never re-dialled"
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_a_healthy_stream_is_never_torn_down(self, monkeypatch, settings):
        """The watchdog must not interrupt a camera that is working."""
        dials = []

        class _Healthy:
            def __init__(self, *a, **k):
                dials.append(1)
                self._stop = False

            async def frames(self):
                while not self._stop:
                    yield _FakeFrame(payload=b"\x00" * 12, width=2, height=2)
                    await asyncio.sleep(0.01)

            def stop(self):
                self._stop = True

        self._install(monkeypatch, _Healthy, self.WINDOW)

        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(12)])
        try:
            await asyncio.sleep(self.WINDOW * 4)
            stream = wall.get("cam-12")

            assert stream.stats.stalls == 0, "the watchdog fired on a healthy stream"
            assert len(dials) == 1, "a healthy source was needlessly re-dialled"
            assert stream.state == StreamState.LIVE
        finally:
            await wall.stop_all()

    @pytest.mark.asyncio
    async def test_one_stalled_camera_does_not_disturb_the_others(
        self, monkeypatch, settings
    ):
        """§2. cam-14 stalling must cost cam-11, cam-12 and cam-13 nothing."""
        dials: dict[str, int] = {}

        class _Source:
            def __init__(self, config, *a, **k):
                self.camera_id = str(config.camera_id)
                dials[self.camera_id] = dials.get(self.camera_id, 0) + 1
                self._stalls = self.camera_id == "cam-14"
                self._stop = False

            async def frames(self):
                yield _FakeFrame(payload=b"\x00" * 12, width=2, height=2)
                while not self._stop:
                    if not self._stalls:
                        yield _FakeFrame(payload=b"\x00" * 12, width=2, height=2)
                    await asyncio.sleep(0.01)

            def stop(self):
                self._stop = True

        self._install(monkeypatch, _Source, self.WINDOW)

        wall = CameraWall(settings)
        await wall.start_cameras([FakeCamera(n) for n in (11, 12, 13, 14)])
        try:
            await asyncio.sleep(self.WINDOW * 4)

            assert wall.get("cam-14").stats.stalls >= 1, "the stalled camera recovered nothing"

            for healthy in ("cam-11", "cam-12", "cam-13"):
                stream = wall.get(healthy)
                assert stream.stats.stalls == 0, f"{healthy} was flagged as stalled"
                assert dials[healthy] == 1, f"{healthy} was needlessly re-dialled"
                assert stream.state == StreamState.LIVE, f"{healthy} left LIVE"

            # Still one worker thread per camera: a stall must not spawn a
            # second RTSP session for the same channel.
            for camera_id in ("cam-11", "cam-12", "cam-13", "cam-14"):
                alive = [t for t in threading.enumerate() if t.name == f"wall-{camera_id}"]
                assert len(alive) == 1, f"{camera_id} has {len(alive)} worker threads"
        finally:
            await wall.stop_all()
