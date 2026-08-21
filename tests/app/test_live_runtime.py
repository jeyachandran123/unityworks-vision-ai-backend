"""Phase 3 — the live runtime.

Test labels used throughout:

    LIVE_SIMULATION   a continuous generated source. Exercises live semantics —
                      unbounded operation, backpressure, reconnect — without
                      pretending anything is a camera.
    REPLAY            recorded media.
    LIVE              a real RTSP camera. **None of these run**: TCP 554 at the
                      restaurant is filtered, and a test that faked a camera
                      would be worse than no test.

The RTSP source is tested through an injected opener, so connect, reconnect,
authentication failure, decoder failure and credential redaction are all covered
without a network.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.configuration.settings import Settings
from app.vision.frames import (
    DropReason,
    FrameSampler,
    LiveFrame,
    LiveFrameQueue,
)
from app.vision.manager import LiveRuntime
from app.vision.secrets import (
    EnvironmentSecretProvider,
    MissingSecretError,
    SecretResolutionError,
)
from app.vision.session import SessionState
from app.vision.sources.base import CameraHealth, SourceKind, SourceState
from app.vision.sources.replay import SyntheticFrameSource
from app.vision.sources.rtsp import (
    REDACTED,
    LiveRtspSource,
    ReconnectPolicy,
    RtspAuthenticationError,
    RtspCameraConfig,
)

SECRET = "sup3r-s3cret-dvr-pw"


def frame(camera: str = "cam-01", sequence: int = 0, captured_ns: int | None = None) -> LiveFrame:
    now = time.time_ns()
    return LiveFrame(
        camera_id=camera,
        sequence=sequence,
        epoch=0,
        captured_at_ns=captured_ns if captured_ns is not None else now,
        received_at_ns=now,
        width=4,
        height=4,
        payload=b"\x00" * 48,
    )


def settings(**overrides) -> Settings:
    base = {
        "app_env": "test",
        "secret_key": "test-only-secret-value-not-for-any-deployment",
        "database_url_override": "sqlite+aiosqlite:///:memory:",
        "redis_enabled": False,
        "metrics_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


# ── bounded memory and backpressure ──────────────────────────────────────────


class TestBoundedQueue:
    """§9 — no unbounded structure in the live path."""

    def test_there_is_no_unbounded_mode(self) -> None:
        with pytest.raises(ValueError, match="no unbounded mode"):
            LiveFrameQueue(0)

    def test_it_never_exceeds_its_capacity(self) -> None:
        queue = LiveFrameQueue(4)
        for index in range(100):
            queue.put(frame(sequence=index))
        assert queue.depth == 4
        assert queue.stats.high_water == 4

    def test_it_drops_the_oldest_and_keeps_the_newest(self) -> None:
        """The policy, stated as a test.

        For live monitoring the newest frame answers the question being asked.
        Keeping the oldest would show an operator a world that has moved on.
        """
        queue = LiveFrameQueue(3)
        for index in range(6):
            queue.put(frame(sequence=index))

        remaining = []
        while queue.depth:
            item = queue._items.popleft()  # noqa: SLF001 - asserting internal order
            remaining.append(item.sequence)

        assert remaining == [3, 4, 5], "the three newest survived, in order"

    def test_a_drop_is_reported_not_silent(self) -> None:
        queue = LiveFrameQueue(1)
        assert queue.put(frame(sequence=0)) is None
        assert queue.put(frame(sequence=1)) is DropReason.QUEUE_FULL
        assert queue.stats.dropped_queue_full == 1

    def test_timestamps_are_never_reordered(self) -> None:
        queue = LiveFrameQueue(4)
        for index in range(20):
            queue.put(frame(sequence=index, captured_ns=1_000 + index))

        captured = [item.captured_at_ns for item in queue._items]  # noqa: SLF001
        assert captured == sorted(captured)

    def test_frames_are_never_duplicated(self) -> None:
        queue = LiveFrameQueue(8)
        for index in range(8):
            queue.put(frame(sequence=index))
        sequences = [item.sequence for item in queue._items]  # noqa: SLF001
        assert len(sequences) == len(set(sequences))

    def test_sampled_out_is_counted_separately_from_queue_full(self) -> None:
        """A healthy 25 fps camera and an overloaded one must not look alike."""
        queue = LiveFrameQueue(2)
        queue.record_sampled_out()
        queue.put(frame())
        queue.put(frame())
        queue.put(frame())

        assert queue.stats.dropped_sampled == 1
        assert queue.stats.dropped_queue_full == 1

    async def test_a_closed_queue_drains_then_returns_none(self) -> None:
        queue = LiveFrameQueue(4)
        queue.put(frame(sequence=1))
        queue.close()

        assert (await queue.get()).sequence == 1
        assert await queue.get() is None


class TestSampler:
    """§12 — sampling happens once, at the source boundary."""

    def test_the_first_frame_is_always_due(self) -> None:
        assert FrameSampler(4.0).accepts(1_000)

    def test_it_thins_a_fast_source_to_the_analysis_rate(self) -> None:
        sampler = FrameSampler(analysis_fps=4.0)  # one per 250 ms
        accepted = sum(
            1 for index in range(100) if sampler.accepts(index * 40_000_000)  # 25 fps
        )
        # 100 frames at 25 fps is 4 s; at 4 fps that is ~16 frames.
        assert 15 <= accepted <= 17

    def test_it_judges_on_capture_time_not_arrival(self) -> None:
        """A burst of buffered frames after a hiccup must not all pass at once."""
        sampler = FrameSampler(analysis_fps=1.0)
        assert sampler.accepts(0)
        # Ten frames that were captured within the same second.
        assert not any(sampler.accepts(index * 10_000_000) for index in range(1, 11))

    def test_a_reset_makes_the_next_frame_due(self) -> None:
        sampler = FrameSampler(1.0)
        sampler.accepts(0)
        sampler.reset()
        assert sampler.accepts(1)


# ── sources ──────────────────────────────────────────────────────────────────


class TestSourceLifecycle:
    """LIVE_SIMULATION — states are explicit and every transition is observable."""

    async def test_a_source_starts_created_and_holds_nothing(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        assert source.status.state is SourceState.CREATED
        assert source.status.frames_produced == 0

    async def test_it_reaches_running_and_then_stopped(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=3, interval_override_s=0)
        produced = [f async for f in source.frames()]

        assert len(produced) == 3
        assert source.status.state is SourceState.STOPPED
        states = [t.to_state for t in source.status.transitions]
        assert SourceState.CONNECTING in states
        assert SourceState.RUNNING in states

    async def test_every_transition_carries_a_reason_and_a_time(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=1, interval_override_s=0)
        [f async for f in source.frames()]

        for transition in source.status.transitions:
            assert transition.reason
            assert transition.at_ns > 0

    async def test_stopping_ends_it_cleanly(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=None, interval_override_s=0)
        seen = 0
        async for _ in source.frames():
            seen += 1
            if seen == 5:
                source.stop()
        assert source.status.state is SourceState.STOPPED

    async def test_camera_id_comes_from_configuration_never_from_pixels(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-kitchen-03", count=2, interval_override_s=0)
        async for produced in source.frames():
            assert produced.camera_id == "cam-kitchen-03"

    async def test_capture_time_is_preserved_and_distinct_from_arrival(self) -> None:
        """§11 — freshness ages against capture time, not arrival."""
        source = SyntheticFrameSource(
            camera_id="cam-01", fps=25.0, count=5, interval_override_s=0
        )
        produced = [f async for f in source.frames()]

        captured = [f.captured_at_ns for f in produced]
        assert captured == sorted(captured)
        # 25 fps → 40 ms between captures, regardless of how fast the producer ran.
        assert captured[1] - captured[0] == pytest.approx(40_000_000, rel=0.01)

        # The two clocks are different concepts, and the divergence is what
        # proves it: capture advances by the frame interval while arrival
        # advances by real elapsed time. (Frame 0 can coincide at nanosecond
        # resolution, which says nothing either way.)
        later = produced[-1]
        assert later.captured_at_ns != later.received_at_ns

        # Capture advanced by four simulated frame intervals — 160 ms — while
        # arrival advanced by however long the loop actually took. That the two
        # spans differ is the property: they are independent clocks, and
        # freshness must age against the first.
        capture_span = later.captured_at_ns - produced[0].captured_at_ns
        arrival_span = later.received_at_ns - produced[0].received_at_ns
        assert capture_span == pytest.approx(160_000_000, rel=0.01)
        assert arrival_span != capture_span

    async def test_a_replay_source_is_never_labelled_live(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        assert source.kind is SourceKind.REPLAY
        assert source.status.to_wire()["kind"] == "replay"


class TestCameraHealth:
    """§22 — health is derived from real source state."""

    def test_running_with_a_recent_frame_is_online(self) -> None:
        assert CameraHealth.of(SourceState.RUNNING, stale=False) is CameraHealth.ONLINE

    def test_running_without_a_recent_frame_is_degraded_not_online(self) -> None:
        """The frozen-frame failure. A silent camera is never healthy."""
        assert CameraHealth.of(SourceState.RUNNING, stale=True) is CameraHealth.DEGRADED

    def test_reconnecting_is_degraded_not_error(self) -> None:
        """Expected to recover. An operator should not be paged for it."""
        assert CameraHealth.of(SourceState.RECONNECTING) is CameraHealth.DEGRADED

    def test_error_is_error(self) -> None:
        assert CameraHealth.of(SourceState.ERROR) is CameraHealth.ERROR

    def test_stopped_is_offline_not_error(self) -> None:
        assert CameraHealth.of(SourceState.STOPPED) is CameraHealth.OFFLINE

    async def test_a_source_that_never_produced_is_not_online(self) -> None:
        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        source._transition(SourceState.RUNNING, "test")  # noqa: SLF001
        assert source.status.stale is True
        assert source.status.health is CameraHealth.DEGRADED
        assert source.status.producing is False


# ── credentials ──────────────────────────────────────────────────────────────


class TestSecretProvider:
    """§6 — Vision OS receives resolved configuration, never fetches secrets."""

    def test_env_references_resolve(self) -> None:
        provider = EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET})
        assert provider.resolve("env:CCTV_PASSWORD") == SECRET

    def test_a_missing_env_variable_is_reported_not_guessed(self) -> None:
        provider = EnvironmentSecretProvider({})
        with pytest.raises(MissingSecretError, match="CCTV_PASSWORD"):
            provider.resolve("env:CCTV_PASSWORD")

    def test_file_references_strip_a_trailing_newline(self, tmp_path) -> None:
        """`echo` adds one, and it is not part of the password."""
        path = tmp_path / "dvr"
        path.write_text(f"{SECRET}\n", encoding="utf-8")
        assert EnvironmentSecretProvider({}).resolve(f"file:{path}") == SECRET

    def test_an_empty_reference_is_refused(self) -> None:
        with pytest.raises(MissingSecretError):
            EnvironmentSecretProvider({}).resolve("")

    def test_an_unknown_scheme_is_refused(self) -> None:
        with pytest.raises(SecretResolutionError, match="scheme"):
            EnvironmentSecretProvider({}).resolve("vault://dvr")

    def test_an_error_never_quotes_a_literal_secret(self) -> None:
        """A `literal:` reference contains the secret; only the scheme survives."""
        provider = EnvironmentSecretProvider({})
        with pytest.raises(SecretResolutionError) as caught:
            provider.resolve(f"literal:")
        assert SECRET not in str(caught.value)

    def test_has_never_returns_the_value(self) -> None:
        provider = EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET})
        assert provider.has("env:CCTV_PASSWORD") is True
        assert provider.has("env:NOTHING") is False


class TestCredentialRedaction:
    """§7 — the password appears in the dial URL and nowhere else."""

    def config(self, **overrides) -> RtspCameraConfig:
        payload = {
            "camera_id": "cam-01",
            "host": "gayatri.freemyip.com",
            "username": "admin",
            "credential_ref": "env:CCTV_PASSWORD",
        }
        payload.update(overrides)
        return RtspCameraConfig(**payload)

    def test_the_dial_url_carries_the_password(self) -> None:
        assert SECRET in self.config().dial_uri(SECRET)

    def test_the_redacted_url_does_not(self) -> None:
        redacted = self.config().redacted_uri()
        assert SECRET not in redacted
        assert REDACTED in redacted
        assert "gayatri.freemyip.com" in redacted, "still diagnosable"
        assert "admin" not in redacted

    def test_source_status_exposes_only_the_redacted_url(self) -> None:
        source = LiveRtspSource(
            self.config(), secrets=EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET})
        )
        assert SECRET not in str(source.status.to_wire())

    async def test_a_decoder_error_quoting_the_url_is_scrubbed(self) -> None:
        """Decoder libraries habitually quote the URL they failed to open."""

        def exploding(uri: str):
            raise OSError(f"failed to open {uri}")

        source = LiveRtspSource(
            self.config(),
            secrets=EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET}),
            reconnect=ReconnectPolicy(initial_ms=1, max_attempts=1),
            opener=exploding,
        )
        [f async for f in source.frames()]

        assert SECRET not in source.status.last_error
        assert REDACTED in source.status.last_error
        assert SECRET not in str(source.status.to_wire())

    async def test_a_missing_credential_is_not_dialled(self) -> None:
        """Retrying a blank password counts toward DVR account lockout."""
        opened: list[str] = []
        source = LiveRtspSource(
            self.config(credential_ref="env:ABSENT"),
            secrets=EnvironmentSecretProvider({}),
            opener=lambda uri: opened.append(uri),
        )
        [f async for f in source.frames()]

        assert opened == []
        assert source.status.state is SourceState.ERROR

    def test_configuration_holds_a_reference_never_a_password(self) -> None:
        config = self.config()
        assert config.credential_ref.startswith("env:")
        assert SECRET not in str(config)


class TestReconnect:
    """§23 — bounded, observable, and it does not retry the unretryable."""

    def config(self) -> RtspCameraConfig:
        return RtspCameraConfig(
            camera_id="cam-01",
            host="dvr.invalid",
            username="admin",
            credential_ref="env:CCTV_PASSWORD",
        )

    def test_backoff_grows_and_is_capped(self) -> None:
        policy = ReconnectPolicy(initial_ms=1_000, multiplier=2.0, max_ms=60_000)
        assert policy.delay_for(1) == 1_000
        assert policy.delay_for(2) == 2_000
        assert policy.delay_for(3) == 4_000
        assert policy.delay_for(50) == 60_000, "capped"

    def test_an_enormous_attempt_count_does_not_overflow(self) -> None:
        """`2.0 ** 9998` raised OverflowError in Phase 7A."""
        policy = ReconnectPolicy(max_attempts=0)
        assert policy.delay_for(9_999) == policy.max_ms
        assert policy.should_retry(9_999)

    async def test_a_transient_failure_reconnects_and_increments_the_epoch(self) -> None:
        attempts = {"n": 0}

        def flaky(uri: str):
            attempts["n"] += 1
            raise OSError("connection reset")

        source = LiveRtspSource(
            self.config(),
            secrets=EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET}),
            reconnect=ReconnectPolicy(initial_ms=1, max_ms=1, max_attempts=3),
            opener=flaky,
        )
        [f async for f in source.frames()]

        assert attempts["n"] >= 2, "it retried"
        assert source.status.reconnects >= 1
        assert source.status.epoch >= 1, "a new epoch, so tracking cannot associate across the gap"
        assert source.status.state is SourceState.ERROR, "and eventually gave up"

    async def test_reconnecting_is_recorded_as_its_own_state(self) -> None:
        def flaky(uri: str):
            raise OSError("connection reset")

        source = LiveRtspSource(
            self.config(),
            secrets=EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET}),
            reconnect=ReconnectPolicy(initial_ms=1, max_ms=1, max_attempts=2),
            opener=flaky,
        )
        [f async for f in source.frames()]

        states = [t.to_state for t in source.status.transitions]
        assert SourceState.RECONNECTING in states

    async def test_bad_credentials_are_not_retried(self) -> None:
        """Retrying a rejected password counts toward DVR account lockout."""
        attempts = {"n": 0}

        def rejecting(uri: str):
            attempts["n"] += 1
            raise RtspAuthenticationError("the DVR rejected these credentials")

        source = LiveRtspSource(
            self.config(),
            secrets=EnvironmentSecretProvider({"CCTV_PASSWORD": SECRET}),
            reconnect=ReconnectPolicy(initial_ms=1, max_attempts=10),
            opener=rejecting,
        )
        [f async for f in source.frames()]

        assert attempts["n"] == 1, "one attempt, then it stopped"
        assert source.status.state is SourceState.ERROR


# ── sessions ─────────────────────────────────────────────────────────────────


class TestSession:
    """LIVE_SIMULATION — the continuous-source boundary."""

    async def test_a_session_processes_a_continuous_source(self) -> None:
        runtime = LiveRuntime(settings())
        seen: list[LiveFrame] = []

        async def handler(_spec, live_frame):
            seen.append(live_frame)

        runtime._on_frame = handler  # noqa: SLF001 - composition seam
        session = await runtime.start_synthetic(
            camera_id="cam-01",
            tenant_id="org-test",
            fps=25.0,
            count=40,
            analysis_fps=25.0,
            interval_override_s=0,
        )

        for _ in range(200):
            if session.stats.frames_processed >= 20:
                break
            await asyncio.sleep(0.01)

        await runtime.stop_all()
        assert session.stats.frames_processed > 0
        assert all(f.camera_id == "cam-01" for f in seen)

    async def test_a_session_and_its_source_agree_on_the_camera(self) -> None:
        from app.vision.session import ReplaySession, SessionSpec

        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        with pytest.raises(ValueError, match="does not match"):
            ReplaySession(SessionSpec(camera_id="cam-02", tenant_id="org"), source)

    async def test_a_live_session_declares_itself_unseekable_and_unbounded(self) -> None:
        from app.vision.session import LiveSession, SessionSpec

        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        session = LiveSession(SessionSpec(camera_id="cam-01", tenant_id="org"), source)
        assert session.seekable is False
        assert session.bounded is False

    async def test_shutdown_leaves_no_task_or_queued_frame(self) -> None:
        """§24 — no orphan task, queue or source."""
        runtime = LiveRuntime(settings())
        session = await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
        )
        await asyncio.sleep(0.05)
        await runtime.stop_all()

        assert session.state is SessionState.STOPPED
        assert session._producer is None  # noqa: SLF001
        assert session._consumer is None  # noqa: SLF001
        assert session._queue.depth == 0  # noqa: SLF001
        assert session.source.status.state.is_terminal

    async def test_stopping_twice_is_safe(self) -> None:
        runtime = LiveRuntime(settings())
        await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=5, interval_override_s=0
        )
        await runtime.stop_all()
        await runtime.stop_all()

    async def test_a_camera_cannot_have_two_active_sessions(self) -> None:
        from app.errors import ConfigurationInvalidError

        runtime = LiveRuntime(settings())
        await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
        )
        with pytest.raises(ConfigurationInvalidError, match="already has an active session"):
            await runtime.start_synthetic(
                camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
            )
        await runtime.stop_all()


class TestBackpressureContract:
    """§35 — a source faster than the processor must stay bounded."""

    async def test_memory_stays_bounded_and_drops_are_counted(self) -> None:
        runtime = LiveRuntime(settings())
        processed = {"n": 0}

        async def slow(_spec, _frame):
            processed["n"] += 1
            # Far slower than the producer. This is the whole scenario.
            await asyncio.sleep(0.02)

        runtime._on_frame = slow  # noqa: SLF001
        session = await runtime.start_synthetic(
            camera_id="cam-01",
            tenant_id="org-test",
            fps=1_000.0,
            count=None,
            analysis_fps=1_000.0,
            queue_capacity=4,
            interval_override_s=0,
        )

        await asyncio.sleep(0.4)
        depth_during = session._queue.depth  # noqa: SLF001
        dropped_during = session._queue.stats.dropped_queue_full  # noqa: SLF001
        await runtime.stop_all()

        # Bounded — not "eventually empties".
        assert depth_during <= 4, "the queue never exceeded its capacity"
        assert session._queue.stats.high_water <= 4  # noqa: SLF001
        assert dropped_during > 0, "the producer genuinely outran the consumer"
        assert processed["n"] > 0, "and the consumer still made progress"

    async def test_the_newest_frames_are_the_ones_kept(self) -> None:
        """Latest-frame policy, observed end to end."""
        runtime = LiveRuntime(settings())
        seen: list[int] = []

        async def slow(_spec, live_frame):
            seen.append(live_frame.sequence)
            await asyncio.sleep(0.02)

        runtime._on_frame = slow  # noqa: SLF001
        await runtime.start_synthetic(
            camera_id="cam-01",
            tenant_id="org-test",
            fps=1_000.0,
            count=None,
            analysis_fps=1_000.0,
            queue_capacity=2,
            interval_override_s=0,
        )
        await asyncio.sleep(0.3)
        await runtime.stop_all()

        assert len(seen) >= 3
        assert seen == sorted(seen), "sequence never went backwards"
        assert len(seen) == len(set(seen)), "no frame was processed twice"
        # Gaps prove old frames were dropped rather than queued.
        assert seen[-1] - seen[0] > len(seen)

    async def test_shutdown_under_load_is_still_clean(self) -> None:
        runtime = LiveRuntime(settings())

        async def slow(_spec, _frame):
            await asyncio.sleep(0.02)

        runtime._on_frame = slow  # noqa: SLF001
        session = await runtime.start_synthetic(
            camera_id="cam-01",
            tenant_id="org-test",
            fps=1_000.0,
            count=None,
            analysis_fps=1_000.0,
            queue_capacity=4,
            interval_override_s=0,
        )
        await asyncio.sleep(0.15)
        await runtime.stop_all()

        assert session.state is SessionState.STOPPED
        assert session._queue.depth == 0  # noqa: SLF001


class TestStreamingState:
    """§21 — `streaming` is derived, never asserted."""

    async def test_no_session_means_not_streaming(self) -> None:
        runtime = LiveRuntime(settings())
        assert runtime.summary().to_wire()["streaming"] is False

    async def test_a_source_with_no_frames_is_not_streaming(self) -> None:
        from app.vision.session import LiveSession, SessionSpec

        source = SyntheticFrameSource(camera_id="cam-01", count=1)
        session = LiveSession(SessionSpec(camera_id="cam-01", tenant_id="org"), source)
        # Connected, has produced nothing.
        source._transition(SourceState.RUNNING, "test")  # noqa: SLF001
        session._state = SessionState.RUNNING  # noqa: SLF001

        assert session.streaming is False, "an open connection is not a stream"

    async def test_a_genuine_frame_makes_it_stream(self) -> None:
        runtime = LiveRuntime(settings())
        session = await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
        )

        for _ in range(200):
            if session.streaming:
                break
            await asyncio.sleep(0.01)

        streaming = session.streaming
        assert runtime.summary().streaming_sessions == 1
        await runtime.stop_all()

        assert streaming is True

    async def test_a_stopped_session_stops_streaming(self) -> None:
        runtime = LiveRuntime(settings())
        session = await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
        )
        for _ in range(200):
            if session.streaming:
                break
            await asyncio.sleep(0.01)

        await runtime.stop_all()
        assert session.streaming is False


class TestMultipleSessions:
    """§36 — structural isolation, not a scale benchmark."""

    async def test_three_sessions_run_without_crossover(self) -> None:
        runtime = LiveRuntime(settings())
        seen: dict[str, set[str]] = {}

        async def handler(spec, live_frame):
            seen.setdefault(spec.camera_id, set()).add(live_frame.camera_id)

        runtime._on_frame = handler  # noqa: SLF001
        for index in (1, 2, 3):
            await runtime.start_synthetic(
                camera_id=f"cam-{index:02d}",
                tenant_id="org-test",
                count=None,
                analysis_fps=100.0,
                interval_override_s=0,
            )

        await asyncio.sleep(0.2)
        sessions = runtime.sessions
        await runtime.stop_all()

        assert len(sessions) == 3
        # Every frame reached the handler under its own camera, and only its own.
        for camera, observed in seen.items():
            assert observed == {camera}, f"{camera} saw frames from {observed}"

    async def test_each_session_has_its_own_queue(self) -> None:
        runtime = LiveRuntime(settings())
        for index in (1, 2):
            await runtime.start_synthetic(
                camera_id=f"cam-{index:02d}",
                tenant_id="org-test",
                count=None,
                interval_override_s=0,
            )
        queues = {id(s._queue) for s in runtime.sessions}  # noqa: SLF001
        await runtime.stop_all()
        assert len(queues) == 2

    async def test_one_session_stopping_leaves_the_others_running(self) -> None:
        runtime = LiveRuntime(settings())
        first = await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-test", count=None, interval_override_s=0
        )
        second = await runtime.start_synthetic(
            camera_id="cam-02", tenant_id="org-test", count=None, interval_override_s=0
        )
        await asyncio.sleep(0.05)

        await runtime.stop(first.session_id, tenant_id="org-test")
        assert second.state.is_active
        await runtime.stop_all()

    async def test_sessions_are_scoped_by_tenant(self) -> None:
        runtime = LiveRuntime(settings())
        await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-a", count=None, interval_override_s=0
        )
        await runtime.start_synthetic(
            camera_id="cam-02", tenant_id="org-b", count=None, interval_override_s=0
        )

        visible = runtime.visible(tenant_id="org-a", camera_ids=None)
        await runtime.stop_all()

        assert [s.camera_id for s in visible] == ["cam-01"]

    async def test_an_empty_camera_scope_grants_nothing(self) -> None:
        """The empty-tuple hazard, again. `()` is none, never all."""
        runtime = LiveRuntime(settings())
        await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-a", count=None, interval_override_s=0
        )
        visible = runtime.visible(tenant_id="org-a", camera_ids=())
        await runtime.stop_all()
        assert visible == []

    async def test_another_tenants_session_reports_not_found(self) -> None:
        from app.errors import NotFoundError

        runtime = LiveRuntime(settings())
        session = await runtime.start_synthetic(
            camera_id="cam-01", tenant_id="org-a", count=None, interval_override_s=0
        )
        # Not "forbidden": telling a caller a session exists elsewhere is itself
        # a disclosure.
        with pytest.raises(NotFoundError):
            runtime.get(session.session_id, tenant_id="org-b")
        await runtime.stop_all()


# ── configuration and autostart ──────────────────────────────────────────────


class TestNoAutostart:
    """§37, §38 — nothing dials a camera by accident."""

    async def test_importing_the_module_starts_nothing(self) -> None:
        runtime = LiveRuntime(settings())
        assert runtime.sessions == ()

    async def test_the_feature_flag_is_off_by_default(self) -> None:
        assert settings().feature_live_cctv is False

    async def test_start_configured_does_nothing_when_the_flag_is_off(self) -> None:
        runtime = LiveRuntime(
            settings(cctv_host="dvr.invalid", cctv_channels="1", feature_live_cctv=False)
        )
        assert await runtime.start_configured() == 0
        assert runtime.sessions == ()

    async def test_an_empty_channel_list_selects_nothing(self) -> None:
        """A 16-channel DVR must not become 16 pipelines by default."""
        runtime = LiveRuntime(
            settings(cctv_host="dvr.invalid", cctv_channels="", feature_live_cctv=True)
        )
        assert await runtime.start_configured() == 0

    def test_only_named_channels_become_cameras(self) -> None:
        runtime = LiveRuntime(
            settings(cctv_host="dvr.invalid", cctv_channels="1,5,7", feature_live_cctv=True)
        )
        cameras = runtime.describe_cameras()
        assert [c["camera_id"] for c in cameras] == ["cam-01", "cam-05", "cam-07"]
        assert len(cameras) == 3, "a 16-channel DVR did not yield 16 cameras"

    def test_a_typo_raises_rather_than_dropping_a_camera(self) -> None:
        """A silently skipped channel is a kitchen nobody is watching."""
        from app.errors import ConfigurationInvalidError

        runtime = LiveRuntime(
            settings(cctv_host="dvr.invalid", cctv_channels="1,two,3", feature_live_cctv=True)
        )
        with pytest.raises(ConfigurationInvalidError, match="not a number"):
            runtime.describe_cameras()

    def test_described_cameras_carry_no_credential(self) -> None:
        runtime = LiveRuntime(
            settings(
                cctv_host="gayatri.freemyip.com",
                cctv_channels="1",
                cctv_username="admin",
                cctv_credential_ref="env:CCTV_PASSWORD",
                feature_live_cctv=True,
            )
        )
        rendered = str(runtime.describe_cameras())
        assert REDACTED in rendered
        assert "admin" not in rendered
        assert "CCTV_PASSWORD" not in rendered
