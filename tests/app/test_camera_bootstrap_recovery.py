"""Camera binding must survive a database that is not up yet.

### The failure this pins

On 2026-08-31 a backend started roughly ten minutes before its PostgreSQL
container. The roster read failed, the handler logged an error, and **nothing
ever tried again**. Everything else recovered on its own — SQLAlchemy reconnects
lazily, so the API began serving the moment the database appeared and
`/health/ready` went green — leaving the cameras dark behind a dashboard that
looked healthy. Their counters were the tell: `frames_decoded=0`,
`reconnects=0`, `last_error=""`, which is the signature of something that never
started rather than something that tried and failed.

The remedy at the time was a manual restart nothing in the system asked for.
These tests exist so that remedy is never needed again.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app import main as app_main


class _Recorder:
    """Stands in for the database roster read.

    `failures` responses raise, then every later call succeeds — the shape of a
    container that is still starting.
    """

    def __init__(self, failures: int, cameras: int = 4) -> None:
        self.failures = failures
        self.cameras = cameras
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionRefusedError("the database is not up yet")
        return self.cameras


@pytest.fixture
def fast_backoff(monkeypatch):
    """Collapse the backoff so a retry test runs in milliseconds."""
    monkeypatch.setattr(app_main, "_BOOTSTRAP_BACKOFF", (0.01, 0.01, 0.01))


def _app_with(wall, live) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(wall=wall, live=live))


@pytest.fixture
def patched(monkeypatch):
    """Replace both roster reads with recorders and report what was called."""

    def install(wall_recorder, live_recorder):
        async def wall(app):
            try:
                return wall_recorder()
            except Exception:
                return None

        async def live(app):
            try:
                return live_recorder()
            except Exception:
                return None

        monkeypatch.setattr(app_main, "_start_camera_wall", wall)
        monkeypatch.setattr(app_main, "_start_cameras_from_database", live)

    return install


class TestTheReadIsDistinguishedFromTheAnswer:
    """`0 cameras` and `nobody answered` are different facts."""

    async def test_an_unreadable_roster_reports_neither_half_read(self, patched):
        patched(_Recorder(failures=99), _Recorder(failures=99))
        assert await app_main._bootstrap_cameras_once(_app_with(None, None)) == (
            False,
            False,
        )

    async def test_zero_enabled_cameras_is_a_successful_read(self, patched):
        """A deployment with no enabled camera is valid, and must not be retried
        forever as though the database were down."""
        patched(_Recorder(failures=0, cameras=0), _Recorder(failures=0, cameras=0))
        assert await app_main._bootstrap_cameras_once(_app_with(None, None)) == (
            True,
            True,
        )

    async def test_a_half_that_fails_is_reported_separately(self, patched):
        """**The bug the live acceptance test caught.**

        The two reads happen seconds apart. On 2026-08-31 the database finished
        starting between them: the wall read failed at 08:21:54, the live read
        succeeded at 08:21:57, and an earlier version of this bootstrap reported
        overall success -- stranding the wall, which is exactly the half
        `/api/v1/wall/cameras` reports and the UI renders.
        """
        patched(_Recorder(failures=99), _Recorder(failures=0, cameras=4))
        assert await app_main._bootstrap_cameras_once(_app_with(None, None)) == (
            False,
            True,
        )

    async def test_both_halves_are_attempted_even_when_the_first_fails(self, patched):
        wall, live = _Recorder(failures=99), _Recorder(failures=0)
        patched(wall, live)
        await app_main._bootstrap_cameras_once(_app_with(None, None))
        assert wall.calls == 1 and live.calls == 1

    async def test_a_half_already_bound_is_not_asked_again(self, patched):
        """The analysis runtime refuses a duplicate camera outright, so retrying
        a half that already bound logs a real error for a non-problem."""
        wall, live = _Recorder(failures=0), _Recorder(failures=0)
        patched(wall, live)
        await app_main._bootstrap_cameras_once(
            _app_with(None, None), need_wall=False, need_live=True
        )
        assert wall.calls == 0, "the wall had already bound"
        assert live.calls == 1


class TestTheSupervisorRecovers:
    """The acceptance criterion: recovery without a restart."""

    async def test_it_retries_until_the_database_answers(self, patched, fast_backoff):
        wall, live = _Recorder(failures=2), _Recorder(failures=2)
        patched(wall, live)
        await asyncio.wait_for(
            app_main._camera_bootstrap_supervisor(_app_with(None, None)), timeout=5
        )
        assert wall.calls == 3, "two refusals then a success"
        assert live.calls == 3

    async def test_it_keeps_going_until_both_halves_are_bound(
        self, patched, fast_backoff
    ):
        """The regression the live acceptance test exposed: it must not stop
        when only one half has bound."""
        wall, live = _Recorder(failures=4), _Recorder(failures=0)
        patched(wall, live)
        await asyncio.wait_for(
            app_main._camera_bootstrap_supervisor(_app_with(None, None)), timeout=10
        )
        assert wall.calls == 5, "retried until the wall itself bound"
        assert live.calls == 1, "and stopped asking the half that already had"

    async def test_it_stops_once_the_roster_is_read(self, patched, fast_backoff):
        """It supervises the *bootstrap*, not the cameras.

        RTSP reconnection belongs to the session's own ReconnectPolicy, and
        enabling a camera later goes through the camera API. A supervisor that
        kept polling would be a second, competing source of truth for which
        cameras should be running.
        """
        wall, live = _Recorder(failures=1), _Recorder(failures=1)
        patched(wall, live)
        await asyncio.wait_for(
            app_main._camera_bootstrap_supervisor(_app_with(None, None)), timeout=5
        )
        before = wall.calls
        await asyncio.sleep(0.05)
        assert wall.calls == before, "it must not keep polling after success"

    async def test_it_survives_an_unexpected_error_and_keeps_trying(
        self, monkeypatch, fast_backoff
    ):
        """A retry that raises must not kill the task and strand the cameras."""
        calls = {"n": 0}

        async def explode(app, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("something unexpected")
            return True, True

        monkeypatch.setattr(app_main, "_bootstrap_cameras_once", explode)
        await asyncio.wait_for(
            app_main._camera_bootstrap_supervisor(_app_with(None, None)), timeout=5
        )
        assert calls["n"] == 3

    async def test_it_keeps_trying_after_the_backoff_is_exhausted(
        self, patched, fast_backoff
    ):
        """A database that returns after ten minutes still gets its cameras."""
        wall = _Recorder(failures=len(app_main._BOOTSTRAP_BACKOFF) + 2)
        live = _Recorder(failures=len(app_main._BOOTSTRAP_BACKOFF) + 2)
        patched(wall, live)
        await asyncio.wait_for(
            app_main._camera_bootstrap_supervisor(_app_with(None, None)), timeout=5
        )
        assert wall.calls > len(app_main._BOOTSTRAP_BACKOFF)

    async def test_it_is_cancellable(self, patched, monkeypatch):
        """Shutdown must be able to stop it before it starts a camera session."""
        monkeypatch.setattr(app_main, "_BOOTSTRAP_BACKOFF", (5.0,))
        patched(_Recorder(failures=99), _Recorder(failures=99))
        task = asyncio.create_task(
            app_main._camera_bootstrap_supervisor(_app_with(None, None))
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestTheBackoffIsSane:
    def test_it_starts_fast_and_settles(self):
        """Fast first, because the common case is a stack coming up together and
        the database being seconds behind. Settled after, because a database
        down for a minute is an outage and polling harder helps nobody."""
        backoff = app_main._BOOTSTRAP_BACKOFF
        assert backoff[0] <= 2.0
        assert backoff[-1] >= 15.0
        assert list(backoff) == sorted(backoff), "must be non-decreasing"


class TestNormalStartupIsUnaffected:
    async def test_a_healthy_database_needs_no_supervisor(self, patched):
        """The ordinary path must not spawn a background task."""
        patched(_Recorder(failures=0), _Recorder(failures=0))
        wall_read, live_read = await app_main._bootstrap_cameras_once(
            _app_with(None, None)
        )
        assert wall_read and live_read
