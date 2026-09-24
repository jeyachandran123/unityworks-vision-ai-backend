"""Deleting a camera must not stop the API answering while it happens.

`CameraService.retire` purges the camera's observation partition before it
deletes the row — the ordering is the safety property, and it stays. The purge
is `FileObservationLog.truncate`, which takes the log's lock, and that lock is
shared with the analysis pipeline appending observations and with readers that
re-scan whole partition files while holding it. Under load the purge waits.

It used to wait *on the API event loop*. Measured on the live deployment on
2026-09-24: a delete sat inside `truncate` for 10.9 s, and an earlier one for
over a minute, during which every other request — including ones touching
nothing but authentication — waited with it. The same shape of fault as the RTSP
producer that ran on the API loop, in a different place.

The purge now waits on a worker thread. The delete still takes as long as it
must; nothing else does.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest

from app.domain.cameras import CameraService

from .conftest import make_recorder

ORG = "org-test"


class _SlowLog:
    """An observation log whose purge holds its thread, as a contended lock does."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self.truncated_on: str | None = None

    def truncate(self, partition, before) -> int:  # noqa: ARG002 - the port's signature
        self.truncated_on = threading.current_thread().name
        time.sleep(self._hold_s)
        return 0


async def _worst_gap_while(awaitable) -> tuple[float, object]:
    """Run `awaitable` while a canary ticks on this loop; return the worst stall."""
    gaps: list[float] = []
    stop = asyncio.Event()

    async def canary() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticking = asyncio.create_task(canary())
    await asyncio.sleep(0.02)
    try:
        result = await awaitable
    finally:
        stop.set()
        ticking.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticking
    return max(gaps), result


@pytest.mark.asyncio
async def test_a_slow_purge_does_not_stall_the_api_event_loop(app) -> None:
    log = _SlowLog(hold_s=0.5)
    async with app.state.database.session_scope() as session:
        await session.merge(make_recorder(ORG))
        await session.flush()
        service = CameraService(session)
        await service.create(
            organization_id=ORG,
            zone_id="zone-01",
            camera_key="cam-77",
            name="Leaving",
            channel=7,
            recorder_id=f"rec-{ORG}",
        )
        await session.flush()

        worst, _ = await _worst_gap_while(
            service.retire(organization_id=ORG, camera_key="cam-77", observation_log=log)
        )

    assert log.truncated_on is not None, "the partition was never purged"
    assert log.truncated_on != threading.main_thread().name, "the purge ran on the API's own thread"
    # The purge holds its thread for 500ms. On the loop, that lands on this
    # canary in full; on a worker thread it cannot reach it.
    assert worst < 0.2, f"the API event loop stalled for {worst * 1000:.0f}ms during a delete"


@pytest.mark.asyncio
async def test_the_row_is_still_only_deleted_after_the_purge(app) -> None:
    """Moving the purge off the loop must not reorder it: if it raises, the
    camera stays, exactly as before."""

    class _BrokenLog:
        def truncate(self, partition, before) -> int:  # noqa: ARG002
            raise RuntimeError("disk full")

    async with app.state.database.session_scope() as session:
        await session.merge(make_recorder(ORG))
        await session.flush()
        service = CameraService(session)
        await service.create(
            organization_id=ORG,
            zone_id="zone-01",
            camera_key="cam-78",
            name="Staying",
            channel=8,
            recorder_id=f"rec-{ORG}",
        )
        await session.flush()

        with pytest.raises(RuntimeError, match="disk full"):
            await service.retire(
                organization_id=ORG, camera_key="cam-78", observation_log=_BrokenLog()
            )

        still_there = await service.get(organization_id=ORG, camera_key="cam-78")
        assert still_there.camera_key == "cam-78"
