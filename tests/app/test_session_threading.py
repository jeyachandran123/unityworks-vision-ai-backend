"""A camera that is being analysed must not slow the API down.

This is the property the whole product rests on once more than one camera is
running: the server answers requests while it watches. It has been broken
twice in the same way — a piece of the frame path left on the API event loop —
and each time the symptom was identical and mystifying from the outside, an
application that seemed to have become slow for no reason.

`CameraWall` runs each camera on a dedicated thread and says so. The analysed
session's *consumer* was moved off the loop after it "took `/health` to 31s and
`/auth/login` to 58s while four cameras ran". Its **producer** was not, so with
four analysed cameras every request took nine seconds while the loop decoded
video. This test states the property directly, so the next piece to be added
to the frame path cannot quietly land on the API loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from app.vision.session import SessionSpec, SessionState, VisionSession
from app.vision.sources.base import FrameSource, SourceKind


class BlockingSource(FrameSource):
    """A source that blocks the thread it runs on, the way a real one does.

    `LiveRtspSource` reads a socket and decodes with PyAV — both of which hold
    the thread rather than yielding to the loop. A synthetic source that
    `await`s politely would pass this test while the real one fails it, which
    is how the defect survived the existing suite.
    """

    def __init__(self, camera_id: str, *, block_s: float = 0.05, count: int = 40) -> None:
        super().__init__(camera_id=camera_id, kind=SourceKind.LIVE)
        self._block_s = block_s
        self._count = count

    async def _produce(self):  # type: ignore[override]
        from app.vision.frames import LiveFrame

        sequence = 0
        while not self._stopping and sequence < self._count:
            # Deliberately synchronous. This is the decoder.
            time.sleep(self._block_s)
            sequence += 1
            captured = time.time_ns()
            yield LiveFrame(
                camera_id=self.camera_id,
                sequence=sequence,
                epoch=0,
                captured_at_ns=captured,
                received_at_ns=captured,
                width=8,
                height=8,
                payload=bytes(192),
            )


async def _canary(gaps: list[float], stop: asyncio.Event) -> None:
    """Tick as fast as the loop allows, recording how long each tick waited.

    Measured *while* the session runs, not afterwards: a loop that was blocked
    for two seconds looks perfectly healthy the moment it is free again, so a
    measurement taken after the fact reports nothing.
    """
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.005)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


@pytest.mark.asyncio
async def test_a_running_session_does_not_stall_the_api_event_loop() -> None:
    """The regression, stated as the thing an operator would notice.

    A blocked API loop is not a slow query and does not look like one: every
    route slows down by the same amount at once, including the ones that touch
    no database.
    """
    session = VisionSession(
        SessionSpec(camera_id="cam-01", tenant_id="org-test", analysis_fps=4.0),
        BlockingSource("cam-01", block_s=0.05, count=20),
    )

    gaps: list[float] = []
    stop = asyncio.Event()
    canary = asyncio.create_task(_canary(gaps, stop))
    await asyncio.sleep(0.05)

    await session.start()
    assert session.state is SessionState.RUNNING
    try:
        await asyncio.sleep(1.0)
    finally:
        stop.set()
        canary.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await canary
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(session.stop(), 10)

    assert gaps, "the canary never ran at all"
    worst = max(gaps)
    # The source holds its thread for 50ms per frame, twenty times over. On the
    # API loop those land on this canary; on the session's own thread they
    # cannot reach it.
    assert worst < 0.1, (
        f"the API event loop stalled for {worst * 1000:.0f}ms while one camera "
        "was being analysed — the frame path is running on the API loop"
    )


@pytest.mark.asyncio
async def test_starting_a_session_schedules_nothing_on_the_caller_loop() -> None:
    """The mechanism behind the property above, pinned separately.

    Stated this way because the symptom is expensive to observe and easy to
    misread as load, while the cause is a one-line choice about where the
    frame path runs.
    """
    session = VisionSession(
        SessionSpec(camera_id="cam-02", tenant_id="org-test"),
        BlockingSource("cam-02", block_s=0.01),
    )

    before = {task.get_name() for task in asyncio.all_tasks()}
    await session.start()
    try:
        added = {task.get_name() for task in asyncio.all_tasks()} - before
        assert not any(
            name.startswith(("produce-", "consume-")) for name in added
        ), f"the frame path was scheduled on the caller's loop: {sorted(added)}"
    finally:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(session.stop(), 10)
