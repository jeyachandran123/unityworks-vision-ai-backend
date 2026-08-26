"""The thread that CPU-bound analysis runs on, so the API loop stays free.

### The measured problem

Frame ingest, YOLO inference and the rest of the perception path are
synchronous, CPU-bound Python. They ran inline on the server's asyncio loop —
`VisionSession._consume` awaited a handler that never yielded — so while a
frame was being analysed no request could be served. Measured, with four
cameras at 1 fps:

    analysis ON   /auth/login 58.5 s · /health 31.5 s · 0 rows 67 s
    analysis OFF  /auth/login  4.0 s · /health  0.6 s

Sixty-seven seconds to return zero rows is not query cost. It is the event
loop never getting a turn.

### Why one thread and not one per camera

Phase 6B.3 solved the same shape for the camera *wall* with a thread per
camera, and that was safe because each camera owned its decoder outright and
shared nothing. **The analysis path is the opposite.** A single registry, a
single tracker, one metrics engine and one `VisionStateManager` are shared by
every camera, and none of them documents itself as thread-safe. Giving each
camera its own analysis thread would put concurrent mutation through all four
at once — trading a latency bug for a correctness bug, which is the worse
trade in a safety product.

So analysis moves *off* the API loop without becoming concurrent with itself:
one worker thread, one private event loop, every camera's work serialised on
it exactly as it is serialised today. The shared platform objects keep seeing
one caller at a time, so **this change introduces no new thread-safety
requirement at all** — it only changes which thread that single caller is.

The cost is that analysis throughput stays what it is. That is the honest
trade: this fixes the API starvation, and it does not pretend to make
detection faster. Per-camera parallelism is available later to whoever audits
those four objects properly.

### Bounded by construction

One thread. One loop. No executor, no task fan-out, no queue of its own — the
session's existing bounded frame queue and drop-oldest policy still provide
the backpressure, unchanged. Nothing here can grow without bound.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from loguru import logger


class AnalysisLoop:
    """A single worker thread running its own asyncio loop.

    Work is submitted from the server's loop and awaited there, so callers
    keep ordinary `await` semantics and frame ordering is untouched: a session
    still finishes one frame before starting the next.
    """

    __slots__ = ("_loop", "_ready", "_thread")

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    @property
    def running(self) -> bool:
        return self._loop is not None and self._loop.is_running()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="vision-analysis", daemon=True
        )
        self._thread.start()
        # Wait for the loop to exist before anyone submits to it, so the first
        # frame after boot is not a race.
        self._ready.wait(timeout=10.0)
        logger.info("analysis worker started — CPU-bound perception is off the API loop")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def run(self, coro) -> Any:
        """Run a coroutine on the analysis thread; await it on the caller's.

        Falls back to running inline when the worker is not up. That keeps a
        misconfiguration slow rather than silently analysing nothing — the
        failure mode this project has hit repeatedly is a pipeline that looks
        healthy and observes nothing, and that must not be reintroduced here.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))

    def stop(self, timeout: float = 10.0) -> None:
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        self._ready.clear()
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if thread.is_alive():
            logger.error("analysis worker did not stop within {}s", timeout)
        else:
            logger.info("analysis worker stopped")


#: One per process. Sessions submit to it; nothing else touches it.
ANALYSIS = AnalysisLoop()

__all__ = ["ANALYSIS", "AnalysisLoop"]
