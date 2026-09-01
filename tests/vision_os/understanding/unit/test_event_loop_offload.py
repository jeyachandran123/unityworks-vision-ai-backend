"""The model call must not run on the loop that consumes camera frames.

### The failure this pins

Measured 2026-09-01 on four live kitchen cameras, `remote_concurrency=2`:

| | |
|---|---|
| analysed frames | **5.7 /min/camera** against a requested 60 |
| queue-full drops | **201-362 per camera per 300 s** |
| VLM latency | p50 2.07 s · p95 14.3 s · **max 40.2 s** |
| CPU | **21% of capacity** — nothing was saturated |
| detection inference | 0 ms · registry apply 0 ms · crop extract 31 ms |

Every stage except the model was at zero. `UnderstandingRuntime._run_ready` called
`engine.understand_batch()` — synchronous — directly on the single ANALYSIS event loop,
which `VisionSession._consume` also uses for **every camera**. While a batch ran, no frame
from any camera could be consumed; session queues (capacity 8) filled and dropped.

The same shape had already been fixed once, one layer up: the ANALYSIS thread exists
because perception ran on the *API* loop and `/health` took 31.5 s. This is that bug on the
next loop down.

### Two things are being defended, and the second is easy to lose

1. The model call runs **off** the loop.
2. Moving it off must not create a **queue of waiters**. While the call blocked the loop,
   no second `on_crops` task could be created — the crop sink runs on that same loop — so
   nothing accumulated. Free the loop and every crop batch becomes a coroutine waiting on
   `_lock`, each holding its crop **pixels**. The bounded `_queue` does not bound those.

`TestNoUnboundedBacklog` is that second property. A one-line `to_thread` inside the existing
lock passes every test in `TestTheModelCallIsOffTheLoop` and fails those.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from vision_os.perception.understanding import UnderstandingRuntime

from ..conftest import CAMERA, POSTURE, make_crop
from .test_runtime import crop_request, evaluation


def build(engine, clock, metrics, health, config, **kw) -> UnderstandingRuntime:
    return UnderstandingRuntime(
        clock=clock, metrics=metrics, health=health, engine=engine, config=config, **kw
    )


class _RecordingEngine:
    """Stands in for the real engine, recording which thread ran the batch."""

    def __init__(self, *, delay: float = 0.0, raises: BaseException | None = None) -> None:
        self.delay = delay
        self.raises = raises
        self.calls = 0
        self.threads: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._guard = threading.Lock()

    def plan_batches(self, requests):
        return ()

    def understand_batch(self, requests, *, crops=None):
        with self._guard:
            self.calls += 1
            self.threads.append(threading.get_ident())
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)          # blocking, exactly like a model call
            if self.raises is not None:
                raise self.raises
            return {}
        finally:
            with self._guard:
                self.concurrent -= 1


async def feed(runtime, n: int = 1, *, object_id: str = "obj-1") -> None:
    """Deliver `n` crop batches.

    The object id must match on **both** sides: `_enqueue` pairs each crop to a
    request by `crop.object_id`, and an unmatched crop is silently skipped — so
    varying only the request's id queues nothing at all.
    """
    for _ in range(n):
        await runtime.on_crops(
            evaluation(crop_request(object_id=object_id)),
            [make_crop(object_id=object_id)],
        )


class TestTheModelCallIsOffTheLoop:
    async def test_the_batch_runs_on_another_thread(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine()
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()
        await feed(runtime)

        assert engine.calls == 1
        assert engine.threads[0] != threading.get_ident(), (
            "understand_batch ran on the event-loop thread"
        )

    async def test_the_loop_keeps_running_while_the_model_blocks(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """**The acceptance criterion.** A slow model must not freeze frame
        consumption — which lives on this same loop."""
        engine = _RecordingEngine(delay=0.40)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await feed(runtime)
        beat.cancel()

        assert engine.calls == 1
        # A blocked loop would have produced 0-1 ticks across a 0.4 s call.
        assert ticks >= 10, f"the loop only got {ticks} turns during a 0.4 s model call"

    async def test_publish_stays_on_the_loop(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """Registry write-back is deliberately NOT offloaded: keeping it on one
        thread preserves write ordering, and it costs 0 ms."""
        seen: list[int] = []
        engine = _RecordingEngine()
        runtime = build(
            engine, clock, metrics, health, understanding_config,
            sink=lambda results: seen.append(threading.get_ident()),
        )
        await runtime.start()
        await feed(runtime)

        assert seen == [threading.get_ident()]


class TestNoUnboundedBacklog:
    """Freeing the loop must not turn crop batches into unbounded lock waiters."""

    async def test_a_second_arrival_sheds_instead_of_waiting(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine(delay=0.30)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        first = asyncio.create_task(feed(runtime))
        await asyncio.sleep(0.05)               # let the batch start
        # 20 more arrivals while the model is busy. None may block.
        started = time.perf_counter()
        await feed(runtime, 20, object_id="obj-2")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.15, (
            f"on_crops waited {elapsed:.2f}s behind a running batch — waiters are "
            f"accumulating, each holding crop pixels"
        )
        await first

    async def test_only_one_batch_is_ever_in_flight(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine(delay=0.15)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        await asyncio.gather(*(feed(runtime, object_id=f"obj-{i}") for i in range(6)))

        assert engine.max_concurrent == 1, (
            f"{engine.max_concurrent} batches ran at once; the runner must be single"
        )

    async def test_the_queue_stays_bounded(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """The deque is the buffer, and it already drops oldest with a counter."""
        engine = _RecordingEngine(delay=0.10)
        runtime = build(
            engine, clock, metrics, health, understanding_config, queue_capacity=4
        )
        await runtime.start()

        await asyncio.gather(*(feed(runtime, object_id=f"obj-{i}") for i in range(30)))

        assert runtime.queue_depth <= 4

    async def test_work_queued_during_a_batch_is_not_stranded(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """A shed arrival is still enqueued, so the running batch must drain it
        rather than leave it until the next camera frame."""
        engine = _RecordingEngine(delay=0.20)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        first = asyncio.create_task(feed(runtime))
        await asyncio.sleep(0.05)
        await feed(runtime, 3, object_id="obj-9")
        await first
        await asyncio.sleep(0.6)

        assert engine.calls >= 2, "items enqueued during a batch were never run"
        assert runtime.queue_depth == 0


class TestFailuresPropagateThroughExistingSemantics:
    async def test_an_exception_in_the_thread_does_not_raise_at_the_seam(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """V9 — the seam is a firewall. An understanding failure may not stop the
        Crop Manager, which may not stop tracking, which may not stop capture."""
        engine = _RecordingEngine(raises=RuntimeError("model exploded"))
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        await feed(runtime)                      # must not raise

        assert runtime.stats.frames_failed == 1

    async def test_the_runner_is_released_after_a_failure(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """A crash must not leave `_running` set and wedge every later batch."""
        engine = _RecordingEngine(raises=RuntimeError("boom"))
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()
        await feed(runtime)

        engine.raises = None
        await feed(runtime, object_id="obj-2")

        assert engine.calls == 2, "the runner never recovered after an exception"

    async def test_cancellation_releases_the_runner(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine(delay=1.0)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()

        task = asyncio.create_task(feed(runtime))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        engine.delay = 0.0
        await feed(runtime, object_id="obj-3")
        assert engine.calls >= 2, "a cancelled batch left the runner stuck"

    async def test_shutdown_during_a_call_still_drains(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine(delay=0.10)
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()
        await feed(runtime, 2)
        await runtime.drain()

        assert runtime.queue_depth == 0


class TestSemanticsAreUnchanged:
    async def test_nothing_is_consumed_before_start(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine()
        runtime = build(engine, clock, metrics, health, understanding_config)
        await feed(runtime)

        assert engine.calls == 0
        assert runtime.stats.frames_consumed == 0

    async def test_empty_crops_are_not_an_error(
        self, clock, metrics, health, understanding_config
    ) -> None:
        engine = _RecordingEngine()
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()
        await runtime.on_crops(evaluation(crop_request()), [])

        assert engine.calls == 0
        assert runtime.stats.frames_failed == 0

    async def test_the_engine_receives_the_same_requests_and_crops(
        self, clock, metrics, health, understanding_config
    ) -> None:
        """The offload moves *where* the call runs, never *what* is asked."""
        captured: dict = {}

        class _Capturing(_RecordingEngine):
            def understand_batch(self, requests, *, crops=None):
                captured["requests"] = list(requests)
                captured["crops"] = dict(crops or {})
                return super().understand_batch(requests, crops=crops)

        engine = _Capturing()
        runtime = build(engine, clock, metrics, health, understanding_config)
        await runtime.start()
        await feed(runtime)

        assert len(captured["requests"]) == 1
        assert len(captured["crops"]) == 1
        request = captured["requests"][0]
        assert request.camera_id == CAMERA
        assert POSTURE in request.requested_attributes
