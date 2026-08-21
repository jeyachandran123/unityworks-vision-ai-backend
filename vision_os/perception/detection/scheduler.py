"""The detection batch scheduler (08_RUNTIME_AND_THREADING section 4).

Single responsibility: *gather work from many cameras into device-efficient
batches. Execute nothing.*

**The critical inversion.** A naive design gives each camera its own model
instance; that fails at ten cameras, because GPU memory goes to duplicate weights
and utilization collapses at batch size 1. Here camera pipelines are logical
flows and the model is a shared service: frames from many pipelines are gathered
into one batch, executed once, and scattered back.

Batching is a **platform** concern rather than an adapter one because the
decision of *what to batch together* spans cameras, and no per-camera component
can make it.

The **dual trigger** is essential. Batch-full alone starves a three-camera
deployment that will never fill a batch of sixteen; timeout alone wastes
throughput at a hundred cameras. Both together mean the same configuration
behaves correctly at either scale — which is precisely what "1, 10 or 100 cameras
without redesign" requires.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ...core.errors import (
    DetectionFailedError,
    DetectionQueueFullError,
    DetectionTimeoutError,
    DetectorContractError,
)
from ...core.model.ids import CameraId, FrameRef
from ...core.model.timebase import Duration
from ...core.ports.clock import Clock
from ...core.ports.detection import DetectionRequest, DetectionResult, FrameView


@dataclass(frozen=True, slots=True)
class BatchKey:
    """What may share a batch.

    Frames only batch together when the model, precision and input geometry
    agree — mixing them would either fail in the framework or silently produce
    results at the wrong scale.
    """

    model_id: str
    model_version: str
    precision: str
    inference_width: int
    inference_height: int
    tier: str = "primary"


@dataclass(slots=True)
class BatchItem:
    """One frame awaiting detection, with the future that will carry its result."""

    frame_ref: FrameRef
    camera_id: CameraId
    view: FrameView
    request: DetectionRequest
    future: asyncio.Future[DetectionResult]
    enqueued_ns: int

    @property
    def cancelled(self) -> bool:
        return self.future.cancelled() or self.future.done()


@dataclass(frozen=True, slots=True)
class BatchStats:
    batches_executed: int = 0
    frames_batched: int = 0
    max_batch_seen: int = 0
    timeouts: int = 0
    rejected_full: int = 0

    @property
    def mean_batch_size(self) -> float:
        return self.frames_batched / self.batches_executed if self.batches_executed else 0.0


#: Dual-trigger defaults. Zero wait is what deterministic mode requires.
_DEFAULT_MAX_WAIT = Duration.from_millis(5)
_DEFAULT_INFERENCE_TIMEOUT = Duration.from_millis(2_000)

BatchExecutor = Callable[
    [BatchKey, Sequence[BatchItem]], Awaitable[Sequence[DetectionResult]]
]


class DetectionScheduler:
    """Accumulates submissions into batches and hands them to an executor.

    Deterministic mode requires ``max_wait`` of zero: batch composition must not
    depend on arrival timing, or a replay produces a different split and V13 is
    lost (08_RUNTIME section 4.3).
    """

    def __init__(
        self,
        *,
        clock: Clock,
        executor: BatchExecutor,
        max_batch_size: int = 8,
        max_wait: Duration = _DEFAULT_MAX_WAIT,
        queue_capacity: int = 64,
        inference_timeout: Duration = _DEFAULT_INFERENCE_TIMEOUT,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if queue_capacity < 1:
            raise ValueError(f"queue_capacity must be >= 1, got {queue_capacity}")
        self._clock = clock
        self._executor = executor
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait
        self._queue_capacity = queue_capacity
        self._inference_timeout = inference_timeout

        self._pending: dict[BatchKey, list[BatchItem]] = {}
        self._lock = asyncio.Lock()
        self._flushing: set[BatchKey] = set()
        self._stats = BatchStats()
        # Strong references to in-flight flush tasks. Without them the event loop
        # may garbage-collect a running task mid-batch.
        self._tasks: set[asyncio.Task[None]] = set()

    def _spawn(self, coro) -> None:
        """Schedule a flush, tolerating a loop that is already shutting down.

        During drain the loop can close between a batch completing and its
        follow-up being scheduled. Closing the coroutine explicitly keeps that
        from surfacing as a spurious "never awaited" warning on every shutdown.
        """
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        """Cancel in-flight flushes and fail any waiter, explicitly.

        A submitter left awaiting a future that will never resolve is the one
        shutdown outcome worse than an error.
        """
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()
        async with self._lock:
            pending = [item for queue in self._pending.values() for item in queue]
            self._pending.clear()
        for item in pending:
            if not item.future.done():
                item.future.set_exception(
                    DetectionFailedError(
                        "detection scheduler closed before this frame was processed",
                        frame_ref=str(item.frame_ref),
                    )
                )

    # --- submission ------------------------------------------------------------ #

    async def submit(
        self,
        *,
        key: BatchKey,
        frame_ref: FrameRef,
        camera_id: CameraId,
        view: FrameView,
        request: DetectionRequest,
    ) -> DetectionResult:
        """Enqueue a frame and await its result.

        Raises:
            DetectionTimeoutError: the batch did not complete within budget.
            DetectionQueueFullError: bounded by ``queue_capacity``. An unbounded
                inference queue is a memory leak with a delayed fuse.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DetectionResult] = loop.create_future()
        item = BatchItem(
            frame_ref=frame_ref,
            camera_id=camera_id,
            view=view,
            request=request,
            future=future,
            enqueued_ns=self._clock.monotonic().ns,
        )

        async with self._lock:
            queue = self._pending.setdefault(key, [])
            if self._depth() >= self._queue_capacity:
                self._stats = _bump(self._stats, rejected_full=1)
                raise DetectionQueueFullError(
                    f"detection queue is full ({self._depth()}/{self._queue_capacity}); "
                    f"shedding rather than growing without bound",
                    depth=self._depth(),
                )
            queue.append(item)
            should_flush = len(queue) >= self._max_batch_size

        # Always flush on a task, never inline. Awaiting the flush here would put
        # inference *before* the timeout guard below, so a hung device would block
        # the caller forever on exactly the path a full batch takes — the busy
        # path, where a stall hurts most.
        self._spawn(self._flush(key) if should_flush else self._flush_after_wait(key))

        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._inference_timeout.seconds
            )
        except TimeoutError as exc:
            self._stats = _bump(self._stats, timeouts=1)
            raise DetectionTimeoutError(
                f"detection for {frame_ref} exceeded "
                f"{self._inference_timeout.millis:.0f}ms",
                frame_ref=str(frame_ref),
            ) from exc

    async def _flush_after_wait(self, key: BatchKey) -> None:
        """The timeout half of the dual trigger.

        With ``max_wait`` of zero this yields once and flushes, which is both the
        deterministic-mode behaviour and the right behaviour for a lightly loaded
        deployment that would otherwise wait for a batch that never forms.
        """
        if self._max_wait.ns > 0:
            await self._clock.sleep(self._max_wait)
        else:
            await asyncio.sleep(0)
        await self._flush(key)

    async def flush_all(self) -> None:
        async with self._lock:
            keys = list(self._pending)
        for key in keys:
            await self._flush(key)

    # --- execution --------------------------------------------------------------- #

    async def _flush(self, key: BatchKey) -> None:
        async with self._lock:
            if key in self._flushing:
                return
            queue = self._pending.get(key)
            if not queue:
                return
            batch = queue[: self._max_batch_size]
            del queue[: len(batch)]
            if not queue:
                self._pending.pop(key, None)
            self._flushing.add(key)

        live = [item for item in batch if not item.cancelled]
        try:
            if live:
                await self._execute(key, live)
        finally:
            async with self._lock:
                self._flushing.discard(key)
            remaining = self._pending.get(key)
            if remaining:
                self._spawn(self._flush(key))

    async def _execute(self, key: BatchKey, batch: Sequence[BatchItem]) -> None:
        try:
            results = await self._executor(key, batch)
        except Exception as exc:  # noqa: BLE001 - a failed batch fails its items, not the platform
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        if len(results) != len(batch):
            error = DetectorContractError(
                f"executor returned {len(results)} results for a batch of "
                f"{len(batch)}; results must map 1:1 and in order (obligation D6)"
            )
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)
            return

        self._stats = _bump(
            self._stats,
            batches_executed=1,
            frames_batched=len(batch),
            max_batch_seen=len(batch),
        )
        for item, result in zip(batch, results, strict=True):
            if not item.future.done():
                item.future.set_result(result)

    # --- introspection ------------------------------------------------------------ #

    def _depth(self) -> int:
        return sum(len(queue) for queue in self._pending.values())

    @property
    def depth(self) -> int:
        return self._depth()

    @property
    def stats(self) -> BatchStats:
        return self._stats

    def queue_wait_ms(self, item: BatchItem) -> float:
        return (self._clock.monotonic().ns - item.enqueued_ns) / 1_000_000


def _bump(
    stats: BatchStats,
    *,
    batches_executed: int = 0,
    frames_batched: int = 0,
    max_batch_seen: int = 0,
    timeouts: int = 0,
    rejected_full: int = 0,
) -> BatchStats:
    return BatchStats(
        batches_executed=stats.batches_executed + batches_executed,
        frames_batched=stats.frames_batched + frames_batched,
        max_batch_seen=max(stats.max_batch_seen, max_batch_seen),
        timeouts=stats.timeouts + timeouts,
        rejected_full=stats.rejected_full + rejected_full,
    )
