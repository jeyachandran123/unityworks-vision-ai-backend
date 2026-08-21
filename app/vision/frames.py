"""The live frame path: one frame type, one bounded queue, one drop policy.

### Why a bounded queue is not a detail

A CCTV deployment runs for months. Anything unbounded in this path is a memory
leak with a schedule attached: 456 frames of 720p BGR24 is ~1.2 GB, so a queue
that grows for an hour is a queue that ends the process.

`LiveFrameQueue` therefore has a fixed capacity and a **stated** overflow rule.
There is no configuration value that means "unlimited".

### The drop policy, and why it is drop-oldest

When the source produces faster than the pipeline consumes, something must go.
For live safety monitoring the choice is not close:

* **The newest frame is the most valuable.** An operator asks "is that person
  wearing a hairnet *now*". A frame from twelve seconds ago answers a question
  nobody is asking.
* **Latency compounds.** Blocking the producer to preserve old frames makes the
  whole stream fall further behind with every dropped deadline, and the
  displayed state drifts from reality without ever saying so.

So the queue evicts the **oldest** frame and keeps the newest. Three properties
hold while it does:

1. **Timestamps are never reordered.** Eviction removes from the front; nothing
   is inserted out of order.
2. **Frames are never fabricated or duplicated.** A dropped frame is gone, and
   counted.
3. **Every drop is measured.** `frames_dropped` is a metric, not a silence.

Recorded replay uses the same queue with the same policy, because a replay whose
backpressure behaves differently is not testing the live path.
"""

from __future__ import annotations

import asyncio
import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any

#: Frames held between the decoder and the pipeline, per camera.
#:
#: Eight, not eighty. At the 4 fps analysis rate this is two seconds of slack —
#: enough to ride out a GC pause or a slow model call, short enough that a frame
#: which does reach the pipeline is still worth looking at. A deep queue on a
#: live stream is just latency with extra memory.
DEFAULT_QUEUE_CAPACITY = 8


class DropReason(enum.Enum):
    """Why a frame did not reach the pipeline. Every value is measured."""

    QUEUE_FULL = "queue_full"
    """The pipeline is slower than the source. The oldest frame was evicted."""

    SAMPLED_OUT = "sampled_out"
    """Deliberate: the source runs faster than the analysis rate."""

    SHUTTING_DOWN = "shutting_down"
    """The session is stopping and will not start new work."""


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """One decoded frame, and everything downstream needs to know about it.

    Immutable. A frame that can be edited after it enters the queue is a frame
    whose timestamp cannot be trusted.
    """

    camera_id: str
    """Set by configuration, never inferred from pixels. It is the partition key
    for tracking, the registry and the observation log, so a wrong value silently
    merges two kitchens."""

    sequence: int
    """Monotonic per source epoch. Survives a reconnect only if the source can
    genuinely continue; otherwise the epoch increments and this restarts."""

    epoch: int
    """Increments on every reconnect. Two frames with the same sequence and
    different epochs are different frames, and tracking must not associate
    across the boundary."""

    captured_at_ns: int
    """**Capture time**, from the source — not arrival time.

    Freshness ages attributes against this. Stamping arrival here would make
    every observation look newer than it is, and a stale answer would present as
    a current one. See §11."""

    received_at_ns: int
    """When this process saw it. Diagnostics only: the gap between this and
    `captured_at_ns` is the transport delay, which is worth graphing and must
    never be mistaken for observation time."""

    width: int
    height: int
    payload: bytes = field(repr=False)
    """BGR24. `repr=False` so a frame never renders megabytes into a log line."""

    @property
    def nbytes(self) -> int:
        return len(self.payload)

    @property
    def transport_delay_ms(self) -> float:
        return max(0.0, (self.received_at_ns - self.captured_at_ns) / 1_000_000)


@dataclass(slots=True)
class QueueStats:
    """What the queue has done. Read by metrics and by the diagnostics route."""

    capacity: int
    depth: int = 0
    high_water: int = 0
    accepted: int = 0
    dropped_queue_full: int = 0
    dropped_sampled: int = 0
    dropped_shutdown: int = 0

    @property
    def dropped_total(self) -> int:
        return self.dropped_queue_full + self.dropped_sampled + self.dropped_shutdown

    def to_wire(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "depth": self.depth,
            "high_water": self.high_water,
            "accepted": self.accepted,
            "dropped_total": self.dropped_total,
            "dropped_queue_full": self.dropped_queue_full,
            "dropped_sampled": self.dropped_sampled,
            "dropped_shutdown": self.dropped_shutdown,
        }


class LiveFrameQueue:
    """A bounded, drop-oldest frame queue for one camera.

    Single producer (the source), single consumer (the session). Not a general
    async queue: `put` **never blocks and never awaits**, because a producer that
    can be blocked by a slow consumer is a producer that stalls the decoder and
    turns a processing problem into a connection problem.
    """

    __slots__ = ("_closed", "_items", "_stats", "_waiter")

    def __init__(self, capacity: int = DEFAULT_QUEUE_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("queue capacity must be at least 1; there is no unbounded mode")
        self._items: deque[LiveFrame] = deque()
        self._stats = QueueStats(capacity=capacity)
        self._waiter: asyncio.Future[None] | None = None
        self._closed = False

    @property
    def stats(self) -> QueueStats:
        return self._stats

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, frame: LiveFrame) -> DropReason | None:
        """Offer a frame. Returns the reason it was dropped, or `None` if kept.

        When full, the **oldest** frame is evicted and this one is kept — the
        dropped frame is the old one, and the return value reports that a drop
        happened at all.
        """
        if self._closed:
            self._stats.dropped_shutdown += 1
            return DropReason.SHUTTING_DOWN

        dropped: DropReason | None = None
        if len(self._items) >= self._stats.capacity:
            self._items.popleft()
            self._stats.dropped_queue_full += 1
            dropped = DropReason.QUEUE_FULL

        self._items.append(frame)
        self._stats.accepted += 1
        self._stats.depth = len(self._items)
        self._stats.high_water = max(self._stats.high_water, self._stats.depth)

        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

        return dropped

    async def get(self) -> LiveFrame | None:
        """The next frame, waiting if necessary. `None` once closed and drained.

        `None` rather than an exception: end-of-stream is the normal way a
        session ends, and a normal ending should not travel as an error.
        """
        while True:
            if self._items:
                frame = self._items.popleft()
                self._stats.depth = len(self._items)
                return frame
            if self._closed:
                return None

            loop = asyncio.get_running_loop()
            self._waiter = loop.create_future()
            try:
                await self._waiter
            finally:
                self._waiter = None

    def record_sampled_out(self) -> None:
        """A frame the sampler declined. Counted separately from a queue drop.

        The distinction matters: sampled-out frames are the system working as
        configured, and queue-full frames are the system falling behind. A single
        `dropped` counter would hide the difference and make a healthy 25 fps
        camera look identical to an overloaded one.
        """
        self._stats.dropped_sampled += 1

    def close(self) -> None:
        """Stop accepting. A consumer still drains what is already queued."""
        self._closed = True
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def clear(self) -> None:
        """Release every held frame. Called on shutdown so nothing outlives it."""
        self._items.clear()
        self._stats.depth = 0


class FrameSampler:
    """Decides which frames reach the pipeline.

    A 25 fps camera must not become 25 fps of detection and VLM work. Sampling
    happens **here, once**, at the source boundary — not in the frontend, and not
    a second time inside the pipeline.

    Interval-based rather than every-Nth-frame: a source whose real rate drifts
    (RTSP under load frequently does) would otherwise deliver a wandering
    analysis rate, and the cost model would drift with it.
    """

    __slots__ = ("_interval_ns", "_last_ns")

    def __init__(self, analysis_fps: float) -> None:
        if analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive")
        self._interval_ns = int(1_000_000_000 / analysis_fps)
        self._last_ns: int | None = None

    def accepts(self, captured_at_ns: int) -> bool:
        """Whether this frame is due. Judged on **capture** time.

        Using arrival time would let a burst of buffered frames after a network
        hiccup all pass the gate at once, spending a second of model budget on a
        second of already-stale video.
        """
        if self._last_ns is None or captured_at_ns - self._last_ns >= self._interval_ns:
            self._last_ns = captured_at_ns
            return True
        return False

    def reset(self) -> None:
        """After a reconnect. The next frame is always due."""
        self._last_ns = None


__all__ = [
    "DEFAULT_QUEUE_CAPACITY",
    "DropReason",
    "FrameSampler",
    "LiveFrame",
    "LiveFrameQueue",
    "QueueStats",
]
