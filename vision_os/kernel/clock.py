"""Clock implementations (08_RUNTIME_AND_THREADING section 6).

Three clocks, one protocol:

``SystemClock``
    Production. Real time, with a monotonic reading that is immune to NTP steps.

``VirtualClock``
    Deterministic mode. Time advances only by explicit control, driven by frame
    PTS. This is what makes replay reproducible (invariant V13) and what lets a
    30-day soak of time-driven behaviour run in minutes.

``ScaledClock``
    Soak testing at N x real time.

Nothing in the platform calls ``time.time()`` directly. The architecture boundary
test enforces this.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import threading
import time

from ..core.model.timebase import Duration, Instant


class SystemClock:
    """Real time. ``monotonic()`` is immune to wall-clock steps.

    Cadence accumulators and timeouts use ``monotonic()`` so that an NTP step or
    a DST transition can never corrupt scheduling phase or make a measured
    duration negative (08_RUNTIME section 9).
    """

    __slots__ = ()

    def now(self) -> Instant:
        return Instant(time.time_ns())

    def monotonic(self) -> Instant:
        return Instant(time.monotonic_ns())

    async def sleep(self, duration: Duration) -> None:
        if duration.ns > 0:
            await asyncio.sleep(duration.seconds)

    @property
    def is_virtual(self) -> bool:
        return False


class ScaledClock:
    """Real time compressed by ``factor``.

    A 30-day soak of retention sweeps, staleness expiry and dormancy transitions
    runs in hours. Memory and handle leaks still require real-time soaking,
    because they are driven by allocation counts rather than clock ticks
    (14_TESTING section 10.2).
    """

    __slots__ = ("_factor", "_origin_wall", "_origin_mono")

    def __init__(self, factor: float = 60.0) -> None:
        if factor <= 0:
            raise ValueError(f"scale factor must be positive, got {factor}")
        self._factor = factor
        self._origin_wall = time.time_ns()
        self._origin_mono = time.monotonic_ns()

    def now(self) -> Instant:
        elapsed = time.monotonic_ns() - self._origin_mono
        return Instant(self._origin_wall + int(elapsed * self._factor))

    def monotonic(self) -> Instant:
        elapsed = time.monotonic_ns() - self._origin_mono
        return Instant(int(elapsed * self._factor))

    async def sleep(self, duration: Duration) -> None:
        if duration.ns > 0:
            await asyncio.sleep(duration.seconds / self._factor)

    @property
    def is_virtual(self) -> bool:
        return False


class VirtualClock:
    """Time advanced only by explicit control. The basis of deterministic mode.

    Sleepers are woken in deadline order, and ties are broken by insertion order,
    so a replay produces the same interleaving every run.
    """

    __slots__ = ("_now_ns", "_lock", "_sleepers", "_counter")

    def __init__(self, start: Instant | None = None) -> None:
        self._now_ns = start.ns if start else 0
        self._lock = threading.RLock()
        self._sleepers: list[tuple[int, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    def now(self) -> Instant:
        with self._lock:
            return Instant(self._now_ns)

    def monotonic(self) -> Instant:
        with self._lock:
            return Instant(self._now_ns)

    async def sleep(self, duration: Duration) -> None:
        if duration.ns <= 0:
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        with self._lock:
            deadline = self._now_ns + duration.ns
            heapq.heappush(self._sleepers, (deadline, next(self._counter), future))
        await future

    def advance(self, duration: Duration) -> None:
        """Move time forward, releasing any sleeper whose deadline has passed."""
        if duration.ns < 0:
            raise ValueError("virtual time cannot move backwards")
        with self._lock:
            target = self._now_ns + duration.ns
            due: list[asyncio.Future[None]] = []
            while self._sleepers and self._sleepers[0][0] <= target:
                _, _, future = heapq.heappop(self._sleepers)
                due.append(future)
            self._now_ns = target
        for future in due:
            if not future.done():
                future.get_loop().call_soon_threadsafe(_resolve, future)

    def set_to(self, instant: Instant) -> None:
        with self._lock:
            if instant.ns < self._now_ns:
                raise ValueError("virtual time cannot move backwards")
            delta = instant.ns - self._now_ns
        self.advance(Duration(delta))

    @property
    def pending_sleepers(self) -> int:
        with self._lock:
            return len(self._sleepers)

    @property
    def is_virtual(self) -> bool:
        return True


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)
