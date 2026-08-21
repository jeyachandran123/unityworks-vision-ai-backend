"""The understanding budget — a hard ceiling on expensive inference per unit time.

> **Single responsibility:** *Decide whether the platform can afford one more
> expensive call. Trigger nothing, crop nothing.*

**The budget is shared across cameras**, and that is deliberate (§M8 Thread
Safety): *"uses atomic counters with periodic reconciliation — the same trade as
M3, and for the same reason."*

A per-camera budget would not be a budget. Understanding cost is a property of
the node's GPU, not of any camera, and a per-camera cap cannot stop 100 cameras
each staying under their own limit while collectively exhausting the device.

What is *not* shared is coordination: no camera ever waits on another. The
critical section is a counter increment, held for nanoseconds, exactly as M3's
global FPS budget already works.

**Exhaustion is never silence.** §M8: skip by priority, emit ``BudgetExhausted``,
and *"publish coverage observations so consumers know attributes are thinned"*
(V8).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from ...core.model.ids import CropId, TenantId
from ...core.model.timebase import Duration, Instant

#: Default reconciliation window — one minute.
#:
#: Short enough that an overspend is corrected within a minute, long enough that
#: the counter is not rolling constantly. §M8 calls for *"periodic
#: reconciliation"*, not continuous metering.
DEFAULT_BUDGET_WINDOW = Duration.from_millis(60_000)

#: The monotonic origin. A budget constructed without a clock reading starts its
#: window at zero and rolls on the first ``try_spend`` past the window.
EPOCH = Instant(0)


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """What ``budget_status()`` returns (§M8 Public API)."""

    ceiling_per_hour: float
    spent_in_window: int
    window_elapsed: Duration
    pressure: float
    """Spend rate over ceiling. Above 1.0 the budget is being exceeded and the
    next request will be shed."""

    exhausted: bool
    shed_in_window: int

    @property
    def remaining_in_window(self) -> float:
        """Calls still affordable this window. Never negative."""
        allowance = self.ceiling_per_hour * (self.window_elapsed.seconds / 3600.0)
        return max(0.0, allowance - self.spent_in_window)


@dataclass(slots=True)
class _DemandSpend:
    """Per-demand accounting, so a consumer's own ceiling can be honoured."""

    calls: int = 0
    window_started_ns: int = 0


class UnderstandingBudget:
    """A hard ceiling on expensive calls, shared across cameras.

    Thread-safe by a short-held lock rather than by partitioning, because the
    resource being protected — the node's inference capacity — is genuinely
    shared. Cameras contend for a counter increment and never for each other's
    progress.
    """

    __slots__ = (
        "_ceiling_per_hour",
        "_credit",
        "_demand_spend",
        "_lock",
        "_shed_in_window",
        "_spent_in_window",
        "_window",
        "_window_started",
    )

    def __init__(
        self,
        *,
        ceiling_per_hour: float,
        window: Duration = DEFAULT_BUDGET_WINDOW,
        now: Instant = EPOCH,
    ) -> None:
        if ceiling_per_hour < 0:
            raise ValueError("ceiling_per_hour must be non-negative")
        if window.ns <= 0:
            raise ValueError("the reconciliation window must be positive")
        self._ceiling_per_hour = ceiling_per_hour
        self._window = window
        self._window_started = now
        self._spent_in_window = 0
        self._shed_in_window = 0
        self._demand_spend: dict[str, _DemandSpend] = {}
        self._lock = threading.Lock()

        # Start full, so the node is not blind for its first window after boot.
        self._credit = self._burst()

    @property
    def ceiling_per_hour(self) -> float:
        return self._ceiling_per_hour

    def try_spend(
        self, now: Instant, *, demand_ids: tuple[str, ...] = (), cost: int = 1
    ) -> bool:
        """Reserve budget for one expensive call.

        Returns ``False`` when the ceiling is reached — the caller sheds by
        priority and records ``BUDGET_EXHAUSTED``. It never blocks: waiting for
        budget would convert a cost problem into a latency problem, and the
        frame will be gone by the time the wait ends.
        """
        with self._lock:
            self._reconcile_locked(now)
            if self._credit < cost:
                self._shed_in_window += 1
                return False
            self._credit -= cost
            self._spent_in_window += cost
            for demand_id in demand_ids:
                spend = self._demand_spend.setdefault(demand_id, _DemandSpend())
                spend.calls += cost
            return True

    def refund(self, cost: int = 1, *, demand_ids: tuple[str, ...] = ()) -> None:
        """Return unspent budget.

        Called when a reserved call did not happen — a gate rejection after
        reservation, or an extraction failure. Without it, a run of failures
        would exhaust the budget having bought nothing.
        """
        with self._lock:
            self._credit = min(self._burst(), self._credit + cost)
            self._spent_in_window = max(0, self._spent_in_window - cost)
            for demand_id in demand_ids:
                spend = self._demand_spend.get(demand_id)
                if spend is not None:
                    spend.calls = max(0, spend.calls - cost)

    def demand_calls(self, demand_id: str) -> int:
        with self._lock:
            spend = self._demand_spend.get(demand_id)
            return spend.calls if spend else 0

    def status(self, now: Instant) -> BudgetStatus:
        with self._lock:
            self._reconcile_locked(now)
            elapsed = Duration(max(1, now.ns - self._window_started.ns))
            return BudgetStatus(
                ceiling_per_hour=self._ceiling_per_hour,
                spent_in_window=self._spent_in_window,
                window_elapsed=elapsed,
                pressure=self._pressure_locked(),
                exhausted=self._credit < 1.0,
                shed_in_window=self._shed_in_window,
            )

    def pressure(self, now: Instant) -> float:
        return self.status(now).pressure

    @property
    def credit(self) -> float:
        """Calls currently affordable. Fractional while a slow rate accrues."""
        with self._lock:
            return self._credit

    # --- internals ------------------------------------------------------------- #

    def _reconcile_locked(self, now: Instant) -> None:
        """Refill the allowance for whole windows elapsed.

        **Periodic reconciliation**, per §M8 — a token bucket refilled on a
        window boundary rather than a sliding window, which would need a
        timestamp per call.

        Credit *carries across windows* rather than being reset, and that is the
        load-bearing detail. A ceiling of 30 calls/hour over a 60-second window
        earns 0.5 calls per window: discarding the remainder each time would mean
        an integer call never becomes affordable, and the platform would go
        permanently blind on a configuration that reads as perfectly reasonable.
        """
        elapsed = now.ns - self._window_started.ns
        if elapsed < self._window.ns:
            return
        windows = elapsed // self._window.ns
        self._window_started = Instant(self._window_started.ns + windows * self._window.ns)
        self._credit = min(self._burst(), self._credit + windows * self._allowance_locked())
        self._spent_in_window = 0
        self._shed_in_window = 0
        self._demand_spend.clear()

    def _allowance_locked(self) -> float:
        return self._ceiling_per_hour * (self._window.seconds / 3600.0)

    def _burst(self) -> float:
        """The most credit that may accumulate.

        Capped at one window's allowance so a quiet night does not buy an
        unbounded morning burst — a ceiling that can be saved up is not a ceiling
        on instantaneous load, which is the thing the GPU actually has.

        The floor of 1.0 is what lets a very low rate ever spend at all: without
        it a sub-one allowance would accrue toward a ceiling it could never
        cross.
        """
        return max(1.0, self._allowance_locked()) if self._ceiling_per_hour > 0 else 0.0

    def _pressure_locked(self) -> float:
        allowance = self._allowance_locked()
        if allowance <= 0:
            return 4.0 if self._spent_in_window else 0.0
        return min(4.0, self._spent_in_window / allowance)


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    entries: int
    capacity: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class CropDeduplicationCache:
    """Bounded LRU of recently produced crops (§M8 failure handling).

    **Every key includes the tenant** — `12_SECURITY` §4: *"Every cache key
    includes `tenant_id` — including crop and understanding caches."* A cache
    keyed on content alone would let one tenant's crop satisfy another tenant's
    request, which is a cross-tenant data path wearing the disguise of an
    optimization.

    Bounded because §M8 names cache growth as a failure mode: an unbounded
    dedup cache is a memory leak that looks like a hit-rate improvement.
    """

    __slots__ = ("_capacity", "_entries", "_evictions", "_hits", "_lock", "_misses")

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be >= 1")
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, str], CropId] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(tenant_id: TenantId, content_hash: str) -> tuple[str, str]:
        """The cache key. Tenant first, so a scan by prefix is tenant-scoped."""
        return (str(tenant_id), content_hash)

    def get(self, tenant_id: TenantId, content_hash: str) -> CropId | None:
        with self._lock:
            key = self.key(tenant_id, content_hash)
            found = self._entries.get(key)
            if found is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return found

    def put(self, tenant_id: TenantId, content_hash: str, crop_id: CropId) -> None:
        with self._lock:
            key = self.key(tenant_id, content_hash)
            self._entries[key] = crop_id
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def forget_tenant(self, tenant_id: TenantId) -> int:
        """Drop one tenant's entries. Used by erasure.

        Erasure that could not reach the cache would leave imagery references
        alive after the data they point at was deleted.
        """
        with self._lock:
            prefix = str(tenant_id)
            doomed = [k for k in self._entries if k[0] == prefix]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=len(self._entries),
                capacity=self._capacity,
                evictions=self._evictions,
            )

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(order=True, slots=True)
class _PrioritisedRequest:
    """A request in the shedding queue.

    Ordered by ``(rank, sequence)`` — the sequence breaks ties by arrival so
    shedding is deterministic. Without it, two requests of equal priority would
    shed in whatever order the heap happened to hold them (V13).
    """

    rank: int
    sequence: int
    payload: object = field(compare=False)


class PriorityQueue:
    """Orders candidates when demand exceeds budget (§M8 responsibility 6).

    Priority is an **opaque class**. This queue maps a class to a rank using a
    configured ordering and never asks what the class means — the reason a class
    exists lives with the consumer (V1/V2).
    """

    __slots__ = ("_order", "_sequence")

    def __init__(self, priority_order: tuple[str, ...] = ()) -> None:
        # Earlier in the tuple means higher priority. An unlisted class sorts
        # last, so an unknown priority degrades to "least important" rather than
        # to an exception — a consumer typo must not stop the pipeline.
        self._order = {name: index for index, name in enumerate(priority_order)}
        self._sequence = 0

    def rank(self, priority_class: str) -> int:
        return self._order.get(priority_class, len(self._order))

    def order(self, requests: list) -> list:
        """Sort requests most-important first, deterministically."""
        decorated = []
        for request in requests:
            self._sequence += 1
            decorated.append(
                _PrioritisedRequest(
                    rank=self.rank(getattr(request, "priority_class", "")),
                    sequence=self._sequence,
                    payload=request,
                )
            )
        decorated.sort()
        return [item.payload for item in decorated]

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(sorted(self._order, key=lambda name: self._order[name]))
