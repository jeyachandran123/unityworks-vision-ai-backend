"""The response cache, and the concurrency machinery around model calls.

> 04_MODULES §M9 State Ownership: *"response cache keyed by `(CropId,
> prompt_version, model_version, attribute_set)`."*

The key is the whole design, and §M9 says why:

> *Because `CropId` is a content hash and prompt/model versions are explicit, the
> cache is **correct by construction** — a cache hit is guaranteed to be the
> answer the current configuration would produce. Caches keyed on object id or
> timestamp instead are the usual source of stale-attribute bugs.*

Read that carefully: correctness here is not achieved by invalidation. There is
nothing to invalidate. Change the prompt and the key changes; change the model
and the key changes; change the pixels and the `CropId` changes. A stale entry
cannot be *reached*, so it cannot be *served*.

**Tenancy is in the key too** (12_SECURITY §4). Content addressing means two
tenants photographing the same scene could produce identical crop hashes, and an
untenanted cache would let one tenant's answer serve another's request.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from ...core.model.ids import AttributeKey, CropId, TenantId
from ...core.model.timebase import Duration, Instant
from ...core.model.understanding import UnderstandingResult

#: The cache key. Every component is load-bearing; see the module docstring.
CacheKey = tuple[str, str, str, str, str]


def cache_key(
    *,
    tenant_id: TenantId,
    crop_id: CropId,
    prompt_version: str,
    model_version: str,
    attributes: Sequence[AttributeKey],
) -> CacheKey:
    """Build the documented key.

    ``attributes`` is **sorted** before joining: asking for ``(a, b)`` and
    ``(b, a)`` is the same question, and a key that distinguished them would halve
    the hit rate for no reason and make the cache's behaviour depend on the order
    a demand happened to list its attributes.
    """
    return (
        str(tenant_id),
        str(crop_id),
        prompt_version,
        model_version,
        ",".join(sorted(str(key) for key in attributes)),
    )


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    entries: int
    capacity: int
    evictions: int
    expirations: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass(slots=True)
class _Entry:
    result: UnderstandingResult
    stored_at: Instant


class ResponseCache:
    """Bounded, tenant-scoped LRU over understanding results.

    Bounded because an unbounded cache in the most memory-hungry layer of the
    platform is a leak that presents as a rising hit rate. TTL'd as well as
    bounded because a result can outlive the *world* it described even though it
    cannot outlive its *configuration* — the same crop is still the same crop, but
    a consumer reading a six-hour-old answer wants to know it is six hours old.
    """

    __slots__ = (
        "_capacity",
        "_entries",
        "_evictions",
        "_expirations",
        "_hits",
        "_lock",
        "_misses",
        "_ttl",
    )

    def __init__(self, *, capacity: int = 2048, ttl: Duration | None = None) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be >= 1")
        self._capacity = capacity
        self._ttl = ttl
        self._entries: OrderedDict[CacheKey, _Entry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._lock = threading.Lock()

    def get(self, key: CacheKey, now: Instant) -> UnderstandingResult | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if self._expired(entry, now):
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.result

    def put(self, key: CacheKey, result: UnderstandingResult, now: Instant) -> None:
        """Store a result.

        **Failures are not cached.** A timeout or an unavailable model is a fact
        about this moment, not about this crop, and caching it would extend a
        transient outage for the life of the entry. Only answers are cached —
        including the honest answer that nothing fit the schema, which *is* a
        property of these pixels and this prompt.
        """
        if result.outcome.is_failure:
            return
        with self._lock:
            self._entries[key] = _Entry(result=result.without_raw_output(), stored_at=now)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1

    def _expired(self, entry: _Entry, now: Instant) -> bool:
        if self._ttl is None:
            return False
        return now.ns - entry.stored_at.ns > self._ttl.ns

    def forget_tenant(self, tenant_id: TenantId) -> int:
        """Drop one tenant's entries. Used by erasure.

        Erasure that could not reach the cache would leave a tenant's model
        output resolvable after the data behind it was deleted.
        """
        with self._lock:
            prefix = str(tenant_id)
            doomed = [key for key in self._entries if key[0] == prefix]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=len(self._entries),
                capacity=self._capacity,
                evictions=self._evictions,
                expirations=self._expirations,
            )

    def __len__(self) -> int:
        return len(self._entries)


class ModelSemaphore:
    """Bounded in-flight requests, per model (08_RUNTIME §1).

    > *A **semaphore per model** caps in-flight requests to what the device or
    > remote endpoint sustains.*

    Per model rather than global, because a remote API's rate limit and a local
    GPU's memory are different constraints with different numbers, and one
    counter cannot express both. 08_RUNTIME §4.4 adds the reason this matters
    beyond throughput: *"Long VLM calls are not preempted; instead concurrency is
    capped so that the detector's latency budget is protected."*
    """

    __slots__ = ("_condition", "_in_flight", "_limit", "_peak", "_rejections", "_waits")

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("concurrency limit must be >= 1")
        self._limit = limit
        self._in_flight = 0
        self._peak = 0
        self._waits = 0
        self._rejections = 0
        self._condition = threading.Condition()

    def try_acquire(self) -> bool:
        """Take a slot without waiting.

        Non-blocking because understanding is best-effort: 08_RUNTIME §5.2 gives
        the crop-to-understanding queue a `drop_oldest` policy precisely because
        *"losing an enrichment is acceptable"*. Blocking here would convert a
        capacity problem into a latency problem in the layer beneath.
        """
        with self._condition:
            if self._in_flight >= self._limit:
                self._rejections += 1
                return False
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
            return True

    def acquire(self, timeout_s: float | None = None) -> bool:
        """Take a slot, waiting up to ``timeout_s``. Used by batch workers."""
        with self._condition:
            if self._in_flight >= self._limit:
                self._waits += 1
                if not self._condition.wait_for(
                    lambda: self._in_flight < self._limit, timeout=timeout_s
                ):
                    self._rejections += 1
                    return False
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
            return True

    def release(self) -> None:
        with self._condition:
            self._in_flight = max(0, self._in_flight - 1)
            self._condition.notify()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def peak(self) -> int:
        return self._peak

    @property
    def rejections(self) -> int:
        return self._rejections

    @property
    def utilization(self) -> float:
        return self._in_flight / self._limit

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class BatchGroup:
    """Requests that may be sent to a model together.

    08_RUNTIME §1: *"Requests are grouped by `(model, prompt_version)` — only
    compatible requests batch together."* Two requests with different prompts are
    two different questions, and batching them would mean sending one prompt and
    attributing the answer to both.
    """

    adapter_id: str
    prompt_pinned: str
    indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.indices)


def group_for_batching(
    keys: Sequence[tuple[str, str]], *, max_batch_size: int
) -> tuple[BatchGroup, ...]:
    """Partition requests into compatible, size-bounded batches.

    ``keys`` is ``(adapter_id, prompt_pinned)`` per request, in request order.
    Groups come back in first-appearance order and each group's indices are
    ascending, so a batch's composition is a pure function of its input — which
    08_RUNTIME §4.3 requires of deterministic mode, and which makes a replay
    reproduce the same batches.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    ordered: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(keys):
        if key not in buckets:
            buckets[key] = []
            ordered.append(key)
        buckets[key].append(index)

    groups: list[BatchGroup] = []
    for key in ordered:
        indices = buckets[key]
        for start in range(0, len(indices), max_batch_size):
            groups.append(
                BatchGroup(
                    adapter_id=key[0],
                    prompt_pinned=key[1],
                    indices=tuple(indices[start : start + max_batch_size]),
                )
            )
    return tuple(groups)
