# P10 Phase 3 · Step 1 — Thread-Safety Audit

**Date:** 2026-09-01
**Scope:** can `UnderstandingEngine.understand_batch()` be moved off the ANALYSIS event
loop via `asyncio.to_thread`, without changing any semantics?
**Production code changed in this step:** **none.**

---

## 0. The finding that reframes the question

**`understand_batch()` already runs its work on worker threads today.**

```python
workers = self._batch_workers(requests)          # min(max_concurrency, remote_concurrency, len)
if workers <= 1:
    for request in requests:                      # ← inline, on the CALLER's thread
        results[...] = self.understand(request, ...)
    return results

with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vos-understand") as pool:
    futures = {pool.submit(self.understand, request, ...): request.request_id ...}
    for future, request_id in futures.items():
        results[request_id] = future.result()     # ← caller BLOCKS here
```

So the question is **not** "is this code safe to run on a thread" — for any batch of 2 or
more it is already doing exactly that, and has been in production throughout. The only
thing the caller does is **block on `future.result()`**, and it does that while sitting on
the ANALYSIS event loop.

`asyncio.to_thread` therefore does not introduce concurrency. It moves *one blocking wait*
off the loop. The inner threading model — how many threads, which objects they touch — is
**unchanged**.

The one path that genuinely runs on the caller's thread is `workers <= 1`, i.e. a batch of
exactly one request. Measured `understanding.batch_size` p50 = 3, p95 = 4, so most batches
already use the pool; single-request batches execute the model inline on the analysis loop.

---

## 1. Question-by-question

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Is `understand_batch()` re-entrant? | **Yes, for the concurrency it permits** | No instance state is held across the call; all shared maps are lock-guarded (§2) |
| 2 | Does it mutate `self`? | **Yes — 4 fields**, 3 correctly guarded | `_requests`, `_failures`, `_rejection_window` under `_guard`; `_breakers`/`_semaphores` guarded on *creation* only (§3) |
| 3 | Does the NVIDIA adapter hold mutable request/session state? | **No request state.** Counters only, all lock-guarded | `_model/_key/_base/_timeout/_max_side/_producible` set once in `__init__`, never reassigned |
| 4 | Are retry counters shared? | **Yes — `CircuitBreaker` is shared and unguarded** | §3 — pre-existing, bounded impact |
| 5 | Is HTTP client state shared? | **No** | `urllib.request.urlopen(...)` per call; no opener, no session, no pool |
| 6 | Are metrics thread-safe? | **Yes** | `metrics/engine.py:131` `threading.Lock` |
| 7 | Are callbacks invoked inside? | **Yes — event-bus publish** | `bus.py:98` `Lock`, `:205` `RLock`; no subscriber touches asyncio (grep of `taps.py`: zero matches) |
| 8 | Does it touch asyncio primitives? | **NO — zero occurrences** | `grep asyncio\|await\|async def` over `engine.py` and `nvidia_vl.py`: **no matches in either** |
| 9 | Does it touch registry/database state? | **No** | Registry write happens in `_publish()`, *after* `understand_batch` returns |
| 10 | Does it depend on the analysis loop thread? | **No** | Follows from 8 and 9 |
| 11 | Does the semaphore protect the actual remote operation? | **Yes** | `_semaphore_for` acquired in `_invoke`, released in `finally`, around `adapter.understand()` |
| 12 | Is `_lock` required around the whole batch, or only scheduling? | **Only scheduling** — and holding it across the batch is the real risk (§4) |

---

## 2. Locking inventory — verified, not assumed

| Object | Protection | Verified at |
|---|---|---|
| `UnderstandingEngine._requests` / `_failures` | `threading.Lock` (`_guard`) | `engine.py:194` |
| `_rejection_window` | `_guard`, correctly held for every append/read/clear | `engine.py:959–973` |
| `_breakers` / `_semaphores` maps | `_guard` + `setdefault` re-check under lock | `engine.py:911–928` |
| `ModelSemaphore` | `threading.Condition` | `cache.py:209` |
| `ResponseCache` | `threading.Lock` | `cache.py:114` |
| `MetricsEngine` | `threading.Lock` | `metrics/engine.py:131` |
| `EventBus` | `Lock` + `RLock` | `bus.py:98`, `:205` |
| `NvidiaVisionUnderstander.stats`, `_recent`, `retired` | `threading.Lock` on **every** mutation | `nvidia_vl.py:330,361,369,377,388,396,413,428,506,576,663` |

The engine's own constructor comment already states the design intent:

> *"Guards this engine's own bookkeeping when `understand_batch` runs requests concurrently
> … Deliberately NOT held across a model call. The semaphore, the cache, the metrics engine
> and the event bus each carry their own lock."*

Every claim in that comment was checked and holds.

---

## 3. Two pre-existing gaps (neither introduced nor worsened by this change)

### 3.1 `CircuitBreaker` mutation is unguarded

`CircuitBreaker` is a plain mutable dataclass with **no lock**. `_breaker_for()` guards the
map, but the mutations are called outside `_guard`:

```python
breaker.record_failure(self._clock.monotonic().ns)   # engine.py:343, 351, 355
breaker.record_success()                             # engine.py:363
```

`consecutive_failures += 1` is an unguarded read-modify-write reachable from up to
`remote_concurrency` threads.

**Impact:** a lost increment delays a breaker trip by one failure. Bounded by concurrency
(2). **Not a safety property** — the breaker is a cost control, and no PPE value depends on
it.

**Already true today**, at exactly the same concurrency, because the pool already runs these
paths in parallel. `to_thread` does not change the number of threads that can reach it.

### 3.2 `capabilities()` reads latency stats without the lock

```python
latency_p50_ms=self.stats.percentile(0.5),   # nvidia_vl.py:313 — no `with self._lock`
```

`percentile()` calls `sorted(self.latencies)` while `observe()` may append or `del [0]`.
Worst case is a stale or briefly-inconsistent percentile in a capability report. **No
attribute, verdict or refusal depends on it.** Also pre-existing.

**Neither is fixed in this phase.** Both are recorded so they are not discovered later and
misattributed to the offload. Fixing them is a separate, small change and does not gate this
one.

---

## 4. The real risk is NOT thread safety — it is lock waiters

This is the finding that determines the implementation shape.

Today, `on_crops` holds an **`asyncio.Lock`** across `_run_ready()`:

```python
async with self._lock:
    self._enqueue(result, crops)
    await self._run_ready()          # ← currently blocks the loop outright
```

Because the loop is blocked, **no other `on_crops` task can even be created** while a batch
runs. The crop sink itself runs on that loop. So there is no backlog today — the whole loop
is simply frozen.

**After the offload, the loop is free.** The sink will keep firing
`loop.create_task(on_crops(...))` for every crop batch, and each of those will queue up on
`async with self._lock`. Awaiting coroutines are **not** bounded by the deque:

| Buffer | Bounded? | |
|---|---|---|
| `self._queue` | ✅ `deque(maxlen=queue_capacity)`, `drop_oldest`, counted | `runtime.py:125, 204` |
| **`asyncio.Lock` waiters** | ❌ **unbounded** | one per `on_crops` task |

Each waiter holds its `crops` tuple — i.e. **pixels** — so unbounded waiters means unbounded
memory, not just unbounded scheduling.

This is precisely what the brief forbids: *"Do not accidentally create an unbounded executor
backlog"* and *"Do not queue an unlimited number of requests waiting for VLM slots."*

**A naive one-line `await asyncio.to_thread(...)` inside the existing `async with self._lock`
would introduce exactly that.** The proposed change is safe with respect to threads and
unsafe with respect to backlog.

---

## 5. Required implementation shape

The offload must keep the existing **shed-not-queue** principle — the same principle the
architecture already states for this edge:

> *"enrichment is shed rather than queued, because a queued call outlives the frame it
> describes"*

So the design is:

1. **Enqueue always** — into the existing bounded deque, which already drops oldest and
   counts overflow. Unchanged semantics, unchanged counters.
2. **One runner at a time** — if a batch is already in flight, `on_crops` returns
   immediately after enqueueing instead of awaiting the lock. No waiter accumulates.
3. **Offload only the blocking call** — `understand_batch` goes to a thread; `_publish`
   (registry write-back) stays on the loop, preserving write ordering and keeping the
   registry single-threaded exactly as today.
4. **Drain what arrived during the batch** — loop until the queue is empty, so items
   enqueued mid-batch are not stranded until the next crop.

What is explicitly **not** changed: the `ModelSemaphore` remains the authority, the cap
remains `remote_concurrency = 2` across all four cameras, `_batch_workers` still bounds the
pool, refusal semantics, `observed_at`, and every compliance rule.

---

## 6. Concurrency ceiling after the change

| Layer | Bound | Source |
|---|---|---|
| Batch executor workers | `min(max_concurrency=4, remote_concurrency=2, len(batch))` = **≤ 2** | `_batch_workers` |
| In-flight remote calls per model | **≤ 2**, globally across all cameras | `ModelSemaphore(limit)` keyed on `adapter_id` |
| `to_thread` executor threads | **1 at a time** — one runner, one `to_thread` call | design §5.2 |
| Understanding queue | `deque(maxlen=…)`, drop-oldest, counted | `runtime.py:125` |
| Lock waiters | **0** — by construction | design §5.2 |

The semaphore stays the authority. `to_thread` adds **one** thread that then blocks on
`future.result()` inside the existing pool — it does not multiply anything.

---

## 7. Verdict

**SAFE TO OFFLOAD**, with the shape in §5 — **not** with the naive one-liner.

- Thread safety: **established**. Every shared object is lock-guarded; neither the engine nor
  the adapter imports asyncio; the work already runs on pool threads today.
- Two pre-existing unguarded spots (§3) are recorded, are not safety-critical, and are
  unaffected by this change.
- The genuine risk is **unbounded `asyncio.Lock` waiters holding crop pixels** (§4), which
  the one-line version would create. §5 closes it by preserving shed-not-queue.

Proceed to implementation.
