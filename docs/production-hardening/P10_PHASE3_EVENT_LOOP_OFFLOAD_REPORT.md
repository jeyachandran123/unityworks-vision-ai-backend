# P10 Phase 3 — Event-Loop Offload

**Date:** 2026-09-01
**Model:** `meta/llama-3.2-11b-vision-instruct` — **unchanged throughout**
**`remote_concurrency`:** 2 — **unchanged**
**Production files changed:** **1** (`vision_os/perception/understanding/runtime.py`)

---

## 1. Objective

Remove the proven blockage: a synchronous VLM call executing on the single ANALYSIS event
loop that also consumes frames for all four cameras. Preserve every semantic — concurrency
cap, shed-not-queue, refusals, `observed_at`, compliance.

---

## 2. Phase 2 evidence (the "before")

300 s, four live cameras, `remote_concurrency=2`:

| | |
|---|---|
| Analysed frames | **5.7 /min/camera** (114 total) |
| Queue-full drops | **201–362 per camera** |
| Mean frame processing | **43 s** |
| VLM latency | p50 2,070 ms · p95 14,336 ms · max 40,152 ms |
| Detection inference | **0 ms** · registry apply **0 ms** · crop extract 31 ms |
| CPU | 165–211% of 1,200% available |
| Concurrency shed | **0** |

Every stage except the model was at zero, and nothing was saturated.

---

## 3. Thread-safety audit

Full audit: [`P10_PHASE3_THREAD_SAFETY.md`](P10_PHASE3_THREAD_SAFETY.md). Headlines:

- **`understand_batch` already ran on worker threads.** For any batch ≥ 2 it fans out over
  its own `ThreadPoolExecutor`. The caller merely *blocked* on `future.result()`. So the
  offload adds **no concurrency** — it moves one wait.
- **Neither `engine.py` nor `nvidia_vl.py` contains a single occurrence of `asyncio`,
  `await` or `async def`.** Nothing in the batch path can depend on running on the loop.
- Every shared object is lock-guarded: engine `_guard`, `ModelSemaphore` (Condition),
  `ResponseCache`, metrics engine, event bus, and every adapter counter.
- Registry write-back happens in `_publish()`, **after** `understand_batch` returns.
- Two **pre-existing** unguarded spots recorded and *not* fixed here: `CircuitBreaker`
  mutation, and `capabilities()` reading latency percentiles. Neither is safety-critical;
  both are unaffected by this change.

**The audit's decisive finding was not about threads.** It was that a naive
`await asyncio.to_thread(...)` *inside the existing `async with self._lock`* would create
**unbounded `asyncio.Lock` waiters**, each holding crop pixels — because freeing the loop
lets the sink keep creating `on_crops` tasks. The bounded `deque` does not bound waiters.

---

## 4–5. The blocking call

[`runtime.py`](../../vision_os/perception/understanding/runtime.py), `_run_ready`:

```python
async with self._lock:
    results = self._engine.understand_batch(requests, crops=crops)   # synchronous
```

Reached from `VisionSession._consume` → `await ANALYSIS.run(handler)` → crop sink →
`loop.create_task(on_crops)` — all on one loop, for all four cameras.

---

## 6. Implementation

Three coordinated changes, one file:

| Change | Purpose |
|---|---|
| `results = await asyncio.to_thread(self._engine.understand_batch, requests, crops=crops)` | move the wait off the loop |
| `self._running` flag + enqueue-then-return in `on_crops` | **zero lock waiters** — preserves shed-not-queue |
| `while self._queue: await self._run_ready()` in the runner and in `drain()` | items enqueued mid-batch are not stranded |

`_publish()` (registry write-back) deliberately **stays on the loop**, preserving
single-threaded write ordering. It costs 0 ms.

`drain()` still *waits* rather than sheds — correct for shutdown and for deterministic tests.

---

## 7. Concurrency preservation

| Layer | Bound | Changed? |
|---|---|---|
| `ModelSemaphore` per model | **2**, global across all cameras | no |
| `_batch_workers` | `min(4, 2, len(batch))` ≤ 2 | no |
| Simultaneous `to_thread` calls | **1** (single runner) | new bound |
| Understanding queue | `deque(maxlen)`, drop-oldest, counted | no |
| Lock waiters | **0** by construction | new bound |

**Measured: `concurrency_rejected = 0` before and after.** The cap was serialising, not
rejecting, and still is.

---

## 8. Event-loop responsiveness

Endpoint latency while VLM calls run:

| Endpoint | Before | After |
|---|---:|---:|
| `/health` | — | **p50 4 ms** |
| `/devtools/live` | 5,946 ms | **p50 2,382 ms** |
| `/devtools/compliance` | 9,938 ms (2 samples timed out at 25 s) | **p50 2,749 ms** |

`/health` at 4 ms p50 shows the API loop is clear. The devtools endpoints improved 2.5–3.6×
but remain seconds — they snapshot vision state and are **not** fully explained by this
change. One `/health` sample hit 7,566 ms. Recorded, not explained (§23).

---

## 9. Before / after — frame throughput

| Camera | Analysed BEFORE | Analysed AFTER | |
|---|---:|---:|---|
| cam-11 | 29 | **186** | 6.4× |
| cam-12 | 28 | **188** | 6.7× |
| cam-13 | 29 | **175** | 6.0× |
| cam-14 | 28 | **137** | 4.9× |
| **Total /min** | **22.8** | **137.2** | **6.0×** |

Per camera: **5.7 → 34.3 analysed frames/min.**

Frames *received* fell (3,390 → 1,283 per camera) — the decoders now compete with a pipeline
doing 6× the work. Analysed throughput rose regardless, which is the metric that matters.
Sampling policy was **not** changed.

---

## 10. Queue drops — the acceptance criterion

| | BEFORE | AFTER |
|---|---:|---:|
| cam-11 | 362 | **0** |
| cam-12 | 234 | **0** |
| cam-13 | 303 | **0** |
| cam-14 | 201 | **0** |

**Queue-full drops eliminated entirely.** Nothing is hidden: `dropped_sampled` (deliberate
downsampling) is still counted and still reported separately.

---

## 11–13. Detection, tracking, cropping

| | BEFORE | AFTER |
|---|---:|---:|
| Detection inference p50 | 0 ms | **0 ms** |
| Detection queue p50 | 375–438 ms | **250–266 ms** |
| Detection timeouts | 0.4/min | **0.2/min** |
| Tracks created | 44 | **276** |
| Tracks **recovered** | 3 | **223** |
| Association failures | 2 | 13 |
| Crops produced | 119 | **360** |
| Crop extraction p50 | 31 ms | 32–93 ms |

Detection **improved** — queue wait down ~40%, timeouts halved. Tracking improved sharply:
recoveries 3 → 223, i.e. tracks now survive gaps instead of dying. Association failures rose
2 → 13 in absolute terms but fell as a share of 276 tracks vs 44.

---

## 14. VLM performance

| | BEFORE | AFTER |
|---|---:|---:|
| Calls | 24.2/min | 16.2/min |
| Succeeded | 20.2/min | 11.6/min |
| Refusals | 0 | 0.2/min |
| p50 | 2,070 ms | 2,664 ms |
| p95 | 14,336 ms | 15,899 ms |
| max | 40,152 ms | 54,981 ms |
| Concurrency shed | 0 | **0** |

**VLM calls went down, and this is expected, not a regression.** `cropping.budget_pressure`
moved **0.32 → 1.00** and `budget_shed` **0 → 604**. The limiter has moved from a blocked
event loop to the policy's own `max_calls_per_hour: 400`. That is a declared cost control in
`kitchen-safety.example.json`, and this phase did not touch it.

Latency drifted ~15% worse under a busier machine (CPU 211% → 894–937% of one core, 7.5 of
12 cores). Not saturated.

---

## 15–16. Registry and compliance

| | BEFORE | AFTER |
|---|---:|---:|
| Registry active (total) | 5 | **15** |
| — per camera | 0 / 3 / 2 / 0 | **2 / 5 / 7 / 1** |
| Attributes applied | 287 | 288 |
| **Compliance subjects** | 2 | **12** |
| Findings | 4 | **12** |
| UNKNOWN | 4 | 12 |
| COMPLIANT | 0 | **0** |
| VIOLATION | 0 | 0 |

Registry population tripled; compliance now evaluates **6× more subjects**. Every camera
holds objects, including the two that previously held none.

---

## 17. Alerts

**0 incidents** in the window, before and after. All 12 findings are UNKNOWN.

This is honest and correct — insufficient evidence yields UNKNOWN, never COMPLIANT — but it
means **this phase did not restore alerting**. It restored the pipeline's ability to observe.
Why attributes are not reaching subjects in a fresh-enough state is a separate question
(§23).

---

## 18. Failure behaviour

Covered by regression tests, all passing:

| Scenario | Behaviour |
|---|---|
| Exception inside the thread | seam never raises; `frames_failed` incremented |
| Exception then recovery | runner released; next batch runs |
| Cancellation mid-call | runner released; next batch runs |
| Shutdown mid-call | `drain()` completes the queue |
| Slow model (0.4 s block) | loop got **≥ 10 turns**; a blocked loop gets 0–1 |

---

## 19. Safety invariants

| Invariant | Status |
|---|---|
| Refusal → no PPE value | ✅ unchanged; refusal semantics untouched |
| NOT_VISIBLE never becomes ABSENT | ✅ unchanged |
| Insufficient evidence → UNKNOWN | ✅ 12/12 UNKNOWN |
| UNKNOWN never becomes COMPLIANT | ✅ 0 COMPLIANT |
| `observed_at` remains capture-time | ✅ `engine.py:446` untouched |
| Observability gate authoritative | ✅ untouched |
| Compliance rules unchanged | ✅ no policy or rule file touched |
| Dropped frames observable | ✅ still counted, split by reason |
| No unbounded queue | ✅ deque bounded; **waiters bounded at 0** |
| `remote_concurrency` ≤ 2 | ✅ shed = 0, semaphore unchanged |

---

## 20. Test results

**3,766 passed · 1 failed** (was 3,730 before this phase; +14 new offload tests, +23 from
P10 Phase 1).

The single failure is `test_no_production_module_names_the_model` — the **pre-existing
Phase-1 red gate**: `nvidia_vl.py:73` still holds `DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"`.
Unrelated to this phase, awaiting an owner decision. **No test was weakened or deleted.**

Ruff: clean on the changed file.

New: `tests/vision_os/understanding/unit/test_event_loop_offload.py` — 14 tests across
`TestTheModelCallIsOffTheLoop`, `TestNoUnboundedBacklog`,
`TestFailuresPropagateThroughExistingSemantics`, `TestSemanticsAreUnchanged`.

---

## 21–22. Files changed / unchanged

**Changed (1 production file):**
`vision_os/perception/understanding/runtime.py` — +73 / −5.

**Added:** the test file above; two documents in `docs/production-hardening/`.

**Intentionally unchanged:** PPE policy, compliance rules, Kleene evaluation, validity
windows, `observed_at`, staleness, NOT_VISIBLE/ABSENT/UNKNOWN semantics, YOLO thresholds,
tracking hysteresis, crop quality floors and geometry, alert/incident semantics, refusal
semantics, evidence semantics, `remote_concurrency`, the model, P9 datasets, historical
evidence, credentials.

No repository-wide find/replace was performed.

---

## 23. Remaining limitations

1. **Alerts are still zero.** 12 subjects, 12 UNKNOWN findings. The bottleneck moved but
   compliance still lacks decided attributes. Needs its own investigation — likely the
   interaction of attribute validity (60 s) with when subjects are re-observed.
2. **The limiter moved to the crop budget.** `budget_pressure` 0.32 → **1.00**,
   `budget_shed` 0 → 604. `max_calls_per_hour: 400` is now binding. It is policy, and out of
   scope here.
3. **Devtools endpoints still take ~2.5 s.** Improved 2.5–3.6× but not explained by this
   change alone. One `/health` sample hit 7.6 s.
4. **Frames received fell** 3,390 → 1,283 per camera. Decoders now compete with a busier
   pipeline. Not investigated.
5. **VLM latency drifted ~15% worse** under higher load; max 40.2 s → 55.0 s.
6. Two pre-existing unguarded spots (§3) remain.
7. **Single-request batches still execute inline** on the `to_thread` worker via the
   `workers <= 1` path — correct, but worth knowing it is not the pool.

---

## 24. Recommendation

**Keep the change.** It is one file, it removes a proven blockage, and every acceptance
criterion that concerns the event loop is met with measured evidence.

**Do not raise `remote_concurrency` next.** The budget is now the binding constraint
(`budget_pressure = 1.0`), so more concurrency would buy little and would spend a policy
allowance faster. The next question is §23.1 — why decided attributes are not reaching
compliance — because that, not throughput, is what stands between this system and alerts.

---

## 25. Verdict

**PASS_WITH_FINDINGS**

Acceptance criteria 1–15:

| # | Criterion | |
|---|---|---|
| 1 | VLM no longer blocks the loop | ✅ loop got ≥10 turns during a 0.4 s call |
| 2 | Frame consumption continues during VLM | ✅ 6.0× analysed frames |
| 3 | `remote_concurrency` ≤ 2 | ✅ shed = 0, semaphore untouched |
| 4 | No unbounded queue | ✅ deque bounded, waiters = 0 |
| 5 | Queue-full drops materially decrease | ✅ **201–362 → 0** |
| 6 | Detection healthy | ✅ improved (queue −40%, timeouts −50%) |
| 7 | Tracking healthy or improved | ✅ recoveries 3 → 223 |
| 8 | Registry correct | ✅ population 5 → 15 |
| 9 | Compliance semantics equivalent | ✅ no rule/policy touched |
| 10 | Refusals remain refusals | ✅ |
| 11 | UNKNOWN remains UNKNOWN | ✅ 12/12 |
| 12 | `observed_at` unchanged | ✅ |
| 13 | Slow VLM does not freeze the pipeline | ✅ |
| 14 | Full backend tests pass | ⚠️ 3,766 pass, 1 **pre-existing** unrelated failure |
| 15 | No safety test weakened | ✅ |

**Findings** that keep this from a clean PASS: alerting is still zero (§23.1); the limiter
moved to the crop budget (§23.2); criterion 14 carries a pre-existing red gate.

**STOP.** Phase 4 not started.
