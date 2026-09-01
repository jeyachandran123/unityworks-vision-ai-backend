# P10 Phase 2 — Throughput Investigation

**Date:** 2026-09-01
**Window:** 300 s controlled baseline, 4 live kitchen cameras
**Production changes made:** **none**

---

## ⚠ Correction to the brief's premise (§7)

The brief states *"Confirm the active model is `meta/llama-3.2-90b-vision-instruct`"* and
*"Do not switch back to 11B during this investigation."*

**The system was already on the 11B before this phase began, and every number quoted in the
brief is 11B data.** Verified three ways:

```
live bound model : meta/llama-3.2-11b-vision-instruct
.env:64          : VISION_NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
source fallback  : DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"   (overridden)
```

I reverted it on 2026-08-31 ~15:56 to restore the APIs after the 90B wedged startup, and
reported that at the time. §7 therefore cannot be executed as written, and §4's sweep would
measure a different model than the brief intends. **Everything below is the 11B.**

---

## 1. Current architecture

```
RTSP → VisionSession._consume ──await ANALYSIS.run(handler)──┐
                                                             │  ONE thread,
   detection → tracking → registry → cropping ───────────────┤  ONE event loop,
                                                             │  ALL 4 cameras
   crop sink → loop.create_task(on_crops) ───────────────────┤
        → _run_ready → engine.understand_batch()  ← SYNCHRONOUS
                                                             │
   registry ← synthesis ← parse ←──────────────────────────── ┘
        → compliance → incidents → API
```

`ANALYSIS` is a single worker thread running its own asyncio loop. It exists because this
exact failure was already solved once, for the API loop — from its own docstring:

> analysis ON  `/auth/login` 58.5 s · `/health` 31.5 s
> analysis OFF `/auth/login`  4.0 s · `/health`  0.6 s
> *"Sixty-seven seconds to return zero rows is not query cost. It is the event loop never
> getting a turn."*

---

## 2. Baseline measurements — 300 s, concurrency = 2

### Per camera

| Camera | Frames rx | **Analysed** | Dropped | Queue-full | Person det. | Tracks | Crop req | Crops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam-11 | 3,390 | 29 | 3,361 | 362 | **0** | 2 | 9 | 7 |
| cam-12 | 3,375 | 28 | 3,347 | 234 | 19 | 18 | 66 | 60 |
| cam-13 | 3,390 | 29 | 3,361 | 303 | 17 | 21 | 57 | 52 |
| cam-14 | 3,390 | 28 | 3,362 | 201 | **0** | 3 | 0 | 0 |

### Totals

| Metric | Window | Per minute |
|---|---:|---:|
| Frames analysed | 114 | **22.8** |
| Detection timeouts | 2 | 0.4 |
| VLM calls | 121 | 24.2 |
| VLM succeeded | 101 | 20.2 |
| VLM refusals / 429 | **0** | 0.0 |
| **Concurrency shed** | **0** | **0.0** |
| Attributes produced | 299 | 59.8 |
| Attributes applied | 287 | 57.4 |

CPU: **165% → 211%** of one core, of 12 available. Memory 646–656 MB, 131–136 threads.
**The machine is not saturated.**

---

## 3. End-to-end timing breakdown

Measured from the platform's own histograms — not estimated.

| Stage | p50 | p95 | max |
|---|---:|---:|---:|
| Detection **inference** | **0 ms** | 0 ms | 0 ms |
| Detection queue wait | 375–438 ms | 750–1,000 ms | 1,938 ms |
| Crop extraction | 16–31 ms | 78–156 ms | 844 ms |
| Crop trigger latency | 0 ms | 0–15 ms | 16 ms |
| Registry apply | **0 ms** | 0 ms | 16 ms |
| State commit / projection | **0 ms** | 0 ms | 31 ms |
| **VLM inference** | **2,070 ms** | **14,336 ms** | **40,152 ms** |

`understanding.batch_size` p50 **3**, p95 **4**.

**Every stage except the VLM is at or near zero.** YOLO inference does not register on the
histogram at all. The 43 s mean frame-processing time is the VLM and nothing else.

---

## 4. VLM latency (11B, n=887)

| | |
|---|---|
| p50 | **2,070 ms** |
| p95 | **14,336 ms** — 6.9× the median |
| max | **40,152 ms** |
| min | 1,116 ms |
| Success | 101/121 (83%) |
| Refusals / 429 | **0** |

The distribution is the problem, not the median. A 2 s median would be tolerable; a p95 of
14 s inside a batch of 3–4 at 2-way concurrency is not.

Effective service rate: 2 concurrent × 60 ÷ 24.2 calls/min ⇒ **~5 s mean per call**, well
above p50 — the tail dominates throughput.

---

## 5. Remote concurrency behaviour (§3 answered)

| Question | Answer | Evidence |
|---|---|---|
| Global or per camera? | **GLOBAL** | `_semaphore_for` keys on `bound.adapter_id`; one adapter ⇒ all 4 cameras share 2 slots |
| Shed when capped? | Shed, not queued — **but never triggered** | `concurrency_rejected = 0` over the window |
| Why no shedding? | `_batch_workers` = `min(max_concurrency, remote_concurrency, len(batch))` = **2** | The executor is pre-throttled to the semaphore, so `try_acquire` always succeeds |
| Calls in flight | 2 (cap) | `understanding.queue_depth = 0` |
| Calls per person | **2** | head band (0.00,0.45) + hand band (0.15,0.55) — distinct groups |
| Calls per analysed frame | **~1.06** measured (121 calls / 114 frames) | earlier snapshot showed 6.9 under a burst |
| **Does VLM block the frame worker?** | **YES** | see §6 |
| Shared lock? | `async with self._lock` in `on_crops`, plus the loop itself | batches serialise |

---

## 6. Root cause

**A synchronous VLM call runs on the single analysis event loop that also consumes frames
for all four cameras.**

[`runtime.py:264`](../../vision_os/perception/understanding/runtime.py)

```python
async def _run_ready(self) -> None:
    async with self._lock:                                     # held across the batch
        results = self._engine.understand_batch(requests, ...)   # SYNCHRONOUS — no to_thread
```

The chain, each link verified:

1. [`session.py:365`](../../app/vision/session.py) — `await ANALYSIS.run(self._handler(frame))`.
   Every frame, every camera, **one thread**.
2. The crop sink schedules `on_crops` with `loop.create_task(...)` — onto **that same loop**.
3. `on_crops` → `_run_ready` → `understand_batch(...)` — a **blocking** call, not offloaded.
4. While it runs, the loop cannot take another turn. No frame from any camera is consumed.
5. Session queues (capacity 8) fill; **~300 frames/camera/300 s are dropped queue-full**.

The design comment on the sink says the hand-off is *"scheduled rather than awaited"*
because *"awaiting would put a 2-second VLM call on the critical path"*. **The scheduling is
correct; the blocking call inside the scheduled task is what puts it back on the critical
path.** The task yields at `create_task`, then blocks the loop for the whole batch.

This is the same failure the `ANALYSIS` thread was created to fix — *"the event loop never
getting a turn"* — reproduced one layer down, on the analysis loop instead of the API loop.

**Corroborating symptom:** during the window `/devtools/live` took **5,946 ms** and
`/devtools/compliance` **9,938 ms**; two collection calls timed out at 25 s entirely. Read
endpoints are being delayed by the same blockage.

**`remote_concurrency=2` is a contributing factor, not the root cause.** Raising it shortens
each batch (more calls in parallel per batch) and would help proportionally — but the loop
still stalls for the batch's duration. Zero shedding proves the cap is *serialising*, not
*rejecting*.

---

## 7. Detection, tracking, crop performance

**All healthy.** Detection inference does not register on its histogram (p50/p95/max all
0 ms); timeouts fell to **2 in 300 s (0.4/min)** after the atlas stack was stopped, from
21/50 (42%) before. Crop extraction p50 31 ms. Registry apply p50 0 ms.

Nothing in the perception path is slow. It is starved of turns.

---

## 8. Registry and compliance impact

```
registry.active : cam-11 0 | cam-12 3 | cam-13 2 | cam-14 0
compliance      : subjects 2, findings 4, compliant 0, violations 0, unknown 4
incidents today : 0
```

**All four findings are UNKNOWN.** The system is behaving exactly as §6 of the brief
requires — insufficient evidence yields UNKNOWN, never COMPLIANT. No safety property is
being violated; the pipeline is simply not gathering enough evidence to reach a verdict.

**Two of four cameras saw no people at all** in the window (cam-11, cam-14: 0 person
detections). Some of the alert absence is genuine absence of subjects, and that must not be
folded into the throughput finding.

---

## 9. Frame-drop accounting (nothing hidden)

Per camera per 300 s: 3,390 received, **29 analysed**, 3,361 dropped.

| Drop reason | Count | Legitimate? |
|---|---:|---|
| `dropped_sampled` | ~3,050 | **Yes** — deliberate 25 fps → `analysis_fps=1.0` downsampling |
| `dropped_queue_full` | **201–362** | **No** — this is the pathological loss |

Even the sampled target (1 fps = 60/min/camera) is missed by ~10×: measured **5.7
analysed frames/min/camera**.

---

## 10. Concurrency = 4 and = 6 — NOT RUN

**Deliberately not executed.** Three reasons:

1. **§7's premise is false.** The active model is the 11B. A sweep now measures the wrong
   model, and its numbers would not transfer — 11B p50 is 2.1 s, the 90B measured 73 s.
2. **The measurement redirects the hypothesis.** The brief asks whether
   `remote_concurrency=2` is the limiting factor. It is a limiter but not *the* one: with
   zero shedding and 21% CPU utilisation, the constraint is a blocked event loop, not a
   rejected call. Raising the cap treats the symptom.
3. **Each level requires a restart**, and restarts on this deployment have twice produced
   multi-minute outages. Spending three of them on the wrong model, to test a secondary
   lever, is not a good trade while the kitchen is live.

The sweep remains worth running — **after** the model question is settled and against the
model that will actually be qualified.

---

## 11. Safety assessment

| Invariant | Status |
|---|---|
| Insufficient evidence → UNKNOWN, never COMPLIANT | ✅ 4/4 findings UNKNOWN |
| NOT_VISIBLE never becomes ABSENT | ✅ unchanged |
| Refusal never becomes a PPE value | ✅ 0 refusals; semantics untouched |
| Dropped frames visible | ✅ counted and reported, split by reason |
| No unbounded queue | ✅ capacity 8, `drop_oldest` |
| Detection not starved of CPU | ✅ 0 ms inference, 0.4 timeouts/min |

**No safety property is violated.** The system is failing safe: it under-reports because it
under-observes, and it says so through UNKNOWN rather than through silence-as-compliance.

That is also the danger. A safety monitor that is *correctly* UNKNOWN on nearly every
subject is not protecting anyone, and from the dashboard it looks the same as a quiet
kitchen.

---

## 12. Changes made / not made

**Made:** none to production code, configuration, or policy. One report; one throwaway
measurement harness in the scratchpad.

**Not made, deliberately:** `remote_concurrency` unchanged at 2; PPE policy, validity
windows, `observed_at`, staleness rules, YOLO thresholds, tracking hysteresis, crop quality
floors, alert rules, refusal semantics all untouched; the model was not switched.

---

## 13. Remaining freshness concern

Untouched, per §8, and now sharper. The 11B at p95 **14.3 s** already consumes a quarter of
`face_covering`/`hand_covering`'s 60 s validity before the answer exists; its max of
**40.2 s** consumes two thirds. The 90B at 73 s exceeds the whole window. Whatever is
decided about throughput, the freshness decision is separate and still open.

---

## 14. Recommendation

**The evidence-backed fix is not the concurrency cap.** It is to stop the blocking call from
occupying the analysis loop:

```python
results = await asyncio.to_thread(self._engine.understand_batch, requests, crops=crops)
```

One line, in the layer that already owns the seam. It preserves the semaphore, the shed
policy, the batch composition, refusal semantics and every safety rule — the calls still run
exactly 2 at a time, and are still shed rather than queued. What changes is that the
analysis loop keeps taking turns while they run.

**This is a production architecture change and I have not made it.** It needs its own
before/after measurement, and `understand_batch` must be confirmed thread-safe under the
existing semaphore before it is trusted.

Raising `remote_concurrency` to 4 or 6 is worth measuring *after* that, as a second lever.
Doing it first would speed up a batch that is still holding the loop hostage.

---

## 15. Verdict

**BLOCKED**

The phase's stated purpose — *"establish whether the current Vision OS architecture can
safely sustain the 90B's inference latency"* — cannot be answered, because the 90B is not
the active model and cannot become it without a multi-minute outage (4 conformance
inference calls at binding × 73 s).

Delivered in full: §1 baseline, §2 end-to-end timing, §3 concurrency semantics, and a root
cause proven from code and measurement rather than inferred from an aggregate.

**Not the concurrency cap. A synchronous VLM call on the single analysis event loop that
feeds all four cameras.**

### Next action — needs your decision

1. Authorise the `to_thread` change plus a before/after measurement, **or**
2. Authorise the concurrency sweep on the 11B as a bounded interim, **or**
3. Settle the 90B question first — its binding cost (~5 min outage) and 73 s latency both
   need a decision before any 90B measurement is worth running.
