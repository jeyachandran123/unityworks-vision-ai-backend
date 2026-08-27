# Phase 6.7 — Understanding Runtime Counter Diagnosis

**Date:** 2026-08-19
**Decision: CASE F — a concrete cause not among A–E. Understanding succeeded completely. The flow
stops at the *write-back guard I added in Phase 6.5*, and my own `except: continue` is hiding why.**

---

## 1. Executive finding

217 head trigger opportunities produced zero `UnderstandingResult`s in Phase 6.6 — **that framing
was wrong.** The counters show understanding worked:

```
crops_consumed        204
dropped_on_overflow     0        <- Hypothesis A REJECTED
requests_made         204
results_produced      202        <- results DID exist
results_failed          0        <- Hypothesis B REJECTED
attributes_produced   384        <- attributes DID exist
sink_failures           0        <- Hypothesis D REJECTED
failure_rate          0.0
-------------------------------
M7 write-backs          0        <- the flow stops HERE
```

**202 results carrying 384 attributes were produced and published without a single failure, and
none reached `apply_attribute()`.**

## 2. Configuration (identical to Phase 6.6)

`Screen Recording 2026-08-17 122832.mp4` · `head_covering` only · `validity_ms` 120 000 ·
`sample_fps` = `target_fps` = 4.0 · `synthetic_time: false` · full production Session ·
**no production change of any kind**.

## 3. Replay status

| | |
|---|---|
| frames decoded | 435 |
| frames processed (`frames_consumed`) | **62** |
| cursor final | 62 / 434, **exhausted: false** |
| virtual elapsed | 16 s (video 115.9 s) |

The clip was not fully replayed — real workload pressure, not a script fault. Sufficient for the
diagnosis: 204 crops and 202 results is ample evidence.

## 4. The accounting chain reconciles cleanly

```
triggers                204   (4 first_sight + 200 attribute_missing)
crops delivered to sink 204
crops_consumed          204   -> every crop reached understanding
dropped_on_overflow       0
requests_made           204
results_produced        202   (2 in flight at pause)
results_failed            0
attributes_produced     384   (~1.9 per result)
sink_failures             0
M7 write-backs            0   <- BREAK
```

Every stage balances until the last. There is no missing term and no redefinition needed.

## 5. Hypotheses A–D, all rejected by evidence

| hypothesis | verdict | evidence |
|---|---|---|
| **A — queue overflow** (448 px saturation) | **REJECTED** | `dropped_on_overflow = 0`. Nothing was dropped. |
| **B — adapter/VLM failure** | **REJECTED** | `results_failed = 0`, `failure_rate = 0.0`. |
| **C — routing/consumption** | **REJECTED** | `crops_consumed = 204` = every trigger. |
| **D — sink failure** | **REJECTED** | `sink_failures = 0`; the sink ran without raising. |

The 448 px hypothesis I proposed in Phase 6.6 is **disproven**. Head crops were produced,
understood and answered without a single drop or failure.

## 6. Where it actually stops — and my own defect

The sink is the function I added in Phase 6.5:

```python
for result in results:
    if result.object_id is None or result.outcome.is_failure:
        continue
    for attribute in result.attributes:
        try:
            engine.apply_attribute(result.object_id, attribute)
        except Exception:      # <- swallows the reason
            continue
```

`sink_failures = 0` means this function ran and returned normally. `results_failed = 0` excludes
`outcome.is_failure`. So the flow is being discarded by exactly one of:

1. **`result.object_id is None`** — results published without object identity, so the guard skips
   every one; or
2. **`apply_attribute()` raising** — most plausibly `ObjectNotFoundError` (the object expired
   during the ~2 s VLM round-trip) or `AttributeRejectedError` (registration/class applicability) —
   **and my bare `except: continue` silently swallowing it.**

**I cannot distinguish these from the counters, because my own exception handler destroys the
evidence.** That handler was written to stop a bad attribute breaking understanding; it also
stops anyone finding out why an attribute was refused. That is a defect I introduced in Phase 6.5,
and it is the reason this phase ends in a candidate pair rather than a single answer.

## 7. Why `hand_covering` wrote 391 and `head_covering` writes 0

| | Phase 6.5 (hand) | Phase 6.6 (head) | **Phase 6.7 (head)** |
|---|---:|---:|---:|
| triggers | 209 | 217 | **204** |
| crops_consumed | not read | not read | **204** |
| dropped_on_overflow | not read | not read | **0** |
| results_produced | not read | not read | **202** |
| results_failed | not read | not read | **0** |
| attributes_produced | not read | not read | **384** |
| sink_failures | not read | not read | **0** |
| **M7 write-backs** | **391** | **0** | **0** |
| `FRESH_ENOUGH` | 0 | 0 | **0** |

Counters were not read in 6.5/6.6, so those cells are honestly blank rather than back-filled.

The attribute is the only changed variable, which points at candidate 2 with an
attribute-specific cause — `head_covering` registration or class applicability differing from
`hand_covering` — but **this is inference, not measurement.**

## 8. Single-track timeline

Longest track: **124 evaluations over 15.2 s**, `writebacks = 0` throughout.

```
FIRST_SIGHT -> ATTRIBUTE_MISSING x123
```

Its crops were consumed, its requests made, its results produced and its attributes generated —
and its registry entry never received one. The reuse opportunity was ample; nothing was stored to
reuse.

## 9. Safety audit

| check | result |
|---|---|
| `NOT_VISIBLE` / `UNKNOWN` never became `ABSENT` | **held** |
| ground truth, compliance, `TriggerPolicy`, `validity_ms` untouched | **held** |
| crop size, quality gate, prompt, detector, tracker untouched | **held** |
| queue size, concurrency, caching unchanged | **held** |
| M9 → M7 wiring unchanged from Phase 6.5 | **held** |
| **no fix applied** | **held** |

## 10. Regression

Backend 2 969 · harness 121 · registry/understanding/cropping 1 297 — all passing. Instrumentation
was read-only and removed.

## 11. Remaining uncertainty

**One binary question is unresolved:** is `result.object_id` None, or is `apply_attribute()`
raising? Both produce exactly the observed counters. My exception handler is why.

---

## 12. Recommended Phase 6.8

**Make the discarded case observable, then re-run. Nothing else.**

The smallest possible change, confined to the Phase 6.5 sink I wrote:

- count results skipped for `object_id is None`;
- count and **record the exception type** rather than swallowing it.

That is instrumentation of my own code, not a production behaviour change, and it converts the
remaining binary into a measured fact in one run. The fix follows once the answer is known —
and the two candidates need different fixes, which is precisely why guessing now would be wrong.

**Do not change 448, `validity_ms`, queue size or concurrency.** Overflow and adapter failure are
now positively excluded; changing any of them would be treating a disproven hypothesis.
