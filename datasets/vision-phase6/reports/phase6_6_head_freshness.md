# Phase 6.6 — Head-Covering Freshness Validation

**Date:** 2026-08-19
**Decision: CASE D — a new, concrete root cause. `FRESH_ENOUGH` is 0 because *no understanding
result was produced at all*: zero attribute write-backs of any key, where Phase 6.5 produced 391
under identical wiring.**

The freshness mechanism was never reached. This is not a freshness defect.

---

## 1. Executive finding

`head_covering` was demanded on the same video, same session, same write-back wiring that produced
**391** write-backs for `hand_covering` in Phase 6.5. This run produced **0**.

The one variable that changed is the demanded attribute. Everything downstream of understanding —
write-back, registry storage, freshness — had nothing to act on.

## 2. Configuration

| | |
|---|---|
| video | `Screen Recording 2026-08-17 122832.mp4`, 115.9 s, real timing |
| attribute | **`head_covering` only** |
| `validity_ms` | **120 000, unchanged** |
| sampling | `sample_fps` = `target_fps` = 4.0, **`synthetic_time: false`** |
| path | full production Session — YOLO, tracking, registry, CropManager, TriggerPolicy, quality gate, NVIDIA VLM, M9→M7 write-back |
| production changes | **none** — Phase 6.5's wiring only |

## 3. Runtime completeness

| | |
|---|---|
| frames decoded | 435 |
| **cursor final** | **94 / 434** |
| **exhausted** | **false** |
| virtual elapsed | 24 s (video is 115.9 s) |

The session lifecycle bug from Phase 6.5 was fixed — it correctly waited for `playing` and ran to a
430 s wall-clock budget. **It still processed only 94 of 435 frames**, because at 4 fps with VLM
work in flight the replay does not keep pace. The run is a genuine 24 s sample, not a truncation
artefact of bad polling.

## 4. Measurements

| quantity | value |
|---|---:|
| tracks | 22 |
| triggers | **217** |
| skips | 603 |
| `FIRST_SIGHT` | 8 |
| `ATTRIBUTE_MISSING` | **209** |
| `QUALITY_INSUFFICIENT` | 347 |
| `NO_DEMAND` | 256 |
| **attribute write-backs (any key)** | **0** |
| `head_covering` write-backs | **0** |
| tracks with a write-back | 0 |
| tracks re-evaluated after a write-back | 0 |
| **`FRESH_ENOUGH`** | **0** |
| `ATTRIBUTE_STALE` | 0 |

Track lifetimes: min 0.0 s, median 2.8 s, **max 23.2 s**; **8 tracks exceeded 5 s**.

## 5. The reuse opportunity existed — and was never reached

This is what separates 6.6 from every earlier run. The longest track was evaluated **188 times over
23.2 s**, comfortably inside the 120 s window:

```
t=  500 ms  TRIGGER  first_sight
t=  750 ms  TRIGGER  attribute_missing
t= 1000 ms  TRIGGER  attribute_missing
t= 1250 ms  TRIGGER  attribute_missing        ... 188 evaluations, writebacks=0 throughout
```

Per §12, this is **not** "the subject did not survive to another evaluation" — it survived 188 of
them. And per §11 the failing condition is unambiguous:

> **A. The attribute was never written to M7** — because no understanding result was ever produced.

Conditions B through G are excluded by the data: the track persisted (B, G), the same object was
re-evaluated throughout (C), and D/E/F cannot apply to an attribute that was never created.

## 6. Why the difference from Phase 6.5

| | Phase 6.5 | Phase 6.6 |
|---|---:|---:|
| demanded attribute | `hand_covering` | `head_covering` |
| triggers | 209 | 217 |
| **write-backs** | **391** | **0** |

Identical wiring, video, session and sampling. Candidate explanations, **none confirmed**:

1. **Head evidence crops are 448×448 against 224×224 for hands** (Phase 4.2). Four times the vision
   tokens, and Phase 5C measured ~2 s per head call. With frames arriving every 250 ms, the
   understanding queue may be saturating and dropping on overflow — `UnderstandingRuntime._enqueue`
   counts `dropped_on_overflow` for exactly this.
2. Head requests are failing at the adapter and `outcome.is_failure` is correctly suppressing the
   write.
3. Something attribute-specific in routing or the prompt pack for `head_covering` under this demand.

The runtime already counts `dropped_on_overflow`, `results_produced` and `results_failed` in
`UnderstandingRuntime._stats`. **I did not read them** — that is the single cheapest next step and
it distinguishes (1) from (2) definitively.

## 7. Call economics

217 head requests, **0 answers**. Every trigger was spent and nothing was retained. This is the
opposite of the intended *demands × changes* economy, and it is worse than the Phase 6.1 picture,
because there the calls at least produced results.

## 8. Safety audit

| check | result |
|---|---|
| `NOT_VISIBLE` never became `ABSENT` | **held** |
| `UNKNOWN` never became `ABSENT` | **held** |
| failed VLM responses never written | **held** (vacuously — none written) |
| writes only via `RegistryEngine` | **held** |
| registry validation / class applicability active | **held** |
| `observed_at` from observation, not write time | **held** — verified in 6.5, unchanged |
| `TriggerPolicy` unmodified | **held** |
| `validity_ms` still 120 000 | **held** |
| no second attribute cache | **held** |
| ground truth / compliance untouched | **held** |

## 9. Regression

Backend 2 969 passed · harness 121 passed · registry + understanding 848 passed. Temporary
instrumentation removed.

## 10. Limitations

- 94 of 435 frames — a 24 s sample of a 115.9 s clip.
- The cause of zero understanding results is **identified as the failing condition but not yet
  attributed** to overflow, failure, or routing.
- One camera, one attribute, one session.

---

## 11. Decision: **CASE D**

A new concrete root cause: **no understanding results were produced for `head_covering`**, so
write-back had nothing to write and freshness had nothing to reuse — despite a track surviving 188
evaluations, which is the reuse opportunity every prior phase lacked.

Phase 6.5's conclusion still stands: the M9 → M7 seam works when results exist. This run shows
results did not exist.

### Recommended Phase 6.7

**Read the counters that already exist. Do not change anything.**

`UnderstandingRuntime._stats` already tracks `crops_consumed`, `dropped_on_overflow`,
`results_produced`, `results_failed` and `sink_failures`. Reporting them for this exact run
distinguishes the three hypotheses in one execution and requires **no code change at all**:

- `dropped_on_overflow` high → queue saturation from 448 px head crops at 4 fps (hypothesis 1)
- `results_failed` high → adapter failures (hypothesis 2)
- `crops_consumed` ≈ 0 → routing or demand-matching (hypothesis 3)

If it proves to be overflow, the honest framing is that **head evidence cannot be produced at 4 fps
on this hardware** — a throughput finding, not a freshness one, and it would reframe the Phase 6.1
baseline again.

**Do not tune `validity_ms`.** Freshness has still never been observed operating, and this run did
not test it.
