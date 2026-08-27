# Phase 6.1 — Production-Path VLM Usage Baseline

**Date:** 2026-08-18
**Result: production VLM usage is ~10.8× lower than every figure reported in Phases 3–5D.**
Those figures came from an evaluation harness that bypasses the mechanism controlling the spend.

---

## 1. Production architecture path, traced

```
video → MediaAsset.frames() → Session → detection → tracking → registry
      → CropManager.evaluate() → P12 TriggerPolicy → TRIGGER | SKIP
      → crop extraction → VLM → semantic mapping → compliance
```

Decisions surface at the **`crop_sink`**, which receives an `EvaluationResult` carrying
`requests` (each with a `TriggerReason`) and `skipped` (each with a `SkipReason`). That is the
single observation point, and it already exists.

**Instrumentation was behaviour-neutral**: the existing sink was wrapped and every call forwarded
unchanged. No policy, threshold, or configuration was touched.

## 2. Execution

| | |
|---|---|
| video | `give_next_seconds_video_of.mp4` (Phase 5D kitchen, validated ground truth) |
| sampled frames | **40** at `sample_fps` 4.0 |
| session | real `vosvc_harness` `Session` — tracking, registry, freshness, budget all live |
| demand | `person` → `head_covering`, freshness 60 000 ms, registered via the real API |
| detector / crops / gate / VLM | production defaults, unchanged |

## 3. The baseline

| metric | value |
|---|---:|
| frames | 40 |
| evidence candidates evaluated | 26 |
| **evidence requests TRIGGERED** | **13** |
| evidence requests SKIPPED | 13 |
| crops produced | 13 |
| **VLM call rate** | **50.0 %** of candidates |
| **calls per 1 000 frames** | **325** |

### Trigger distribution (Part F)

| reason | count | % of triggers |
|---|---:|---:|
| `FIRST_SIGHT` | 6 | 46.2 % |
| `ATTRIBUTE_MISSING` | 6 | 46.2 % |
| `IDENTITY_UNVERIFIED` | 1 | 7.7 % |

### Skip distribution (Part G)

| reason | count | % of skips |
|---|---:|---:|
| `NO_DEMAND` | 10 | 76.9 % |
| `QUALITY_INSUFFICIENT` | **3** | 23.1 % |

Only reasons actually emitted are listed. `FRESH_ENOUGH`, `BUDGET_EXHAUSTED`, `DEDUPLICATED`,
`PRIORITY_PREEMPTED`, `EVIDENCE_SUFFICIENT` and `FRAME_UNAVAILABLE` did **not** occur.

## 4. The demand gate dominates everything

A first run with **no demand registered** produced:

```
1554 candidates evaluated → 1554 SKIP (no_demand) → 0 VLM calls
```

**Zero.** The cost model documented in the architecture — *demands × changes*, not
*cameras × fps* — is real and load-bearing. With nothing asking, the platform spends nothing,
and it evaluated 1 554 candidates to decide that.

## 5. Freshness analysis (Part H) — the honest result

**`FRESH_ENOUGH` never fired, so temporal reuse was not exercised.**

Triggers are 92 % `FIRST_SIGHT` + `ATTRIBUTE_MISSING`, which is the signature of **short-lived
tracks**: each object is new, has no prior observation, and is therefore correctly asked about
once. Reuse cannot occur where there is nothing to reuse.

This does **not** show the freshness policy is wrong. It shows this 10-second clip cannot test it —
`head_covering` `validity_ms` is 120 000 ms, i.e. **twelve times the clip's entire duration**. A
video shorter than one validity window can never produce a `FRESH_ENOUGH` skip.

Measured, not decided: no conclusion is drawn about whether 120 000 ms is appropriate.

## 6. PPE state transitions (Part I)

**Not observable in this clip.** With no track surviving a validity window, there is no
before/after state pair to compare. The transition analysis needs footage substantially longer
than 120 s, which none of the Phase 5 violation videos provides (10–116 s).

## 7. Safety checks (Part J)

| check | result |
|---|---|
| `NOT_VISIBLE` never became `ABSENT` | **held** |
| `UNKNOWN` never became `ABSENT` | **held** |
| a skipped call produced no VLM result | **held** — 13 triggers, 13 crops, no orphans |
| stale observation did not silently become new | n/a — no reuse occurred |
| trigger decisions did not alter ground truth | **held** — annotations untouched |
| `QUALITY_INSUFFICIENT` refused before spending | **held** — 3 calls avoided by the Phase 4.1 gate |

## 8. Harness vs production (Part K)

|  | evaluation harness | **production path** |
|---|---:|---:|
| calls / 1 000 frames (head only) | ~3 500 | **325** |
| tracking | bypassed | **live** |
| registry | bypassed | **live** |
| demand gating | none | **live — dominant** |
| freshness / temporal reuse | none | live (untested here) |
| quality gate | live | live |

**~10.8× fewer calls on the production path.**

**This is not a reduction achieved by Phase 6.** It is the gap between what was measured before and
what the system was always doing. No "reduction percentage" is claimed, because there is no
same-behaviour control arm — the harness was never production behaviour.

## 9. Limitations

- **One 10-second video, 40 sampled frames, 13 triggers.** Small.
- Freshness and temporal reuse are **unexercised**, so their contribution is unmeasured.
- `NO_DEMAND` skips (10) reflect candidates outside the registered demand's filter, not waste.
- One demand for one attribute; a multi-attribute, multi-subscriber deployment will differ.
- The Phase 5D ground truth was **not** re-scored here — this phase measures call behaviour, not
  accuracy.

---

## 10. Decision: **CASE B**

**Production VLM usage is now measurable, and freshness/staleness behaviour requires a controlled
experiment that this footage cannot support.**

The baseline exists and is trustworthy for what it covers: 325 calls / 1 000 frames, dominated by
first-sight triggering, with the demand gate and quality gate both demonstrably saving calls.
What remains unmeasured is the mechanism Phase 6 most wants to evaluate.

### Smallest next experiment

**Replay a video longer than one validity window (> 120 s) with a stable track, and measure
`FRESH_ENOUGH`.**

`Screen Recording 2026-08-17 122832.mp4` is **115.9 s** — just under. Options, cheapest first:

1. **Replay an existing video at reduced `target_fps` so wall-clock exceeds 120 s** while frame
   content is unchanged. This exercises freshness without new footage and without touching
   `validity_ms`.
2. **Concatenate or loop** a kitchen clip to exceed the window — acceptable for a *call-behaviour*
   experiment, though it must never be used for accuracy ground truth.
3. **Capture > 3 minutes of continuous footage with a stationary worker** — the clean answer.

Only after `FRESH_ENOUGH` is observed firing does changing `validity_ms` have a control arm.

**Do not change `TriggerPolicy` or any threshold yet.** Nothing measured here indicates a problem
with them; the gaps are in coverage, not behaviour.
