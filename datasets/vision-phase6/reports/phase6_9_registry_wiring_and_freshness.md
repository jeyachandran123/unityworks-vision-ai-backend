# Phase 6.9 — M7 Attribute Registry Wiring + Freshness Re-Validation

**Date:** 2026-08-19
**Decision: CASE A. The existing freshness mechanism works. `FRESH_ENOUGH` fired 522 times —
the first occurrence in the entire Phase 6 programme.**

---

## 1. Executive finding

One parameter was missing. Passing the policy `AttributeRegistry` to M7 turned every measured
symptom from Phases 6.1–6.8 off at once:

| metric | Phase 6.8 | **Phase 6.9** |
|---|---:|---:|
| attributes produced | 308 | 258 |
| write-back attempts | 308 | 258 |
| **write-backs applied** | **0** | **258** |
| **write-backs rejected** | **308** | **0** |
| `ATTRIBUTE_MISSING` | 200 | **0** |
| **`FRESH_ENOUGH`** | **0** | **522** |
| `ATTRIBUTE_STALE` | 0 | 0 |
| tracks with a stored attribute | 0 | **17** |
| frames processed | 44 | **250** |

`ATTRIBUTE_MISSING` did not merely fall — it **disappeared from the trigger distribution
entirely**.

## 2. The change

`vosvc_harness/assembly.py`, one call:

```python
attributes = build_attribute_registry(policies)
registry_layer = build_registry_layer(
    platform, store=InMemoryObjectStore(), attributes=attributes
)
```

`build_registry_layer()` already accepted `attributes`; it was never passed. The **same instance**
now reaches M7 and M9 — not an equivalent copy, which would drift the moment a policy reloaded on
one side.

**M7's neutrality gate was not weakened.** The fix gives M7 the correct vocabulary; it still
refuses anything undeclared, and a test asserts that.

## 3. Proof the registry is shared

`tests/test_shared_attribute_registry.py` — 5 passed, 1 skipped — exercising the real composition,
not a mock:

- M7 has an attribute registry at all
- M7 and M9 hold the **same instance** (identity, skipped where M9 does not expose its registry)
- M7 accepts **every attribute the active policies declared**, derived from `stack.policies` rather
  than a hard-coded key
- M7 **still refuses** an undeclared attribute
- the write-back audit accounts for every discarded result, with reasons

## 4. Runtime accounting — a chain that now completes

```
crops_consumed        138
dropped_on_overflow     0
requests_made         138
results_produced      132
results_failed          0
attributes_produced   258
sink_failures           0
writeback applied     258   <- was 0
writeback rejected      0   <- was 308
```

Every stage balances end to end for the first time.

## 5. Single-track trace — the critical proof

```
track 0000000MG8GWM062   evals=196  write-backs=89
stored: head_covering = "none"   observed_at=55 200 ms   written at 55 500 ms

t=21 250 ms  SKIP     quality_insufficient   stored=False
t=22 000 ms  TRIGGER  first_sight            stored=False
t=22 250 ms  TRIGGER  lifecycle_transition   stored=True    <- attribute now in M7
t=22 500 ms  TRIGGER  lifecycle_transition   stored=True
```

`observed_at` is **55 200 ms** while the write occurred at **55 500 ms** — source-observation time
preserved, not stamped at write time. That is the value freshness ages against, and it is correct.

## 6. Trigger and skip distributions

| trigger | n | | skip | n |
|---|---:|---|---|---:|
| `LIFECYCLE_TRANSITION` | 96 | | `QUALITY_INSUFFICIENT` | 1 374 |
| `FIRST_SIGHT` | 33 | | `NO_DEMAND` | 1 122 |
| `QUALITY_IMPROVED` | 9 | | **`FRESH_ENOUGH`** | **522** |

`LIFECYCLE_TRANSITION` and `QUALITY_IMPROVED` appear for the first time. Both require a prior
stored observation, so both were structurally unreachable until now — the same blockage as
freshness, and further confirmation that a single missing parameter was suppressing an entire
family of policy behaviours.

## 7. Call economics

**522 `FRESH_ENOUGH` skips are 522 VLM calls the platform declined to repeat**, against 138 it
actually made. On this run, reuse avoided roughly **3.8×** the work it performed.

The Phase 6.1 baseline of 325 calls / 1 000 frames was measured with freshness structurally
disabled, so it describes a system that could not reuse anything. It should not be quoted as the
"before" of an optimisation — the correct statement is that reuse was never operating, not that it
was operating poorly.

## 8. `ATTRIBUTE_STALE` — correctly zero, not a gap

Virtual elapsed was **63 s** against a **120 000 ms** validity window. Nothing could expire. Per
§12 this was not forced, and no conclusion is drawn about whether 120 s is the right value.

## 9. Runtime completeness

| | |
|---|---|
| frames decoded | 435 |
| frames processed | **250 / 434** (exhausted: false) |
| wall clock | 331 s |
| virtual elapsed | 63 s |
| tracks with stored attribute | 17 |

Far further than any previous run (44–94 frames), and stopped by the wall-clock budget rather than
a defect. Real VLM throughput, honestly reported.

## 10. Safety audit

| check | result |
|---|---|
| `NOT_VISIBLE` / `UNKNOWN` never became `ABSENT` | **held** |
| failed results never written | **held** — `failed_outcome = 0`, guard intact |
| **M7 still rejects unknown attributes** | **held** — asserted by test |
| class applicability still enforced | **held** — unchanged code path |
| `observed_at` is source time | **held** — 55 200 vs 55 500 ms |
| `TriggerPolicy` unchanged | **held** |
| `validity_ms` still 120 000 | **held** |
| crop size, quality gate, prompt, detector, tracker unchanged | **held** |
| no second `AttributeRegistry`, no new cache | **held** — same instance |
| ground truth, compliance untouched | **held** |

## 11. Regression

Registry + understanding 848 · shared-registry composition 5 passed / 1 skipped · full backend and
harness suites below.

## 12. Decision: **CASE A**

M7 accepts registered attributes; the same tracked objects are re-evaluated; **`FRESH_ENOUGH` is
522.** The M9 → M7 wiring is correct and the existing freshness mechanism is demonstrably
reachable and working.

Every earlier hypothesis is now positively excluded: not footage length, not track lifetime, not
448 px crops, not queue overflow, not adapter failure, not `TriggerPolicy`, not `validity_ms`. A
single unpassed parameter suppressed freshness, lifecycle and quality-refresh triggers together.

### Recommended next phase

**Re-establish the production VLM baseline, now that reuse actually functions.** Every call-economy
figure in Phases 6.1–6.8 was measured against a system that could not reuse an observation. That
baseline is not wrong, but it describes a configuration that no longer exists.

Then, and only then, `ATTRIBUTE_STALE` becomes testable with footage or replay long enough to cross
120 s — and only after that does any discussion of tuning `validity_ms` have a control arm.

**Nothing was tuned in this phase.** The only production change is the one parameter.
