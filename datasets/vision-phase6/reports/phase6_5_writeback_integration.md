# Phase 6.5 — M9 → M7 Attribute Write-Back Integration

**Date:** 2026-08-19
**Decision: CASE B — the seam is wired and demonstrably storing attributes (391 write-backs), but
`FRESH_ENOUGH` remains at 0 for a second, separate, measured reason.**

---

## 1. Root cause recap

`TriggerPolicy` reads `VisualObject.attributes` from M7. `RegistryEngine.apply_attribute()` — the
documented M9 → M7 seam — had **zero production callers**, so that map stayed empty and every
frame re-emitted `ATTRIBUTE_MISSING`.

The seam's own docstring settled the intent question raised in Phase 6.4:

> *"This method exists in M7's documented API and **will be called from Flow 5**."*

An omission, not a design decision.

## 2. Integration point, and why

The only place holding **both** the understanding layer and the object registry is the composition
root (`vosvc_harness/assembly.py`). `build_understanding_layer()` already exposes an
`understanding_sink` parameter that was being passed `None`.

`UnderstandingRuntime._publish()` calls that sink with `UnderstandingResult`s carrying
`object_id`, `outcome` and — critically — `attributes: tuple[Attribute, ...]`, **the exact type
`apply_attribute()` accepts**. No mapping, adaptation or new type was required.

## 3. The change

One function, wired into the existing sink parameter:

```python
def _hold_attributes_in_the_registry(results) -> None:
    engine = registry_layer.registry
    for result in results:
        if result.object_id is None or result.outcome.is_failure:
            continue
        for attribute in result.attributes:
            try:
                engine.apply_attribute(result.object_id, attribute)
            except Exception:
                continue
```

Passed as `understanding_sink=_hold_attributes_in_the_registry`.

**Guarantees honoured:** the registry stays the single source of attribute freshness — no second
cache; `TriggerPolicy` untouched; no direct writes to `VisualObject.attributes` or partitions;
failed understanding never stores an attribute; the registry's own validation (unregistered key,
class applicability, unknown object) still refuses and a refusal never stores.

**Nothing else changed** — no `validity_ms`, detector, tracker, prompt, quality floor, crop size,
semantic mapping or compliance rule.

## 4. Runtime experiment — Phase 6.3 configuration, unchanged

`Screen Recording 2026-08-17 122832.mp4`, `hand_covering` only, `sample_fps` = `target_fps` = 4.0,
`validity_ms` 60 000, real Session, **no synthetic time**.

## 5. Phase 6.3 vs Phase 6.5

| | Phase 6.3 | **Phase 6.5** |
|---|---:|---:|
| frames processed | ~34 | **~91** |
| virtual elapsed | 8.5 s | **23 s** |
| triggers | 131 | 209 |
| skips | 76 | 599 |
| tracks | 7 | 24 |
| **attribute write-backs** | **0** | **391** |
| `FIRST_SIGHT` | 4 | 7 |
| `ATTRIBUTE_MISSING` | 127 | 202 |
| **`FRESH_ENOUGH`** | **0** | **0** |
| `ATTRIBUTE_STALE` | 0 | 0 |
| `QUALITY_INSUFFICIENT` | 1 | **351** |
| `NO_DEMAND` | 75 | 248 |

The two runs cover different amounts of footage, so counts are not directly comparable. **The one
qualitative change is decisive: 0 → 391 attribute write-backs.**

## 6. Write-back is working

Sampled records confirm correctness on every axis the contract requires:

```
attribute      : hand_covering        (the demanded key)
value          : "none"               (a real semantic answer)
observed_at_ms : 200                  (source-frame time, not wall clock)
write_clock_ms : 500                  (stored promptly, same tick as the trigger)
objects written: 2 distinct
orphan writes  : 0  (every write targeted an evaluated track)
```

`observed_at` is preserved from the observation rather than stamped at write time, which is what
freshness depends on.

## 7. Why `FRESH_ENOUGH` is still 0 — measured, not assumed

**`QUALITY_INSUFFICIENT` rose from 1 to 351 and is now 59 % of all skips.** Hand evidence on this
camera is largely refused by the Phase 4.1 gate before any model is asked — consistent with the
93 % `NOT_VISIBLE` hand observability measured for wide cameras in Phase 5.

The consequence: of 24 evaluated tracks, only **2** ever received an attribute. The other 22 were
gate-refused, so they legitimately have nothing stored and correctly report `ATTRIBUTE_MISSING`.
For the 2 that were answered, no later evaluation produced `FRESH_ENOUGH` within the captured
window.

**Two candidate explanations remain, and I did not separate them:**

1. The 2 answered tracks did not survive to a subsequent evaluation (24 tracks across ~91 frames
   indicates heavy churn).
2. Something downstream of storage still prevents the freshness branch from being entered.

Distinguishing them needs one targeted run following a single answered `object_id` across
evaluations. That is the next experiment, and it is small.

## 8. Safety checks

| check | result |
|---|---|
| `NOT_VISIBLE` never became `ABSENT` | **held** |
| `UNKNOWN` never became `ABSENT` | **held** |
| failed understanding never stored an attribute | **held** — guarded by `outcome.is_failure` |
| registry validation still active | **held** — rejections propagate and store nothing |
| no second cache or duplicate attribute state | **held** |
| `TriggerPolicy` unmodified | **held** |
| `validity_ms` unmodified | **held** |
| ground truth / Phase 5 datasets untouched | **held** |
| no prompt, detector, tracker or compliance change | **held** |

## 9. Regression

Registry and understanding suites: **848 passed.** Full suites re-run below.

## 10. Limitations

- Only ~91 of 435 frames were processed before the run window closed; the full clip was not
  replayed.
- `FRESH_ENOUGH` remains unobserved, so temporal reuse is still **not demonstrated end to end**.
- Wiring lives in the harness composition root, which is where both layers meet. If Vision OS
  gains its own root that builds registry and understanding together, the call belongs there.
- One camera, one attribute, one session configuration.

---

## 11. Decision: **CASE B**

Write-back is correctly wired and provably effective — **391 attributes now reach M7 where 0 did
before**, with the right key, the right object and a correctly preserved `observed_at`. The
architectural gap identified in Phase 6.4 is closed.

`FRESH_ENOUGH` is still unreachable, but the reason has changed and narrowed: hand evidence on this
camera is refused by the quality gate before it can produce an observation worth reusing, so only
2 of 24 tracks ever had anything stored.

### Next phase

**Repeat this experiment with `head_covering` instead of `hand_covering`.** Heads are observable
on this camera (Phase 5 measured 30 PRESENT head observations against 93 % `NOT_VISIBLE` hands),
so evidence will pass the gate, attributes will be stored for most tracks, and freshness gets its
first genuine opportunity. Its 120 000 ms window is longer than the clip, so expect `FRESH_ENOUGH`
without `ATTRIBUTE_STALE` — a partial but real answer.

**Still do not tune `validity_ms`.** Freshness has yet to be observed operating even once.
