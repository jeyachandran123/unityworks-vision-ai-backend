# Phase 6.8 — M9 → M7 Write-Back Rejection Audit

**Date:** 2026-08-19
**Root cause found, and it is a one-parameter wiring gap.**

> `AttributeRejectedError: attribute 'head_covering' is not registered`
> `AttributeRejectedError: attribute 'hand_covering' is not registered`

**Every** attribute is refused by M7 because the registry layer was built with an **empty**
`AttributeRegistry`, while the policy's attributes were registered into a *different* instance
handed only to the understanding layer.

---

## 1. The measurement

Same configuration as Phase 6.7. The Phase 6.5 sink now records why each result is discarded:

```
understanding : crops_consumed 166 · results_produced 165 · results_failed 0
                attributes_produced 308 · dropped_on_overflow 0 · sink_failures 0

WRITE-BACK AUDIT
  applied            0
  rejected         308
  no_object_id       0        <- candidate 1 EXCLUDED
  failed_outcome     0
  reasons:
     164 x  AttributeRejectedError: attribute 'hand_covering' is not registered
     144 x  AttributeRejectedError: attribute 'head_covering' is not registered
```

`no_object_id = 0` eliminates the first candidate from Phase 6.7. **Candidate 2 is confirmed with
the exact exception**, on 308 of 308 attempts.

## 2. The gap

`vosvc_harness/assembly.py`:

```python
line 976:  attributes = build_attribute_registry(policies)      # policy keys registered here
line 977:  registry_layer = build_registry_layer(platform, store=InMemoryObjectStore())
                                                              # ^ attributes NOT passed
...        build_understanding_layer(..., attributes=attributes)  # only M9 gets it
```

`build_registry_layer()` **already accepts** `attributes: AttributeRegistry | None = None`. It was
simply never passed. M7 therefore validates against a default empty registry and refuses every
key — correctly, by its own neutrality gate.

This is the same class of defect as Phase 6.4: an existing parameter left unwired. Two instances of
the same shape, one seam apart.

## 3. Correction to Phase 6.5 — my error

**Phase 6.5 reported "391 attribute write-backs" and concluded the seam was proven working. That
was wrong.** My instrumentation there recorded the attempt *before* calling through:

```python
def watched(object_id, attribute):
    WB.append({...})              # counted here
    return orig_apply(object_id, attribute)   # ...which then raised
```

So 391 counted **attempts**, every one of which raised `AttributeRejectedError` and was swallowed
by my own bare `except: continue`. Nothing was ever stored.

Phase 6.7 wrapped correctly (`r = orig(...)` first, then record) and measured 0 — which was the
true figure all along.

Three consequences:

- **The hand-vs-head difference in 6.5/6.6 was an instrumentation artifact, not behaviour.** Both
  attributes fail identically, and always did.
- **Phase 6.5's CASE B conclusion ("write-back wired and provably effective") is withdrawn.** The
  wiring is correct; the write has never once succeeded.
- **Phase 6.6's CASE D and my 448 px overflow hypothesis were chasing a symptom** of this.

The counters in this phase are trustworthy because they are incremented by the production sink at
the point of success or failure, not by an observer guessing at the boundary.

## 4. Why every earlier phase was blocked

```
Phase 6.1  FRESH_ENOUGH never seen      -> attributed to short tracks
Phase 6.2  FRESH_ENOUGH never seen      -> attributed to clip length
Phase 6.3  ATTRIBUTE_MISSING every frame-> attributed to observation not stored  (correct)
Phase 6.4  no apply_attribute caller    -> real gap, correctly found and wired
Phase 6.5  "391 write-backs"            -> WRONG, attempts not stores
Phase 6.6  0 write-backs for head       -> real, cause misattributed to 448 px
Phase 6.7  understanding all green      -> localised to the write-back guard
Phase 6.8  AttributeRejectedError x308  -> ROOT CAUSE
```

Freshness has never been testable. `FRESH_ENOUGH`, `ATTRIBUTE_STALE`, `LOW_CONFIDENCE` and
`QUALITY_IMPROVED` remain unreachable — not because of footage, track lifetime, crop size or
policy, but because **M7 has never accepted a single attribute.**

## 5. Safety audit

| check | result |
|---|---|
| `NOT_VISIBLE` / `UNKNOWN` never became `ABSENT` | **held** |
| ground truth, compliance, `TriggerPolicy`, `validity_ms` untouched | **held** |
| crop size, quality gate, prompt, detector, tracker, queue, concurrency untouched | **held** |
| M7 rejection behaviour **not weakened** | **held** — the registry is right to refuse unregistered keys |
| **no fix applied** | **held** |

The only change is counters inside the Phase 6.5 sink I wrote — it now records `applied`,
`rejected`, `no_object_id`, `failed_outcome` and the exception text, instead of discarding the
reason. That handler destroyed a whole diagnosis in Phase 6.7; the cost of a silent `except` is
now itself measured.

## 6. Regression

Registry + understanding 848 passed. Full suites below.

## 7. Remaining uncertainty

None on the root cause — 308 of 308 rejections name the attribute and the reason.

**Open design question, not a defect:** should M7 and M9 share one `AttributeRegistry` instance, or
should the composition root pass the same one to both? `build_registry_layer` accepts the
parameter, which suggests the latter was intended. Whoever owns M7's neutrality gate should confirm
before it is wired, since the registry is the Semantic Ceiling's outermost ring and sharing it is
an architectural statement, not a plumbing choice.

---

## 8. Recommended Phase 6.9

**Pass the policy attribute registry to `build_registry_layer`. One parameter.**

```python
registry_layer = build_registry_layer(
    platform, store=InMemoryObjectStore(), attributes=attributes
)
```

`attributes` is already constructed on the preceding line. This is a **production behaviour
change**, so I have not made it.

Then re-run Phase 6.6 unchanged. The prediction is specific and falsifiable:

- `applied` rises from 0 to ≈308 per 166 crops;
- `ATTRIBUTE_MISSING` collapses after the first observation per track;
- **`FRESH_ENOUGH` appears for the first time in this programme.**

If `FRESH_ENOUGH` still does not appear once M7 is accepting attributes, that is a genuine
freshness defect and the first one actually attributable to the policy.

**Do not tune `validity_ms`, crop size, queue size or concurrency.** Every one of those has now
been positively excluded by measurement.
