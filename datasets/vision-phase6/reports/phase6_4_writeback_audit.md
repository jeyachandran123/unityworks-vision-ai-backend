# Phase 6.4 — Observation Lifecycle / Write-Back Audit

**Date:** 2026-08-18
**Root cause found: the M9 → M7 attribute write-back seam is implemented but has no production
caller. Understanding results never return to the registry, so the trigger policy can never see
an attribute as satisfied.**

This explains Phase 6.3 exactly, and it is an architectural gap rather than a policy fault.

---

## 1. The question

Phase 6.3 observed the same tracked person re-triggering `ATTRIBUTE_MISSING` every frame, 250 ms
apart, never `FRESH_ENOUGH`. Three candidate causes were listed and none confirmed. This audit
determines which.

## 2. What the trigger policy actually reads

`TriggerPolicy` decides `ATTRIBUTE_MISSING` here (`adapters/cropping/triggers.py:109`):

```python
missing = tuple(
    key for key in sorted(wanted)
    if key not in candidate.attributes
    or candidate.attributes[key].observed_at is None
)
```

`candidate.attributes` is built by the crop engine (`engine.py:820`) from the **registry's**
`VisualObject.attributes`, with the comment:

> *"Read from M7, never cached here. A second copy of attribute freshness would drift from the
> registry's."*

So the policy consults **one** source of truth: the registry object.

## 3. The write path exists

`RegistryEngine.apply_attribute()` (`perception/registry/engine.py:442`) is the documented M9 → M7
seam. It validates the key against the `AttributeRegistry`, checks class applicability, writes via
`partition.apply_attribute()`, and increments `ATTRIBUTES_APPLIED`. It is complete and correct.

## 4. Nothing calls it

Every `apply_attribute` reference in the repository:

| location | kind |
|---|---|
| `registry/engine.py:442` | the implementation itself |
| `registry/partition.py:451` | the internal store write |
| `state/projection.py:187, 249, 286` | **a different function** — `_apply_attributes`, observation → published state |
| `tests/.../test_registry_engine.py` ×5 | tests |
| `tests/.../test_ownership_and_recovery.py`, `test_understanding_architecture.py` | port-surface assertions |

**Production callers: zero.** Neither `app/` nor the harness ever invokes it. The understanding
layer produces a result, the result becomes an observation, the observation reaches synthesis and
the published state — and **it never returns to the registry object the trigger policy reads**.

## 5. Consequences

1. **`FRESH_ENOUGH` is unreachable.** Freshness compares an observation age against a validity
   window. With no stored attribute there is nothing to age, so the branch cannot be entered — for
   *any* attribute, on *any* camera, at *any* validity setting.
2. **`ATTRIBUTE_STALE`, `LOW_CONFIDENCE` and `QUALITY_IMPROVED` are equally unreachable**, since
   all three test a prior observation that never exists.
3. **VLM calls are unbounded per tracked person.** Every frame in which a demand covers a tracked
   object produces a fresh request, limited only by `NO_DEMAND`, `QUALITY_INSUFFICIENT` and budget.
4. **The published state is unaffected.** Observations still flow M9 → M11 → M12 through
   `projection.py`. Compliance results and the demo remain correct. This is a *cost and
   re-computation* defect, not a correctness defect in what the system reports.

## 6. This corrects three earlier statements of mine

| earlier claim | correction |
|---|---|
| Phase 6 audit: "temporal reuse already exists and is configured" | The *mechanism* exists; it is **not reachable**, because its precondition is never established. |
| Phase 6.1: "VLM usage is already well controlled" | Controlled by demand and quality gating **only**. Freshness contributes nothing. |
| Phase 6.2: "the footage is too short to test freshness" | Footage length was never the binding constraint. No clip of any length could have produced `FRESH_ENOUGH`. |

The Phase 6.1 measurement of **325 calls / 1 000 frames stands** — it was measured correctly. Only
the attribution of *why* was wrong.

## 7. Evidence quality

- **Static:** exhaustive repo-wide search for callers; the seam's own docstring in
  `registry/engine.py:14` lists `apply_attribute(object_id, attr) -> void` as part of the port
  surface, and `test_understanding_architecture.py:128` asserts it is present on the interface —
  i.e. the contract is tested, the wiring is not.
- **Runtime:** Phase 6.3 recorded 131 triggers across 7 stable tracks with `ATTRIBUTE_MISSING`
  recurring at exactly the frame interval, and zero `FRESH_ENOUGH`. The static finding predicts
  precisely that trace.

Two independent lines agreeing. I did not modify anything to test this.

## 8. Limitations

- The audit is static plus the existing 6.3 trace. **I did not build a fix and observe
  `FRESH_ENOUGH` appear**, which would be the conclusive demonstration.
- Whether the seam is unwired deliberately (an architectural decision pending) or by omission is
  **not** something I can determine from the code. The docstrings describe it as a real seam, which
  suggests omission, but that is inference.
- One session configuration examined.

---

## 9. Decision

**The next step is a wiring question, not a measurement question — and it is a production change,
so it is yours to authorise.**

The smallest change that would make freshness reachable is connecting the understanding result to
`RegistryEngine.apply_attribute()` at the existing M9 → M7 seam. That is a **production behaviour
change**, explicitly outside every constraint I have been given, so I have not made it.

Recommended order:

1. **Confirm intent.** If the seam is unwired deliberately, the entire freshness/temporal-reuse
   line of work is moot and Phase 6 should be re-scoped. This is worth one question to whoever
   designed M7/M9 before any code is written.
2. **If it is an omission**, wire it, then re-run the Phase 6.3 hand experiment unchanged. The
   prediction is specific and falsifiable: `FRESH_ENOUGH` should appear within one frame of the
   first observation, and hand VLM calls should collapse from ~19 per track to ~1 per 60 s.
3. **Only then** does tuning `validity_ms` have any meaning.

**No production code, policy, threshold or configuration was changed in this phase.** No model was
trained. Ground truth was neither read nor modified.
