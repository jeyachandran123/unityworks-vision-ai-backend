# Phase 6 Part A — Architecture Audit

**Date:** 2026-08-18
**Finding: the mechanism Phase 6 asks me to build already exists in production. The VLM-usage
figures this phase would optimise against were never production behaviour.**

---

## 1. The headline

Parts B, D and E ask for a VLM call-decision gate, temporal reuse of recent verified observations,
and staleness invalidation on visual change.

**All three are implemented, shipped, and configured.** They live in the P12 `TriggerPolicy`
(`adapters/cropping/triggers.py`), which every crop request already passes through.

### What already decides to *spend* a call

`first_sight` · `attribute_missing` · `attribute_stale` · `appearance_changed` ·
`low_confidence` · `identity_unverified` · `quality_improved` · `periodic_refresh` ·
`explicit_request` · `lifecycle_transition`

### What already decides to *save* one

`no_demand` · `budget_exhausted` · `quality_insufficient` · **`fresh_enough`** ·
`evidence_sufficient` · `deduplicated` · `priority_preempted` · `frame_unavailable`

Mapping the brief onto what exists:

| Phase 6 asks for | already implemented as |
|---|---|
| "reuse a recent verified observation" (Part D/E) | `SkipReason.FRESH_ENOUGH` — *"all demanded attributes are fresh"* |
| "state becomes stale when evidence meaningfully changes" | `TriggerReason.APPEARANCE_CHANGED`, `ATTRIBUTE_STALE` |
| "re-request when quality improves" | `TriggerReason.QUALITY_IMPROVED` |
| "don't call merely because a person exists" (Part B) | `SkipReason.NO_DEMAND` |
| "budget ceiling" | `SkipReason.BUDGET_EXHAUSTED`, `sustainable_freshness()` |
| "observability gate before the call" | `SkipReason.QUALITY_INSUFFICIENT` (Phase 4.1) |

Freshness is configured, not merely available: `head_covering` `validity_ms` **120 000**,
`hand_covering` **60 000**, demand `freshness_ms` **60 000**.

## 2. The measurement gap this exposes

`tools/vision_eval/predict.py` — the harness behind every number in Phases 3 through 5D — states
in its own docstring that it drives the components directly and **deliberately bypasses tracking
and the registry**. It therefore issues one VLM call per evidence group per subject per frame *by
construction*. There is no freshness path for it to take.

**Consequence: the "2.0 calls per person", "5 733 calls per 1 000 frames" and related figures
describe the evaluation harness, not the production pipeline.** They are an upper bound. Real
production usage is unmeasured.

This matters directly for Phase 6's acceptance criteria, which require a before/after VLM
reduction percentage. **Computing a reduction against a baseline that was never production
behaviour would produce a large, meaningless improvement number** — the system would appear to
gain what it already had.

## 3. Attribute dependency table (Part A deliverable)

### `head_covering`

```
final state source : VLM
supporting signals : YOLOv8n person box
                     evidence_region (top 0.00, height 0.45)
                     output_size 448
                     quality gate (min_scale_pixels 130, max_blur 0.5)
                     pose observability — EVALUATED (Phase 4.4), NOT wired into runtime
deterministic      : NONE validated
VLM required       : YES
```

### `hand_covering`

```
final state source : VLM
supporting signals : YOLOv8n person box
                     evidence_region (top 0.15, height 0.55)
                     output_size 224 (deployment default)
                     quality gate (min_scale_pixels 150, max_blur 0.85 — provisional)
deterministic      : NONE validated
VLM required       : YES
```

### Value provenance across the pipeline

| value | produced by |
|---|---|
| person box | YOLOv8n (`adapters/detection/yolo.py`) |
| head observability | pose module — **evaluated, not wired** |
| evidence-region geometry | `SemanticPolicy.evidence_regions` → `PartFocusedCropStrategy` |
| crop resolution | `SemanticPolicy.output_sizes` → `PartFocusedCropStrategy` |
| quality grades | `HeuristicQualityEstimator` |
| gate verdict | `QualityGate.evaluate(grades, attributes)` |
| **call / skip decision** | **P12 `TriggerPolicy`** |
| PPE state | VLM via `UnderstanderPort` → semantic mapping |
| compliance state | `ComplianceEvaluator` (three-valued Kleene) |

## 4. Part C — no deterministic PPE path exists, and none may be invented

**Neither attribute has a validated deterministic signal.** Part C forbids inventing one, and the
programme's own history is the argument for that prohibition: the Camera C investigation nearly
recorded a dozen fabricated violations because clear gloves *looked like* bare skin at reduced
resolution. A "light pixels = hairnet" rule would have made exactly that error permanent.

So the honest dependency answer for both attributes is **VLM required: yes**, and the reduction
available to Phase 6 is entirely in *how often* it is asked — which is the machinery in §1.

## 5. Part I — open-set safety, verified intact

`PRESENT` / `ABSENT` / `NOT_VISIBLE` / `UNKNOWN` remain distinct. The `AttributeState.is_decided`
predicate, the compliance evaluator's refusal check, and the Phase 5 validator's
`decided_state_without_observability` rule all still enforce that missing evidence cannot become a
violation. 2 969 tests green.

## 6. Part J — dark-covering regression fixture

The known failure is preserved and **not tuned for**:

| source | subject | truth | system | status |
|---|---|---|---|---|
| kitchen-01 | f00900 s0 — "dark knitted cap, close to the lens" | PRESENT | ABSENT | fails every run |
| cam5d | w3 — dark cap, two frames | PRESENT | ABSENT | 2 false violations |

Both remain in their datasets with original labels. Nothing was changed to make them pass.

---

## 7. Decision: **CASE C — VLM usage cannot yet be safely reduced, because the reduction already
shipped and has never been measured.**

This is not "no reduction is possible". It is that **Phase 6 cannot honestly claim one from the
current baseline**, and building a second reuse mechanism alongside the existing one would be
duplicating `TriggerPolicy`, which the standing constraints forbid.

### What is missing

A production-path baseline. Specifically:

1. **Run the Phase 5 datasets through the harness session** (`vosvc_harness`), which wires the
   registry, tracking and demand freshness — not through `predict.py`. Record the real
   `TriggerReason` / `SkipReason` distribution.
2. **Then** the questions Phase 6 asks become answerable with real numbers: how many calls does
   `fresh_enough` already save, and is `validity_ms` 120 000 too generous or too tight for CCTV
   where a worker can remove a hairnet between frames?
3. Only after that does a freshness-policy experiment (Part F arms B/C/D) have a meaningful
   control arm.

### What should not happen

- **No new temporal-reuse component.** One exists; a second would compete with it.
- **No deterministic PPE heuristic.** Part C, and the Camera C near-miss.
- **No specialist PPE model** (Part H): 6 violating workers, 2 cameras, 2 restaurants, **2 PRESENT
  observations** and effectively one dark-covering example. **SPECIALIST MODEL NOT YET
  JUSTIFIED.**
- **No prompt change** aimed at the dark cap.

### Honest scope note

This report delivers Part A, the Part C/H/I/J determinations, and the decision. Parts B, E and F
were **not implemented**, because the audit shows their premise needs re-grounding first — building
a call-decision gate on top of the existing one, and measuring it against a harness artifact, would
produce an impressive reduction figure that reflects nothing real. That is precisely the outcome
the brief's closing principle warns against.
