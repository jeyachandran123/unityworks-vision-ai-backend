# Vision OS — Perception Architecture Repair

Repair of the five proven defects from
[`VISION_OS_PERCEPTION_FORENSIC_AUDIT.md`](VISION_OS_PERCEPTION_FORENSIC_AUDIT.md),
plus the minimum scoped fix for the observation-log gate.

| | |
|---|---|
| Repository | `unityworks-vision-ai-backend` |
| Branch | `feat/unityworks-vision-os-prod-hardining` |
| HEAD | `9bfbe8f` |
| Date | 2026-09-03 |
| Tests | **3,928 → 3,982 passing** (+54), 1 pre-existing failure unchanged |
| Production code changed | 11 files, +377 −43 |

---

## 1. Executive summary

All five defects are repaired, and the observation-log gate is repaired with a
minimum scoped change. **No threshold was tuned, no incident suppressed, no model
replaced, no detector changed, and no test weakened.**

| # | Defect | Repair | Verified by |
|---|---|---|---|
| 1 | Policy `lifecycle`/`min_confidence` declared, never enforced | `SubjectFilter` gained the two predicates; `matching()` consults them; the app now registers the **policy's own** demand | 12 tests |
| 2 | Production ran the fallback tracker by construction | `tracker_id` moved to settings, default `tracker.sort` | 10 tests |
| 3 | Fragmentation from motion-free association | Motion model + optimal assignment (same class, config change) | e2e scenarios |
| 4 | Transition ids destroyed, then guessed from states | `TrackingOutcome` carries the real `TrackUpdate`; the bridge forwards it | 7 tests |
| 5 | Re-entry binding structurally unreachable | Objects whose track ended are released **before** the absorb loop | 8 tests |
| — | Conformance kit poisoned the durable log | `for_conformance()` twin; kit never touches production | 7 tests |

The headline result, measured end to end on the real tracker and real registry:

```
scenario                          tracker.iou (before)   tracker.sort (after)
short occlusion, 2 blind frames   2 logical objects      1 logical object
                                  0 recoveries           1 recovery
```

**One person, one identity, across an occlusion that previously created a second
person — and therefore a second incident.**

### What this does not fix

Two things, stated up front because they bound what you should expect:

1. **Frame rate is a real and separate constraint.** Measured below: a briskly
   walking person fragments at 1 and 2 fps on *every* tracker in the repository,
   and holds one identity from 3 fps upward. The four kitchen cameras currently
   run `analysis_fps = 2.0` — on the wrong side of that cliff.
2. **The observation-log repair has not yet been proven on the live deployment.**
   It is proven by test and by a five-boot experiment. The running process still
   needs a restart before alerts can flow.

---

## 2. Baseline (Stage 1, recorded before any change)

```
branch : feat/unityworks-vision-os-prod-hardining
commit : 9bfbe8f
tests  : 3,928 passed, 1 failed
         (tests/vision_os/understanding/test_ninety_b_configuration.py —
          nvidia_vl.py:73 hard-codes the 90B model name; pre-existing,
          untouched per rule 20)

tracker      : app/vision/runtime.py:656  "tracker_id": "tracker.iou"  (hard-coded)
observation  : data/observations/ holds 7 kit-*.jsonl artefacts
               → durable log refuses binding on boot 2+
identity     : 1,990 incidents / 1,990 distinct object_ids  = ratio 1.0000
               created=False returned 0 times, ever
pipeline stop: observation publication (synthesis never binds)
```

---

## 3. Actual runtime pipeline map

Traced from the composition root, with the information loss marked.

```
RTSP → VisionSession._consume → ANALYSIS.run          app/vision/analysis_loop.py:99
   ↓
DetectionRuntime (yolov8n, person, conf 0.4)
   ↓ DetectionOutcome
TrackingRuntime.on_detected  (per-camera lock)        perception/tracking/runtime.py:108
   ↓
TrackingEngine.track                                  perception/tracking/engine.py:138
   ├─ GeometricTracker.update → TrackUpdate           adapters/tracking/geometric.py:575
   │    new · terminated · coasting · recovered ·
   │    associations · refused · unmatched_detections     ← complete, always was
   ↓
TrackingOutcome                                       engine.py:182
   ✔ REPAIRED — now carries `update` whole; counts kept for existing readers
   ↓
TrackingToRegistryBridge                              app/vision/bridges.py:152
   ✔ REPAIRED — forwards the update; no longer re-derives ids from states
   ↓
ObjectRegistry._ingest                                perception/registry/engine.py:668
   ✔ REPAIRED — terminated tracks release their objects BEFORE the absorb loop
   ↓ RegistryUpdate.present
CroppingRuntime → CropTriggerEngine.evaluate
   ├─ DefaultTriggerPolicy._wanted_for                adapters/cropping/triggers.py:222
   │    ✔ REPAIRED — supplies lifecycle + identity confidence
   └─ DemandRegistry.required_attributes/matching     perception/cropping/demands.py:347
        ✔ REPAIRED — consults lifecycle and min_confidence
   ↓
Understanding (NVIDIA VLM) → AttributeObservation
   ↓
Synthesis (M7)                                        vision_os/synthesis_bootstrap.py
   ✔ REPAIRED — conformance gated on a disposable twin
   ↓
Exposure (M14) → ComplianceDriver (5 s) → IncidentService.open
                 dedup on (org, camera_key, object_id, rule_id)   ← unchanged
```

### Boundary contracts

| Boundary | Identity key | Lifecycle carried | Loss before | Loss now |
|---|---|---|---|---|
| Tracker → Engine | `TrackId` | full `TrackUpdate` | none | none |
| Engine → Bridge | — | **4 integers** | **all ids** | none |
| Bridge → Registry | `TrackId` | reconstructed from states | **wrong `new`, empty `terminated`/`recovered`** | none |
| Registry → Cropping | `ObjectId` | `lifecycle` on the object | none | none |
| Cropping → Demand | `ObjectId` | **class only** | **lifecycle, confidence** | none |
| Compliance → Incident | `ObjectId` | — | none | none |

---

## 4. Root cause → repair

### Defect 1 — the policy safety gate

**Symptom.** A one-frame false person reached the model and could become a `high`
severity incident.

**Mechanism.** `kitchen-safety.example.json` declares
`lifecycle: ["active","occluded"]` and `min_confidence: 0.4`. `SubjectFilter`
carried both fields and had exactly one method, `matches_class`. Nothing in the
repository read either field. Worse, `app/vision/demands.py` built its own
aggregate demand carrying `class_ids` only — so the parsed values never even
reached the registry that would have enforced them. `VisualObject.is_present`
deliberately admits `PROVISIONAL`, so nothing else stopped it either.

**Repair, two halves.**

* `SubjectFilter.matches_lifecycle` / `matches_confidence`, consulted by
  `DemandRegistry.matching` and `required_attributes`; the trigger adapter passes
  the candidate's `lifecycle` and `identity_confidence`, both of which
  `TriggerCandidate` already carried.
* `register_policy_demands` now registers **one demand per policy, built by the
  policy itself** via `SemanticPolicy.build_demand`. The function was always
  named that and always documented "one demand per policy"; it now does it. This
  also restores the trigger hints, priority class and per-demand budget that the
  hand-built demand had been silently dropping.

**A bug I introduced and caught.** My first version defaulted `lifecycle: str = ""`
and then tested `"" in ("active","occluded")` — so any caller using the old
three-argument form would have matched **nothing**, silently stopping all
analysis. `test_an_unsupplied_state_is_not_read_as_a_state_name` exists because
of it. "The policy did not narrow" and "the caller did not say" are different
questions and only one of them may exclude.

### Defect 2 — the fallback tracker in production

**Mechanism.** `app/vision/runtime.py:656` hard-coded `"tracker.iou"` — the
platform's documented *universal fallback*: no motion model, greedy single-stage
association, `handles_occlusion="none"`. Nothing had fallen back;
`TrackingManager.is_fallback` correctly reported `False`, so every health check
agreed the tracker was fine.

**Repair.** `vision_tracker_id` in settings, default `tracker.sort`.

I inspected what the repository already provides before changing anything, as
instructed:

| Adapter | Motion model | Association | Occlusion | Class |
|---|---|---|---|---|
| `tracker.iou` | none | greedy | none | `GeometricTracker` |
| `tracker.sort` | **linear** | **optimal** | short | `GeometricTracker` |
| `tracker.bytetrack` | linear | two-stage optimal | short | `GeometricTracker` |

**All three are the same class with a different `GeometricConfig`.** The intended
production tracker already existed and was merely disconnected, so this is a
reconnection, not a new tracking framework. No new dependency.

**Why `sort` and not `bytetrack`.** ByteTrack's advantage is a second association
pass over *low-confidence* detections. The detector runs at conf 0.4 and emits
only `person`, so there is little weak-detection tail for it to exploit; its
benefit is unproven here. `sort` addresses the mechanism the audit actually
proved. `bytetrack` remains one settings value away.

**Verified no hidden regression:** `tracker_factory` passes the config-derived
`LifecyclePolicy` to every tracker, so I checked that `TrackingSection` and
`LifecyclePolicy` defaults are identical (`min_hits_to_confirm` 3,
`max_coast_frames` 5, `max_lost_frames` 15). They are — the switch moves no track
memory bound. Pinned by test.

### Defect 3 — fragmentation

Addressed by Defect 2's motion model, and measured in §6. **No threshold was
touched**: `iou_threshold`, `max_association_cost`, `ambiguity_margin`,
`min_hits_to_confirm` and every lifecycle bound are exactly as they were.

### Defect 4 — transition data destroyed then guessed

**Mechanism.** `TrackingEngine.track` collapsed a complete `TrackUpdate` to four
integers. The bridge then rebuilt one from track **states**:

```python
new=ids_in(TrackState.TENTATIVE)      # a state, not an event
```

`TENTATIVE` persists for `min_hits_to_confirm` frames, so one track reported as
"new" on three consecutive frames, while a track created-and-confirmed in one
frame never reported as new at all. `terminated` and `recovered` had no
counterpart among the counts and took their empty defaults **on every frame,
forever** — which is precisely the information the registry needed for Defect 5.

**Repair.** `TrackingOutcome` gained `update: TrackUpdate | None`, populated with
the tracker's own object. The bridge forwards it. The four count fields are
**kept**, so no existing reader changes. The state-based reconstruction survives
only for an outcome with no update — a failed frame — so degradation is
unchanged.

### Defect 5 — re-entry structurally unreachable

**Mechanism.** `bind_reentry` considers only `OCCLUDED`/`DORMANT`, unbound
records. A fragmenting track dies and its replacement is born in the *same*
frame, and the absorb loop ran before the ageing loop — so the predecessor was
still `ACTIVE` and still bound, failing both guards. `no_candidates`, every time.

**Repair — the ordering, documented.** The order inside `_ingest` is now:

```
1. advance camera time, resolve epoch
2. RELEASE  — objects bound to tracks in `update.terminated`:      ← NEW
              close the binding, apply the lifecycle machine's
              ordinary `on_unmeasured` edge (ACTIVE → OCCLUDED)
3. ABSORB   — for each track in `update.active`:
              bind_continuing → bind_reentry → mint
4. AGE      — objects untouched this frame: close bindings, age
5. EXPIRE / persist on their own cadences
```

Step 2 is derived from the architecture, not assumed: `on_unmeasured` on an
`ACTIVE` object already yields `OCCLUDED` via the machine's own
`MEASUREMENT_LOST` edge, and it acts on the tracker's **declared** `terminated`
list rather than inferring death from a track's absence — which is only possible
because Defect 4 was repaired first.

**This widens no threshold.** `max_reentry_distance` (0.25), `max_reentry_gap`
(30 s), `class_must_match`, `min_binding_confidence` (0.3) and the
`ambiguity_margin` (0.15) refusal are untouched. The change puts the predecessor
into the *candidate set*; the existing scoring still decides, and two plausible
predecessors still refuse to merge and mint a new object.

### The observation-log gate — minimum scoped repair

**Mechanism.** `_gate` ran the kit against the **live** adapter. The kit writes
real records; `FileObservationLog` has no `reset()` (the one at `stores.py:342`
belongs to another class), so seven `kit-*` partitions stayed in the deployment's
log. On boot 2 the kit ran again over its own leftovers and failed L2, L3 and L7.
Synthesis never bound, exposure was never built, compliance read zero subjects.

**Repair.** `FileObservationLog.for_conformance()` returns a twin rooted in a
temp directory plus a `dispose`. `_gate` uses it when an adapter offers one.

**Why a twin, not a cleanup.** Anything that deletes `kit-*` partitions from a
live log is one bad glob away from deleting a camera's observations — the system
of record — to tidy a test fixture. A twin cannot make that mistake.

Two small additions only. Nothing else from the reverted patch set was restored:
no `can_raise_alerts`, no degraded-reason plumbing, no budget change.

---

## 5. Attribute semantics — preserved, not changed

Per instruction, nothing here was altered.

* **Four states intact.** No `not_visible` was converted to compliant or
  non-compliant anywhere. `hand_covering` continues to answer `not_visible` →
  `unresolved` / `not_observable` when hands are inside a pot or behind a body,
  which is the designed behaviour and the honest one.
* **`face_covering` severity unchanged.** Still `informational`; still raises no
  incident.

**On observability, investigated as asked:** informational findings *are*
recorded and *are* exposed. `CompliancePass.record()` runs for every finding at
`compliance_driver.py:261`, before and independently of the `RAISES_INCIDENTS`
filter at line 478, and `to_wire()` publishes `by_rule` — so
`kitchen.person.face_covering.v1` appears in `/api/v1/devtools/compliance` with
its compliant / violation / unknown counts.

The gap is that the **product surface** shows incidents, and an informational
rule never becomes one — so a correctly-working evaluation reads to an operator
as a feature that was never built. Surfacing informational findings is a product
decision and touches the frontend, which this phase may not do. **Escalated, not
changed** (§9).

---

## 6. Before / after identity metrics

Real `GeometricTracker` driving a real `ObjectRegistry`. Only the clock and the
metrics sink are stubs. Person box 0.12 of frame width.

### Scenario matrix, 1 fps

| Scenario | Tracker | Track creations | Recoveries | **Logical objects** | Lifecycles |
|---|---|---|---|---|---|
| A normal walk (0.075 w/s, 10 f) | iou | 1 | 0 | **1** | active |
| A normal walk | **sort** | 1 | 0 | **1** | active |
| **B 2-frame occlusion** | iou | 2 | 0 | **2** ✗ | active, occluded |
| **B 2-frame occlusion** | **sort** | **1** | **1** | **1** ✓ | active |
| C two people crossing | iou | 2 | 0 | **2** ✓ | active |
| C two people crossing | **sort** | 2 | 0 | **2** ✓ | active |
| D empty scene | both | 0 | 0 | **0** ✓ | — |
| F one-frame false positive | both | 1 | 0 | 1 | **provisional** |

Scenario **B** is the repair. Scenario **C** is the guard against having "fixed"
it by merging everybody. Scenario **F** shows the false positive stopping at
`provisional` — never `active`, which is the only lifecycle
`kitchen-safety`'s scope admits to the model, so Defect 1's gate is what keeps it
out of the VLM.

### Frame-rate sweep — the honest boundary

Same physical motion (0.20 frame-widths **per second**), sampled at different
rates. Only the frame interval changes.

| fps | Displacement/frame | vs box width | iou objects | sort objects |
|---|---|---|---|---|
| 1 | 0.200 | 1.67× | 4 | 4 |
| 2 | 0.100 | 0.83× | 8 | 8 |
| **3** | 0.067 | 0.56× | **1** | **1** |
| 4 | 0.050 | 0.42× | 1 | 1 |
| 5 | 0.040 | 0.33× | 1 | 1 |

**The continuity cliff sits between 2 and 3 fps**, where per-frame displacement
falls below roughly half the box width.

A motion model predicts from *observed* velocity, so it needs two measurements
before it can help. When the very first inter-frame displacement already exceeds
the box width, no track ever gets a second hit, no velocity is ever estimated,
and `sort` and `iou` behave identically. **This is a sampling limit, not an
architecture defect,** and the correct fix is frame rate — not looser
association, which would start merging different people.

This refines the forensic audit, which called frame rate "a partial mitigation".
The measurement is more precise: **fps is decisive for fast subjects and
irrelevant for occlusion.** The architecture repairs fix the second; only fps
fixes the first.

> **The four kitchen cameras currently run `analysis_fps = 2.0`** — on the wrong
> side of the cliff for anyone moving briskly. Raising them to 3–4 is the single
> highest-value operational change available, and it is **cheap**: model spend is
> governed by demand freshness (60 s) and the call budget, not by frame rate.
> Better continuity *reduces* distinct objects, which *reduces* model calls.
> Detection CPU rises; VLM spend should fall.

### Projected effect on the production symptom

The `1990 / 1990` ratio was produced by every fragment minting a fresh
`ObjectId`. With re-entry reachable and occlusion survivable, repeated
observations of one unresolved condition now reach `IncidentService.open` under a
**stable** `object_id`, so the dedup key matches and `created=False` becomes
reachable for the first time. The ratio cannot be re-measured until the
deployment restarts and accumulates new data (§9).

---

## 7. Performance

| Concern | Assessment |
|---|---|
| Association complexity | `OptimalAssociator` is O(n³) worst case vs greedy O(k log k), on *gated* candidates only — a handful per track at kitchen densities. Unchanged gating. |
| Registry release step | One pass over a partition's records, only when `update.terminated` is non-empty. O(objects), no new allocation, no new lookup structure. |
| VLM calls | Unchanged per object, and **fewer in total**: fragmentation was inflating the object count, and each object costs its own crops. |
| Identity store growth | Bounded as before — `max_tracks_per_camera` 256, `max_age_frames` 36,000, registry expiry sweep untouched. No new cache was introduced. |
| Retention window | Unchanged: `max_reentry_gap` 30 s, occlusion and dormancy horizons as configured. |
| Conformance twin | One temp directory per gated adapter per boot, removed in a `finally`. |

No unbounded structure was added anywhere.

---

## 8. Files changed

### Production (11 files, +377 −43)

| File | Why |
|---|---|
| `vision_os/core/model/demand.py` | `matches_lifecycle`, `matches_confidence` |
| `vision_os/perception/cropping/demands.py` | `matching`/`required_attributes` consult them |
| `vision_os/adapters/cropping/triggers.py` | supply lifecycle + confidence; `TypeError` fallback for older resolvers |
| `app/vision/demands.py` | register the policy's own demand, one per policy |
| `app/configuration/settings.py` | `vision_tracker_id` |
| `app/vision/runtime.py` | tracker from settings, not a literal |
| `vision_os/perception/tracking/engine.py` | `TrackingOutcome.update` |
| `app/vision/bridges.py` | forward the real update |
| `vision_os/perception/registry/engine.py` | release terminated tracks before absorption |
| `vision_os/adapters/synthesis/stores.py` | `FileObservationLog.for_conformance()` |
| `vision_os/synthesis_bootstrap.py` | `_gate` uses the twin; `_conformance_twin` |

### Tests (6 files, +54)

`tests/vision_os/cropping/test_policy_enforcement.py` (12) ·
`tests/app/test_tracker_selection.py` (10) ·
`tests/app/test_transition_propagation.py` (7) ·
`tests/vision_os/registry/test_reentry_ordering.py` (8) ·
`tests/vision_os/synthesis/test_conformance_isolation.py` (7) ·
`tests/vision_os/integration/test_identity_continuity.py` (10)

### Intentionally not changed

Detector, weights, confidence thresholds · VLM, prompts, model selection ·
`nvidia_vl.py` (rule 20) · IoU / association / lifecycle thresholds ·
`IncidentService` dedup semantics · attribute state semantics ·
compliance rule severities · policy documents · `analysis_enabled` work ·
frontend · reporting · Phase 0 design system.

**One assertion was corrected, none weakened.** In
`test_identity_continuity.py` I first asserted that a one-frame false positive
leaves no *present* object. That was wrong about the architecture, not a code
defect: `is_present` deliberately includes `PROVISIONAL`, and refusing to record
what was detected would be its own kind of dishonesty. The test now asserts the
property that actually matters — such an object never reaches `ACTIVE`, the only
lifecycle the policy admits to the model.

---

## 9. What remains uncertain, and escalations

1. **Not yet observed on live data.** Everything here is proven by test and by
   an end-to-end harness on the real tracker and registry. The running process
   still serves the pre-repair code. **A restart is required**, and only then can
   the incident/object ratio and `created=False` count be re-measured. I have not
   restarted it.
2. **The 7 `kit-*.jsonl` files remain in `data/observations/`.** They are now
   **inert** — the kit no longer reads that directory — but they still appear as
   partitions no camera uses. Deleting them is safe and cosmetic; I did not,
   because it is outside this phase's mandate.
3. **`analysis_fps = 2.0` is below the measured continuity cliff.** A
   configuration change I have not made, because changing what the cameras run at
   is your call. Recommended: 3–4.
4. **`face_covering` observability is a product decision.** The evaluation is
   recorded and exposed via `by_rule`; it is invisible on the product surface
   because informational rules raise no incident. Surfacing informational
   findings touches the frontend — **escalated, unchanged**.
5. **Detector false positives are still detector false positives.** The
   architecture now stops one from becoming a compliance violation (it stalls at
   `provisional`, which the policy scope excludes). It does not stop yolov8n from
   producing them. Distinguishing a genuine detector error rate from tracking
   error would need labelled footage from these cameras, which does not exist in
   the repository — `datasets/kitchen-01` has 15 frames and 43 annotated
   subjects, enough to calibrate crop quality and not enough to score a detector.
6. **`bytetrack` unevaluated.** Deliberate: its benefit depends on a
   low-confidence detection tail this configuration does not produce. Revisit
   with measurements if crowding becomes the problem.

---

## 10. Tests

```
before : 3,928 passed, 1 failed   (3,929 collected)
after  : 3,982 passed, 1 failed   (3,983 collected)
delta  : +54 passed, 0 new failures
```

The single failure is unchanged and pre-existing:
`test_ninety_b_configuration.py::test_no_production_module_names_the_model`
(`nvidia_vl.py:73`). Untouched, per rule 20.

Every new test asserts an architectural property. The ones worth naming:

* `test_the_old_fallback_tracker_fragments_on_the_same_frames` — pins the
  regression itself, so a revert to `tracker.iou` fails loudly.
* `test_a_distant_newcomer_is_a_different_person`,
  `test_a_different_class_never_inherits_an_identity` — false-merge prevention,
  weighted equally with duplicate prevention as instructed.
* `test_an_unsupplied_state_is_not_read_as_a_state_name` — exists because I
  wrote that bug and it would have stopped all analysis silently.
* `test_a_real_partition_survives_the_gate_byte_for_byte` — the guarantee a
  cleanup-based observation-log fix could not have made.
* `test_at_two_fps_a_brisk_walk_fragments_on_every_tracker` — a test that
  asserts a **limitation**, so a future "fix" by loosening association is caught.

---

## 11. Go / no-go

**The perception architecture defects are repaired and verified in code.**

To see it working in production:

1. **Restart the backend.** Required for all six repairs, and it is what proves
   the observation-log fix: synthesis should bind, and alerts should resume.
2. Confirm `/api/v1/devtools/compliance` reports non-zero `subjects` and that
   `by_rule` lists all three rules.
3. Raise `analysis_fps` to 3–4 on cam-11…14.
4. After a day, re-measure `incidents / distinct object_ids`. It was 1.0000.
   Anything above 1.0 means deduplication is finally being exercised.

I have not restarted the process or changed camera settings — both are yours to
authorise.
