# Vision OS — Perception Pipeline Forensic Audit

**Stages 1 and 2 only.** Execution-path map and forensic findings. No code has been
modified. Stage 3 below is a *proposal* awaiting approval, not an implementation record.

| | |
|---|---|
| Repository | `unityworks-vision-ai-backend` |
| Branch | `feat/unityworks-vision-os-prod-hardining` |
| HEAD at audit | `9bfbe8f` |
| Date | 2026-09-03 |
| Method | Static trace of the real composition root + query of the live production database |
| Production evidence | 1,990 incidents across cam-11 … cam-14 |

---

## 0. Executive summary

Six reported symptoms trace to **five distinct defects**. None of them is the VLM, and
none of them is the detector model choice.

| # | Symptom | Root cause | Confidence |
|---|---|---|---|
| 1 | False person → violation | Policy's `lifecycle` and `min_confidence` scope is **parsed and never enforced** | **Proven** |
| 2 | Track IDs fragment | `tracker.iou` — the documented *fallback* tracker — is hard-coded as the production tracker; it has no motion model | **Proven** |
| 3 | Detection becomes a semantic subject immediately | `is_present` includes `PROVISIONAL`; no admission gate consults track or object confirmation state | **Proven** |
| 4 | Tracking → registry transition information lost | `TrackingOutcome` collapses exact IDs to **counts**; the bridge then re-derives them incorrectly | **Proven** |
| 5 | Repeated violations for one person | Registry re-entry binder is structurally unreachable in the frame where fragmentation happens | **Proven** |
| 6 | Only head covering appears | Three different causes, none of them the VLM | **Proven** |

The single most consequential number in this audit:

```
camera    incidents   distinct object_ids
cam-12         767            767
cam-13         703            703
cam-11         392            392
cam-14         128            128
```

**Incidents equal distinct object IDs exactly, on every camera.** In 1,990 incidents the
deduplication path in `IncidentService.open` has returned `created=False` **zero times**.
No registry object has ever survived long enough to be observed violating twice. This is
not a deduplication failure — deduplication is correct and never gets the chance to run.
It is total identity churn upstream.

---

## STAGE 1 — Execution path map

### 1.1 The actual runtime path

Traced from the composition root, not from the architecture documents.

```
RTSP source  (app/vision/sources/)
      │
      ▼
VisionSession._consume  ──►  ANALYSIS.run(...)      app/vision/analysis_loop.py:99
      │                       one worker thread, one private event loop,
      │                       every camera serialised on it
      ▼
DetectionRuntime                                    vision_os/perception/detection/runtime.py
      │   yolov8n.onnx, classes=person
      │   VISION_DETECTOR_CONFIDENCE (.env.example: 0.4)
      ▼   DetectionOutcome
TrackingRuntime.on_detected                         vision_os/perception/tracking/runtime.py:108
      │   per-camera asyncio.Lock, bounded by frame_timeout_ms
      ▼
TrackingEngine.track                                vision_os/perception/tracking/engine.py:122
      │
      ├─► GeometricTracker.update  ──► TrackUpdate   vision_os/adapters/tracking/geometric.py:575
      │      COMPLETE: new, terminated, coasting, recovered,
      │      associations, refused, unmatched_detections
      │
      ▼   ✗ INFORMATION DESTROYED HERE
    TrackingOutcome                                 engine.py:166
      │   created:int  terminated:int  recovered:int  coasting:int
      │   (counts only — every ID discarded)
      ▼
TrackingRuntime._publish → sink                     runtime.py:164
      ▼
TrackingToRegistryBridge.__call__                   app/vision/bridges.py:101
      │   ✗ RECONSTRUCTS a TrackUpdate by guessing from track states
      ▼   bridges.py:175
RegistryRuntime.on_tracked                          vision_os/perception/registry/runtime.py:131
      ▼
ObjectRegistry._ingest                              vision_os/perception/registry/engine.py:668
      │   line 698: for track in update.active   ← absorb loop
      │   line 743: for record in ...            ← ageing / unbinding loop
      │   ✗ ORDERING DEFECT between these two loops
      ▼   RegistryUpdate
CroppingRuntime.on_registered                       vision_os/perception/cropping/runtime.py:150
      │   evaluates update.present
      │   ✗ is_present includes PROVISIONAL
      ▼
CropTriggerEngine.evaluate → _admit                 cropping/engine.py:250, 375
      │   DemandRegistry.matching()                 cropping/demands.py:347
      │   ✗ filters class/camera/region — NOT lifecycle, NOT min_confidence
      ▼
UnderstandingRuntime → NVIDIA VLM                   vision_os/perception/understanding/
      ▼   AttributeObservation
Synthesis (M7) → Exposure (M14) → ObservationApi
      ▼
ComplianceDriver._loop (every 5 s)                  app/vision/compliance_driver.py:170
      │   compliance_interval_s = 5.0
      ▼
ComplianceDriver.apply                              compliance_driver.py:454
      │   severity ∈ {low, medium, high, critical}  compliance_driver.py:478
      ▼
IncidentService.open  → dedup on                    app/domain/incidents.py:78
      (organization_id, camera_key, object_id, rule_id)
```

### 1.2 The actual configuration

The platform configuration is **not** a config file. It is a Python literal in
`app/vision/runtime.py:_config_document()`. Values that matter here:

| Setting | Value | Source |
|---|---|---|
| `tracking.tracker_id` | **`tracker.iou`** | `app/vision/runtime.py:656` — hard-coded literal |
| `tracking.min_hits_to_confirm` | 3 | schema default |
| `tracking.max_coast_frames` | 5 | schema default |
| `tracking.max_lost_frames` | 15 | schema default |
| `tracking.iou_threshold` | 0.1 | schema default |
| `tracking.max_association_cost` | 0.7 | schema default |
| `tracking.ambiguity_margin` | 0.05 | schema default |
| `registry.min_observations_to_confirm` | 2 | `runtime.py` |
| `cropping.understanding_calls_per_hour` | 3,600 | `runtime.py` |
| `profiles[0].target_fps` | 4.0 | `runtime.py` |
| `compliance_interval_s` | 5.0 | `app/configuration/settings.py:122` |
| `vision_demand_freshness_ms` | 60,000 | `app/configuration/settings.py:172` |
| Detector | `yolov8n.onnx`, `classes=person`, conf **0.4** | `.env.example:89-92` |
| Policies | `kitchen-safety.example.json` v2.1.0 + `object-identity.example.json` | `.env.example:78` |
| Rules | `config/rules/site-safety.example.json` | — |

> The live `.env` could not be read (blocked by the environment's file classifier). Detector
> and policy values above are from `.env.example` and must be confirmed against the running
> deployment before Stage 3 lands.

---

## STAGE 2 — Forensic findings

### A. Actual active architecture vs intended architecture

The intended architecture is layered and each layer is individually well built. The tracker
in particular is **correct**: `GeometricTracker.update` returns a complete `TrackUpdate`
carrying every transition ID, every association, every refusal and every unmatched detection
index (`geometric.py:575-589`). Nothing is wrong with it.

The defects are all in the **seams between layers**, and in **configuration that is declared
but never read**. That is why every layer looks healthy in isolation and the system is
nonetheless untrustworthy end to end.

Three seams are load-bearing and all three leak:

1. `TrackingEngine.track` → `TrackingOutcome` — exact IDs → integers.
2. `TrackingToRegistryBridge._to_update` — integers → guessed IDs.
3. `DemandRegistry.matching` — a policy scope that is parsed, carried, and never consulted.

---

### B. Root-cause candidates ranked by evidence

#### **PROVEN — 1. The policy's subject scope is never enforced**

`config/policies/kitchen-safety.example.json` declares:

```json
"scope": {
  "object_classes": ["person"],
  "lifecycle": ["active", "occluded"],
  "min_confidence": 0.4
}
```

The policy author's intent is unambiguous: *do not analyse provisional objects, and do not
analyse objects below 0.4 identity confidence.*

`SubjectFilter` (`vision_os/core/model/demand.py:140-158`) has exactly three fields —
`class_ids`, `lifecycle`, `min_confidence` — and exactly **one** method: `matches_class`
(line 151). There is no `matches_lifecycle`. There is no confidence check.

`DemandRegistry.matching` (`cropping/demands.py:347-357`) is the only consumer:

```python
return tuple(
    state
    for state in self.active()
    if state.demand.scope.covers_camera(camera_id)
    and state.demand.scope.covers_region(region_ids)
    and state.demand.subject_filter.matches_class(class_id)   # ← the only filter
)
```

A repository-wide search for any read of `subject_filter.lifecycle` or
`subject_filter.min_confidence` returns **nothing**. The fields are write-only.

It is worse than an unenforced field. The production demand is built by
`app/vision/demands.py:146`:

```python
subject_filter=SubjectFilter(class_ids=tuple(ClassId(c) for c in classes)),
```

The application constructs its own standing demand (`uwv/standing`) and **never passes
`lifecycle` or `min_confidence` at all**. It also ignores the policy's entire `demand` block
— `freshness_ms: 60000`, `triggers: [on_first_sight, on_region_entry, on_change]`, and
`budget.max_calls_per_hour: 400` — substituting `vision_demand_freshness_ms` and empty
defaults for the rest. `vision_os/exposure/demands.py:410` independently drops the same two
fields when re-exporting a demand to the wire.

**This is the primary root cause of Problem 1.** The gate the operator configured to keep
false positives out of the compliance pipeline does not exist at runtime.

#### **PROVEN — 2. The production tracker is the fallback tracker**

`app/vision/runtime.py:656`:

```python
"tracking": {"enabled": True, "tracker_id": "tracker.iou"},
```

Hard-coded. Not an environment variable, not a config file, not a per-site setting.

`tracker.iou` is described in its own factory (`adapters/tracking/trackers.py:35-45`) as:

> *"the universal fallback … Deliberately minimal. It exists to keep the pipeline running,
> and every property it lacks — motion prediction, occlusion handling — is one it would need
> a working model or tuning to provide."*

It is built with `use_motion_model=False`, `two_stage=False`, `handles_occlusion="none"` and
a `GreedyAssociator`. The repository ships two strictly better trackers — `tracker.sort`
(linear motion, optimal assignment) and `tracker.bytetrack` (two-stage association) — and
**all three are the same `GeometricTracker` class with a different `GeometricConfig`**.
Switching is a configuration change, not a code change, and carries no new dependency.

> **The fallback did not occur.** This is not a silent degradation — `tracker.iou` is what
> the composition root explicitly asks for. `TrackingManager.is_fallback` would be `False`.
> Any diagnostic that only checks "did we fall back?" will report healthy.

#### **PROVEN — 3. Why continuity breaks — the arithmetic**

`association.py:172` is the entire admission test for a detection-to-track pair:

```python
if overlap < policy.min_iou and separation > gate:
    continue          # not even a candidate
```

with `gate = policy.gate_multiplier * prediction.uncertainty`.

With `use_motion_model=False` the predictor is `StationaryPredictor`
(`adapters/tracking/motion.py:62-68`), which predicts **the box does not move** and reports
`uncertainty = 0.05 × elapsed_seconds`. So:

```
gate = 3.0 × 0.05 × Δt  =  0.15 × Δt      (normalized frame widths)
```

A pair survives gating if **either**:

* **overlap branch** — `IoU(last_box, detection) ≥ 0.1`, or
* **gate branch** — `centre_separation ≤ 0.15 × Δt`.

For a person of normalized box width *w* moving at *v* frame-widths/second:

| Branch | Survives when | With w ≈ 0.1 |
|---|---|---|
| overlap | `v·Δt ≲ 0.8·w` | `Δt ≲ 0.08 / v` |
| gate | `v·Δt ≤ 0.15·Δt` → **`v ≤ 0.15`** | independent of frame rate |

Two consequences, and the second is the important one:

1. **At the observed cadence the overlap branch is dead.** A person walking 1.4 m/s across a
   frame spanning ~6 m moves ≈ 0.23 frame-widths/s. At 1 fps that is 0.23 per frame —
   more than two box widths. IoU is exactly **0**.
2. **The gate branch does not scale with frame rate.** Displacement grows as `v·Δt`; the gate
   grows as `0.15·Δt`. Both are linear in `Δt`, so they cancel: the gate branch admits a
   track **iff `v ≤ 0.15` frame-widths/second**, at *any* frame rate. Lowering or raising
   fps cannot rescue a person moving faster than that.

So for anyone moving at ordinary walking pace, both branches fail, the detection is not a
candidate for the existing track, the track coasts (5 frames), goes lost (15), terminates —
and the detection **spawns a new track in the same frame** (`geometric.py:565-570`).

That is the fragmentation, and it is a deterministic property of running a motion-free
tracker on a low-frame-rate stream. It is not jitter and it is not tuning.

> **Raising the frame rate is a partial mitigation only.** It revives the overlap branch
> (a walking person needs roughly ≥ 3 fps to keep IoU ≥ 0.1), but it does nothing for the
> gate branch and nothing for occlusion. A motion model fixes both, because the prediction
> moves the box to where the person is *going*, restoring overlap at any cadence.

#### **PROVEN — 4. Transition information is destroyed, then guessed**

The tracker produces everything (`geometric.py:575-589`):

```python
return TrackUpdate(
    active=active, new=tuple(new_ids), terminated=tuple(terminated),
    coasting=tuple(coasting), recovered=tuple(recovered),
    associations=tuple(associations), refused=tuple(refused),
    unmatched_detections=tuple(...),
)
```

`TrackingEngine.track` then throws the IDs away (`engine.py:166-179`):

```python
return TrackingOutcome(
    tracks=update.active,
    created=len(update.new),          # ← ID list → integer
    terminated=len(update.terminated),
    recovered=len(update.recovered),
    coasting=len(update.coasting),
)
```

`TrackingOutcome` has no field for `terminated` IDs, `recovered` IDs, `associations`,
`refused` or `unmatched_detections`. They cease to exist at this line.

The bridge then rebuilds a `TrackUpdate` from what survived (`app/vision/bridges.py:175-184`):

```python
new=ids_in(TrackState.TENTATIVE),
coasting=ids_in(TrackState.COASTING),
# terminated, recovered, associations, refused, unmatched_detections
# all silently take their empty defaults
```

Two separate defects here:

* **`new` is wrong, not merely lossy.** `new` means *created this frame*. `TENTATIVE` is a
  state that persists for `min_hits_to_confirm = 3` frames. A track therefore reports as
  "new" on three consecutive frames, and a track created-and-confirmed in one frame never
  reports as new at all.
* **`terminated` and `recovered` are always empty.** The registry is told nothing died and
  nothing came back, on every frame, forever.

The bridge's own docstring claims the ids are *"recovered from the tracks' own states rather
than invented, so a track this bridge reports as new is one the tracker marked new."* That
is not what the code does. `TrackState.TENTATIVE` is not a creation marker.

This is Problem 4 exactly as suspected, and the loss originates one layer higher than the
bridge — in `TrackingOutcome` itself.

#### **PROVEN — 5. Re-entry binding is structurally unreachable**

This is the finding that explains the perfect 767/767 ratio, and it is the most important one
in the report.

The registry has a correct mechanism for exactly this problem. `bind_reentry`
(`registry/binding.py:171`) exists to match a *new* track to an *existing* object by position
and class — precisely what should absorb a fragmented track back into the same person.

It never fires. Two guards make it unreachable in the frame where it is needed:

```python
# binding.py:191
if record.lifecycle not in (LifecycleState.OCCLUDED, LifecycleState.DORMANT):
    continue
# binding.py:193
if record.bound_track is not None:
    continue
```

Now read `_ingest`'s loop order (`registry/engine.py`):

| Line | Loop | What it does |
|---|---|---|
| **698** | `for track in update.active:` | `_absorb` — binds tracks, **mints new objects** |
| **743** | `for record in partition.records():` | ages untouched objects, **closes their bindings** |

When a person's track fragments, both the death of T101 and the birth of T204 land in the
**same frame**. The absorb loop at 698 runs first, so at the moment `_absorb` calls
`bind_reentry` for the new track T204:

* the predecessor object is still `ACTIVE` — the ageing loop that would move it to
  `OCCLUDED`/`DORMANT` has not run yet → **fails the guard at line 191**;
* the predecessor object is still bound to T101 — `close_bindings` is at line 751, inside the
  loop that has not run yet → **fails the guard at line 193**.

`bind_reentry` returns `reason="no_candidates"`, and `_absorb` falls through to `_mint`
(`engine.py:869`) — a brand-new `ObjectId`. By the next frame the new object exists and is
`ACTIVE`, and the old one simply ages alone to `DEPARTED`.

The re-entry window (`max_reentry_gap` 30 s, `max_reentry_distance` 0.25,
`min_binding_confidence` 0.3) is generously sized and completely irrelevant, because the
candidate set is empty before any threshold is consulted.

**A fragmented track can never be re-bound to its own object.** That is why the production
data shows a 1:1 ratio with no exceptions.

---

### C. Tracking analysis — why IDs fragment

Summarised causally:

```
tracker.iou hard-coded (runtime.py:656)
   → StationaryPredictor: predicted box = last box, uncertainty = 0.05·Δt
      → gate = 0.15·Δt, admits only v ≤ 0.15 frame-widths/s
      → at ~1 fps a walking person has IoU = 0 and separation > gate
         → no candidate → track coasts → lost → terminated
         → the same detection spawns a NEW track in the SAME frame
```

Against the reported failure signature:

| Reported | Explanation |
|---|---|
| Frame 1→2 same ID (101) | Person briefly slow or stationary — gate branch admits |
| Frame 3 → 204 | Person moved > 0.15 frame-widths in the interval |
| Frames 4, 5 → 315, 412 | Same cause repeating; each break mints a fresh ID |
| Person continuously visible throughout | Correct — detection never failed; **association** did |

Note the diagnostic implication: `BreakReason.ASSOCIATION_FAILURE`, not
`BreakReason.DETECTOR_MISS`. The detector is not the problem, and any investigation that
starts at the detector will find nothing.

---

### D. False detection analysis — where a false positive becomes trusted

There is **no admission or stability layer** between raw detection and expensive semantic
analysis. The confirmation machinery exists but is bypassed at every gate:

| Stage | Gate that exists | Whether it applies | Evidence |
|---|---|---|---|
| Detection | conf ≥ 0.4 | ✅ applies | `.env.example:92` |
| Track creation | `min_hits_to_confirm = 3` → `TENTATIVE` | ⚠️ track is created immediately; state is advisory | `geometric.py:565` |
| Bridge | — | ❌ forwards **all** `active`, including `TENTATIVE` | `bridges.py:164, 175` |
| Registry absorb | — | ❌ `_absorb` mints an object for any track, any state | `engine.py:698, 869` |
| Object lifecycle | `min_observations_to_confirm = 2` → `PROVISIONAL` | ⚠️ state is set but not gated on | `runtime.py` |
| Crop candidate set | `update.present` | ❌ **`is_present` returns `True` for `PROVISIONAL`** | `visual_object.py:87-95` |
| Demand match | `subject_filter.lifecycle = [active, occluded]` | ❌ **never read** | `demands.py:347` |
| Demand match | `subject_filter.min_confidence = 0.4` | ❌ **never read** | `demands.py:347` |

So a **single-frame** false person detection at confidence 0.41:

1. spawns a `TENTATIVE` track,
2. is forwarded by the bridge in `active`,
3. is minted as a `PROVISIONAL` registry object,
4. is included in `RegistryUpdate.present` because `is_present` admits `PROVISIONAL`,
5. passes `matching()` because only its class is checked,
6. is cropped, sent to the VLM, and answered `head_covering: none`,
7. fails `kitchen.person.ppe.v1`, and
8. **becomes a `high` severity incident with evidence and a notification.**

Every one of those steps is doing what its own contract says. The failure is that the two
gates written to stop it — the policy's `lifecycle` and `min_confidence` — are inert, and the
one structural gate (`is_present`) deliberately admits provisional objects.

---

### E. Attribute pipeline analysis — head vs face vs hand

**None of the three is a VLM failure.** The three attributes diverge at three different
layers, and one of them is not broken at all.

A live finding payload, taken from the newest incident in production:

```json
"conditions": [
  { "attribute": "head_covering", "operator": "ne", "expected": "none",
    "observed": "none",        "outcome": "failed",     "unknown_reason": null },
  { "attribute": "hand_covering", "operator": "eq", "expected": "gloves",
    "observed": "not_visible", "outcome": "unresolved", "unknown_reason": "not_observable" }
]
```

Per-attribute trace against the 18 stages requested:

| Stage | `head_covering` | `face_covering` | `hand_covering` |
|---|---|---|---|
| 1. Declared in policy | ✅ v2.1.0 | ✅ v2.1.0 | ✅ v2.1.0 |
| 2–4. Enabled / capability / demand | ✅ | ✅ | ✅ |
| 5–6. Crop planned, region | ✅ band (0.00, 0.45) | ✅ **same band** → shares head's crop | ✅ band (0.15, 0.55) → **its own crop + call** |
| 7–8. Crop generated, quality | ✅ `min_scale 130, blur 0.5` | ✅ identical floors | ⚠️ `min_scale 150, blur 0.85` — policy states these are **PROVISIONAL, fitted to 3 examples** |
| 9–11. VLM request / asked / answered | ✅ | ✅ (free, same call) | ✅ **confirmed present in live payload** |
| 12–14. Parsed, coerced, accepted | ✅ | ✅ | ✅ → `not_visible` |
| 15–17. Registry / synthesis / rule | ✅ `kitchen.person.ppe.v1` | ✅ `kitchen.person.face_covering.v1` | ✅ same rule as head |
| 18. Violation raised | ✅ **severity `high`** | ❌ **severity `informational`** | ⛔ `unresolved` — by design |

**`face_covering` — cause: severity, not perception.** The rule
`kitchen.person.face_covering.v1` is declared `"severity": "informational"`
(`config/rules/site-safety.example.json`). `ComplianceDriver.apply` filters at
`compliance_driver.py:478`:

```python
RAISES_INCIDENTS = frozenset({"low", "medium", "high", "critical"})
...
if finding.severity not in RAISES_INCIDENTS:
    continue
```

`informational` is excluded, so the rule is evaluated and its result is discarded before it
can become an incident. The rule carries `"status": "unmeasured-informational"` — this was a
**deliberate** decision, because the face quality floors were copied from head_covering and
never calibrated. It is correct engineering that is invisible to the operator, who reasonably
reads "no face covering results" as "the feature is broken".

**`hand_covering` — cause: nothing. It is working.** It is requested, cropped (its own call,
because band (0.15,0.55) ≠ (0.00,0.45)), answered, parsed and evaluated. The answer is
`not_visible`, which the rule's `unknown_values` correctly maps to `unresolved` /
`not_observable` — never a violation and never a pass. The policy documents why: of 43
annotated subjects in `datasets/kitchen-01`, **only 3 had hands a human could read**; the
rest were inside a pot, behind a body, or out of frame. The platform is refusing to guess,
which is exactly the designed behaviour.

> **A cost caveat for Stage 3.** `hand_covering` costs a *second* VLM call per person because
> its band differs from head's. At `understanding_calls_per_hour = 3,600` and a budget that
> sheds under pressure (`cropping/engine.py:375 _admit`), this is the attribute most likely
> to be dropped first when the estate is busy — which would make it intermittently invisible
> for a reason unrelated to everything above. This should be measured before it is assumed.

**`head_covering` — working, and the only attribute that can currently produce an incident.**
Which is why it is the only one the operator ever sees.

---

### F. Registry and violation analysis — why one person makes many violations

The deduplication is **correct**. `IncidentService.open` (`app/domain/incidents.py:78`)
looks up an open incident on `(organization_id, camera_key, object_id, rule_id)` and, on a
hit, updates `observed_at` and returns `created=False` — suppressing the duplicate evidence
capture and the duplicate notification (`compliance_driver.py:503`).

It has never once returned `created=False` in production.

The causal chain, end to end, with each link now proven:

```
tracker.iou has no motion model
  → association fails for any normally-moving person          [C]
  → track fragments; a new TrackId is minted in the same frame
  → the absorb loop (engine.py:698) runs BEFORE the ageing loop (engine.py:743)
  → the predecessor object is still ACTIVE and still bound
  → bind_reentry rejects it at binding.py:191 and :193 → "no_candidates"
  → _mint creates a NEW ObjectId                              [B.5]
  → the compliance pass (every 5 s) sees an unfamiliar object_id
  → the dedup key (…, object_id, …) does not match any open incident
  → created=True → NEW incident + NEW evidence capture + NEW notification
```

**Do not add suppression.** Suppressing duplicates here would hide the identity churn while
leaving every downstream count — dwell time, per-person compliance rate, unique-visitor
figures, model evaluation — silently wrong. The duplicates are a symptom that is currently
telling the truth about a broken identity chain.

Corroborating detail — the largest single-pass bursts, grouped by identical `observed_at`
(one compliance pass):

```
2026-08-25 01:46:44.671513  →  15 incidents across 4 cameras
2026-08-25 09:40:40.073729  →  13 incidents across 2 cameras
2026-09-03 04:53:50.291029  →  10 incidents across 3 cameras
2026-08-25 03:38:01.766002  →   9 incidents across 1 camera
```

The last row is the clearest single piece of evidence in the dataset: **nine distinct
"people", each with its own object ID, each raising its own violation, on one camera within
one 5-second window.** A kitchen camera does not see nine simultaneous new people. It saw a
small number of real people whose identities had shattered.

---

## STAGE 3 — Proposed repair plan (NOT IMPLEMENTED — awaiting approval)

Ordered by evidence strength and by ratio of harm removed to risk taken. Each is
independently landable and independently testable.

### R1 — Enforce the subject filter that already exists

| | |
|---|---|
| **Problem** | 1, 3 — false detections reach the VLM and compliance |
| **Evidence** | `demands.py:347`; `SubjectFilter` has no lifecycle/confidence accessor; `app/vision/demands.py:146` never passes them |
| **Root cause** | Declared configuration is never read |
| **Smallest correction** | Add `matches_lifecycle` and `matches_confidence` to `SubjectFilter`; consult them in `DemandRegistry.matching`; pass the policy's `scope.lifecycle` and `scope.min_confidence` through `app/vision/demands.py`; stop dropping them in `exposure/demands.py:410` |
| **Files** | `core/model/demand.py`, `perception/cropping/demands.py`, `app/vision/demands.py`, `exposure/demands.py` |
| **Risk** | **Low–medium.** Fewer crops will be taken. This is the intended effect, but it changes crop volume, so it must land with the instrumentation in R5 to prove what is being excluded and why |
| **Back-compat** | An empty `lifecycle` tuple must continue to mean "every lifecycle", matching `covers_camera`'s existing empty-means-all convention. Existing demands are unaffected |
| **Tests** | Scenario F (one-frame false person → no crop, no violation); a provisional object is excluded; an empty filter still matches everything |

### R2 — Stop destroying transition information

| | |
|---|---|
| **Problem** | 4 |
| **Evidence** | `engine.py:166` collapses IDs to counts; `bridges.py:180` mis-derives `new` from `TENTATIVE` |
| **Root cause** | A lossy intermediate type between two types that both carry the full information |
| **Smallest correction** | Add the ID-carrying fields to `TrackingOutcome` (keeping the existing count fields as derived properties, so no consumer breaks) and have the bridge forward them verbatim instead of reconstructing |
| **Files** | `perception/tracking/engine.py`, `app/vision/bridges.py` |
| **Risk** | **Low.** Purely additive; the counts keep working |
| **Back-compat** | `created`/`terminated`/`recovered`/`coasting` become `@property` returning `len(...)`. Every existing reader is unchanged |
| **Tests** | `new` contains exactly the tracks the tracker created this frame; a 3-frame tentative track is reported new **once**, not three times; `terminated` and `recovered` arrive non-empty |

### R3 — Make re-entry binding reachable

| | |
|---|---|
| **Problem** | 5 — and this is the fix that collapses the 1:1 ratio |
| **Evidence** | `engine.py:698` absorbs before `engine.py:743` ages; `binding.py:191/193` exclude `ACTIVE` and bound objects |
| **Root cause** | Loop ordering, not thresholds |
| **Smallest correction** | Within `_ingest`, close bindings for tracks named in `update.terminated` (available once R2 lands) **before** the absorb loop, so a track that died this frame releases its object in time for the new track to re-bind it |
| **Files** | `perception/registry/engine.py` |
| **Risk** | **Medium.** This changes identity semantics — the thing most worth being careful with. It must not allow two live tracks to bind one object; `bound_track is not None` stays the guard for that. Requires R2 first, because it needs the real terminated ID list |
| **Back-compat** | `RegistryUpdate` shape unchanged. Object IDs become *more* stable, which is the goal, but any downstream consumer that assumed high object churn should be checked |
| **Tests** | Scenario B (continuous person → one object ID); Scenario C (jitter); Scenario D (confidence drop); an ambiguous two-candidate re-entry still refuses and mints, per M7 |

### R4 — Select a tracker with a motion model

| | |
|---|---|
| **Problem** | 2 — the upstream cause of 5 |
| **Evidence** | Section C's arithmetic; `runtime.py:656` |
| **Root cause** | The documented fallback tracker is the production tracker |
| **Smallest correction** | Move `tracker_id` out of the hard-coded literal into settings, defaulting to **`tracker.sort`**. Same class, same port, no new dependency — `use_motion_model=True` plus `OptimalAssociator` |
| **Files** | `app/vision/runtime.py`, `app/configuration/settings.py` |
| **Risk** | **Low mechanically, medium behaviourally.** Trivial to apply and to revert; but it changes tracking behaviour across the estate, so it needs Scenarios B–E before it is trusted |
| **Back-compat** | None broken. `tracker.iou` remains available and remains the automatic fallback |
| **Why `sort` and not `bytetrack`** | `bytetrack`'s advantage is its second association pass over *low-confidence* detections. The detector runs at conf 0.4 and emits only `person`, so there is little weak-detection tail for it to exploit; its benefit is unproven **here**. `sort` addresses the mechanism Section C actually proves. Revisit `bytetrack` with measurements once the crowded-scene case is characterised. |
| **Caveat to verify first** | `tracker_factory` (`tracking_bootstrap.py:88`) passes the config-derived `LifecyclePolicy` to **every** tracker, overriding the tuned per-tracker defaults. Selecting a tracker therefore does *not* pick up its intended `max_coast_frames`/`max_lost_frames`. This is worth confirming and possibly correcting as part of R4 |

### R5 — Forensic instrumentation

| | |
|---|---|
| **Problem** | All six — and the reason this audit needed a database query to reach ground truth |
| **Smallest correction** | An opt-in, per-camera, bounded frame-trace recorder writing the structure requested in the brief: detection → admission → association (score, margin, refusal) → track state → object ID → crop → VLM → rule → violation |
| **Constraints** | Off by default; zero production behaviour when disabled; a bounded ring, never unbounded logging; **no image or biometric data** — box coordinates and IDs only, honouring `RetentionMode.NEVER_PERSIST`; existing privacy architecture untouched |
| **Risk** | **Low** when disabled; must be proven to be a genuine no-op in that state |
| **Note** | This should land **first** or alongside R1, so every subsequent change is measured rather than argued |

### Explicitly rejected

* **Raising the detector confidence threshold.** No evidence supports it. The false-positive path is an unenforced admission gate (R1), not a threshold value, and raising it would cost real detections while leaving the actual hole open.
* **Replacing the VLM.** Section E shows all three attributes reach it and are answered correctly. The VLM is the best-behaved component in the chain.
* **Suppressing duplicate incidents.** Section F — it would hide the identity defect and corrupt every downstream metric.
* **Rewriting the tracking architecture.** The tracker is correct and already emits everything needed. The defects are in seams and configuration.
* **Making `is_present` exclude `PROVISIONAL`.** Tempting, and wrong: `is_present` is a broad domain predicate with other callers, and changing it would alter unrelated semantics. The admission decision belongs in the demand filter (R1), where the policy already tries to express it.

---

## Required test scenarios — current expected status

Against today's code, before any repair:

| Scenario | Expected | Would pass today |
|---|---|---|
| A — empty scene | no subject, no call, no incident | ✅ likely |
| B — continuous person | stable track + object ID | ❌ **fails** (C, F) |
| C — bbox jitter | track continuous | ⚠️ marginal — survives only while `v ≤ 0.15` |
| D — brief confidence drop | no fragmentation | ❌ **fails** — single-stage association drops the detection outright |
| E — short occlusion | per tracker capability | ❌ `handles_occlusion="none"` |
| F — one-frame false person | never a confirmed subject | ❌ **fails** (D) |
| G — head covering trace | full path | ✅ works |
| H — face covering trace | full path | ⚠️ evaluated, then dropped at severity |
| I — hand covering trace | full path | ✅ works, answers `not_visible` |

---

## Appendix — Evidence index

| Claim | Location |
|---|---|
| Tracker hard-coded to the fallback | `app/vision/runtime.py:656` |
| Full `TrackUpdate` produced by the tracker | `vision_os/adapters/tracking/geometric.py:575` |
| IDs collapsed to counts | `vision_os/perception/tracking/engine.py:166` |
| `TrackUpdate` reconstructed from states | `app/vision/bridges.py:175` |
| `new` derived from `TENTATIVE` | `app/vision/bridges.py:180` |
| Association gate | `vision_os/perception/tracking/association.py:172` |
| Stationary prediction + uncertainty growth | `vision_os/adapters/tracking/motion.py:62` |
| Absorb loop before ageing loop | `vision_os/perception/registry/engine.py:698`, `:743` |
| Re-entry excludes `ACTIVE` and bound objects | `vision_os/perception/registry/binding.py:191`, `:193` |
| New object minted on no match | `vision_os/perception/registry/engine.py:869` |
| `is_present` admits `PROVISIONAL` | `vision_os/core/model/visual_object.py:87-95` |
| `SubjectFilter` has only `matches_class` | `vision_os/core/model/demand.py:151` |
| Demand matching ignores lifecycle/confidence | `vision_os/perception/cropping/demands.py:347` |
| App demand omits both fields | `app/vision/demands.py:146` |
| Exposure re-export drops both fields | `vision_os/exposure/demands.py:410` |
| Incident dedup key | `app/domain/incidents.py:78` |
| `informational` never raises an incident | `app/vision/compliance_driver.py:109`, `:478` |
| Face rule is `informational` | `config/rules/site-safety.example.json` |
| Policy scope declares lifecycle + min_confidence | `config/policies/kitchen-safety.example.json` |
| 1,990 incidents = 1,990 distinct object IDs | production database, `incidents` table |
