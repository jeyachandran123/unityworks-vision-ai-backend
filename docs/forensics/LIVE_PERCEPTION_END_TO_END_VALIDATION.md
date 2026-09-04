# Vision OS — Live Perception End-to-End Validation

Validation of the repaired perception architecture against the **running**
pipeline. No architecture was changed in this phase.

| | |
|---|---|
| Repository | `unityworks-vision-ai-backend` |
| Branch | `feat/unityworks-vision-os-prod-hardining` |
| HEAD | `6992489` |
| Date | 2026-09-03, 14:55–15:10 IST |
| Tests | **3,982 passing / 1 pre-existing failure — unchanged** |
| Code changed | **none** |
| Configuration changed | `analysis_fps` 2.0 → 3.0 on cam-11…14 |

---

## 1. Executive verdict

## **PERCEPTION STILL BLOCKED — at layer 0, camera input**

Not by the architecture. The repaired code is confirmed running, and every layer
I could exercise without a camera is confirmed working. The chain is broken
before detection:

> **The DVR's RTSP port is unreachable from this machine.**
> `203.118.57.154:554` — three consecutive connection attempts, 12 s timeout
> each, all timed out. Port 80 on the same host answers in 0.04 s, and DNS
> resolves normally, so the host is up and the network path exists; the RTSP
> service specifically does not answer.

It was working earlier today: the durable observation logs were last appended at
**10:51:24**, and RTSP must have been connected for that to happen. Between then
and now the DVR stopped accepting RTSP.

**What this means for the repair.** Nothing in this finding contradicts it, and
nothing in it validates it either. Four of the seven layers I set out to prove
are now proven on live infrastructure; three cannot be proven until frames flow.

| Layer | Status |
|---|---|
| Repaired code actually running | ✅ **proven** |
| Observation-log gate repaired | ✅ **proven on the real production log** |
| Synthesis + exposure binding | ✅ **proven** |
| Policy gate consuming real policy | ✅ **proven** |
| Tracker selection | ✅ **proven** (`tracker.sort` composed) |
| VLM reachability | ✅ **proven** (model offered) |
| Camera input | ❌ **BLOCKED — RTSP 554 unreachable** |
| Live identity continuity | ⛔ cannot measure — no frames |
| Live dedup behaviour | ⛔ cannot measure — no frames |
| False-positive origin | ⛔ cannot measure — no frames |

**I did not restart the backend.** Reasons in §4. **I did not touch code.**

---

## 2. Running-code verification

The prior report warned "the running process still serves pre-repair code". That
was true when written. It is no longer.

### Processes, by exact PID

| PID | Parent | Started | What |
|---|---|---|---|
| 30000 | 13156 | 12:09:48 | `python -m uvicorn app.main:app --reload --port 8010` (launcher) |
| **11364** | 30000 | 12:09:48 | uvicorn **reload supervisor** |
| **24840** | 11364 | **14:38:15** | **the live worker** (`multiprocessing.spawn`, child of 11364) |
| 33332 / 33068 | — | 12:09:49 | frontend `npm run dev` + vite |
| 20296 / 29904 | — | 02-09 | vite :5277 |
| 17920 / 14472 | — | 02-09 | vite :5278 |
| 36272 | 4656 | 11:09:48 | pgAdmin |
| 26864 | 29860 | 02-09 | AmazonQ language server |
| 19168 / 28840 | — | 09:49:38 | stale scratchpad `x.py` from an earlier session |

### Proof the worker holds the repaired code

Mtimes of the eleven repaired production files, against the worker's start:

```
latest repaired file   vision_os/core/model/demand.py   14:22:52
worker 24840 spawned                                    14:38:15   ← 15m later
```

Under `--reload` the supervisor respawns the worker on file change, so the
worker that exists now was created **after** the last edit. Every repair marker
is present on disk:

```
vision_tracker_id     app/configuration/settings.py            ✓
for_conformance       vision_os/adapters/synthesis/stores.py   ✓
_conformance_twin     vision_os/synthesis_bootstrap.py         ✓
matches_lifecycle     vision_os/core/model/demand.py           ✓
update=update         vision_os/perception/tracking/engine.py  ✓
ended = {track_id     vision_os/perception/registry/engine.py  ✓
```

### Independent, behavioural corroboration

Stronger than a timestamp. The conformance kit used to write seven `kit-*.jsonl`
files into the production observation log on **every** boot. Those files are
still dated **10:52:35** — the pre-repair boot. The 14:38 boot **added nothing**.

```
data/observations/
  cam-11..14.jsonl          10:51:24 – 10:51:26   (last real observations)
  kit-*.jsonl  × 7          10:52:35              (pre-repair boot, unchanged)
```

The live process booted and left the durable log untouched. That is the
conformance-twin repair working in production, observed rather than asserted.

Note also: `git status` shows the perception repair is **uncommitted** working
tree. Commit `6992489` contains only the earlier `analysis_enabled` work.

---

## 3. Camera configuration inventory

Runtime settings actually loaded:

```
vision_tracker_id        = tracker.sort          ← repaired default in force
vision_semantic_policy   = ./config/policies/kitchen-safety.example.json,
                           ./config/policies/object-identity.example.json
observation_log          = file  ./data/observations
vision_demand_freshness_ms = 60000
compliance_interval_s      = 5.0
```

| Camera | enabled | analysis_enabled | analysis_fps | Stream | Runtime state | Role |
|---|---|---|---|---|---|---|
| cam-01 … cam-10 | false | false | 4.0 | main | not started | off |
| **cam-11** | **true** | **true** | **2.0 → 3.0** | main | **no session — RTSP unreachable** | kitchen, analysed |
| **cam-12** | **true** | **true** | **2.0 → 3.0** | main | **no session** | kitchen, analysed |
| **cam-13** | **true** | **true** | **2.0 → 3.0** | main | **no session** | kitchen, analysed |
| **cam-14** | **true** | **true** | **2.0 → 3.0** | main | **no session** | kitchen, analysed |
| cam-15, cam-16 | false | false | 4.0 | main | not started | off |

`analysis_enabled` is honoured: only the four kitchen cameras are scheduled for
perception. Transport for all four: `gayatri.freemyip.com:554`, channel = camera
number, `stream_type=main`, `credential_ref=env:CCTV_PASSWORD` (a pointer, never
a value).

### Runtime state, established by socket inspection

Worker 24840's established connections:

```
::1        → 5432   PostgreSQL   × 2      ✓
::1        → 6379   Redis        × 1      ✓
127.0.0.1  ↔ 127.0.0.1  × 12             internal self-pipes
--- non-local remote endpoints:  NONE ---
```

**Zero connections to 203.118.57.154.** No RTSP session exists. Not one being
retried, not one stalled mid-handshake — none.

---

## 4. FPS decision

**Reported before applying, as required.**

| | |
|---|---|
| Current value | `2.0` on cam-11…14 |
| Proposed | `3.0` |
| Applied | ✅ yes, via `CameraService.update` (audited path, not raw SQL) |

**Why 3.0.** The repair phase measured, on the real tracker and registry, that
continuity for a briskly-walking subject breaks between 2 and 3 fps and holds
from 3 upward — because per-frame displacement must fall below roughly half the
subject's box width for association to survive. 3.0 is the first value on the
correct side of that boundary. Higher values buy nothing for identity continuity
and cost detection CPU linearly, which is why 4, 10, 15 and 30 were all rejected.

**Verified, not assumed — does this increase model spend?**

The prior report claimed model spend is governed by freshness and budget rather
than frame rate. I checked the implementation rather than repeating it:

* `analysis_fps` reaches the sampler and nothing else:
  `Camera.analysis_fps` → `to_rtsp_config` → `manager.py:240` → `session.py:143`
  `FrameSampler(spec.analysis_fps)`. The platform profile's `target_fps: 4.0` is
  a separate scheduler concept and does **not** override it — so 2.0 was genuinely
  the effective rate, with no other configuration overriding it.
* Re-analysis is gated on **time**, not frames
  (`adapters/cropping/triggers.py`): rule 3 is
  `candidate.attributes[key].is_stale(now, freshness)`; rule 7 is
  `now.ns - candidate.last_analysed.ns >= refresh_interval.ns`; the terminal case
  is `FRESH_ENOUGH` — *"the platform correctly spends nothing"*.
* The frame-triggered rules are bounded by **events**, not frames: `FIRST_SIGHT`
  fires once per object per attribute, `LIFECYCLE_TRANSITION` on transitions,
  `APPEARANCE_CHANGED` needs an embedding provider (none ships, disabled).

**Conclusion: VLM calls scale with the number of distinct logical objects, not
with frame rate.** So the expected effect is:

| | Effect |
|---|---|
| Detection inference | **+50 %** per analysed camera (2→3 fps × 4 cameras = 12 fps aggregate, against `scheduler.global_budget_fps` 60.0 — ample headroom) |
| Model-call budget | unchanged (`understanding_calls_per_hour`) |
| VLM calls | **expected to fall**, not rise — fragmentation was inflating the object count and each object costs its own crops |

**This change takes effect at the next camera start.** It is inert today because
no camera can start.

**Rollback:**

```bash
cd unityworks-vision-ai-backend
./.venv/Scripts/python.exe - <<'EOF'
import asyncio, sys; sys.path.insert(0, '.')
from app.configuration.settings import get_settings
from app.domain.cameras import CameraService
from app.infrastructure.database import Database
from app.domain import models as _d
from app.users import models as _i
async def m():
    s = get_settings(); db = Database(s); db.connect()
    async with db.session_scope() as ses:
        for k in ('cam-11','cam-12','cam-13','cam-14'):
            await CameraService(ses).update(organization_id=s.default_tenant_id,
                camera_key=k, assigned_by='rollback', analysis_fps=2.0)
asyncio.run(m())
EOF
```

---

## 5. Observation publication verification

This is the strongest positive result in the phase, and readiness was
deliberately not used as evidence — the previous outage proved `/health/ready`
reports `vision_os: true` while publication is dead.

Instead the boot path was executed against the **real** production observation
log directory, with the seven poisoned kit files still in it:

```
observation_log      : file ./data/observations
tracker (settings)   : tracker.sort
tracker (composed)   : tracker.sort
log dir BEFORE       : cam-11..14.jsonl + kit-log-{idempotent,monotonic,order,
                       part-a,part-b,tail}.jsonl + kit-obs-cam.jsonl

=== BINDING RESULT ===
  synthesis bound    : True      ← was None; this is exactly what was broken
  exposure bound     : True      ← was never built
  observation log    : FileObservationLog   (durable, not the in-memory fallback)

log dir AFTER        : (identical)
new files written    : NONE
files removed        : NONE
```

Four things are established at once:

1. **Synthesis binds.** The `ObservationError` that unbound it is gone.
2. **Exposure is built**, so observations are readable and compliance has a
   source of subjects.
3. **The kit wrote nothing** into the durable log — the twin is doing its job.
4. **It recovered an already-poisoned log.** The seven pre-existing kit files did
   not have to be deleted first. That was the open question from the previous
   phase, and the answer is that no manual cleanup is required.

The VLM also bound and answered:

```
understanding bound — provider=understander.nvidia_vl producible=4
understander 'understander.nvidia_vl' reachable,
             model 'meta/llama-3.2-11b-vision-instruct' offered
```

---

## 6. End-to-end pipeline trace

| Stage | Status | Evidence |
|---|---|---|
| Camera input | ❌ **STOPS HERE** | RTSP 554 × 3 attempts, 12 s timeout each; zero sockets to the DVR |
| Detection | ⛔ not reached | no frames |
| Tracking | ⛔ not reached | — |
| Track continuity | ⛔ not reached | — |
| Registry identity | ⛔ not reached | — |
| Policy demand filtering | ✅ verified out-of-band | §7 |
| VLM / attributes | ✅ bound + reachable | model offered |
| Observation publication | ✅ **bound** | §5 |
| Compliance evaluation | ⏸ running, 0 subjects | driver alive on a 5 s interval |
| Incident dedup | ⛔ not exercised | no findings |
| Incident storage | ⏸ static | 1,990, latest `05:06:41Z` |
| Product / API visibility | ✅ serving | `/health` 200 |

Observation logs last appended **10:51:24**; the current time at audit was
**14:55 IST (09:25 UTC)**. Roughly four hours of silence, coincident with RTSP
becoming unreachable.

---

## 7. Policy gate validation

Verified against the **real** policy files the running deployment loads, not a
test fixture. The composed demand now carries the whole contract:

```
--- kitchen-safety@2.1.0 ---
  class_ids       : ('person',)
  lifecycle       : ('active', 'occluded')   ← previously dropped
  min_confidence  : 0.4                      ← previously dropped
  attributes      : ('head_covering', 'face_covering', 'hand_covering')
  freshness_ms    : 60000
  trigger_hints   : ('on_first_sight', 'on_region_entry', 'on_change')  ← previously ()
  priority_class  : safety-observation                                  ← previously ""
  budget/hour     : 400                                                 ← previously unbounded
```

Gate behaviour, exercised through the real `SubjectFilter` predicates:

| Subject | lifecycle | conf | Outcome |
|---|---|---|---|
| one-frame false positive | `provisional` | 0.9 | **excluded** |
| low-confidence subject | `active` | 0.2 | **excluded** |
| confirmed person | `active` | 0.9 | **ANALYSED** |
| person behind a counter | `occluded` | 0.9 | **ANALYSED** |
| departed, retained | `dormant` | 0.9 | **excluded** |

All five behave as the policy document specifies. Provisional detections are
excluded from active-only analysis; active and occluded are eligible. **No policy
threshold was changed** — these are the values already in the file, now actually
consulted.

Two side-effects worth naming, both cost reductions that were previously being
discarded: the per-demand budget of **400 calls/hour** is now enforced where it
was unbounded, and the trigger hints now reach the platform.

`object-identity@1.0.0` likewise now carries `lifecycle=('active',)`,
`min_confidence=0.25`, `freshness=300000`, `budget=120/hour`.

---

## 8. False-positive investigation

**Not performed. No evidence available.** The original symptom — "no person in
the image, but the system marks a person" — requires live frames to trace, and
there are none. Attributing it to any layer today would be guesswork.

What *is* established, from the repair phase's end-to-end tests: a one-frame
detection reaches `provisional` and never `active`, and §7 shows `provisional` is
now excluded from analysis. So a single spurious detection can no longer become a
compliance violation **through the architecture**. Whether yolov8n produces such
detections on these cameras, and at what rate, remains unmeasured and needs
frames plus labelled footage. The repository's `datasets/kitchen-01` has 15
frames and 43 annotated subjects — enough to calibrate crop quality, not enough
to score a detector.

---

## 9. Attribute visibility results

**Not re-measured live.** No new findings have been produced since 05:06 UTC.

The state semantics are intact and unchanged — nothing in this phase or the
repair phase collapsed them:

| State | Meaning | Behaviour |
|---|---|---|
| positive (e.g. `hairnet`) | model saw a covering | condition passes |
| negative (`none`) | model saw a bare head | condition **fails** → violation |
| `not_visible` | model looked and could not tell | mapped by `unknown_values` → **unresolved**, never a pass, never a violation |
| `unresolved` / `not_observable` | the rule's reading of the above | no incident |
| informational | rule severity | evaluated, recorded, **raises no incident** |

`face_covering` observability, re-confirmed by code path:
`CompliancePass.record()` runs for **every** finding at
`compliance_driver.py:261`, before and independently of the `RAISES_INCIDENTS`
filter at line 478, and `to_wire()` publishes `by_rule`. So informational
findings are counted and exposed at `/api/v1/devtools/compliance`. They are
absent from the product surface only because informational rules raise no
incident — a product decision, still escalated, still unchanged.

---

## 10. Incident deduplication behaviour

**Cannot be exercised.** No compliance evaluation has produced a finding since
the camera feed stopped.

```
incidents = 1990   distinct object_ids = 1990   ratio = 1.0000
latest    = 2026-09-03 05:06:41Z   (pre-repair)
```

The ratio is unchanged because no new incident has been created. **This is not
evidence that the repair failed** — it is evidence that nothing has been
measured. No duplicate suppression was manufactured, and no incident was merged
or hidden.

---

## 11. Before / after measurements

| Metric | Before repair | After repair | Comparability |
|---|---|---|---|
| Synthesis binds at boot | ❌ No | ✅ **Yes** | **COMPARABLE — improved** |
| Exposure layer built | ❌ No | ✅ **Yes** | **COMPARABLE — improved** |
| Kit files written per boot | 7 | **0** | **COMPARABLE — improved** |
| Production observations deleted by gate | n/a | **0** | **COMPARABLE** |
| Composed tracker | `tracker.iou` | **`tracker.sort`** | **COMPARABLE — improved** |
| Policy `lifecycle` enforced | ❌ No | ✅ **Yes** | **COMPARABLE — improved** |
| Policy `min_confidence` enforced | ❌ No | ✅ **Yes** | **COMPARABLE — improved** |
| Per-demand budget enforced | ❌ unbounded | ✅ **400/hr** | **COMPARABLE — improved** |
| `analysis_fps` | 2.0 | 3.0 | config change, effect unmeasured |
| Incident / object ratio | 1.0000 | 1.0000 | **NOT COMPARABLE** — no new data |
| `created=False` events | 0 | 0 | **NOT COMPARABLE** — no new data |
| Tracks created per minute | unmeasured | unmeasured | **NOT COMPARABLE** |
| Track continuity duration | unmeasured | unmeasured | **NOT COMPARABLE** |
| VLM calls per logical person | unmeasured | unmeasured | **NOT COMPARABLE** |
| Observations published/min | 0 since 10:51 | 0 | **NOT COMPARABLE** — no frames |

No trend line is drawn. Everything above is a point-in-time measurement and is
labelled as such.

---

## 12. New defects

**None found in the perception architecture.**

One **operational** condition found, classified as required:

| | |
|---|---|
| Layer | 0 — camera input / acquisition |
| Code path | none. `app/main.py:_start_cameras_from_database` → `live.start_from_records` → RTSP dial to `gayatri.freemyip.com:554` |
| Evidence | 3 × `TimeoutError` at 12 s on `203.118.57.154:554`; port 80 open in 0.04 s; DNS resolves; zero non-local sockets on worker 24840 |
| Reproduction | `socket.create_connection(('203.118.57.154', 554), timeout=12)` |
| Existed before? | **No.** Observations were written at 10:51:24 today, which requires a live RTSP session |
| Blocks the repaired architecture? | **Yes — completely.** No frames means no detection, tracking, identity, attributes, findings or incidents |
| Minimal repair | **Not a code repair.** DVR/router/network — RTSP service, port forward, or ISP path. Outside this repository |
| Regression risk | none (no change made) |

I made no code change for it, per rule 10: there is no evidence of a code defect
here.

---

## 13. Changes made

**Code: none.** `git status` shows exactly the eleven files from the prior repair
phase, unmodified since, plus the six new test files and the two reports.

**Configuration: one change.**

| What | From | To | How | Why |
|---|---|---|---|---|
| `cameras.analysis_fps` on cam-11…14 | 2.0 | 3.0 | `CameraService.update` (audited domain path) | §4 — measured continuity boundary |

Rollback in §4.

**Not done, deliberately:**

* **No restart.** Three reasons: the worker already runs the repaired code
  (§2, proven two independent ways); with RTSP unreachable a restart cannot
  validate anything; and restarting the user's live server has real cost and no
  benefit right now. The `--reload` supervisor will pick the new fps up on the
  next respawn, or it can be restarted deliberately when the DVR returns.
* **No process was killed.** Every PID above was recorded and observed only.
* **The 7 `kit-*.jsonl` files were left in place.** They are now inert — §5 proves
  the boot path ignores them — and deleting them is cosmetic.
* **`nvidia_vl.py` untouched**, per rule 6.

---

## 14. Test results

```
entering this phase : 3,982 passed, 1 failed
after this phase    : 3,982 passed, 1 failed
delta               : 0
```

Targeted suites re-run clean: tracking, registry, cropping/policy, synthesis,
integration, transition propagation, tracker selection, compliance incidents,
analysis_enabled — **all passing**.

The single failure is unchanged and pre-existing:
`test_ninety_b_configuration.py::test_no_production_module_names_the_model`
(`vision_os/adapters/understanding/nvidia_vl.py:73`). Not attributed to this
work, not touched.

No test was added — this was a validation phase, and adding tests to prove an
unreachable camera would prove nothing.

---

## 15. Rollback instructions

| Change | Rollback |
|---|---|
| `analysis_fps` 3.0 → 2.0 | script in §4 |
| Perception repair (if ever needed) | `git checkout -- <the 11 files>`; they are uncommitted working-tree changes |
| Tracker selection alone | set `VISION_TRACKER_ID=tracker.iou` — no code change, no redeploy |
| Runtime | nothing was started or stopped; nothing to roll back |

---

## 16. Final recommendation

## **NOT READY — BLOCKED BY LAYER 0 (CAMERA INPUT / RTSP ACQUISITION)**

The blocker is infrastructure, not software. Everything below the camera that
could be exercised without frames is working, and three of the repairs are now
proven on live production state rather than only in tests.

**To unblock, in order:**

1. **Restore RTSP on the DVR.** `203.118.57.154:554` must accept connections.
   Port 80 answers and DNS resolves, so check the RTSP service itself, the port
   forward, and whether the ISP path to 554 changed. It was working at 10:51
   today.
2. **Restart the backend** once RTSP answers, so the four cameras start at the
   new 3 fps:
   ```
   # stop exactly: 30000 (launcher), 11364 (supervisor), 24840 (worker)
   cd unityworks-vision-ai-backend
   ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8010
   ```
   Consider dropping `--reload` for a real run: every file save tears down all
   RTSP sessions.
3. **Confirm frames** — non-zero sockets to 203.118.57.154, and
   `data/observations/cam-*.jsonl` growing again.
4. **Then, and only then, re-run this validation.** The three unproven layers —
   identity continuity, dedup behaviour, false-positive origin — need roughly an
   hour of real footage with people in frame. The measurement that matters is:

   ```
   incidents / distinct object_ids
   ```

   It was **1.0000**. Anything above 1.0 means deduplication is finally being
   exercised, which is the whole point of the repair.

Until then the honest position is that the architecture is repaired and
**partially** proven in production: the parts that do not need a camera are
verified working; the parts that do are untested.
