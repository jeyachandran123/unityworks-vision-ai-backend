# FINAL — Live PPE Violation Detection Hardening

**Date:** 2026-08-31
**Deployment:** UnityWorks Vision OS, kitchen CCTV `cam-11`…`cam-14`, tenant `org-unityworks`
**Scope:** trace the live pipeline from CCTV frame to dashboard alert, find where real PPE
violations are lost, fix it with evidence.

---

## 1. Executive summary

Violations stopped because **the VLM stopped answering**, not because the kitchen became
compliant and not because any pipeline stage broke.

At 09:56:35 IST the deployment's model was changed from
`meta/llama-3.2-11b-vision-instruct` to `minimaxai/minimax-m3`. The uvicorn `--reload`
worker respawned 48 seconds later at 09:57:23. The last violation this system produced was
at **09:52:16 IST — five minutes before that reload.** In the 50 minutes that followed,
the pipeline detected 5,477 people, held 3,807 tracks, cut 2,720 crops, and produced
**51 answered inferences against 1,558 refusals — a 3.2% success rate.** Every refusal was
HTTP 429.

The account is rate limited **on that model specifically**. Measured back-to-back on the
same key, endpoint, crops and prompt, in three orderings including strict alternation:

| Model | Answered | HTTP 429 |
|---|---:|---:|
| `meta/llama-3.2-11b-vision-instruct` | **16 / 16** answered¹ | 0 |
| `minimaxai/minimax-m3` | **1 / 16** | 15 |

A dedicated retry-with-backoff collection run later the same day sent **87 requests for
`minimaxai/minimax-m3` and received 87 × 429 — zero answers.**

¹ *"Answered" means the endpoint replied, not that the reply parsed. See §24a — the
comparison model's usable-output rate is lower than this table implies, which does not
affect the root cause but does affect how the comparison should be read.*

Everything downstream of inference was verified working and is unchanged. The alert path in
particular was never broken: **500 active incidents** were already stored and being served
to the Alerts page, including 200 examined in detail with correct condition-level evidence.

Two defects were found and fixed:

1. **The failure was silent.** `health()` reported `{"available": true, "state": "ok"}`
   while 96.8% of inference was dying. Fixed — see §3.
2. **Historical Llama evidence had been relabelled as MiniMax** by a repo-wide
   find-and-replace, including files marked "EVIDENCE, not a target" and "never edited
   afterwards". Restored — see §22.

One structural limitation was found and **not** fixed, because fixing it means changing
production policy: the system cannot evaluate left and right hands independently (§13).

---

## 2. Which failure mode was it

The brief lists eleven candidates. Measured, on live traffic:

| | Candidate | Verdict | Evidence |
|---|---|---|---|
| A | Person never detected | **No** | 5,477 person detections across 4 cameras |
| B | Wrong crop created | **No** | 2,720 crops produced; 75% graded `excellent` |
| C | Region not observable | **No** | correctly reported as `not_observable`, distinct from absent |
| D | VLM never called | **No** | 1,609 calls issued |
| E | **VLM rate-limited / failed** | **YES** | **1,558 × HTTP 429; 51 answered (3.2%)** |
| F | VLM returned wrong state | No | correct on every answered call inspected |
| G | Response parsed incorrectly | **No** | 0 unparseable; fenced JSON handled |
| H | Attribute never reached registry | **No** | 204 attributes applied |
| I | Rule misinterpreted it | **No** | 200 incidents show correct condition logic |
| J | Alert delivery failed | **No** | incidents persisted with evidence refs |
| K | Dashboard didn't display | **No** | 500 active incidents served on the page's own query |

**The loss is entirely at E, and it is 96.8% of all PPE evidence.**

---

## 3. Root cause

`minimaxai/minimax-m3` **is listed** by `https://integrate.api.nvidia.com/v1/models`, so
`probe()` passed at binding and reported the model healthy. The account simply has
effectively no serving quota for it.

That combination walked straight past the guard built after the previous outage. The
2026-08-26 incident — `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` retired upstream, 18 hours
of silent "no alerts" — produced a 404/410 retirement latch. A 429 is *not* a retirement:
it is retryable and a quota can reset, so latching it would leave a deployment dark after a
spike. The latch was right to ignore it. Reporting `"ok"` was not.

So the same silence returned wearing a different status code, and `health()` said the
analysis was fine while the product went blind.

---

## 4. The fix

### 4.1 Code — `vision_os/adapters/understanding/nvidia_vl.py`

| Change | Why |
|---|---|
| `RateLimitedError` raised on 429 | its own type, so no layer matches on message text |
| `_Stats.rate_limited` counter, added to `to_wire()` | `failed` alone cannot separate "quiet kitchen" from "over quota" |
| Bounded rolling outcome window (`HEALTH_WINDOW = 32`) | health is a question about *now*; lifetime totals would call a dead adapter healthy |
| `health()` reports `rate_limited` / `failing` | sustained failure is reported; the wording names the operator's next action |

`HEALTH_MIN_SUCCESS = 0.5` is **measured, not chosen**: the failing configuration ran at
3.2% and the working one at 100% on the same account, so the floor separates them by an
order of magnitude in both directions without being fitted to either.
`HEALTH_MIN_SAMPLES = 10` stops a cold restart looking like an outage.

This changes what is **reported**, never what is **produced**. A 429 was always a refusal
and never became an attribute — U2 was not broken. The bug was that nobody was told.
`health()["available"]` is report-only; it gates no pipeline stage.

Verified live on the running deployment before the model was switched back:

```json
{"available": false, "state": "rate_limited", "model": "minimaxai/minimax-m3",
 "reason": "the model service is rate limiting this account: only 0 of the last 32 crops
            were answered (0%). No PPE attribute is being produced for the rest, so an
            empty Alerts page reflects the quota and not the scene. Raise the quota for
            'minimaxai/minimax-m3' or name a model this account can serve."}
```

### 4.2 Configuration — two changes to `.env`

**A typo meant the model line was never read at all.** Line 66 said
`VISION_NVIDIA_MODE` — missing the `L` — and the correctly named line was commented out.
Pydantic's `extra="ignore"` discarded it silently; `Settings().vision_nvidia_model`
resolved to `''`. The deployment ran MiniMax only because the source `DEFAULT_MODEL` had
also been changed to it. **Any future model change through `.env` would have done nothing.**
Corrected to `VISION_NVIDIA_MODEL=` and verified to reach the factory.

**The model was then pointed at one this account can serve:**
`VISION_NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct`. This is a one-line, fully
reversible deployment setting, and §16/§23 cannot be satisfied without it — the live chain
cannot be demonstrated through a model that answers 0% of requests. **This is your call to
keep or revert; see §26.**

---

## 5. Current architecture trace

```
RTSP  →  detection (YOLOv8n)  →  tracking  →  registry
                                                  │
                              demand + budget  ───┤
                                                  ▼
                                        cropping (part-focused)
                                                  │
                                        quality gate ── rejected
                                                  ▼
                                    understanding (NVIDIA VLM)  ←── THE LOSS WAS HERE
                                                  │
                                    parse → schema split
                                                  ▼
                                       synthesis → registry
                                                  ▼
                                     compliance rules (Kleene)
                                                  ▼
                                   findings → incidents → API → Alerts page
```

Two attribute groups per person: head band `(0.00, 0.45)` at 448px answering
`head_covering` **and** `face_covering` in one call, and hand band `(0.15, 0.55)` at the
default size answering `hand_covering` — 2.00 model calls per person.

---

## 6. Person detection results (§4)

Measured over the 50-minute rate-limited window:

| Camera | Frames processed | Person detections | Tracks created | Registry objects |
|---|---:|---:|---:|---:|
| cam-11 | 1,819 | 996 | 713 | 623 |
| cam-12 | 1,811 | 1,631 | 1,546 | 1,556 |
| cam-13 | 1,847 | 2,239 | 961 | 926 |
| cam-14 | 1,697 | 611 | 587 | 514 |
| **Total** | **7,174** | **5,477** | **3,807** | **3,619** |

All four cameras `online`, `producing: true`, **0 reconnects, 0 errors**, ~24,600 frames
produced each, analysis sampled at 1.0 fps. Detection timeouts: 7 across all four cameras
over 7,174 frames (0.1%).

**Detection is not the problem.** No threshold was touched.

---

## 7. Tracking results

3,807 tracks created from 5,477 detections, 3,619 reaching the registry. Ratios are stable
across cameras and consistent with people being tracked across frames rather than
re-created per frame. `tracking.association_failures` and `tracking.recovered` counters
exist and were not elevated.

---

## 8. Crop quality results (§5)

| Metric | Count |
|---|---:|
| Crop requests | 3,354 (`first_sight` 1,300, `attribute_missing` 2,054) |
| Crops produced | **2,720 (81.1%)** |
| Gate rejected | 634 — `too_occluded` 351, `too_blurry` 249, `too_small` 34 |
| Quality `excellent` | 2,034 |
| Quality `good` | 53 |
| Quality `marginal` | 633 |

18.9% loss at the crop gate is real but is an order of magnitude smaller than the inference
loss, and the rejections are the gate doing its job — a blurred or occluded crop yields a
guess, which is the failure this platform was built to avoid.

Head-band geometry is calibrated on `datasets/kitchen-01`: `output_size` 448 raised head
accuracy from 23.3% to 74.4% and cut false-ABSENT-on-a-covered-head from 20 to 4.
`min_scale_pixels 130` / `max_blur 0.5` are measured floors, not guesses.

Hand-band floors are **provisional and labelled as such in the policy** — only 3 of 43
kitchen-01 subjects had hands a human could read, which cannot support a threshold.

---

## 9. Head / hairnet results (§8)

Working. Live evidence from the new incidents produced after the fix:

```
head_covering  ne  none   observed=none  ->  failed  (violation)
```

The policy's question explicitly handles the failure that once cost three chefs a false
violation: *"answer it EVEN IF loose hair is also visible below or around it, which is
normal and does not make the head uncovered."* Verified present and unmodified.

Region observability for the head is produced by `PoseRegionObservability`, measured on
kitchen-01 against **human** annotation only: precision 96.6%, recall 87.5%, unsafe
acceptance 1/11.

---

## 10. Face / mask results (§7)

The rule exists (`kitchen.person.face_covering.v1`), reads `face_covering`, and is
**informational** severity by deliberate design — no face has been annotated in this
repository, so its precision and recall are *unmeasured, not poor*.

Domain: `none | mask | mask_below_nose | other | not_visible`. `mask_below_nose` is a
decided state, not a refusal, so a chin mask fails the requirement rather than becoming
"we could not tell".

**§7's specific concern — profile faces — is correctly handled.** The policy question says
*"A face seen from the side IS visible — answer it normally."* This is now pinned by
`test_a_profile_is_answerable_by_policy`.

Live: 0 face violations were raised in the observation window. No mask-related finding
reached `violation` state during the measured period.

---

## 11. Left-hand / right-hand glove results (§6) — **LIMITATION**

**The system cannot do what §6 requires, and no amount of tuning will change that.**

The shipped policy declares **one** `hand_covering` attribute over one crop band, with the
domain `none | gloves | not_visible`. There is no place to record a second hand. The rule
reads `hand_covering eq gloves`.

The consequence is a real, currently unmeasured false-negative path. The policy question
says *"Answer 'gloves' ONLY if you can actually see a covering on a **hand**"* — singular.
A person with one gloved hand and one bare hand can legitimately be answered `gloves`, and
the rule then reads **COMPLIANT for both hands**.

Live condition outcomes across 200 incidents:

| `hand_covering` outcome | Count |
|---|---:|
| `unresolved` / `not_visible` | 179 |
| `failed` / `none` | 15 |
| `held` / `gloves` | 2 |
| `unresolved` / stale `gloves` or `none` | 4 |

89.5% of hand observations are refusals — hands in this kitchen are usually inside a pot,
behind a body, or out of frame.

**Not fixed, deliberately.** Closing it means changing the attribute domain, the question,
the crop geometry and the rule together, then re-measuring against annotated hands. That is
a policy change; §13 forbids inventing one and the standing constraints forbid modifying
production PPE policy or prompts. It is recorded as executable tests
(`TestPerHandStateIsNotRepresentable`) rather than a paragraph, so it cannot be forgotten
and any future per-hand work must come past it deliberately. See §26 for what it would take.

One guard is in place and is why this has not produced a flood of false compliance: the
question spends four clauses on when to refuse and one on when to answer, including
*"A visible forearm, sleeve or cuff is NOT a visible hand."*

---

## 12. MiniMax M3 inference results (§9, §18)

**Accuracy is not measurable on this account.** That is the finding, not a gap in the work.

| Measurement | Result |
|---|---|
| Listed by `/v1/models` | **Yes** |
| Live production, 50 min | **51 answered / 1,609 calls = 3.2%** |
| Controlled A/B, 3 orderings | **1 / 16** |
| Retry-with-backoff collection | **0 / 87 — every request 429** |
| Latency when it answers | 21.4s cold, ~2.3s warm |
| Correctness when it answers | correct on every inspected call; agreed with human ground truth 2/2 |

Per-attribute accuracy for head, face and each hand **cannot be reported**. Producing those
numbers requires answers, and 87 consecutive requests produced none. Publishing an accuracy
figure from the two successful calls earlier in the day would be fitting a metric to noise.

Per §18: a model that is accurate when it answers but answers ~0% of requests is **not
production-ready on this account**.

New MiniMax evidence is at `experiments/vlm_prompt/runs/minimax_m3_20260831.json`, in its
own file, never mixed with the Llama runs.

---

## 13. Rate-limit results

| Window | Requests | 429 | Answered |
|---|---:|---:|---:|
| Live, rate-limited (MiniMax) | 1,609 | 1,558 | 51 (3.2%) |
| Controlled A/B (MiniMax) | 16 | 15 | 1 |
| Controlled A/B (Llama) | 16 | **0** | **16** |
| Backoff collection (MiniMax) | 87 | 87 | **0** |
| Live, after fix (Llama) | 108 | **0** | **93 (86%)** |
| Live, after fix, sustained 80 min | 1,698 | **2** | **1,696 (99.9%)** |

The 429 rate **worsened over the day** under sustained retrying — from ~94% to 100% — which
is consistent with a quota that degrades rather than a transient spike, and is a further
argument against retrying harder.

---

## 14. Parser results (§10)

**No parse failure occurred at any point.** `understanding.unparseable` was 0 throughout.

MiniMax M3 returns fenced JSON, verbatim:

```
```json
{
  "head_covering": "hairnet",
  "face_covering": "not_visible",
  "hand_covering": "not_visible"
}
```
```

This is a **different shape from Llama's**, which answers the object body with no braces at
all. Both are handled. §10's requirement not to use one model's samples as evidence for
another is now enforced by a separate test file with verbatim MiniMax samples.

Tested and passing: valid JSON, fenced JSON, fence without language tag, surrounding prose,
leading/trailing whitespace, missing field, unknown field, empty, blank, prose-only,
unclosed, truncated, broken-fenced. **Every unusable shape returns `None` → refusal →
`structured={}`. None becomes a PPE value.**

---

## 15. Observability gate results (§11)

`PoseRegionObservability` claims `head_covering` and `face_covering` only. `hand_covering`
is deliberately **not** claimed — wrist keypoints locate a wrist, and the policy's own
wording is that a forearm is not a hand. Everything unclaimed returns `UNSUPPORTED` and
behaves exactly as before the adapter existed: **237 `unsupported` results, no suppression.**

The gate reports *where a head is*, never *what is on it*. There is no code path in the
adapter that can express a covering.

**The gate is not suppressing genuine violations.** The three post-fix incidents all passed
through it and reached `violation`.

---

## 16. Registry results (§12)

Attributes flow `VLM → parsed → synthesis → registry → rule` without loss. Live:
204 attributes applied during the rate-limited window, 315 after the fix. `head_covering =
none` arrives at the compliance engine as `none` — visible in every incident's condition
record. No second attribute cache exists.

`synthesis.skipped reason=refused` (2,415) is the correct behaviour: a refusal produces no
observation, so no attribute is written, so the rule sees `attribute_absent` → UNKNOWN.

---

## 17. Compliance rule results (§13)

The shipped rule, unmodified:

| Rule | Severity | Conditions |
|---|---|---|
| `kitchen.person.ppe.v1` | high | `head_covering ne none`, `hand_covering eq gloves`; `not_visible` → UNKNOWN |
| `kitchen.person.face_covering.v1` | informational | `face_covering eq mask`; `not_visible` → UNKNOWN |

Kleene three-valued, no short-circuit, any `false` → VIOLATION even beside an UNKNOWN. An
absent attribute is UNKNOWN, never `false`.

Verified live on a real incident — a bare head **and** unseen hands on the same person:

```
head_covering  ne none   observed=none         -> failed      → VIOLATION reported
hand_covering  eq gloves observed=not_visible  -> unresolved  → NOT reported
summary: "is not wearing a head covering"      ← says nothing about gloves
```

The real failure survived the unknown beside it, and the sentence never mentions the body
part nobody saw.

---

## 18. Alert pipeline results (§14)

**Never broken.** Before any change: 500 active incidents stored and served, spanning
2026-08-26 to 2026-08-31, on the exact query the Alerts page makes (`incidentsApi.list('active')`).

| Summary | Count (of 200 examined) |
|---|---:|
| is not wearing a head covering | 185 |
| is not wearing gloves | 14 |
| both | 1 |

Evidence blobs retrievable: `/api/v1/evidence/{ref}` → **200**, `/image` → **200**.

The user's report of "few or no alerts" is precisely accurate and precisely explained: the
page is not empty, but **no new alert had been produced since 09:52:16**, four minutes
before the model swap.

---

## 19. Live-camera validation (§16, §23)

Model switched to the servable one at 05:42:42Z; `--reload` respawned the worker
(pid 27800); nothing else touched.

| Elapsed | VLM ok | Refusals | Attributes | Applied | New incidents |
|---|---:|---:|---:|---:|---:|
| ~0 min | 5 | 0 | 15 | 6 | 0 |
| ~1 min | 19 | 0 | 57 | 66 | 0 |
| ~2 min | 46 | 0 | 138 | 159 | 0 |
| ~3 min | 66 | 0 | 198 | 231 | 0 |
| **~4 min** | **69** | **0** | **207** | **240** | **1** |
| **~80 min** | **1,696** | **2** | **5,088** | **5,268** | **13** |

Sustained over the following 80 minutes: **1,696 answered against 2 refusals — 99.9%** —
and violations continued to accrue, the most recent at 07:01:29Z.

**The first three new violations**, within five minutes of the switch, on two cameras:

| Observed (UTC) | Camera | Severity | Finding |
|---|---|---|---|
| 05:46:05 | cam-14 | high | head `none` → failed; hand `not_visible` → unresolved |
| 05:46:20 | cam-14 | high | head `none` → failed; hand `not_visible` → unresolved |
| 05:46:20 | cam-13 | high | head `none` → failed; hand `not_visible` → unresolved |

**§23 acceptance chain, demonstrated end to end on live kitchen CCTV:**

| Link | Evidence |
|---|---|
| Real person without hairnet | cam-14, live RTSP |
| person detected | YOLO, object `01M1B5MRENAFSWF414Z651GKPD` |
| tracking retained | track id equals object id |
| head observable | pose gate passed |
| useful head evidence sent | crop cut, quality gate passed, 448px head band |
| model called and answered | 0 refusals |
| **ABSENT** | `head_covering = none` |
| registry | attribute applied |
| compliance violation | `kitchen.person.ppe.v1`, `failed`, high |
| alert persisted | incident `1464dc2a264743c98e3a2ceb6dd4b33b` |
| dashboard | served under `status=active` |

**Mask chain:** not demonstrated — no unmasked-face violation occurred in the observation
window, and the rule is informational by design. **Glove chain:** not demonstrated at
per-hand granularity — see §11; it is not representable.

---

## 20. Controlled test results (§15)

| Case | Expected | Result |
|---|---|---|
| A — visible head, hairnet PRESENT | no violation | ✅ `COMPLIANT` |
| B — visible head, hairnet ABSENT | violation | ✅ `VIOLATION` |
| C — visible face, mask PRESENT | no violation | ✅ `COMPLIANT` |
| D — visible face, mask ABSENT | violation | ✅ `VIOLATION` |
| E — left glove ABSENT, right PRESENT | violation | ⚠️ **not representable** — §11 |
| F — left NOT_VISIBLE, right PRESENT | no fabrication, no both-compliant claim | ⚠️ **not representable** — §11 |
| G — VLM returns 429 | no PPE verdict, VLM DEGRADED | ✅ `UNKNOWN` + `state: rate_limited` |

Cases E and F are recorded as executable limitation tests asserting exactly what the system
does today, with the false-negative path documented in the test docstring.

---

## 21. Regression tests added

**+63 tests, 3,730 passing, 0 failing** (3,667 before this phase).

| File | Tests | Pins |
|---|---:|---|
| `tests/vision_os/understanding/unit/test_rate_limit_health.py` | 23 | 429 classification, sustained-failure reporting, no flapping, recovery without restart, refusal still a refusal |
| `tests/vision_os/understanding/unit/test_minimax_response_shapes.py` | 21 | verbatim MiniMax output, 6 degraded shapes, nothing unusable becomes a value |
| `tests/compliance/test_ppe_alert_chain.py` | 19 | §15 cases C, D, G; per-hand limitation; profile-face wording |

No existing test was weakened or deleted.

---

## 22. Historical evidence restored (§19)

A repo-wide find-and-replace (all files sharing mtime `09:56:35`) had rewritten
`meta/llama-3.2-11b-vision-instruct` → `minimaxai/minimax-m3` across files that record
*measurements*, not configuration. It was committed in `4bcbd7a`.

| File | Problem | Action |
|---|---|---|
| `tests/compliance/kitchen01_model_answers.json` | 43 answers recorded 2026-08-27, relabelled without re-recording. Header reads "This file is EVIDENCE, not a target." | restored |
| `experiments/vlm_prompt/runs/*.json` (13) | 12 variant runs executed 2026-08-27, four days before the swap; marked "never edited afterwards" | restored |
| `tests/compliance/test_dataset_regression.py` | docstring attribution | restored |
| `vision_os/adapters/understanding/payload.py` | attributes Llama's braceless-JSON quirk to MiniMax — and MiniMax demonstrably fences instead | restored |
| `tests/app/test_vlm_model_replacement.py:220` | `BRACELESS` sample labelled "Verbatim from minimax" | restored |

The `DEFAULT_MODEL` change in `nvidia_vl.py` was legitimate configuration and was left alone.

Verified: `datasets/p9-v1` and `p9-v2` digests unchanged at `fe16a44bc39e01e4`; dataset
guard **PASS**, 0 new CCTV exposure.

---

## 23. Final diagnostic table

| Stage | Working? | Evidence | Failure rate |
|---|---|---|---|
| Camera | ✅ | 4/4 online, ~24,600 frames each, 0 reconnects, 0 errors | 0% |
| Person detection | ✅ | 5,477 detections / 7,174 frames | 0.1% timeouts |
| Tracking | ✅ | 3,807 tracks → 3,619 registry objects | — |
| Head crop | ✅ | 448px band, calibrated 23.3%→74.4% | 18.9% gate (all stages) |
| Face crop | ✅ | shares head band, no extra call | as above |
| Left-hand crop | ⚠️ | **not separable** — one hand band | n/a |
| Right-hand crop | ⚠️ | **not separable** — one hand band | n/a |
| MiniMax inference | ❌ | 51/1,609 live; 0/87 with backoff | **96.8% → 100%** |
| Response parsing | ✅ | 0 unparseable; fenced JSON handled | 0% |
| Observability gate | ✅ | 237 `unsupported`, no suppression | 0% |
| Registry | ✅ | 204 applied during outage, 315 after | 0% |
| Compliance rule | ✅ | 200 incidents, correct Kleene outcomes | 0% |
| Alert persistence | ✅ | 500 active, evidence blobs 200 OK | 0% |
| Dashboard | ✅ | served on the page's own query | 0% |
| **VLM health reporting** | ✅ **fixed** | was `"ok"` at 3.2% success; now `rate_limited` | was 100% |

---

## 24. Remaining failure modes

1. **Per-hand PPE is not representable.** §11. A person with one bare hand can read
   COMPLIANT. Unmeasured in magnitude.
2. **Face/mask accuracy is unmeasured.** No annotated faces exist in this repository. The
   rule is informational for that reason and must not be promoted on hope.
3. **Hand quality floors are provisional** — 3 positive examples cannot support a threshold.
4. **`vision_os.cropping.budget_spent` can never move** — emitted as `.increment(0)` at
   `perception/cropping/runtime.py:409`. Cosmetic; it suppresses nothing. Reported rather
   than changed, because whether it is a placeholder series or a bug is the owner's call.
5. **`test_identity_fields_survive_reordering` is timing-flaky under load** — asserts exact
   completion order across 5 threads with 30 ms margins. Failed once during a full-suite run
   with 4 RTSP decoders and a probe running; passed 5/5 in isolation. Pre-existing, not
   weakened.
6. **A DB outage still surfaces as `500` on login**, not `503`. Carried from Pre-P9.9.

---

## 24a. Correction — what "answered" was counted as (2026-08-31, later)

The controlled A/B in §1 and §13 reports `meta/llama-3.2-11b-vision-instruct` at **16/16**.
That number counted **`refused == False`**, which is *reachability*, not usable output. An
unparseable reply is not a refusal — it returns `structured={}` with `refused=False` — so
those 16 conflate "the endpoint answered" with "an attribute was produced".

Re-measured later the same day over 20 kitchen-01 crops, counting **parsed** rather than
merely answered:

| Outcome | Count | |
|---|---:|---|
| Parsed → attribute produced | **12 / 20** | **60%** |
| Answered but unusable | 1 / 20 | 5% |
| Refused (transport, no 429) | 7 / 20 | 35% |

The model's output shape is **unstable across calls**. Observed on identical prompts:

```
"head_covering": "cap"      ← braceless, quoted     → parses (12/13)
"head_covering": none       ← braceless, unquoted   → UNPARSEABLE
**Head Covering:** none     ← markdown headings     → UNPARSEABLE
```

**What this does and does not change.**

It does **not** change the root cause. MiniMax at 0/87 was a total loss of evidence; that
finding stands and is unaffected by how the comparison model was scored.

It does **not** contradict the live post-fix measurement in §19. There, 1,696 engine-level
successes produced 5,088 attributes — exactly 3.00 per success — which is only possible if
those replies genuinely parsed. Live production splits head and hand into two calls with
narrower schemas; this experiment asks all three keys in one. The narrower production
request appears to yield the stable shape more often, but that is an observation from two
differently-shaped measurements, **not** a controlled comparison, and it has not been tested.

**Open question, not a conclusion:** whether the 60% here reflects prompt shape, endpoint
load at the time (a 120 s timeout was also seen on `llama-3.2-90b-vision-instruct` in the
same window), or genuine format instability that the live split-call path partly hides.
Resolving it needs a controlled run of the production two-call shape against the same crops.
Until then, no single accuracy figure for this model should be quoted as settled.

---

## 25. Model limitations

- **MiniMax M3 cannot be evaluated on this account.** Zero answers in 87 attempts. It is not
  wrong; it is unreachable.
- When it does answer it is well-formed and was correct on every inspected call — but two
  successes is not a measurement and no accuracy figure is published here.
- Cold latency 21.4s, warm ~2.3s; Llama measured ~4.3s mean.
- Neither model has measured per-attribute recall for face or hands on live footage.

---

## 26. Decisions for the owner

1. **Model.** The deployment currently runs `meta/llama-3.2-11b-vision-instruct` because
   that is what this account can serve and §23 could not otherwise be demonstrated. To go
   back to MiniMax you need serving quota for `minimaxai/minimax-m3` on the NVIDIA account —
   nothing in the code prevents it, and `health()` will now tell you immediately if the quota
   is insufficient. One line in `.env`.
2. **Per-hand gloves.** Requires a policy change: split `hand_covering` into left/right with
   their own crop bands and questions, extend the rule, and annotate hands on live footage to
   measure it. Roughly a phase of work, and it needs annotated hands first — kitchen-01 has 3.
3. **`--reload` is on in this deployment.** Convenient in development; a stray file save
   restarts analysis in production.

---

## 27. Verification summary

| Check | Result |
|---|---|
| Full backend suite | **3,730 passed, 0 failed** (was 3,667) |
| `tests/vision_os/understanding` | 379 passed |
| `tests/compliance` | 123 passed |
| `tests/app` | 461 passed |
| `tests/tools` (P9 dataset) | 220 passed |
| Ruff | 6 findings, all pre-existing in `migrations/` and `scripts/`; **0 in changed code** |
| P9 dataset guard | **PASS** — 0 new CCTV exposure |
| P9-v1 / P9-v2 digests | unchanged `fe16a44bc39e01e4` |
| Live chain, real CCTV | **3 new violations in 5 min; 13 over 80 min** |
| Sustained inference after fix | **1,696 answered / 2 refused (99.9%)** |

---

## 28. Statement

**ROOT CAUSE:**
The deployment was switched to `minimaxai/minimax-m3`, for which this NVIDIA account has
effectively no serving quota. The model is listed by `/v1/models`, so binding and `probe()`
both passed, and the account then returned HTTP 429 to 96.8% of inference calls — later
100%. Every PPE attribute for those crops was lost, so the compliance rules correctly
reported UNKNOWN rather than guessing, and no new alert could be raised. Detection,
tracking, cropping, parsing, the observability gate, the registry, the rules, persistence
and the dashboard were all measured working throughout. A secondary defect made this
invisible: `health()` reported `"ok"` because only a 404/410 retirement could mark the
analysis unavailable, and a 429 is not a retirement.

**FIX:**
1. `nvidia_vl.py` — 429 raised as `RateLimitedError`; a bounded 32-call outcome window;
   `health()` now reports `rate_limited` / `failing` on sustained failure with an
   operator-actionable reason; `rate_limited` counter added to the model panel. Thresholds
   measured, not chosen. Reporting only — refusal behaviour unchanged.
2. `.env` — `VISION_NVIDIA_MODE` typo corrected to `VISION_NVIDIA_MODEL` (the setting was
   never being read), and the model pointed at one this account can serve.
3. Historical Llama evidence restored across 17 files after a repo-wide relabelling.

**VERIFIED:**
3,730 tests passing (+63 new, none weakened). Ruff clean in changed code. P9 artefacts
intact, guard PASS. On live kitchen CCTV after the fix: VLM success **3.2% → 99.9%**
(1,696 answered against 2 refusals over 80 minutes), and **13 new high-severity
head-covering violations — the first three within five minutes**,
each traced from RTSP frame through detection, tracking, crop, inference, registry and rule
to a persisted incident with a retrievable evidence blob served on the dashboard's own
query. The degraded-health fix was confirmed live on production traffic before the model was
switched, reporting `rate_limited — only 0 of the last 32 crops were answered (0%)`.

**REMAINING LIMITATION:**
Left and right hands cannot be evaluated independently — the policy declares one
`hand_covering` attribute, so one gloved hand can read as compliant for both. Not fixed
because it requires a production policy change and annotated hand data that does not exist.
Face/mask accuracy is unmeasured and its rule is informational for that reason. MiniMax M3's
accuracy could not be measured at all: 0 answers in 87 attempts.

**PRODUCTION STATUS:**

**READY** — on `meta/llama-3.2-11b-vision-instruct`, for head-covering detection, which is
the chain demonstrated end to end on live cameras.

**NOT READY** — on `minimaxai/minimax-m3` until serving quota exists; and **not ready for
per-hand glove compliance on any model**, which is a policy limitation rather than a
model one.
