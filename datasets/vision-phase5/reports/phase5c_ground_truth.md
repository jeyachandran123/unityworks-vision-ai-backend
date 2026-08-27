# Phase 5C — Ground Truth and First Baseline on Genuine Violation Footage

**Date:** 2026-08-18
**Source:** `i_want_sec_video_of_this_vi.mp4` — 1280×720, 24 fps, 10.0 s, 240 frames, h264,
new by content hash. Commercial kitchen, timestamp `2026-08-17 14:50`.

**This is the first time in the programme that `ABSENT` precision has been measurable.**

---

## 1. The hypothesis, verified

> "Real CCTV of an in-scope food worker actually working without a hairnet."

Treated as a hypothesis and checked predicate by predicate against raw frames, **before any model
was run**:

| # | predicate | verdict | evidence |
|---|---|---|---|
| 1 | person | **YES** ×2 | two men, plainly visible |
| 2 | in-scope staff | **YES** | chef whites; one in a kitchen apron |
| 3 | working | **YES** | worker B handling food containers at a prep shelf; worker A at the range/wok station |
| 4 | food/prep activity | **YES** | commercial kitchen — range, wok, bulk food tubs, prep surfaces |
| 5 | head observable | **YES** | both heads visible in all 10 sampled frames |
| 6 | covering absent | **YES** | dark hair, clear hairline, no net texture, no cap edge |
| 7 | policy scope | **YES** | uniformed kitchen staff working in a kitchen — the scope the policy targets |

Unlike Phase 5A/5B, no predicate is unresolved. Those clips showed uniformed staff **seated in a
dining room, not handling food**, which left scope genuinely ambiguous. Here all seven hold.

**All 20 head crops (10 frames × 2 workers) show uncovered heads.** No shadow, transparency or
compression ambiguity of the kind that made the Camera C gloves misleading.

## 2. Sampling and ground truth

Every 24th frame — 1.0 fps, **10 frames across the full 10 s**, deterministic. Person boxes drawn
by hand from native frames; **no detector box was copied**, and no model output was rendered
during annotation.

| | |
|---|---:|
| frames | 10 |
| **person observations** | **20** |
| **distinct workers** | **2** |
| `head_covering` ABSENT | **20** |
| `hand_covering` NOT_VISIBLE | 20 |

Validation: **0 issues**.

`hand_covering` is `NOT_VISIBLE` throughout — hands are not resolvable at this distance and angle.
Correctly recorded as unobservable rather than guessed.

## 3. Person detection recall — measurable for the first time

Because ground truth was built independently of the detector, detection recall is finally a real
number rather than a tautology:

| | |
|---|---:|
| ground-truth people | 20 |
| detected and matched (IoU ≥ 0.5) | 19 |
| **missed by YOLOv8n** | **1** |
| detections outside ground truth | 1 |
| **person detection recall** | **95.0 %** |

The miss is `cam5c/0192 workerA`, best IoU 0.451 — a near-miss on the box, not an absent
detection. The person was **kept in ground truth**, as required.

## 4. Baseline: the current Vision OS, unchanged

Production configuration exactly as Phase 4.2 left it — YOLOv8n, 448 head evidence, 224 hand,
quality gate, blur estimator, VLM, prompt, semantic mapping, compliance. Nothing tuned for this
video.

### `head_covering` — MEASURED

| metric | value | n |
|---|---:|---:|
| **ABSENT precision** | **100.0 %** | 19/19 |
| **ABSENT recall** | **100.0 %** | 19/19 |
| **false violations** | **0** | — |
| **missed violations** | **0** | — |
| unsupported ABSENT | 0 | — |
| accuracy | 100.0 % | 19 |

Confusion matrix — every cell:

```
                 system
              ABSENT
truth ABSENT    19
```

**When a worker is genuinely uncovered and the head is observable, the current system identified
the violation in 19 of 19 observations, with no false alarms and nothing missed.**

### NOT MEASURABLE from this video

- `PRESENT` precision / recall — **0 PRESENT observations**. Every ground-truth label here is
  ABSENT, so the system's behaviour on compliant workers cannot be assessed from this clip.
- `NOT_VISIBLE` / `UNKNOWN` precision and recall — no head observations in those states.
- `hand_covering` — all 20 truths are `NOT_VISIBLE`.

## 5. The combined picture, stated carefully

This video measures violation *detection*. The kitchen-01 dataset measures behaviour on
*compliant* workers. Same system, same configuration, different footage:

| dataset | truth | system said ABSENT | outcome |
|---|---|---:|---|
| kitchen-01 (30 PRESENT heads) | compliant | **4** | 4 false violations |
| cam5c (19 ABSENT heads) | violating | **19** | 19 true violations |

Treating them as one population — legitimate only because the configuration is identical —
gives an **indicative ABSENT precision of 19 / 23 = 82.6 %**.

That figure carries real caveats and should not be quoted without them: it merges two cameras,
two scenes and two annotation sessions, and rests on 2 violating workers. It is the best estimate
available and it is not a production number.

## 6. VLM usage and latency — MEASURED

| | |
|---|---:|
| VLM calls | 40 |
| calls per person observation | 2.0 |
| calls per frame | 4.0 |
| calls avoided by gates | **0** |
| gate rejections | 0 |
| mean latency | **2 052 ms** |
| median latency | 1 637 ms |
| p95 latency | 3 209 ms |
| total wall clock | 83 s |

Latency is **6× better than the kitchen-01 runs** (mean 13–16 s, p95 60 s). The crops are the same
448 px; the difference is endpoint load, not configuration — which is worth recording, because the
alarming p95 numbers reported in Phase 4.2 were network variance rather than a property of the
system.

## 7. Failure attribution

**One failure, one category.**

| category | count | detail |
|---|---:|---|
| `DETECTION_FAILURE` | 1 | `cam5c/0192 workerA`, best IoU 0.451 — box drift below the matching threshold |
| VLM / semantic | **0** | every evidence crop it saw was read correctly |
| pose / quality gate | 0 | no rejections; no observability failures |
| compliance logic | 0 | — |

No failure required `UNKNOWN_FAILURE_REASON`.

## 8. Temporal behaviour — observed, not changed

The violation persists across all 10 sampled frames for both workers, and the system reported
`ABSENT` consistently in all 19 matched observations. **No single-frame flicker, no transitions,
no intermittent false negatives.** The one gap is the detection miss at frame 192, which appears
in a single frame and recovers.

Current tracking and temporal logic were **not modified and not evaluated further**; this is an
observation of the existing behaviour only.

## 9. Anti-circularity verification

| check | status |
|---|---|
| ground truth independent of YOLO / pose / VLM / compliance | **held** — annotation completed before any model ran |
| no model output shown during annotation | **held** — raw frames only |
| person boxes hand-drawn, not detector-copied | **held** |
| people annotatable even when the detector misses them | **held** — the missed person is in ground truth |
| every ABSENT reviewed at highest available resolution | **held** |
| ground truth unchanged after seeing model output | **held** — the system agreed with all 19; nothing was revised |

Had the system disagreed, the label would have stood pending independent re-inspection. It did
not disagree, so no revision question arose.

## 10. Sample-size honesty

**2 distinct workers. 20 observations. 1 camera. 1 kitchen. 10 seconds.**

19 matched observations of 2 people are **not** 19 independent samples. The correct reading of the
100 % is: *on two uncovered workers in one kitchen under one lighting condition, the system did not
make a single error.* That is genuinely good and genuinely narrow.

No claim of production readiness. No claim of generalisation.

---

## Decision: CASE B

**Useful, genuine violation evidence exists — and the dataset is still insufficient.**

| target | required | now | status |
|---|---:|---:|---|
| `head_covering` ABSENT | ≥20 | **20** | **met (observations)** |
| distinct workers in violation | ≥8 | **2** | **short** |
| `head_covering` PRESENT | ≥20 | 30 (kitchen-01) | met, different camera |
| distinct cameras with violations | ≥3 | **1** | short |
| distinct restaurants | ≥2 | 1 | short |

The observation target is met; the **worker and camera diversity targets are not**, and those are
what a precision figure actually depends on.

### The 10 questions

1. **Genuine ABSENT obtained?** **Yes** — first time in the programme.
2. **Distinct workers?** **2.**
3. **Valid ABSENT observations?** **20** annotated, 19 matched and evaluated.
4. **ABSENT precision measurable?** **Yes — 100.0 % (19/19)**, on this footage.
5. **ABSENT recall measurable?** **Yes — 100.0 % (19/19)**.
6. **Did YOLO detect all in-scope workers?** **No — 19/20, 95.0 % recall.**
7. **Did pose determine observability correctly?** Not exercised — no crop was rejected and no head
   was unobservable, so the gate had nothing to refuse.
8. **False violations from the current system?** **0** on this video. (4 remain on kitchen-01's
   compliant workers.)
9. **What component caused failures?** **The detector** — one box drift. Zero VLM failures.
10. **Phase 6?** Below.

### Recommended next experiment

**Collect violation footage from more workers and more cameras — not model work.**

The measurement just made is the strongest result this system has produced, and its weakness is
entirely sample diversity: 2 workers, 1 camera, 10 seconds. Another engineering change cannot fix
that, and optimising against a 2-worker dataset would be fitting to it.

Specifically, and in priority order:

1. **More violating workers across more cameras** — the ≥8-worker, ≥3-camera targets. Until then,
   82.6 % combined ABSENT precision rests on 2 violating individuals.
2. **Re-examine the 4 kitchen-01 false violations against this result.** They are the same 4
   subjects every run, and this video shows the VLM reading uncovered heads perfectly. That
   sharpens the question: those 4 are *covered* heads read as uncovered, so the failure is
   specific to recognising a covering, not to judging bare heads. That is a narrower and more
   tractable problem than previously framed.
3. **Do not start a specialist PPE model.** The VLM made zero semantic errors here. The evidence
   does not support replacing it.

**Nothing was fixed, tuned or trained in this phase**, per the measure-first workflow.
