# Phase 5D — Decision: the first false violations measured against real ground truth

**Date:** 2026-08-18
**Headline: ABSENT precision fell from 100 % to 60 % the moment the dataset contained workers who
were actually wearing head coverings — and the failure has a single, specific cause.**

---

## 1. Video inventory

Two new files, both confirmed new by content hash:

| filename | resolution | fps | duration | frames | codec |
|---|---|---:|---:|---:|---|
| `give_next_seconds_video_of.mp4` | 1280×720 | 24 | 10.0 s | 240 | h264 |
| `i_want_sec_video_of_this_vi (1).mp4` | 1280×720 | 24 | 10.0 s | 240 | h264 |

Both are **the same camera** — a commercial kitchen with gas ranges, large sauce vessels, a bulk
container shelf. Timestamps `2026-08-13 13:46:57` and `13:56:47`, ten minutes apart. Machine
inventory: `phase5d_video_inventory.json`.

**This is a new camera and a new kitchen**, distinct from the Phase 5C kitchen and from cameras
A/B/C. Filenames again read as messages rather than capture identifiers.

## 2. Why this footage matters: it contains both states

Every previous dataset had only one decided state per camera. This one has **PRESENT and ABSENT
workers side by side, in the same frame, under identical lighting and camera geometry.**

That is the discrimination test, and nothing before it could run.

| | |
|---|---:|
| frames annotated | 2 |
| person observations | 7 |
| **distinct workers** | **4** |
| `head_covering` ABSENT | 4 |
| `head_covering` **PRESENT** | **2** |
| `head_covering` NOT_VISIBLE | 1 |

Validation: **0 issues**. Boxes hand-drawn from native 1280×720 frames; no detector output
consulted, no model shown during annotation.

Only the two frames verified at full native resolution were annotated. The remaining sampled
frames were left unannotated rather than labelled from reduced-resolution crops.

## 3. Baseline: the current Vision OS, unchanged

| metric | value | n |
|---|---:|---:|
| **ABSENT precision** | **60.0 %** | 3 TP / 5 predicted |
| ABSENT recall | 100.0 % | 3/3 |
| **PRESENT recall** | **0.0 %** | 0/2 |
| PRESENT precision | n/a — never predicted | 0 predicted |
| NOT_VISIBLE precision / recall | 100 % / 100 % | 1/1 |
| **FALSE VIOLATIONS** | **2** | — |
| missed violations | 0 | — |
| unsupported ABSENT | 0 | — |
| detection recall | 85.7 % | 6/7 |
| VLM calls | 14 | — |
| latency mean / median / p95 | 2 264 / 1 813 / 4 132 ms | — |

Confusion:

```
                    system
              ABSENT   NOT_VISIBLE
truth ABSENT      3          0
truth PRESENT     2          0        <- both false violations
truth NOT_VIS     0          1
```

## 4. The cause: dark head coverings

**Both false violations are the same worker — w3, wearing a dark cap — in both frames.**

The system read a dark head covering as a bare head. It did not fail on the bare heads (3/3
correct) and it did not fail on the unobservable head (correctly refused). It failed specifically
and only on the **dark covering**.

This connects directly to a failure that has persisted since Phase 4.2. Of the four kitchen-01
subjects that produce a false violation in *every* run, one is annotated
**"dark knitted cap, close to the lens"**. The pattern now has two independent confirmations from
two different kitchens:

| covering type | evidence | result |
|---|---|---|
| blue hairnet | kitchen-01, 30 subjects | mostly correct |
| **dark cap / bandana** | kitchen-01 + cam5d | **read as ABSENT → false violation** |
| bare head | cam5c 19/19, cam5d 3/3 | **correct** |

**The system is good at recognising a bare head and good at recognising a light hairnet. It
mistakes a dark covering for hair.** That is a narrow, concrete, testable bottleneck — not "the
VLM is unreliable".

## 5. Comparison with Phase 5C

| | Phase 5C | Phase 5D |
|---|---:|---:|
| observations | 20 (19 matched) | 7 (6 matched) |
| distinct workers | 2 | **4** |
| cameras | 1 | 1 (**new**) |
| ground-truth PRESENT | **0** | **2** |
| ABSENT precision | **100 %** | **60 %** |
| false violations | 0 | **2** |
| detection recall | 95.0 % | 85.7 % |

**The 5C figure of 100 % was not wrong — it was unfalsifiable.** With no PRESENT observations in
that footage, a system that answered `ABSENT` unconditionally would have scored identically. I
flagged that at the time; this video is the proof, and it arrived within one phase.

Pooled across both violation videos: ABSENT TP = 22, FP = 2 → **91.7 % pooled ABSENT precision**
over 6 workers and 2 cameras. Adding kitchen-01's 4 persistent false violations, which use the
same configuration, gives 22 / 28 = **78.6 %**. Both figures are indicative, not production
numbers.

## 6. Diversity contribution: **HIGH VALUE**

| dimension | contribution |
|---|---|
| new violating workers | **+3** (w1, w2 bare; plus w3 covered) |
| new camera | **+1** |
| new restaurant/site | **+1** (distinct kitchen) |
| **PRESENT + ABSENT in one scene** | **first time** |
| new conditions | multi-worker, occlusion, truncated head, mixed PPE, varied distance |

This is the first footage that could *disprove* anything, and it immediately did.

## 7. Limitations — stated plainly

**2 frames. 7 observations. 4 workers. 1 camera. 2 PRESENT examples.**

The 60 % precision rests on **two** false violations from **one** worker. That is not a rate; it is
an existence proof that the failure mode is real. Equally, 2 PRESENT examples cannot establish
PRESENT recall — 0.0 % means "it got both wrong", not "it never works".

Annotation covers only the two natively-verified frames; the other sampled frames were left
unannotated rather than guessed.

No production-readiness claim. No generalisation claim.

## 8. Decision: **CASE B**

Real, high-value violation evidence exists. Diversity is still insufficient for a production
figure.

| target | required | now | status |
|---|---:|---:|---|
| ABSENT observations | ≥20 | 24 (5C+5D) | met |
| **distinct violating workers** | ≥8 | **6** | short |
| **PRESENT observations with violations present** | ≥20 | **2** | **far short** |
| distinct cameras with violations | ≥3 | **2** | short |
| distinct restaurants | ≥2 | **2** | **met** |

### Recommended next phase

**Collect footage containing dark head coverings, then re-measure. Do not change the model yet.**

The bottleneck is now specific enough to name, which it has never been before. But it rests on
one worker in two frames, and optimising against that would be fitting to a single person's cap.

In priority order:

1. **Footage of workers in dark caps, bandanas and dark hairnets** — enough to establish whether
   the failure is systematic across dark coverings or specific to this style. This is the single
   highest-value collection target, and it is much narrower than "more violation footage".
2. **More PRESENT observations generally** — 2 is far too few to measure PRESENT recall at all.
3. **Then** decide between a prompt change and a specialist covering classifier — with a dataset
   that can actually distinguish the two hypotheses.

**Do not start a specialist PPE model yet.** The VLM reads bare heads perfectly (22/22 across two
kitchens) and light hairnets well. A model replacement aimed at a failure measured on one cap
would be premature.

---

## 9. Confirmations

Production Vision OS **unmodified and unchanged**: YOLOv8n, pose as configured, head 448, hand 224,
quality floors, blur estimator, VLM, prompt, semantic mapping, compliance. Nothing trained, nothing
tuned, no prompt touched. Ground truth was written before any model ran, and **was not revised
after seeing the system's answers** — the two disagreements stand as model failures, which is what
they are.
