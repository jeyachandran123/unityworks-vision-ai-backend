# Phase 5 — `hand_covering` Evaluation, Camera C

**Date:** 2026-08-18
**Policy:** operator confirmed gloves are **required** for this food-preparation task. Used only
to establish that the requirement applies — never as evidence about what any person is wearing.

## Headline: the workers are wearing gloves

**Camera C is hand-compliant. There are no hand violations to measure.**

The gloves are **clear disposable gloves**, and they are identifiable only at native resolution.
At thumbnail scale they read as bare skin, because the glove is transparent and the forearm above
it is bare.

### Correction to my own earlier read

In the previous inventory pass I reported that camera C showed "hands and forearms of uniform bare
skin tone, no glove edge or cuff line visible" and flagged them as likely violations. **That was
wrong.** Inspecting the same footage at native resolution shows the glove cuff at the wrist and
the crinkled translucent material over the hand, clearly, in multiple frames.

This is the same failure that produced the head-covering false violations in Phase 4.2 — a PPE
item misread as absent because it was viewed at insufficient resolution — reproduced here in a
human observer rather than the VLM. It is a useful independent confirmation of the resolution
finding, and it is exactly why rule 5 requires the absence to be *visually clear* before `ABSENT`
is written.

Had this been annotated from the thumbnail impression, the dataset would have contained roughly a
dozen fabricated violations, and every subsequent violation-precision number would have been
measuring that mistake.

---

## Method

| | |
|---|---|
| source | `media/Screen Recording 2026-08-17 122553.mp4` (1714×966, 30 fps, 49.8 s) |
| sampling | every 75th frame — 0.4 fps, 20 frames, deterministic and reproducible |
| review resolution | 857×560 native-pixel crops of the work zone, **no downscaling** |
| ground truth | human visual inspection only |
| models consulted | **none** — no YOLO boxes, pose, VLM or compliance output was shown or used |

Person boxes were drawn by hand, not copied from any detector.

---

## Ground truth

| `hand_covering` | count |
|---|---:|
| **PRESENT** | **3** |
| **ABSENT** | **0** |
| NOT_VISIBLE | 1 |
| UNKNOWN | 12 |
| person observations | 16 |
| **distinct persons** | **1** |

Validation: **0 issues**.

The 12 `UNKNOWN` are honest. Those frames were reviewed only in reduced-resolution batches, where
the glove state cannot be determined confidently. Rule 7 requires `UNKNOWN` there, and rule 11's
identity persistence does not license inferring a frame's label from its neighbours — the worker
could have removed a glove between samples. They are recorded as annotator uncertainty, not as a
system target.

---

## What is measurable

| metric | result |
|---|---|
| `hand_covering` **ABSENT precision** | **NOT MEASURABLE** — 0 ground-truth ABSENT |
| `hand_covering` **ABSENT recall** | **NOT MEASURABLE** — 0 ground-truth ABSENT |
| `hand_covering` PRESENT precision | **NOT MEASURABLE** — 3 observations, 1 person |
| `hand_covering` PRESENT recall | **NOT MEASURABLE** — 3 observations, 1 person |
| NOT_VISIBLE precision / recall | **NOT MEASURABLE** — 1 observation |
| UNKNOWN precision / recall | **NOT MEASURABLE** |
| false violation rate | **NOT MEASURABLE** |
| unsupported ABSENT | **NOT MEASURABLE** |
| person detection recall | **NOT MEASURABLE** — single-person scene |

**Nothing requested is measurable from this camera.**

## The pipeline run was not executed

Deliberately. With 0 ground-truth `ABSENT` and 3 decided `PRESENT` observations from one person,
every metric in the request would be either undefined or computed over n ≤ 3. A run would emit an
`ABSENT` precision figure describing an empty denominator and a `PRESENT` precision over three
samples — numbers that describe this sample, not the system, and that would be quoted later as
though they described the system.

The harness is built, tested and ready. It needs violation footage, not a run.

---

## Why this outcome was still worth the work

Three things changed:

1. **`hand_covering` PRESENT is no longer zero.** Before this phase the attribute had *never* been
   observed in its positive state anywhere in the repository. There are now 3 confirmed
   glove-positive observations. Small, but it is the first evidence that the attribute is
   observable at all on some camera.
2. **A camera that can see hands exists.** Cameras A and B measured 93 % `NOT_VISIBLE` for hands.
   Camera C sees them closely. Future hand data collection should use this mounting position.
3. **A near-miss was caught.** The clear-glove misread would have injected ~12 fabricated
   violations into the first dataset ever built to measure violation precision.

---

## Status: CASE B — insufficient, with specifics

| target | required | camera C | overall |
|---|---:|---:|---|
| `hand_covering` PRESENT | ≥20 | **3** | 3 |
| `hand_covering` **ABSENT** | ≥20 | **0** | 3 (camera A, Phase 3) |
| `head_covering` PRESENT | ≥20 | available | ~30 |
| `head_covering` **ABSENT** | ≥20 | **0** | **0** |
| distinct workers in violation | ≥8 | **0** | **0** |

**Still DATA REQUIRED, and the requirement is now sharper:** all three cameras show compliant
kitchens. 226 seconds of footage across three angles contains **zero head violations and zero hand
violations**.

The missing input is not more footage of this restaurant. It is footage of **non-compliance**,
which by definition will not be found in a kitchen that is complying. Options, in order of
evidential value:

1. Footage from a site or shift with known compliance problems, collected as normal operation.
2. Footage from a period *before* a PPE policy was enforced, if any is retained.
3. If neither exists, a deliberately staged session — **labelled as staged, reported separately,
   and never merged into the headline number**, since posed footage differs systematically in pose,
   framing and duration.

---

## Not a production-readiness claim

One restaurant, one camera, one worker, three confirmed observations. This report supports no
statement about how Vision OS performs on hand compliance, in this kitchen or any other.
