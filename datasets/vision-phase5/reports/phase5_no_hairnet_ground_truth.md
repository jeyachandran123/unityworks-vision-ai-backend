# Phase 5A — Ground Truth: `Screen Recording 2026-08-18 103417.mp4`

**Date:** 2026-08-18
**Status: BLOCKED ON A POLICY QUESTION — no labels written.**

There are bare heads in this video. Whether they are *violations* depends on a scope decision I
cannot make from pixels, and writing labels either way would fabricate the exact metric Phase 5
exists to measure.

---

## Video

| field | value |
|---|---|
| filename | `Screen Recording 2026-08-18 103417.mp4` |
| sha256 | `1a14bed30ab97a30381a11cc1f087d87b749deff5ecff78031b402e89650aa8d` |
| resolution | 1714 × 970 |
| fps | 30 |
| duration | 36.6 s |
| frames | 1098 |
| codec | h264 |

Confirmed new by content hash. **Camera D — a customer dining room**, not a kitchen: wooden
tables and chairs, wall art, pendant lighting, a beer-tap station and kegs. Burnt-in timestamp
`2026-08-18 13:03:43`, which the kitchen cameras lack.

## Sampling

Every 40th frame — 0.75 fps, **28 frames** covering the full 36.6 s. Deterministic and
reproducible. Frames written to `review/frames_nohairnet/`. Two frames were then examined at full
1714×970 native resolution (200 and 880, 22 seconds apart) per the mandatory native-resolution
rule.

---

## What is actually in the frame

The scene is stable across the clip. Two clearly distinct groups of people:

**Group 1 — uniformed staff (white shirts).** Six or so individuals seated and standing. Every one
of them has a head covering:

- four or more in **blue hairnets**
- one in a **black cap**

**Group 2 — people in civilian clothing, bare-headed.** Seated at dining tables:

- a man in a dark button shirt with glasses
- a man in a red polo shirt
- (a further person crosses the room with a backpack, wearing a dark cap)

**Within the uniformed staff, I found no uncovered head.** The bare heads all belong to people in
ordinary clothes, sitting at dining tables, in a dining room, at 13:03.

---

## The question that blocks annotation

> **Does the head-covering requirement apply to non-uniformed people in the dining area?**

Both answers are defensible from the footage, and they produce opposite datasets:

**If the policy covers food handlers only** — the bare-headed individuals are customers, or
managers off the line. They are **out of scope**, this video contains **zero** `head_covering`
ABSENT observations, and the ~28 staff observations are all `PRESENT`.

**If the policy covers everyone on the premises** — those individuals are in scope and their heads
are plainly, natively visible. That would yield the **first genuine ABSENT examples in this
repository**, from at least 2–3 distinct people.

The visual evidence is identical under both readings. Only the rule differs.

### Why I did not simply pick one

This is the same shape as the camera C glove question, and there I was nearly wrong in the other
direction. Labelling seated customers as PPE violations would inject fabricated violations into
the first dataset ever built to measure violation precision — and *every* subsequent number would
be measuring that decision rather than the system. Labelling them out of scope, if the site
actually requires coverings throughout, would discard the only real violation evidence found so
far.

Rule 6 permits `ABSENT` only when "the required head covering is not present" — which presupposes
that the covering is required *for that person*. That predicate is unresolved, so the label is
unresolved.

---

## Ground-truth summary

```
head_covering:
    PRESENT      : 0 written
    ABSENT       : 0 written
    NOT_VISIBLE  : 0 written
    UNKNOWN      : 0 written

distinct people : ~8 observed, 0 annotated
```

**No annotations were written.** The dataset file for camera D does not exist, so validation has
nothing to validate — reporting "VALID DATASET" over an empty file would be meaningless.

Nothing here is fabricated, and nothing is discarded: the frames are extracted and on disk, and
annotation can begin immediately once the scope question is answered.

---

## Anti-circularity status

| # | check | status |
|---|---|---|
| 1 | no annotation from YOLO | **held** — detector never run on this video |
| 2 | no annotation from pose | **held** — pose never run on this video |
| 3 | no annotation from VLM | **held** |
| 4 | no annotation from compliance output | **held** |
| 5 | every ABSENT has native-resolution evidence | n/a — none written |
| 6 | every NOT_VISIBLE genuinely unobservable | n/a |
| 7 | every PRESENT visually identifiable | n/a |
| 8 | every UNKNOWN has a reason | n/a |
| 9 | any visible person annotatable regardless of detector | **held** — no model proposals used |
| 10 | no model output shown during annotation | **held** |

Inspection was done on raw extracted frames only. No box, skeleton or model answer was rendered.

---

## Classification: pending

Not CASE A, B or C yet — the case is determined by the policy answer:

- **policy covers food handlers only** → **CASE C**, DATA REQUIRED. All four videos, 263 s across
  four cameras, contain zero in-scope head violations.
- **policy covers all persons present** → likely **CASE B**: real ABSENT observations from 2–3
  people in one 36-second clip on one camera. Enough to make ABSENT precision *measurable* for the
  first time, still short of the ≥20-observations / ≥8-people target, and from a single camera and
  scene — so no generalisation claim would be available.

## Limitations regardless of the answer

One restaurant, one camera, one 36-second clip, one scene, ~8 people. Even in the best case this
supports no statement about generalisation across cameras, lighting or restaurants.

---

## Confirmations

- **Production Vision OS was not modified.** Detector still YOLOv8n, pose unwired, head 448, hand
  224, quality gate, blur estimator, VLM, prompt, semantic mapping, compliance and tracking all
  untouched.
- **Vision OS evaluation was not run.** No detection, pose, crop, VLM or compliance was executed
  against this video.
- **No code changes were made** in this phase. Existing Phase 5 schema and validator are adequate;
  no extension proved necessary.
- **No labels fabricated in either direction.**
