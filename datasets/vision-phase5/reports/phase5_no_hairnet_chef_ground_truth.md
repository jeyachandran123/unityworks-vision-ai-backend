# Phase 5B — Ground Truth: `i_want_sec_video_not_a_im.mp4`

**Date:** 2026-08-18
**Verification result: the bare heads are real. The "working" part of the hypothesis is not
supported.**
**Status: no labels written — one predicate remains unresolved, and it is not a visual one.**

---

## 1. What the hypothesis claimed

> "This video contains an in-scope chef working without a hairnet."

Treated as a hypothesis and checked against the pixels, it splits into three claims. Two are
confirmed; one is contradicted.

| claim | verdict |
|---|---|
| heads are uncovered | **CONFIRMED** at native resolution |
| the people are staff | **CONFIRMED** — uniformed |
| they are *working* / handling food | **CONTRADICTED** — they are seated at dining tables |

---

## 2. What is visible

Native-resolution inspection (1280×720, no downscaling) of frames sampled across the 10 s clip.

**Uniformed staff, bare-headed.** At least four people in white staff shirts, several with
lanyards, seated around dining tables. Their hair is plainly visible — dark, uncovered, no
hairnet, cap, or net of any kind. **This is unambiguous at native resolution**, unlike the Camera C
gloves, which needed magnification to resolve. Nothing here is a transparency artefact.

**Civilians, bare-headed.** A man in a dark button shirt with glasses and a man in a red polo,
seated at adjacent tables — the same two individuals present in the Phase 5A camera D footage.

**Nobody in the frame is handling food.** There is no food, no prep surface, no service action.
The staff are seated at dining tables in a dining room, hands on the table, apparently in a
meeting or on a break.

### This differs from Phase 5A

In the camera D footage (`Screen Recording 2026-08-18 103417.mp4`, since deleted), the uniformed
staff in this same room **were wearing blue hairnets**. In this clip they are not. Same room, same
civilians present, different head state.

That is a real and interesting difference. It is also why the labels cannot be written yet.

---

## 3. The unresolved predicate

Part 5 permits `ABSENT` only when **the person is in scope** and **the required head covering is
clearly absent**. The second is satisfied. The first is not, and it cannot be settled from pixels:

> **Does the head-covering requirement apply to uniformed staff seated in the dining area, not
> handling food?**

Both readings are coherent:

- **Requirement attaches to food handling** — staff on a break or in a briefing are not in scope,
  removing a hairnet at the table is normal, and this clip contains **zero** in-scope violations.
- **Requirement attaches to being on duty in uniform anywhere on the premises** — these are
  **genuine in-scope ABSENT observations**, and the first real violation evidence in this
  repository.

The instruction "do not label customers or unrelated people as PPE violations" (Part 3) is
precisely what makes this decisive: the whole question is whether seated, non-handling staff are
"unrelated" for this attribute at this moment.

I raised this same scope question in Phase 5A. It has not yet been answered, and the answer is the
only thing standing between this footage and a validated dataset.

---

## 4. Ground truth

```
head_covering (in-scope observations):
    PRESENT      : 0 written
    ABSENT       : 0 written
    NOT_VISIBLE  : 0 written
    UNKNOWN      : 0 written

uniformed staff observed, bare-headed : ~4 distinct people
civilians observed, bare-headed       : 2 distinct people
people observed handling food         : 0
```

No annotation file was written, so validation had nothing to run against. Frames are extracted to
`review/frames_chef/` and annotation can start the moment the scope question is answered.

---

## 5. Evidence-quality caveat

This file is **1280×720**, about 55 % of the pixel count of the camera D file it replaced
(1714×970), re-encoded at 24 fps. Part 6 requires ABSENT candidates be confirmed "at the highest
available source resolution" — and the highest resolution *previously available for this room* was
higher than what is now on disk.

For these particular observations it does not change the reading: the uncovered heads are obvious
even at 720p. But if this footage becomes the basis of the violation dataset, the original
higher-resolution capture should be restored, because subtler cases on this camera will need it.

---

## 6. Anti-circularity status

| check | status |
|---|---|
| no annotation from YOLO / pose / VLM / compliance | **held** — none was run on this video |
| no model output shown during inspection | **held** — raw frames only, no boxes or skeletons |
| every visible person annotatable regardless of detector | **held** |
| ABSENT candidates reviewed at native resolution | **held** |

Vision OS was **not run**. Part 12 gates the evaluation on validated ground truth, and there is
none yet.

---

## 7. Classification

**Pending the scope answer**, exactly as in Phase 5A:

- **staff-on-break are out of scope** → **CASE C**, DATA REQUIRED. Four videos, ~273 s, four
  cameras, zero in-scope violations.
- **uniformed staff are always in scope** → **CASE B**: roughly 4 distinct bare-headed staff in a
  10-second clip. Genuine, clearly observable, and the first such evidence here — but far short of
  the ≥20-observation / ≥8-worker targets, from one camera, one room, one moment, with nobody
  working. A violation-precision figure from it would describe a staff meeting, not a kitchen.

---

## 8. What I would ask for

If uniformed staff are in scope regardless of activity, say so and I will annotate this
immediately — the visual evidence is solid.

But the more useful footage, and what the phase title implies, is a **chef at a prep or cooking
station, working, without a hairnet**. That is the observation Vision OS actually has to get
right, and it is what cameras A, B and C would capture. None of the four videos currently contains
it.

---

## 9. Confirmations

- **Production Vision OS not modified** — YOLOv8n, pose unwired, 448/224, quality gate, blur
  estimator, VLM, prompt, semantic mapping, compliance, tracking all untouched.
- **Vision OS evaluation not run.**
- **No code changes** — existing schema and validator remain adequate.
- **No labels fabricated**, in either direction.
- **No model trained.**
