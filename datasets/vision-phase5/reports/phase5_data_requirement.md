# Phase 5 — DATA REQUIRED

**Status: no PPE violation footage exists in this repository.** The dataset schema,
validator, annotation workflow and evaluation harness are built and tested. They are waiting on
footage, and no labels have been invented to fill the gap.

---

## What was searched

Every video in the repository, inspected frame-by-frame:

| file | resolution | duration | content | usable? |
|---|---|---|---|---|
| `4fa5c3ab-Screen Recording 2026-08-13` | 1712×1032 | 60.6 s | **kitchen CCTV, camera 1** | partly — 0 violations |
| `f72107ff-Screen Recording 2026-08-17` | 1718×978 | 115.9 s | **kitchen CCTV, camera 2** | partly — 0 violations |
| `4137e8cc-WhatsApp 2026-08-12` | 848×478 | 13.4 s | office interior, desks | no |
| `ca850784-WhatsApp 2026-08-12` | 478×850 | 13.8 s | close-up of a pen and notebook | no |
| `2cbb14ab-mixkit-worried-and-sad-woman` | 1280×720 | 15.0 s | stock: woman outdoors | no |
| `4c8a7ef7-mixkit-street-with-people` | 1280×720 | 19.0 s | stock: street at dusk | no |
| `898322a6-mixkit-couple-dance-floor` | 1920×1080 | 14.4 s | stock: dance floor | no |
| `514f9c3b-2980656-uhd` | 3840×2160 | 20.9 s | stock | no |
| `76e9a984-19613728-uhd` | 3840×2160 | 53.3 s | stock | no |

Camera 2 was newly discovered in this phase and is genuinely useful — a second angle, different
lighting and layout, which the split policy needs. It was scanned densely: **62 head regions
sampled across all 116 seconds. Every worker wears a blue hairnet.**

## The gap

| attribute | PRESENT | **ABSENT** | NOT_VISIBLE | can measure violation precision? |
|---|---:|---:|---:|---|
| `head_covering` | 30 | **0** | 13 | **no** |
| `hand_covering` | 0 | **3** | 40 | **no** — 3 is not a rate |

Both cameras show a compliant kitchen. That is good operational news and useless evaluation
material: **a dataset with no violations cannot measure violation precision.** Every `ABSENT` the
system emits is wrong by construction, so the number is undefined rather than poor.

## What is required

Footage in which **the PPE is genuinely absent while the relevant body region is clearly
visible**. Not staged mimicry of a violation — real working conditions in which someone is not
wearing the equipment.

### Minimum, per attribute

| need | count | why |
|---|---:|---|
| `head_covering` **ABSENT**, head clearly visible | **≥ 20** | the floor below which one annotation error moves the rate by more than a few points |
| `hand_covering` **ABSENT**, hands clearly visible | ≥ 20 | current dataset has 3 |
| `hand_covering` **PRESENT** (gloves visible) | ≥ 20 | currently **zero**; the attribute has never been seen in its positive state |
| distinct people in violation | ≥ 8 | 20 frames of one person measures one person |
| distinct cameras | ≥ 3 | two are available; a third breaks the tie when they disagree |
| distinct restaurants | ≥ 2 | the split policy prefers restaurant-level isolation |

### Conditions to cover

The failures already measured are concentrated in particular conditions, so the footage should
contain them rather than only clean examples:

- workers **bent over** counters and pots — the pose where the head leaves the detector box
- workers **turned away** from the camera
- **distant** workers (≈150 px person height) — where the 4 known semantic failures live
- **close** workers (≈700 px) — where the head sits mid-box rather than at the top
- **motion blur** from fast work
- **multiple overlapping** workers
- **hands visible** at all — the current 93 % NOT_VISIBLE rate means this camera cannot support
  hand compliance regardless of PPE
- varied lighting, including the dawn/dusk exposure range

### What must not happen

Do not stage violations by asking a compliant worker to remove a hairnet for the camera. Posed
footage differs systematically from real footage in pose, framing and duration, and a precision
figure measured on it would not transfer. If staged footage is the only option, it must be
**labelled as staged and reported separately**, never merged into the headline number.

---

## What is ready now

```
backend/datasets/vision-phase5/
    manifest.json     attributes, split policy, status
    schema.json       states, observability, the ABSENT invariant
    annotations/      empty — awaiting footage
    reports/          this file, and the quality report
```

```
python -m tools.vision_eval.annotate_ppe extract <video> --dataset datasets/vision-phase5 \
    --restaurant r1 --camera cam-1 --fps 0.5
python -m tools.vision_eval.annotate_ppe review   --dataset datasets/vision-phase5
python -m tools.vision_eval.annotate_ppe validate --dataset datasets/vision-phase5
```

The validator refuses to write an annotation asserting `ABSENT` for a region marked unobservable,
and the quality report prints `INSUFFICIENT DATA` with the shortfall rather than a percentage.
Both were verified against the empty dataset.
