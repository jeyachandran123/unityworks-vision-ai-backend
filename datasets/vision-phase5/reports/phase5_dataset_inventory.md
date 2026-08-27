# Phase 5 — Video Inventory (`backend/media/`)

**Date:** 2026-08-18
**Source of truth:** `backend/media/` — three videos, 226.3 s total, all distinct (verified by
content hash, not filename).

---

## 1. Inventory

| # | file | resolution | fps | duration | frames | camera |
|---|---|---|---:|---:|---:|---|
| 1 | `Screen Recording 2026-08-13 112749.mp4` | 1712×1032 | 30 | 60.6 s | 1817 | kitchen A — wide, cooking range |
| 2 | `Screen Recording 2026-08-17 122832.mp4` | 1718×978 | 30 | 115.9 s | 3476 | kitchen B — wide, pot station |
| 3 | `Screen Recording 2026-08-17 122553.mp4` | 1714×966 | 30 | 49.8 s | 1495 | **kitchen C — close-up prep bench** |

Machine-readable: `phase5_dataset_inventory.json`.

Video 1 is the clip used in Phases 3–4.4. Videos 2 and 3 are **new to Phase 5**. Video 3 had
never been inspected before this phase.

All three are genuine CCTV (fixed overhead mounts, timestamp-free, continuous). All show
restaurant kitchen work.

---

## 2. PPE observability by camera

| camera | workers visible | heads observable | **hands observable** | usable for `head_covering` | usable for `hand_covering` |
|---|---|---|---|---|---|
| A (video 1) | 2–5 per frame | frequently | **rarely** (93 % `NOT_VISIBLE` measured) | yes | no |
| B (video 2) | 1–4 per frame | frequently | rarely | yes | poor |
| C (video 3) | 1–2 per frame | frequently | **frequently and closely** | yes | **yes** |

**Camera C is the significant find.** It is a close-up overhead bench view where a worker handles
food with hands filling a large part of the frame — the first footage in this repository where
`hand_covering` is inspectable at all. Cameras A and B look down at torsos, which is why hand
annotation there was 93 % `NOT_VISIBLE`.

---

## 3. Head coverings: searched thoroughly, none absent

`head_covering = ABSENT` was searched for deliberately and at increasing magnification:

- **Camera A** — 43 subjects fully annotated in Phase 3. **0 ABSENT.**
- **Camera B** — 62 head regions sampled across all 116 s, plus 8 full frames, plus targeted
  zooms. **0 ABSENT.**
- **Camera C** — 21 head regions plus 8 full frames. **0 ABSENT.**

One candidate was pursued and rejected: a figure at camera B frame ~1780 appears dark-headed at
thumbnail scale. Zoomed to native resolution across frames 1700–1900 it is **plainly wearing a
blue hairnet** — the dark appearance was shadow at small scale.

That negative result is worth keeping. It is the same failure mode the VLM shows at 224 px,
reproduced in a human observer: **at low resolution a shadowed hairnet reads as a bare head.** It
is evidence for the Phase 4.2 resolution finding from an independent direction.

**Conclusion: all three kitchens are head-compliant. `head_covering` ABSENT precision remains
unmeasurable, and no amount of annotating this footage will change that.**

---

## 4. Hand coverings: camera C contains apparent violations

At native resolution on camera C, the worker handles food (dough/buns) with **hands and forearms
of uniform bare skin tone, no glove edge or cuff line visible**, in direct food contact, across
many frames spanning the clip.

If the site policy requires gloves for food contact, these are **genuine `hand_covering = ABSENT`
observations occurring in normal work** — exactly the material Phase 5 has been blocked on.

**This is not yet ground truth.** Two things must happen before it is:

1. **Confirm the policy.** `ABSENT` means the PPE that *should* be there is not. If this site does
   not require gloves for this task, bare hands are compliant and the correct label is `ABSENT`
   only against a rule that exists. This is a question for the operator, not the annotator.
2. **Confirm bare skin against a flesh-toned or clear glove** at native resolution, per person per
   frame. At this camera's resolution the forearm/hand tone match is strong evidence but not
   proof, and the honest label where it is not clear is `UNKNOWN`, not `ABSENT`.

Neither has been done. **No hand labels have been written.**

---

## 5. Status

| target | required | available | status |
|---|---:|---|---|
| `head_covering` PRESENT | ≥20 | ~30 (camera A) + more available | met |
| `head_covering` **ABSENT** | ≥20 | **0** | **DATA REQUIRED** |
| `hand_covering` PRESENT (gloves) | ≥20 | **0 observed** | **DATA REQUIRED** |
| `hand_covering` **ABSENT** | ≥20 | **candidates on camera C** | **pending annotation** |
| distinct workers | ≥8 | ~6–8 across three cameras | plausible |
| distinct cameras | ≥3 | **3** | **met** |
| distinct restaurants | ≥2 | 1–2 (A/B may be one site) | uncertain |

**Case B applies, not Case C.** The head attribute is still unmeasurable, but camera C makes the
hand attribute potentially measurable in one direction for the first time.

Note the asymmetry: even with camera C fully annotated, `hand_covering` would have many `ABSENT`
and **zero `PRESENT`** — the mirror of the head problem. Violation *precision* would become
measurable; violation *recall* would not, and a system that answered `ABSENT` unconditionally
would score perfectly on precision. Both directions are needed before the attribute is genuinely
evaluable.
