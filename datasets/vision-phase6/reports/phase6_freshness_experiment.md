## Replay methodology (Part B) — and exactly what is synthetic about it

The production `Session` drives a **`VirtualClock`**, advanced once per replayed frame by
`1000 / target_fps` milliseconds (`session.py:505`). Simulated elapsed time is therefore

    frames x (1000 / target_fps) ms

and it is controllable **without altering a single pixel**.

| | |
|---|---|
| video | `Screen Recording 2026-08-17 122832.mp4` — 115.9 s, the longest available |
| `sample_fps` | 2.0 — frames taken from the container |
| `target_fps` | 1.0 — replay rate, so the clock advances 1000 ms per frame |
| effect | ~232 frames x 1000 ms ≈ **232 s simulated**, against a 120 s head window and a 60 s hand window |

**What is real:** every frame, in original order, unmodified. No looping, no repetition, no
edited or synthesised imagery.

**What is synthetic:** the *implied interval between frames*. The clip really spans 115.9 s; the
session treats it as ~232 s. Part B permits exactly this for call-behaviour testing, and it is the
only way to cross a 120 s window with 115.9 s of footage.

**Therefore:** these results are valid for **call/skip behaviour only**. No accuracy, precision or
recall claim may be derived from this run, because the timeline it measures is not the timeline the
footage was captured on. Ground truth was neither read nor modified.

---

# Phase 6.2 — Freshness / Temporal Reuse Experiment

**Date:** 2026-08-18
**Result: BOUNDARY NOT MEASURABLE. `FRESH_ENOUGH` was not observed, and the reason is structural
rather than a policy fault.**

## Objective

Does the existing `TriggerPolicy` reuse a prior observation (`FRESH_ENOUGH`) while it is inside the
configured validity window, and stop reusing it (`ATTRIBUTE_STALE`) once it expires?

## Production path used

Identical to Phase 6.1: real `vosvc_harness` `Session` — detection, tracking, registry,
`CropManager`, `TriggerPolicy`, quality gate and VLM all live. `tools/vision_eval/predict.py` was
**not** used. Instrumentation wrapped the existing `crop_sink` and forwarded every call unchanged.
Demand registered through the real API for **both** `head_covering` and `hand_covering`, so the
120 s and 60 s windows could be characterised independently.

## Runs attempted

| # | sample_fps | target_fps | frames | simulated span | outcome |
|---|---:|---:|---:|---:|---|
| 1 | 2.00 | 1.00 | 232 | 232 s | killed at the 900 s wall-clock limit before reporting |
| 2 | 0.45 | 0.25 | 52 | 208 s | step-driven; my drain-wait logic was wrong — **script fault**, 0 candidates |
| 3 | 0.45 | 0.25 | 52 | **436 s virtual** | play-driven, completed — **0 triggers, 0 skips, 0 tracks** |

Run 3 completed cleanly: the demand was `ACCEPTED` and 436 s of virtual time elapsed. Yet **nothing
reached the crop sink at all** — not even a skip.

## Why: sampling rate and tracking are coupled

| phase | sample_fps | one frame every | tracking | candidates |
|---|---:|---|---|---:|
| 6.1 | 4.00 | 0.25 s of video | associates | **13 triggers** |
| 6.2 | 0.45 | 2.2 s of video | cannot associate | **0** |

At 0.45 fps a worker moves too far between frames for the tracker to link detections, so the
registry never confirms an object, so no crop candidate is ever evaluated. **Freshness cannot be
tested on objects the registry never confirmed.**

This produces a hard tension on the available footage — the longest clip is 115.9 s against a 120 s
head window:

```
sample_fps 4.00 / target_fps 4.00 -> 463 frames -> 116 s simulated  (never crosses 120 s)
sample_fps 2.00 / target_fps 1.00 -> 231 frames -> 231 s simulated  (wall-clock >= 231 s)
sample_fps 0.45 / target_fps 0.25 ->  52 frames -> 208 s simulated  (tracking breaks)
```

Crossing the window requires either **dense sampling and a long wall-clock run**, or **sparse
sampling that destroys the identity freshness depends on**. Neither was achievable here.

## Measured results

| quantity | value |
|---|---|
| `FRESH_ENOUGH` count | **0 — not observed** |
| `ATTRIBUTE_STALE` count | 0 — not observed |
| `APPEARANCE_CHANGED` | 0 — **NOT OBSERVABLE** |
| `QUALITY_IMPROVED` | 0 — **NOT OBSERVABLE** |
| stable tracks | **0** in run 3 |
| observation ages at reuse | **no data** |
| validity-boundary behaviour | **BOUNDARY NOT MEASURABLE** |
| hand freshness (60 s) | **HAND FRESHNESS NOT OBSERVABLE** |
| PPE state transitions | not observed |

**No freshness figure is reported, because none was measured.**

## Safety checks

| check | result |
|---|---|
| `FRESH_ENOUGH` never created a VLM observation | vacuously held — none occurred |
| a skipped request never produced a fabricated result | **held** |
| `NOT_VISIBLE` / `UNKNOWN` never became `ABSENT` | **held** |
| no ground-truth annotation modified | **held** — none was read or written |
| no VLM answer used to alter ground truth | **held** |
| production configuration unchanged | **held** — no `validity_ms`, policy or threshold touched |

## Limitations

- Longest available clip is **115.9 s** against a **120 s** head window. The footage is
  structurally too short.
- Reaching 120 s of simulated time needs a wall-clock run longer than a single foreground
  execution allowed here.
- Run 2 failed on my own step-drain logic; that is a harness-script defect, not a system finding,
  and is recorded as such.

---

## Decision: **CASE C**

**Freshness still cannot be reliably exercised.** Stable identity could not be maintained at the
sampling rate required to cross the validity window on the available footage.

This is **not** evidence that freshness is broken. It is evidence that the available footage cannot
test it. Phase 6.1 already showed the surrounding machinery working — `NO_DEMAND` suppressed 1 554
candidates and `QUALITY_INSUFFICIENT` saved 3 calls.

### Exactly what is missing

**Continuous footage longer than 150 s in which one worker remains trackable throughout**, replayed
at `sample_fps` ≥ 2 so tracking holds. Three routes, cheapest first:

1. **Capture > 3 minutes of continuous CCTV with a stationary worker** — the clean answer, and the
   only one that also makes PPE state transitions (Part I) observable.
2. **Run the existing 115.9 s clip at `sample_fps` 4 / `target_fps` 1** — 463 frames, 463 s
   simulated, tracking intact. Needs a **background run of roughly 8-10 minutes wall-clock**, which
   is entirely feasible; it simply exceeded the budget available here.
3. **Measure `hand_covering` alone** on the same clip: its window is **60 s**, which a 115.9 s clip
   crosses naturally at `target_fps` = `sample_fps` — no time stretching required at all. This is
   the cheapest genuine test of the mechanism and should be tried first.

Route 3 is the recommended next step: it needs no new footage, no synthetic time, and no policy
change, and it would produce the first real `FRESH_ENOUGH` observation.

**No policy, threshold, or production behaviour was changed in this phase.**
