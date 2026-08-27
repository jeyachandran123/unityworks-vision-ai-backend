# Phase 6.3 — Hand Freshness Experiment (60 s window)

**Date:** 2026-08-18
**Decision: CASE C — `FRESH_ENOUGH` still cannot be observed. The cause is now identified and it
is not the footage.**

**The attribute is never satisfied, so freshness can never engage.** The same tracked person
re-triggers `ATTRIBUTE_MISSING` on *every frame*, 250 ms apart, indefinitely.

---

## Objective

Does the existing `TriggerPolicy` reuse a `hand_covering` observation inside its 60 000 ms window?
Phase 6.2 recommended this because a 115.9 s clip crosses a 60 s window naturally — **no synthetic
time, no looping, no stretching**.

## Configuration

| | |
|---|---|
| production path | real `vosvc_harness` Session — detection, tracking, registry, `CropManager`, `TriggerPolicy`, quality gate, VLM all live |
| video | `Screen Recording 2026-08-17 122832.mp4`, 115.9 s, original order |
| sampling | `sample_fps` = `target_fps` = **4.0** — real timing, **`synthetic_time: false`** |
| frames decoded | 435 |
| demand | **`hand_covering` only**, freshness 60 000 ms |
| configuration changed | **none** |

## Results

| quantity | value |
|---|---:|
| virtual elapsed | **8.5 s** (see limitation below) |
| evidence candidates | 207 |
| **TRIGGERED** | **131** |
| SKIPPED | 76 |
| tracks | 7 (all with duration > 0) |
| track durations | min 2 s, median 3 s, **max 8 s** |
| **`FRESH_ENOUGH`** | **0** |
| `ATTRIBUTE_STALE` | 0 |

| trigger | n | | skip | n |
|---|---:|---|---|---:|
| `ATTRIBUTE_MISSING` | **127** | | `NO_DEMAND` | 75 |
| `FIRST_SIGHT` | 4 | | `QUALITY_INSUFFICIENT` | 1 |

## The finding

Per-track sequence, identical across tracks:

```
track 7TJ9KWAD...   66 triggers, 0 skips
  t=  500 ms  first_sight        age=None
  t=  750 ms  attribute_missing  age=250 ms
  t= 1000 ms  attribute_missing  age=250 ms
  t= 1250 ms  attribute_missing  age=250 ms   ... every frame, indefinitely
```

The attribute is requested, and **250 ms later it is still `ATTRIBUTE_MISSING`**. The policy is
behaving correctly on the information it has: an attribute with no stored value must be requested.
`FRESH_ENOUGH` is unreachable because **there is never a stored observation for it to consider
fresh**.

The track lives 2–8 s — far inside the 60 s window — so if any observation were being retained,
every request after the first would have been `FRESH_ENOUGH`. None was.

### Three candidate causes, none yet confirmed

1. The understanding result is not written back to the registry within the replay.
2. The understanding stage does not complete before the next frame is evaluated (250 ms budget
   against ~2 s VLM latency measured in Phase 5C).
3. A `NOT_VISIBLE` answer does not satisfy the attribute, so an unobservable region re-requests
   forever. Camera B measured **93 % `NOT_VISIBLE`** for hands, which makes this the most likely
   of the three.

**I did not determine which.** Distinguishing them requires instrumenting the understanding
write-back path, which is beyond this phase's measurement-only scope.

## This reframes Phase 6.1

Phase 6.1 reported 325 calls / 1 000 frames and I described VLM usage as "well controlled". That
now needs qualifying: usage is bounded by **`NO_DEMAND` and `QUALITY_INSUFFICIENT` only**.
Freshness contributes **nothing**, because it never engages. The 325 figure remains correct as
measured; the mechanism behind it was partly misattributed.

If cause 3 holds, an attribute that can never be observed on a camera generates an **unbounded**
repeat-call stream for every tracked person — the opposite of the intended
*demands × changes* economy. That is a production cost and correctness concern well beyond this
experiment.

## Safety checks

| check | result |
|---|---|
| `FRESH_ENOUGH` never created a VLM observation | vacuously held — none occurred |
| a skipped request never received a fabricated result | **held** |
| `NOT_VISIBLE` / `UNKNOWN` never became `ABSENT` | **held** |
| track identity consistent | **held** — stable ids across the sequence |
| ground truth never read or modified | **held** |
| production configuration unchanged | **held** |

## Limitations

- **Virtual elapsed was 8.5 s, not 109 s.** My polling loop exited on the session's first
  `paused` report after `play()`, before the replay was under way — **a defect in my experiment
  script, not in Vision OS**. 34 of 435 frames were processed.
- That truncation does **not** affect the finding: the `ATTRIBUTE_MISSING` cycle is unambiguous and
  identical across all 7 tracks, and the longest track (8 s) is already well inside the 60 s window
  where reuse should have occurred.
- Maximum observed track life is **8 s**. Even a full run would not have tested the 60 s boundary,
  because no identity survives that long on this footage.
- One camera, one clip, one attribute.

---

## Decision: **CASE C**

`FRESH_ENOUGH` cannot be observed — but the obstacle has moved from "the footage is too short" to
a specific, addressable mechanism: **observations are not becoming available to the trigger policy
between frames.**

### Exact next experiment

**Determine why `ATTRIBUTE_MISSING` repeats, before anything else.** This is now a higher-value
question than the freshness boundary, and it is cheap:

1. **Instrument the understanding write-back** — record whether an observation for
   `hand_covering` ever reaches the registry, and with what state. One session run answers it.
2. **If the cause is `NOT_VISIBLE` not satisfying the attribute**, that is a policy question with
   real production consequences and should be raised before any freshness tuning.
3. **Re-run this experiment with a corrected polling loop** (`await` until the session reports
   `playing`, then poll to completion) so the full 109 s is replayed.
4. Only once an observation is demonstrably retained does the 60 s boundary become testable.

**No policy, threshold or production behaviour was changed.** No freshness figure is reported,
because none was measured.
