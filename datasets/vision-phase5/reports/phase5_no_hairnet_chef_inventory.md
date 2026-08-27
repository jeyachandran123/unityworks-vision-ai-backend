# Phase 5B — Inventory

## The new file

| field | value |
|---|---|
| filename | `i_want_sec_video_not_a_im.mp4` |
| sha256 | `1047d864b36dde0803801df7773311f5825561c0850878aadca0ad54914fa7dd` |
| resolution | **1280 × 720** |
| fps | 24.0 |
| duration | 10.0 s |
| frames | 240 |
| codec | h264 |
| size | 2.2 MB |

Genuinely new by content hash.

## Two things to flag before anything else

**1. The camera D file from Phase 5A is gone.** `Screen Recording 2026-08-18 103417.mp4`
(1714×970, 36.6 s) is no longer in `backend/media/`. It has been replaced, not supplemented.

**2. This file is lower resolution than the footage it appears to replace.** 1280×720 against
1714×970 — about 55 % of the pixel count, and re-encoded at 24 fps rather than 30. It shows what
looks like the same dining room.

That matters directly for this phase: Part 6 mandates native-resolution verification, and the
Camera C glove finding showed PPE can look absent at reduced resolution while being present at
native. Here the *available* native resolution is itself lower than what was previously supplied
for this room. Any ABSENT label from this file rests on weaker source evidence than the deleted
file could have supported.

The filename also reads like a message rather than a capture identifier, which may indicate the
upload was not the intended one.

## All files currently in `backend/media/`

| filename | resolution | duration | status |
|---|---|---:|---|
| `i_want_sec_video_not_a_im.mp4` | 1280×720 | 10.0 s | **NEW** |
| `Screen Recording 2026-08-13 112749.mp4` | 1712×1032 | 60.6 s | known |
| `Screen Recording 2026-08-17 122553.mp4` | 1714×966 | 49.8 s | known |
| `Screen Recording 2026-08-17 122832.mp4` | 1718×978 | 115.9 s | known |

## Scene

Same dining room as Phase 5A camera D: wooden tables, wall art, pendant lights, beer-tap station
and kegs at the left, burnt-in timestamp. Continuous CCTV, single fixed mount, no cuts.
