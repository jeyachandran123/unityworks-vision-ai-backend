"""P9.6 Phase 1 — why wall-clock sampling failed, measured from the P9.5 corpus.

    python -m tools.p9_dataset.baseline --write

This module makes no recommendation. It reads the 520 frames P9.5 actually
collected and reports what a fixed-interval timer bought for them, so the design
of the event sampler answers a measurement rather than an intuition.

### The measurement that matters

Duplicate *rate* was already published in P9.5. What it could not say is **why** —
whether the kitchen is genuinely static or the interval merely landed badly. The
distribution of consecutive-frame Hamming distance answers that: a static scene
piles up at zero, an active scene spreads out, and a scene that does both means
one interval cannot serve it, because the number has to be simultaneously too
slow for the activity and far too fast for the stillness.

That shape is the whole argument for event-aware sampling, so it is computed here
rather than asserted in the report.

### Information yield

The honest denominator is not "frames kept" but "frames decoded". A collector
that decodes 40,654 frames and produces 304 person candidates has an information
yield of well under one percent, and every intermediate ratio that looks
healthier than that is measuring a stage rather than the pipeline.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from .dedupe import (
    DEFAULT_THRESHOLD,
    MAX_ADJACENT_GAP,
    FrameHash,
    find_duplicates,
    frame_key,
    hamming,
    load_frames,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "p9-live"
OUT = LIVE / "sampling_baseline.json"


def _sessions() -> dict:
    out = {}
    for directory in sorted(LIVE.glob("live-*")):
        record = directory / "session.json"
        if record.exists():
            payload = json.loads(record.read_text(encoding="utf-8"))
            out[payload["session_id"]] = payload
    return out


def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    zero = sum(1 for v in ordered if v == 0)
    return {
        "n": len(ordered),
        "zero": zero,
        "zero_rate": round(zero / len(ordered), 4),
        "at_or_below_threshold": sum(1 for v in ordered if v <= DEFAULT_THRESHOLD),
        "median": statistics.median(ordered),
        "p25": ordered[len(ordered) // 4],
        "p75": ordered[3 * len(ordered) // 4],
        "p90": ordered[min(len(ordered) - 1, 9 * len(ordered) // 10)],
        "max": ordered[-1],
    }


def consecutive_distances(frames: list[FrameHash]) -> dict:
    """Hamming distance between each frame and its immediate predecessor.

    Grouped by camera and session, because a distance measured across a session
    boundary is not a measure of anything. The percentiles are the reading that
    sets the event sampler's change threshold — taken from this distribution
    rather than from a round number that looks plausible.
    """
    per_camera: dict[str, list[int]] = collections.defaultdict(list)
    grouped: dict[tuple[str, str], list[FrameHash]] = collections.defaultdict(list)
    for frame in frames:
        grouped[(frame.session_id, frame.camera_id)].append(frame)

    for (_, camera_id), group in grouped.items():
        group.sort(key=lambda f: f.order)
        for previous, current in zip(group, group[1:], strict=False):
            per_camera[camera_id].append(hamming(previous.bits, current.bits))

    everything = [v for values in per_camera.values() for v in values]
    return {
        "overall": describe(everything),
        "by_camera": {c: describe(v) for c, v in sorted(per_camera.items())},
        "histogram": dict(sorted(collections.Counter(everything).items())),
    }


def report() -> dict:
    frames = load_frames()
    if not frames:
        return {"frames_sampled": 0, "note": "no live frames collected yet"}

    sessions = _sessions()
    queue_path = LIVE / "review_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
    candidates = queue.get("candidates", [])

    duplicates = find_duplicates(frames)
    identical = find_duplicates(frames, threshold=0)
    retained = [f for f in frames if frame_key(f.path) not in duplicates]
    decoded = sum(s["totals"]["frames_decoded"] for s in sessions.values())

    by_camera: dict[str, dict] = {}
    for frame in frames:
        entry = by_camera.setdefault(
            frame.camera_id,
            {"sampled": 0, "bit_identical": 0, "near_duplicate": 0, "retained": 0},
        )
        entry["sampled"] += 1
        key = frame_key(frame.path)
        entry["bit_identical"] += key in identical
        if key in duplicates:
            entry["near_duplicate"] += 1
        else:
            entry["retained"] += 1

    for session in sessions.values():
        for camera in session["cameras"]:
            entry = by_camera.setdefault(camera["camera_id"], {})
            entry["decoded"] = entry.get("decoded", 0) + camera["frames_decoded"]
            entry["seconds"] = round(entry.get("seconds", 0.0) + camera["seconds"], 1)

    for candidate in candidates:
        entry = by_camera.setdefault(candidate["camera_id"], {})
        entry["candidates"] = entry.get("candidates", 0) + 1

    for entry in by_camera.values():
        entry.setdefault("candidates", 0)
        minutes = entry.get("seconds", 0.0) / 60.0
        entry["candidates_per_camera_minute"] = (
            round(entry["candidates"] / minutes, 2) if minutes else None
        )
        entry["retained_per_decoded"] = (
            round(entry["retained"] / entry["decoded"], 5) if entry.get("decoded") else None
        )

    by_session: dict[str, dict] = {}
    for session_id, session in sessions.items():
        sampled = [f for f in frames if f.session_id == session_id]
        kept = [f for f in sampled if frame_key(f.path) not in duplicates]
        found = sum(1 for c in candidates if c["session_id"] == session_id)
        by_session[session_id] = {
            "collected_at": session["collected_at"],
            "interval_seconds": session["sampling"]["interval_seconds"],
            "window_seconds": session["sampling"]["window_seconds"],
            "decoded": session["totals"]["frames_decoded"],
            "sampled": len(sampled),
            "retained": len(kept),
            "candidates": found,
            "candidates_per_retained": round(found / len(kept), 3) if kept else None,
        }

    by_interval: dict[str, dict] = {}
    for entry in by_session.values():
        key = f"{entry['interval_seconds']:g}s"
        bucket = by_interval.setdefault(
            key, {"sessions": 0, "sampled": 0, "retained": 0, "candidates": 0}
        )
        bucket["sessions"] += 1
        bucket["sampled"] += entry["sampled"]
        bucket["retained"] += entry["retained"]
        bucket["candidates"] += entry["candidates"]
    for bucket in by_interval.values():
        bucket["duplicate_rate"] = (
            round(1 - bucket["retained"] / bucket["sampled"], 4) if bucket["sampled"] else None
        )

    frames_with_people = {c["frame_id"] for c in candidates}
    person_free = len(retained) - len(frames_with_people)

    return {
        "_comment": [
            "P9.6 Phase 1. A measurement of the P9.5 wall-clock corpus, not a",
            "recommendation. The consecutive-distance histogram is the argument:",
            "a fixed interval cannot serve a scene that is static for long",
            "stretches and then briefly busy, because one number has to be both",
            "too slow for the activity and far too fast for the stillness.",
        ],
        "corpus": "datasets/p9-live (P9.5, wall-clock sampling)",
        "threshold": DEFAULT_THRESHOLD,
        "max_adjacent_gap": MAX_ADJACENT_GAP,
        "frames_decoded": decoded,
        "frames_sampled": len(frames),
        "bit_identical": len(identical),
        "bit_identical_rate": round(len(identical) / len(frames), 4),
        "near_duplicates": len(duplicates),
        "near_duplicate_rate": round(len(duplicates) / len(frames), 4),
        "retained": len(retained),
        "retained_with_person": len(frames_with_people),
        "retained_person_free": person_free,
        "person_free_rate": round(person_free / len(retained), 4) if retained else None,
        "candidates": len(candidates),
        "information_yield": {
            "_comment": "candidates per DECODED frame is the pipeline's true rate",
            "sampled_per_decoded": round(len(frames) / decoded, 5) if decoded else None,
            "retained_per_decoded": round(len(retained) / decoded, 5) if decoded else None,
            "candidates_per_decoded": round(len(candidates) / decoded, 5) if decoded else None,
            "candidates_per_sampled": round(len(candidates) / len(frames), 4),
            "wasted_sampled_frames": len(duplicates) + person_free,
            "wasted_fraction_of_sampled": round(
                (len(duplicates) + person_free) / len(frames), 4
            ),
        },
        "consecutive_distance": consecutive_distances(frames),
        "by_camera": dict(sorted(by_camera.items())),
        "by_session": by_session,
        "by_interval": dict(sorted(by_interval.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = report()
    if not result.get("frames_sampled"):
        print("no live frames found under", LIVE)
        return 1

    print(f"decoded            : {result['frames_decoded']:,}")
    print(f"sampled            : {result['frames_sampled']}")
    print(f"bit-identical      : {result['bit_identical']} ({result['bit_identical_rate']:.1%})")
    print(f"near-duplicate     : {result['near_duplicates']} ({result['near_duplicate_rate']:.1%})")
    print(f"retained           : {result['retained']}")
    print(f"person-free        : {result['retained_person_free']} ({result['person_free_rate']:.1%})")
    print(f"candidates         : {result['candidates']}")
    yields = result["information_yield"]
    print(f"candidates/decoded : {yields['candidates_per_decoded']:.5f}")
    print(f"wasted of sampled  : {yields['wasted_fraction_of_sampled']:.1%}")

    print("\nconsecutive-frame Hamming distance:")
    overall = result["consecutive_distance"]["overall"]
    print(
        f"  n={overall['n']}  zero={overall['zero']} ({overall['zero_rate']:.1%})  "
        f"median={overall['median']}  p75={overall['p75']}  "
        f"p90={overall['p90']}  max={overall['max']}"
    )
    for camera, entry in result["consecutive_distance"]["by_camera"].items():
        print(
            f"  {camera}  n={entry['n']:3d}  zero={entry['zero_rate']:6.1%}  "
            f"median={entry['median']:5.1f}  p75={entry['p75']:3d}  "
            f"p90={entry['p90']:3d}  max={entry['max']:3d}"
        )

    print("\nby camera:")
    for camera, entry in result["by_camera"].items():
        print(
            f"  {camera}  decoded={entry.get('decoded', 0):6d} sampled={entry['sampled']:4d} "
            f"retained={entry['retained']:4d} candidates={entry['candidates']:4d} "
            f"per-minute={entry['candidates_per_camera_minute']}"
        )

    print("\nby interval:")
    for interval, entry in result["by_interval"].items():
        print(
            f"  {interval:>4s}  sessions={entry['sessions']} sampled={entry['sampled']:4d} "
            f"retained={entry['retained']:4d} "
            f"duplicate_rate={entry['duplicate_rate']:.1%} "
            f"candidates={entry['candidates']}"
        )

    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
