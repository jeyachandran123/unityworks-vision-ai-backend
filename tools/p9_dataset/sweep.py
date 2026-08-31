"""Phase 4 — choose the sampling parameters by measurement, on recorded footage.

    python -m tools.p9_dataset.sweep --build-cache
    python -m tools.p9_dataset.sweep --write

### Why recorded footage and not live cameras

A parameter comparison needs the *same* input for every setting. A live stream
gives each configuration a different kitchen, so any difference between two runs
is confounded with whatever happened to be going on — which is precisely the
mistake §7 of the baseline caught wall-clock sampling making, where two sessions
at an identical interval differed sixfold in yield.

So the sweep runs against 226 seconds of real production recordings. The result
transfers because the sampler consumes hashes and boxes, not pixels: the policy
under test is identical, only its input is replayed.

### Decode once, detect once, sweep many times

The perception cache holds `(hash, boxes)` per sampled frame. Building it runs
the detector; sweeping does not. That makes forty configurations cost about as
much as one, and it makes the comparison **exactly** reproducible — a rerun that
disagreed would be a bug, not noise.

The cache is sampled at 6 fps of video time, which bounds `detect_every_seconds`
from below: a configuration cannot be evaluated at a detection rate finer than
the cache holds, and the sweep refuses rather than silently rounding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .dedupe import (
    DEFAULT_THRESHOLD,
    MAX_ADJACENT_GAP,
    FrameHash,
    Redundancy,
    audit_rescues,
    classification_summary,
    classify,
    hamming,
)
from .events import EventSampler, SamplingConfig, SamplingReason

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT.parent
OUT = ROOT / "datasets" / "p9-sweep"
CACHE = OUT / "perception_cache.json"
RESULT = OUT / "parameter_sweep.json"

#: Frames of video time between cached observations. 5 at 30 fps is 6 Hz.
CACHE_STRIDE = 5

#: Real production recordings, with the camera each carries in its overlay.
CLIPS = (
    {
        "clip_id": "rec-20260813",
        "camera_id": "cam-unknown",
        "path": ATLAS / "media" / "Screen Recording 2026-08-13 112749.mp4",
    },
    {
        "clip_id": "rec-20260817-a",
        "camera_id": "cam-11",
        "path": ATLAS / "media" / "Screen Recording 2026-08-17 122553.mp4",
    },
    {
        "clip_id": "rec-20260817-b",
        "camera_id": "cam-13",
        "path": ATLAS / "media" / "Screen Recording 2026-08-17 122832.mp4",
    },
)


def build_cache() -> dict:
    """Decode each clip once, hash and detect, and store the perception.

    The detector is the production one, bound exactly as `candidates.py` binds
    it, so the sweep measures the sampler's policy rather than a second
    detector's idiosyncrasies.
    """
    import numpy as np
    from PIL import Image

    from .candidates import _detector, propose_people

    detector = _detector()
    clips = []
    for clip in CLIPS:
        path = clip["path"]
        if not path.exists():
            continue
        import av

        observations = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 30)
            for index, frame in enumerate(container.decode(stream)):
                if index % CACHE_STRIDE:
                    continue
                image = frame.to_image().convert("RGB")
                small = np.asarray(
                    image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16
                )
                bits = 0
                for bit in (small[:, 1:] > small[:, :-1]).flatten():
                    bits = (bits << 1) | int(bit)
                observations.append(
                    {
                        "frame_index": index,
                        "timestamp": round(index / fps, 4),
                        "hash": bits,
                        "boxes": [
                            [list(box), round(score, 4)]
                            for box, score in propose_people(detector, image)
                        ],
                    }
                )
        clips.append(
            {
                "clip_id": clip["clip_id"],
                "camera_id": clip["camera_id"],
                "source": path.name,
                "fps": fps,
                "stride": CACHE_STRIDE,
                "observations": observations,
            }
        )
        print(
            f"  {clip['clip_id']:16s} {len(observations):4d} observations, "
            f"{sum(len(o['boxes']) for o in observations):4d} boxes"
        )

    payload = {
        "_comment": [
            "Perception cache for the P9.6 parameter sweep. Hashes and PERSON",
            "boxes only — no PPE signal of any kind is recorded or consulted.",
            "Built once so every configuration sees identical input; a sweep",
            "against live streams would confound the parameter with the kitchen.",
        ],
        "stride": CACHE_STRIDE,
        "clips": clips,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def _accepted_duplicate_rate(hashes: list[int]) -> tuple[int, float | None]:
    """Near-duplicates *among the frames the sampler kept*.

    The same rule `dedupe.py` applies to a corpus — similar **and** adjacent —
    so the number here is comparable with the 55.8 % the baseline measured for
    wall-clock sampling.
    """
    duplicates = 0
    kept: list[tuple[int, int]] = []
    for order, bits in enumerate(hashes):
        recent = [(o, h) for o, h in kept if order - o <= MAX_ADJACENT_GAP]
        if any(hamming(bits, h) <= DEFAULT_THRESHOLD for _, h in recent):
            duplicates += 1
        else:
            kept.append((order, bits))
    rate = round(duplicates / len(hashes), 4) if hashes else None
    return duplicates, rate


def evaluate(config: SamplingConfig, cache: dict) -> dict:
    """Replay every clip through one configuration."""
    stride_seconds = None
    totals = {
        "offered": 0,
        "accepted": 0,
        "person_frames": 0,
        "candidates": 0,
        "baseline": 0,
        "tracks": 0,
    }
    reasons: dict[str, int] = {}
    suppressed: dict[str, int] = {}
    hashes: list[int] = []
    kept: list[FrameHash] = []
    per_clip = {}

    for clip in cache["clips"]:
        fps = clip["fps"]
        stride_seconds = clip["stride"] / fps
        if config.detect_every_seconds < stride_seconds - 1e-9:
            raise ValueError(
                f"detect_every_seconds={config.detect_every_seconds} is finer than "
                f"the cache stride ({stride_seconds:.4f}s); rebuild the cache with a "
                f"smaller stride rather than evaluating a rate it cannot represent"
            )
        observations = clip["observations"]
        index = {o["frame_index"]: o for o in observations}
        sampler = EventSampler(
            config,
            hash_of=lambda key: index[key]["hash"],
            detect=lambda key: [
                (tuple(box), score) for box, score in index[key]["boxes"]
            ],
        )
        clip_hashes: list[int] = []
        clip_accepted = 0
        clip_people = 0
        clip_candidates = 0
        for observation in observations:
            decision = sampler.offer(
                observation["frame_index"],
                observation["timestamp"],
                observation["frame_index"],
            )
            if not decision.accepted:
                continue
            clip_accepted += 1
            clip_hashes.append(observation["hash"])
            kept.append(
                FrameHash(
                    path=Path(f"{clip['clip_id']}/{observation['frame_index']:06d}.jpg"),
                    camera_id=clip["camera_id"],
                    session_id=clip["clip_id"],
                    order=len(clip_hashes) - 1,
                    bits=observation["hash"],
                    reason=decision.reason.value if decision.reason else "",
                    people=decision.people,
                )
            )
            boxes = len(observation["boxes"])
            clip_candidates += boxes
            clip_people += boxes > 0

        statistics = sampler.statistics()
        for reason, count in statistics["by_reason"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        for key, count in statistics["suppressed"].items():
            suppressed[key] = suppressed.get(key, 0) + count

        totals["offered"] += statistics["frames_offered"]
        totals["accepted"] += clip_accepted
        totals["person_frames"] += clip_people
        totals["candidates"] += clip_candidates
        totals["baseline"] += statistics["baseline_triggered"]
        totals["tracks"] += statistics["candidate_subject_tracks"]
        hashes.extend(clip_hashes)
        per_clip[clip["clip_id"]] = {
            "offered": statistics["frames_offered"],
            "accepted": clip_accepted,
            "candidates": clip_candidates,
            "duplicate_rate": _accepted_duplicate_rate(clip_hashes)[1],
        }

    duplicates, duplicate_rate = _accepted_duplicate_rate(hashes)
    accepted = totals["accepted"]
    event_aware = classification_summary(classify(kept))
    rescue = audit_rescues(kept)
    return {
        "config": config.as_dict(),
        "cache_stride_seconds": round(stride_seconds, 4) if stride_seconds else None,
        "offered": totals["offered"],
        "accepted": accepted,
        "acceptance_rate": round(accepted / totals["offered"], 4) if totals["offered"] else None,
        "duplicates_among_accepted": duplicates,
        "duplicate_rate": duplicate_rate,
        "event_aware": event_aware,
        "event_aware_removal_rate": event_aware["removal_rate"],
        "rescue_audit": rescue,
        "person_frames": totals["person_frames"],
        "person_frame_rate": round(totals["person_frames"] / accepted, 4) if accepted else None,
        "candidates": totals["candidates"],
        "candidates_per_accepted": round(totals["candidates"] / accepted, 3) if accepted else None,
        "baseline_triggered": totals["baseline"],
        "candidate_subject_tracks": totals["tracks"],
        "baseline_share": round(totals["baseline"] / accepted, 4) if accepted else None,
        "distinct_reasons": len(reasons),
        "by_reason": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "suppressed": dict(sorted(suppressed.items())),
        "by_clip": per_clip,
    }


def grid() -> list[tuple[str, SamplingConfig]]:
    """The comparison. One axis varied at a time, from a common base.

    Deliberately not a full factorial: the point is to read each parameter's
    effect, and a 4×4×4 grid would report 64 numbers in which no single
    parameter's contribution is legible.
    """
    base = SamplingConfig()
    out: list[tuple[str, SamplingConfig]] = [("base", base)]
    for gap in (0.0, 1.0, 2.0, 4.0, 8.0):
        out.append((f"min_gap={gap:g}s", replace(base, min_gap_seconds=gap)))
    for floor in (0.0, 0.5, 1.0, 2.0):
        out.append((f"hard_floor={floor:g}s", replace(base, hard_floor_seconds=floor)))
    for beat in (15.0, 30.0, 45.0, 90.0, 100000.0):
        out.append((f"heartbeat={beat:g}s", replace(base, heartbeat_seconds=beat)))
    for bits in (4, 6, 8, 12, 16):
        out.append((f"scene_bits={bits}", replace(base, scene_change_bits=bits)))
    for cap in (5, 10, 20, 40):
        out.append((f"max_per_reason={cap}", replace(base, max_per_reason=cap)))
    for rate in (0.2, 0.5, 1.0, 2.0):
        out.append((f"detect_every={rate:g}s", replace(base, detect_every_seconds=rate)))
    for hits in (1, 2, 3, 4):
        out.append((f"min_hits={hits}", replace(base, track_min_hits=hits)))
    for age in (0, 1, 2, 4):
        out.append((f"max_age={age}", replace(base, track_max_age=age)))
    out.append(
        (
            "no_hysteresis",
            replace(base, track_min_hits=1, track_max_age=0),
        )
    )

    # Combined candidates. The single-axis rows above show what each parameter
    # does; these are the settings actually up for selection, evaluated on the
    # same cache so the comparison is like for like.
    out.append(("cand-A/base", base))
    out.append(
        (
            "cand-B/detect1",
            replace(base, detect_every_seconds=1.0, max_per_reason=12),
        )
    )
    out.append(
        (
            "cand-C/detect1-gap3",
            replace(
                base, detect_every_seconds=1.0, max_per_reason=12, min_gap_seconds=3.0
            ),
        )
    )
    out.append(
        (
            "cand-D/detect2",
            replace(base, detect_every_seconds=2.0, max_per_reason=12),
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.build_cache or not CACHE.exists():
        print("building perception cache (runs the detector once per clip)...")
        cache = build_cache()
    else:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    observations = sum(len(c["observations"]) for c in cache["clips"])
    boxes = sum(len(o["boxes"]) for c in cache["clips"] for o in c["observations"])
    print(f"\ncache: {len(cache['clips'])} clips, {observations} observations, {boxes} boxes")

    rows = []
    print(
        f"\n{'setting':22s} {'kept':>5s} {'rate':>6s} {'dup':>6s} {'ev-dup':>7s} "
        f"{'corrob':>7s} {'trks':>5s} {'person':>7s} {'cands':>6s} {'base%':>6s} {'kinds':>5s}"
    )
    for label, config in grid():
        try:
            result = evaluate(config, cache)
        except ValueError as error:
            print(f"{label:22s} SKIPPED — {error}")
            continue
        result["label"] = label
        rows.append(result)
        print(
            f"{label:22s} {result['accepted']:5d} "
            f"{result['acceptance_rate']:6.1%} {result['duplicate_rate']:6.1%} "
            f"{result['event_aware_removal_rate']:7.1%} "
            f"{(result['rescue_audit']['corroborated_rate'] or 0):7.1%} "
            f"{result['candidate_subject_tracks']:5d} "
            f"{result['person_frame_rate']:7.1%} {result['candidates']:6d} "
            f"{result['baseline_share']:6.1%} {result['distinct_reasons']:5d}"
        )

    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(
            json.dumps(
                {
                    "_comment": [
                        "P9.6 Phase 4. One axis varied at a time from a common base,",
                        "replayed against a fixed perception cache so every row sees",
                        "identical input. No row is 'best' on every metric and none",
                        "was selected on one: the chosen configuration is argued in",
                        "P9_6_EVENT_SAMPLING_REPORT.md against all of them.",
                    ],
                    "clips": [
                        {k: v for k, v in c.items() if k != "observations"}
                        for c in cache["clips"]
                    ],
                    "observations": observations,
                    "results": rows,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten: {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
