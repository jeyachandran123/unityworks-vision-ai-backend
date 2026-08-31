"""Frozen sampling policies — P9.6 Phase 2, Phase 1 of the brief.

    python -m tools.p9_dataset.baselines --write

A superseded policy that can still be **executed** is evidence; one that can only
be described is a claim. Every policy this programme has run in production is
pinned here by value, so any later result can be re-derived against the exact
configuration that produced the corpus rather than against whatever the defaults
have drifted to.

Nothing in this module reads the current defaults. The numbers are literals on
purpose: if `SamplingConfig`'s defaults change tomorrow, `PHASE1` must not move
with them, and a test asserts the frozen record still round-trips.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .events import DepartureRule, SamplingConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "p9-baselines"

#: The policy that produced the P9.6 Phase 1 corpus (12 sessions, 1,095 frames).
#:
#: Pinned by value. `DepartureRule.ON_EXPIRY` is the behaviour Phase 1 shipped
#: and the defect Phase 2 exists to test: the event fires when a track expires,
#: so the frame it keeps is the one *after* the last person walked out.
PHASE1 = SamplingConfig(
    version="p9.6-events-1",
    scene_change_bits=8,
    move_fraction=0.15,
    bbox_iou=0.70,
    bbox_area_ratio=1.25,
    overlap_iou=0.10,
    overlap_release_iou=0.05,
    edge_epsilon=0.01,
    grid=3,
    match_iou=0.30,
    match_distance_diagonals=1.20,
    match_area_ratio=2.0,
    track_min_hits=2,
    track_max_age=2,
    min_gap_seconds=2.0,
    hard_floor_seconds=0.5,
    heartbeat_seconds=45.0,
    detect_every_seconds=0.5,
    max_per_reason=10,
    max_samples=150,
    departure_rule=DepartureRule.ON_EXPIRY,
    departure_confirm_misses=0,
)

#: Variant B — departure evidence. Same detection everywhere; when a track
#: expires, keep the **last frame in which it was confirmed** rather than the
#: frame in which its absence was finally admitted.
PHASE2_B = replace(
    PHASE1, version="p9.6-events-2b", departure_rule=DepartureRule.LAST_CONFIRMED
)

#: Variant C — confirmed departure. As B, and additionally require more missed
#: detections before believing the person left at all, so a temporary occlusion
#: or a detector stutter does not become a departure.
PHASE2_C = replace(
    PHASE1,
    version="p9.6-events-2c",
    departure_rule=DepartureRule.LAST_CONFIRMED,
    track_max_age=5,
    departure_confirm_misses=4,
)

#: Variant D — B, plus routing the low-information reasons out of the annotation
#: pool rather than out of the corpus. Implemented in `dedupe`/`live_queue` by
#: sample class, not here; listed so the sweep can name it.
PHASE2_D = replace(PHASE2_B, version="p9.6-events-2d")

POLICIES = {
    "phase1": PHASE1,
    "phase2-b": PHASE2_B,
    "phase2-c": PHASE2_C,
    "phase2-d": PHASE2_D,
}

#: The corpus digest at the moment Phase 1 closed. Must not move until a human
#: annotates something.
PHASE1_CORPUS = {
    "p9_v1_digest": "fe16a44bc39e01e4",
    "p9_v2_digest": "fe16a44bc39e01e4",
    "live_sessions": 16,
    "live_frames_sampled": 1615,
    "live_frames_retained": 1194,
    "candidates": 1260,
    "frames_decoded": 164282,
    "person_free_retained": 402,
    "person_free_rate": 0.3373,
    "event_sessions": 12,
    "event_frames_sampled": 1095,
    "event_frames_retained": 964,
    "event_candidates": 956,
    "event_person_free": 323,
    "event_person_free_rate": 0.335,
    "event_reason_blind_duplicate_rate": 0.469,
    "event_aware_duplicate_rate": 0.120,
}


def record() -> dict:
    return {
        "_comment": [
            "Frozen sampling policies. Pinned BY VALUE, not read from current",
            "defaults: if the defaults drift, these must not drift with them, or",
            "a later A/B would compare two things neither of which ran in",
            "production. PHASE1 is the policy that produced the P9.6 Phase 1",
            "corpus and remains executable for replay.",
        ],
        "policies": {name: policy.as_dict() for name, policy in POLICIES.items()},
        "phase1_corpus": PHASE1_CORPUS,
        "deduplication": {
            "threshold_hamming": 5,
            "max_adjacent_gap": 3,
            "scope": "same camera AND same session AND bounded index gap",
        },
        "cameras": {
            "cam-11": {"channel": 11, "resolution": "1920x1080", "codec": "hevc", "fps": 15},
            "cam-12": {"channel": 12, "resolution": "960x576", "codec": "hevc", "fps": 25},
            "cam-13": {"channel": 13, "resolution": "1920x1080", "codec": "hevc", "fps": 15},
            "cam-14": {"channel": 14, "resolution": "960x576", "codec": "hevc", "fps": 25},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    payload = record()
    for name, policy in payload["policies"].items():
        print(f"{name:10s} {policy['version']:18s} departure={policy['departure_rule']}")
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "p9.6-phase1.json"
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
