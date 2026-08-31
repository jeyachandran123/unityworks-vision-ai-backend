"""Collection quality and dataset diversity — Phases 13, 14 and 18.

    python -m tools.p9_dataset.quality --write

Reads the sessions, the event-aware deduplication and the review queue, and
reports what the collection actually bought. It computes no labels and proposes
none.

### Why several yields rather than one

Phase 13 lists eight ratios and warns against optimising for any single one, and
the P9.5 corpus shows why in one line: equal sampling gave every camera 130
frames, and candidate yield still differed by 12x. A collector tuned on frames
kept would have called that a success.

So each stage gets its own denominator — decoded, sampled, retained, per
camera-minute — and they are reported side by side. Where two disagree, the
disagreement is the finding.

### The over/under rule, stated rather than intuited

A category is flagged against the **uniform share** for its dimension:

* `OVERREPRESENTED` — more than 2x the uniform share
* `UNDERREPRESENTED` — less than 0.5x the uniform share
* `BALANCED` — in between
* `UNKNOWN` — the dimension cannot be measured from what exists

Uniform is a reference point, not a target. cam-12 genuinely holds more people
than cam-14, and a corpus that mirrored the kitchen would be skewed too. The flag
says "an annotation pass drawn from this pool would be dominated by X", which is
a fact about the pool; whether to correct it is a judgement, and the only legal
correction is **more observation**, never duplication.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .dedupe import (
    Redundancy,
    audit_rescues,
    classification_summary,
    classify,
    frame_key,
    load_frames,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "p9-live"
OUT = LIVE / "quality.json"

#: Share of the uniform expectation above which a category is over-represented.
OVER = 2.0

#: ...and below which it is under-represented.
UNDER = 0.5


def _sessions() -> dict:
    out = {}
    for directory in sorted(LIVE.glob("live-*")):
        record = directory / "session.json"
        if record.exists():
            payload = json.loads(record.read_text(encoding="utf-8"))
            out[payload["session_id"]] = payload
    return out


def balance(counts: dict) -> dict:
    """Flag each category against the uniform share for its dimension."""
    total = sum(counts.values())
    if not total or not counts:
        return {"total": 0, "categories": {}, "note": "nothing to measure"}
    uniform = total / len(counts)
    out = {}
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        ratio = count / uniform
        out[name] = {
            "count": count,
            "share": round(count / total, 4),
            "vs_uniform": round(ratio, 2),
            "flag": (
                "OVERREPRESENTED"
                if ratio > OVER
                else "UNDERREPRESENTED"
                if ratio < UNDER
                else "BALANCED"
            ),
        }
    return {"total": total, "categories": out}


def _by_class(retained, frames_with_people: set) -> dict:
    """Event and baseline scored apart — Phase 3 of the P9.6 Phase 2 brief.

    Phase 1 published one person-positive rate across both classes and it
    misled in both directions. A baseline frame is the record of stillness; it
    is *supposed* to be empty, so folding it into a person-yield metric
    penalises it for doing its job, while the event frames simultaneously hide
    how much of the session the event layer failed to explain.
    """
    out: dict[str, dict] = {}
    for frame in retained:
        label = frame.sample_class or "unclassified"
        cell = out.setdefault(label, {"frames": 0, "with_person": 0})
        cell["frames"] += 1
        frame_id = f"{frame.session_id}.{frame.camera_id}.{frame.order:04d}"
        if frame_id in frames_with_people:
            cell["with_person"] += 1
    for cell in out.values():
        cell["person_free"] = cell["frames"] - cell["with_person"]
        cell["person_positive_rate"] = round(cell["with_person"] / cell["frames"], 4)
        cell["person_free_rate"] = round(cell["person_free"] / cell["frames"], 4)
    out.setdefault("_note", {})
    out["_note"] = {
        "unclassified": "P9.5 wall-clock frames, which predate the distinction",
        "baseline_is_not_scored_on_people": (
            "a baseline frame is the record of stillness and detector failure; "
            "its purpose is temporal coverage, not person yield"
        ),
    }
    return out


def report() -> dict:
    sessions = _sessions()
    frames = load_frames()
    if not frames:
        return {"frames": 0, "note": "no live frames collected yet"}

    verdicts = classify(frames)
    dedup = classification_summary(verdicts)
    rescue = audit_rescues(frames)
    retained = [
        f
        for f in frames
        if verdicts.get(frame_key(f.path))
        in (Redundancy.UNIQUE, Redundancy.MEANINGFUL_CHANGE)
    ]

    queue_path = LIVE / "review_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
    candidates = queue.get("candidates", [])

    decoded = sum(s["totals"]["frames_decoded"] for s in sessions.values())
    event_triggered = sum(s["totals"].get("event_triggered", 0) for s in sessions.values())
    baseline = sum(s["totals"].get("baseline_triggered", 0) for s in sessions.values())

    # ---- per camera -----------------------------------------------------
    by_camera: dict[str, dict] = {}
    for frame in frames:
        entry = by_camera.setdefault(
            frame.camera_id,
            {"sampled": 0, "retained": 0, "decoded": 0, "seconds": 0.0, "candidates": 0},
        )
        entry["sampled"] += 1
        if verdicts.get(frame_key(frame.path)) in (
            Redundancy.UNIQUE,
            Redundancy.MEANINGFUL_CHANGE,
        ):
            entry["retained"] += 1
    for session in sessions.values():
        for camera in session["cameras"]:
            entry = by_camera.setdefault(
                camera["camera_id"],
                {"sampled": 0, "retained": 0, "decoded": 0, "seconds": 0.0, "candidates": 0},
            )
            entry["decoded"] += camera["frames_decoded"]
            entry["seconds"] = round(entry["seconds"] + camera["seconds"], 1)
            entry["sessions"] = entry.get("sessions", 0) + 1
            entry["tracks"] = entry.get("tracks", 0) + camera.get(
                "candidate_subject_tracks", 0
            )
    frames_with_people: set[str] = set()
    for candidate in candidates:
        entry = by_camera.setdefault(candidate["camera_id"], {})
        entry["candidates"] = entry.get("candidates", 0) + 1
        frames_with_people.add(candidate["frame_id"])
    for entry in by_camera.values():
        minutes = entry.get("seconds", 0.0) / 60.0
        entry["candidates_per_camera_minute"] = (
            round(entry.get("candidates", 0) / minutes, 2) if minutes else None
        )
        entry["collection_efficiency"] = (
            round(entry["retained"] / entry["decoded"], 5) if entry.get("decoded") else None
        )
        entry["duplicate_rate"] = (
            round(1 - entry["retained"] / entry["sampled"], 4) if entry.get("sampled") else None
        )

    # ---- per session ----------------------------------------------------
    by_session: dict[str, dict] = {}
    for session_id, session in sessions.items():
        sampled = [f for f in frames if f.session_id == session_id]
        kept = [
            f
            for f in sampled
            if verdicts.get(frame_key(f.path))
            in (Redundancy.UNIQUE, Redundancy.MEANINGFUL_CHANGE)
        ]
        found = [c for c in candidates if c["session_id"] == session_id]
        by_session[session_id] = {
            "collected_at": session["collected_at"],
            "day": session.get("collection_day", ""),
            "period": session.get("period", "unspecified"),
            "strategy": session["sampling"].get("strategy", "wall-clock interval"),
            "cameras": len(session["cameras"]),
            "decoded": session["totals"]["frames_decoded"],
            "sampled": len(sampled),
            "retained": len(kept),
            "candidates": len(found),
            "candidates_per_retained": round(len(found) / len(kept), 3) if kept else None,
        }

    # ---- distributions --------------------------------------------------
    reasons = collections.Counter(f.reason or "unrecorded" for f in retained)
    flags: collections.Counter = collections.Counter()
    for candidate in candidates:
        for flag in candidate.get("review_flags", ()):
            flags[flag] += 1

    person_counts = collections.Counter(
        str(candidate.get("people_in_frame", "?")) for candidate in candidates
    )
    hints = collections.Counter(
        candidate.get("head_observability_hint", "unknown") for candidate in candidates
    )

    return {
        "_comment": [
            "P9.6 collection quality and diversity. NO LABELS ARE COMPUTED HERE.",
            "Every stage carries its own denominator because the stages disagree:",
            "P9.5 sampled all four cameras equally and still saw a 12x spread in",
            "candidate yield. Where two ratios disagree, that IS the finding.",
            "Balance flags compare against a uniform share, which is a reference",
            "point and not a target — the kitchen itself is not uniform. The only",
            "legal correction for an under-represented category is more",
            "observation; duplication is forbidden.",
        ],
        "sessions": len(sessions),
        "days": sorted({s.get("collection_day", "") for s in sessions.values()}),
        "cameras": sorted(by_camera),
        "efficiency": {
            "frames_decoded": decoded,
            "frames_sampled": len(frames),
            "frames_retained": len(retained),
            "collection_efficiency": round(len(retained) / decoded, 5) if decoded else None,
            "event_yield": round(event_triggered / decoded, 5) if decoded else None,
            "duplicate_rate": dedup["removal_rate"],
            "person_yield": (
                round(len(frames_with_people) / len(retained), 4) if retained else None
            ),
            "candidate_yield": (
                round(len(candidates) / len(retained), 3) if retained else None
            ),
            "candidates_per_decoded": (
                round(len(candidates) / decoded, 5) if decoded else None
            ),
            "event_triggered": event_triggered,
            "baseline_triggered": baseline,
            "baseline_share": (
                round(baseline / (event_triggered + baseline), 4)
                if event_triggered + baseline
                else None
            ),
        },
        "deduplication": dedup,
        "rescue_audit": rescue,
        "by_sample_class": _by_class(retained, frames_with_people),
        "by_camera": dict(sorted(by_camera.items())),
        "by_session": by_session,
        "diversity": {
            "camera": balance({c: e.get("candidates", 0) for c, e in by_camera.items()}),
            "day": balance(
                collections.Counter(c.get("collection_day", "?") for c in candidates)
            ),
            "session": balance(
                collections.Counter(c["session_id"] for c in candidates)
            ),
            "period": balance(
                collections.Counter(c.get("period", "unspecified") for c in candidates)
            ),
            "event": balance(dict(reasons)),
            "people_per_frame": balance(dict(person_counts)),
            "review_flag": balance(dict(flags)),
            "head_observability_hint": balance(dict(hints)),
            "subject": {
                "flag": "UNKNOWN",
                "note": (
                    "Subject identity is not established. Track ids are association "
                    "aids that survive no occlusion and no session boundary, and "
                    "promoting one to a person would put the same human on both "
                    "sides of a split while every leakage check reported clean."
                ),
                "candidate_subject_tracks": sum(
                    e.get("tracks", 0) for e in by_camera.values()
                ),
            },
            "hard_case": {
                "flag": "UNKNOWN",
                "note": (
                    "Hard-case tags are populated by human annotation, which has "
                    "not occurred. The 33-tag taxonomy exists and is validated; "
                    "inferring tags from a model would make the corpus describe "
                    "the model."
                ),
            },
            "ppe_state": {
                "flag": "UNKNOWN",
                "note": (
                    "No PPE label exists for any live frame, by design. P9.6 is "
                    "data acquisition only."
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = report()
    if not result.get("sessions"):
        print("no live sessions found under", LIVE)
        return 1

    efficiency = result["efficiency"]
    print(f"sessions           : {result['sessions']}  days {result['days']}")
    print(f"decoded            : {efficiency['frames_decoded']:,}")
    print(f"sampled            : {efficiency['frames_sampled']}")
    print(f"retained           : {efficiency['frames_retained']}")
    print(f"collection eff.    : {efficiency['collection_efficiency']:.5f}")
    print(f"duplicate rate     : {efficiency['duplicate_rate']:.1%}")
    print(f"person yield       : {efficiency['person_yield']:.1%}")
    print(f"candidate yield    : {efficiency['candidate_yield']}")
    print(f"baseline share     : {efficiency['baseline_share']:.1%}")

    print(
        f"\n{'camera':8s} {'runtime':>8s} {'decoded':>8s} {'sampled':>8s} "
        f"{'retain':>7s} {'cands':>6s} {'/cam-min':>9s}"
    )
    for camera, entry in result["by_camera"].items():
        print(
            f"{camera:8s} {entry.get('seconds', 0):7.0f}s {entry.get('decoded', 0):8d} "
            f"{entry['sampled']:8d} {entry['retained']:7d} "
            f"{entry.get('candidates', 0):6d} "
            f"{entry.get('candidates_per_camera_minute', 0):9.2f}"
        )

    print("\ndiversity flags:")
    for dimension, payload in result["diversity"].items():
        if "categories" in payload:
            flagged = {
                name: entry["flag"]
                for name, entry in payload["categories"].items()
                if entry["flag"] != "BALANCED"
            }
            print(f"  {dimension:26s} {flagged or 'all balanced'}")
        else:
            print(f"  {dimension:26s} {payload['flag']}")

    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
