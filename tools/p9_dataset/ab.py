"""A/B replay of sampling policies over identical evidence — Phases 3–6.

    python -m tools.p9_dataset.ab --write

Every policy is replayed over the **same** perception traces, so a difference
between two rows is a difference between the policies and not between two
kitchens. P9.6 Phase 1 established that session-to-session variance swamps
parameter effects — two sessions at identical settings differed 4.4x in candidate
yield — which is exactly why a live A/B would prove nothing.

### Event and baseline are scored apart

Phase 1 published one efficiency number across both and it misled in both
directions: the baseline dragged the person-positive rate down while the event
frames concealed how little of the session the baseline was covering. A baseline
frame is *supposed* to be empty — it is the record of stillness — so scoring it
on person yield penalises it for working. Here:

* **event frames** are judged on person-positive rate, duplication and yield;
* **baseline frames** are judged on temporal coverage and camera health.

The combined figure is still printed, because it is what an annotation pool would
actually contain, but it is never the headline.

### What a replay can and cannot claim

It can compare policies. It cannot state the absolute rate a policy would achieve
live, because a trace is sampled at a fixed cadence while the live sampler hashes
every decoded frame. Deltas come from here; absolutes come from live collection.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .baselines import POLICIES
from .event_audit import audit
from .dedupe import (
    DEFAULT_THRESHOLD,
    MAX_ADJACENT_GAP,
    PROTECTED_REASONS,
    FrameHash,
    Redundancy,
    classification_summary,
    classify,
    hamming,
)
from .trace import load_traces, replay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "p9-traces" / "ab_results.json"


def _naive_duplicates(entries: list[dict]) -> int:
    """Reason-blind duplication, so the number is comparable with P9.5's 55.8 %."""
    duplicates = 0
    kept: list[tuple[int, int]] = []
    for order, entry in enumerate(entries):
        recent = [(o, h) for o, h in kept if order - o <= MAX_ADJACENT_GAP]
        if any(hamming(entry["hash"], h) <= DEFAULT_THRESHOLD for _, h in recent):
            duplicates += 1
        else:
            kept.append((order, entry["hash"]))
    return duplicates


def _score(kept: list[dict], camera_id: str, trace_id: str) -> dict:
    """Metrics for one bag of kept frames, split by sample class."""
    frames = [
        FrameHash(
            path=Path(f"{trace_id}/{camera_id}/{e['captured']:06d}"),
            camera_id=camera_id,
            session_id=trace_id,
            order=order,
            bits=e["hash"],
            reason=e["reason"],
            people=e["people"],
        )
        for order, e in enumerate(kept)
    ]
    verdicts = classify(frames)
    summary = classification_summary(verdicts)
    retained = [
        e
        for e, f in zip(kept, frames, strict=False)
        if verdicts[str(f.path)] in (Redundancy.UNIQUE, Redundancy.MEANINGFUL_CHANGE)
    ]

    def bag(entries: list[dict]) -> dict:
        if not entries:
            return {"frames": 0}
        with_person = sum(1 for e in entries if e["people"] > 0)
        return {
            "frames": len(entries),
            "with_person": with_person,
            "person_free": len(entries) - with_person,
            "person_positive_rate": round(with_person / len(entries), 4),
            "person_free_rate": round(1 - with_person / len(entries), 4),
            "candidates": sum(e["people"] for e in entries),
            "naive_duplicate_rate": round(_naive_duplicates(entries) / len(entries), 4),
        }

    return {
        "sampled": len(kept),
        "retained": summary["retained"],
        "event_aware_duplicate_rate": summary["removal_rate"],
        "naive_duplicate_rate": (
            round(_naive_duplicates(kept) / len(kept), 4) if kept else None
        ),
        "exact_duplicates": summary["exact_duplicates"],
        "near_duplicates": summary["near_duplicates"],
        "meaningful_change": summary["meaningful_change"],
        "all": bag(retained),
        "event": bag([e for e in retained if e["sample_class"] == "event"]),
        "baseline": bag([e for e in retained if e["sample_class"] == "baseline"]),
        "by_reason": dict(
            collections.Counter(e["reason"] for e in retained).most_common()
        ),
        "retrospective_captures": sum(1 for e in kept if e["retrospective"]),
        "retrospective_fallbacks": sum(1 for e in kept if e["fallback"]),
    }


def _per_reason(kept: list[dict]) -> dict:
    """Person-positive rate per event type — the diagnosis Phase 1 needed."""
    out: dict[str, dict] = {}
    for entry in kept:
        cell = out.setdefault(
            entry["reason"], {"frames": 0, "with_person": 0, "person_free": 0}
        )
        cell["frames"] += 1
        if entry["people"] > 0:
            cell["with_person"] += 1
        else:
            cell["person_free"] += 1
    for cell in out.values():
        cell["person_free_rate"] = round(cell["person_free"] / cell["frames"], 4)
        cell["protected"] = None
    for reason, cell in out.items():
        cell["protected"] = reason in PROTECTED_REASONS
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["frames"]))


def evaluate(policy_name: str, traces: list[dict]) -> dict:
    config = POLICIES[policy_name]
    per_camera: dict[str, list[dict]] = collections.defaultdict(list)
    per_trace: dict[str, dict] = {}
    decoded = 0
    observations = 0
    all_kept: list[dict] = []
    tracks = 0

    for trace in traces:
        result = replay(trace, config)
        trace_kept: list[dict] = []
        for camera_id, payload in result["by_camera"].items():
            decoded += payload["frames_decoded"]
            observations += payload["observations"]
            tracks += payload["statistics"]["candidate_subject_tracks"]
            for entry in payload["kept"]:
                entry = dict(entry, camera_id=camera_id, trace_id=trace["trace_id"])
                per_camera[camera_id].append(entry)
                trace_kept.append(entry)
                all_kept.append(entry)
        per_trace[trace["trace_id"]] = {
            "period": trace.get("period", "unspecified"),
            "day": trace.get("collection_day", ""),
            **_score(trace_kept, "mixed", trace["trace_id"]),
        }

    cameras = {
        camera_id: _score(entries, camera_id, "all")
        for camera_id, entries in sorted(per_camera.items())
    }
    overall = _score(all_kept, "all", "all")
    overall["frames_decoded"] = decoded
    overall["observations"] = observations
    overall["candidates_per_decoded"] = (
        round(overall["all"].get("candidates", 0) / decoded, 6) if decoded else None
    )
    overall["candidates_per_retained_event_frame"] = (
        round(overall["event"]["candidates"] / overall["event"]["frames"], 3)
        if overall["event"].get("frames")
        else None
    )
    # Coverage, so a policy cannot buy its person-free rate by missing people.
    # Phase 8 is explicit that such a trade is a regression, not a win.
    overall["candidate_subject_tracks"] = tracks
    overall["baseline_contribution"] = (
        round(overall["baseline"].get("frames", 0) / overall["retained"], 4)
        if overall["retained"]
        else None
    )
    return {
        "policy": policy_name,
        "config": config.as_dict(),
        "overall": overall,
        "per_reason": _per_reason(all_kept),
        "event_audit": audit(all_kept, config),
        "event_audit_earliest": audit(all_kept, config, reference="reference_earliest"),
        "by_camera": cameras,
        "by_trace": per_trace,
    }


def _delta(a, b) -> str:
    if a is None or b is None:
        return "—"
    absolute = b - a
    relative = (absolute / a * 100) if a else float("inf")
    sign = "+" if absolute >= 0 else ""
    if abs(relative) == float("inf"):
        return f"{sign}{absolute:.4f} (n/a)"
    return f"{sign}{absolute:.4f} ({sign}{relative:.1f}%)"


def compare(results: dict, base: str = "phase1") -> dict:
    rows = {}
    reference = results[base]["overall"]
    for name, result in results.items():
        if name == base:
            continue
        overall = result["overall"]
        rows[name] = {
            metric: {
                "phase1": _get(reference, metric),
                name: _get(overall, metric),
                "delta": _delta(_get(reference, metric), _get(overall, metric)),
            }
            for metric in METRICS
        }
    return rows


METRICS = (
    "sampled",
    "retained",
    "naive_duplicate_rate",
    "event_aware_duplicate_rate",
    "all.person_free_rate",
    "all.candidates",
    "event.frames",
    "event.person_free_rate",
    "event.person_positive_rate",
    "event.candidates",
    "event.naive_duplicate_rate",
    "baseline.frames",
    "baseline.person_free_rate",
    "candidates_per_decoded",
    "candidates_per_retained_event_frame",
    "baseline_contribution",
    "candidate_subject_tracks",
    "retrospective_captures",
    "retrospective_fallbacks",
)


def _get(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--policies", default="phase1,phase2-b,phase2-c")
    args = parser.parse_args()

    traces = load_traces()
    if not traces:
        print("no perception traces found; run `trace record` first")
        return 1

    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    results = {name: evaluate(name, traces) for name in names}

    observations = results[names[0]]["overall"]["observations"]
    decoded = results[names[0]]["overall"]["frames_decoded"]
    print(
        f"replayed {len(traces)} trace(s), {observations} observations, "
        f"{decoded:,} decoded frames — identical input for every policy\n"
    )

    width = max(len(n) for n in names)
    print(f"{'metric':40s} " + " ".join(f"{n:>{max(width, 10)}s}" for n in names))
    for metric in METRICS:
        values = []
        for name in names:
            value = _get(results[name]["overall"], metric)
            values.append("—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value))
        print(f"{metric:40s} " + " ".join(f"{v:>{max(width, 10)}s}" for v in values))

    # frames / person-free per reason, for EVERY policy side by side. This is
    # the coverage evidence: a policy that improved its person-free rate by
    # ceasing to fire an event has not improved, it has stopped looking.
    reasons = sorted(
        {r for name in names for r in results[name]["per_reason"]},
        key=lambda r: -results[names[0]]["per_reason"].get(r, {}).get("frames", 0),
    )
    print("\nframes kept / person-free, by reason and policy:")
    print(f"  {'reason':22s} " + " ".join(f"{n:>20s}" for n in names))
    for reason in reasons:
        cells = []
        for name in names:
            cell = results[name]["per_reason"].get(reason)
            cells.append(
                "—".rjust(20)
                if cell is None
                else (
                    f"{cell['frames']:4d} /{cell['person_free']:4d} "
                    f"({cell['person_free_rate']:4.0%})"
                ).rjust(20)
            )
        print(f"  {reason:22s} " + " ".join(cells))

    print("\nevent audit (testable consequences only), under both references:")
    for name in names:
        near = results[name]["event_audit"]["overall_testable"]
        far = results[name]["event_audit_earliest"]["overall_testable"]

        def rate(summary):
            return "—" if summary["rate"] is None else format(summary["rate"], ".1%")

        print(
            f"  {name:20s} nearest {near['held']:4d}/{near['compared']:4d} "
            f"= {rate(near):>6s}    earliest {far['held']:4d}/{far['compared']:4d} "
            f"= {rate(far):>6s}"
        )

    print("\nby camera:")
    for camera in sorted(results[names[0]]["by_camera"]):
        line = f"  {camera:8s}"
        for name in names:
            cell = results[name]["by_camera"].get(camera, {})
            event = cell.get("event", {})
            line += (
                f"  {name}: kept={cell.get('sampled', 0):3d} "
                f"ev={event.get('frames', 0):3d} "
                f"free={event.get('person_free_rate', 0):.0%}"
            )
        print(line)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            json.dumps(
                {
                    "_comment": [
                        "A/B replay over identical perception traces. Deltas between",
                        "policies are valid; absolute rates are not, because a trace",
                        "is sampled at a fixed cadence while the live sampler hashes",
                        "every decoded frame. Event and baseline frames are scored",
                        "apart: a baseline frame is supposed to be empty.",
                    ],
                    "traces": [
                        {
                            "trace_id": t["trace_id"],
                            "recorded_at": t["recorded_at"],
                            "period": t.get("period"),
                            "observations": t["totals"]["observations"],
                        }
                        for t in traces
                    ],
                    "results": results,
                    "comparison": compare(results),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
