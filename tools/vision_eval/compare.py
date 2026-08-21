"""Side-by-side comparison of measurement runs.

    python -m tools.vision_eval.compare datasets/kitchen-01 baseline crop448 phase42

Prints the tables a report needs, straight from the saved result files, so no
number in a report is transcribed by hand. Runs are kept in separate columns and
never merged: they measure different configurations, and averaging them would
describe a system nobody ran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def load(root: Path, tag: str) -> dict:
    return json.loads((root / "results" / f"{tag}.json").read_text(encoding="utf-8"))


def attribute(run: dict, name: str) -> dict | None:
    return next((a for a in run["attributes"] if a["attribute"] == name), None)


def state(report: dict, name: str) -> dict:
    return next(
        (s for s in report["states"] if s["state"] == name),
        {"support": 0, "predicted": 0, "precision": None, "recall": None},
    )


def latency(run: dict, key: str) -> float | None:
    block = run.get("latency")
    if isinstance(block, dict):
        return block.get(key)
    # Older result files recorded only the mean.
    return run.get("mean_vlm_latency_ms") if key == "mean_ms" else None


def main() -> int:
    root = Path(sys.argv[1])
    tags = sys.argv[2:]
    runs = {tag: load(root, tag) for tag in tags}
    width = max(len(t) for t in tags) + 2

    def row(label: str, values) -> None:
        cells = "".join(f"{str(v):>{width + 6}}" for v in values)
        print(f"  {label:<38}{cells}")

    print("\n" + "=" * 100)
    print("CONFIGURATION")
    print("=" * 100)
    row("run", tags)
    row("evidence output size", [
        json.dumps(r["configuration"].get("output_sizes"))
        if r["configuration"].get("output_sizes")
        else r["configuration"].get("crop_size", "224 (default)")
        for r in runs.values()
    ])
    row("evidence regions", [r["configuration"]["evidence_regions"] for r in runs.values()])
    row("quality gate", [r["configuration"]["quality_gate"] for r in runs.values()])

    for name in ("head_covering", "hand_covering"):
        reports = {t: attribute(r, name) for t, r in runs.items()}
        if not any(reports.values()):
            continue
        print("\n" + "=" * 100)
        print(f"{name.upper()}")
        print("=" * 100)
        first = next(r for r in reports.values() if r)
        counts = {
            s: state(first, s)["support"]
            for s in ("present", "absent", "not_visible", "unknown")
        }
        print(f"  ground truth: {counts}   (support is identical across runs)")
        decided = counts["present"] + counts["absent"]
        if decided < 10:
            print(f"  ** {decided} decided ground-truth examples — "
                  f"insufficient to evaluate this attribute **")
        print()
        row("accuracy", [pct(r["accuracy"]) for r in reports.values()])
        row("correct / matched",
            [f"{r['correct']}/{r['matched']}" for r in reports.values()])
        for label in ("present", "absent", "not_visible"):
            row(f"{label} precision",
                [pct(state(r, label)["precision"]) for r in reports.values()])
            row(f"{label} recall",
                [pct(state(r, label)["recall"]) for r in reports.values()])
        row("UNKNOWN produced",
            [state(r, "unknown")["predicted"] for r in reports.values()])
        row("NOT_VISIBLE produced",
            [state(r, "not_visible")["predicted"] for r in reports.values()])
        row("FALSE ABSENT on a true PRESENT",
            [r["confusion"].get("present", {}).get("absent", 0) for r in reports.values()])
        row("FALSE PRESENT on a true ABSENT",
            [r["confusion"].get("absent", {}).get("present", 0) for r in reports.values()])
        row("unsupported claims",
            [r["unsupported_claims"] for r in reports.values()])

    print("\n" + "=" * 100)
    print("QUALITY GATE")
    print("=" * 100)
    gates = [r.get("quality_gate", {}) for r in runs.values()]
    row("evidence crops", [g.get("evidence_crops", r["evidence_groups"])
                           for g, r in zip(gates, runs.values())])
    row("accepted", [g.get("accepted", "-") for g in gates])
    row("rejected", [g.get("rejected", r["gate_rejections"])
                     for g, r in zip(gates, runs.values())])
    row("rejection rate", [pct(g.get("rejection_rate")) for g in gates])
    row("VLM calls avoided", [g.get("vlm_calls_avoided", r["gate_rejections"])
                              for g, r in zip(gates, runs.values())])
    row("NOT_VISIBLE from gate", [g.get("not_visible_from_gate", "-") for g in gates])
    row("NOT_VISIBLE from model", [g.get("not_visible_from_model", "-") for g in gates])

    print("\n" + "=" * 100)
    print("VLM USAGE AND LATENCY")
    print("=" * 100)
    row("VLM calls", [r["vlm_calls"] for r in runs.values()])
    row("calls per person",
        [f"{r['vlm_calls'] / max(1, r['matched_subjects']):.2f}" for r in runs.values()])
    row("calls per 1000 frames",
        [f"{r['vlm_calls_per_1000_frames']:.0f}" for r in runs.values()])
    row("mean latency ms", [latency(r, "mean_ms") or "-" for r in runs.values()])
    row("median latency ms", [latency(r, "median_ms") or "-" for r in runs.values()])
    row("p95 latency ms", [latency(r, "p95_ms") or "-" for r in runs.values()])
    row("total wall clock s", [latency(r, "total_wall_clock_s") or "-" for r in runs.values()])

    print("\n" + "=" * 100)
    print("FAILURE CATEGORIES")
    print("=" * 100)
    every = sorted({k for r in runs.values() for k in r["failures_by_category"]})
    for category in every:
        row(category, [r["failures_by_category"].get(category, 0) for r in runs.values()])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
