"""Run the current pipeline over an annotated dataset and score it.

    python -m tools.vision_eval.run_baseline datasets/kitchen-01 --tag baseline

The result is written to ``results/<tag>.json`` and becomes the reference point
a later change is compared against. Nothing here decides anything; it measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate
from .predict import PerceptionRunner
from .schema import load_annotations

DEFAULT_POLICY = Path("config/policies/kitchen-safety.example.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--provider", default="nvidia")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="where the API key is read from; never from source")
    parser.add_argument("--no-regions", action="store_true",
                        help="one whole-subject crop for every attribute (pre-Phase-4)")
    parser.add_argument("--no-gate", action="store_true",
                        help="ask the model even about crops the gate rejects")
    parser.add_argument("--crop-size", type=int, default=0,
                        help="canonical crop side; 0 keeps the platform default")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", action="store_true",
                        help="reuse answers from a previous run of the identical "
                             "configuration, so an interrupted run resumes")
    args = parser.parse_args()

    from PIL import Image

    root: Path = args.dataset
    frames = load_annotations(root / "annotations" / f"{root.name}.json")
    if args.limit:
        frames = frames[: args.limit]

    runner = PerceptionRunner(
        policy_path=args.policy,
        provider=args.provider,
        use_regions=not args.no_regions,
        use_quality_gate=not args.no_gate,
        env_file=args.env_file if args.env_file.exists() else None,
        crop_size=args.crop_size,
        cache_path=(root / "results" / f"{args.tag}.cache.json") if args.cache else None,
    )
    print(f"understander: {runner.binding_note}")
    if "static" in runner.binding_note.lower():
        print("REFUSING: the understander resolved to a static stub; "
              "a measurement of a stub is not a measurement of the system")
        return 2

    images = {f.frame_id: Image.open(root / f.image_path).convert("RGB") for f in frames}
    predicted = runner.run(frames, images)
    report = evaluate(frames, predicted, attributes=runner.attributes)

    out = root / "results" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_wire(report, runner, args), indent=2), encoding="utf-8")
    _print(report, runner)
    print(f"\nwritten: {out}")
    return 0


def _wire(report, runner, args) -> dict:
    return {
        "tag": args.tag,
        "configuration": {
            "policy": str(args.policy),
            "provider": args.provider,
            "evidence_regions": not args.no_regions,
            "quality_gate": not args.no_gate,
            "crop_size_override": runner.crop_size or None,
            "output_sizes": {k: list(v) for k, v in runner.output_sizes.items()},
            "understander": runner.binding_note,
        },
        "frames": report.frames,
        "annotated_subjects": report.annotated_subjects,
        "matched_subjects": report.matched_subjects,
        "unmatched_truth": report.unmatched_truth,
        "spurious_predictions": report.spurious_predictions,
        "vlm_calls": report.vlm_calls,
        "vlm_calls_per_1000_frames": report.vlm_calls_per_1000_frames,
        "latency": {
            "mean_ms": round(runner.stats.mean_latency_ms, 1),
            "median_ms": round(runner.stats.median_latency_ms, 1),
            "p95_ms": round(runner.stats.p95_latency_ms, 1),
            "total_wall_clock_s": round(runner.stats.wall_clock_ms / 1000, 1),
        },
        "mean_vlm_latency_ms": round(runner.stats.mean_latency_ms, 1),
        "quality_gate": {
            "evidence_crops": runner.stats.evidence_groups,
            "accepted": runner.stats.evidence_groups - runner.stats.gate_rejections,
            "rejected": runner.stats.gate_rejections,
            "rejection_rate": (
                runner.stats.gate_rejections / runner.stats.evidence_groups
                if runner.stats.evidence_groups else None
            ),
            "vlm_calls_avoided": runner.stats.gate_rejections,
            "not_visible_from_gate": runner.stats.not_visible_from_gate,
            "not_visible_from_model": runner.stats.not_visible_from_model,
        },
        "answers_replayed_from_cache": runner.stats.cached_answers,
        "gate_rejections": runner.stats.gate_rejections,
        "evidence_groups": runner.stats.evidence_groups,
        "reasons": runner.stats.reasons,
        "attributes": [
            {
                "attribute": a.attribute,
                "matched": a.matched,
                "correct": a.correct,
                "accuracy": a.accuracy,
                "unsupported_claims": a.unsupported_claims,
                "confusion": {k: dict(v) for k, v in a.confusion.items()},
                "states": [
                    {
                        "state": s.state, "support": s.support, "predicted": s.predicted,
                        "precision": s.precision, "recall": s.recall, "f1": s.f1,
                    }
                    for s in a.states
                ],
            }
            for a in report.attributes
        ],
        "failures_by_category": report.failures_by_category(),
        "failures": [
            {
                "frame_id": f.frame_id, "subject_id": f.subject_id,
                "attribute": f.attribute, "truth": f.truth, "predicted": f.predicted,
                "category": f.category.value, "crop_size": f.crop_size,
                "quality": f.quality, "vlm_used": f.vlm_used, "detail": f.detail,
            }
            for f in report.failures
        ],
    }


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _print(report, runner) -> None:
    print(f"\nframes {report.frames}  annotated {report.annotated_subjects}  "
          f"matched {report.matched_subjects}  unmatched {report.unmatched_truth}  "
          f"spurious {report.spurious_predictions}")
    stats = runner.stats
    print(f"evidence crops {stats.evidence_groups}  accepted "
          f"{stats.evidence_groups - stats.gate_rejections}  rejected {stats.gate_rejections}"
          f"  ({stats.gate_rejections / max(1, stats.evidence_groups) * 100:.1f}%)")
    print(f"vlm calls {report.vlm_calls}  avoided {stats.gate_rejections}"
          f"  per person {report.vlm_calls_per_person:.2f}"
          f"  per 1000 frames {report.vlm_calls_per_1000_frames:.0f}")
    print(f"not_visible from gate {stats.not_visible_from_gate}"
          f"  from model {stats.not_visible_from_model}")
    print(f"latency mean {stats.mean_latency_ms:.0f}ms  median "
          f"{stats.median_latency_ms:.0f}ms  p95 {stats.p95_latency_ms:.0f}ms"
          f"  total {stats.wall_clock_ms / 1000:.0f}s")
    for a in report.attributes:
        print(f"\n{a.attribute}: accuracy {_pct(a.accuracy)} over {a.matched}"
              f"   unsupported claims {a.unsupported_claims}")
        for s in a.states:
            if s.support or s.predicted:
                print(f"    {s.state:<12} support {s.support:>3}  predicted {s.predicted:>3}"
                      f"  precision {_pct(s.precision)}  recall {_pct(s.recall)}")
        for truth, row in sorted(a.confusion.items()):
            print(f"    truth {truth:<12} -> {dict(sorted(row.items()))}")
        # The production failure, named rather than left to be read off a table.
        false_absent = a.confusion.get("present", {}).get("absent", 0)
        false_present = a.confusion.get("absent", {}).get("present", 0)
        print(f"    FALSE ABSENT on a true PRESENT: {false_absent}"
              f"   false PRESENT on a true ABSENT: {false_present}")
    print(f"\nfailures: {report.failures_by_category()}")


if __name__ == "__main__":
    raise SystemExit(main())
