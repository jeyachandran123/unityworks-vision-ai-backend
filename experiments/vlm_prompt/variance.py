"""Estimate the noise floor, so variant differences can be judged against it.

The provider does not return the same answer twice at temperature 0. Measured on
this corpus, the production prompt agreed with its own recorded run on **34 of 43
subjects (79.1 %)** — nine subjects flipped with the model, the prompt, the crop
bytes and the temperature all held constant.

That number is the experiment's error bar. A variant that beats the baseline by
fewer subjects than the baseline differs from itself has demonstrated nothing,
and reporting such a difference as an improvement would be the small-sample
mistake this whole programme is built to avoid.

So every variant is run several times, and this module reports:

* **within-variant agreement** — how much a variant disagrees with itself;
* **per-metric spread** — min/median/max across repeats;
* **between-variant separation** — whether the gap between two variants exceeds
  the spread within either of them.

    python -m experiments.vlm_prompt.variance
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .score import NOT_VISIBLE, PRESENT, REFUSING, per_state, unsupported_claims, verdicts

RUNS = Path(__file__).resolve().parent / "runs"


def repeats(code: str) -> list[dict]:
    """Every recorded run of one variant, in filename order."""
    found = sorted(RUNS.glob(f"variant_{code}.json")) + sorted(
        RUNS.glob(f"variant_{code}_*.json")
    )
    return [json.loads(p.read_text(encoding="utf-8")) for p in found]


def _answers(run: dict) -> dict[str, str | None]:
    return {f"{c['frame']}/{c['subject']}": c["predicted"] for c in run["cases"]}


def self_agreement(code: str) -> tuple[float | None, int]:
    """Mean pairwise agreement between repeats of the same variant."""
    runs = repeats(code)
    if len(runs) < 2:
        return None, len(runs)
    maps = [_answers(r) for r in runs]
    scores = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            keys = set(maps[i]) & set(maps[j])
            scores.append(sum(maps[i][k] == maps[j][k] for k in keys) / len(keys))
    return statistics.mean(scores), len(runs)


def metrics_of(run: dict) -> dict[str, float | int | None]:
    cases = run["cases"]
    ungated = [(c["truth"], c["predicted"]) for c in cases]
    gated = [
        (c["truth"], None if c["pose_state"] in REFUSING else c["predicted"])
        for c in cases
    ]
    states_u = per_state(ungated)
    counts = {k: len(v) for k, v in verdicts(cases, gated=True).items()}
    return {
        "not_visible_recall": states_u[NOT_VISIBLE].recall,
        "not_visible_precision": states_u[NOT_VISIBLE].precision,
        "present_recall": states_u[PRESENT].recall,
        "present_precision": states_u[PRESENT].precision,
        "unsupported_ungated": unsupported_claims(ungated),
        "unsupported_gated": unsupported_claims(gated),
        "false_violation": counts["violation_unobservable"] + counts["violation_semantic"],
        "violation_semantic": counts["violation_semantic"],
        "violation_justified": counts["violation_justified"],
        "compliant": counts["compliant"],
        "missed_violation": counts["missed_violation"],
        "unparsed": sum(1 for _, p in ungated if p is None),
        "latency_p50_ms": run["latency_p50_ms"],
    }


def spread(code: str) -> dict[str, dict]:
    runs = repeats(code)
    collected: dict[str, list] = {}
    for run in runs:
        for key, value in metrics_of(run).items():
            collected.setdefault(key, []).append(value)
    out = {}
    for key, values in collected.items():
        clean = [v for v in values if v is not None]
        if not clean:
            out[key] = {"n": 0, "min": None, "median": None, "max": None}
            continue
        out[key] = {
            "n": len(clean),
            "min": min(clean),
            "median": statistics.median(clean),
            "max": max(clean),
        }
    return out


def _cell(entry: dict) -> str:
    if not entry or entry["n"] == 0:
        return "        n/a"
    if isinstance(entry["median"], float):
        return f"{entry['median']:.3f} [{entry['min']:.3f}-{entry['max']:.3f}]"
    return f"{entry['median']:>5.0f} [{entry['min']:.0f}-{entry['max']:.0f}]"


def report(codes: list[str]) -> None:
    print("\n" + "=" * 92)
    print("RUN-TO-RUN VARIANCE — the error bar on everything else")
    print("=" * 92)
    for code in codes:
        mean, count = self_agreement(code)
        label = f"{mean * 100:.1f}%" if mean is not None else "n/a (1 run)"
        print(f"  variant {code}: {count} run(s), mean self-agreement {label}")

    print(
        "\n  Same model, same prompt, same crop bytes, temperature 0. Disagreement"
        "\n  here is provider-side non-determinism, not a prompt effect."
    )

    print("\n" + "=" * 92)
    print("METRIC SPREAD ACROSS REPEATS — median [min-max]")
    print("=" * 92)
    keys = [
        "not_visible_recall",
        "not_visible_precision",
        "present_recall",
        "present_precision",
        "unsupported_ungated",
        "unsupported_gated",
        "false_violation",

        "compliant",
        "missed_violation",
        "unparsed",
        "violation_semantic",
        "violation_justified",
    ]
    header = f"{'metric':24s}" + "".join(f"{('   ' + c):>22s}" for c in codes)
    print(header)
    print("-" * len(header))
    spreads = {c: spread(c) for c in codes}
    for key in keys:
        line = f"{key:24s}"
        for code in codes:
            line += f"{_cell(spreads[code].get(key, {})):>22s}"
        print(line)

    print("\n" + "=" * 92)
    print("SEPARATION TEST — is a gap bigger than the noise it sits in?")
    print("=" * 92)
    base = codes[0]
    for key in ("not_visible_recall", "present_recall", "false_violation"):
        print(f"\n  {key}")
        b = spreads[base].get(key, {})
        if not b or b["n"] == 0:
            continue
        for code in codes[1:]:
            v = spreads[code].get(key, {})
            if not v or v["n"] == 0:
                continue
            overlap = not (v["min"] > b["max"] or v["max"] < b["min"])
            verdict = (
                "OVERLAPS baseline range — not separable"
                if overlap
                else "SEPARATED from baseline range"
            )
            print(
                f"    {base} {b['min']}-{b['max']}  vs  {code} {v['min']}-{v['max']}"
                f"   -> {verdict}"
            )


if __name__ == "__main__":
    import sys

    report(sys.argv[1:] or ["A", "B", "C", "D"])
