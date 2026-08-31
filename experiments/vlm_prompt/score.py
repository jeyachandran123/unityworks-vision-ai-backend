"""Score every recorded run on the same corpus, the same way.

Three views, because they answer three different questions and only reporting one
of them would mislead:

**UNGATED** — every subject reaches the model. This is the only view that can
answer the question this experiment exists for: *can the prompt make the VLM
refuse for itself?* P8's gate is deliberately absent here.

**P8-GATED** — the pose gate refuses first, exactly as production does since P8.
This is the operational reality, and the view a promotion decision uses.

**VERDICTS** — the P8-gated answers put through the real shipped compliance rule.
Aggregate accuracy is not the deliverable; what a kitchen manager is told is.

Ground truth has **zero** `absent` examples for `head_covering`. Every metric
that would need them reports `null`, never `0.0` — a metric with no support is
undefined, not bad, and printing a number there would be inventing one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
ROOT = Path(__file__).resolve().parents[2]

PRESENT, ABSENT, NOT_VISIBLE = "present", "absent", "not_visible"
STATES = (PRESENT, ABSENT, NOT_VISIBLE)

#: Pose states on which P8 withholds the crop.
REFUSING = {"not_located", "low_confidence"}


def load(code: str) -> dict:
    return json.loads((RUNS / f"variant_{code}.json").read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class StateScore:
    state: str
    support: int
    predicted: int
    precision: float | None
    recall: float | None
    f1: float | None


def per_state(pairs: list[tuple[str, str | None]]) -> dict[str, StateScore]:
    """Precision/recall/F1 per state. `None` where support or predictions are 0."""
    out: dict[str, StateScore] = {}
    for state in STATES:
        support = sum(1 for truth, _ in pairs if truth == state)
        predicted = sum(1 for _, pred in pairs if pred == state)
        hit = sum(1 for truth, pred in pairs if truth == state and pred == state)
        precision = _ratio(hit, predicted)
        recall = _ratio(hit, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        out[state] = StateScore(state, support, predicted, precision, recall, f1)
    return out


def confusion(pairs: list[tuple[str, str | None]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for truth, pred in pairs:
        key = pred or "unparsed"
        matrix.setdefault(truth, {}).setdefault(key, 0)
        matrix[truth][key] += 1
    return matrix


def unsupported_claims(pairs: list[tuple[str, str | None]]) -> int:
    """Truth NOT_VISIBLE, model decided anyway. The headline safety number."""
    return sum(
        1 for truth, pred in pairs if truth == NOT_VISIBLE and pred in (PRESENT, ABSENT)
    )


def verdicts(cases: list[dict], *, gated: bool) -> dict[str, list[dict]]:
    """The real shipped rule, over these answers, with a SHARPER taxonomy.

    `tests/compliance/test_dataset_regression.py` calls a violation "false" only
    when **both** body parts were unobservable, and "true" otherwise. That is too
    coarse for this experiment: a violation raised against a subject whose head
    is *visibly covered* is a false alarm caused by misreading the picture, and
    the inherited bucketing files it under `true_violation`.

    So violations are split three ways here by **why** they were raised:

    * `violation_justified` — a real breach existed. On this corpus the only
      genuine breaches are the 3 subjects with `hand_covering = absent`; there is
      no `head_covering = absent` example at all, so a head-driven violation can
      never be justified here.
    * `violation_unobservable` — raised where the annotator could not see the
      head. The failure P8 attacks.
    * `violation_semantic` — raised where the head was plainly covered. The
      failure this experiment attacks.

    The inherited definition is not wrong, it is coarser, and the P8 pinned test
    keeps using it unchanged. This one is used only inside this experiment, and
    both are reported so neither hides the other.

    `hand_covering` is held at the production model's own recorded answer for
    every variant — this experiment varies the head prompt only, so letting the
    hand answer move would confound the comparison.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from compliance import ComplianceState
    from tests.compliance.test_dataset_regression import CASES, verdict

    hands = {(c["frame"], c["subject"]): c["model"].get("hand_covering") for c in CASES}
    truth_hand = {(c["frame"], c["subject"]): c["truth"].get("hand_covering") for c in CASES}
    to_domain = {PRESENT: "cap", ABSENT: "none", NOT_VISIBLE: "not_visible"}

    buckets: dict[str, list[dict]] = {
        "violation_justified": [], "violation_unobservable": [],
        "violation_semantic": [], "missed_violation": [],
        "compliant": [], "unknown": [],
    }
    for case in cases:
        key = (case["frame"], case["subject"])
        head = case["predicted"]
        if gated and case["pose_state"] in REFUSING:
            head = None
        state = verdict(to_domain.get(head), hands.get(key))
        head_truth, hand_truth = case["truth"], truth_hand.get(key)
        breach = ABSENT in (head_truth, hand_truth)

        if state is ComplianceState.VIOLATION:
            if breach:
                buckets["violation_justified"].append(case)
            elif head_truth == NOT_VISIBLE:
                buckets["violation_unobservable"].append(case)
            else:
                buckets["violation_semantic"].append(case)
        elif state is ComplianceState.UNKNOWN:
            buckets["missed_violation" if breach else "unknown"].append(case)
        else:
            buckets["compliant"].append(case)
    return buckets


def false_violations(buckets: dict[str, list[dict]]) -> int:
    """Every violation raised against a subject who was not in breach."""
    return len(buckets["violation_unobservable"]) + len(buckets["violation_semantic"])


def score(code: str) -> dict:
    run = load(code)
    cases = run["cases"]

    ungated = [(c["truth"], c["predicted"]) for c in cases]
    gated = [
        (c["truth"], None if c["pose_state"] in REFUSING else c["predicted"])
        for c in cases
    ]

    latencies = sorted(c["latency_ms"] for c in cases)
    calls_gated = sum(1 for c in cases if c["pose_state"] not in REFUSING)

    return {
        "variant": code,
        "title": run["title"],
        "prompt_sha256": run["prompt_sha256"],
        "model": run["model"],
        "corpus_digest": run["corpus_digest"],
        "n": len(cases),
        "ungated": {
            "states": {k: asdict(v) for k, v in per_state(ungated).items()},
            "confusion": confusion(ungated),
            "unsupported_claims": unsupported_claims(ungated),
            "unparsed": sum(1 for _, p in ungated if p is None),
            "accuracy_over_parsed": _ratio(
                sum(1 for t, p in ungated if t == p),
                sum(1 for _, p in ungated if p is not None),
            ),
        },
        "p8_gated": {
            "states": {k: asdict(v) for k, v in per_state(gated).items()},
            "confusion": confusion(gated),
            "unsupported_claims": unsupported_claims(gated),
            "model_calls": calls_gated,
            "calls_saved_by_p8": len(cases) - calls_gated,
        },
        "verdicts_gated": {k: len(v) for k, v in verdicts(cases, gated=True).items()},
        "verdicts_ungated": {k: len(v) for k, v in verdicts(cases, gated=False).items()},
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "latency_total_s": round(sum(latencies) / 1000.0, 1),
        "prompt_chars": len(run["prompt"]),
        "max_output_tokens": run["max_output_tokens"],
    }


def _fmt(value) -> str:
    if value is None:
        return "  n/a"
    if isinstance(value, float):
        return f"{value:5.3f}"
    return f"{value:5d}"


def table(codes: list[str]) -> None:
    scored = {c: score(c) for c in codes}

    print("\n" + "=" * 78)
    print("UNGATED — every subject reaches the model")
    print("  answers the question this experiment exists for: can the prompt")
    print("  make the VLM refuse for itself?")
    print("=" * 78)
    rows = [
        ("NOT_VISIBLE precision", lambda s: s["ungated"]["states"][NOT_VISIBLE]["precision"]),
        ("NOT_VISIBLE recall", lambda s: s["ungated"]["states"][NOT_VISIBLE]["recall"]),
        ("NOT_VISIBLE F1", lambda s: s["ungated"]["states"][NOT_VISIBLE]["f1"]),
        ("PRESENT precision", lambda s: s["ungated"]["states"][PRESENT]["precision"]),
        ("PRESENT recall", lambda s: s["ungated"]["states"][PRESENT]["recall"]),
        ("PRESENT F1", lambda s: s["ungated"]["states"][PRESENT]["f1"]),
        ("ABSENT precision (support 0)", lambda s: s["ungated"]["states"][ABSENT]["precision"]),
        ("ABSENT recall (support 0)", lambda s: s["ungated"]["states"][ABSENT]["recall"]),
        ("unsupported claims", lambda s: s["ungated"]["unsupported_claims"]),
        ("unparsed", lambda s: s["ungated"]["unparsed"]),
    ]
    _emit(rows, scored, codes)

    print("\n" + "=" * 78)
    print("P8-GATED — operational reality since P8")
    print("=" * 78)
    rows = [
        ("NOT_VISIBLE recall", lambda s: s["p8_gated"]["states"][NOT_VISIBLE]["recall"]),
        ("PRESENT recall", lambda s: s["p8_gated"]["states"][PRESENT]["recall"]),
        ("unsupported claims", lambda s: s["p8_gated"]["unsupported_claims"]),
        ("model calls", lambda s: s["p8_gated"]["model_calls"]),
        ("calls saved by P8", lambda s: s["p8_gated"]["calls_saved_by_p8"]),
    ]
    _emit(rows, scored, codes)

    print("\n" + "=" * 78)
    print("END-TO-END VERDICTS — the real shipped rule, P8-gated")
    print("=" * 78)
    rows = [
        ("FALSE violations (total)", lambda s: s["verdicts_gated"]["violation_unobservable"]
                                             + s["verdicts_gated"]["violation_semantic"]),
        ("  ... unobservable-driven", lambda s: s["verdicts_gated"]["violation_unobservable"]),
        ("  ... semantic (visible head)", lambda s: s["verdicts_gated"]["violation_semantic"]),
        ("justified violations", lambda s: s["verdicts_gated"]["violation_justified"]),
        ("compliant", lambda s: s["verdicts_gated"]["compliant"]),
        ("missed violations", lambda s: s["verdicts_gated"]["missed_violation"]),
        ("unknown", lambda s: s["verdicts_gated"]["unknown"]),
    ]
    _emit(rows, scored, codes)

    print("\n" + "=" * 78)
    print("END-TO-END VERDICTS — WITHOUT P8 (prompt acting alone)")
    print("=" * 78)
    rows = [
        ("FALSE violations (total)", lambda s: s["verdicts_ungated"]["violation_unobservable"]
                                             + s["verdicts_ungated"]["violation_semantic"]),
        ("  ... unobservable-driven", lambda s: s["verdicts_ungated"]["violation_unobservable"]),
        ("  ... semantic (visible head)", lambda s: s["verdicts_ungated"]["violation_semantic"]),
        ("justified violations", lambda s: s["verdicts_ungated"]["violation_justified"]),
        ("compliant", lambda s: s["verdicts_ungated"]["compliant"]),
        ("missed violations", lambda s: s["verdicts_ungated"]["missed_violation"]),
    ]
    _emit(rows, scored, codes)

    print("\n" + "=" * 78)
    print("COST")
    print("=" * 78)
    rows = [
        ("latency p50 ms", lambda s: int(s["latency_p50_ms"])),
        ("latency p95 ms", lambda s: int(s["latency_p95_ms"])),
        ("total wall s (43 calls)", lambda s: int(s["latency_total_s"])),
        ("prompt chars", lambda s: s["prompt_chars"]),
        ("max output tokens", lambda s: s["max_output_tokens"]),
    ]
    _emit(rows, scored, codes)


def _emit(rows, scored, codes) -> None:
    header = f"{'metric':32s}" + "".join(f"{('  ' + c):>8s}" for c in codes)
    print(header)
    print("-" * len(header))
    for label, getter in rows:
        line = f"{label:32s}"
        for code in codes:
            line += f"{_fmt(getter(scored[code])):>8s}"
        print(line)


if __name__ == "__main__":
    import sys

    codes = sys.argv[1:] or ["A", "B", "C", "D"]
    table(codes)
    out = RUNS / "scores.json"
    out.write_text(
        json.dumps({c: score(c) for c in codes}, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nwritten: {out}")
