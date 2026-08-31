"""Per-case comparison. Aggregate metrics decide nothing on a corpus this small.

P8 already produced one lesson about this: an aggregate identical to the
published baseline concealed the loss of the corpus's only true violation. So
every safety-critical subject is listed individually here, and a variant that
improves an average while destroying one of them is rejected on this table
rather than promoted on the other one.

    python -m experiments.vlm_prompt.percase
    python -m experiments.vlm_prompt.percase --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .score import load, verdicts

RUNS = Path(__file__).resolve().parent / "runs"

#: Subjects whose outcome a promotion decision turns on.
#:
#: Chosen from ground truth and from the P8 record — never from any variant's
#: answers, so the table cannot be curated after seeing the results.
CRITICAL = {
    "f01500/s2": "the corpus's ONLY true violation - losing it rejects a variant",
    "f00060/s0": "genuinely bare hands, head covered (missed violation in baseline)",
    "f01500/s1": "genuinely bare hands (missed violation in baseline)",
    "f00780/s2": "P8's one surviving false violation; ambiguous ground truth",
    "f00300/s0": "known semantic failure - clear blue hairnet read as none",
    "f00420/s3": "known semantic failure",
    "f00900/s0": "known semantic failure",
    "f01140/s1": "known semantic failure",
    "f00660/s1": "P8 corrected this one - must stay corrected",
    "f00780/s3": "P8 corrected this one - must stay corrected",
    "f01620/s1": "P8 corrected this one - must stay corrected",
}


def bucket_of(code: str, *, gated: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, cases in verdicts(load(code)["cases"], gated=gated).items():
        for case in cases:
            out[f"{case['frame']}/{case['subject']}"] = name
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("codes", nargs="*", default=None)
    args = parser.parse_args()
    codes = args.codes or ["A", "B", "C", "D"]

    runs = {c: load(c) for c in codes}
    answers = {
        c: {f"{k['frame']}/{k['subject']}": k for k in runs[c]["cases"]} for c in codes
    }
    gated = {c: bucket_of(c, gated=True) for c in codes}
    base = answers[codes[0]]
    keys = sorted(base) if args.all else [k for k in sorted(base) if k in CRITICAL]

    print("\nPER-CASE - attribute answer (P8 gate NOT applied, so the prompt is visible)")
    head = f"{'case':12s} {'truth':12s} {'pose':16s}" + "".join(f"{c:>13s}" for c in codes)
    print(head)
    print("-" * len(head))
    for key in keys:
        row = base[key]
        line = f"{key:12s} {row['truth']:12s} {row['pose_state']:16s}"
        for code in codes:
            got = answers[code][key]["predicted"] or "UNPARSED"
            mark = "+" if got == row["truth"] else "-"
            line += f"{mark + ' ' + got:>13s}"
        print(line)
        if not args.all:
            print(f"{'':12s}  -> {CRITICAL[key]}")

    print("\nPER-CASE - end-to-end verdict (P8 gate APPLIED, real shipped rule)")
    head = f"{'case':12s} {'truth':12s}" + "".join(f"{c:>19s}" for c in codes)
    print(head)
    print("-" * len(head))
    for key in keys:
        line = f"{key:12s} {base[key]['truth']:12s}"
        for code in codes:
            line += f"{gated[code].get(key, '-'):>19s}"
        print(line)

    print("\nSAFETY SUMMARY (P8-gated verdicts over the whole corpus)")
    for code in codes:
        counts = {k: len(v) for k, v in verdicts(runs[code]["cases"], gated=True).items()}
        false = counts["violation_unobservable"] + counts["violation_semantic"]
        print(
            f"  {code}: FALSE={false:2d} "
            f"(unobservable={counts['violation_unobservable']:2d} "
            f"semantic={counts['violation_semantic']:2d})  "
            f"justified={counts['violation_justified']:2d}  "
            f"compliant={counts['compliant']:2d}  "
            f"missed={counts['missed_violation']:2d}  "
            f"unknown={counts['unknown']:2d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
