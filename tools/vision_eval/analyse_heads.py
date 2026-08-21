"""Where every head answer came from, and why the wrong ones were wrong.

    python -m tools.vision_eval.analyse_heads datasets/kitchen-01 phase42

Two questions the aggregate metrics cannot answer:

**What happened to the 13 heads a human could not read?** Splitting them by
whether the quality gate stopped them separates a fixable evidence problem from
one no threshold can reach. A head that is sharp, large, and still unreadable is
unreadable for a reason quality does not measure.

**Why is each remaining failure wrong?** Categories come from what the platform
recorded and what the annotator wrote, never from a guess. A cause that cannot be
established stays ``UNKNOWN_FAILURE_REASON``; forcing a tidy distribution would
defeat the point of measuring one.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .schema import load_annotations

#: Words an annotator used for a head that could not be read, mapped to the cause.
#:
#: Read from the annotator's own free-text note rather than inferred from the
#: states, because the note is the only record of *why* a human could not see it.
POSE_WORDS = {
    "turned away": "HEAD_TURNED_AWAY",
    "turned": "HEAD_TURNED_AWAY",
    "bent": "HEAD_BENT_DOWN",
    "bent over": "HEAD_BENT_DOWN",
    "cropped off": "TRUNCATION",
    "not in the box": "TRUNCATION",
    "out of frame": "TRUNCATION",
    "outside the box": "TRUNCATION",
    "out of view": "TRUNCATION",
    "only partly in the box": "TRUNCATION",
    "distant": "POOR_VIEW_ANGLE",
    "ambiguous": "POOR_VIEW_ANGLE",
    "unresolvable": "POOR_VIEW_ANGLE",
}


def cause_from_note(note: str) -> str:
    lowered = note.lower()
    for word, cause in POSE_WORDS.items():
        if word in lowered:
            return cause
    return "UNKNOWN_FAILURE_REASON"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/kitchen-01")
    tag = sys.argv[2] if len(sys.argv) > 2 else "phase42"

    truth = {
        (f.frame_id, s.subject_id): s
        for f in load_annotations(root / "annotations" / f"{root.name}.json")
        for s in f.subjects
    }
    result = json.loads((root / "results" / f"{tag}.json").read_text(encoding="utf-8"))
    head = next(a for a in result["attributes"] if a["attribute"] == "head_covering")
    failures = [f for f in result["failures"] if f["attribute"] == "head_covering"]

    print(f"=== {tag}: head_covering ===\n")

    confusion = head["confusion"]
    print("Ground-truth NOT_VISIBLE heads (13) — what the system said:")
    row = confusion.get("not_visible", {})
    for state, count in sorted(row.items()):
        print(f"    -> {state:<14} {count}")

    # Did the gate stop it, or did the model see it and fail?
    gated = [f for f in failures if f["quality"] not in ("passed", "")]
    passed = [f for f in failures if f["quality"] == "passed"]
    print(f"\n  rejected by the quality gate : {len(gated)}")
    print(f"  passed the gate, still wrong : {len(passed)}")

    print("\nWhy a human could not read them (from the annotator's own note):")
    causes = Counter(
        cause_from_note(subject.note)
        for (_, _), subject in truth.items()
        if subject.attributes.get("head_covering")
        and subject.attributes["head_covering"].value == "not_visible"
    )
    for cause, count in causes.most_common():
        print(f"    {cause:<26} {count}")

    print("\nRemaining head failures, one line each:")
    for f in sorted(failures, key=lambda x: x["frame_id"]):
        subject = truth.get((f["frame_id"], f["subject_id"]))
        note = subject.note if subject else ""
        cause = (
            f["category"].upper()
            if f["quality"] not in ("passed", "")
            else cause_from_note(note) if f["truth"] == "not_visible"
            else f["category"].upper()
        )
        print(
            f"    {f['frame_id'].split('/')[-1]:<8} {f['subject_id']:<4}"
            f" truth={f['truth']:<12} said={f['predicted']:<12}"
            f" crop={f['crop_size']:<10} quality={f['quality']:<12} {cause}"
        )
        if note:
            print(f"             note: {note}")

    false_absent = confusion.get("present", {}).get("absent", 0)
    print(f"\nFALSE ABSENT on a truly covered head: {false_absent}")
    print("(the production failure: worker is compliant, system reports a violation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
