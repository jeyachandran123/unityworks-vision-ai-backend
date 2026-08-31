"""Refuse to let production CCTV enter Git — P9.7, Phase 5.

    python -m tools.p9_dataset.guard          # audit, exit 1 on new exposure
    python -m tools.p9_dataset.guard --json

### Why a guard and not just `.gitignore`

`.gitignore` only governs files Git is not already tracking. It is a rule about
*future* additions, it is easy to override with `git add -f`, and it says nothing
about what is already in history. The audit that opened P9.7 found exactly that
gap: `datasets/` held 723 MB of CCTV-derived frames with no ignore rule, and 153
image files were already committed.

So the ignore rules stop the next mistake and this guard reports the standing
one. The desired failure mode — a developer copies 5 GB of frames into the
working tree and the tooling refuses — needs something that actually runs.

### What counts as an exposure

A tracked or stageable file whose extension carries a recoverable image of a
person. **Not** manifests, digests, boxes, counts or reports: those are the
artefacts that make a dataset citable, they identify nobody, and a guard that
blocked them would push people to disable it.

### Known exposures are recorded, not excused

`KNOWN_TRACKED_PIXELS` lists what was already committed when the guard was
written. They are reported at every run and counted separately, so the number
can only go down deliberately. Adding to that list to silence the guard would
defeat it; the list exists so that *new* exposure is distinguishable from the
inherited one, and it is not a permission to add more.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .store import PIXEL_SUFFIXES, REPO, DatasetStore, StoreError

#: Directories whose committed images predate P9.7. Recorded, not forgiven.
#:
#: `datasets/kitchen-01` is the source of the only human PPE annotations this
#: programme has (P9-v1/v2, 43 subjects). Removing it from history is a decision
#: with consequences for reproducing that ground truth, so the guard reports it
#: and leaves the decision to a human — see the P9.7 report.
KNOWN_TRACKED_PIXELS = (
    "datasets/kitchen-01/",
    "datasets/vision-phase5/",
)


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _is_pixel(path: str) -> bool:
    return Path(path).suffix.lower() in PIXEL_SUFFIXES


def audit() -> dict:
    """Everything the guard can determine about CCTV exposure in this repo."""
    tracked = [p for p in _git("ls-files") if _is_pixel(p)]
    staged = [
        line[3:].strip('"')
        for line in _git("status", "--porcelain", "--untracked-files=all")
        if line[:2] in ("A ", "AM", "M ", "MM") and _is_pixel(line[3:])
    ]
    stageable = [
        line[3:].strip('"')
        for line in _git("status", "--porcelain", "--untracked-files=all")
        if line.startswith("??") and _is_pixel(line[3:])
    ]

    known = [p for p in tracked if any(p.startswith(k) for k in KNOWN_TRACKED_PIXELS)]
    new = sorted(set(tracked) - set(known))

    try:
        store = DatasetStore.resolve()
        root = str(store.root)
        root_error = ""
    except StoreError as error:
        root = ""
        root_error = str(error)

    return {
        "tracked_pixel_files": len(tracked),
        "known_pre_p97": len(known),
        "known_prefixes": list(KNOWN_TRACKED_PIXELS),
        "new_tracked": new,
        "staged_pixel_files": sorted(staged),
        "stageable_pixel_files": sorted(stageable),
        "store_root": root,
        "store_root_error": root_error,
        "ok": not new and not staged and not stageable and not root_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit()
    if args.json:
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1

    print(f"tracked image/video files : {report['tracked_pixel_files']}")
    print(f"  known, pre-P9.7         : {report['known_pre_p97']} "
          f"({', '.join(report['known_prefixes'])})")
    print(f"  NEW exposure            : {len(report['new_tracked'])}")
    for path in report["new_tracked"][:20]:
        print(f"      {path}")
    print(f"staged now                : {len(report['staged_pixel_files'])}")
    for path in report["staged_pixel_files"][:20]:
        print(f"      {path}")
    print(f"stageable (untracked)     : {len(report['stageable_pixel_files'])}")
    for path in report["stageable_pixel_files"][:20]:
        print(f"      {path}")
    print(f"store root                : {report['store_root'] or report['store_root_error']}")
    print()
    print("PASS — no new CCTV exposure" if report["ok"] else "FAIL — see above")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
