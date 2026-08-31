"""Coverage ledger and stopping criteria — P9.8.

    python -m tools.p9_dataset.coverage
    python -m tools.p9_dataset.coverage --write

### Why a ledger rather than a frame count

"Collect 5 GB" is not a stopping criterion; it is a budget. The question that
actually decides whether a corpus can support a benchmark is *what has been
observed*, and the honest unit of that is a **cell**: one camera, on one calendar
day, in one block of the working day.

P9.6 Phase 2 measured 13.90 candidates per camera-minute across the corpus, so
volume has never been the constraint. 26 sessions produced 3,250 candidates — and
23 of those sessions fell on a single calendar day, which is why the corpus still
cannot support a day-held-out split.

### The stopping criteria, stated as measurements

Each is a property of the ledger, not a judgement:

* **day spread** — no single calendar day holds more than `MAX_DAY_SHARE` of
  candidates. A corpus that is 95 % one morning describes that morning.
* **day count** — at least `MIN_DAYS` distinct days, so a day can be held out
  without taking the whole corpus with it.
* **block coverage** — at least `MIN_BLOCKS` distinct time blocks observed, so
  the corpus is not one lighting condition and one staffing pattern.
* **camera presence** — every camera present on at least `MIN_DAYS_PER_CAMERA`
  days, because a camera seen on one day cannot be distinguished from a camera
  with one unusual day.
* **group sufficiency** — at least `MIN_GROUPS` independent split groups.

None of these is about how many images exist.

### What the ledger cannot tell you

It counts observations, not people. Identity is unverified throughout this
programme, so a cell with 400 candidates may hold four workers or one worker
photographed four hundred times. The ledger reports cells; it never claims
subjects.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "p9-live"
OUT = LIVE / "coverage.json"

#: Blocks of the working day, in UTC hours. Local time is UTC+5:30, so these
#: correspond to early morning, morning, midday, afternoon and evening on site.
#: Named by clock, never by an asserted operating condition — P9.6 Phase 1
#: asserted shift labels over a 35-minute window and had to retract them.
BLOCKS = (
    ("00-03Z", range(0, 3)),
    ("03-06Z", range(3, 6)),
    ("06-09Z", range(6, 9)),
    ("09-12Z", range(9, 12)),
    ("12-15Z", range(12, 15)),
    ("15-18Z", range(15, 18)),
    ("18-24Z", range(18, 24)),
)

MIN_DAYS = 5
MAX_DAY_SHARE = 0.40
MIN_BLOCKS = 4
MIN_DAYS_PER_CAMERA = 3
MIN_GROUPS = 12
CAMERAS = ("cam-11", "cam-12", "cam-13", "cam-14")


def block_of(hour: int) -> str:
    for name, hours in BLOCKS:
        if hour in hours:
            return name
    return "unknown"


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    met: bool
    observed: str
    required: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "met": self.met,
            "observed": self.observed,
            "required": self.required,
        }


def _sessions() -> dict:
    out = {}
    for directory in sorted(LIVE.glob("live-*")):
        record = directory / "session.json"
        if record.exists():
            payload = json.loads(record.read_text(encoding="utf-8"))
            out[payload["session_id"]] = payload
    return out


def report() -> dict:
    sessions = _sessions()
    queue_path = LIVE / "review_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
    candidates = queue.get("candidates", [])

    when = {
        session_id: (
            payload.get("collection_day", ""),
            block_of(int(payload["collected_at"][11:13])),
        )
        for session_id, payload in sessions.items()
    }

    cells: dict[tuple[str, str, str], int] = collections.Counter()
    by_day: collections.Counter = collections.Counter()
    by_block: collections.Counter = collections.Counter()
    for candidate in candidates:
        day, block = when.get(candidate["session_id"], ("", "unknown"))
        cells[(day, block, candidate["camera_id"])] += 1
        by_day[day] += 1
        by_block[block] += 1

    camera_days: dict[str, set] = collections.defaultdict(set)
    for (day, _, camera), count in cells.items():
        if count:
            camera_days[camera].add(day)

    days = sorted({d for d, _ in when.values() if d})
    blocks = sorted({b for _, b in when.values()})
    total = sum(by_day.values())
    top_share = (max(by_day.values()) / total) if total else 0.0

    criteria = [
        Criterion(
            "calendar_days",
            len(days) >= MIN_DAYS,
            f"{len(days)} ({', '.join(days)})",
            f">= {MIN_DAYS}",
        ),
        Criterion(
            "no_day_dominates",
            top_share <= MAX_DAY_SHARE,
            f"{top_share:.1%} on {by_day.most_common(1)[0][0] if by_day else '-'}",
            f"<= {MAX_DAY_SHARE:.0%}",
        ),
        Criterion(
            "time_blocks",
            len(blocks) >= MIN_BLOCKS,
            f"{len(blocks)} ({', '.join(blocks)})",
            f">= {MIN_BLOCKS}",
        ),
        Criterion(
            "cameras_across_days",
            all(len(camera_days.get(c, ())) >= MIN_DAYS_PER_CAMERA for c in CAMERAS),
            ", ".join(f"{c}:{len(camera_days.get(c, ()))}d" for c in CAMERAS),
            f"every camera on >= {MIN_DAYS_PER_CAMERA} days",
        ),
        Criterion(
            "split_groups",
            len(sessions) >= MIN_GROUPS,
            f"{len(sessions)} sessions",
            f">= {MIN_GROUPS}",
        ),
    ]

    # Gaps, ordered by what would move the most criteria.
    observed_cells = {(d, b, c) for (d, b, c), n in cells.items() if n}
    gaps = [
        {"day": day, "block": block, "camera": camera}
        for day in days
        for block, _ in BLOCKS
        for camera in CAMERAS
        if (day, block, camera) not in observed_cells
    ]

    return {
        "_comment": [
            "P9.8 coverage ledger. The unit is a CELL: one camera, one calendar",
            "day, one block of the working day. Volume has never been the",
            "constraint — 13.90 candidates per camera-minute — so the criteria",
            "below are about spread, not count.",
            "Cells count OBSERVATIONS, never people. Identity is unverified, so a",
            "cell with 400 candidates may hold four workers or one worker",
            "photographed four hundred times.",
        ],
        "sessions": len(sessions),
        "candidates": total,
        "days": days,
        "blocks_observed": blocks,
        "by_day": dict(by_day.most_common()),
        "by_block": dict(by_block.most_common()),
        "camera_days": {c: sorted(camera_days.get(c, ())) for c in CAMERAS},
        "cells_observed": len(observed_cells),
        "cells_possible": len(days) * len(BLOCKS) * len(CAMERAS),
        "gaps": gaps[:60],
        "gap_count": len(gaps),
        "criteria": [c.as_dict() for c in criteria],
        "all_criteria_met": all(c.met for c in criteria),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    payload = report()
    print(f"sessions        : {payload['sessions']}")
    print(f"candidates      : {payload['candidates']:,}")
    print(f"calendar days   : {payload['days']}")
    print(f"blocks observed : {payload['blocks_observed']}")
    print(f"cells observed  : {payload['cells_observed']} of {payload['cells_possible']} possible")
    print()
    print("candidates by day:")
    for day, count in payload["by_day"].items():
        share = count / payload["candidates"] if payload["candidates"] else 0
        print(f"  {day or '(unknown)':12s} {count:6,}  {share:6.1%}")
    print()
    print("candidates by block (UTC):")
    for block, count in payload["by_block"].items():
        print(f"  {block:10s} {count:6,}")
    print()
    print("camera presence, in days:")
    for camera, days in payload["camera_days"].items():
        print(f"  {camera}  {len(days)} day(s)  {days}")
    print()
    print("STOPPING CRITERIA")
    for entry in payload["criteria"]:
        mark = "MET " if entry["met"] else "NOT "
        print(f"  [{mark}] {entry['name']:22s} observed {entry['observed']:38s} need {entry['required']}")
    print()
    print("ALL CRITERIA MET" if payload["all_criteria_met"] else "CRITERIA NOT MET — collection incomplete")

    if args.write:
        OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
