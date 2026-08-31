"""Run a sequence of collection sessions — P9.6 Phases 7, 8 and 9.

    python -m tools.p9_dataset.campaign --plan default

### Why a driver rather than a shell loop

One process, so stopping it stops everything. A shell loop spawning collectors
leaves orphans when it is interrupted, and an orphaned collector holds four RTSP
sessions and a detector: the next measurement then reads the contention rather
than the change it was meant to test. That happened during this phase and cost a
misdiagnosis, so the loop lives here where it can be killed as a unit.

### Why many short sessions rather than a few long ones

The session is the group a split may not straddle — `SubjectAnnotation.group_key`
collapses an unverified session to a single indivisible group — so **sessions,
not frames, are the scarce resource**. P9.5 collected four and the spec derives
twelve as a working minimum, which is why the default plan is twelve short
sessions rather than four long ones.

### Why sparse cameras get their own sessions

cam-11 and cam-14 yielded 1.95 and 1.63 candidates per camera-minute against
cam-12's 19.26. Equalising that by duplicating frames is forbidden and would be
a lie about the corpus; the only legal correction is more observation time, so
the plan interleaves sessions that watch only the sparse pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: `(label, seconds, period, cameras)`.
#:
#: Periods are the operator's assertion about the operating condition. Nothing
#: verifies them, and the report presents them as assertions.
DEFAULT_PLAN = (
    ("e", 150, "morning-prep", [11, 12, 13, 14]),
    ("f", 210, "morning-prep", [11, 14]),
    ("g", 150, "morning-prep", [11, 12, 13, 14]),
    ("h", 150, "morning-service", [11, 12, 13, 14]),
    ("i", 210, "morning-service", [11, 14]),
    ("j", 150, "morning-service", [11, 12, 13, 14]),
    ("k", 150, "mid-morning", [11, 12, 13, 14]),
    ("l", 210, "mid-morning", [11, 14]),
    ("m", 150, "mid-morning", [11, 12, 13, 14]),
    ("n", 150, "mid-morning", [11, 12, 13, 14]),
    ("o", 210, "mid-morning", [11, 14]),
    ("p", 150, "mid-morning", [11, 12, 13, 14]),
)

#: P9.6 Phase 2's campaign, run under the policy the replay selected.
#:
#: Periods are **not named**. Phase 1 asserted `morning-prep` and `mid-morning`
#: over a 35-minute window, which the report had to retract as unsupported. A
#: label an operator types is not an observation, so these sessions carry only
#: the clock time they actually ran at, and the operating condition is
#: characterised afterwards from measured activity.
PHASE2_PLAN = tuple(
    (label, seconds, "recorded-by-clock", cameras)
    for label, seconds, cameras in (
        ("q", 150, [11, 12, 13, 14]),
        ("r", 210, [11, 14]),
        ("s", 150, [11, 12, 13, 14]),
        ("t", 150, [11, 12, 13, 14]),
        ("u", 210, [11, 14]),
        ("v", 150, [11, 12, 13, 14]),
        ("w", 150, [11, 12, 13, 14]),
        ("x", 210, [11, 14]),
        ("y", 150, [11, 12, 13, 14]),
        ("z", 150, [11, 12, 13, 14]),
    )
)

#: P9.8. Distinct labels so a same-day re-run cannot collide with `phase2`,
#: whose q..z labels are already spent on 2026-08-28.
P98_PLAN = tuple(
    (label, seconds, "recorded-by-clock", cameras)
    for label, seconds, cameras in (
        ("n1", 150, [11, 12, 13, 14]),
        ("n2", 210, [11, 14]),
        ("n3", 150, [11, 12, 13, 14]),
        ("n4", 150, [11, 12, 13, 14]),
        ("n5", 210, [11, 14]),
        ("n6", 150, [11, 12, 13, 14]),
        ("n7", 150, [11, 12, 13, 14]),
        ("n8", 150, [11, 12, 13, 14]),
    )
)

PLANS = {
    "default": DEFAULT_PLAN,
    "phase2": PHASE2_PLAN,
    "p98": P98_PLAN,
    "short": DEFAULT_PLAN[:2],
}


def run(plan, *, gap: float = 5.0) -> list[dict]:
    from .baselines import PHASE2_B
    from .collect import collect

    # The policy the Phase 2 replay selected, named explicitly rather than taken
    # from the current defaults — a corpus must record the policy it was
    # collected under, not whatever the defaults happen to be later.
    sampling = PHASE2_B
    records = []
    for label, seconds, period, cameras in plan:
        started = time.strftime("%H:%M:%S", time.gmtime())
        print(
            f"\n=== {label}  {period:16s} cams={cameras} {seconds}s  (started {started}Z)",
            flush=True,
        )
        record = collect(
            cameras,
            label=label,
            seconds=float(seconds),
            interval=3.0,
            max_frames=sampling.max_samples,
            sampling=sampling,
            # An observed clock time, not an asserted operating condition.
            period=(
                f"observed-{started[:5].replace(':', '')}Z"
                if period == "recorded-by-clock"
                else period
            ),
        )
        totals = record["totals"]
        print(
            f"    decoded={totals['frames_decoded']:6d} kept={totals['frames_kept']:4d} "
            f"event={totals['event_triggered']:4d} baseline={totals['baseline_triggered']:3d} "
            f"ok={totals['cameras_ok']}/{len(cameras)} "
            f"failed={totals['cameras_failed']} reconnects={totals['reconnects']}",
            flush=True,
        )
        print(f"    reasons: {totals['by_reason']}", flush=True)
        records.append(record)
        time.sleep(gap)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=sorted(PLANS), default="default")
    args = parser.parse_args()

    plan = PLANS[args.plan]
    budget = sum(entry[1] for entry in plan)
    print(f"plan '{args.plan}': {len(plan)} sessions, ~{budget / 60:.0f} minutes of observation")

    records = run(plan)
    print("\n=== campaign complete")
    print(f"sessions   : {len(records)}")
    print(f"decoded    : {sum(r['totals']['frames_decoded'] for r in records):,}")
    print(f"kept       : {sum(r['totals']['frames_kept'] for r in records)}")
    print(f"failures   : {sum(r['totals']['cameras_failed'] for r in records)}")
    print(f"reconnects : {sum(r['totals']['reconnects'] for r in records)}")
    summary = ROOT / "datasets" / "p9-live" / "campaign.json"
    summary.write_text(
        json.dumps(
            {
                "_comment": [
                    "P9.6 collection campaign. Sessions are the unit a split may",
                    "not straddle, so the plan buys sessions rather than frames.",
                    "Sparse-camera sessions exist because cam-11 and cam-14 yield",
                    "roughly a tenth of cam-12's candidates per camera-minute, and",
                    "the only legal correction is observation time.",
                ],
                "plan": [
                    {
                        "label": label,
                        "seconds": seconds,
                        "period": period,
                        "cameras": cameras,
                    }
                    for label, seconds, period, cameras in plan
                ],
                "sessions": [r["session_id"] for r in records],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
