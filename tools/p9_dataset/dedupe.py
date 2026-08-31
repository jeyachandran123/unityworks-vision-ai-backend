"""Temporal deduplication — remove redundant frames, never difficult ones.

    python -m tools.p9_dataset.dedupe

### The distinction that governs everything here

*Redundant* and *difficult* both look like "similar to something else". They are
not the same thing and confusing them destroys the dataset:

* **redundant** — the same worker, in the same posture, in the same place, two
  sampling intervals apart. Adds no information.
* **difficult** — a dark corner, a distant figure, an occluded head. These
  *also* resemble each other, because hard cases cluster in appearance space.

So similarity alone is never sufficient grounds for removal. A frame is dropped
only when it is similar **and adjacent in time within the same camera and
session**. Two visually similar frames from different cameras, different
sessions, or far apart in the same session are kept — they are independent
observations of a recurring condition, which is exactly what a benchmark needs.

### The hash

A 64-bit difference hash: greyscale, resize to 9x8, compare each pixel with its
right neighbour. Robust to the JPEG noise and auto-exposure drift that dominate
CCTV, sensitive to a person moving. Pure `Pillow` + `numpy`, deterministic, no
new dependency.

### The threshold is measured, not assumed

`report()` prints the duplication rate at several Hamming thresholds so the
choice can be read off real data rather than defended in prose. If almost
everything is duplicate at the sampling interval, the interval is wrong — and
that is a finding about the collector, reported rather than silently patched.

### Event-aware classification (P9.6, Phase 12)

Perceptual similarity alone over-counts redundancy on precisely the frames the
dataset most needs. A distant worker who moves a full step may barely disturb a
9x8 hash, and a person entering at the edge of a 1920x1080 frame may not disturb
it at all — yet those are the entries, exits and occlusion transitions that
produce hard cases.

So when a frame's **sampling reason** is available, it overrides similarity.
Reasons that record a change in the person population or in what can be observed
are never removed for looking alike:

| reason | similarity may remove it | why |
|---|---|---|
| `person_entered`, `person_left`, `person_count_changed` | **no** | the population changed; the pixels are not the evidence |
| `occlusion_changed`, `region_transition` | **no** | observability changed, which is the axis the schema exists to separate |
| `periodic_heartbeat` | **no** | it is the record of stillness and of event-detector failure; deleting it destroys the measurement Phase 6 asks for |
| `manual_review` | **no** | a person asked for it |
| `bbox_changed`, `person_moved`, `scene_changed` | yes | the reasons most likely to fire on detector jitter or on a change too small to matter |

This yields the three-way split Phase 12 requires — exact duplicate, near
duplicate, meaningful temporal change — rather than a single similarity verdict.
"""

from __future__ import annotations

import argparse
import enum
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "p9-live"

#: Hamming distance at or below which two adjacent frames are the same picture.
#:
#: 64-bit hash. 0 is bit-identical; ~5 tolerates JPEG and exposure noise while
#: still separating a moved arm. Reported across a range by `report()` so the
#: value is a reading, not a preference.
DEFAULT_THRESHOLD = 5

#: Frames further apart than this within a session are never treated as
#: duplicates of each other, however similar they look. An empty kitchen at
#: 09:00 and an empty kitchen at 14:00 are two genuine observations of an empty
#: kitchen, and collapsing them would understate how often the cameras see
#: nothing — which is itself a property the dataset must represent.
MAX_ADJACENT_GAP = 3


#: Sampling reasons that similarity may never override.
#:
#: Each records a change in the person population or in what could be observed —
#: the two things a 64-bit hash of a downscaled greyscale image is worst at
#: seeing, and the two things this dataset exists to capture. Keeping them is
#: therefore not a concession to caution; it is the correction of a known blind
#: spot in the instrument doing the measuring.
PROTECTED_REASONS = frozenset(
    {
        "manual_review",
        "person_entered",
        "person_left",
        "person_count_changed",
        "occlusion_changed",
        "region_transition",
        "periodic_heartbeat",
    }
)


#: Protected reasons whose claim a person count can actually test.
#:
#: These assert the population changed, so an unchanged count is evidence
#: against them. The remaining protected reasons assert that the *same* people
#: changed state, where an unchanged count is the prediction rather than the
#: refutation — see `audit_rescues`.
COUNT_TESTABLE_REASONS = frozenset(
    {"person_entered", "person_left", "person_count_changed"}
)


class Redundancy(enum.Enum):
    """The three-way verdict Phase 12 requires, instead of duplicate/not."""

    UNIQUE = "unique"
    """Nothing recent resembles it."""

    EXACT_DUPLICATE = "exact_duplicate"
    """Bit-identical hash to a recent frame, and no protected event to explain
    why it should be kept anyway."""

    NEAR_DUPLICATE = "near_duplicate"
    """Within the Hamming threshold of a recent frame, unprotected."""

    MEANINGFUL_CHANGE = "meaningful_change"
    """Looks like a recent frame **and is kept regardless**, because its
    sampling reason records a change the hash cannot see. This is the category
    that separates P9.6 from a similarity filter: without it, a distant worker
    stepping across a wide shot is indistinguishable from a still image."""


@dataclass(frozen=True, slots=True)
class FrameHash:
    path: Path
    camera_id: str
    session_id: str
    order: int
    bits: int
    reason: str = ""
    """The sampling reason, when the frame came from the event sampler. Empty
    for the P9.5 wall-clock corpus, which had no reasons — and where the absence
    is itself the finding."""

    people: int = -1
    """Confirmed person count, or -1 when unknown. Used only by `audit_rescues`
    to check the protection rule against evidence."""

    sample_class: str = ""
    """`event`, `baseline`, or empty for the P9.5 corpus that predates the
    distinction. Metrics for the two classes are reported apart and never
    averaged: a baseline frame is the record of stillness and is *supposed* to
    be empty, so scoring it on person yield penalises it for working."""


def dhash_image(image) -> int:
    """64-bit difference hash of an in-memory PIL image.

    Split out from `dhash` so the live collector can hash a frame it is holding
    without a round trip through the filesystem — and so both paths provably
    produce the same number, which they must, or a frame's hash would depend on
    whether it had been saved yet.
    """
    import numpy as np
    from PIL import Image

    small = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def dhash(path: Path) -> int:
    """64-bit difference hash of a frame on disk."""
    from PIL import Image

    with Image.open(path) as image:
        return dhash_image(image)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def frame_key(path) -> str:
    """A stable identifier for a frame.

    Repo-relative where possible, absolute otherwise. Deduplication must not
    depend on where the corpus happens to live — a frame outside the repository
    is still a frame, and crashing on one would make the tool unusable against
    an archived collection.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except (ValueError, TypeError):
        return str(path)


def session_reasons(session_dir: Path) -> dict[str, dict]:
    """Map frame filename -> its sample record, from the session's own manifest.

    A P9.5 session predates sampling reasons and yields an empty map, which is
    the correct answer for it rather than an error: the corpus is heterogeneous
    by history, and the loader has to read both halves.
    """
    record = session_dir / "session.json"
    if not record.exists():
        return {}
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        sample["file"]: sample
        for camera in payload.get("cameras", [])
        for sample in camera.get("samples", [])
    }


def load_frames(root: Path = LIVE) -> list[FrameHash]:
    out: list[FrameHash] = []
    for session_dir in sorted(p for p in root.glob("live-*") if p.is_dir()):
        samples = session_reasons(session_dir)
        for camera_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            for order, path in enumerate(sorted(camera_dir.glob("*.jpg"))):
                sample = samples.get(path.name, {})
                out.append(
                    FrameHash(
                        path=path,
                        camera_id=camera_dir.name,
                        session_id=session_dir.name,
                        order=order,
                        bits=dhash(path),
                        reason=sample.get("sampling_reason", ""),
                        people=sample.get("people_detected", -1),
                        sample_class=sample.get(
                            "sample_class",
                            # Derived for P9.6 Phase 1 records, which recorded a
                            # reason but not yet a class.
                            "baseline"
                            if sample.get("sampling_reason") == "periodic_heartbeat"
                            else "event"
                            if sample.get("sampling_reason")
                            else "",
                        ),
                    )
                )
    return out


def find_duplicates(
    frames: list[FrameHash],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_gap: int = MAX_ADJACENT_GAP,
) -> dict[str, str]:
    """Map duplicate frame -> the earlier frame it repeats.

    Compared only within the same camera **and** session, and only across a
    bounded index gap. That triple restriction is what keeps a recurring hard
    condition from being deleted as a repeat of itself.
    """
    duplicates: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[FrameHash]] = {}
    for frame in frames:
        grouped.setdefault((frame.session_id, frame.camera_id), []).append(frame)

    for group in grouped.values():
        group.sort(key=lambda f: f.order)
        kept: list[FrameHash] = []
        for frame in group:
            recent = [k for k in kept if frame.order - k.order <= max_gap]
            match = next((k for k in recent if hamming(frame.bits, k.bits) <= threshold), None)
            if match is None:
                kept.append(frame)
            else:
                duplicates[frame_key(frame.path)] = frame_key(match.path)
    return duplicates


def classify(
    frames: list[FrameHash],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_gap: int = MAX_ADJACENT_GAP,
) -> dict[str, Redundancy]:
    """Label every frame `UNIQUE`, `EXACT_`/`NEAR_DUPLICATE` or `MEANINGFUL_CHANGE`.

    Same triple restriction as `find_duplicates` — one camera, one session, a
    bounded index gap — with the sampling reason consulted before removal.

    A frame classified `MEANINGFUL_CHANGE` **stays in the corpus and stays
    comparable**: later frames may be found redundant against it, because it is a
    genuine observation, not an exemption from the rules.
    """
    verdicts: dict[str, Redundancy] = {}
    grouped: dict[tuple[str, str], list[FrameHash]] = {}
    for frame in frames:
        grouped.setdefault((frame.session_id, frame.camera_id), []).append(frame)

    for group in grouped.values():
        group.sort(key=lambda f: f.order)
        kept: list[FrameHash] = []
        for frame in group:
            recent = [k for k in kept if frame.order - k.order <= max_gap]
            match = next(
                (k for k in recent if hamming(frame.bits, k.bits) <= threshold), None
            )
            if match is None:
                verdicts[frame_key(frame.path)] = Redundancy.UNIQUE
                kept.append(frame)
            elif frame.reason in PROTECTED_REASONS:
                verdicts[frame_key(frame.path)] = Redundancy.MEANINGFUL_CHANGE
                kept.append(frame)
            elif hamming(frame.bits, match.bits) == 0:
                verdicts[frame_key(frame.path)] = Redundancy.EXACT_DUPLICATE
            else:
                verdicts[frame_key(frame.path)] = Redundancy.NEAR_DUPLICATE
    return verdicts


def audit_rescues(
    frames: list[FrameHash],
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_gap: int = MAX_ADJACENT_GAP,
) -> dict:
    """Check the protection rule against evidence, rather than trusting it.

    `PROTECTED_REASONS` keeps a frame that looks like its neighbour. That is
    right when the population or the visible geometry really did change, and it
    is **redundancy laundering** when it did not — a rule that always fires is
    indistinguishable from no rule at all, and it would quietly re-admit exactly
    the duplicates the corpus is trying to shed.

    So this reports, for every rescued frame, whether the confirmed person count
    actually differs from the frame it resembled. A low corroborated share means
    the protection is too broad and should be narrowed — a finding about the
    rule, published rather than tuned away.

    ### What this instrument can and cannot test

    Person count only corroborates reasons that **claim** the population changed.
    `occlusion_changed` and `region_transition` describe the same people changing
    state, so an unchanged count is what they *predict*, not evidence against
    them; scoring them here would manufacture a failure. They are counted under
    `not_count_testable` and left unscored rather than scored wrongly.

    Even where it applies the check is necessary, not sufficient: two people
    swapping places is a real change no count can see.

    Frames whose person count is unknown (`people < 0`) are counted separately
    and never scored either way.
    """
    checked = changed = unchanged = unknown = 0
    untestable = 0
    by_reason: dict[str, dict[str, int]] = {}
    grouped: dict[tuple[str, str], list[FrameHash]] = {}
    for frame in frames:
        grouped.setdefault((frame.session_id, frame.camera_id), []).append(frame)

    for group in grouped.values():
        group.sort(key=lambda f: f.order)
        kept: list[FrameHash] = []
        for frame in group:
            recent = [k for k in kept if frame.order - k.order <= max_gap]
            match = next(
                (k for k in recent if hamming(frame.bits, k.bits) <= threshold), None
            )
            if match is None:
                kept.append(frame)
                continue
            if frame.reason not in PROTECTED_REASONS:
                continue
            kept.append(frame)
            checked += 1
            entry = by_reason.setdefault(
                frame.reason,
                {
                    "rescued": 0,
                    "count_changed": 0,
                    "count_unchanged": 0,
                    "count_testable": frame.reason in COUNT_TESTABLE_REASONS,
                },
            )
            entry["rescued"] += 1
            if frame.reason not in COUNT_TESTABLE_REASONS:
                untestable += 1
            elif frame.people < 0 or match.people < 0:
                unknown += 1
            elif frame.people != match.people:
                changed += 1
                entry["count_changed"] += 1
            else:
                unchanged += 1
                entry["count_unchanged"] += 1

    scored = changed + unchanged
    return {
        "_comment": [
            "Does the protection rule earn its keep? A rescued frame is",
            "corroborated when the confirmed person count differs from the frame",
            "it resembled. A low corroborated share means the rule is too broad.",
            "Scored ONLY for reasons that claim the population changed: an",
            "unchanged count is what occlusion_changed and region_transition",
            "predict, so scoring them here would manufacture a failure.",
            "Necessary, not sufficient: two people swapping places is a real",
            "change no count can see.",
        ],
        "rescued": checked,
        "count_testable": scored + unknown,
        "not_count_testable": untestable,
        "corroborated_by_count_change": changed,
        "count_unchanged": unchanged,
        "count_unknown": unknown,
        "corroborated_rate": round(changed / scored, 4) if scored else None,
        "by_reason": dict(sorted(by_reason.items())),
    }


def classification_summary(verdicts: dict[str, Redundancy]) -> dict:
    """The counts Phase 12 asks for, plus the rates that make them readable."""
    counts = {verdict: 0 for verdict in Redundancy}
    for verdict in verdicts.values():
        counts[verdict] += 1
    total = len(verdicts)
    removed = counts[Redundancy.EXACT_DUPLICATE] + counts[Redundancy.NEAR_DUPLICATE]
    return {
        "frames": total,
        "unique": counts[Redundancy.UNIQUE],
        "exact_duplicates": counts[Redundancy.EXACT_DUPLICATE],
        "near_duplicates": counts[Redundancy.NEAR_DUPLICATE],
        "meaningful_change": counts[Redundancy.MEANINGFUL_CHANGE],
        "removed": removed,
        "retained": total - removed,
        "removal_rate": round(removed / total, 4) if total else None,
        "rescued_by_event": counts[Redundancy.MEANINGFUL_CHANGE],
        "rescued_rate": (
            round(counts[Redundancy.MEANINGFUL_CHANGE] / total, 4) if total else None
        ),
    }


def report(frames: list[FrameHash] | None = None) -> dict:
    frames = frames if frames is not None else load_frames()
    if not frames:
        return {"frames": 0, "note": "no live frames collected yet"}

    sweep = {}
    for threshold in (0, 2, 5, 8, 12, 16):
        found = find_duplicates(frames, threshold=threshold)
        sweep[threshold] = {
            "duplicates": len(found),
            "rate": round(len(found) / len(frames), 4),
        }

    chosen = find_duplicates(frames)
    by_camera: dict[str, dict[str, int]] = {}
    for frame in frames:
        entry = by_camera.setdefault(frame.camera_id, {"collected": 0, "duplicate": 0})
        entry["collected"] += 1
        if frame_key(frame.path) in chosen:
            entry["duplicate"] += 1
    for entry in by_camera.values():
        entry["retained"] = entry["collected"] - entry["duplicate"]

    return {
        "_comment": [
            "Duplication measured, not assumed. The threshold sweep is the",
            "evidence for the chosen threshold; if the rate is high at the",
            "sampling interval, the INTERVAL is wrong and that is a finding",
            "about the collector rather than something to tune away here.",
            "Frames are compared only within one camera AND session AND a",
            "bounded index gap, so a recurring hard condition is never deleted",
            "as a repeat of itself.",
        ],
        "frames_collected": len(frames),
        "threshold_used": DEFAULT_THRESHOLD,
        "max_adjacent_gap": MAX_ADJACENT_GAP,
        "threshold_sweep": sweep,
        "duplicates": len(chosen),
        "retained": len(frames) - len(chosen),
        "duplicate_rate": round(len(chosen) / len(frames), 4),
        "by_camera": by_camera,
        "duplicate_map": chosen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = report()
    if not result.get("frames_collected"):
        print("no live frames found under", LIVE)
        return 1

    print(f"frames collected : {result['frames_collected']}")
    print(f"duplicates       : {result['duplicates']}  ({result['duplicate_rate']:.1%})")
    print(f"retained         : {result['retained']}")
    print("\nthreshold sweep (Hamming distance -> duplicate rate):")
    for threshold, entry in result["threshold_sweep"].items():
        print(f"  <= {threshold:2d}  {entry['duplicates']:4d}  {entry['rate']:.1%}")
    print("\nby camera:")
    for camera, entry in sorted(result["by_camera"].items()):
        print(
            f"  {camera}  collected={entry['collected']:4d} "
            f"duplicate={entry['duplicate']:4d} retained={entry['retained']:4d}"
        )

    if args.write:
        path = LIVE / "deduplication.json"
        path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
