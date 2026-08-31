"""Build, split, count and freeze a P9 dataset version.

A dataset version is **immutable**. `build` writes `manifest.json` with a content
digest over every sample; `verify` recomputes it. Changing a label produces a new
version or a failed verification — never a silently mutated `P9-v1`.

### Splitting

Group-aware, by `session::subject`, allocated whole. The splitter will **refuse**
to produce a split it cannot make honest:

* fewer groups than requested partitions → refuses rather than emitting an empty
  or single-group partition that would look like a split and measure nothing;
* a group that would straddle partitions → impossible by construction, since
  allocation is per group.

Allocation is deterministic — groups are sorted and hashed with a fixed seed —
so two people building `P9-v1` from the same annotations get the same split.
"""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .schema import (
    ANNOTATION_SCHEMA_VERSION,
    Region,
    Split,
    SubjectAnnotation,
)
from .validate import errors_only, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"


def digest_of(subjects: Sequence[SubjectAnnotation]) -> str:
    """Content digest over the annotations, order-independent.

    Sorted by `sample_id` before hashing so a reordered file is the same dataset,
    and a changed label is not.
    """
    payload = json.dumps(
        [s.to_dict() for s in sorted(subjects, key=lambda s: s.sample_id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


class SplitRefused(RuntimeError):
    """The requested split cannot be made honestly from this data."""


def assign_splits(
    subjects: Sequence[SubjectAnnotation],
    *,
    ratios: dict[Split, float] | None = None,
    seed: str = "P9",
    hard_test_groups: Iterable[str] = (),
) -> list[SubjectAnnotation]:
    """Allocate whole `session::subject` groups to splits, deterministically.

    Args:
        hard_test_groups: groups pinned to HARD_TEST regardless of the ratios.
            Safety-critical cases are chosen, not sampled.

    Raises:
        SplitRefused: there are fewer groups than partitions. A "split" that puts
            every group in one partition is not a split, and returning one would
            let a leakage check pass on a dataset that cannot support evaluation.
    """
    from dataclasses import replace

    ratios = ratios or {Split.TRAIN: 0.5, Split.VALIDATION: 0.2, Split.TEST: 0.3}
    pinned = set(hard_test_groups)

    groups = sorted({s.group_key for s in subjects})
    free = [g for g in groups if g not in pinned]

    wanted = [s for s, r in ratios.items() if r > 0]
    if len(free) < len(wanted):
        raise SplitRefused(
            f"{len(free)} allocatable group(s) for {len(wanted)} partitions "
            f"({[s.value for s in wanted]}). A partition with no independent "
            f"group measures nothing; collect more subjects or sessions before "
            f"splitting. Groups present: {groups}"
        )

    ordered = sorted(
        free, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
    )
    placement: dict[str, Split] = {g: Split.HARD_TEST for g in pinned}
    cursor = 0
    total = sum(ratios[s] for s in wanted)
    for index, split in enumerate(wanted):
        share = ratios[split] / total
        take = len(free) - cursor if index == len(wanted) - 1 else round(len(free) * share)
        for group in ordered[cursor : cursor + take]:
            placement[group] = split
        cursor += take

    return [replace(s, split=placement.get(s.group_key, Split.UNASSIGNED)) for s in subjects]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Statistics:
    payload: dict

    def as_dict(self) -> dict:
        return self.payload


def statistics(subjects: Sequence[SubjectAnnotation]) -> Statistics:
    """Every distribution the P9 report has to publish."""
    by_region: dict[str, dict[str, int]] = {}
    for region in Region:
        counts: collections.Counter = collections.Counter()
        for subject in subjects:
            annotation = subject.region_of(region)
            if annotation is None:
                counts["not_annotated"] += 1
                continue
            counts[f"obs:{annotation.observability.value}"] += 1
            counts[f"state:{annotation.state.value}"] += 1
        by_region[region.value] = dict(sorted(counts.items()))

    tags: collections.Counter = collections.Counter()
    for subject in subjects:
        for annotation in subject.regions:
            tags.update(annotation.hard_case_tags)

    return Statistics(
        {
            "subjects": len(subjects),
            "frames": len({s.frame_id for s in subjects}),
            "groups": len({s.group_key for s in subjects}),
            "subject_ids": len({s.subject_id for s in subjects}),
            "sessions": sorted({s.session_id for s in subjects}),
            "cameras": sorted({s.camera_id for s in subjects}),
            "source_videos": sorted({s.source_video for s in subjects if s.source_video}),
            "by_split": dict(
                sorted(collections.Counter(s.split.value for s in subjects).items())
            ),
            "by_label_provenance": dict(
                sorted(
                    collections.Counter(
                        s.label_provenance.value for s in subjects
                    ).items()
                )
            ),
            "by_box_provenance": dict(
                sorted(
                    collections.Counter(s.box_provenance.value for s in subjects).items()
                )
            ),
            "by_quality_status": dict(
                sorted(
                    collections.Counter(s.quality_status.value for s in subjects).items()
                )
            ),
            "by_camera": dict(
                sorted(collections.Counter(s.camera_id for s in subjects).items())
            ),
            "by_region": by_region,
            "hard_case_tags": dict(tags.most_common()),
            "evaluation_grade": sum(1 for s in subjects if s.is_evaluation_grade),
            "ground_truth_violations": sum(
                1
                for s in subjects
                for r in s.regions
                if r.is_ground_truth_violation
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Build / verify
# --------------------------------------------------------------------------- #


def build(
    subjects: Sequence[SubjectAnnotation],
    *,
    version: str,
    notes: Sequence[str] = (),
    excluded: Sequence[dict] = (),
    strict: bool = True,
) -> dict:
    """Assemble an immutable dataset version.

    Args:
        strict: refuse to build when any ERROR-severity check fails. A dataset
            that cannot pass its own integrity checks must not acquire a version
            number, because a version number is what makes it citable.
    """
    findings = validate_manifest(subjects)
    errors = errors_only(findings)
    if strict and errors:
        raise ValueError(
            f"{len(errors)} integrity error(s); refusing to freeze:\n"
            + "\n".join(str(e) for e in errors[:20])
        )

    return {
        "_comment": [
            "An IMMUTABLE P9 dataset version.",
            "Any change to a label, a box or a split produces a NEW version.",
            "`digest` covers every annotation; `verify` recomputes it.",
            "Labels marked machine_proposed are NOT ground truth and are excluded",
            "from every evaluation split by the validator, not by convention.",
        ],
        "dataset_version": version,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "digest": digest_of(subjects),
        "statistics": statistics(subjects).as_dict(),
        "validation": {
            "errors": [str(e) for e in errors],
            "warnings": [str(e) for e in findings if e.severity == "warning"],
        },
        "excluded": list(excluded),
        "notes": list(notes),
        "samples": [s.to_dict() for s in sorted(subjects, key=lambda s: s.sample_id)],
    }


def write(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> list[SubjectAnnotation]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [SubjectAnnotation.from_dict(s) for s in document["samples"]]


def verify(path: Path) -> tuple[bool, str]:
    """Has this dataset version been mutated since it was frozen?"""
    document = json.loads(path.read_text(encoding="utf-8"))
    subjects = [SubjectAnnotation.from_dict(s) for s in document["samples"]]
    actual = digest_of(subjects)
    expected = document["digest"]
    if actual == expected:
        return True, f"{document['dataset_version']} intact ({actual[:16]})"
    return False, (
        f"{document['dataset_version']} HAS BEEN MUTATED\n"
        f"  frozen digest : {expected[:32]}\n"
        f"  actual digest : {actual[:32]}\n"
        f"  A dataset version is immutable. Create a new version instead."
    )
