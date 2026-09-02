"""Dataset manifests — what was labelled, how it splits, and what it cannot measure.

Reads `datasets/kitchen-01/dataset.json` and `datasets/vision-phase5/manifest.json`.

### The `notes` field is the most important thing here

`kitchen-01/dataset.json` says, in its own words:

    "Boxes are detector proposals visually confirmed to be real people. People
     the detector never proposed were therefore never annotated, so this dataset
     CANNOT measure detection recall."

A coverage panel that showed "43 annotated subjects, 0 unmatched" without that
sentence would invite a reader to conclude the detector misses nobody. The note
is carried into `limitations` and rendered beside the counts, not behind them.

### An empty dataset is not a dataset scoring zero

`vision-phase5/manifest.json` records `status: "AWAITING FOOTAGE"`. It has a
purpose, a schema and an annotation policy, and no data. That is surfaced as its
recorded status with no counts at all — never as zeros, which would read as a
dataset that was labelled and found to contain nothing.

### Imagery is not touched

`datasets/kitchen-01/frames/` holds 4,036 JPEG frames of real people and
`experiments/vlm_prompt/crops/` holds 43 PNG crops. Nothing here reads, lists,
counts or references them. See `app/evaluation/adapters/__init__` and the phase
report for why no image surface was built.
"""

from __future__ import annotations

from pathlib import Path

from app.evaluation.adapters.common import ROOT, read_json, repo_relative, whole
from app.evaluation.model import DatasetCoverage

MANIFESTS: tuple[tuple[str, Path], ...] = (
    ("kitchen-01", ROOT / "datasets" / "kitchen-01" / "dataset.json"),
    ("vision-phase5", ROOT / "datasets" / "vision-phase5" / "manifest.json"),
)


def _splits(payload: dict) -> tuple[dict[str, list[str]], str]:
    split = payload.get("split")
    if not isinstance(split, dict):
        return {}, str(payload.get("split_by", "") or "")
    members = {
        name: [str(v) for v in value]
        for name, value in split.items()
        if isinstance(value, list)
    }
    return members, str(split.get("split_by") or payload.get("split_by") or "")


def _limitations(payload: dict) -> tuple[str, ...]:
    """Whatever the manifest says about its own limits, verbatim.

    Copied rather than summarised. These sentences were written by whoever built
    the dataset and they are more precise than anything this adapter could infer.
    """
    found: list[str] = []
    for key in ("notes", "_note", "limitations", "caveats"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
        elif isinstance(value, list):
            found.extend(str(v).strip() for v in value if str(v).strip())
    return tuple(found)


def load() -> tuple[DatasetCoverage, ...]:
    coverages: list[DatasetCoverage] = []

    for name, path in MANIFESTS:
        payload = read_json(path)
        if not isinstance(payload, dict):
            coverages.append(
                DatasetCoverage(
                    name=name,
                    artifact=repo_relative(path),
                    available=False,
                    reason=(
                        f"{repo_relative(path)} is missing or could not be read. "
                        "No counts are shown, because an unreadable manifest is "
                        "not an empty dataset."
                    ),
                )
            )
            continue

        splits, split_by = _splits(payload)
        counts = payload.get("attribute_counts")
        status = str(payload.get("status", "") or "")

        coverages.append(
            DatasetCoverage(
                name=str(payload.get("name") or name),
                artifact=repo_relative(path),
                # `None` rather than 0 when the manifest records no count. A
                # dataset awaiting footage has no frames, not zero frames.
                frames=whole(payload.get("frames")),
                subjects=whole(payload.get("subjects")),
                splits=splits,
                split_by=split_by,
                attribute_counts=counts if isinstance(counts, dict) else {},
                status=status,
                limitations=_limitations(payload),
                annotation_source=str(payload.get("annotation_source", "") or ""),
                available=True,
                reason=status if status else "",
            )
        )

    return tuple(coverages)


__all__ = ["MANIFESTS", "load"]
