"""Dataset layout and splitting.

### Why splitting is by video, never by frame

Consecutive CCTV frames are nearly identical. Randomly assigning individual
frames from one video to train and test puts near-duplicates on both sides, and
a model that has memorised a chef's jacket scores beautifully on footage it has
effectively already seen. The number is real; the generalisation it implies is
not.

So the unit of splitting is a **video**, and preferably a camera or a restaurant.
``split_by`` names the grouping key, and every frame carrying that key lands
wholly on one side. A caller asking for a split that would put one video in two
places gets an error rather than a leaky dataset.

### Layout

    datasets/<name>/
        videos/         source footage, referenced not copied
        frames/         extracted stills
        annotations/    human labels
        predictions/    what a run of Vision OS said
        results/        evaluation output
        dataset.json    manifest: what is here, and how it splits
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import AnnotatedFrame

#: Grouping keys a split may use, coarsest first.
#:
#: Coarser is safer. Splitting by restaurant means the evaluation set shares no
#: kitchen, no lighting and no uniforms with training; splitting by video still
#: shares the room. Use the coarsest the dataset can support while leaving
#: enough on both sides.
SPLIT_KEYS = ("restaurant_id", "camera_id", "video_id")


class LeakageError(ValueError):
    """A split would put the same group on both sides.

    Raised rather than silently tolerated: leakage inflates every metric
    downstream and is invisible in the result.
    """


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    split_by: str = "video_id"

    def __post_init__(self) -> None:
        if self.split_by not in SPLIT_KEYS:
            raise ValueError(f"split_by must be one of {SPLIT_KEYS}, got {self.split_by!r}")
        seen: dict[str, str] = {}
        for name, group in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            for key in group:
                if key in seen:
                    raise LeakageError(
                        f"{self.split_by} {key!r} appears in both {seen[key]} and "
                        f"{name}; consecutive CCTV frames are near-duplicates, so "
                        f"this would score the system on footage it trained on"
                    )
                seen[key] = name

    def side_of(self, frame: AnnotatedFrame) -> str | None:
        key = getattr(frame, self.split_by)
        if key in self.train:
            return "train"
        if key in self.validation:
            return "validation"
        if key in self.test:
            return "test"
        return None


@dataclass(frozen=True, slots=True)
class Dataset:
    """A named collection of annotated frames and how it divides."""

    name: str
    root: Path
    frames: tuple[AnnotatedFrame, ...] = ()
    split: DatasetSplit | None = None
    notes: str = ""

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(sorted({f.video_id for f in self.frames}))

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted({f.camera_id for f in self.frames if f.camera_id}))

    @property
    def restaurants(self) -> tuple[str, ...]:
        return tuple(sorted({f.restaurant_id for f in self.frames if f.restaurant_id}))

    @property
    def subject_count(self) -> int:
        return sum(len(f.subjects) for f in self.frames)

    def attribute_counts(self) -> dict[str, dict[str, int]]:
        """How many times each state was annotated, per attribute.

        The first thing to read before any metric: a recall computed over three
        examples is not a measurement, and this is what makes that visible.
        """
        counts: dict[str, dict[str, int]] = {}
        for frame in self.frames:
            for subject in frame.subjects:
                for key, state in subject.attributes.items():
                    counts.setdefault(key, {})
                    counts[key][state.value] = counts[key].get(state.value, 0) + 1
        return counts

    def side(self, name: str) -> tuple[AnnotatedFrame, ...]:
        if self.split is None:
            return self.frames if name == "train" else ()
        return tuple(f for f in self.frames if self.split.side_of(f) == name)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frames": len(self.frames),
            "subjects": self.subject_count,
            "videos": list(self.videos),
            "cameras": list(self.cameras),
            "restaurants": list(self.restaurants),
            "attribute_counts": self.attribute_counts(),
            "split_by": self.split.split_by if self.split else None,
            "notes": self.notes,
        }


def group_split(
    frames: Sequence[AnnotatedFrame],
    *,
    split_by: str = "video_id",
    test: Iterable[str] = (),
    validation: Iterable[str] = (),
) -> DatasetSplit:
    """Build a split, putting everything not named into train.

    Explicit rather than random. Which camera is held out is an engineering
    judgment — a deployment wants its hardest camera in the test set, and a
    shuffle cannot know that.
    """
    if split_by not in SPLIT_KEYS:
        raise ValueError(f"split_by must be one of {SPLIT_KEYS}")
    groups = {getattr(f, split_by) for f in frames}
    held = set(test) | set(validation)
    unknown = held - groups
    if unknown:
        raise ValueError(f"{split_by} {sorted(unknown)} are not present in the frames")
    return DatasetSplit(
        train=tuple(sorted(groups - held)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
        split_by=split_by,
    )


def save_dataset(dataset: Dataset) -> Path:
    manifest = dataset.root / "dataset.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset.describe()
    if dataset.split is not None:
        payload["split"] = {
            "split_by": dataset.split.split_by,
            "train": list(dataset.split.train),
            "validation": list(dataset.split.validation),
            "test": list(dataset.split.test),
        }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "SPLIT_KEYS",
    "Dataset",
    "DatasetSplit",
    "LeakageError",
    "group_split",
    "save_dataset",
]
