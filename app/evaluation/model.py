"""The normalized shape. **A metric cannot exist here without its provenance.**

### Why `Provenance` is a required field and not a nice-to-have

An evaluation number is meaningless without four things: what produced it, what
was evaluated, on what data, and when. A dashboard that shows `0.23` under the
word "accuracy" and nothing else invites exactly one reaction — that the model is
23% accurate — when the truth in this repository is that *head-covering
attribute agreement on 43 human-annotated subjects from one 15-frame test split,
against one prompt and one VLM build, with no recorded evaluation date* is 0.23.

Those are different claims. `MetricEntry` has no default for `provenance` and no
default for `definition`, so a value cannot reach the API without both.

### Absent provenance is data, not a gap to fill

Several real artifacts in this repository carry **no evaluation timestamp at
all** — `datasets/kitchen-01/results/*.json` among them. The honest response is
to say so. `Provenance.evaluated_at is None` with
`timestamp_source = "absent"` renders as "no evaluation date recorded", never as
the file's modification time, which records when git touched it rather than when
anybody measured anything.

### Comparability is a property of a pair, never of a chart

`ComparabilityKey` exists so that two runs can be asked whether they may share an
axis. Two evaluations belong on one trend only if their metric definition, model,
dataset, split and configuration all agree. Anything less is two snapshots, and
this module makes that a computed answer rather than a design decision somebody
makes at chart-drawing time.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class MetricKind(enum.Enum):
    """What a number *means*. Never a label chosen for how it reads.

    The distinctions here are the ones this repository's own artifacts make, and
    conflating any two of them would be inventing a claim:

    * `ATTRIBUTE_AGREEMENT` is how often the system's attribute answer matched a
      human annotation. `tools/vision_eval/metrics.py` calls it `accuracy`, and
      it is accuracy **of one attribute against one annotated split** — not
      "model accuracy", and emphatically not mAP.
    * `ACCURACY_OVER_PARSED` is the VLM experiment's own name for agreement
      computed only over responses that parsed. Runs with unparsed answers have a
      different denominator, which is exactly why it has its own name.
    * `DETECTION_COUNT` is a count of boxes. `datasets/kitchen-01/dataset.json`
      states in its own notes that the set **cannot measure detection recall**,
      so no recall is derived from it here.
    * `PINNED_COUNT` is a regression measurement pinned at a known value — a
      number that must not change silently. It is not a pass rate.
    """

    ATTRIBUTE_AGREEMENT = "attribute_agreement"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ACCURACY_OVER_PARSED = "accuracy_over_parsed"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    DETECTION_COUNT = "detection_count"
    LATENCY_MS = "latency_ms"
    THROUGHPUT_FPS = "throughput_fps"
    COUNT = "count"
    RATIO = "ratio"
    PINNED_COUNT = "pinned_count"
    CONFIGURED_THRESHOLD = "configured_threshold"


class Freshness(enum.Enum):
    """How much weight a reader should give an artifact's date.

    `UNDATED` is its own state rather than "very old". An artifact with no
    recorded evaluation date is not stale — nobody knows what it is, and a UI
    that greyed it out like an old one would be asserting something false.
    """

    UNDATED = "undated"
    RECENT = "recent"
    AGEING = "ageing"
    STALE = "stale"


#: Boundaries in days. Chosen to be visibly arbitrary rather than dressed up as
#: an SLA: this repository defines no evaluation cadence, so these describe how
#: far in the past a date is and nothing more. The UI says exactly that.
FRESHNESS_DAYS = {"recent": 30, "ageing": 120}


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a number came from. Required on every metric.

    Every field that is unknown is `None` or empty **and says so** — none of them
    is inferred, defaulted or reconstructed from the filesystem.
    """

    #: Repository-relative path of the file this was read from. A path inside the
    #: repository, never an absolute one: an API response has no business
    #: revealing where the checkout lives.
    artifact: str
    #: The adapter family — `ppe_evaluation`, `vlm_prompt`, `dataset_manifest`…
    source: str
    #: The run's own identity as the artifact names it (`baseline`, `variant_A`),
    #: or empty when the artifact names none.
    run_id: str = ""
    #: What was evaluated. `meta/llama-3.2-11b-vision-instruct`, `yolov8n.onnx`.
    model: str = ""
    #: Configuration identity — a policy id and version, a prompt digest.
    configuration: str = ""
    #: Dataset and split, as the artifact records them.
    dataset: str = ""
    split: str = ""
    #: **Read from inside the artifact only.** Never a file modification time.
    evaluated_at: datetime | None = None
    #: `artifact` when a real timestamp was found, `absent` when none exists.
    #: The second is a fact worth rendering, not a blank to hide.
    timestamp_source: str = "absent"
    #: Sample size behind the number.
    sample_size: int | None = None
    #: Anything the artifact itself says limits its interpretation. Copied
    #: verbatim from the source — `dataset.json`'s own note that the set cannot
    #: measure detection recall is the reason this field exists.
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "source": self.source,
            "run_id": self.run_id,
            "model": self.model,
            "configuration": self.configuration,
            "dataset": self.dataset,
            "split": self.split,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "timestamp_source": self.timestamp_source,
            "sample_size": self.sample_size,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ComparabilityKey:
    """What must match before two results may share an axis.

    Deliberately coarse and deliberately strict. Two runs are comparable only if
    every field agrees; anything else and the UI shows them as separate
    snapshots. A false negative here costs a chart. A false positive draws a line
    between a llama-3.2 run and a minimax run and calls it a trend.
    """

    metric_kind: str
    model: str
    dataset: str
    split: str
    #: The configuration digest the artifact itself records — a prompt sha, a
    #: policy version. Empty means the artifact did not record one, which by
    #: itself makes a run incomparable rather than comparable-by-default.
    configuration: str

    def comparable_with(self, other: ComparabilityKey) -> bool:
        if not self.configuration or not other.configuration:
            # An unrecorded configuration is not a matching one. Two runs that
            # both forgot to say how they were configured are not thereby the
            # same run.
            return False
        return self == other

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_kind": self.metric_kind,
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "configuration": self.configuration,
        }


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """One number, its meaning, and where it came from.

    `value is None` is a first-class state and means **undefined**, not zero.
    `experiments/vlm_prompt/score.py` says it plainly: *"a metric with no support
    is undefined, not bad, and printing a number there would be inventing one."*
    Recall over zero ground-truth examples is `None` here and renders as `—`.
    """

    key: str
    label: str
    kind: MetricKind
    value: float | int | None
    #: One sentence stating exactly what this number measures. Required — a
    #: metric whose definition nobody wrote down is a metric nobody can act on.
    definition: str
    provenance: Provenance
    #: `null` for a ratio that has no denominator; the reason is here.
    undefined_reason: str = ""
    unit: str = ""
    #: The count the value was computed over, when the artifact records one.
    support: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind.value,
            "value": self.value,
            "definition": self.definition,
            "undefined_reason": self.undefined_reason,
            "unit": self.unit,
            "support": self.support,
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class MetricGroup:
    """Metrics that belong on one screen together — one attribute, one view."""

    key: str
    title: str
    description: str
    metrics: tuple[MetricEntry, ...] = ()
    #: A confusion matrix as `{truth: {predicted: count}}`, when the artifact
    #: has one. Rendered as a table rather than a heat map: at this size the
    #: numbers are the point and a colour ramp would only hide them.
    confusion: dict[str, dict[str, int]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "metrics": [m.as_dict() for m in self.metrics],
            "confusion": self.confusion,
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One artifact, normalized. The unit the API serves and the UI lists."""

    run_id: str
    title: str
    summary: str
    source: str
    provenance: Provenance
    comparability: ComparabilityKey
    groups: tuple[MetricGroup, ...] = ()
    #: False when the artifact was found but could not be read, or is
    #: deliberately incomplete. Consumers branch on this before anything else,
    #: exactly as they do for observations and module capabilities.
    available: bool = True
    reason: str = ""
    #: `complete` or `partial`, from what the artifact itself says about its own
    #: coverage — never inferred from whether numbers are present.
    completeness: str = "complete"

    @property
    def freshness(self) -> Freshness:
        """Derived from the artifact's own recorded date, or `UNDATED`."""
        if self.provenance.evaluated_at is None:
            return Freshness.UNDATED

        from datetime import UTC

        age = (datetime.now(UTC) - self.provenance.evaluated_at).days
        if age <= FRESHNESS_DAYS["recent"]:
            return Freshness.RECENT
        if age <= FRESHNESS_DAYS["ageing"]:
            return Freshness.AGEING
        return Freshness.STALE

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "completeness": self.completeness,
            "freshness": self.freshness.value,
            "provenance": self.provenance.as_dict(),
            "comparability": self.comparability.as_dict(),
            "groups": [g.as_dict() for g in self.groups],
        }


@dataclass(frozen=True, slots=True)
class DatasetCoverage:
    """What a dataset actually contains, from its own manifest.

    `limitations` is the load-bearing field. `datasets/kitchen-01/dataset.json`
    states that the set cannot measure detection recall, and a coverage panel
    that omitted that while showing 43 annotated subjects would invite exactly
    the reading the note exists to prevent.
    """

    name: str
    artifact: str
    frames: int | None = None
    subjects: int | None = None
    #: `{split: [members]}` exactly as recorded. Empty lists are shown as empty
    #: rather than hidden: a dataset that is entirely a test split is a fact
    #: about what it can be used for.
    splits: dict[str, list[str]] = field(default_factory=dict)
    split_by: str = ""
    #: `{attribute: {state: count}}` — the label distribution. Zero support for
    #: a state is why several precision and recall figures are undefined.
    attribute_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    status: str = ""
    limitations: tuple[str, ...] = ()
    annotation_source: str = ""
    available: bool = True
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifact": self.artifact,
            "frames": self.frames,
            "subjects": self.subjects,
            "splits": self.splits,
            "split_by": self.split_by,
            "attribute_counts": self.attribute_counts,
            "status": self.status,
            "limitations": list(self.limitations),
            "annotation_source": self.annotation_source,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ArtifactFamily:
    """One adapter's worth of results, or an honest statement of its absence."""

    key: str
    title: str
    description: str
    available: bool
    reason: str = ""
    runs: tuple[EvaluationRun, ...] = ()
    #: Files the adapter looked for. Listed so a reader can see what is expected
    #: to exist, rather than guessing why a family is empty.
    expected_artifacts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "available": self.available,
            "reason": self.reason,
            "runs": [r.as_dict() for r in self.runs],
            "expected_artifacts": list(self.expected_artifacts),
        }


def comparable_groups(runs: tuple[EvaluationRun, ...]) -> list[list[str]]:
    """Partition runs into sets that may share an axis.

    Returned as explicit groups rather than as a boolean per pair, because the
    UI's question is "which of these can I put on one chart" and answering it
    per pair invites drawing a line through a chain of pairwise-compatible but
    collectively incomparable points.
    """
    buckets: dict[tuple, list[str]] = {}
    for run in runs:
        if not run.available:
            continue
        key = run.comparability
        if not key.configuration:
            # Its own group of one: an unrecorded configuration cannot be shown
            # to match anything, so it is never silently joined to a trend.
            buckets[("__unpaired__", run.run_id)] = [run.run_id]
            continue
        bucket = (key.metric_kind, key.model, key.dataset, key.split, key.configuration)
        buckets.setdefault(bucket, []).append(run.run_id)
    return list(buckets.values())


__all__ = [
    "FRESHNESS_DAYS",
    "ArtifactFamily",
    "ComparabilityKey",
    "DatasetCoverage",
    "EvaluationRun",
    "Freshness",
    "MetricEntry",
    "MetricGroup",
    "MetricKind",
    "Provenance",
    "comparable_groups",
]
