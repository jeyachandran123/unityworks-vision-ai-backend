"""Assembling every adapter into one answer, and deciding what may be compared.

### Nothing here computes a headline

There is deliberately no "model health score". Four families measure four
different things on two different denominators, and any single number combining
them would be a figure with no definition — the one thing this phase must not
produce. The API returns the families; the reader draws the conclusion.

### Comparability is decided here, once

`comparison_sets` partitions runs into groups that may legitimately share an
axis, using `ComparabilityKey`. The frontend receives the partition and can only
draw within a group; it is never handed a flat list of runs and left to decide.

The partition is deliberately strict. In this repository it produces:

    the four llama-3.2 prompt variants   → one comparable set (same corpus digest)
    the minimax probe                    → alone (different model, no usable answers)
    the four PPE configuration runs      → one set per configuration string
    the regression recording             → alone (a different measurement entirely)

The PPE runs share a dataset and a model binding but each varied a configuration
knob, so they compare *settings* rather than *time* — and none of them records a
date, which is why the UI shows them as a configuration comparison and not a
trend.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.adapters import datasets as dataset_adapter
from app.evaluation.adapters import policy as policy_adapter
from app.evaluation.adapters import ppe as ppe_adapter
from app.evaluation.adapters import regression as regression_adapter
from app.evaluation.adapters import vlm_prompt as vlm_adapter
from app.evaluation.model import ArtifactFamily, EvaluationRun

#: Every adapter, in the order the dashboard presents them. Adding a source is
#: adding one line here plus a module — the frontend does not change.
FAMILY_LOADERS = (
    ppe_adapter.load,
    vlm_adapter.load,
    regression_adapter.load,
)


def families() -> tuple[ArtifactFamily, ...]:
    """Every artifact family, each stating its own availability."""
    return tuple(load() for load in FAMILY_LOADERS)


def all_runs(loaded: tuple[ArtifactFamily, ...]) -> tuple[EvaluationRun, ...]:
    return tuple(run for family in loaded for run in family.runs)


def comparison_sets(loaded: tuple[ArtifactFamily, ...]) -> list[dict[str, Any]]:
    """Groups of runs that may share an axis, with why they are grouped.

    Every returned group carries the key that made it a group, so a client can
    render *why* two runs sit together rather than asking the reader to trust
    the grouping.
    """
    buckets: dict[tuple, list[EvaluationRun]] = {}
    solo: list[EvaluationRun] = []

    for run in all_runs(loaded):
        if not run.available:
            continue
        key = run.comparability
        if not key.configuration:
            # An unrecorded configuration cannot be shown to match anything. It
            # gets its own group rather than being joined to whatever else
            # happens to share a model.
            solo.append(run)
            continue
        buckets.setdefault(
            (key.metric_kind, key.model, key.dataset, key.split, key.configuration), []
        ).append(run)

    groups: list[dict[str, Any]] = []
    for bucket, members in buckets.items():
        metric_kind, model, dataset, split, configuration = bucket
        # A single run is not a comparison. Labelled as a standalone snapshot so
        # a client never renders a one-point "trend".
        groups.append(
            {
                "key": "|".join(bucket),
                "metric_kind": metric_kind,
                "model": model,
                "dataset": dataset,
                "split": split,
                "configuration": configuration,
                "run_ids": [r.run_id for r in members],
                "comparable": len(members) > 1,
                "dated": all(r.provenance.evaluated_at is not None for r in members),
                "why": _why(members),
            }
        )

    for run in solo:
        groups.append(
            {
                "key": f"unpaired|{run.run_id}",
                "metric_kind": run.comparability.metric_kind,
                "model": run.comparability.model,
                "dataset": run.comparability.dataset,
                "split": run.comparability.split,
                "configuration": "",
                "run_ids": [run.run_id],
                "comparable": False,
                "dated": run.provenance.evaluated_at is not None,
                "why": (
                    "This run records no configuration identity, so it cannot be "
                    "shown to match any other. Displayed on its own."
                ),
            }
        )

    return groups


def _why(members: list[EvaluationRun]) -> str:
    if len(members) == 1:
        return (
            "The only run with this combination of metric, model, dataset and "
            "configuration. A single point is a snapshot, not a trend."
        )
    dated = [r for r in members if r.provenance.evaluated_at is not None]
    if len(dated) != len(members):
        return (
            f"{len(members)} runs share a metric, model, dataset and "
            "configuration, so their figures are directly comparable. They "
            "record no evaluation dates, so they compare configurations rather "
            "than time and are not plotted as a series."
        )
    return (
        f"{len(members)} runs share a metric, model, dataset and configuration "
        "and each records its own evaluation date. Directly comparable, and "
        "shown as discrete evaluation runs rather than continuous monitoring."
    )


def summary() -> dict[str, Any]:
    """The whole dashboard payload, assembled once."""
    loaded = families()
    coverages = dataset_adapter.load()
    policy_available, policy_reason, policy_groups = policy_adapter.load()

    runs = all_runs(loaded)
    dated = [r for r in runs if r.available and r.provenance.evaluated_at is not None]
    latest = max((r.provenance.evaluated_at for r in dated), default=None)

    return {
        "families": [f.as_dict() for f in loaded],
        "datasets": [c.as_dict() for c in coverages],
        "configuration": {
            "available": policy_available,
            "reason": policy_reason,
            "groups": [g.as_dict() for g in policy_groups],
        },
        "comparison_sets": comparison_sets(loaded),
        "totals": {
            "families": len(loaded),
            "families_available": sum(1 for f in loaded if f.available),
            "runs": len(runs),
            "runs_available": sum(1 for r in runs if r.available),
            # Counted, and named for what it is: how many artifacts carry a real
            # evaluation date. The rest are undated, which is not the same as old.
            "runs_dated": len(dated),
            "runs_undated": sum(1 for r in runs if r.available and r.provenance.evaluated_at is None),
        },
        "latest_evaluation_at": latest.isoformat() if latest else None,
        # Stated in the payload so no client has to decide it. There is no
        # aggregate score here, and this says why.
        "headline_metric": None,
        "headline_reason": (
            "No single figure summarises these artifacts. They measure different "
            "things — attribute agreement against annotation, prompt-variant "
            "agreement over parsed responses, and recorded-answer counts — over "
            "different denominators. Any combined score would be a number with "
            "no definition."
        ),
    }


__all__ = ["FAMILY_LOADERS", "all_runs", "comparison_sets", "families", "summary"]
