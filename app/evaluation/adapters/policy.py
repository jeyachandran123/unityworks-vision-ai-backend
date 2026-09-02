"""Configured thresholds — read from `config/policies/kitchen-safety*.json`.

These are not measurements. They are the values the running system is configured
with, and they belong on an evaluation dashboard because a metric computed at one
confidence threshold means something different at another.

Every number here is read from the policy document rather than restated, so the
dashboard cannot drift from what the deployment actually uses. `min_confidence`
is the policy's own key; `validity_ms` and `freshness_ms` are the attribute and
demand windows the same file declares.

Nothing here writes to a policy, and the compliance evaluator is not touched —
this reads the same file it reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.evaluation.adapters.common import ROOT, ratio, read_json, repo_relative, whole
from app.evaluation.model import MetricEntry, MetricGroup, MetricKind, Provenance

POLICIES = ROOT / "config" / "policies"

#: Read in a fixed order. `kitchen-safety.example.json` is the one the evaluation
#: runs name in their configuration; the head-variant is the alternative that
#: `kitchen-safety.head-variant.json` exists to compare against.
DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("kitchen-safety.example.json", "Kitchen safety (shipped example)"),
    ("kitchen-safety.head-variant.json", "Kitchen safety (head variant)"),
)


def _provenance(path: Path, payload: dict) -> Provenance:
    policy_id = str(payload.get("policy_id", "") or "")
    version = str(payload.get("version", "") or "")
    return Provenance(
        artifact=repo_relative(path),
        source="policy_configuration",
        run_id=f"{policy_id}@{version}" if policy_id else "",
        model="",
        configuration=f"{policy_id}@{version}" if policy_id else "not recorded",
        dataset="",
        split="",
        # A policy document is configuration, not an evaluation. It has no
        # evaluation date and it would be wrong to give it one.
        evaluated_at=None,
        timestamp_source="not_applicable",
        limitations=(
            "Configuration in force, not a measurement. A metric computed under "
            "one threshold does not describe behaviour under another.",
        ),
    )


def _group(path: Path, payload: dict) -> MetricGroup:
    prov = _provenance(path, payload)
    scope = payload.get("scope") or {}
    demand = payload.get("demand") or {}
    budget = demand.get("budget") or {}

    metrics: list[MetricEntry] = []

    def add(key: str, label: str, value: Any, definition: str, unit: str = "") -> None:
        if value is None:
            return
        metrics.append(
            MetricEntry(
                key=key,
                label=label,
                kind=MetricKind.CONFIGURED_THRESHOLD,
                value=value,
                definition=definition,
                provenance=prov,
                unit=unit,
            )
        )

    add(
        "scope.min_confidence",
        "Minimum detection confidence",
        ratio(scope.get("min_confidence")),
        "Detections below this score are not considered for attribute evaluation "
        "at all. Read from the policy's scope.min_confidence.",
    )
    add(
        "demand.freshness_ms",
        "Attribute freshness window",
        whole(demand.get("freshness_ms")),
        "How long an existing answer is reused before the understander is asked "
        "again. From the policy's demand.freshness_ms.",
        unit="ms",
    )
    add(
        "demand.max_calls_per_hour",
        "Model call budget",
        whole(budget.get("max_calls_per_hour")),
        "Ceiling on understander invocations per hour, from the policy's demand budget.",
        unit="calls/h",
    )

    # Per-attribute validity windows, which differ between attributes and are
    # part of what any agreement figure was measured under.
    for attribute in payload.get("attributes") or []:
        if not isinstance(attribute, dict):
            continue
        key = str(attribute.get("key", ""))
        add(
            f"attribute.{key}.validity_ms",
            f"{key.replace('_', ' ')} validity",
            whole(attribute.get("validity_ms")),
            f"How long an observed {key.replace('_', ' ')} value stays valid before "
            "it is treated as stale. From the policy's attribute declaration.",
            unit="ms",
        )

    return MetricGroup(
        key=repo_relative(path),
        title=f"{payload.get('policy_id', 'policy')} v{payload.get('version', '?')}",
        description=(
            "Thresholds the deployment is configured with. Evaluation figures "
            "elsewhere on this page were produced under these values."
        ),
        metrics=tuple(metrics),
    )


def load() -> tuple[bool, str, tuple[MetricGroup, ...]]:
    """`(available, reason, groups)` — thresholds, or an honest absence."""
    groups: list[MetricGroup] = []
    missing: list[str] = []

    for filename, _label in DOCUMENTS:
        path = POLICIES / filename
        payload = read_json(path)
        if not isinstance(payload, dict):
            missing.append(repo_relative(path))
            continue
        groups.append(_group(path, payload))

    if not groups:
        return (
            False,
            "No kitchen-safety policy document could be read, so no configured "
            f"threshold is shown. Expected: {', '.join(missing) or 'config/policies/'}",
            (),
        )
    return True, "", tuple(groups)


__all__ = ["DOCUMENTS", "load"]
