"""PPE attribute evaluation — `tools/vision_eval`'s committed reports.

Reads `datasets/kitchen-01/results/{baseline,crop448,crop448_gated,phase42}.json`,
each an `EvaluationReport` serialised by the offline harness against the
human-annotated `kitchen-01` split.

### What these numbers are, precisely

`accuracy` in these files is `correct / matched` for **one attribute** over the
subjects the detector matched to an annotation. `tools/vision_eval/metrics.py`
defines it that way and this adapter carries that definition through verbatim.
It is not "model accuracy", it is not mAP, and it is not a compliance pass rate.

`unsupported_claims` is the harness's own headline safety figure: decided answers
(`present`/`absent`) where the annotator recorded `not_visible`. Every one is the
system asserting something about pixels a human judged unreadable. It is surfaced
prominently because the harness itself calls it *"the failure mode this whole
programme is built around"*.

### What is deliberately dropped

Each report carries a `failures` array of individual cases with `frame_id`,
`subject_id` and a human `detail` sentence. Those are per-person records tied to
identifiable frames, and a dashboard has no need of them to answer "is this model
good enough" — the confusion matrix and the failure-category tally answer that.
Only the category counts survive.

### No timestamp exists in these files

They record `tag` and `configuration` and no date at all. `Provenance` therefore
carries `evaluated_at = None` and `timestamp_source = "absent"`. The file's
modification time is **not** substituted: it records when git wrote the file, not
when anybody ran an evaluation.
"""

from __future__ import annotations

from pathlib import Path

from app.evaluation.adapters.common import (
    ROOT,
    basename,
    ratio,
    read_json,
    repo_relative,
    whole,
)
from app.evaluation.model import (
    ArtifactFamily,
    ComparabilityKey,
    EvaluationRun,
    MetricEntry,
    MetricGroup,
    MetricKind,
    Provenance,
)

RESULTS = ROOT / "datasets" / "kitchen-01" / "results"

#: The reports this adapter reads, in the order a reviewer would read them.
#: A fixed list rather than a glob: `phase42.cache.json` and `detector_bench.json`
#: sit in the same directory and are not attribute evaluations, and a glob would
#: eventually pick up whatever else lands there.
REPORTS: tuple[tuple[str, str, str], ...] = (
    (
        "baseline",
        "Baseline",
        "The shipped configuration at the time: evidence regions on, quality gate on.",
    ),
    ("crop448", "448px crops", "Larger crops handed to the understander."),
    ("crop448_gated", "448px crops, gated", "448px crops with the quality gate applied."),
    ("phase42", "Phase 4.2", "Per-attribute output sizing."),
)

#: Attribute agreement, defined once. Every metric that uses it says the same
#: sentence, so two surfaces cannot describe the same number differently.
AGREEMENT_DEFINITION = (
    "Correct answers divided by matched subjects, for this one attribute, "
    "against human annotation on this split. Not overall model accuracy and "
    "not a compliance pass rate."
)


def _provenance(path: Path, payload: dict, tag: str) -> Provenance:
    configuration = payload.get("configuration") or {}
    policy = basename(str(configuration.get("policy", "")))
    understander = str(configuration.get("understander", ""))

    bits = [b for b in (policy, understander) if b]
    return Provenance(
        artifact=repo_relative(path),
        source="ppe_evaluation",
        run_id=tag,
        # The report names its understander binding rather than a model version.
        # Recorded as what it says, not upgraded into a version string.
        model=understander or "not recorded",
        configuration=" / ".join(bits),
        dataset="kitchen-01",
        split="test",
        # No date exists inside these artifacts. Left absent rather than taken
        # from the filesystem.
        evaluated_at=None,
        timestamp_source="absent",
        sample_size=whole(payload.get("matched_subjects")),
        limitations=(
            "This report records no evaluation date. Its position in any "
            "ordering is by configuration, not by time.",
            "kitchen-01 annotates only detector proposals, so it cannot "
            "measure detection recall — see the dataset's own note.",
        ),
    )


def _attribute_group(attribute: dict, provenance: Provenance) -> MetricGroup:
    name = str(attribute.get("attribute", "unknown"))
    matched = whole(attribute.get("matched"))
    metrics: list[MetricEntry] = [
        MetricEntry(
            key=f"{name}.agreement",
            label="Attribute agreement",
            kind=MetricKind.ATTRIBUTE_AGREEMENT,
            value=ratio(attribute.get("accuracy")),
            definition=AGREEMENT_DEFINITION,
            provenance=provenance,
            support=matched,
            undefined_reason=""
            if attribute.get("accuracy") is not None
            else "No subject was matched for this attribute, so the ratio has no denominator.",
        ),
        MetricEntry(
            key=f"{name}.unsupported_claims",
            label="Unsupported claims",
            kind=MetricKind.UNSUPPORTED_CLAIMS,
            value=whole(attribute.get("unsupported_claims")),
            definition=(
                "Decided answers (present or absent) where the human annotator "
                "recorded not_visible. Each one is the system asserting "
                "something about a region a person judged unreadable. Lower is "
                "better; zero is the target."
            ),
            provenance=provenance,
            support=matched,
        ),
    ]

    # Per-state precision / recall / F1, exactly as the harness computed them.
    for state in attribute.get("states") or []:
        label = str(state.get("state", ""))
        support = whole(state.get("support"))
        undefined = (
            "No annotated example of this state exists in the split, so the "
            "metric has no denominator. Undefined, not zero."
            if not support
            else ""
        )
        for metric_key, kind, definition in (
            (
                "precision",
                MetricKind.PRECISION,
                f"Of the subjects the system called '{label}', the share that truly were.",
            ),
            (
                "recall",
                MetricKind.RECALL,
                f"Of the subjects that truly were '{label}', the share the system found.",
            ),
            ("f1", MetricKind.F1, f"Harmonic mean of precision and recall for '{label}'."),
        ):
            raw = state.get(metric_key)
            metrics.append(
                MetricEntry(
                    key=f"{name}.{label}.{metric_key}",
                    label=f"{label} {metric_key}",
                    kind=kind,
                    value=ratio(raw),
                    definition=definition,
                    provenance=provenance,
                    support=support,
                    undefined_reason="" if raw is not None else undefined or
                    "The artifact recorded this metric as undefined.",
                )
            )

    confusion = attribute.get("confusion")
    return MetricGroup(
        key=name,
        title=name.replace("_", " ").capitalize(),
        description=(
            f"Agreement against human annotation for {name.replace('_', ' ')}, "
            f"over {matched if matched is not None else 'an unrecorded number of'} "
            "matched subjects."
        ),
        metrics=tuple(metrics),
        confusion=confusion if isinstance(confusion, dict) else None,
    )


def _run_group(payload: dict, provenance: Provenance) -> MetricGroup:
    """The run-level counts. Counts, never rates that the split cannot support."""
    metrics = [
        MetricEntry(
            key="frames",
            label="Frames evaluated",
            kind=MetricKind.COUNT,
            value=whole(payload.get("frames")),
            definition="Annotated frames the harness ran over.",
            provenance=provenance,
        ),
        MetricEntry(
            key="annotated_subjects",
            label="Annotated subjects",
            kind=MetricKind.COUNT,
            value=whole(payload.get("annotated_subjects")),
            definition="People a human annotated across those frames.",
            provenance=provenance,
        ),
        MetricEntry(
            key="matched_subjects",
            label="Matched subjects",
            kind=MetricKind.COUNT,
            value=whole(payload.get("matched_subjects")),
            definition=(
                "Annotated people the detector also found, and therefore the "
                "denominator of every agreement figure below."
            ),
            provenance=provenance,
        ),
        MetricEntry(
            key="unmatched_truth",
            label="Annotated but not detected",
            kind=MetricKind.COUNT,
            value=whole(payload.get("unmatched_truth")),
            definition=(
                "Annotated people the detector never proposed. Counted, but not "
                "turned into a recall figure: kitchen-01 annotates detector "
                "proposals, so it cannot measure what the detector missed."
            ),
            provenance=provenance,
        ),
        MetricEntry(
            key="vlm_calls",
            label="Model calls",
            kind=MetricKind.COUNT,
            value=whole(payload.get("vlm_calls")),
            definition="Understander invocations the run made.",
            provenance=provenance,
        ),
    ]
    latency = ratio(payload.get("mean_vlm_latency_ms"))
    if latency is not None:
        metrics.append(
            MetricEntry(
                key="mean_vlm_latency_ms",
                label="Mean model latency",
                kind=MetricKind.LATENCY_MS,
                value=latency,
                definition="Mean wall-clock time per understander call, in milliseconds.",
                provenance=provenance,
                unit="ms",
            )
        )

    return MetricGroup(
        key="run",
        title="Run totals",
        description="What the harness ran over, and what it cost.",
        metrics=tuple(metrics),
    )


def _failure_group(payload: dict, provenance: Provenance) -> MetricGroup | None:
    """Failure *categories* only. The individual cases are left in the file.

    Each `failures` entry names a frame and a subject and describes what the
    person was doing. That is per-person material tied to an identifiable frame,
    and the category tally answers the dashboard's question without it.
    """
    categories = payload.get("failures_by_category")
    if not isinstance(categories, dict) or not categories:
        return None
    return MetricGroup(
        key="failures",
        title="Failure categories",
        description=(
            "How the harness classified each disagreement. Individual failing "
            "cases name a frame and a person and are deliberately not surfaced "
            "here."
        ),
        metrics=tuple(
            MetricEntry(
                key=f"failure.{name}",
                label=name.replace("_", " "),
                kind=MetricKind.COUNT,
                value=whole(count),
                definition=f"Disagreements the harness classified as '{name}'.",
                provenance=provenance,
            )
            for name, count in sorted(categories.items(), key=lambda kv: -int(kv[1]))
        ),
    )


def load() -> ArtifactFamily:
    """Every committed PPE evaluation report, normalized."""
    runs: list[EvaluationRun] = []
    expected = tuple(f"datasets/kitchen-01/results/{tag}.json" for tag, _, _ in REPORTS)

    for tag, title, summary in REPORTS:
        path = RESULTS / f"{tag}.json"
        payload = read_json(path)

        if not isinstance(payload, dict):
            runs.append(
                EvaluationRun(
                    run_id=tag,
                    title=title,
                    source="ppe_evaluation",
                    summary=summary,
                    provenance=Provenance(
                        artifact=repo_relative(path), source="ppe_evaluation", run_id=tag
                    ),
                    comparability=ComparabilityKey(
                        metric_kind=MetricKind.ATTRIBUTE_AGREEMENT.value,
                        model="",
                        dataset="kitchen-01",
                        split="test",
                        configuration="",
                    ),
                    available=False,
                    reason=(
                        f"{repo_relative(path)} is missing or could not be read as "
                        "an evaluation report. No figure is shown for it, and its "
                        "absence is not a score."
                    ),
                )
            )
            continue

        provenance = _provenance(path, payload, tag)
        groups: list[MetricGroup] = [_run_group(payload, provenance)]
        for attribute in payload.get("attributes") or []:
            if isinstance(attribute, dict):
                groups.append(_attribute_group(attribute, provenance))
        failures = _failure_group(payload, provenance)
        if failures is not None:
            groups.append(failures)

        runs.append(
            EvaluationRun(
                run_id=tag,
                title=title,
                summary=summary,
                source="ppe_evaluation",
                provenance=provenance,
                comparability=ComparabilityKey(
                    metric_kind=MetricKind.ATTRIBUTE_AGREEMENT.value,
                    model=provenance.model,
                    dataset="kitchen-01",
                    split="test",
                    configuration=provenance.configuration,
                ),
                groups=tuple(groups),
                # Each of these varied the configuration on purpose. They are a
                # comparison of settings, not a series over time, and nothing
                # about them is partial.
                completeness="complete",
            )
        )

    readable = [r for r in runs if r.available]
    return ArtifactFamily(
        key="ppe_evaluation",
        title="PPE attribute evaluation",
        description=(
            "Offline evaluation of the understander's PPE answers against the "
            "human-annotated kitchen-01 split, produced by tools/vision_eval. "
            "Each run varied one configuration choice; none records an "
            "evaluation date."
        ),
        available=bool(readable),
        reason=""
        if readable
        else (
            "No PPE evaluation report could be read. The harness lives in "
            "tools/vision_eval and writes to datasets/kitchen-01/results."
        ),
        runs=tuple(runs),
        expected_artifacts=expected,
    )


__all__ = ["AGREEMENT_DEFINITION", "REPORTS", "load"]
