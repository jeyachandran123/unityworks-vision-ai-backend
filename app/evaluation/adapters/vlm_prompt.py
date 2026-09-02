"""VLM prompt experiments — `experiments/vlm_prompt/runs/`.

Reads `scores.json` (the scored variants) and each `variant_*.json` for its
recorded provenance. These are the only artifacts in the repository with a real
evaluation timestamp, a real model identity and a real corpus digest.

### The metric is not called accuracy, and this adapter does not rename it

`experiments/vlm_prompt/score.py` computes `accuracy_over_parsed` — agreement
with human annotation over the responses that **parsed**. A run with unparsed
answers has a different denominator from one without, which is precisely why the
experiment gave it its own name. Calling it "model accuracy" on a dashboard
would be the relabelling this phase exists to avoid, so `MetricKind` has a
distinct member for it and the definition string says what the denominator is.

### Three views, kept three

The experiment scores each variant three ways and its own docstring explains why
only reporting one would mislead:

    ungated    every subject reaches the model — can the prompt make the VLM refuse?
    p8_gated   the pose gate refuses first, as production does — the operational reality
    verdicts   the gated answers through the real shipped rule — what a manager is told

They answer different questions and are surfaced as separate groups. `verdicts`
is `null` in the current scoring output and is therefore reported as absent
rather than as an empty result.

### Comparability

Every variant shares `corpus_digest = bbe9e0559523b9b0`, the same model and the
same attribute, so A/B/C/D **are** comparable with one another. The
`minimax_m3_20260831` run is a different model entirely and produced no usable
answers at all — it is loaded, reported, and deliberately given a comparability
key that matches nothing.
"""

from __future__ import annotations

from app.evaluation.adapters.common import (
    ROOT,
    parse_instant,
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

RUNS = ROOT / "experiments" / "vlm_prompt" / "runs"

VIEWS: tuple[tuple[str, str, str], ...] = (
    (
        "ungated",
        "Ungated",
        "Every subject reaches the model. The only view that answers whether the "
        "prompt itself can make the VLM refuse.",
    ),
    (
        "p8_gated",
        "Pose-gated",
        "The pose gate refuses first, exactly as production has since P8. The "
        "operational reality, and the view a promotion decision uses.",
    ),
)

ACCURACY_OVER_PARSED_DEFINITION = (
    "Agreement with human annotation over the responses that parsed. The "
    "experiment's own name for this figure — a run with unparsed answers has a "
    "different denominator, so it is deliberately not called accuracy."
)


def _variant_provenance(variant: str, scored: dict) -> tuple[Provenance, str]:
    """Provenance from the scores plus the variant file's recorded metadata."""
    path = RUNS / f"variant_{variant}.json"
    detail = read_json(path)
    detail = detail if isinstance(detail, dict) else {}

    recorded_at = parse_instant(detail.get("recorded_at"))
    prompt_sha = str(scored.get("prompt_sha256") or detail.get("prompt_sha256") or "")
    model = str(scored.get("model") or detail.get("model") or "")
    corpus = str(scored.get("corpus_digest") or detail.get("corpus_digest") or "")
    attribute = str(detail.get("attribute") or "")

    configuration_bits = [b for b in (f"prompt:{prompt_sha}" if prompt_sha else "", attribute) if b]

    limitations = [
        "Ground truth for head_covering contains no 'absent' examples, so every "
        "metric needing them is undefined rather than zero.",
        "Scored on a 43-subject corpus from one 15-frame split. A difference of "
        "one or two cases moves these figures visibly.",
    ]
    if not recorded_at:
        limitations.append("This variant file records no timestamp.")

    return (
        Provenance(
            artifact=repo_relative(RUNS / "scores.json"),
            source="vlm_prompt",
            run_id=f"variant_{variant}",
            model=model or "not recorded",
            configuration=" / ".join(configuration_bits),
            dataset=str(detail.get("dataset") or "kitchen-01"),
            split="test",
            evaluated_at=recorded_at,
            timestamp_source="artifact" if recorded_at else "absent",
            sample_size=whole(scored.get("n")),
            limitations=tuple(limitations),
        ),
        corpus,
    )


def _view_group(view: str, title: str, description: str, payload: dict, prov: Provenance) -> MetricGroup:
    metrics: list[MetricEntry] = [
        MetricEntry(
            key=f"{view}.accuracy_over_parsed",
            label="Accuracy over parsed",
            kind=MetricKind.ACCURACY_OVER_PARSED,
            value=ratio(payload.get("accuracy_over_parsed")),
            definition=ACCURACY_OVER_PARSED_DEFINITION,
            provenance=prov,
            support=prov.sample_size,
        ),
        MetricEntry(
            key=f"{view}.unsupported_claims",
            label="Unsupported claims",
            kind=MetricKind.UNSUPPORTED_CLAIMS,
            value=whole(payload.get("unsupported_claims")),
            definition=(
                "Answers of present or absent where the annotator recorded "
                "not_visible. The number this experiment was run to reduce."
            ),
            provenance=prov,
        ),
        MetricEntry(
            key=f"{view}.unparsed",
            label="Unparsed responses",
            kind=MetricKind.COUNT,
            value=whole(payload.get("unparsed")),
            definition=(
                "Responses the harness could not read as an answer. They are "
                "excluded from the accuracy denominator, which is why that "
                "metric carries the name it does."
            ),
            provenance=prov,
        ),
    ]

    for state, scores in (payload.get("states") or {}).items():
        if not isinstance(scores, dict):
            continue
        support = whole(scores.get("support"))
        for key, kind in (
            ("precision", MetricKind.PRECISION),
            ("recall", MetricKind.RECALL),
            ("f1", MetricKind.F1),
        ):
            raw = scores.get(key)
            metrics.append(
                MetricEntry(
                    key=f"{view}.{state}.{key}",
                    label=f"{state} {key}",
                    kind=kind,
                    value=ratio(raw),
                    definition=f"{key.capitalize()} for the state '{state}' in the {title.lower()} view.",
                    provenance=prov,
                    support=support,
                    undefined_reason=""
                    if raw is not None
                    else (
                        "No annotated example of this state exists in the corpus, "
                        "so the metric is undefined. The experiment reports null "
                        "here rather than zero, deliberately."
                    ),
                )
            )

    confusion = payload.get("confusion")
    return MetricGroup(
        key=view,
        title=title,
        description=description,
        metrics=tuple(metrics),
        confusion=confusion if isinstance(confusion, dict) else None,
    )


def _minimax_run() -> EvaluationRun | None:
    """The run that produced nothing, reported as such.

    Loaded on purpose. A dashboard that silently omitted a failed experiment
    would leave a reader believing every attempt at a second model succeeded, and
    this one is the clearest illustration in the repository of a run whose
    numbers must not be compared with anything.
    """
    path = RUNS / "minimax_m3_20260831.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None

    ran_at = parse_instant(payload.get("ran_at"))
    prov = Provenance(
        artifact=repo_relative(path),
        source="vlm_prompt",
        run_id="minimax_m3_20260831",
        model=str(payload.get("model") or "not recorded"),
        configuration=f"variant:{payload.get('variant', '')}",
        dataset="kitchen-01",
        split="test",
        evaluated_at=ran_at,
        timestamp_source="artifact" if ran_at else "absent",
        sample_size=whole(payload.get("subjects_attempted")),
        limitations=(
            "Every request was rate-limited. The run produced no usable answer, "
            "so it measures the provider's throttling rather than the model.",
            "A different model from the llama-3.2 variants. Its figures share no "
            "axis with theirs.",
        ),
    )

    answered = ratio(payload.get("answer_rate"))
    agreement = payload.get("agreement_when_answered")
    metrics = (
        MetricEntry(
            key="answer_rate",
            label="Answer rate",
            kind=MetricKind.RATIO,
            value=answered,
            definition="Share of attempted subjects for which the provider returned any answer.",
            provenance=prov,
            support=whole(payload.get("subjects_attempted")),
        ),
        MetricEntry(
            key="rate_limited_total",
            label="Rate-limited requests",
            kind=MetricKind.COUNT,
            value=whole(payload.get("rate_limited_total")),
            definition="Requests the provider refused for rate limiting.",
            provenance=prov,
        ),
        MetricEntry(
            key="agreement_when_answered",
            label="Agreement when answered",
            kind=MetricKind.ACCURACY_OVER_PARSED,
            value=ratio(agreement),
            definition=(
                "Agreement with annotation across answered subjects. The "
                "artifact's own field name — and it is null here because nothing "
                "was answered."
            ),
            provenance=prov,
            undefined_reason=""
            if agreement is not None
            else "No subject was answered, so there is nothing to agree with. Undefined, not zero.",
        ),
    )

    return EvaluationRun(
        run_id="minimax_m3_20260831",
        title="minimax-m3 probe",
        summary=(
            "An attempt to score a second model on the same corpus. Every request "
            "was rate-limited and no answer was recorded."
        ),
        source="vlm_prompt",
        provenance=prov,
        # Deliberately keyed so it matches nothing: a different model, and a
        # configuration string no variant shares.
        comparability=ComparabilityKey(
            metric_kind=MetricKind.ACCURACY_OVER_PARSED.value,
            model=str(payload.get("model") or ""),
            dataset="kitchen-01",
            split="test",
            configuration="minimax-probe",
        ),
        groups=(
            MetricGroup(
                key="probe",
                title="Probe outcome",
                description="What the provider did, rather than what the model can do.",
                metrics=metrics,
            ),
        ),
        completeness="partial",
        reason=(
            "The provider rate-limited every request. This run measures "
            "throttling, not model quality, and shares no axis with the "
            "llama-3.2 variants."
        ),
    )


def load() -> ArtifactFamily:
    scores = read_json(RUNS / "scores.json")
    expected = ("experiments/vlm_prompt/runs/scores.json",)

    runs: list[EvaluationRun] = []
    if isinstance(scores, dict):
        for variant, scored in sorted(scores.items()):
            if not isinstance(scored, dict):
                continue
            prov, corpus = _variant_provenance(variant, scored)
            groups = [
                _view_group(view, title, description, scored[view], prov)
                for view, title, description in VIEWS
                if isinstance(scored.get(view), dict)
            ]
            missing_views = [v for v, _, _ in VIEWS if not isinstance(scored.get(v), dict)]
            verdicts_absent = scored.get("verdicts_gated") is None

            runs.append(
                EvaluationRun(
                    run_id=f"variant_{variant}",
                    title=f"Variant {variant} — {scored.get('title', '')}".strip(" —"),
                    summary=(
                        f"Prompt variant {variant}, scored on {scored.get('n', '?')} "
                        "subjects in two views."
                    ),
                    source="vlm_prompt",
                    provenance=prov,
                    comparability=ComparabilityKey(
                        metric_kind=MetricKind.ACCURACY_OVER_PARSED.value,
                        model=prov.model,
                        dataset=prov.dataset,
                        split=prov.split,
                        # The corpus digest is what makes these comparable: same
                        # subjects, same annotations, same order.
                        configuration=f"corpus:{corpus}" if corpus else "",
                    ),
                    groups=tuple(groups),
                    completeness="partial" if (missing_views or verdicts_absent) else "complete",
                    reason=(
                        "The verdict views are null in the scoring output, so the "
                        "compliance-rule outcome is not shown for this variant."
                        if verdicts_absent
                        else ""
                    ),
                )
            )

    probe = _minimax_run()
    if probe is not None:
        runs.append(probe)

    return ArtifactFamily(
        key="vlm_prompt",
        title="VLM prompt experiments",
        description=(
            "Recorded prompt variants scored against the same 43-subject corpus. "
            "The only artifacts in this repository that carry a real evaluation "
            "timestamp, model identity and corpus digest."
        ),
        available=bool(runs),
        reason=""
        if runs
        else (
            "experiments/vlm_prompt/runs/scores.json could not be read, so no "
            "prompt variant is scored here."
        ),
        runs=tuple(runs),
        expected_artifacts=expected,
    )


__all__ = ["ACCURACY_OVER_PARSED_DEFINITION", "VIEWS", "load"]
