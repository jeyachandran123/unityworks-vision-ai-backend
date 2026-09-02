"""The dataset regression recording — `tests/compliance/kitchen01_model_answers.json`.

43 real answers from `meta/llama-3.2-11b-vision-instruct` on real crops from the
human-annotated kitchen-01 split, recorded once and never edited. The test file
beside it replays them through the shipped compliance rule; this adapter reads
the recording itself.

### What this adapter computes, and what it deliberately does not

It counts. For each attribute it compares the recorded `truth` value with the
recorded `model` value and tallies agreement. That is arithmetic over a file.

It does **not** recompute compliance verdicts. Deciding whether a pair of
attribute values is a violation is `compliance/`'s job, the rule lives in one
place, and a second implementation here would be exactly the duplication the
Phase 3 close-out just removed from the observation fold. The verdict
distribution is asserted in `tests/compliance/test_dataset_regression.py`, where
it is **pinned** — `assert len(buckets["false_violation"]) == 11` — as a
measurement that must not change silently rather than as a gate. That pinning is
described here and its number is not restated, because a copy would drift from
the assertion.

### Per-case detail stays in the file

Each case carries a `frame`, a `subject` and a `note` describing what the person
was doing ("blue hairnet; both hands bare on the mixing bowl"). Those are
per-person records tied to identifiable frames. Only the tallies leave.
"""

from __future__ import annotations

from app.evaluation.adapters.common import ROOT, parse_instant, read_json, repo_relative
from app.evaluation.model import (
    ArtifactFamily,
    ComparabilityKey,
    EvaluationRun,
    MetricEntry,
    MetricGroup,
    MetricKind,
    Provenance,
)

RECORDING = ROOT / "tests" / "compliance" / "kitchen01_model_answers.json"

TEST_MODULE = "tests/compliance/test_dataset_regression.py"


def _tally(cases: list[dict], attribute: str) -> tuple[int, int, int, int]:
    """`(compared, agreed, model_silent, unsupported)` for one attribute.

    `unsupported` counts the safety failure directly: the annotator recorded
    `not_visible` and the model answered something decided. Counted here the same
    way every other surface in this repository counts it.
    """
    compared = agreed = silent = unsupported = 0
    for case in cases:
        truth = (case.get("truth") or {}).get(attribute)
        answer = (case.get("model") or {}).get(attribute)
        if truth is None:
            continue
        compared += 1
        if answer is None:
            silent += 1
            continue
        # An exact match, or a specific covering where the annotation says one
        # is present. `present` is the annotation vocabulary; `cap`, `hairnet`
        # and friends are the model's. Anything not in the four-state vocabulary
        # is treated as a covering, which is what the annotation means by
        # `present`.
        decided_absent = answer in ("none", "absent")
        decided = answer not in ("not_visible", "unknown")
        if truth == "present" and decided and not decided_absent:
            agreed += 1
        elif truth == answer:
            agreed += 1
        if truth == "not_visible" and decided:
            unsupported += 1
    return compared, agreed, silent, unsupported


def load() -> ArtifactFamily:
    payload = read_json(RECORDING)
    expected = (repo_relative(RECORDING),)

    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        return ArtifactFamily(
            key="dataset_regression",
            title="Dataset regression recording",
            description=(
                "Recorded model answers replayed against the shipped compliance "
                "rule by the test suite."
            ),
            available=False,
            reason=(
                f"{repo_relative(RECORDING)} is missing or could not be read as a "
                "recording. No regression figure is shown; an unreadable "
                "recording is not a passing one."
            ),
            expected_artifacts=expected,
        )

    cases = [c for c in payload["cases"] if isinstance(c, dict)]
    recorded_at = parse_instant(payload.get("recorded_at"))

    prov = Provenance(
        artifact=repo_relative(RECORDING),
        source="dataset_regression",
        run_id="kitchen01_model_answers",
        model=str(payload.get("model") or "not recorded"),
        configuration=f"replayed by {TEST_MODULE}",
        dataset=str(payload.get("dataset") or "kitchen-01"),
        split="test",
        evaluated_at=recorded_at,
        timestamp_source="artifact" if recorded_at else "absent",
        sample_size=len(cases),
        limitations=(
            "A recording, not a live evaluation. It is deterministic and needs no "
            "network, and it is never edited to make a test pass.",
            "The compliance-verdict distribution is asserted in "
            f"{TEST_MODULE} and is pinned there as a measurement that must not "
            "change silently. It is not recomputed on this page, because the "
            "rule lives in compliance/ and a second copy would drift.",
            "recorded_at is a date with no time. Treated as recorded on that day.",
        ),
    )

    groups: list[MetricGroup] = [
        MetricGroup(
            key="recording",
            title="The recording",
            description="What was recorded, from what, and when.",
            metrics=(
                MetricEntry(
                    key="cases",
                    label="Recorded cases",
                    kind=MetricKind.COUNT,
                    value=len(cases),
                    definition=(
                        "Subject answers captured from the model on real crops "
                        "and replayed by the test suite on every run."
                    ),
                    provenance=prov,
                ),
            ),
        )
    ]

    attributes = sorted(
        {key for case in cases for key in (case.get("truth") or {}) if isinstance(key, str)}
    )
    for attribute in attributes:
        compared, agreed, silent, unsupported = _tally(cases, attribute)
        groups.append(
            MetricGroup(
                key=attribute,
                title=attribute.replace("_", " ").capitalize(),
                description=(
                    "Direct comparison of the recorded annotation with the "
                    "recorded model answer. Counting only — no compliance rule "
                    "is applied here."
                ),
                metrics=(
                    MetricEntry(
                        key=f"{attribute}.agreement",
                        label="Recorded agreement",
                        kind=MetricKind.ATTRIBUTE_AGREEMENT,
                        value=(agreed / compared) if compared else None,
                        definition=(
                            "Recorded model answers that matched the recorded "
                            "annotation, over annotated cases. A comparison of two "
                            "recorded fields, not a compliance outcome."
                        ),
                        provenance=prov,
                        support=compared,
                        undefined_reason=""
                        if compared
                        else "No case carries an annotation for this attribute.",
                    ),
                    MetricEntry(
                        key=f"{attribute}.unsupported_claims",
                        label="Unsupported claims",
                        kind=MetricKind.UNSUPPORTED_CLAIMS,
                        value=unsupported,
                        definition=(
                            "Cases where the annotator recorded not_visible and "
                            "the model answered something decided."
                        ),
                        provenance=prov,
                        support=compared,
                    ),
                    MetricEntry(
                        key=f"{attribute}.model_silent",
                        label="No model answer",
                        kind=MetricKind.COUNT,
                        value=silent,
                        definition=(
                            "Annotated cases the recording holds no model answer "
                            "for. Counted apart from a wrong answer: silence and "
                            "error have different causes."
                        ),
                        provenance=prov,
                        support=compared,
                    ),
                ),
            )
        )

    return ArtifactFamily(
        key="dataset_regression",
        title="Dataset regression recording",
        description=(
            "43 recorded answers from the production VLM on human-annotated "
            "crops. Replayed against the shipped compliance rule by the test "
            "suite on every run; the counts here are of the recording itself."
        ),
        available=True,
        runs=(
            EvaluationRun(
                run_id="kitchen01_model_answers",
                title="kitchen-01 recorded answers",
                summary=(
                    "The fixture behind the compliance regression test. Evidence, "
                    "and never edited to make a test pass."
                ),
                source="dataset_regression",
                provenance=prov,
                comparability=ComparabilityKey(
                    metric_kind=MetricKind.ATTRIBUTE_AGREEMENT.value,
                    model=prov.model,
                    dataset=prov.dataset,
                    split="test",
                    # Its own configuration string. Agreement counted by direct
                    # field comparison is not the same measurement as the
                    # harness's matched-subject accuracy, so it must never share
                    # an axis with the PPE evaluation runs even though both are
                    # called "agreement".
                    configuration="recorded-replay",
                ),
                groups=tuple(groups),
            ),
        ),
        expected_artifacts=expected,
    )


__all__ = ["RECORDING", "TEST_MODULE", "load"]
