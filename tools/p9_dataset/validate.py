"""Dataset integrity checks — the ones a dataset cannot be trusted without.

Split into two severities, and the distinction is deliberate:

**ERROR** — the dataset is wrong and must not be used. Contradictory states,
duplicate ids, leakage across splits, model labels inside an evaluation split.

**WARNING** — the dataset is *thin* and the report must say so. Missing states,
unknown hard-case tags, a class with too little support to score. These are
findings about the data, not defects in it, and suppressing them would hide the
one thing this programme most needs to know.

The check that matters most is `no_group_leakage`. Everything else guards
against mistakes; that one guards against a benchmark that reports a number it
has not earned.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from dataclasses import dataclass

from .schema import (
    HARD_CASE_TAGS,
    AttributeState,
    LabelProvenance,
    Observability,
    QualityStatus,
    Region,
    Split,
    SubjectAnnotation,
)

#: Splits from which a published metric may be computed.
EVALUATION_SPLITS = (Split.TEST, Split.HARD_TEST, Split.VALIDATION)

#: Minimum examples of a state before a metric over it is worth quoting.
#:
#: Not a statistical threshold — with n this small nothing is. It is the point
#: below which the P9 report prints `insufficient support` instead of a number,
#: chosen so that the current corpus's zero-support classes cannot silently
#: acquire a percentage.
MIN_SUPPORT_FOR_METRIC = 30


@dataclass(frozen=True, slots=True)
class ValidationError:
    severity: str
    check: str
    detail: str
    sample_id: str = ""

    def __str__(self) -> str:
        where = f" [{self.sample_id}]" if self.sample_id else ""
        return f"{self.severity.upper():7s} {self.check}{where}: {self.detail}"


def validate_subject(subject: SubjectAnnotation) -> list[ValidationError]:
    """Per-subject checks that the type system does not already enforce.

    The impossible state combinations (`NOT_VISIBLE` + `PRESENT`/`ABSENT`,
    `VISIBLE` + `NOT_EVALUATED`) are rejected in `RegionAnnotation.__post_init__`
    and therefore cannot reach here — a construction-time guard rather than a
    validation pass, so a bad annotation cannot exist in memory long enough to
    be written.
    """
    out: list[ValidationError] = []

    for region in subject.regions:
        for tag in region.hard_case_tags:
            if tag not in HARD_CASE_TAGS:
                out.append(
                    ValidationError(
                        "warning", "unknown_hard_case_tag",
                        f"{region.region.value}: {tag!r} is not in the known set",
                        subject.sample_id,
                    )
                )
        if region.observability is Observability.UNCERTAIN and not region.note:
            out.append(
                ValidationError(
                    "warning", "uncertain_without_reason",
                    f"{region.region.value}: UNCERTAIN carries no note; a marginal "
                    f"call nobody can re-read cannot be adjudicated later",
                    subject.sample_id,
                )
            )

    if subject.quality_status is QualityStatus.ACCEPTED and (
        subject.label_provenance is LabelProvenance.MACHINE_PROPOSED
    ):
        out.append(
            ValidationError(
                "error", "machine_label_accepted",
                "a machine-proposed label was marked ACCEPTED without human "
                "verification; model output must never become ground truth",
                subject.sample_id,
            )
        )
    return out


def _duplicate_ids(subjects: list[SubjectAnnotation]) -> list[ValidationError]:
    counts = collections.Counter(s.sample_id for s in subjects)
    return [
        ValidationError("error", "duplicate_sample_id", f"{n} occurrences", sample_id)
        for sample_id, n in counts.items()
        if n > 1
    ]


def _group_leakage(subjects: list[SubjectAnnotation]) -> list[ValidationError]:
    """No `session::subject` group may appear in more than one split.

    **The check this whole module exists for.** The same person seconds apart is
    one example photographed twice; putting those either side of a split
    boundary produces a benchmark that measures memorisation and reports it as
    accuracy.
    """
    placement: dict[str, set[str]] = collections.defaultdict(set)
    for subject in subjects:
        if subject.split is not Split.UNASSIGNED:
            placement[subject.group_key].add(subject.split.value)
    return [
        ValidationError(
            "error", "group_leakage",
            f"group {group!r} appears in splits {sorted(splits)} — the same "
            f"subject in the same session must live in exactly one split",
        )
        for group, splits in placement.items()
        if len(splits) > 1
    ]


def _frame_leakage(subjects: list[SubjectAnnotation]) -> list[ValidationError]:
    """A frame may not be split across partitions either.

    Two people in one frame share lighting, camera, occlusion and moment. Even
    with different subject ids they are not independent samples.
    """
    placement: dict[str, set[str]] = collections.defaultdict(set)
    for subject in subjects:
        if subject.split is not Split.UNASSIGNED:
            placement[subject.frame_id].add(subject.split.value)
    return [
        ValidationError(
            "error", "frame_leakage",
            f"frame {frame!r} appears in splits {sorted(splits)}",
        )
        for frame, splits in placement.items()
        if len(splits) > 1
    ]


def _machine_labels_in_evaluation(
    subjects: list[SubjectAnnotation],
) -> list[ValidationError]:
    return [
        ValidationError(
            "error", "machine_label_in_evaluation",
            f"split={s.split.value} carries a {s.label_provenance.value} label; "
            f"an evaluation split may contain only human-verified ground truth",
            s.sample_id,
        )
        for s in subjects
        if s.split in EVALUATION_SPLITS
        and s.label_provenance
        not in (LabelProvenance.HUMAN_VERIFIED, LabelProvenance.HUMAN_ADJUDICATED)
    ]


def _disputed_in_evaluation(subjects: list[SubjectAnnotation]) -> list[ValidationError]:
    return [
        ValidationError(
            "error", "disputed_in_evaluation",
            "an unresolved annotator disagreement is in an evaluation split; "
            "adjudicate it or exclude it, but never resolve it silently",
            s.sample_id,
        )
        for s in subjects
        if s.split in EVALUATION_SPLITS and s.quality_status is QualityStatus.DISPUTED
    ]


def _support(subjects: list[SubjectAnnotation]) -> list[ValidationError]:
    """Warn per region+state where evaluation support is too thin to quote."""
    out: list[ValidationError] = []
    evaluable = [s for s in subjects if s.split in EVALUATION_SPLITS]
    for region in Region:
        counts = collections.Counter()
        for subject in evaluable:
            annotation = subject.region_of(region)
            if annotation is None:
                continue
            if annotation.observability is Observability.VISIBLE:
                counts[annotation.state.value] += 1
            else:
                counts[annotation.observability.value] += 1
        for state in (AttributeState.PRESENT.value, AttributeState.ABSENT.value,
                      Observability.NOT_VISIBLE.value):
            n = counts.get(state, 0)
            if n < MIN_SUPPORT_FOR_METRIC:
                out.append(
                    ValidationError(
                        "warning", "insufficient_support",
                        f"{region.value}/{state}: {n} example(s) in evaluation "
                        f"splits, below the {MIN_SUPPORT_FOR_METRIC} needed to "
                        f"quote a metric",
                    )
                )
    return out


def validate_manifest(subjects: Iterable[SubjectAnnotation]) -> list[ValidationError]:
    """Every check, in one pass. Errors first, then warnings."""
    items = list(subjects)
    errors: list[ValidationError] = []
    for subject in items:
        errors.extend(validate_subject(subject))
    errors.extend(_duplicate_ids(items))
    errors.extend(_group_leakage(items))
    errors.extend(_frame_leakage(items))
    errors.extend(_machine_labels_in_evaluation(items))
    errors.extend(_disputed_in_evaluation(items))
    errors.extend(_support(items))
    return sorted(errors, key=lambda e: (e.severity != "error", e.check, e.sample_id))


def errors_only(findings: Iterable[ValidationError]) -> list[ValidationError]:
    return [f for f in findings if f.severity == "error"]
