"""Double annotation, adjudication and inter-annotator agreement.

The machinery P9's gold set needs and could not exercise, because producing the
second independent annotation requires a second human.

### Three rules this module enforces

**Independence.** Two annotations of the same sample must come from different
annotators. A second pass by the same person measures their consistency, not the
task's difficulty, and calling that agreement would flatter a hard dataset.

**No silent resolution.** Disagreement produces `DISPUTED`, never a coin flip and
never "take the first one". Only an explicit `Adjudication` by a third party
resolves it, and the losing annotation is kept — deleting the disagreement
destroys the evidence that the example was hard.

**Agreement is reported per region and per field, never as one number.** Head
observability and left-hand state are different tasks with different difficulty;
one aggregate would hide a region nobody can label reliably behind three that
everyone can.

### Why raw agreement rather than kappa, for now

Cohen's kappa corrects for chance using the marginal distribution, and on this
corpus the marginals are extreme — one class dominates most regions. Kappa
becomes unstable and famously counter-intuitive there (high agreement, near-zero
kappa), which reads as a broken dataset rather than a skewed one. Raw agreement
plus explicit per-class counts is the honest report at this scale; kappa is
computed and shown **alongside** it once support allows, never instead of it.
"""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .schema import (
    LabelProvenance,
    QualityStatus,
    Region,
    SubjectAnnotation,
)

#: Support below which an agreement figure is reported as a count, not a rate.
MIN_PAIRS_FOR_RATE = 20


@dataclass(frozen=True, slots=True)
class Adjudication:
    """A third party's resolution of a disagreement. Keeps both originals."""

    sample_id: str
    region: Region
    annotator_a: str
    annotator_b: str
    observability_a: str
    observability_b: str
    state_a: str
    state_b: str
    adjudicator: str
    resolved_observability: str
    resolved_state: str
    reason: str
    resolved_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(
                f"{self.sample_id}/{self.region.value}: an adjudication without a "
                f"reason cannot be reviewed, and an unreviewable resolution is "
                f"indistinguishable from a coin flip"
            )
        if self.adjudicator in (self.annotator_a, self.annotator_b):
            raise ValueError(
                f"{self.sample_id}: the adjudicator was one of the original "
                f"annotators; a tie cannot be broken by a player"
            )


@dataclass(frozen=True, slots=True)
class Disagreement:
    sample_id: str
    region: Region
    field_name: str
    value_a: str
    value_b: str
    annotator_a: str
    annotator_b: str


def pair_annotations(
    first: list[SubjectAnnotation], second: list[SubjectAnnotation]
) -> list[tuple[SubjectAnnotation, SubjectAnnotation]]:
    """Match two independent passes by `sample_id`.

    Raises when the same annotator produced both. Independence is the property
    that makes the number mean anything, so it is checked rather than assumed.
    """
    index = {s.sample_id: s for s in second}
    pairs = []
    for sample in first:
        other = index.get(sample.sample_id)
        if other is None:
            continue
        if sample.annotator == other.annotator:
            raise ValueError(
                f"{sample.sample_id}: both annotations are by {sample.annotator!r}. "
                f"A second pass by the same person measures self-consistency, not "
                f"inter-annotator agreement."
            )
        pairs.append((sample, other))
    return pairs


def disagreements(
    pairs: list[tuple[SubjectAnnotation, SubjectAnnotation]],
) -> list[Disagreement]:
    """Every field on which two annotators differ, region by region."""
    out: list[Disagreement] = []
    for a, b in pairs:
        for region in Region:
            ra, rb = a.region_of(region), b.region_of(region)
            if ra is None or rb is None:
                continue
            if ra.observability is not rb.observability:
                out.append(
                    Disagreement(
                        a.sample_id, region, "observability",
                        ra.observability.value, rb.observability.value,
                        a.annotator, b.annotator,
                    )
                )
            if ra.state is not rb.state:
                out.append(
                    Disagreement(
                        a.sample_id, region, "state",
                        ra.state.value, rb.state.value,
                        a.annotator, b.annotator,
                    )
                )
    return out


def agreement(pairs: list[tuple[SubjectAnnotation, SubjectAnnotation]]) -> dict:
    """Per region, per field. Never one number.

    `rate` is `None` below `MIN_PAIRS_FOR_RATE`: a percentage from four
    comparisons is noise wearing a decimal point.
    """
    report: dict[str, dict] = {}
    for region in Region:
        for field_name in ("observability", "state"):
            agreed = total = 0
            confusion: collections.Counter = collections.Counter()
            for a, b in pairs:
                ra, rb = a.region_of(region), b.region_of(region)
                if ra is None or rb is None:
                    continue
                va = getattr(ra, field_name).value
                vb = getattr(rb, field_name).value
                total += 1
                agreed += va == vb
                confusion[(va, vb)] += 1
            report[f"{region.value}.{field_name}"] = {
                "pairs": total,
                "agreed": agreed,
                "disagreed": total - agreed,
                "rate": round(agreed / total, 4) if total >= MIN_PAIRS_FOR_RATE else None,
                "rate_suppressed_reason": (
                    None
                    if total >= MIN_PAIRS_FOR_RATE
                    else f"only {total} pair(s); below {MIN_PAIRS_FOR_RATE}"
                ),
                "kappa": _kappa(confusion) if total >= MIN_PAIRS_FOR_RATE else None,
                "confusion": {f"{k[0]}|{k[1]}": v for k, v in sorted(confusion.items())},
            }
    return report


def _kappa(confusion: collections.Counter) -> float | None:
    """Cohen's kappa. Shown beside raw agreement, never instead of it."""
    total = sum(confusion.values())
    if not total:
        return None
    observed = sum(v for (a, b), v in confusion.items() if a == b) / total
    marginal_a: collections.Counter = collections.Counter()
    marginal_b: collections.Counter = collections.Counter()
    for (a, b), v in confusion.items():
        marginal_a[a] += v
        marginal_b[b] += v
    expected = sum(
        (marginal_a[k] / total) * (marginal_b[k] / total)
        for k in set(marginal_a) | set(marginal_b)
    )
    if expected >= 1.0:
        return None  # degenerate: one class only, kappa undefined
    return round((observed - expected) / (1 - expected), 4)


def agreement_by(
    pairs: list[tuple[SubjectAnnotation, SubjectAnnotation]], key: str = "camera_id"
) -> dict:
    """Agreement stratified by camera, session or day.

    A dataset can agree well overall and be unlabelable on one camera. Reporting
    only the aggregate would hide exactly the camera worth fixing.
    """
    grouped: dict[str, list] = {}
    for a, b in pairs:
        grouped.setdefault(getattr(a, key), []).append((a, b))
    return {
        value: {
            "pairs": len(group),
            "regions": agreement(group),
        }
        for value, group in sorted(grouped.items())
    }


def apply_adjudications(
    annotations: list[SubjectAnnotation], adjudications: list[Adjudication]
) -> list[SubjectAnnotation]:
    """Fold resolutions in, marking the result `HUMAN_ADJUDICATED`.

    The provenance change is the point: a sample that needed a third opinion is
    permanently marked as one, so a later failure on it is recognisable as a hard
    case rather than a surprise.
    """
    from dataclasses import replace

    from .schema import AttributeState, Observability, RegionAnnotation

    resolved: dict[str, list[Adjudication]] = {}
    for entry in adjudications:
        resolved.setdefault(entry.sample_id, []).append(entry)

    out = []
    for sample in annotations:
        entries = resolved.get(sample.sample_id)
        if not entries:
            out.append(sample)
            continue
        regions = list(sample.regions)
        for entry in entries:
            for index, region in enumerate(regions):
                if region.region is entry.region:
                    regions[index] = RegionAnnotation(
                        region=region.region,
                        observability=Observability(entry.resolved_observability),
                        state=AttributeState(entry.resolved_state),
                        hard_case_tags=region.hard_case_tags,
                        note=f"adjudicated by {entry.adjudicator}: {entry.reason}"[:160],
                    )
        out.append(
            replace(
                sample,
                regions=tuple(regions),
                label_provenance=LabelProvenance.HUMAN_ADJUDICATED,
                quality_status=QualityStatus.ACCEPTED,
            )
        )
    return out


def summary(
    pairs: list[tuple[SubjectAnnotation, SubjectAnnotation]],
    adjudications: list[Adjudication] = (),
) -> dict:
    found = disagreements(pairs)
    per_region = agreement(pairs)
    rates = [v["rate"] for v in per_region.values() if v["rate"] is not None]
    return {
        "_comment": [
            "Agreement is reported PER REGION and PER FIELD. There is deliberately",
            "no single headline number: head observability and left-hand state are",
            "different tasks, and one average would hide a region nobody can label.",
            "A rate is suppressed below 20 pairs — a percentage from four",
            "comparisons is noise wearing a decimal point.",
        ],
        "pairs": len(pairs),
        "disagreements": len(found),
        "adjudications": len(adjudications),
        "unresolved": len(found) - len(adjudications),
        "median_rate_where_reportable": (
            round(statistics.median(rates), 4) if rates else None
        ),
        "by_region_field": per_region,
        "by_camera": agreement_by(pairs, "camera_id"),
        "disagreement_detail": [
            {
                "sample_id": d.sample_id,
                "region": d.region.value,
                "field": d.field_name,
                "a": d.value_a,
                "b": d.value_b,
                "annotator_a": d.annotator_a,
                "annotator_b": d.annotator_b,
            }
            for d in found
        ],
    }
