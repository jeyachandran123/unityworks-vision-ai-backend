"""Scoring ground truth against Vision OS predictions.

Three decisions shape everything here, and each exists because the obvious
alternative produces a flattering number.

**Per attribute, never one figure.** A single "accuracy" hides the case that
matters: a system that reads heads well and guesses at hands scores respectably
overall while being dangerous about hands.

**``NOT_VISIBLE`` is scored, not excluded.** It is a claim about the footage, and
agreeing with the annotator that the hands could not be seen is a *correct*
answer. Dropping those rows would delete the behaviour Phase 2 exists to produce
and reward a system that guesses.

**Precision and recall are computed per state.** "Precision" of a three-state
attribute is meaningless without saying precision *of what*. The interesting
number for PPE is precision of ``ABSENT`` — how often a reported violation was
real — and it is invisible in a macro average.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .schema import (
    AnnotatedFrame,
    AttributeState,
    FailureCategory,
    PredictedFrame,
    PredictedSubject,
)

#: Overlap at which a prediction is considered to be about the same subject.
#:
#: 0.5 is the long-standing detection convention. It is a **matching** threshold,
#: not an accuracy claim: a lower value would credit sloppier boxes, a higher one
#: would score correct attributes as misses because the box was loose.
DEFAULT_IOU = 0.5


@dataclass(frozen=True, slots=True)
class Failure:
    """One wrong answer, with enough context to open the evidence."""

    frame_id: str
    video_id: str
    subject_id: str
    object_id: str
    attribute: str
    truth: str
    predicted: str
    category: FailureCategory
    crop_id: str = ""
    crop_size: str = ""
    quality: str = ""
    skip_reason: str = ""
    model_id: str = ""
    vlm_used: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class StateScore:
    """Precision and recall of one state of one attribute."""

    state: str
    support: int
    """How many times the annotator recorded this state. A precision computed
    over two rows is noise, and this is what says so."""

    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float | None:
        """``None`` when the state was never predicted — not zero.

        Zero would read as "always wrong"; the truth is "never attempted", and a
        mean over fabricated zeros is a fabricated mean.
        """
        return None if self.predicted == 0 else self.true_positive / self.predicted

    @property
    def recall(self) -> float | None:
        return None if self.support == 0 else self.true_positive / self.support

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass(frozen=True, slots=True)
class AttributeReport:
    attribute: str
    matched: int
    correct: int
    states: tuple[StateScore, ...] = ()
    confusion: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    missing_prediction: int = 0
    """Annotated subjects the system said nothing about. Counted separately from
    a wrong answer: silence and error have different causes."""

    @property
    def accuracy(self) -> float | None:
        return None if self.matched == 0 else self.correct / self.matched

    def rate_of(self, state: AttributeState) -> float | None:
        """How often the **system** produced this state, over matched subjects."""
        if self.matched == 0:
            return None
        produced = sum(
            count
            for truth_row in self.confusion.values()
            for predicted, count in truth_row.items()
            if predicted == state.value
        )
        return produced / self.matched

    def score(self, state: AttributeState) -> StateScore | None:
        return next((s for s in self.states if s.state == state.value), None)

    @property
    def unsupported_claims(self) -> int:
        """Decided answers where the annotator could not see the region.

        The headline safety number. Every one of these is the system asserting
        something about pixels a human judged unreadable — the failure mode this
        whole programme is built around.
        """
        row = self.confusion.get(AttributeState.NOT_VISIBLE.value, {})
        return row.get(AttributeState.PRESENT.value, 0) + row.get(
            AttributeState.ABSENT.value, 0
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    frames: int
    annotated_subjects: int
    matched_subjects: int
    unmatched_truth: int
    """Annotated people the detector never found. A detection failure, and it
    must not be silently excluded — a system that detects one person perfectly
    would otherwise score 100%."""

    spurious_predictions: int
    attributes: tuple[AttributeReport, ...] = ()
    failures: tuple[Failure, ...] = ()
    vlm_calls: int = 0
    vlm_call_reasons: Mapping[str, int] = field(default_factory=dict)

    @property
    def detection_recall(self) -> float | None:
        total = self.matched_subjects + self.unmatched_truth
        return None if total == 0 else self.matched_subjects / total

    @property
    def vlm_calls_per_1000_frames(self) -> float | None:
        return None if self.frames == 0 else self.vlm_calls * 1000.0 / self.frames

    @property
    def vlm_calls_per_person(self) -> float | None:
        m = self.matched_subjects
        return None if m == 0 else self.vlm_calls / m

    def attribute(self, name: str) -> AttributeReport | None:
        return next((a for a in self.attributes if a.attribute == name), None)

    def failures_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self.failures:
            counts[failure.category.value] = counts.get(failure.category.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _categorise(
    truth: AttributeState, predicted: AttributeState, subject: PredictedSubject,
    attribute: str,
) -> tuple[FailureCategory, str]:
    """Attribute a failure to a stage, or admit it cannot be attributed.

    Reads the platform's own recorded reasons before inferring anything. A crop
    the gate rejected for blur is a blur failure because the gate said so — not
    because this function guessed from the states.
    """
    quality = subject.quality.get(attribute, "")
    skip = subject.skip_reason.get(attribute, "")

    for token, category in (
        ("too_blurry", FailureCategory.MOTION_BLUR),
        ("too_small", FailureCategory.LOW_RESOLUTION),
        ("too_occluded", FailureCategory.OCCLUSION),
        ("too_truncated", FailureCategory.TRUNCATION),
        ("degenerate", FailureCategory.CROP_FAILURE),
    ):
        if token in quality or token in skip:
            return category, f"gate recorded {quality or skip}"

    # The system claimed something about a region the annotator could not read.
    if truth is AttributeState.NOT_VISIBLE and predicted.is_decided:
        return (
            FailureCategory.UNKNOWN_HANDLING_FAILURE,
            "claimed a decided state where the annotator saw no usable evidence",
        )

    # The region was readable and the system declined. Not dangerous, but it is
    # coverage lost, and its cause is usually the gate or the prompt.
    if truth.is_decided and predicted is AttributeState.NOT_VISIBLE:
        return (
            FailureCategory.SEMANTIC_MODEL_FAILURE,
            "declined a region the annotator could read",
        )

    if truth.is_decided and predicted.is_decided:
        return (
            FailureCategory.VLM_FAILURE if subject.vlm_used
            else FailureCategory.SEMANTIC_MODEL_FAILURE,
            "the region was readable and the answer was wrong",
        )

    return FailureCategory.UNKNOWN_FAILURE_REASON, ""


def evaluate(
    truth_frames: Iterable[AnnotatedFrame],
    predicted_frames: Iterable[PredictedFrame],
    *,
    attributes: Sequence[str],
    iou: float = DEFAULT_IOU,
) -> EvaluationReport:
    """Score predictions against ground truth. Pure — no I/O, no clock."""
    predictions = {f.frame_id: f for f in predicted_frames}
    truth_list = list(truth_frames)

    matched = unmatched = spurious = 0
    annotated = 0
    failures: list[Failure] = []
    confusion: dict[str, dict[str, dict[str, int]]] = {
        a: {} for a in attributes
    }
    tallies: dict[str, dict[str, int]] = {a: {"matched": 0, "correct": 0, "missing": 0} for a in attributes}
    vlm_calls = 0
    reasons: dict[str, int] = {}

    for frame in truth_list:
        annotated += len(frame.subjects)
        predicted_frame = predictions.get(frame.frame_id)
        if predicted_frame is None:
            unmatched += len(frame.subjects)
            continue

        vlm_calls += predicted_frame.vlm_calls
        for reason, count in predicted_frame.vlm_call_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count

        available = list(predicted_frame.subjects)
        for subject in frame.subjects:
            best, best_iou = None, 0.0
            for candidate in available:
                if candidate.box is None:
                    continue
                score = subject.box.iou(candidate.box)
                if score > best_iou:
                    best, best_iou = candidate, score

            if best is None or best_iou < iou:
                unmatched += 1
                continue

            available.remove(best)
            matched += 1

            for attribute in attributes:
                truth_state = subject.state(attribute)
                if truth_state is None:
                    continue
                predicted_state = best.state(attribute)
                tallies[attribute]["matched"] += 1

                if predicted_state is None:
                    tallies[attribute]["missing"] += 1
                    row = confusion[attribute].setdefault(truth_state.value, {})
                    row["<none>"] = row.get("<none>", 0) + 1
                    continue

                row = confusion[attribute].setdefault(truth_state.value, {})
                row[predicted_state.value] = row.get(predicted_state.value, 0) + 1

                if predicted_state is truth_state:
                    tallies[attribute]["correct"] += 1
                    continue

                category, detail = _categorise(truth_state, predicted_state, best, attribute)
                failures.append(
                    Failure(
                        frame_id=frame.frame_id,
                        video_id=frame.video_id,
                        subject_id=subject.subject_id,
                        object_id=best.object_id,
                        attribute=attribute,
                        truth=truth_state.value,
                        predicted=predicted_state.value,
                        category=category,
                        crop_id=best.crop_ids.get(attribute, ""),
                        crop_size=best.crop_size.get(attribute, ""),
                        quality=best.quality.get(attribute, ""),
                        skip_reason=best.skip_reason.get(attribute, ""),
                        model_id=best.model_id,
                        vlm_used=best.vlm_used,
                        detail=detail,
                    )
                )

        spurious += len(available)

    reports = []
    for attribute in attributes:
        table = confusion[attribute]
        states = []
        for state in AttributeState:
            support = sum(table.get(state.value, {}).values())
            predicted_count = sum(
                row.get(state.value, 0) for row in table.values()
            )
            tp = table.get(state.value, {}).get(state.value, 0)
            states.append(
                StateScore(
                    state=state.value,
                    support=support,
                    predicted=predicted_count,
                    true_positive=tp,
                    false_positive=predicted_count - tp,
                    false_negative=support - tp,
                )
            )
        reports.append(
            AttributeReport(
                attribute=attribute,
                matched=tallies[attribute]["matched"],
                correct=tallies[attribute]["correct"],
                missing_prediction=tallies[attribute]["missing"],
                states=tuple(states),
                confusion={k: dict(v) for k, v in table.items()},
            )
        )

    return EvaluationReport(
        frames=len(truth_list),
        annotated_subjects=annotated,
        matched_subjects=matched,
        unmatched_truth=unmatched,
        spurious_predictions=spurious,
        attributes=tuple(reports),
        failures=tuple(failures),
        vlm_calls=vlm_calls,
        vlm_call_reasons=reasons,
    )


__all__ = [
    "DEFAULT_IOU",
    "AttributeReport",
    "EvaluationReport",
    "Failure",
    "StateScore",
    "evaluate",
]
