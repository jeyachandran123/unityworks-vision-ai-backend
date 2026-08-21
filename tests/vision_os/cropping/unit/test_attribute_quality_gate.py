"""Phase 4.1 — what counts as a usable crop depends on the question asked.

One global quality floor is wrong for any deployment that asks more than one
kind of question. A 60px-tall person is ample evidence for "is this a person";
the head band inside that box is 27px and cannot support "is the head covered".
A single floor has to be wrong for one of them, and it fails silently either
way — too low and a blurred head produces a confident answer, too high and every
distant worker goes unexamined.

These tests pin the behaviour that makes the difference: **an unusable crop
becomes NOT_VISIBLE with a named reason, and the expensive call is never made.**
"""

from __future__ import annotations

import pytest

from vision_os.core.model.crop import GateRejection
from vision_os.core.model.detection import ExposureLevel, QualityGrades, QualityLevel
from vision_os.perception.cropping.gate import GateThresholds, QualityGate

HEAD = "head_covering"
HANDS = "hand_covering"


def grades(**overrides) -> QualityGrades:
    """A crop that passes every default check unless a test spoils one axis."""
    base = {
        "scale_pixels": 400.0,
        "truncation": 0.0,
        "occlusion": 0.0,
        "blur": 0.1,
        "crowding": 0.0,
        "exposure": ExposureLevel.OK,
        "overall": QualityLevel.GOOD,
    }
    return QualityGrades(**{**base, **overrides})


@pytest.fixture
def gate() -> QualityGate:
    """A gate configured the way the kitchen policy configures one.

    Heads need a far larger region than hands: reading a hairnet needs the top
    of the head resolved, and that band is a fraction of a box that is itself a
    fraction of the frame.
    """
    return QualityGate(
        GateThresholds(),
        per_attribute={
            HEAD: GateThresholds(min_scale_pixels=220.0, max_blur=0.45),
            HANDS: GateThresholds(min_scale_pixels=120.0, max_blur=0.6),
        },
    )


# --- the five crop conditions the brief names -------------------------------- #


def test_a_good_crop_passes_and_reaches_the_model(gate: QualityGate) -> None:
    result = gate.evaluate(grades(), (HEAD,))
    assert result.passed
    assert result.reason is None


def test_a_blurred_crop_is_rejected_as_blur(gate: QualityGate) -> None:
    result = gate.evaluate(grades(blur=0.7), (HEAD,))
    assert not result.passed
    assert result.reason is GateRejection.TOO_BLURRY


def test_a_small_crop_is_rejected_as_scale(gate: QualityGate) -> None:
    result = gate.evaluate(grades(scale_pixels=90.0), (HEAD,))
    assert not result.passed
    assert result.reason is GateRejection.TOO_SMALL


def test_an_occluded_crop_is_rejected_as_occlusion(gate: QualityGate) -> None:
    result = gate.evaluate(grades(occlusion=0.85), (HEAD,))
    assert not result.passed
    assert result.reason is GateRejection.TOO_OCCLUDED


def test_a_truncated_crop_is_rejected_as_truncation(gate: QualityGate) -> None:
    result = gate.evaluate(grades(truncation=0.8), (HEAD,))
    assert not result.passed
    assert result.reason is GateRejection.TOO_TRUNCATED


# --- the floors are per attribute, which is the whole point ------------------- #


def test_the_same_crop_can_serve_one_question_and_not_another(gate: QualityGate) -> None:
    """The measurement that justifies this phase.

    150px is above the hand floor and below the head floor. Before Phase 4.1
    both questions were answered from it; one of those answers was unsupported.
    """
    marginal = grades(scale_pixels=150.0)
    assert gate.evaluate(marginal, (HANDS,)).passed
    assert not gate.evaluate(marginal, (HEAD,)).passed


def test_blur_tolerance_is_also_per_attribute(gate: QualityGate) -> None:
    smeared = grades(blur=0.55)
    assert gate.evaluate(smeared, (HANDS,)).passed
    assert not gate.evaluate(smeared, (HEAD,)).passed


def test_a_shared_crop_is_held_to_the_strictest_declared_floor(gate: QualityGate) -> None:
    """A crop serving both questions must satisfy the harder one.

    Otherwise the strict attribute is answered from evidence its own policy
    called insufficient — the exact silent failure the floors exist to stop.
    """
    marginal = grades(scale_pixels=150.0)
    assert not gate.evaluate(marginal, (HEAD, HANDS)).passed
    assert gate.evaluate(marginal, (HANDS, HEAD)).reason is GateRejection.TOO_SMALL


def test_an_undeclared_attribute_falls_back_to_the_default(gate: QualityGate) -> None:
    """A deployment that declares nothing behaves exactly as it did before."""
    assert gate.evaluate(grades(scale_pixels=60.0), ("garment_colour",)).passed


def test_omitting_attributes_uses_the_default_thresholds(gate: QualityGate) -> None:
    """Every pre-existing caller keeps its old behaviour."""
    assert gate.evaluate(grades(scale_pixels=60.0)).passed
    assert gate.thresholds_for(()) is gate.thresholds


def test_thresholds_for_picks_the_strictest_not_the_first(gate: QualityGate) -> None:
    assert gate.thresholds_for((HANDS, HEAD)).min_scale_pixels == 220.0
    assert gate.thresholds_for((HEAD, HANDS)).min_scale_pixels == 220.0


def test_would_pass_agrees_with_evaluate_for_the_same_attributes(gate: QualityGate) -> None:
    """The pre-check and the post-extraction check must not disagree.

    They are asked at different moments about the same crop; if they used
    different floors the engine would pay to extract a crop it then discards.
    """
    marginal = grades(scale_pixels=150.0)
    for attributes in ((HEAD,), (HANDS,), (HEAD, HANDS), ()):
        assert gate.would_pass(marginal, attributes) == gate.evaluate(
            marginal, attributes
        ).passed


# --- independence: one attribute failing must not silence the other ---------- #


def test_a_head_rejection_does_not_suppress_the_hand_answer(gate: QualityGate) -> None:
    """Grouped evidence must fail independently.

    A head band too small to read says nothing about whether the hands were
    readable. Coupling them would throw away good evidence and, worse, make an
    unrelated failure look like a hand failure.
    """
    head_only_bad = grades(scale_pixels=150.0)
    assert not gate.evaluate(head_only_bad, (HEAD,)).passed
    assert gate.evaluate(head_only_bad, (HANDS,)).passed


def test_the_rejection_reason_is_specific_not_generic(gate: QualityGate) -> None:
    """A NOT_VISIBLE nobody can explain cannot be engineered against.

    Knowing the head band failed on *scale* points at the camera or the crop;
    knowing it failed on *blur* points at shutter speed. 'Quality' points at
    nothing.
    """
    result = gate.evaluate(grades(scale_pixels=90.0), (HEAD,))
    assert result.reason is GateRejection.TOO_SMALL
    assert "90.0px" in result.detail and "220.0px" in result.detail


def test_scale_wins_attribution_over_blur_when_both_fail(gate: QualityGate) -> None:
    """A 9px object is TOO_SMALL, not TOO_BLURRY, though it is also blurred."""
    result = gate.evaluate(grades(scale_pixels=9.0, blur=0.99), (HEAD,))
    assert result.reason is GateRejection.TOO_SMALL


def test_degenerate_geometry_is_caught_before_any_threshold(gate: QualityGate) -> None:
    result = gate.evaluate(grades(scale_pixels=0.0), (HEAD,))
    assert result.reason is GateRejection.DEGENERATE_GEOMETRY


def test_per_attribute_thresholds_are_validated_like_any_other(gate: QualityGate) -> None:
    with pytest.raises(ValueError):
        GateThresholds(min_scale_pixels=-1.0)
    with pytest.raises(ValueError):
        GateThresholds(max_blur=1.5)


def test_the_gate_stays_pure(gate: QualityGate) -> None:
    """Same grades, same verdict — what lets a rejection be reproduced (V13)."""
    sample = grades(scale_pixels=150.0)
    first = gate.evaluate(sample, (HEAD,))
    second = gate.evaluate(sample, (HEAD,))
    assert (first.passed, first.reason, first.detail) == (
        second.passed,
        second.reason,
        second.detail,
    )
