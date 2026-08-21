"""Phase 4.2 — how much resolution a claim needs depends on the claim.

`evidence_region` and `output_size` fix two different losses, and the difference
is the reason this exists. The region decides *what* is in the crop; the size
decides how much *detail* survives being resampled into it. Narrowing to the head
band stopped spending the canvas on legs, but the band is still squeezed into the
canonical square — so on 1712x1032 footage a hairnet arrived as roughly 30px of
fabric and read as a bare head. Framing could not fix that; only resolution could.

Measured on `datasets/kitchen-01`: identical model, prompt, region and detector,
23.3 % head accuracy at 224 and 74.4 % at 448.

Raised per attribute rather than globally because vision tokens scale with
**area** — 448 costs 4x the tokens of 224, and paying that on every question to
fix one would be a cost with no measured return.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import PartFocusedCropStrategy
from vision_os.core.model.ids import AttributeKey, ClassId
from vision_os.core.model.space import Box

HEAD = AttributeKey("head_covering")
HANDS = AttributeKey("hand_covering")
PERSON = ClassId("person")
SUBJECT = Box(0.3, 0.1, 0.5, 0.9)


def plan_for(strategy: PartFocusedCropStrategy, *attributes: AttributeKey):
    return strategy.plan(
        box=SUBJECT,
        class_id=PERSON,
        source_width=1712,
        source_height=1032,
        attributes=attributes,
    )


@pytest.fixture
def strategy() -> PartFocusedCropStrategy:
    """Configured the way the kitchen policy configures one: heads need the
    detail, hands have no measurement saying they do."""
    return PartFocusedCropStrategy(
        regions={"head_covering": (0.0, 0.45), "hand_covering": (0.15, 0.55)},
        output_sizes={"head_covering": (448, 448)},
    )


def test_a_declared_attribute_gets_its_own_size(strategy) -> None:
    plan = plan_for(strategy, HEAD)
    assert (plan.output_width, plan.output_height) == (448, 448)


def test_an_undeclared_attribute_keeps_the_deployment_default(strategy) -> None:
    """The safety property: configuring one attribute must not move another."""
    plan = plan_for(strategy, HANDS)
    assert (plan.output_width, plan.output_height) == (224, 224)


def test_a_strategy_declaring_nothing_behaves_exactly_as_before() -> None:
    """No deployment is silently upgraded. 448 costs 4x the tokens, and a
    platform that quietly started paying it would be a bill nobody approved."""
    strategy = PartFocusedCropStrategy(regions={"head_covering": (0.0, 0.45)})
    assert plan_for(strategy, HEAD).output_width == 224


def test_a_shared_crop_takes_the_largest_declared_size(strategy) -> None:
    """Mirrors the gate's strictest-floor rule, for the same reason.

    A crop answering both questions rendered at the smaller size would answer the
    demanding one from detail its own policy said was insufficient — and the loss
    would be invisible in the result.
    """
    assert plan_for(strategy, HEAD, HANDS).output_width == 448
    assert plan_for(strategy, HANDS, HEAD).output_width == 448


def test_the_size_is_chosen_by_area_not_by_a_single_side() -> None:
    """Area is what both the cost and the detail scale with."""
    strategy = PartFocusedCropStrategy(
        output_sizes={"head_covering": (600, 100), "hand_covering": (300, 300)}
    )
    assert plan_for(strategy, HEAD, HANDS).output_width == 300


def test_a_non_positive_size_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="output_size"):
        PartFocusedCropStrategy(output_sizes={"head_covering": (0, 448)})
    with pytest.raises(ValueError, match="output_size"):
        PartFocusedCropStrategy(output_sizes={"head_covering": (448, -1)})


def test_resolution_does_not_disturb_the_evidence_region(strategy) -> None:
    """Framing and resolution must stay independent.

    If raising the size also moved the band, the 448 measurement would be
    confounded and could not be attributed to resolution at all.
    """
    narrow = PartFocusedCropStrategy(
        regions={"head_covering": (0.0, 0.45)}, output_sizes={"head_covering": (448, 448)}
    )
    plain = PartFocusedCropStrategy(regions={"head_covering": (0.0, 0.45)})
    assert plan_for(narrow, HEAD).padded_box == plan_for(plain, HEAD).padded_box


def test_the_union_of_bands_is_unchanged_by_sizing(strategy) -> None:
    """Existing evidence-grouping behaviour must survive untouched."""
    both = plan_for(strategy, HEAD, HANDS).padded_box
    head_only = plan_for(strategy, HEAD).padded_box
    assert both.y2 > head_only.y2, "the union must still reach into the hand band"


def test_an_attribute_with_no_region_still_gets_its_size() -> None:
    """Size and region are declared independently; neither implies the other."""
    strategy = PartFocusedCropStrategy(output_sizes={"head_covering": (448, 448)})
    plan = plan_for(strategy, HEAD)
    assert plan.output_width == 448
    assert plan.padded_box.y2 > SUBJECT.y2 - 0.01, "no region means the whole subject"
