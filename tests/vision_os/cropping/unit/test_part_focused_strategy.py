"""P14 ``crop.part_focused`` — spending the crop on the region a question is about.

The measurement behind this file. A standing person is roughly 1:3 and the
canonical crop is square, so ``crop.padded`` letterboxes one into the other and
about **83%** of the image is black bar; the head lands in a few dozen pixels.
On real kitchen CCTV that was enough to lose plainly visible blue hairnets — the
model answered ``none`` for a covered head and the rule turned it into a
violation.

The strategy narrows to the band the demanded attributes declared and widens it
to something near square, so the same 224x224 is spent on the subject instead of
on bars. It reads geometry from the policy document and never learns what an
attribute means, which is what obligation C5 permits and requires.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import PaddedCropStrategy, PartFocusedCropStrategy
from vision_os.core.model.ids import AttributeKey, ClassId
from vision_os.core.model.space import Box

#: A standing person as this camera actually frames one: tall, thin, mid-frame.
STANDING = Box(0.40, 0.20, 0.46, 0.55)

HEAD = AttributeKey("head_covering")
HANDS = AttributeKey("hand_covering")

REGIONS = {"head_covering": (0.0, 0.45), "hand_covering": (0.15, 0.55)}


def plan_for(*attributes, regions=REGIONS, **kwargs):
    strategy = PartFocusedCropStrategy(regions=regions, **kwargs)
    return strategy.plan(
        box=STANDING,
        class_id=ClassId("person"),
        source_width=1712,
        source_height=1032,
        attributes=attributes,
    )


def aspect(plan) -> float:
    box = plan.padded_box
    return (box.x2 - box.x1) / (box.y2 - box.y1)


class TestItStopsWastingTheCropOnBlackBars:
    def test_the_whole_person_letterboxes_badly(self) -> None:
        """The baseline this exists to beat, asserted rather than asserted about."""
        padded = PaddedCropStrategy().plan(
            box=STANDING, class_id=ClassId("person"), source_width=1712, source_height=1032
        )

        assert aspect(padded) < 0.25

    @pytest.mark.parametrize(
        "attributes", [(HEAD,), (HANDS,), (HEAD, HANDS), (AttributeKey("undeclared"),)]
    )
    def test_every_plan_is_near_square(self, attributes) -> None:
        assert aspect(plan_for(*attributes)) >= 0.75 - 1e-6

    def test_a_wide_subject_is_left_alone(self) -> None:
        """Widening exists to fix tall boxes. A box already wider than the floor
        keeps its own width — inflating it would add context nobody asked for."""
        wide = PartFocusedCropStrategy(regions={}).plan(
            box=Box(0.10, 0.40, 0.90, 0.60),
            class_id=ClassId("person"),
            source_width=1712,
            source_height=1032,
            attributes=(),
        )

        assert wide.padded_box.x1 < 0.10
        assert wide.padded_box.x2 > 0.90


class TestItLooksWhereTheQuestionIs:
    def test_a_head_question_gets_the_top_of_the_subject(self) -> None:
        head = plan_for(HEAD)

        assert head.padded_box.y1 < STANDING.y1 + STANDING.height * 0.10
        assert head.padded_box.y2 < STANDING.y1 + STANDING.height * 0.60

    def test_a_hand_question_reaches_further_down(self) -> None:
        assert plan_for(HANDS).padded_box.y2 > plan_for(HEAD).padded_box.y2

    def test_both_questions_union_rather_than_pick_one(self) -> None:
        """One crop has to answer everything asked of it.

        M8 admits one request per decision, so a band satisfying only the first
        attribute would leave the rest answered from pixels that do not contain
        them — the exact failure this strategy exists to remove.
        """
        both = plan_for(HEAD, HANDS)

        assert both.padded_box.y1 <= plan_for(HEAD).padded_box.y1 + 1e-9
        assert both.padded_box.y2 >= plan_for(HANDS).padded_box.y2 - 1e-9


class TestItDegradesToThePreviousBehaviour:
    def test_an_undeclared_attribute_gets_the_whole_subject(self) -> None:
        """A policy that declared no region must not be cropped by guesswork."""
        span = plan_for(AttributeKey("something_nobody_declared"))
        padded = PaddedCropStrategy().plan(
            box=STANDING, class_id=ClassId("person"), source_width=1712, source_height=1032
        )

        assert span.padded_box.y1 == pytest.approx(padded.padded_box.y1, abs=1e-6)
        assert span.padded_box.y2 == pytest.approx(padded.padded_box.y2, abs=1e-6)

    def test_no_regions_at_all_is_a_working_configuration(self) -> None:
        assert plan_for(HEAD, regions={}).padded_box.y2 > STANDING.y2 - 1e-6


class TestItRefusesAMalformedRegion:
    """At construction, while someone is looking at the document."""

    @pytest.mark.parametrize("span", [(-0.1, 0.5), (0.0, 0.0), (0.8, 0.5), (0.0, 1.5)])
    def test_a_band_outside_the_subject_is_refused(self, span) -> None:
        with pytest.raises(ValueError, match="region for"):
            PartFocusedCropStrategy(regions={"k": span})

    def test_it_declares_its_identity(self) -> None:
        assert PartFocusedCropStrategy().strategy_id == "crop.part_focused"


class TestItStaysWithinTheFrame:
    """Obligation C1 — the planned box lies inside the frame after clamping."""

    @pytest.mark.parametrize(
        "box",
        [
            Box(0.0, 0.0, 0.05, 0.30),      # against the left/top edge
            Box(0.95, 0.70, 1.0, 1.0),      # against the right/bottom edge
            Box(0.48, 0.0, 0.52, 1.0),      # full height
        ],
    )
    def test_an_edge_subject_still_produces_a_legal_plan(self, box) -> None:
        plan = PartFocusedCropStrategy(regions=REGIONS).plan(
            box=box,
            class_id=ClassId("person"),
            source_width=1712,
            source_height=1032,
            attributes=(HEAD, HANDS),
        )
        planned = plan.padded_box

        assert 0.0 <= planned.x1 < planned.x2 <= 1.0
        assert 0.0 <= planned.y1 < planned.y2 <= 1.0
