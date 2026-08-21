"""Crop strategies, extraction, and the transform record.

02_VOM section 10.7 on why ``transform`` exists: *"two models evaluated on
differently-letterboxed crops are not comparable, and without this field nobody
finds out."*

So the tests that matter here are not "does it produce bytes" — they are "does
the record match what actually happened". A crop whose transform record and real
transform disagree is worse than one with no record at all, because it invites a
comparison that looks valid and is not.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import (
    PaddedCropStrategy,
    ReferenceCropExtractor,
    TightCropStrategy,
)
from vision_os.core.errors import CropExtractionError
from vision_os.core.model.crop import CropTransform
from vision_os.core.model.ids import ClassId
from vision_os.core.model.space import Box
from vision_os.core.ports.cropping import CropPlan

from ..conftest import FRAME_HEIGHT, FRAME_WIDTH, PERSON, sharp_frame


def plan_for(strategy, box: Box, *, class_id: ClassId = PERSON) -> CropPlan:
    return strategy.plan(
        box=box,
        class_id=class_id,
        source_width=FRAME_WIDTH,
        source_height=FRAME_HEIGHT,
    )


class TestCropStrategies:
    def test_tight_applies_no_padding(self) -> None:
        strategy = TightCropStrategy(output_size=(64, 64))
        plan = plan_for(strategy, Box(0.4, 0.3, 0.55, 0.85))
        assert plan.padding_applied == 0.0
        assert plan.padded_box == Box(0.4, 0.3, 0.55, 0.85)

    def test_padded_adds_proportional_context(self) -> None:
        """Padding is a fraction of the box's own size, not a fixed pixel count.

        A fixed pad would swamp a distant object and barely touch a near one, so
        two objects at different depths would get incomparable crops.
        """
        strategy = PaddedCropStrategy(padding=0.5, output_size=(64, 64))
        plan = plan_for(strategy, Box(0.4, 0.4, 0.5, 0.6))
        assert plan.padded_box.x1 == pytest.approx(0.35)
        assert plan.padded_box.x2 == pytest.approx(0.55)
        assert plan.padded_box.y1 == pytest.approx(0.30)
        assert plan.padded_box.y2 == pytest.approx(0.70)

    def test_padding_clamps_at_the_frame_edge(self) -> None:
        """Obligation C1. An unclamped box reads outside the buffer."""
        strategy = PaddedCropStrategy(padding=0.9, output_size=(64, 64))
        plan = plan_for(strategy, Box(0.02, 0.02, 0.12, 0.3))
        assert plan.padded_box.x1 >= 0.0
        assert plan.padded_box.y1 >= 0.0
        assert plan.padded_box.x2 <= 1.0

    def test_the_requested_ratio_is_still_recorded_after_clamping(self) -> None:
        """An edge object gets less context than asked for — and the crop says so.

        ``padding_applied`` records the *request*; the transform records the
        rectangle actually read. Recording only one would hide which of the two
        happened.
        """
        strategy = PaddedCropStrategy(padding=0.5, output_size=(64, 64))
        plan = plan_for(strategy, Box(0.0, 0.0, 0.1, 0.2))
        assert plan.padding_applied == 0.5

    def test_output_size_is_fixed_regardless_of_object_size(self) -> None:
        """One crop format for every model (§M8 STANDARDIZATION)."""
        strategy = PaddedCropStrategy(output_size=(224, 224))
        tiny = plan_for(strategy, Box(0.5, 0.5, 0.51, 0.52))
        huge = plan_for(strategy, Box(0.05, 0.02, 0.95, 0.98))
        assert (tiny.output_width, tiny.output_height) == (224, 224)
        assert (huge.output_width, huge.output_height) == (224, 224)

    def test_a_degenerate_box_still_produces_a_plan(self) -> None:
        """The gate refuses hopeless crops *with a reason*; geometry must not raise.

        A strategy that raised would produce the same outcome with no statistic
        attached — an invisible failure rather than a counted one.
        """
        strategy = PaddedCropStrategy(output_size=(64, 64))
        plan = plan_for(strategy, Box(0.5, 0.5, 0.5001, 0.5001))
        assert plan.padded_box.area > 0

    def test_an_entirely_out_of_frame_box_still_plans(self) -> None:
        strategy = TightCropStrategy(output_size=(64, 64))
        plan = plan_for(strategy, Box(1.5, 1.5, 1.8, 1.9))
        assert plan.padded_box.area > 0

    def test_plans_are_deterministic(self) -> None:
        strategy = PaddedCropStrategy(output_size=(64, 64))
        box = Box(0.33, 0.21, 0.58, 0.79)
        assert plan_for(strategy, box) == plan_for(strategy, box)

    @pytest.mark.parametrize(
        "kwargs",
        [{"padding": 5.0}, {"padding": -0.1}, {"output_size": (0, 64)}],
    )
    def test_invalid_strategy_configuration_is_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            PaddedCropStrategy(**kwargs)


class TestExtraction:
    def test_extraction_produces_the_declared_output_size(self, extractor) -> None:
        strategy = PaddedCropStrategy(output_size=(32, 32))
        plan = plan_for(strategy, Box(0.4, 0.3, 0.55, 0.85))
        payload, transform = extractor.extract(
            sharp_frame(),
            plan=plan,
            source_width=FRAME_WIDTH,
            source_height=FRAME_HEIGHT,
            channels=3,
            colour_space="bgr24",
        )
        assert len(payload) == 32 * 32 * 3
        assert (transform.output_width, transform.output_height) == (32, 32)

    def test_the_transform_records_what_actually_happened(self, extractor) -> None:
        """Not what was requested — what happened."""
        strategy = TightCropStrategy(output_size=(32, 32), preserve_aspect=False)
        plan = plan_for(strategy, Box(0.25, 0.5, 0.5, 0.75))
        _, transform = extractor.extract(
            sharp_frame(),
            plan=plan,
            source_width=FRAME_WIDTH,
            source_height=FRAME_HEIGHT,
            channels=3,
            colour_space="bgr24",
        )
        assert transform.crop_x == pytest.approx(0.25 * FRAME_WIDTH, abs=1)
        assert transform.crop_y == pytest.approx(0.5 * FRAME_HEIGHT, abs=1)
        assert transform.crop_width == pytest.approx(0.25 * FRAME_WIDTH, abs=2)
        assert transform.source_width == FRAME_WIDTH
        assert transform.colour_space == "bgr24"

    def test_letterboxing_is_recorded(self, extractor) -> None:
        """A tall object in a square output must be padded, and say so."""
        strategy = TightCropStrategy(output_size=(64, 64), preserve_aspect=True)
        plan = plan_for(strategy, Box(0.45, 0.1, 0.5, 0.9))
        _, transform = extractor.extract(
            sharp_frame(),
            plan=plan,
            source_width=FRAME_WIDTH,
            source_height=FRAME_HEIGHT,
            channels=3,
            colour_space="bgr24",
        )
        assert transform.letterboxed
        assert transform.pad_left > 0
        assert transform.aspect_preserved, (
            "aspect_preserved must agree with the recorded padding; a crop that "
            "claims preservation while squashing invites an invalid comparison"
        )

    def test_squashing_is_recorded_as_not_preserved(self, extractor) -> None:
        strategy = TightCropStrategy(output_size=(64, 64), preserve_aspect=False)
        plan = plan_for(strategy, Box(0.45, 0.1, 0.5, 0.9))
        _, transform = extractor.extract(
            sharp_frame(),
            plan=plan,
            source_width=FRAME_WIDTH,
            source_height=FRAME_HEIGHT,
            channels=3,
            colour_space="bgr24",
        )
        assert not transform.letterboxed
        assert not transform.aspect_preserved

    def test_extraction_is_deterministic(self, extractor) -> None:
        strategy = PaddedCropStrategy(output_size=(32, 32))
        plan = plan_for(strategy, Box(0.4, 0.3, 0.55, 0.85))
        pixels = sharp_frame()
        first = extractor.extract(
            pixels, plan=plan, source_width=FRAME_WIDTH, source_height=FRAME_HEIGHT,
            channels=3, colour_space="bgr24",
        )
        second = extractor.extract(
            pixels, plan=plan, source_width=FRAME_WIDTH, source_height=FRAME_HEIGHT,
            channels=3, colour_space="bgr24",
        )
        assert first == second

    def test_a_short_buffer_raises_rather_than_fabricating(self, extractor) -> None:
        """Extracting from a truncated buffer would silently fabricate evidence.

        §M8 RELIABILITY: *"Never silently fabricate crops."*
        """
        strategy = PaddedCropStrategy(output_size=(32, 32))
        plan = plan_for(strategy, Box(0.4, 0.3, 0.55, 0.85))
        with pytest.raises(CropExtractionError, match="fabricate"):
            extractor.extract(
                memoryview(b"\x00" * 100),
                plan=plan,
                source_width=FRAME_WIDTH,
                source_height=FRAME_HEIGHT,
                channels=3,
                colour_space="bgr24",
            )

    def test_a_zero_dimension_frame_raises(self, extractor) -> None:
        strategy = PaddedCropStrategy(output_size=(32, 32))
        plan = plan_for(strategy, Box(0.4, 0.3, 0.55, 0.85))
        with pytest.raises(CropExtractionError):
            extractor.extract(
                sharp_frame(),
                plan=plan,
                source_width=0,
                source_height=0,
                channels=3,
                colour_space="bgr24",
            )

    def test_the_extractor_reads_the_right_pixels(self) -> None:
        """A colour-coded frame proves the crop came from where it claims.

        Left half red, right half blue. A crop of the right half must be blue —
        a test that would catch an off-by-one in the source rectangle, which is
        otherwise invisible in a checkerboard.
        """
        width, height = 64, 64
        buffer = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                base = (y * width + x) * 3
                if x < width // 2:
                    buffer[base + 2] = 255  # red in BGR
                else:
                    buffer[base] = 255  # blue in BGR

        extractor = ReferenceCropExtractor()
        strategy = TightCropStrategy(output_size=(8, 8), preserve_aspect=False)
        plan = strategy.plan(
            box=Box(0.6, 0.2, 0.9, 0.8),
            class_id=PERSON,
            source_width=width,
            source_height=height,
        )
        payload, _ = extractor.extract(
            memoryview(bytes(buffer)),
            plan=plan,
            source_width=width,
            source_height=height,
            channels=3,
            colour_space="bgr24",
        )
        blues = payload[0::3]
        reds = payload[2::3]
        assert all(value == 255 for value in blues), "the right half is blue"
        assert all(value == 0 for value in reds), "no red may leak in"


class TestTransformRecordIntegrity:
    def test_a_degenerate_transform_cannot_be_constructed(self) -> None:
        with pytest.raises(ValueError, match="not evidence"):
            CropTransform(
                source_width=100,
                source_height=100,
                output_width=32,
                output_height=32,
                crop_x=0,
                crop_y=0,
                crop_width=0,
                crop_height=10,
            )

    def test_non_positive_dimensions_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            CropTransform(
                source_width=0,
                source_height=100,
                output_width=32,
                output_height=32,
                crop_x=0,
                crop_y=0,
                crop_width=10,
                crop_height=10,
            )

    def test_aspect_preserved_is_computed_not_asserted(self) -> None:
        """The flag is derived from the numbers, so it cannot lie independently."""
        square = CropTransform(
            source_width=100, source_height=100,
            output_width=32, output_height=32,
            crop_x=0, crop_y=0, crop_width=20, crop_height=20,
        )
        assert square.aspect_preserved

        squashed = CropTransform(
            source_width=100, source_height=100,
            output_width=32, output_height=32,
            crop_x=0, crop_y=0, crop_width=10, crop_height=40,
        )
        assert not squashed.aspect_preserved
