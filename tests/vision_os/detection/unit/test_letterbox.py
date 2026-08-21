"""Letterboxing and its exact inverse.

The highest-value tests in the detection layer. A wrong inverse drifts boxes by a
few percent: detection still "works", tracking quietly degrades, object counts
skew, and the cause is found months later — if ever.

Every case here is pure arithmetic, so the correctness-critical part of every
YOLO-family adapter is verified in CI without a GPU, a model, or an image
library.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.detection.letterbox import (
    compute_transform,
    invert_to_normalized,
    truncation_of,
)

ASPECTS = [
    (640, 640),
    (1920, 1080),
    (1080, 1920),
    (3840, 2160),
    (1920, 60),
    (60, 1920),
    (33, 97),
    (100, 100),
    (1280, 40),
    (7, 5000),
]


class TestTransform:
    def test_square_source_into_square_target_needs_no_padding(self) -> None:
        transform = compute_transform(
            source_width=100, source_height=100, target_width=640, target_height=640
        )
        assert transform.scale == pytest.approx(6.4)
        assert transform.pad_x == pytest.approx(0.0)
        assert transform.pad_y == pytest.approx(0.0)

    def test_wide_source_is_padded_vertically(self) -> None:
        """A 16:9 image in a square box gets top and bottom bars."""
        transform = compute_transform(
            source_width=1920, source_height=1080, target_width=640, target_height=640
        )
        assert transform.scale == pytest.approx(640 / 1920)
        assert transform.pad_x == pytest.approx(0.0)
        assert transform.pad_y > 0

    def test_tall_source_is_padded_horizontally(self) -> None:
        transform = compute_transform(
            source_width=1080, source_height=1920, target_width=640, target_height=640
        )
        assert transform.pad_x > 0
        assert transform.pad_y == pytest.approx(0.0)

    def test_scale_preserves_aspect_ratio(self) -> None:
        for width, height in ASPECTS:
            transform = compute_transform(
                source_width=width, source_height=height,
                target_width=640, target_height=640,
            )
            scaled_width = width * transform.scale
            scaled_height = height * transform.scale
            assert scaled_width <= 640 + 1e-6
            assert scaled_height <= 640 + 1e-6
            # One dimension must fill the target exactly, or we scaled too small.
            assert scaled_width == pytest.approx(640) or scaled_height == pytest.approx(640)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="source size"):
            compute_transform(
                source_width=0, source_height=10, target_width=640, target_height=640
            )
        with pytest.raises(ValueError, match="target size"):
            compute_transform(
                source_width=10, source_height=10, target_width=0, target_height=640
            )


class TestInverseExactness:
    """The check that justifies the entire conformance mechanism."""

    @pytest.mark.parametrize(("width", "height"), ASPECTS)
    def test_full_frame_box_round_trips_to_the_unit_square(
        self, width: int, height: int
    ) -> None:
        """A box covering the whole source must invert to exactly [0,1].

        This is the single assertion that catches a mis-ordered inverse: get the
        padding or the scale wrong and the result is off by the letterbox bars.
        """
        transform = compute_transform(
            source_width=width, source_height=height, target_width=640, target_height=640
        )
        x1 = transform.pad_x
        y1 = transform.pad_y
        x2 = transform.pad_x + width * transform.scale
        y2 = transform.pad_y + height * transform.scale

        box = invert_to_normalized(transform, x1, y1, x2, y2)
        assert box.x1 == pytest.approx(0.0, abs=1e-9)
        assert box.y1 == pytest.approx(0.0, abs=1e-9)
        assert box.x2 == pytest.approx(1.0, abs=1e-9)
        assert box.y2 == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(("width", "height"), ASPECTS)
    def test_centre_box_round_trips_to_the_centre(
        self, width: int, height: int
    ) -> None:
        """A quarter-size centred box must land at (0.375, 0.625) in both axes.

        Asymmetric aspect ratios are where a padding error shows up as an offset
        that a square test image would hide entirely.
        """
        transform = compute_transform(
            source_width=width, source_height=height, target_width=640, target_height=640
        )
        source_x1, source_y1 = width * 0.375, height * 0.375
        source_x2, source_y2 = width * 0.625, height * 0.625

        box = invert_to_normalized(
            transform,
            source_x1 * transform.scale + transform.pad_x,
            source_y1 * transform.scale + transform.pad_y,
            source_x2 * transform.scale + transform.pad_x,
            source_y2 * transform.scale + transform.pad_y,
        )
        assert box.x1 == pytest.approx(0.375, abs=1e-9)
        assert box.y1 == pytest.approx(0.375, abs=1e-9)
        assert box.x2 == pytest.approx(0.625, abs=1e-9)
        assert box.y2 == pytest.approx(0.625, abs=1e-9)

    @pytest.mark.parametrize(("width", "height"), ASPECTS)
    def test_arbitrary_box_round_trips(self, width: int, height: int) -> None:
        transform = compute_transform(
            source_width=width, source_height=height, target_width=640, target_height=640
        )
        for nx1, ny1, nx2, ny2 in (
            (0.0, 0.0, 0.1, 0.1),
            (0.9, 0.9, 1.0, 1.0),
            (0.2, 0.05, 0.44, 0.93),
            (0.49, 0.49, 0.51, 0.51),
        ):
            box = invert_to_normalized(
                transform,
                nx1 * width * transform.scale + transform.pad_x,
                ny1 * height * transform.scale + transform.pad_y,
                nx2 * width * transform.scale + transform.pad_x,
                ny2 * height * transform.scale + transform.pad_y,
            )
            assert box.x1 == pytest.approx(nx1, abs=1e-9)
            assert box.y1 == pytest.approx(ny1, abs=1e-9)
            assert box.x2 == pytest.approx(nx2, abs=1e-9)
            assert box.y2 == pytest.approx(ny2, abs=1e-9)

    def test_padding_is_never_folded_into_the_object(self) -> None:
        """Clamping must happen after inverting, never before.

        A box sitting entirely in the letterbox bar is outside the image, and
        clamping in letterboxed space would drag it onto the image edge — a
        detection that never existed.
        """
        transform = compute_transform(
            source_width=1920, source_height=1080, target_width=640, target_height=640
        )
        assert transform.pad_y > 10
        box = invert_to_normalized(transform, 100.0, 0.0, 200.0, transform.pad_y / 2)
        assert box.y1 == pytest.approx(0.0)
        assert box.y2 == pytest.approx(0.0, abs=1e-5)

    def test_result_is_always_within_the_unit_square(self) -> None:
        """Obligation D1 holds even for wildly out-of-range model output."""
        transform = compute_transform(
            source_width=800, source_height=600, target_width=640, target_height=640
        )
        box = invert_to_normalized(transform, -5000.0, -5000.0, 9000.0, 9000.0)
        assert box.is_within_unit()

    def test_degenerate_box_is_widened_rather_than_rejected(self) -> None:
        """A zero-area box from the model becomes a minimal valid box.

        ``Box`` refuses degenerate geometry, so the adapter must not hand it one;
        widening keeps a real (if tiny) detection instead of crashing the batch.
        """
        transform = compute_transform(
            source_width=640, source_height=640, target_width=640, target_height=640
        )
        box = invert_to_normalized(transform, 100.0, 100.0, 100.0, 100.0)
        assert box.x2 > box.x1
        assert box.y2 > box.y1


class TestTruncation:
    def test_fully_visible_box_is_not_truncated(self) -> None:
        transform = compute_transform(
            source_width=640, source_height=640, target_width=640, target_height=640
        )
        assert truncation_of(transform, 100.0, 100.0, 200.0, 200.0) == pytest.approx(0.0)

    def test_half_off_frame_reports_half_truncated(self) -> None:
        """An object continuing past the edge understates its true extent."""
        transform = compute_transform(
            source_width=640, source_height=640, target_width=640, target_height=640
        )
        truncation = truncation_of(transform, -50.0, 100.0, 50.0, 200.0)
        assert truncation == pytest.approx(0.5, abs=0.01)

    def test_fully_off_frame_reports_total_truncation(self) -> None:
        transform = compute_transform(
            source_width=640, source_height=640, target_width=640, target_height=640
        )
        assert truncation_of(transform, -200.0, -200.0, -100.0, -100.0) == pytest.approx(1.0)

    def test_degenerate_box_reports_no_truncation(self) -> None:
        transform = compute_transform(
            source_width=640, source_height=640, target_width=640, target_height=640
        )
        assert truncation_of(transform, 10.0, 10.0, 10.0, 10.0) == 0.0


class TestNonSquareTargets:
    def test_rectangular_target_is_supported(self) -> None:
        """Dynamic resolution can select a non-square inference size."""
        transform = compute_transform(
            source_width=1920, source_height=1080, target_width=1280, target_height=736
        )
        box = invert_to_normalized(
            transform,
            transform.pad_x,
            transform.pad_y,
            transform.pad_x + 1920 * transform.scale,
            transform.pad_y + 1080 * transform.scale,
        )
        assert box.x1 == pytest.approx(0.0, abs=1e-9)
        assert box.x2 == pytest.approx(1.0, abs=1e-9)
        assert box.y2 == pytest.approx(1.0, abs=1e-9)

    def test_scale_is_independent_of_target_shape(self) -> None:
        """Fidelity tiers must not change where a box lands, only its precision."""
        for target in (320, 640, 1280):
            transform = compute_transform(
                source_width=1920, source_height=1080,
                target_width=target, target_height=target,
            )
            box = invert_to_normalized(
                transform,
                0.25 * 1920 * transform.scale + transform.pad_x,
                0.25 * 1080 * transform.scale + transform.pad_y,
                0.75 * 1920 * transform.scale + transform.pad_x,
                0.75 * 1080 * transform.scale + transform.pad_y,
            )
            assert box.x1 == pytest.approx(0.25, abs=1e-9)
            assert box.x2 == pytest.approx(0.75, abs=1e-9)
