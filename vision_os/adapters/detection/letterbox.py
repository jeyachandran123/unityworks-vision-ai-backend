"""Letterboxing and its exact inverse.

This module is small and matters more than its size suggests.

Getting the inverse subtly wrong is **the highest-frequency, lowest-visibility
adapter bug in computer vision**: boxes drift by a few percent, detection still
"works", tracking quietly degrades, and the cause is found months later — if
ever. It is the single divergence that the whole conformance-kit mechanism exists
to catch (06_PORTS section 5.2).

It is pure arithmetic with no framework dependency, so it is exhaustively
testable in CI without a GPU, a model, or an image library.
"""

from __future__ import annotations

from ...core.model.space import Box
from ..models.runtimes import LetterboxTransform


def compute_transform(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxTransform:
    """Fit a source image into a target box, preserving aspect ratio.

    The image is scaled by the *smaller* of the two ratios so it fits entirely,
    then centred with symmetric padding. Both facts must be inverted exactly.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"invalid source size {source_width}x{source_height}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"invalid target size {target_width}x{target_height}")

    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = source_width * scale
    scaled_height = source_height * scale
    return LetterboxTransform(
        scale=scale,
        pad_x=(target_width - scaled_width) / 2.0,
        pad_y=(target_height - scaled_height) / 2.0,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
    )


def invert_to_normalized(
    transform: LetterboxTransform,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Box:
    """Map a box from letterboxed pixel space back to normalized source space.

    Undoes padding first, then scale, then normalizes by the *source* dimensions
    — the order matters, and doing it the other way round produces boxes that are
    plausible and wrong.

    Obligation D1: the result is normalized against the rectified source image,
    origin top-left, ``x1 < x2`` and ``y1 < y2``.
    """
    source_x1 = (x1 - transform.pad_x) / transform.scale
    source_y1 = (y1 - transform.pad_y) / transform.scale
    source_x2 = (x2 - transform.pad_x) / transform.scale
    source_y2 = (y2 - transform.pad_y) / transform.scale

    normalized_x1 = source_x1 / transform.source_width
    normalized_y1 = source_y1 / transform.source_height
    normalized_x2 = source_x2 / transform.source_width
    normalized_y2 = source_y2 / transform.source_height

    # Clamp *after* inverting, never before: clamping in letterboxed space would
    # fold padding into the object and shift the box.
    left = max(0.0, min(1.0, min(normalized_x1, normalized_x2)))
    top = max(0.0, min(1.0, min(normalized_y1, normalized_y2)))
    right = max(0.0, min(1.0, max(normalized_x1, normalized_x2)))
    bottom = max(0.0, min(1.0, max(normalized_y1, normalized_y2)))

    if right <= left:
        right = min(1.0, left + 1e-6)
    if bottom <= top:
        bottom = min(1.0, top + 1e-6)
    return Box(left, top, right, bottom)


def truncation_of(
    transform: LetterboxTransform, x1: float, y1: float, x2: float, y2: float
) -> float:
    """How much of a box fell outside the source image before clamping.

    A positive value means the object continues past the frame edge, so anything
    measured from the visible part understates its true extent (obligation D8).
    """
    source_x1 = (x1 - transform.pad_x) / transform.scale
    source_y1 = (y1 - transform.pad_y) / transform.scale
    source_x2 = (x2 - transform.pad_x) / transform.scale
    source_y2 = (y2 - transform.pad_y) / transform.scale

    full_width = abs(source_x2 - source_x1)
    full_height = abs(source_y2 - source_y1)
    if full_width <= 0 or full_height <= 0:
        return 0.0

    visible_width = min(max(source_x1, source_x2), transform.source_width) - max(
        min(source_x1, source_x2), 0.0
    )
    visible_height = min(max(source_y1, source_y2), transform.source_height) - max(
        min(source_y1, source_y2), 0.0
    )
    visible = max(0.0, visible_width) * max(0.0, visible_height)
    return max(0.0, min(1.0, 1.0 - visible / (full_width * full_height)))
