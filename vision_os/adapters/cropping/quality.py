"""The default quality estimator — P13.

> §M8: *"heuristic sharpness/scale today; learned quality predictors later."*

02_VOM section 10.8 fixes the grades; this adapter fixes only *how they are
measured*. Six grades, each measured independently, then folded into one
``overall`` verdict. Nothing here is a heuristic invented for the occasion —
every grade in the output is one the architecture names.

**Unmeasured is ``None``, never zero** (obligation Q2). Pre-extraction the
estimator sees a box and gets scale, truncation and crowding; blur and exposure
need pixels and stay ``None`` until a crop exists. A zeroed blur grade reads as
*perfectly sharp*, which is the opposite of what "not measured" means.

``core`` may not import numpy, and neither does this — the sharpness measure is
a strided Laplacian in pure Python over a subsampled grid, which is O(hundreds)
of operations per crop rather than O(pixels).
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.model.detection import ExposureLevel, QualityGrades, QualityLevel
from ...core.model.space import Box
from ...core.ports.cropping import QualityRequest

#: Object height in source pixels below which nothing defensible can be claimed.
#:
#: §M8 names scale as *"the strongest single predictor"*. The floor is
#: configurable because it depends on the model downstream, not on the platform.
DEFAULT_MIN_SCALE_PIXELS = 48.0

#: Scale at or above which the input is considered excellent on that axis.
DEFAULT_GOOD_SCALE_PIXELS = 160.0

#: Blur (0 sharp, 1 featureless) above which a claim is not defensible.
DEFAULT_MAX_BLUR = 0.85

#: Truncation above which the crop shows a fragment rather than an object.
DEFAULT_MAX_TRUNCATION = 0.5

#: Occlusion (approximated by crowding overlap) above which the crop is unusable.
DEFAULT_MAX_OCCLUSION = 0.7

#: Variance-of-Laplacian value treated as "fully sharp" when normalising.
#:
#: Empirical scaling constant, not a threshold: it converts an unbounded variance
#: into the [0,1] grade 02_VOM requires. The *decision* threshold is
#: ``DEFAULT_MAX_BLUR`` above.
#:
#: Calibrated against real kitchen CCTV (``datasets/kitchen-01``): native-
#: resolution patches of that footage score roughly 40–120 sharp, and the same
#: crops under a 6px Gaussian fall below 5. 60.0 puts the midpoint of the grade
#: inside that gap rather than above the whole range.
SHARPNESS_SATURATION = 60.0

#: Side of each patch sampled for blur, and how many patches across the crop.
#:
#: Blur must be measured between **adjacent** pixels. A stride sample spread over
#: the whole crop compares pixels far enough apart to be uncorrelated, which
#: reports a large variance for any textured scene no matter how smeared it is —
#: measuring how busy the kitchen is, not whether the lens was in focus.
#:
#: Several small patches rather than one large one because blur is local: a still
#: counter and a moving arm share a crop, and one centred window would grade
#: whichever happened to be in the middle.
BLUR_PATCH_SIDE = 12
BLUR_PATCH_GRID = 3

#: Mean luma below/above which exposure is called under/over.
UNDEREXPOSED_LUMA = 40.0
OVEREXPOSED_LUMA = 215.0

#: Cap on samples taken from a crop when measuring blur and exposure.
#:
#: Bounds the estimator's cost regardless of crop size: quality estimation must
#: never become the expensive step it exists to avoid paying for.
MAX_SAMPLES = 1024


class HeuristicQualityEstimator:
    """Scale, truncation, crowding, blur, exposure — measured, then folded.

    Pure and deterministic (obligation Q4): the same request always produces the
    same grades, which is what lets a gate rejection be reproduced from a replay
    six months later (V13).
    """

    __slots__ = (
        "_good_scale_pixels",
        "_max_blur",
        "_max_occlusion",
        "_max_truncation",
        "_min_scale_pixels",
    )

    def __init__(
        self,
        *,
        min_scale_pixels: float = DEFAULT_MIN_SCALE_PIXELS,
        good_scale_pixels: float = DEFAULT_GOOD_SCALE_PIXELS,
        max_blur: float = DEFAULT_MAX_BLUR,
        max_truncation: float = DEFAULT_MAX_TRUNCATION,
        max_occlusion: float = DEFAULT_MAX_OCCLUSION,
    ) -> None:
        if min_scale_pixels <= 0:
            raise ValueError("min_scale_pixels must be positive")
        if good_scale_pixels < min_scale_pixels:
            raise ValueError("good_scale_pixels must be >= min_scale_pixels")
        for name, value in (
            ("max_blur", max_blur),
            ("max_truncation", max_truncation),
            ("max_occlusion", max_occlusion),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        self._min_scale_pixels = min_scale_pixels
        self._good_scale_pixels = good_scale_pixels
        self._max_blur = max_blur
        self._max_truncation = max_truncation
        self._max_occlusion = max_occlusion

    @property
    def estimator_id(self) -> str:
        return "quality.heuristic"

    @property
    def min_scale_pixels(self) -> float:
        return self._min_scale_pixels

    def estimate(self, request: QualityRequest) -> QualityGrades:
        """Grade the input. Never raises on legal-but-extreme geometry (Q5)."""
        if request.source_width <= 0 or request.source_height <= 0:
            # A frame with no dimensions cannot be graded. Degenerate rather
            # than exceptional: the gate rejects it and the reason is recorded.
            return QualityGrades(overall=QualityLevel.INSUFFICIENT)

        scale_pixels = self._scale(request.box, request.source_height)
        truncation = self._truncation(request.box)
        crowding = self._crowding(request.box, request.neighbour_boxes)
        occlusion = crowding
        """Occlusion is approximated by overlap with neighbours. An honest
        approximation, and the only one available without a segmentation mask —
        recorded as the same number twice rather than as a second invented
        measurement."""

        blur: float | None = None
        exposure: ExposureLevel | None = None
        if request.pixels is not None and request.crop_width > 0 and request.crop_height > 0:
            blur = self._blur(request)
            exposure = self._exposure(request)

        overall = self._fold(
            scale_pixels=scale_pixels,
            truncation=truncation,
            occlusion=occlusion,
            blur=blur,
            exposure=exposure,
        )
        return QualityGrades(
            scale_pixels=scale_pixels,
            truncation=truncation,
            occlusion=occlusion,
            blur=blur,
            crowding=crowding,
            exposure=exposure,
            overall=overall,
        )

    # --- individual grades ----------------------------------------------------- #

    def _scale(self, box: Box, source_height: int) -> float:
        """Object height in source pixels.

        Height rather than area, because the models downstream care about how
        many pixels tall a person is; a wide, short box is not a substitute.
        """
        return max(0.0, box.height * source_height)

    def _truncation(self, box: Box) -> float:
        """Fraction of the box lying outside the frame.

        Measured against the *unclamped* box, which is why detection passes the
        raw geometry through: after clamping the evidence of truncation is gone.
        """
        full_area = box.area
        if full_area <= 0.0:
            return 1.0
        clamped = box.clamped_to_unit()
        visible_width = max(0.0, clamped.x2 - clamped.x1)
        visible_height = max(0.0, clamped.y2 - clamped.y1)
        visible = visible_width * visible_height
        return min(1.0, max(0.0, 1.0 - visible / full_area))

    def _crowding(self, box: Box, neighbours: Sequence[Box]) -> float:
        """Fraction of the box overlapped by the union bound of its neighbours.

        Approximated by the largest single overlap rather than a true union, so
        the measure stays O(n) and never exceeds 1. Overstating crowding would
        reject usable crops; the largest overlap understates it, which fails in
        the direction of still trying.
        """
        area = box.area
        if area <= 0.0 or not neighbours:
            return 0.0
        worst = 0.0
        for other in neighbours:
            overlap = self._intersection_area(box, other)
            if overlap > worst:
                worst = overlap
        return min(1.0, worst / area)

    @staticmethod
    def _intersection_area(a: Box, b: Box) -> float:
        width = min(a.x2, b.x2) - max(a.x1, b.x1)
        height = min(a.y2, b.y2) - max(a.y1, b.y1)
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return width * height

    def _blur(self, request: QualityRequest) -> float:
        """Normalized variance-of-Laplacian: 0 is sharp, 1 is featureless.

        A 4-neighbour Laplacian over **adjacent** pixels, taken from a few small
        patches spread across the crop. Adjacency is the whole point: the
        Laplacian measures how fast intensity changes from one pixel to the next,
        and pixels sampled far apart change fast in any textured scene, blurred
        or not.

        The patches are read at native resolution and the sharpest is kept. A
        crop where anything is in focus was focused; averaging would let a large
        flat wall outvote the subject and grade a sharp chef as blurred.

        Cost stays bounded by ``BLUR_PATCH_SIDE`` and ``BLUR_PATCH_GRID`` rather
        than by crop size — quality estimation must never become the expensive
        step it exists to avoid paying for.
        """
        best = 0.0
        found = False
        for patch in self._blur_patches(request):
            side = BLUR_PATCH_SIDE
            responses: list[float] = []
            for row in range(1, side - 1):
                for col in range(1, side - 1):
                    centre = patch[row * side + col]
                    responses.append(
                        patch[(row - 1) * side + col]
                        + patch[(row + 1) * side + col]
                        + patch[row * side + col - 1]
                        + patch[row * side + col + 1]
                        - 4.0 * centre
                    )
            if len(responses) < 2:
                continue
            mean = sum(responses) / len(responses)
            variance = sum((r - mean) ** 2 for r in responses) / len(responses)
            best = max(best, variance)
            found = True

        if not found:
            return 0.0
        sharpness = min(1.0, best / SHARPNESS_SATURATION)
        return round(1.0 - sharpness, 6)

    def _blur_patches(self, request: QualityRequest) -> list[list[float]]:
        """Contiguous native-resolution patches, evenly spaced over the crop.

        Deterministic placement, never random: the same crop must grade the same
        way on replay six months later (obligation Q4, invariant V13).
        """
        pixels = request.pixels
        width, height = request.crop_width, request.crop_height
        side = BLUR_PATCH_SIDE
        if pixels is None or width < side or height < side:
            return []
        total = width * height
        channels = max(1, len(pixels) // total)
        if channels * total > len(pixels):
            return []
        raw = pixels.tobytes() if pixels.format != "B" else pixels
        stride = width * channels

        # Evenly spaced origins, inset so every patch lies wholly inside.
        steps = max(1, BLUR_PATCH_GRID)
        xs = self._origins(width, side, steps)
        ys = self._origins(height, side, steps)

        patches: list[list[float]] = []
        for top in ys:
            for left in xs:
                patch: list[float] = []
                for row in range(side):
                    base = (top + row) * stride + left * channels
                    for col in range(side):
                        offset = base + col * channels
                        if channels >= 3:
                            patch.append(
                                0.114 * raw[offset]
                                + 0.587 * raw[offset + 1]
                                + 0.299 * raw[offset + 2]
                            )
                        else:
                            patch.append(float(raw[offset]))
                patches.append(patch)
        return patches

    @staticmethod
    def _origins(extent: int, side: int, steps: int) -> list[int]:
        span = extent - side
        if span <= 0:
            return [0]
        if steps == 1:
            return [span // 2]
        return [round(span * i / (steps - 1)) for i in range(steps)]

    def _exposure(self, request: QualityRequest) -> ExposureLevel:
        luma = self._sample_luma(request)
        if not luma:
            return ExposureLevel.OK
        mean = sum(luma) / len(luma)
        if mean < UNDEREXPOSED_LUMA:
            return ExposureLevel.UNDER
        if mean > OVEREXPOSED_LUMA:
            return ExposureLevel.OVER
        return ExposureLevel.OK

    # --- sampling -------------------------------------------------------------- #

    def _sample_luma(self, request: QualityRequest) -> list[float]:
        """Subsample the crop into a bounded list of luma values.

        Deterministic stride sampling, not random: a random sample would make
        the same crop grade differently on replay, breaking V13.
        """
        pixels = request.pixels
        if pixels is None:
            return []
        total_pixels = request.crop_width * request.crop_height
        if total_pixels <= 0:
            return []
        channels = max(1, len(pixels) // total_pixels)
        if channels * total_pixels > len(pixels):
            return []

        step = max(1, total_pixels // MAX_SAMPLES)
        samples: list[float] = []
        raw = pixels.tobytes() if pixels.format != "B" else pixels
        for index in range(0, total_pixels, step):
            base = index * channels
            if channels >= 3:
                # BGR24 luma, the platform's declared crop colour space.
                blue = raw[base]
                green = raw[base + 1]
                red = raw[base + 2]
                samples.append(0.114 * blue + 0.587 * green + 0.299 * red)
            else:
                samples.append(float(raw[base]))
        return samples

    @staticmethod
    def _grid_stride(sample_count: int) -> int:
        """The largest square grid fitting the samples, so neighbours are real.

        Sampling with a stride makes the samples a coarse grid of the crop;
        treating them as a square of side ``isqrt(n)`` keeps the Laplacian's
        neighbours spatially adjacent rather than arbitrary.
        """
        stride = 1
        while (stride + 1) * (stride + 1) <= sample_count:
            stride += 1
        return stride

    # --- folding --------------------------------------------------------------- #

    def _fold(
        self,
        *,
        scale_pixels: float,
        truncation: float,
        occlusion: float,
        blur: float | None,
        exposure: ExposureLevel | None,
    ) -> QualityLevel:
        """Fold the grades into one verdict.

        **Worst-axis wins.** A crop that is sharp, well-exposed and 12 pixels
        tall is not a good crop; averaging the axes would say otherwise and would
        send an unanswerable question to an expensive model.
        """
        if scale_pixels < self._min_scale_pixels:
            return QualityLevel.INSUFFICIENT
        if truncation > self._max_truncation:
            return QualityLevel.INSUFFICIENT
        if occlusion > self._max_occlusion:
            return QualityLevel.INSUFFICIENT
        if blur is not None and blur > self._max_blur:
            return QualityLevel.INSUFFICIENT

        marginal = (
            scale_pixels < self._good_scale_pixels
            or truncation > self._max_truncation / 2.0
            or occlusion > self._max_occlusion / 2.0
            or (blur is not None and blur > self._max_blur / 2.0)
            or exposure in (ExposureLevel.UNDER, ExposureLevel.OVER)
        )
        if marginal:
            return QualityLevel.MARGINAL

        if blur is None or exposure is None:
            # Ungraded on two axes. "Good" is the honest ceiling for a verdict
            # that has not seen pixels — claiming excellence from geometry alone
            # would overstate what was measured.
            return QualityLevel.GOOD
        if blur < self._max_blur / 4.0 and scale_pixels >= self._good_scale_pixels:
            return QualityLevel.EXCELLENT
        return QualityLevel.GOOD


class AlwaysUsableEstimator:
    """Grades everything ``GOOD``. For tests that are not about quality.

    Named so its use is obvious in a test that would otherwise silently depend
    on the heuristic estimator's thresholds.
    """

    __slots__ = ()

    @property
    def estimator_id(self) -> str:
        return "quality.always_usable"

    def estimate(self, request: QualityRequest) -> QualityGrades:
        scale = max(0.0, request.box.height * max(1, request.source_height))
        return QualityGrades(
            scale_pixels=scale,
            truncation=0.0,
            occlusion=0.0,
            blur=0.0,
            crowding=0.0,
            exposure=ExposureLevel.OK,
            overall=QualityLevel.GOOD,
        )
