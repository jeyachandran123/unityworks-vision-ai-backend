"""Crop strategies and the reference extractor — P14.

> §M8 STANDARDIZATION: *"No YOLO crop. No CLIP crop. No Florence crop. No Qwen
> crop. No InternVL crop."*

There is one crop format. A strategy decides **geometry** — how much of the
frame to read and at what output size — and the extractor performs the read. No
strategy here is named after a model, and none encodes a model's preprocessing:
mean subtraction, channel order flips, and tensor layout belong to the model
adapter in M9, downstream of the canonical crop.

Two strategies ship, both named in §M8: ``crop.tight`` and ``crop.padded``.
Multi-scale, part-focused and temporal-stack strategies plug in here without M8
changing, which is the point of the port.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.errors import CropExtractionError
from ...core.model.crop import CropTransform
from ...core.model.ids import AttributeKey, ClassId
from ...core.model.space import Box
from ...core.ports.cropping import CropPlan

#: The platform's canonical crop size. One format, for every model.
#:
#: 224x224 is the input size the majority of vision backbones accept natively.
#: A model wanting something else resizes *down* from this in its own adapter —
#: which is lossy but honest, and far better than every model cropping its own
#: pixels from the frame (§M8 responsibility 4).
DEFAULT_OUTPUT_SIZE = (224, 224)

#: Context ratio added around the box by ``crop.padded``.
#:
#: §M8: context matters for attributes that depend on surroundings. 15% is the
#: platform default; the value is configuration, not a constant of nature.
DEFAULT_PADDING = 0.15


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


class TightCropStrategy:
    """The object's box, unpadded. ``crop.tight``.

    Maximises the object's pixel share of the output, which is what a
    fine-grained attribute wants. It also throws away every contextual cue,
    which is why it is not the default.
    """

    __slots__ = ("_output_size", "_preserve_aspect")

    def __init__(
        self,
        *,
        output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
        preserve_aspect: bool = True,
    ) -> None:
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size must be positive")
        self._output_size = output_size
        self._preserve_aspect = preserve_aspect

    @property
    def strategy_id(self) -> str:
        return "crop.tight"

    def plan(
        self,
        *,
        box: Box,
        class_id: ClassId,
        source_width: int,
        source_height: int,
        attributes: Sequence[AttributeKey] = (),
    ) -> CropPlan:
        width, height = self._output_size
        return CropPlan(
            source_box=box,
            padded_box=_clamped(box.x1, box.y1, box.x2, box.y2),
            padding_applied=0.0,
            output_width=width,
            output_height=height,
            preserve_aspect=self._preserve_aspect,
        )


class PaddedCropStrategy:
    """The box plus a context ratio. ``crop.padded``. The platform default.

    Padding is expressed as a fraction of the box's own dimensions, so a distant
    object gets proportionally the same context as a near one — a fixed pixel
    pad would swamp a small object and barely touch a large one.

    Clamping to the frame is what obligation C1 requires, and it is also why
    ``padding_applied`` records the *requested* ratio while the transform records
    the rectangle actually read: an object at the frame edge gets less context
    than asked for, and the crop says so.
    """

    __slots__ = ("_output_size", "_padding", "_preserve_aspect")

    def __init__(
        self,
        *,
        padding: float = DEFAULT_PADDING,
        output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
        preserve_aspect: bool = True,
    ) -> None:
        if not 0.0 <= padding <= 4.0:
            raise ValueError("padding must be in [0,4]")
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size must be positive")
        self._padding = padding
        self._output_size = output_size
        self._preserve_aspect = preserve_aspect

    @property
    def strategy_id(self) -> str:
        return "crop.padded"

    @property
    def padding(self) -> float:
        return self._padding

    def plan(
        self,
        *,
        box: Box,
        class_id: ClassId,
        source_width: int,
        source_height: int,
        attributes: Sequence[AttributeKey] = (),
    ) -> CropPlan:
        pad_x = box.width * self._padding
        pad_y = box.height * self._padding
        width, height = self._output_size
        return CropPlan(
            source_box=box,
            padded_box=_clamped(
                box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y
            ),
            padding_applied=self._padding,
            output_width=width,
            output_height=height,
            preserve_aspect=self._preserve_aspect,
        )


class PartFocusedCropStrategy:
    """The part of the subject a question is actually about. ``crop.part_focused``.

    §M8 names this among the strategies that plug in without the platform
    changing — *"part-focused (head region for headwear, torso for hi-vis)"* —
    and P14's obligation C5 permits it explicitly: *"a strategy may use the
    object's class to choose geometry; it may never use what the class means to a
    business."* This uses the **demanded attributes** to choose geometry and
    never learns what any of them mean.

    ### Why a whole-person crop is not adequate evidence

    A standing person is roughly 1:3, and the canonical crop is square. Letterbox
    one into the other and about 55% of the image is black bar; the head lands in
    perhaps 40x40 pixels of a 224x224 canvas. Reviewing real kitchen footage,
    that was enough to lose a plainly visible blue hairnet: the model answered
    ``none`` for a head that was covered, and the rule turned it into a violation.

    Narrowing to the region a question is about spends the same 224x224 on the
    part that answers it. The head band of the same person fills the frame.

    ### The regions are data

    ``regions`` maps an attribute key to a vertical band of the subject box, and
    every value comes from the policy document that declared the attribute. This
    class contains no attribute name, no body part, and no fraction of its own —
    an attribute with no declared region falls back to the whole box, which is
    the previous behaviour exactly.

    When several attributes are demanded together the bands are **unioned**, so
    one crop still answers every question asked of it. That is what keeps this a
    single crop per object per frame: M8 admits one request per decision, and
    inventing a second would be a parallel pipeline.
    """

    __slots__ = (
        "_min_aspect",
        "_output_size",
        "_output_sizes",
        "_padding",
        "_preserve_aspect",
        "_regions",
    )

    def __init__(
        self,
        *,
        regions: dict[str, tuple[float, float]] | None = None,
        padding: float = DEFAULT_PADDING,
        output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
        output_sizes: dict[str, tuple[int, int]] | None = None,
        preserve_aspect: bool = True,
        min_aspect: float = 0.75,
    ) -> None:
        """
        Args:
            regions: ``{attribute_key: (top, height)}`` as fractions of the
                subject box, from the policy document.
            output_sizes: ``{attribute_key: (width, height)}`` for attributes
                that need a canonical crop larger than the deployment default.

                **How much resolution a claim needs depends on the claim.**
                Narrowing to the head band recovers framing but not detail: the
                band is still resampled to the canonical size, so a hairnet in a
                224px crop survives as roughly 30px of fabric. Measured on
                annotated kitchen footage, raising only the head band to 448
                moved head accuracy from 23.3% to 74.4% with the same model,
                prompt, region and detector.

                Absent an entry the deployment default applies, so a deployment
                that declares none behaves exactly as before. Declared per
                attribute rather than raised globally because vision tokens
                scale with *area*: paying 4x on every crop to fix one question
                would be a cost with no measured return.
            min_aspect: The narrowest width/height the planned region may have
                before it is widened. A tall band letterboxes into a square crop
                exactly as a whole person does, so narrowing vertically without
                widening horizontally would trade one waste for another.
        """
        if not 0.0 <= padding <= 4.0:
            raise ValueError("padding must be in [0,4]")
        if min_aspect <= 0.0:
            raise ValueError("min_aspect must be positive")
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size must be positive")
        for key, size in (output_sizes or {}).items():
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"output_size for '{key}' is {size}; both must be positive")
        for key, span in (regions or {}).items():
            top, extent = span
            if not 0.0 <= top <= 1.0 or not 0.0 < extent <= 1.0 or top + extent > 1.0001:
                raise ValueError(
                    f"region for '{key}' is {span}; a band must be (top, height) "
                    f"fractions of the subject box lying inside it"
                )
        self._regions = dict(regions or {})
        self._padding = padding
        self._output_size = output_size
        self._output_sizes = dict(output_sizes or {})
        self._preserve_aspect = preserve_aspect
        self._min_aspect = min_aspect

    @property
    def strategy_id(self) -> str:
        return "crop.part_focused"

    @property
    def regions(self) -> dict[str, tuple[float, float]]:
        return dict(self._regions)

    def plan(
        self,
        *,
        box: Box,
        class_id: ClassId,
        source_width: int,
        source_height: int,
        attributes: Sequence[AttributeKey] = (),
    ) -> CropPlan:
        top, bottom = self._span(attributes)

        band_y1 = box.y1 + box.height * top
        band_y2 = box.y1 + box.height * bottom
        band_height = max(band_y2 - band_y1, _EPSILON)

        # Widen a narrow band rather than letterboxing it. The band is centred on
        # the subject, so the extra width is context on both sides — which is
        # what a reviewer looking at a head wants anyway.
        band_width = box.width
        wanted = band_height * self._min_aspect
        if band_width < wanted:
            centre = (box.x1 + box.x2) / 2.0
            band_x1, band_x2 = centre - wanted / 2.0, centre + wanted / 2.0
        else:
            band_x1, band_x2 = box.x1, box.x2

        pad_x = (band_x2 - band_x1) * self._padding
        pad_y = band_height * self._padding
        width, height = self._output_for(attributes)
        return CropPlan(
            source_box=box,
            padded_box=_clamped(
                band_x1 - pad_x, band_y1 - pad_y, band_x2 + pad_x, band_y2 + pad_y
            ),
            padding_applied=self._padding,
            output_width=width,
            output_height=height,
            preserve_aspect=self._preserve_aspect,
        )

    def _output_for(self, attributes: Sequence[AttributeKey]) -> tuple[int, int]:
        """The canonical size for a crop serving these attributes.

        The **largest** declared size wins when one crop answers several
        questions, mirroring the gate's strictest-floor rule and for the same
        reason: a crop rendered at the smaller size would answer the demanding
        question from detail its own policy said was not enough, and the loss
        would be invisible in the result.

        Compared by area, because that is what the cost and the detail both
        scale with.
        """
        declared = [
            self._output_sizes[str(key)]
            for key in attributes
            if str(key) in self._output_sizes
        ]
        if not declared:
            return self._output_size
        return max(declared, key=lambda size: size[0] * size[1])

    def _span(self, attributes: Sequence[AttributeKey]) -> tuple[float, float]:
        """The union of the bands the demanded attributes declared.

        Union, not intersection: one crop has to answer every question asked of
        it, and a band that satisfied only the first attribute would leave the
        rest being answered from pixels that do not contain them — which is the
        failure this class exists to remove.

        No declared region for any demanded attribute means the whole box, which
        is exactly what ``crop.padded`` would have produced.
        """
        spans = [
            self._regions[str(key)] for key in attributes if str(key) in self._regions
        ]
        if not spans:
            return 0.0, 1.0
        top = min(span[0] for span in spans)
        bottom = max(span[0] + span[1] for span in spans)
        return top, min(bottom, 1.0)


#: Minimum box extent, in normalized units. Roughly a tenth of a pixel at 1080p.
_EPSILON = 1e-4


def _clamped(x1: float, y1: float, x2: float, y2: float) -> Box:
    """Clamp to the frame and guarantee a non-degenerate result.

    An object entirely outside the frame clamps to zero area, and ``Box`` refuses
    to exist with none — so clamping through ``Box.clamped_to_unit`` would raise
    on exactly the input a real camera produces when something walks out of shot.

    Widening by one part in ten thousand keeps the plan constructible so the
    *gate* rejects it with ``DEGENERATE_GEOMETRY`` — a recorded, countable
    outcome rather than an exception thrown from geometry code that has no idea
    what to do about it.
    """
    x1, y1 = _clamp_unit(x1), _clamp_unit(y1)
    x2, y2 = _clamp_unit(x2), _clamp_unit(y2)
    if x2 - x1 < _EPSILON:
        x1 = max(0.0, min(x1, 1.0 - _EPSILON))
        x2 = x1 + _EPSILON
    if y2 - y1 < _EPSILON:
        y1 = max(0.0, min(y1, 1.0 - _EPSILON))
        y2 = y1 + _EPSILON
    return Box(x1, y1, x2, y2)


class ReferenceCropExtractor:
    """Nearest-neighbour extraction in pure Python. The reference implementation.

    Correct, dependency-free and slow — which is the right trade for a reference:
    it defines what a crop *is* so a fast implementation can be checked against
    it. A production node replaces this with an OpenCV or GPU extractor and
    nothing else changes, because extraction sits behind its own port.

    **The transform is measured, not assumed** (02_VOM section 10.7). Every
    number in the returned ``CropTransform`` describes what this function
    actually did.
    """

    __slots__ = ("_interpolation",)

    def __init__(self, *, interpolation: str = "nearest") -> None:
        self._interpolation = interpolation

    @property
    def extractor_id(self) -> str:
        return f"crop.extractor.reference.{self._interpolation}"

    def extract(
        self,
        pixels: memoryview,
        *,
        plan: CropPlan,
        source_width: int,
        source_height: int,
        channels: int,
        colour_space: str,
    ) -> tuple[bytes, CropTransform]:
        if source_width <= 0 or source_height <= 0:
            raise CropExtractionError(
                f"cannot extract from a {source_width}x{source_height} frame"
            )
        expected = source_width * source_height * channels
        if len(pixels) < expected:
            raise CropExtractionError(
                f"frame buffer holds {len(pixels)} bytes but "
                f"{source_width}x{source_height}x{channels} needs {expected}; "
                f"extracting from a short buffer would silently fabricate evidence"
            )

        crop_x, crop_y, crop_w, crop_h = self._source_rect(
            plan.padded_box, source_width, source_height
        )
        if crop_w <= 0 or crop_h <= 0:
            raise CropExtractionError(
                "the requested region has no area in source pixels; a crop with "
                "no area is not evidence"
            )

        out_w, out_h = plan.output_width, plan.output_height
        if plan.preserve_aspect:
            drawn_w, drawn_h, pad_left, pad_top = self._letterbox(crop_w, crop_h, out_w, out_h)
        else:
            drawn_w, drawn_h, pad_left, pad_top = out_w, out_h, 0, 0

        buffer = self._resample(
            pixels,
            source_width=source_width,
            channels=channels,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            out_w=out_w,
            out_h=out_h,
            drawn_w=drawn_w,
            drawn_h=drawn_h,
            pad_left=pad_left,
            pad_top=pad_top,
        )

        transform = CropTransform(
            source_width=source_width,
            source_height=source_height,
            output_width=out_w,
            output_height=out_h,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_w,
            crop_height=crop_h,
            scale=drawn_w / crop_w,
            drawn_width=drawn_w,
            drawn_height=drawn_h,
            pad_left=pad_left,
            pad_top=pad_top,
            interpolation=self._interpolation,
            colour_space=colour_space,
            letterboxed=pad_left > 0 or pad_top > 0,
        )
        return bytes(buffer), transform

    # --- geometry -------------------------------------------------------------- #

    @staticmethod
    def _source_rect(
        box: Box, source_width: int, source_height: int
    ) -> tuple[int, int, int, int]:
        """Normalized box to an integer source-pixel rectangle.

        Floor the origin and ceil the extent, so the rectangle covers the whole
        requested region rather than shaving a pixel off each edge — rounding
        inward loses object boundary at exactly the small scales where every
        pixel counts.
        """
        x1 = int(_clamp_unit(box.x1) * source_width)
        y1 = int(_clamp_unit(box.y1) * source_height)
        x2 = int(-(-_clamp_unit(box.x2) * source_width // 1))
        y2 = int(-(-_clamp_unit(box.y2) * source_height // 1))
        x1 = max(0, min(x1, source_width - 1))
        y1 = max(0, min(y1, source_height - 1))
        x2 = max(x1 + 1, min(x2, source_width))
        y2 = max(y1 + 1, min(y2, source_height))
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _letterbox(
        crop_w: int, crop_h: int, out_w: int, out_h: int
    ) -> tuple[int, int, int, int]:
        """Fit the crop inside the output without distorting it.

        A squashed crop produces attributes about a distorted object, and the
        distortion is invisible in the output — which is exactly the failure the
        transform record exists to make detectable.
        """
        scale = min(out_w / crop_w, out_h / crop_h)
        drawn_w = max(1, int(crop_w * scale))
        drawn_h = max(1, int(crop_h * scale))
        return drawn_w, drawn_h, (out_w - drawn_w) // 2, (out_h - drawn_h) // 2

    @staticmethod
    def _resample(
        pixels: memoryview,
        *,
        source_width: int,
        channels: int,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
        out_w: int,
        out_h: int,
        drawn_w: int,
        drawn_h: int,
        pad_left: int,
        pad_top: int,
    ) -> bytearray:
        """Nearest-neighbour into a zero-filled (black-padded) output buffer."""
        buffer = bytearray(out_w * out_h * channels)
        row_stride = source_width * channels
        for out_y in range(drawn_h):
            src_y = crop_y + (out_y * crop_h) // drawn_h
            src_row = src_y * row_stride
            dst_row = ((out_y + pad_top) * out_w + pad_left) * channels
            for out_x in range(drawn_w):
                src_x = crop_x + (out_x * crop_w) // drawn_w
                src = src_row + src_x * channels
                dst = dst_row + out_x * channels
                buffer[dst : dst + channels] = pixels[src : src + channels]
        return buffer
