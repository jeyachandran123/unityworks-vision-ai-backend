"""The Frame object (02_VISION_OBJECT_MODEL §10.2).

``privacy_state`` travels with the frame so any component can assert that masking
happened. A ``MASK_FAILED`` frame is dropped rather than processed, because a
masking failure that proceeds is a compliance incident (12_SECURITY §2.1).

``decode_quality`` matters because anything inferred from a corrupted delta frame
deserves lower trust.

Pixels are referenced through the ``PixelBuffer`` protocol rather than stored
inline. ``core`` may not depend on numpy or any imaging library; an adapter wraps
its native buffer (a numpy array, a CUDA allocation, a decoder surface) behind
this protocol, and the platform only ever sees a read-only view.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .ids import FrameRef
from .timebase import FrameTime


class PrivacyState(enum.Enum):
    MASKED = "masked"
    """Policy applied successfully."""

    UNMASKED_PERMITTED = "unmasked_permitted"
    """No masking policy is configured for this camera."""

    MASK_FAILED = "mask_failed"
    """Masking was required and failed. Such frames are never emitted."""

    @property
    def emittable(self) -> bool:
        return self is not PrivacyState.MASK_FAILED


class DecodeQuality(enum.Enum):
    KEYFRAME = "keyframe"
    DELTA = "delta"
    RECOVERED_FROM_ERROR = "recovered_from_error"


@runtime_checkable
class PixelBuffer(Protocol):
    """An opaque, read-only pixel store owned by the Frame Buffer.

    Deliberately minimal: the platform never interprets pixels, it only lends
    them. Everything that *does* interpret pixels is an adapter, and adapters
    unwrap this into their native representation.
    """

    @property
    def nbytes(self) -> int:
        """Size of the underlying allocation in bytes."""
        ...

    def readonly_view(self) -> memoryview:
        """A read-only view. No stage may mutate shared pixels (01_LAYERED §4.3)."""
        ...


@dataclass(frozen=True, slots=True)
class FrameDimensions:
    width: int
    height: int
    colour_space: str = "bgr24"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid dimensions {self.width}x{self.height}")


@dataclass(frozen=True, slots=True)
class SourceMeta:
    """Stream telemetry carried alongside the frame for quality reasoning."""

    codec: str = "unknown"
    bitrate_bps: int = 0
    packet_loss: float = 0.0
    jitter_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """Frame-level quality signals available at acquisition time.

    Deliberately *frame*-scoped. Object-level quality grades (occlusion,
    truncation, scale, crowding) belong to the crop stage and are not part of
    Flow 1.
    """

    blur: float = 0.0
    """Normalized sharpness deficit in [0,1]. 0 = sharp."""

    exposure: str = "ok"
    """One of ``under`` | ``ok`` | ``over``."""

    decode_quality: DecodeQuality = DecodeQuality.KEYFRAME

    @property
    def usable(self) -> bool:
        return self.blur < 0.9 and self.decode_quality is not DecodeQuality.RECOVERED_FROM_ERROR


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded instant, honestly identified and honestly timestamped.

    Immutable once published: readers need no lock at all, which is what makes
    multi-consumer reads free (01_LAYERED §4.3).
    """

    frame_ref: FrameRef
    time: FrameTime
    dimensions: FrameDimensions
    pixels: PixelBuffer
    privacy_state: PrivacyState
    quality: FrameQuality = FrameQuality()
    source_meta: SourceMeta = SourceMeta()

    def __post_init__(self) -> None:
        if not self.privacy_state.emittable:
            raise ValueError(
                f"refusing to construct Frame {self.frame_ref} with privacy_state="
                f"{self.privacy_state.value}; masking failures must drop the frame"
            )
