"""P8 ``DetectorPort`` — the detection seam (06_PORTS_AND_ADAPTERS section 4).

The most churn-exposed port in the platform after the understander. Replacing
YOLO with RT-DETR, or either with an open-vocabulary detector, is an adapter plus
a conformance run — never a platform change.

An interface alone would not achieve that. Two detectors can implement this
protocol perfectly and still break the platform on swap: one returns boxes in
letterboxed coordinates and the other in original-image coordinates; one applies
NMS and the other does not; one returns an empty list for "nothing found" and the
other raises. Every one of those divergences is an **adapter** responsibility
here, and every one is checked by ``kit.detector`` before an adapter may load.

Adapters emit ``RawDetection`` — platform taxonomy, normalized coordinates, raw
score. Platform substrate (identity, provenance assembly, confidence calibration,
quality grading) is added by the Detection Engine, because an adapter must not be
able to forge it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.frame import FrameDimensions
from ..model.health import ComponentHealth
from ..model.ids import CalibrationId, ClassId, FrameRef
from ..model.provenance import InferenceTiming, ModelMeta
from ..model.space import Box
from ..model.taxonomy import GeometryKind


@dataclass(frozen=True, slots=True)
class FrameView:
    """Read-only pixels handed to a detector, with the geometry to interpret them.

    The adapter receives a *view*, never ownership: the Frame Buffer still owns
    the memory and the Detection Engine still holds the lease.
    """

    frame_ref: FrameRef
    dimensions: FrameDimensions
    pixels: memoryview

    def __post_init__(self) -> None:
        if not self.pixels.readonly:
            raise ValueError(
                "a FrameView must wrap a read-only view; no stage may mutate shared pixels"
            )


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    """What the platform asks for. Every field is a *hint* except the limits."""

    target_classes: tuple[ClassId, ...] = ()
    """Hint. An adapter may return more; the platform filters."""

    min_confidence: float | None = None
    """Adapter-side pre-filter, to avoid transporting detections nobody wants."""

    max_detections: int | None = None
    inference_width: int | None = None
    inference_height: int | None = None
    tier: str = "primary"
    deterministic: bool = False
    """When true the adapter must pin seeds and avoid any nondeterministic kernel
    it can avoid, or declare ``deterministic: false`` in capabilities."""


@dataclass(frozen=True, slots=True)
class RawDetection:
    """One detection as an adapter emits it.

    Obligations D1-D3 apply here: ``box`` is normalized against the rectified
    source image with letterboxing exactly inverted, ``class_id`` is a **platform
    taxonomy** class, and ``score`` is the model's own raw output.
    """

    class_id: ClassId
    box: Box
    score: float
    class_scores: tuple[tuple[ClassId, float], ...] = ()
    truncation: float | None = None
    geometry_kind: GeometryKind = GeometryKind.BOX
    native_label: str | None = None
    """Retained for diagnostics only. The Detection Engine never propagates it
    into a Detection — a native label must never escape the adapter (D2)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"detection score must be in [0,1], got {self.score}")
        if not self.class_id:
            raise ValueError("a RawDetection must carry a platform class_id (D2)")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """One frame's worth of adapter output.

    An empty ``detections`` tuple is a **valid, non-error** outcome: "nothing
    detected" and "detection failed" are different results and are never
    conflated (D5).
    """

    frame_ref: FrameRef
    detections: tuple[RawDetection, ...]
    model_meta: ModelMeta
    timing: InferenceTiming


@dataclass(frozen=True, slots=True)
class NmsDeclaration:
    """Whether the adapter already suppressed overlapping boxes.

    Declared because **a platform cannot correct for what it does not know**
    (D4). An adapter that silently applies NMS while declaring otherwise makes
    object counts wrong in a way no downstream stage can detect.
    """

    applied: bool
    iou_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.applied and self.iou_threshold is None:
            raise ValueError("an adapter applying NMS must declare its IoU threshold")


@dataclass(frozen=True, slots=True)
class BatchProfile:
    supported: bool = False
    max_size: int = 1
    optimal_size: int = 1

    def __post_init__(self) -> None:
        if self.max_size < 1 or self.optimal_size < 1:
            raise ValueError("batch sizes must be >= 1")
        if self.optimal_size > self.max_size:
            raise ValueError("optimal_size cannot exceed max_size")


@dataclass(frozen=True, slots=True)
class InputConstraints:
    min_width: int = 32
    min_height: int = 32
    max_width: int = 7680
    max_height: int = 4320
    colour_space: str = "bgr24"
    aspect_handling: str = "letterbox"


@dataclass(frozen=True, slots=True)
class DetectorCapabilities:
    """Declared honestly — adapter obligation A1.

    ``producible_classes`` is what makes a capability gap detectable: a consumer
    asking for a class this detector cannot produce learns so immediately rather
    than waiting forever for data that will never arrive (invariant V8).
    """

    producible_classes: tuple[ClassId, ...]
    geometry_kinds: tuple[GeometryKind, ...] = (GeometryKind.BOX,)
    input_constraints: InputConstraints = InputConstraints()
    batch: BatchProfile = BatchProfile()
    nms: NmsDeclaration = NmsDeclaration(applied=False)
    precision: str = "fp32"
    deterministic: bool = True
    calibration_profile: CalibrationId | None = None
    cost_class: float = 1.0
    device_id: str = "cpu"
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.producible_classes:
            raise ValueError(
                "a detector must declare at least one producible class; an "
                "undeclared capability is an undetectable gap (V8)"
            )

    def can_produce(self, class_id: ClassId) -> bool:
        return any(
            produced == class_id or produced.startswith(f"{class_id}.")
            for produced in self.producible_classes
        )


@runtime_checkable
class DetectorPort(Protocol):
    """P8 — find things in a frame.

    Implementations: YOLO, RT-DETR, DINO / Grounding-DINO, segmentation, pose,
    cloud APIs, and composite cascades. The platform sees one detector in every
    case, which is why a cascade needs no platform support.
    """

    def capabilities(self) -> DetectorCapabilities:
        """Declared capability. Must be stable across calls."""
        ...

    def detect(
        self, frames: Sequence[FrameView], request: DetectionRequest
    ) -> Sequence[DetectionResult]:
        """Detect on a batch.

        Results map **1:1 and in order** to ``frames`` (D6). The adapter is
        **stateless across calls** (D7): frame N's result must not depend on
        frame N-1.

        Raises:
            DetectionFailedError: and only that. An adapter never returns a
                fabricated result on failure — a plausible wrong answer is worse
                than an admitted failure, because nothing downstream can detect
                it (obligation A4).
        """
        ...

    def warm(self) -> None:
        """Run representative input once.

        Mandatory before a detector is considered ready: a cold model's first
        inference can be 10-100x slower, which reads as a performance regression
        rather than as an unwarmed model.
        """
        ...

    def health(self) -> ComponentHealth:
        """Self-report. The kernel decides what it means."""
        ...
