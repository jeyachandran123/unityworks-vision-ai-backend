"""P8 — the YOLO detector adapter.

**The platform does not know YOLO exists.** Nothing above this file imports it,
names it, or branches on it; the Detection Engine holds a ``DetectorPort`` and
the composition root decides what fills it. Replacing this with RT-DETR is a new
adapter plus a configuration change.

What this adapter absorbs, so the platform never sees it:

* **Letterboxing and its exact inverse** (obligation D1) — delegated to
  ``letterbox``, which is pure arithmetic and exhaustively tested without a GPU.
* **The COCO label space** (D2) — mapped to platform taxonomy here; a native
  label never leaves this file.
* **YOLO's built-in NMS** (D4) — declared honestly, because a platform cannot
  correct for suppression it does not know about.
* **Framework choice** — inference goes through ``DetectorSession``, so the same
  adapter runs on ultralytics locally and ONNX at the edge.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.errors import DetectionFailedError
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import ClassId, ModelId, ModuleId
from ...core.model.provenance import InferenceTiming, ModelMeta
from ...core.model.taxonomy import GeometryKind, TaxonomyMapping, UnmappedPolicy
from ...core.ports.clock import Clock
from ...core.ports.detection import (
    BatchProfile,
    DetectionRequest,
    DetectionResult,
    DetectorCapabilities,
    FrameView,
    InputConstraints,
    NmsDeclaration,
    RawDetection,
)
from ..models.runtimes import DetectorSession, LetterboxedImage, RawBox
from .letterbox import compute_transform, invert_to_normalized, truncation_of

DEFAULT_INPUT_SIZE = 640
_NANOS_PER_MS = 1_000_000


class YoloDetector:
    """A production YOLO detector behind ``DetectorPort``.

    Stateless across calls (obligation D7): every call derives everything from
    its arguments, so frame N's result can never depend on frame N-1 and a swap
    cannot carry hidden state.

    The clock is injected rather than read, because an adapter that reads the
    wall clock cannot be replayed and quietly costs the platform invariant V13.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        session: DetectorSession,
        mapping: TaxonomyMapping,
        model_id: ModelId,
        model_version: str,
        artifact_hash: str,
        device_id: str = "cpu",
        precision: str = "fp32",
        input_size: int = DEFAULT_INPUT_SIZE,
        nms_iou_threshold: float = 0.45,
        max_batch_size: int = 16,
        deterministic: bool = True,
        adapter_version: str = "1.0.0",
    ) -> None:
        self._clock = clock
        self._session = session
        self._mapping = mapping
        self._model_id = model_id
        self._model_version = model_version
        self._artifact_hash = artifact_hash
        self._device_id = device_id
        self._precision = precision
        self._input_size = input_size
        self._nms_iou = nms_iou_threshold
        self._max_batch = max_batch_size
        self._deterministic = deterministic
        self._adapter_version = adapter_version
        self._warmed = False
        self._failures = 0
        self._native_names = tuple(session.class_names())

    @property
    def adapter_version(self) -> str:
        return self._adapter_version

    # --- capability ------------------------------------------------------------ #

    def capabilities(self) -> DetectorCapabilities:
        """Declared honestly (adapter obligation A1).

        ``producible_classes`` comes from the *mapping*, not from the model's
        native label list: what this detector can deliver to the platform is what
        it can translate, and claiming more would create a capability gap nobody
        could detect (invariant V8).
        """
        return DetectorCapabilities(
            producible_classes=self._mapping.producible_classes,
            geometry_kinds=(GeometryKind.BOX,),
            input_constraints=InputConstraints(
                min_width=32,
                min_height=32,
                max_width=self._input_size,
                max_height=self._input_size,
                colour_space="bgr24",
                aspect_handling="letterbox",
            ),
            batch=BatchProfile(
                supported=True,
                max_size=self._max_batch,
                optimal_size=min(8, self._max_batch),
            ),
            # YOLO applies NMS internally. Declaring it means the platform does
            # not double-suppress, which would silently halve object counts.
            nms=NmsDeclaration(applied=True, iou_threshold=self._nms_iou),
            precision=self._precision,
            deterministic=self._deterministic,
            cost_class=1.0,
            device_id=self._device_id,
            extra={"native_label_space": self._mapping.native_label_space or "coco"},
        )

    # --- inference -------------------------------------------------------------- #

    def detect(
        self, frames: Sequence[FrameView], request: DetectionRequest
    ) -> Sequence[DetectionResult]:
        """Detect on a batch.

        Raises:
            DetectionFailedError: and only that. Never a fabricated result — a
                plausible wrong answer is worse than an admitted failure, because
                nothing downstream can detect it (obligation A4).
        """
        if not frames:
            return ()

        target_width = request.inference_width or self._input_size
        target_height = request.inference_height or target_width

        preprocess_started = self._clock.monotonic().ns
        prepared = [
            LetterboxedImage(
                pixels=frame.pixels,
                width=frame.dimensions.width,
                height=frame.dimensions.height,
                transform=compute_transform(
                    source_width=frame.dimensions.width,
                    source_height=frame.dimensions.height,
                    target_width=target_width,
                    target_height=target_height,
                ),
            )
            for frame in frames
        ]
        preprocess_ms = (self._clock.monotonic().ns - preprocess_started) / _NANOS_PER_MS

        inference_started = self._clock.monotonic().ns
        try:
            outputs = self._session.infer(prepared)
        except Exception as exc:  # noqa: BLE001 - normalise every framework failure
            self._failures += 1
            raise DetectionFailedError(
                f"YOLO inference failed for a batch of {len(frames)}: "
                f"{type(exc).__name__}: {exc}",
                model_id=str(self._model_id),
                device_id=self._device_id,
            ) from exc
        inference_ms = (self._clock.monotonic().ns - inference_started) / _NANOS_PER_MS

        if len(outputs) != len(frames):
            self._failures += 1
            raise DetectionFailedError(
                f"session returned {len(outputs)} results for {len(frames)} frames; "
                f"batch results must map 1:1 and in order (obligation D6)",
                model_id=str(self._model_id),
            )

        postprocess_started = self._clock.monotonic().ns
        translated = [
            self._translate(boxes, image, request)
            for image, boxes in zip(prepared, outputs, strict=True)
        ]
        postprocess_ms = (
            self._clock.monotonic().ns - postprocess_started
        ) / _NANOS_PER_MS

        model_meta = ModelMeta(
            model_id=self._model_id,
            model_version=self._model_version,
            artifact_hash=self._artifact_hash,
            precision=self._precision,
            device_id=self._device_id,
        )
        timing = InferenceTiming(
            preprocess_ms=preprocess_ms / len(frames),
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms / len(frames),
            batch_size=len(frames),
            device_id=self._device_id,
            model_load_state="warm" if self._warmed else "cold",
        )
        return [
            DetectionResult(
                frame_ref=frame.frame_ref,
                detections=detections,
                model_meta=model_meta,
                timing=timing,
            )
            for frame, detections in zip(frames, translated, strict=True)
        ]

    def _translate(
        self,
        boxes: Sequence[RawBox],
        image: LetterboxedImage,
        request: DetectionRequest,
    ) -> tuple[RawDetection, ...]:
        """Native output to platform vocabulary. Where D1 and D2 are honoured."""
        minimum = request.min_confidence if request.min_confidence is not None else 0.0
        translated: list[RawDetection] = []

        for box in boxes:
            if box.score < minimum:
                continue

            native_label = self._native_label(box.class_index)
            entry = self._mapping.lookup(native_label) if native_label else None
            if entry is None:
                if self._mapping.unmapped_policy is UnmappedPolicy.DROP:
                    continue
                class_id = ClassId("unknown")
            else:
                class_id = entry.class_id

            translated.append(
                RawDetection(
                    class_id=class_id,
                    box=invert_to_normalized(
                        image.transform, box.x1, box.y1, box.x2, box.y2
                    ),
                    score=max(0.0, min(1.0, box.score)),
                    truncation=truncation_of(
                        image.transform, box.x1, box.y1, box.x2, box.y2
                    ),
                    geometry_kind=GeometryKind.BOX,
                    # Diagnostics only. The engine never propagates it, so a
                    # native label cannot escape (obligation D2).
                    native_label=native_label,
                )
            )

        if request.max_detections is not None and len(translated) > request.max_detections:
            translated.sort(key=lambda d: d.score, reverse=True)
            translated = translated[: request.max_detections]
        return tuple(translated)

    def _native_label(self, class_index: int) -> str | None:
        if 0 <= class_index < len(self._native_names):
            return self._native_names[class_index]
        return None

    # --- lifecycle -------------------------------------------------------------- #

    def warm(self) -> None:
        """Run one representative inference.

        Mandatory before the detector counts as ready: a cold first inference can
        be 10-100x slower and would otherwise read as a performance regression.
        """
        blank = memoryview(bytearray(64 * 64 * 3)).toreadonly()
        transform = compute_transform(
            source_width=64,
            source_height=64,
            target_width=self._input_size,
            target_height=self._input_size,
        )
        try:
            self._session.infer(
                [LetterboxedImage(pixels=blank, width=64, height=64, transform=transform)]
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionFailedError(
                f"YOLO warmup failed: {type(exc).__name__}: {exc}",
                model_id=str(self._model_id),
            ) from exc
        self._warmed = True

    def health(self) -> ComponentHealth:
        state = HealthState.HEALTHY if self._warmed else HealthState.STARTING
        if self._failures > 0:
            state = HealthState.DEGRADED
        return ComponentHealth(
            component_id=ModuleId(f"detector.yolo.{self._device_id}"),
            state=state,
            reported_at=self._clock.now(),
            detail=f"{self._failures} inference failure(s)",
            metrics={"failures": float(self._failures)},
        )
