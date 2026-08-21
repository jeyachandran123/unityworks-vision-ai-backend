"""P8 — the reference detector.

Fully deterministic and dependency-free, so the entire detection pipeline can be
exercised in CI without a GPU, a model file, or a network. It is to Flow 2 what
``InMemoryRawSource`` was to Flow 1.

It is a *reference*, not a fake: it honours every obligation the port declares —
normalized coordinates, platform taxonomy, honest NMS declaration, empty results
as a valid outcome, statelessness — and it passes the same ``kit.detector`` a
production adapter must pass. An adapter that only tests are allowed to use would
prove nothing about the contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ...core.errors import DetectionFailedError
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import ClassId, ModelId, ModuleId
from ...core.model.provenance import InferenceTiming, ModelMeta
from ...core.model.space import Box
from ...core.model.taxonomy import GeometryKind
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

_REFERENCE_MODEL_ID = ModelId("reference-detector")
_EMPTY_MODEL_ID = ModelId("empty-detector")


@dataclass(frozen=True, slots=True)
class ScriptedDetection:
    """One detection the reference detector will emit, already normalized."""

    class_id: ClassId
    box: Box
    score: float
    truncation: float | None = None


class ReferenceDetector:
    """Emits a scripted, deterministic set of detections per frame."""

    def __init__(
        self,
        *,
        clock: Clock,
        producible_classes: Sequence[ClassId],
        script: Sequence[ScriptedDetection] = (),
        per_frame: dict[str, Sequence[ScriptedDetection]] | None = None,
        model_id: ModelId = _REFERENCE_MODEL_ID,
        model_version: str = "1.0.0",
        artifact_hash: str = "blake2b:reference",
        device_id: str = "cpu",
        max_batch_size: int = 16,
        nms_applied: bool = False,
        fail_on_call: int = 0,
        emit_native_label: str | None = None,
        emit_out_of_range: bool = False,
    ) -> None:
        if not producible_classes:
            raise ValueError("a detector must declare at least one producible class")
        self._clock = clock
        self._producible = tuple(producible_classes)
        self._script = tuple(script)
        self._per_frame = dict(per_frame or {})
        self._model_id = model_id
        self._model_version = model_version
        self._artifact_hash = artifact_hash
        self._device_id = device_id
        self._max_batch = max_batch_size
        self._nms_applied = nms_applied
        self._fail_on_call = fail_on_call
        self._emit_native_label = emit_native_label
        self._emit_out_of_range = emit_out_of_range
        self._calls = 0
        self._warmed = False

    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            producible_classes=self._producible,
            geometry_kinds=(GeometryKind.BOX,),
            input_constraints=InputConstraints(
                min_width=1, min_height=1, max_width=4096, max_height=4096
            ),
            batch=BatchProfile(
                supported=True, max_size=self._max_batch, optimal_size=self._max_batch
            ),
            nms=NmsDeclaration(
                applied=self._nms_applied,
                iou_threshold=0.45 if self._nms_applied else None,
            ),
            precision="fp32",
            deterministic=True,
            device_id=self._device_id,
        )

    def detect(
        self, frames: Sequence[FrameView], request: DetectionRequest
    ) -> Sequence[DetectionResult]:
        self._calls += 1
        if self._fail_on_call and self._calls >= self._fail_on_call:
            raise DetectionFailedError(
                "reference detector scripted failure", model_id=str(self._model_id)
            )

        started = self._clock.monotonic().ns
        results: list[DetectionResult] = []
        for frame in frames:
            scripted = self._per_frame.get(str(frame.frame_ref), self._script)
            results.append(
                DetectionResult(
                    frame_ref=frame.frame_ref,
                    detections=self._emit(scripted, request),
                    model_meta=ModelMeta(
                        model_id=self._model_id,
                        model_version=self._model_version,
                        artifact_hash=self._artifact_hash,
                        precision="fp32",
                        device_id=self._device_id,
                    ),
                    timing=InferenceTiming(
                        inference_ms=(self._clock.monotonic().ns - started) / 1_000_000,
                        batch_size=len(frames),
                        device_id=self._device_id,
                        model_load_state="warm" if self._warmed else "cold",
                    ),
                )
            )
        return results

    def _emit(
        self, scripted: Sequence[ScriptedDetection], request: DetectionRequest
    ) -> tuple[RawDetection, ...]:
        minimum = request.min_confidence if request.min_confidence is not None else 0.0
        emitted: list[RawDetection] = []
        for item in scripted:
            if item.score < minimum:
                continue
            box = (
                Box(0.5, 0.5, 1.5, 1.5) if self._emit_out_of_range else item.box
            )
            emitted.append(
                RawDetection(
                    class_id=(
                        ClassId(self._emit_native_label)
                        if self._emit_native_label
                        else item.class_id
                    ),
                    box=box,
                    score=item.score,
                    truncation=item.truncation,
                    geometry_kind=GeometryKind.BOX,
                )
            )

        # Honour the caller's cap, keeping the strongest. Returning more than was
        # asked for wastes transport on detections the platform will discard, and
        # makes the request parameter a lie.
        if request.max_detections is not None and len(emitted) > request.max_detections:
            emitted.sort(key=lambda d: d.score, reverse=True)
            emitted = emitted[: request.max_detections]
        return tuple(emitted)

    def warm(self) -> None:
        self._warmed = True

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            component_id=ModuleId("detector.reference"),
            state=HealthState.HEALTHY if self._warmed else HealthState.STARTING,
            reported_at=self._clock.now(),
            detail=f"{self._calls} call(s)",
        )

    @property
    def calls(self) -> int:
        return self._calls


@dataclass(slots=True)
class EmptyDetector:
    """Always finds nothing.

    Exists because "nothing detected" is a **valid, non-error** outcome that the
    platform must handle identically to a populated one (obligation D5). An empty
    scene is the common case in most real deployments, and treating it as an edge
    case is how "no detections" quietly becomes "detection failed".
    """

    clock: Clock
    producible_classes: tuple[ClassId, ...] = (ClassId("person"),)
    model_id: ModelId = _EMPTY_MODEL_ID
    warmed: bool = False
    calls: int = field(default=0)

    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            producible_classes=self.producible_classes,
            batch=BatchProfile(supported=True, max_size=8, optimal_size=8),
            nms=NmsDeclaration(applied=False),
        )

    def detect(
        self, frames: Sequence[FrameView], request: DetectionRequest
    ) -> Sequence[DetectionResult]:
        self.calls += 1
        return [
            DetectionResult(
                frame_ref=frame.frame_ref,
                detections=(),
                model_meta=ModelMeta(
                    model_id=self.model_id,
                    model_version="1.0.0",
                    artifact_hash="blake2b:empty",
                ),
                timing=InferenceTiming(batch_size=len(frames)),
            )
            for frame in frames
        ]

    def warm(self) -> None:
        self.warmed = True

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            component_id=ModuleId("detector.empty"),
            state=HealthState.HEALTHY,
            reported_at=self.clock.now(),
        )
