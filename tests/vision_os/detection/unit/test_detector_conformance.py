"""``kit.detector`` — and proof that it has teeth.

Every shipped detector must pass. Equally important, the kit must **reject**
adapters that violate the port: a kit that passes everything proves nothing, and
"every model is replaceable" would already be false.

Each broken adapter below is modelled on a real failure mode, not an invented
one — a leaked native label, a fabricated result on bad input, an undeclared NMS,
a mismatched batch length, and a stateful detector.

A *drifted* letterbox inverse is deliberately absent: no generic kit can detect
it without ground truth, because a correct inverse also moves with aspect ratio.
That obligation is proven by ``test_letterbox.py`` instead, which exercises the
arithmetic directly across ten aspect ratios.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vision_os.adapters.detection import (
    EmptyDetector,
    ReferenceDetector,
    ScriptedDetection,
    YoloDetector,
)
from vision_os.adapters.models import RawBox, ScriptedSession
from vision_os.conformance import DETECTOR_KIT, KitSection
from vision_os.core.model.ids import ClassId
from vision_os.core.model.provenance import InferenceTiming, ModelMeta
from vision_os.core.model.space import Box
from vision_os.core.model.taxonomy import (
    GeometryKind,
    MappingEntry,
    TaxonomyMapping,
    UnmappedPolicy,
)
from vision_os.core.ports.detection import (
    BatchProfile,
    DetectionRequest,
    DetectionResult,
    DetectorCapabilities,
    FrameView,
    NmsDeclaration,
    RawDetection,
)

from ..conftest import ADAPTER_ID, MODEL_ID

PRODUCIBLE = (ClassId("person"), ClassId("vehicle.forklift"))


def _yolo_mapping() -> TaxonomyMapping:
    return TaxonomyMapping(
        adapter_id=ADAPTER_ID,
        model_id=MODEL_ID,
        entries=(
            MappingEntry("person", ClassId("person")),
            MappingEntry("forklift", ClassId("vehicle.forklift")),
        ),
        unmapped_policy=UnmappedPolicy.DROP,
        native_label_space="coco",
    )


class TestShippedAdaptersPass:
    def test_reference_detector_passes_the_full_kit(self, clock) -> None:
        detector = ReferenceDetector(
            clock=clock,
            producible_classes=PRODUCIBLE,
            script=(
                ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),
                ScriptedDetection(
                    ClassId("vehicle.forklift"), Box(0.5, 0.4, 0.8, 0.9), 0.7
                ),
            ),
        )
        report = DETECTOR_KIT.run(detector, fast_only=False)
        assert report.passed, report.failures

    def test_empty_detector_passes(self, clock) -> None:
        """"Nothing detected" is a first-class outcome, not an edge case."""
        report = DETECTOR_KIT.run(EmptyDetector(clock=clock), fast_only=False)
        assert report.passed, report.failures

    def test_yolo_adapter_passes_the_full_kit(self, clock) -> None:
        """The production adapter, exercised without ultralytics or a GPU."""
        session = ScriptedSession(
            names=("person", "car", "forklift"),
            default=(
                RawBox(64.0, 64.0, 192.0, 320.0, 0.88, 0),
                RawBox(300.0, 200.0, 480.0, 400.0, 0.62, 2),
            ),
        )
        detector = YoloDetector(
            clock=clock,
            session=session,
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        report = DETECTOR_KIT.run(detector, fast_only=False)
        assert report.passed, report.failures

    def test_kit_covers_every_section(self) -> None:
        covered = DETECTOR_KIT.sections_covered()
        for section in (
            KitSection.SHAPE,
            KitSection.SEMANTICS,
            KitSection.FAILURE,
            KitSection.RESOURCE,
        ):
            assert section in covered


# --- deliberately broken adapters ---------------------------------------------- #


class _BaseBroken:
    """Minimal conforming detector, subclassed to break exactly one obligation."""

    def __init__(self, clock) -> None:
        self._clock = clock
        self._warmed = False

    def capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            producible_classes=PRODUCIBLE,
            batch=BatchProfile(supported=True, max_size=8, optimal_size=8),
            nms=NmsDeclaration(applied=False),
            deterministic=True,
        )

    def _result(self, frame, detections) -> DetectionResult:
        return DetectionResult(
            frame_ref=frame.frame_ref,
            detections=detections,
            model_meta=ModelMeta(
                model_id=MODEL_ID, model_version="1.0.0", artifact_hash="blake2b:x"
            ),
            timing=InferenceTiming(batch_size=1),
        )

    @staticmethod
    def _keep(detections, request):
        """Honour min_confidence, so only the obligation under test is broken."""
        minimum = request.min_confidence if request.min_confidence is not None else 0.0
        return tuple(d for d in detections if d.score >= minimum)

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        return [
            self._result(
                frame,
                self._keep(
                    (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
                    request,
                ),
            )
            for frame in frames
        ]

    def warm(self) -> None:
        self._warmed = True

    def health(self):
        from vision_os.core.model.health import ComponentHealth, HealthState
        from vision_os.core.model.ids import ModuleId

        return ComponentHealth(
            component_id=ModuleId("detector.broken"),
            state=HealthState.HEALTHY,
            reported_at=self._clock.now(),
        )


class _LeaksNativeLabel(_BaseBroken):
    """Emits a COCO label instead of a platform class (obligation D2)."""

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        return [
            self._result(
                frame,
                self._keep(
                    (
                        RawDetection(
                            ClassId("pedestrian"), Box(0.1, 0.1, 0.3, 0.6), 0.9
                        ),
                    ),
                    request,
                ),
            )
            for frame in frames
        ]


class _WrongBatchLength(_BaseBroken):
    """Returns one result regardless of batch size (obligation D6)."""

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        if not frames:
            return []
        return [
            self._result(
                frames[0],
                self._keep(
                    (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
                    request,
                ),
            )
        ]


class _IgnoresThreshold(_BaseBroken):
    """Ignores ``min_confidence``, so a raised threshold changes nothing."""

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        return [
            self._result(
                frame,
                (RawDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.05),),
            )
            for frame in frames
        ]


class _UndeclaredNms(_BaseBroken):
    """Claims to apply NMS but declares no threshold (obligation D4)."""

    def capabilities(self) -> DetectorCapabilities:
        base = super().capabilities()
        return DetectorCapabilities(
            producible_classes=base.producible_classes,
            batch=base.batch,
            nms=NmsDeclaration.__new__(NmsDeclaration),
            deterministic=True,
        )


class _Stateful(_BaseBroken):
    """Remembers the previous call — which is tracking, not detection (D7)."""

    def __init__(self, clock) -> None:
        super().__init__(clock)
        self._offset = 0.0

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        # Cycles rather than saturating: a capped drift would settle into a fixed
        # value after enough calls and start looking stateless again.
        self._offset = round((self._offset + 0.05) % 0.25, 4)
        return [
            self._result(
                frame,
                self._keep(
                    (
                        RawDetection(
                            ClassId("person"),
                            Box(0.1 + self._offset, 0.1, 0.3 + self._offset, 0.6),
                            0.9,
                        ),
                    ),
                    request,
                ),
            )
            for frame in frames
        ]


class _FabricatesOnBadInput(_BaseBroken):
    """Guesses when handed unusable pixels rather than failing.

    The most dangerous adapter possible: fully-provenanced fiction that looks
    entirely legitimate downstream.
    """

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        results = []
        for frame in frames:
            if frame.dimensions.width <= 1 or len(frame.pixels) < 100:
                results.append(
                    self._result(
                        frame,
                        (
                            RawDetection(
                                ClassId("person"), Box(0.0, 0.0, 1.6, 1.6), 0.99
                            ),
                        ),
                    )
                )
            else:
                results.append(
                    self._result(
                        frame,
                        self._keep(
                            (
                                RawDetection(
                                    ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9
                                ),
                            ),
                            request,
                        ),
                    )
                )
        return results


class _RaisesUntyped(_BaseBroken):
    """Leaks a framework exception across the port boundary."""

    def detect(self, frames, request) -> Sequence[DetectionResult]:
        for frame in frames:
            if len(frame.pixels) < 100:
                raise RuntimeError("cuda kernel launch failed")
        return super().detect(frames, request)


class _NoProducibleClasses:
    """Declares nothing, making its capability gap undetectable (V8)."""

    def __init__(self, clock) -> None:
        self._clock = clock

    def capabilities(self):
        return DetectorCapabilities.__new__(DetectorCapabilities)

    def detect(self, frames, request):
        return []

    def warm(self) -> None: ...

    def health(self):
        from vision_os.core.model.health import ComponentHealth, HealthState
        from vision_os.core.model.ids import ModuleId

        return ComponentHealth(
            ModuleId("x"), HealthState.HEALTHY, self._clock.now()
        )


class TestKitRejectsBrokenAdapters:
    """A kit that passes everything proves nothing."""

    def test_native_label_leak_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_LeaksNativeLabel(clock), fast_only=True)
        assert not report.passed
        assert any("taxonomy_mapping_complete" in f for f in report.failures)

    def test_wrong_batch_length_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_WrongBatchLength(clock), fast_only=True)
        assert not report.passed
        assert any("batch_order_preserved" in f for f in report.failures)

    def test_ignored_threshold_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_IgnoresThreshold(clock), fast_only=True)
        assert not report.passed
        assert any("threshold_behaviour" in f for f in report.failures)

    def test_undeclared_nms_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_UndeclaredNms(clock), fast_only=True)
        assert not report.passed

    def test_stateful_detector_is_caught(self, clock) -> None:
        """A detector that remembers is doing tracking."""
        report = DETECTOR_KIT.run(_Stateful(clock), fast_only=True)
        assert not report.passed
        assert any(
            "statelessness" in f or "determinism" in f for f in report.failures
        ), report.failures

    def test_fabrication_on_bad_input_is_caught(self, clock) -> None:
        """The single most important check in the kit."""
        report = DETECTOR_KIT.run(_FabricatesOnBadInput(clock), fast_only=True)
        assert not report.passed
        assert any("no_fabrication_on_failure" in f for f in report.failures)

    def test_untyped_exception_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_RaisesUntyped(clock), fast_only=True)
        assert not report.passed
        assert any(
            "corrupt_input_is_typed" in f or "no_fabrication" in f
            for f in report.failures
        )

    def test_undeclared_capability_is_caught(self, clock) -> None:
        report = DETECTOR_KIT.run(_NoProducibleClasses(clock), fast_only=True)
        assert not report.passed


class TestFastSubset:
    def test_fast_subset_skips_resource_checks(self, clock) -> None:
        """Seconds at load, before a single real frame is processed."""
        detector = ReferenceDetector(clock=clock, producible_classes=PRODUCIBLE)
        report = DETECTOR_KIT.run(detector, fast_only=True)
        assert report.passed
        assert any("resource/" in skipped for skipped in report.skipped)

    def test_fast_subset_still_catches_the_catastrophic_class(self, clock) -> None:
        for broken in (_LeaksNativeLabel, _FabricatesOnBadInput, _WrongBatchLength):
            report = DETECTOR_KIT.run(broken(clock), fast_only=True)
            assert not report.passed, broken.__name__


class TestYoloAdapterBehaviour:
    def test_unmapped_native_label_is_dropped(self, clock) -> None:
        """``car`` is in the model's label space but not in the mapping."""
        session = ScriptedSession(
            names=("person", "car", "forklift"),
            default=(
                RawBox(64.0, 64.0, 192.0, 320.0, 0.9, 0),
                RawBox(200.0, 200.0, 300.0, 300.0, 0.9, 1),
            ),
        )
        detector = YoloDetector(
            clock=clock,
            session=session,
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        frames = [_view(clock)]
        result = detector.detect(frames, DetectionRequest(min_confidence=0.0))[0]
        assert len(result.detections) == 1
        assert result.detections[0].class_id == "person"

    def test_emit_as_unknown_keeps_unmapped_detections_visible(self, clock) -> None:
        """Silently discarding them would under-report without saying why (V8)."""
        mapping = TaxonomyMapping(
            adapter_id=ADAPTER_ID,
            model_id=MODEL_ID,
            entries=(MappingEntry("person", ClassId("person")),),
            unmapped_policy=UnmappedPolicy.EMIT_AS_UNKNOWN,
        )
        session = ScriptedSession(
            names=("person", "car"),
            default=(RawBox(200.0, 200.0, 300.0, 300.0, 0.9, 1),),
        )
        detector = YoloDetector(
            clock=clock,
            session=session,
            mapping=mapping,
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        result = detector.detect([_view(clock)], DetectionRequest(min_confidence=0.0))[0]
        assert result.detections[0].class_id == "unknown"

    def test_declares_its_built_in_nms(self, clock) -> None:
        """Double-suppression would silently halve object counts."""
        detector = YoloDetector(
            clock=clock,
            session=ScriptedSession(names=("person",)),
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        nms = detector.capabilities().nms
        assert nms.applied
        assert nms.iou_threshold is not None

    def test_capabilities_come_from_the_mapping_not_the_model(self, clock) -> None:
        """Claiming a class the mapping cannot yield is an undetectable gap."""
        detector = YoloDetector(
            clock=clock,
            session=ScriptedSession(names=("person", "car", "forklift", "bicycle")),
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        assert set(detector.capabilities().producible_classes) == {
            "person",
            "vehicle.forklift",
        }

    def test_inference_failure_is_typed(self, clock) -> None:
        from vision_os.core.errors import DetectionFailedError

        session = ScriptedSession(names=("person",), fail_after=1)
        detector = YoloDetector(
            clock=clock,
            session=session,
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        with pytest.raises(DetectionFailedError, match="YOLO inference failed"):
            detector.detect([_view(clock)], DetectionRequest())

    def test_geometry_kind_is_declared(self, clock) -> None:
        detector = YoloDetector(
            clock=clock,
            session=ScriptedSession(names=("person",)),
            mapping=_yolo_mapping(),
            model_id=MODEL_ID,
            model_version="1.0.0",
            artifact_hash="blake2b:yolo",
        )
        assert GeometryKind.BOX in detector.capabilities().geometry_kinds


def _view(clock, width: int = 640, height: int = 360) -> FrameView:
    from vision_os.core.model.frame import FrameDimensions
    from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch

    return FrameView(
        frame_ref=FrameRef(CameraId("cam-01"), StreamEpoch(1), FrameSeq(0)),
        dimensions=FrameDimensions(width=width, height=height),
        pixels=memoryview(bytearray(width * height * 3)).toreadonly(),
    )
