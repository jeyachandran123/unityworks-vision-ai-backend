"""Camera to detection, through the real Flow 1 platform.

The proof that Flow 2 attaches at the documented seam and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission, and detection resumes the path — with Flow 1 holding
nothing but a protocol.

Also proves the negative: with no consumer attached, the platform behaves exactly
as Flow 1 did.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.acquisition import SourceBindings
from vision_os.adapters.acquisition import (
    ArrivalTimeClockSync,
    InMemoryRawSource,
    NoMaskPolicy,
    PassthroughDecoder,
)
from vision_os.adapters.configuration import InMemoryConfigSource
from vision_os.adapters.detection import ReferenceDetector, ScriptedDetection
from vision_os.adapters.models import InMemoryArtifactStore, ScriptedRuntime
from vision_os.bootstrap import build_platform
from vision_os.core.model.camera import Camera
from vision_os.core.model.detection import Detection
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import CameraId, ClassId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.detection_bootstrap import build_detection_layer
from vision_os.kernel.config import ConfigLayer
from vision_os.kernel.metrics import MetricName
from vision_os.perception.detection import DetectionRuntime

from ...conftest import HEIGHT, WIDTH, base_config_document, make_frames

DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)

REFERENCE_WEIGHTS = b"reference-weights"
REFERENCE_HASH = (
    "blake2b:" + __import__("hashlib").blake2b(REFERENCE_WEIGHTS, digest_size=32).hexdigest()
)
"""The declared hash must be correct *at validation time*, not patched in later:
an artifact without a verified hash never loads (12_SECURITY section 6)."""


def _detection_document(cameras: int = 1) -> dict:
    """Flow 1 configuration plus the four Flow 2 channels.

    Taxonomy, detectors, and their mappings are *data*. Swapping the model here
    changes no platform code (invariant V3).
    """
    document = base_config_document(cameras=cameras, target_fps=1000.0)
    document["detection"] = {
        "enabled": True,
        "confidence_threshold": 0.25,
        "max_batch_size": 4,
        "batch_max_wait_ms": 0,
        "inference_timeout_ms": 2000,
        "queue_capacity": 32,
    }
    document["models"] = {"warmup_enabled": True, "allow_cpu_fallback": True}
    document["taxonomy"] = [
        {"class_id": "person"},
        {"class_id": "vehicle"},
        {"class_id": "vehicle.forklift"},
    ]
    document["detectors"] = [
        {
            "detector_id": "reference-primary",
            "adapter_id": "detector.reference",
            "model_id": "reference-detector",
            "model_version": "1.0.0",
            "artifact_uri": "mem://reference.bin",
            "artifact_hash": REFERENCE_HASH,
            "role": "primary_detector",
            "mappings": [
                {"native_label": "person", "class_id": "person"},
                {"native_label": "forklift", "class_id": "vehicle.forklift"},
            ],
        }
    ]
    return document


def _bindings_factory(clock, frames: int = 6, interpacket_ms: int = 0):
    """Build acquisition adapters.

    ``interpacket_ms`` paces frame arrival on the injected clock. Without it a
    source delivers its whole script in one event-loop burst at a single instant,
    and the scheduler correctly admits one frame and drops the rest on cadence —
    accurate behaviour, but not what a camera does.
    """

    def factory(camera: Camera) -> SourceBindings:
        return SourceBindings(
            source=InMemoryRawSource(
                make_frames(frames),
                clock=clock,
                semantics=camera.source_semantics,
                interpacket=(
                    Duration.from_millis(interpacket_ms) if interpacket_ms else None
                ),
            ),
            decoder=PassthroughDecoder(dimensions=DIMENSIONS),
            privacy=NoMaskPolicy(),
            clock_sync=ArrivalTimeClockSync(),
        )

    return factory


def _detector_factory(clock):
    def factory(declaration):
        return ReferenceDetector(
            clock=clock,
            producible_classes=(ClassId("person"), ClassId("vehicle.forklift")),
            script=(
                ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.92),
                ScriptedDetection(
                    ClassId("vehicle.forklift"), Box(0.5, 0.4, 0.8, 0.9), 0.71
                ),
            ),
        )

    return factory


async def _pump(clock, predicate, *, steps: int = 300, step_ms: int = 5) -> None:
    """Advance virtual time while yielding to the loop.

    Cadence is measured on the injected clock, so a test that never advances it
    admits exactly one frame per camera and then throttles forever — which is the
    scheduler working correctly, not a bug.
    """
    for _ in range(steps):
        if predicate():
            return
        clock.advance(Duration.from_millis(step_ms))
        for _ in range(6):
            await asyncio.sleep(0)


async def _build(clock, document: dict, *, collected: list[Detection] | None = None):
    """Boot Flow 1, attach Flow 2, and wire the seam."""
    artifacts = InMemoryArtifactStore()
    artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)

    platform = build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=_bindings_factory(clock),
        clock=clock,
    )
    layer = build_detection_layer(
        platform,
        detector_factory=_detector_factory(clock),
        artifacts=artifacts,
        runtimes=(ScriptedRuntime(),),
        detection_sink=(collected.extend if collected is not None else None),
    )
    return platform, layer


class TestEndToEnd:
    async def test_frames_become_detections(self, clock) -> None:
        """The whole of Flow 2, from a camera to standardized detections."""
        collected: list[Detection] = []
        platform, layer = await _build(clock, _detection_document(), collected=collected)

        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=layer.engine,
            sink=collected.extend,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001 - seam under test
        await layer.start()
        await runtime.start()
        await platform.boot()

        await _pump(clock, lambda: len(collected) >= 4)

        assert collected, "frames must reach the detector through the seam"
        for detection in collected:
            assert detection.class_id in ("person", "vehicle.forklift")
            assert detection.spatial.bbox.is_within_unit()
            assert detection.provenance.model_artifact_hash
            assert detection.provenance.adapter_id == "detector.reference"
            assert detection.taxonomy_version

        await layer.stop()
        await platform.shutdown()

    async def test_no_consumer_means_flow_one_behaviour(self, clock) -> None:
        """Detection is opt-in: absent, the platform is exactly as it was."""
        platform = build_platform(
            config_sources={
                ConfigLayer.SITE: InMemoryConfigSource(
                    base_config_document(target_fps=1000.0)
                )
            },
            bindings_factory=_bindings_factory(clock),
            clock=clock,
        )
        await platform.boot()
        await _pump(
            clock,
            lambda: platform.runtime.pipeline_stats(CameraId("cam-01")).admitted >= 3,
        )

        stats = platform.runtime.pipeline_stats(CameraId("cam-01"))
        assert stats.admitted > 0
        assert platform.metrics.snapshot().counter_value(
            MetricName.PIPELINE_CONSUMER_FAILURES, camera_id="cam-01"
        ) == 0
        await platform.shutdown()

    async def test_detection_failure_never_stops_acquisition(self, clock) -> None:
        """The seam is a firewall (invariant V9)."""

        class HostileConsumer:
            def __init__(self) -> None:
                self.calls = 0

            async def on_admitted(self, frame_ref, fidelity) -> None:
                self.calls += 1
                raise RuntimeError("detection layer exploded")

        consumer = HostileConsumer()
        platform = build_platform(
            config_sources={
                ConfigLayer.SITE: InMemoryConfigSource(
                    base_config_document(target_fps=1000.0)
                )
            },
            bindings_factory=_bindings_factory(clock, frames=12, interpacket_ms=5),
            clock=clock,
            admitted_frame_consumer=consumer,
        )
        await platform.boot()
        await _pump(
            clock,
            lambda: platform.runtime.pipeline_stats(CameraId("cam-01")).admitted >= 3,
        )

        assert consumer.calls > 0
        stats = platform.runtime.pipeline_stats(CameraId("cam-01"))
        assert stats.admitted > 1, "acquisition kept running through the failures"
        assert platform.metrics.snapshot().counter_value(
            MetricName.PIPELINE_CONSUMER_FAILURES, camera_id="cam-01"
        ) > 0, "a consumer that raises is counted, never hidden"
        await platform.shutdown()

    async def test_multi_camera_batching(self, clock) -> None:
        """Frames from several cameras share one model, as designed."""
        collected: list[Detection] = []
        platform, layer = await _build(
            clock, _detection_document(cameras=3), collected=collected
        )
        runtime = DetectionRuntime(
            clock=platform.clock, bus=platform.bus, metrics=platform.metrics,
            health=platform.health, engine=layer.engine, sink=collected.extend,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        await layer.start()
        await runtime.start()
        await platform.boot()

        await _pump(
            clock, lambda: len({d.frame_ref.camera_id for d in collected}) >= 2
        )

        cameras = {d.frame_ref.camera_id for d in collected}
        assert len(cameras) >= 2, "several cameras fed the shared detector"
        await layer.stop()
        await platform.shutdown()

    async def test_detection_layer_declines_when_no_detector_is_declared(
        self, clock
    ) -> None:
        """A site with no detector should not run a layer producing nothing."""
        from vision_os.core.errors import DetectionError

        document = _detection_document()
        document["detectors"] = []
        platform = build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
            bindings_factory=_bindings_factory(clock),
            clock=clock,
        )
        with pytest.raises(DetectionError, match="no enabled detector"):
            build_detection_layer(
                platform,
                detector_factory=_detector_factory(clock),
                artifacts=InMemoryArtifactStore(),
                runtimes=(ScriptedRuntime(),),
            )
        await platform.shutdown(graceful=False)


class TestConfigurationIsTheOnlyVerticalChannel:
    def test_taxonomy_rejects_a_role(self) -> None:
        """No crop evidences a role, so no role may name a class (invariant V1)."""
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["taxonomy"].append({"class_id": "staff_member"})
        violations = validate(document)
        assert any("names a role" in v for v in violations)

    def test_taxonomy_rejects_a_judgment(self) -> None:
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["taxonomy"].append({"class_id": "unsafe_behaviour"})
        violations = validate(document)
        assert any("judgment" in v for v in violations)

    def test_visual_kinds_are_accepted(self) -> None:
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["taxonomy"].extend(
            [
                {"class_id": "container"},
                {"class_id": "container.tray"},
                {"class_id": "equipment"},
                {"class_id": "equipment.wheelchair"},
            ]
        )
        assert validate(document) == ()

    def test_orphan_class_is_rejected(self) -> None:
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["taxonomy"].append({"class_id": "furniture.bed"})
        violations = validate(document)
        assert any("no declared parent" in v for v in violations)

    def test_mapping_to_an_undeclared_class_is_rejected(self) -> None:
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["detectors"][0]["mappings"].append(
            {"native_label": "cat", "class_id": "animal.cat"}
        )
        violations = validate(document)
        assert any("undeclared taxonomy class" in v for v in violations)

    def test_detector_requires_an_artifact_hash(self) -> None:
        """Unverified weights are a supply-chain hole."""
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["detectors"][0]["artifact_hash"] = ""
        violations = validate(document)
        assert any("artifact_hash is required" in v for v in violations)

    def test_unknown_precision_is_rejected(self) -> None:
        from vision_os.kernel.config import validate

        document = _detection_document()
        document["detectors"][0]["precision"] = "fp8"
        violations = validate(document)
        assert any("precision must be one of" in v for v in violations)

    def test_the_schema_still_admits_no_business_rule(self) -> None:
        """Flow 2 widened the schema by exactly four data channels, not by a rule."""
        from vision_os.kernel.config import validate

        for attempt in (
            {"detection_rules": [{"if": "person_count>5", "then": "alert"}]},
            {"alerts": {"enabled": True}},
        ):
            document = _detection_document()
            document.update(attempt)
            assert validate(document), f"{attempt} should have been rejected"
