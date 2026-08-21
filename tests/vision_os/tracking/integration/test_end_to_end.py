"""Camera to tracked objects, through the real Flow 1 and Flow 2 platform.

The proof that Flow 3 attaches at documented seams and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission; detection resumes that path; tracking resumes the
detection path — with Flow 1 holding only ``AdmittedFrameConsumer`` and Flow 2
holding only ``DetectionConsumer``.

Also proves the negatives: with no tracking consumer attached, the platform
behaves exactly as Flow 2 did, and nothing in the pipeline knows which tracker
is bound.
"""

from __future__ import annotations

import asyncio
import hashlib

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
from vision_os.conformance import platform_registry
from vision_os.core.model.camera import Camera
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import CameraId, ClassId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.model.track import TrackState
from vision_os.detection_bootstrap import build_detection_layer
from vision_os.kernel.config import ConfigLayer
from vision_os.kernel.events import TrackCreated
from vision_os.perception.detection import DetectionRuntime
from vision_os.tracking_bootstrap import build_tracking_layer

from ...conftest import HEIGHT, WIDTH, base_config_document, make_frames

DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)
REFERENCE_WEIGHTS = b"reference-weights"
REFERENCE_HASH = (
    "blake2b:" + hashlib.blake2b(REFERENCE_WEIGHTS, digest_size=32).hexdigest()
)


def tracking_document(cameras: int = 1, *, tracker_id: str = "tracker.sort") -> dict:
    """Flow 1 + Flow 2 + Flow 3 configuration.

    The tracker is named **only here**, as a config value. Swapping it changes no
    platform code (invariant V3).
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
    document["tracking"] = {
        "enabled": True,
        "tracker_id": tracker_id,
        "min_hits_to_confirm": 2,
        "max_coast_frames": 4,
        "max_lost_frames": 8,
        "max_tracks_per_camera": 32,
    }
    document["taxonomy"] = [{"class_id": "person"}, {"class_id": "vehicle"}]
    document["detectors"] = [
        {
            "detector_id": "reference-primary",
            "adapter_id": "detector.reference",
            "model_id": "reference-detector",
            "model_version": "1.0.0",
            "artifact_uri": "mem://reference.bin",
            "artifact_hash": REFERENCE_HASH,
            "role": "primary_detector",
            "mappings": [{"native_label": "person", "class_id": "person"}],
        }
    ]
    return document


def bindings_factory(clock, frames: int = 10, interpacket_ms: int = 20):
    def factory(camera: Camera) -> SourceBindings:
        return SourceBindings(
            source=InMemoryRawSource(
                make_frames(frames),
                clock=clock,
                semantics=camera.source_semantics,
                interpacket=Duration.from_millis(interpacket_ms),
            ),
            decoder=PassthroughDecoder(dimensions=DIMENSIONS),
            privacy=NoMaskPolicy(),
            clock_sync=ArrivalTimeClockSync(),
        )

    return factory


def detector_factory(clock):
    """A detector that always reports one person in the same place.

    A stationary object is the cleanest end-to-end signal: any fragmentation is
    unambiguously the pipeline's fault rather than the scenario's.
    """

    def factory(declaration):
        return ReferenceDetector(
            clock=clock,
            producible_classes=(ClassId("person"),),
            script=(ScriptedDetection(ClassId("person"), Box(0.3, 0.3, 0.5, 0.8), 0.92),),
        )

    return factory


async def pump(clock, predicate, *, steps: int = 400, step_ms: int = 5) -> None:
    """Advance virtual time while yielding to the loop."""
    for _ in range(steps):
        if predicate():
            return
        clock.advance(Duration.from_millis(step_ms))
        for _ in range(6):
            await asyncio.sleep(0)


async def build_stack(clock, document: dict, *, collected: list | None = None):
    """Boot Flow 1, attach Flow 2, attach Flow 3, wire both seams."""
    artifacts = InMemoryArtifactStore()
    artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)

    platform = build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=bindings_factory(clock),
        clock=clock,
        conformance=platform_registry(),
    )
    tracking = build_tracking_layer(
        platform,
        tracking_sink=(collected.append if collected is not None else None),
    )
    detection = build_detection_layer(
        platform,
        detector_factory=detector_factory(clock),
        artifacts=artifacts,
        runtimes=(ScriptedRuntime(),),
        detection_consumer=tracking.runtime,
    )
    return platform, detection, tracking


class TestEndToEnd:
    async def test_frames_become_tracked_objects(self, clock) -> None:
        """The whole pipeline: pixels to temporal continuity."""
        collected: list = []
        platform, detection, tracking = await build_stack(
            clock, tracking_document(), collected=collected
        )

        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            consumer=tracking.runtime,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001 - seam under test
        await detection.start()
        await runtime.start()
        await tracking.runtime.start()
        await platform.boot()

        await pump(clock, lambda: len(collected) >= 4)

        assert collected, "frames must reach the tracker through both seams"
        tracked = [r for r in collected if r.count]
        assert tracked, "detections must become tracks"
        for result in tracked:
            for track in result.tracks:
                assert track.camera_id == CameraId("cam-01")
                assert track.confidence.semantics.value == "association"
                assert track.spatial.bbox.is_within_unit()
                assert track.provenance.adapter_id

        await detection.stop()
        await platform.shutdown()

    async def test_a_stationary_object_keeps_one_track_id(self, clock) -> None:
        """Any fragmentation here is the pipeline's fault, not the scenario's."""
        collected: list = []
        platform, detection, tracking = await build_stack(
            clock, tracking_document(), collected=collected
        )
        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            consumer=tracking.runtime,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        subscription = platform.bus.subscribe([TrackCreated])
        await detection.start()
        await runtime.start()
        await tracking.runtime.start()
        await platform.boot()

        await pump(clock, lambda: len(collected) >= 6)

        created = subscription.drain()
        assert len(created) == 1, f"object fragmented into {len(created)} tracks"

        await detection.stop()
        await platform.shutdown()

    async def test_a_track_confirms_end_to_end(self, clock) -> None:
        collected: list = []
        platform, detection, tracking = await build_stack(
            clock, tracking_document(), collected=collected
        )
        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            consumer=tracking.runtime,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        await detection.start()
        await runtime.start()
        await tracking.runtime.start()
        await platform.boot()

        await pump(
            clock,
            lambda: any(
                t.state is TrackState.CONFIRMED for r in collected for t in r.tracks
            ),
        )
        states = {t.state for r in collected for t in r.tracks}
        assert TrackState.CONFIRMED in states

        await detection.stop()
        await platform.shutdown()


class TestSeamIsOptional:
    async def test_without_a_tracking_consumer_flow_2_is_unchanged(self, clock) -> None:
        """The Flow 3 seam defaults to ``None``; adding tracking is explicit."""
        artifacts = InMemoryArtifactStore()
        artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)
        platform = build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(tracking_document())},
            bindings_factory=bindings_factory(clock),
            clock=clock,
            conformance=platform_registry(),
        )
        detection = build_detection_layer(
            platform,
            detector_factory=detector_factory(clock),
            artifacts=artifacts,
            runtimes=(ScriptedRuntime(),),
        )
        collected: list = []
        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            sink=collected.extend,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        await detection.start()
        await runtime.start()
        await platform.boot()

        await pump(clock, lambda: len(collected) >= 3)
        assert collected, "detection must still work with no tracker attached"

        await detection.stop()
        await platform.shutdown()

    async def test_a_broken_tracking_consumer_does_not_stop_detection(
        self, clock
    ) -> None:
        """Invariant V9, one layer deeper: tracking may not take detection down."""

        class _Explodes:
            calls = 0

            async def on_detected(self, outcome) -> None:
                type(self).calls += 1
                raise RuntimeError("tracking is broken")

        artifacts = InMemoryArtifactStore()
        artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)
        platform = build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(tracking_document())},
            bindings_factory=bindings_factory(clock),
            clock=clock,
            conformance=platform_registry(),
        )
        detection = build_detection_layer(
            platform,
            detector_factory=detector_factory(clock),
            artifacts=artifacts,
            runtimes=(ScriptedRuntime(),),
        )
        broken = _Explodes()
        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            consumer=broken,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        await detection.start()
        await runtime.start()
        await platform.boot()

        await pump(clock, lambda: _Explodes.calls >= 3)

        assert _Explodes.calls >= 1
        assert runtime.stats.frames_detected >= 1, "detection stopped when tracking broke"
        assert runtime.stats.consumer_failures >= 1

        await detection.stop()
        await platform.shutdown()


class TestTrackerSelectionIsConfiguration:
    async def test_the_configured_tracker_is_the_one_that_binds(self, clock) -> None:
        for tracker_id in ("tracker.iou", "tracker.sort", "tracker.bytetrack"):
            platform = build_platform(
                config_sources={
                    ConfigLayer.SITE: InMemoryConfigSource(
                        tracking_document(tracker_id=tracker_id)
                    )
                },
                bindings_factory=bindings_factory(clock),
                clock=clock,
                conformance=platform_registry(),
            )
            tracking = build_tracking_layer(platform)
            assert tracking.tracker_id == tracker_id
            assert not tracking.is_fallback

    async def test_an_unknown_tracker_falls_back_rather_than_failing_boot(
        self, clock
    ) -> None:
        """Degrade, never die (invariant V9). The IoU fallback needs no weights
        and no device, so this path cannot itself fail."""
        platform = build_platform(
            config_sources={
                ConfigLayer.SITE: InMemoryConfigSource(
                    tracking_document(tracker_id="tracker.does-not-exist")
                )
            },
            bindings_factory=bindings_factory(clock),
            clock=clock,
            conformance=platform_registry(),
        )
        tracking = build_tracking_layer(platform)
        assert tracking.is_fallback
        assert tracking.tracker_id == "tracker.iou"
        assert "does-not-exist" in tracking.manager.fallback_reason

    async def test_tracking_config_reaches_the_tracker(self, clock) -> None:
        document = tracking_document()
        document["tracking"]["max_tracks_per_camera"] = 7
        platform = build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
            bindings_factory=bindings_factory(clock),
            clock=clock,
            conformance=platform_registry(),
        )
        tracking = build_tracking_layer(platform)
        assert tracking.manager.capabilities.max_objects == 7

    async def test_a_missing_kit_refuses_to_build_the_layer(self, clock) -> None:
        """An ungated tracker is never activated, so this is fatal."""
        import pytest

        from vision_os.conformance import flow1_registry
        from vision_os.core.errors import TrackingError

        platform = build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(tracking_document())},
            bindings_factory=bindings_factory(clock),
            clock=clock,
            conformance=flow1_registry(),
        )
        with pytest.raises(TrackingError, match="no conformance kit"):
            build_tracking_layer(platform)
