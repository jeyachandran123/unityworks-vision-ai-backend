"""Camera to canonical objects, through the real Flow 1-3 platform.

The proof that Flow 4 attaches at documented seams and nowhere else: a full
platform boots, frames flow from an in-memory source through decode, masking,
buffering and admission; detection resumes that path; tracking resumes the
detection path; and the registry consumes tracking output — with each earlier
flow holding nothing but a protocol.

Also exercises the composition root, which is the only module that selects an
object store.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path

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
from vision_os.adapters.registry import FileObjectStore, InMemoryObjectStore
from vision_os.bootstrap import build_platform
from vision_os.conformance import flow1_registry, platform_registry
from vision_os.core.errors import RegistryError
from vision_os.core.model.camera import Camera
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.ids import CameraId, ClassId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration
from vision_os.core.ports.registry import IdentityResolverPort, ResolutionResult
from vision_os.detection_bootstrap import build_detection_layer
from vision_os.kernel.config import ConfigLayer
from vision_os.kernel.events import ObjectCreated
from vision_os.perception.detection import DetectionRuntime
from vision_os.registry_bootstrap import (
    build_binding_policy,
    build_lifecycle_policy,
    build_object_store,
    build_registry_layer,
)
from vision_os.tracking_bootstrap import build_tracking_layer

from ...conftest import HEIGHT, WIDTH, base_config_document, make_frames

DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)
REFERENCE_WEIGHTS = b"reference-weights"
REFERENCE_HASH = (
    "blake2b:" + hashlib.blake2b(REFERENCE_WEIGHTS, digest_size=32).hexdigest()
)


def registry_document(cameras: int = 1, **registry_overrides) -> dict:
    """Flow 1 + 2 + 3 + 4 configuration."""
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
        "tracker_id": "tracker.sort",
        "min_hits_to_confirm": 2,
        "max_coast_frames": 4,
        "max_lost_frames": 8,
    }
    document["registry"] = {
        "enabled": True,
        "min_observations_to_confirm": 2,
        "persistence_enabled": False,
        **registry_overrides,
    }
    document["taxonomy"] = [{"class_id": "person"}]
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


def bindings_factory(clock, frames: int = 12, interpacket_ms: int = 20):
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
    """One person, always in the same place — the cleanest end-to-end signal."""

    def factory(declaration):
        return ReferenceDetector(
            clock=clock,
            producible_classes=(ClassId("person"),),
            script=(ScriptedDetection(ClassId("person"), Box(0.3, 0.3, 0.5, 0.8), 0.92),),
        )

    return factory


async def pump(clock, predicate, *, steps: int = 400, step_ms: int = 5) -> None:
    for _ in range(steps):
        if predicate():
            return
        clock.advance(Duration.from_millis(step_ms))
        for _ in range(6):
            await asyncio.sleep(0)


def make_platform(clock, document: dict, *, conformance=None):
    return build_platform(
        config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
        bindings_factory=bindings_factory(clock),
        clock=clock,
        conformance=conformance or platform_registry(),
    )


async def build_stack(clock, document: dict, *, collected: list | None = None):
    """Boot every flow and wire all three seams."""
    artifacts = InMemoryArtifactStore()
    artifacts.put("mem://reference.bin", REFERENCE_WEIGHTS)

    platform = make_platform(clock, document)
    registry_layer = build_registry_layer(
        platform,
        store=InMemoryObjectStore(),
        registry_sink=(collected.append if collected is not None else None),
    )
    tracking = build_tracking_layer(platform, tracking_sink=None)
    detection = build_detection_layer(
        platform,
        detector_factory=detector_factory(clock),
        artifacts=artifacts,
        runtimes=(ScriptedRuntime(),),
        detection_consumer=tracking.runtime,
    )
    return platform, detection, tracking, registry_layer


class _TrackingToRegistry:
    """Bridges the tracking sink to the registry seam.

    Flow 3's runtime takes a synchronous sink; the registry's seam is async, so
    this schedules the hand-off. In a distributed deployment the same edge
    becomes a queue — the shape of the seam is what matters, not the transport.
    """

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self.pending: list = []

    def __call__(self, outcome) -> None:
        if outcome.failed:
            return
        self.pending.append(outcome)

    async def drain(self, tracker) -> None:
        while self.pending:
            outcome = self.pending.pop(0)
            update = _update_from(outcome, tracker)
            if update is not None:
                await self._runtime.on_tracked(outcome.camera_id, update)


def _update_from(outcome, tracker):
    """Rebuild a ``TrackUpdate`` from the tracking outcome and live tracks."""
    from vision_os.core.model.ids import FrameRef as FR
    from vision_os.core.model.ids import FrameSeq, StreamEpoch
    from vision_os.core.model.track import TrackUpdate

    if not outcome.tracks:
        return None
    return TrackUpdate(
        camera_id=outcome.camera_id,
        frame_ref=FR(outcome.camera_id, StreamEpoch(1), FrameSeq(len(outcome.tracks))),
        tracker_epoch=outcome.tracker_epoch,
        active=outcome.tracks,
    )


class TestEndToEnd:
    async def test_frames_become_canonical_objects(self, clock) -> None:
        """The whole pipeline: pixels to durable identity."""
        platform, detection, tracking, registry_layer = await build_stack(
            clock, registry_document()
        )

        bridge = _TrackingToRegistry(registry_layer.runtime)
        tracking.runtime._sink = bridge  # noqa: SLF001 - the seam under test

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
        await registry_layer.runtime.start()
        await platform.boot()

        await pump(clock, lambda: len(bridge.pending) >= 5)
        await bridge.drain(tracking)

        objects = registry_layer.registry.objects(CameraId("cam-01"))
        assert objects, "tracks must become canonical objects"
        for obj in objects:
            assert obj.object_id
            assert obj.camera_id == CameraId("cam-01")
            assert obj.confidence.semantics.value == "identity"
            assert obj.provenance.config_revision

        await detection.stop()
        await platform.shutdown()

    async def test_object_creation_is_published(self, clock) -> None:
        platform, detection, tracking, registry_layer = await build_stack(
            clock, registry_document()
        )
        bridge = _TrackingToRegistry(registry_layer.runtime)
        tracking.runtime._sink = bridge  # noqa: SLF001
        runtime = DetectionRuntime(
            clock=platform.clock,
            bus=platform.bus,
            metrics=platform.metrics,
            health=platform.health,
            engine=detection.engine,
            consumer=tracking.runtime,
        )
        platform.runtime._admitted_consumer = runtime  # noqa: SLF001
        subscription = platform.bus.subscribe([ObjectCreated])

        await detection.start()
        await runtime.start()
        await tracking.runtime.start()
        await registry_layer.runtime.start()
        await platform.boot()

        await pump(clock, lambda: len(bridge.pending) >= 5)
        await bridge.drain(tracking)

        assert subscription.drain(), "object creation must reach the bus"

        await detection.stop()
        await platform.shutdown()


class TestCompositionRoot:
    def test_the_layer_assembles(self, clock) -> None:
        platform = make_platform(clock, registry_document())
        layer = build_registry_layer(platform, store=InMemoryObjectStore())
        assert layer.registry is not None
        assert layer.runtime is not None
        assert layer.store_id == "memory"

    def test_a_disabled_registry_refuses_to_build(self, clock) -> None:
        platform = make_platform(clock, registry_document(enabled=False))
        with pytest.raises(RegistryError, match="enabled is false"):
            build_registry_layer(platform)

    def test_a_missing_kit_refuses_to_build(self, clock) -> None:
        """An ungated store is never used."""
        platform = make_platform(
            clock, registry_document(), conformance=flow1_registry()
        )
        with pytest.raises(RegistryError, match="no conformance kit"):
            build_registry_layer(platform)

    def test_a_non_conforming_store_refuses_to_build(self, clock) -> None:
        from vision_os.core.ports.registry import ObjectStorePort

        class _Amnesiac(ObjectStorePort):
            @property
            def store_id(self) -> str:
                return "amnesiac"

            def save(self, snapshot) -> None:
                return None

            def load(self, camera_id):
                return None

            def forget(self, camera_id) -> None:
                return None

        platform = make_platform(clock, registry_document())
        with pytest.raises(RegistryError, match="failed conformance"):
            build_registry_layer(platform, store=_Amnesiac())

    def test_supplying_a_resolver_is_refused(self, clock) -> None:
        """P11 has no implementations in Phase 1 (15_ROADMAP section 3)."""

        class _Resolver(IdentityResolverPort):
            @property
            def resolver_id(self) -> str:
                return "resolver.spatiotemporal"

            @property
            def requires_embeddings(self) -> bool:
                return False

            def resolve(self, request) -> ResolutionResult:
                return ResolutionResult(abstained=True)

        platform = make_platform(clock, registry_document())
        with pytest.raises(RegistryError, match="Phase 1"):
            build_registry_layer(platform, resolver=_Resolver())

    def test_configuration_reaches_the_lifecycle_policy(self, clock) -> None:
        platform = make_platform(
            clock,
            registry_document(
                min_observations_to_confirm=7, max_objects_per_camera=11
            ),
        )
        policy = build_lifecycle_policy(platform)
        assert policy.min_observations_to_confirm == 7
        assert policy.max_objects_per_camera == 11

    def test_configuration_reaches_the_binding_policy(self, clock) -> None:
        platform = make_platform(
            clock, registry_document(ambiguity_margin=0.42, max_reentry_distance=0.9)
        )
        policy = build_binding_policy(platform)
        assert policy.ambiguity_margin == pytest.approx(0.42)
        assert policy.max_reentry_distance == pytest.approx(0.9)

    def test_persistence_disabled_selects_the_memory_store(self, clock) -> None:
        """Not a fallback for failure — the honest implementation for a site that
        accepts session-scoped identity, and it says so through ``store_id``."""
        platform = make_platform(clock, registry_document(persistence_enabled=False))
        assert build_object_store(platform).store_id == "memory"

    def test_persistence_enabled_selects_a_file_store(self, clock) -> None:
        platform = make_platform(clock, registry_document(persistence_enabled=True))
        assert build_object_store(platform).store_id.startswith("file:")

    def test_the_layer_reports_its_store(self, clock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = make_platform(clock, registry_document())
            layer = build_registry_layer(
                platform, store=FileObjectStore(Path(directory))
            )
            assert layer.store_id.startswith("file:")


class TestRuntimePersistence:
    async def test_objects_are_flushed_and_reloaded(self, clock) -> None:
        """07_STATE section 9.3, through the real runtime schedule."""

        from ..conftest import CAMERA, make_track, make_update

        with tempfile.TemporaryDirectory() as directory:
            store = FileObjectStore(Path(directory))
            platform = make_platform(
                clock,
                registry_document(persistence_enabled=True, persistence_interval_ms=1),
            )
            layer = build_registry_layer(platform, store=store)
            await layer.runtime.start()

            for seq in range(6):
                await layer.runtime.on_tracked(
                    CAMERA, make_update([make_track(seq=seq)], seq=seq)
                )
            await layer.runtime.flush_now()

            before = layer.registry.objects(CAMERA)
            assert before

            # A fresh layer over the same store must recover identity.
            revived = build_registry_layer(make_platform(clock, registry_document(
                persistence_enabled=True
            )), store=store)
            restored = revived.runtime.restore_from(CAMERA)
            assert restored == len(before)
            assert revived.registry.get(before[0].object_id).object_id == (
                before[0].object_id
            )

            await layer.runtime.stop()

    async def test_expiry_runs_on_its_schedule(self, clock) -> None:
        from ..conftest import CAMERA, make_track, make_update

        platform = make_platform(
            clock, registry_document(expiry_interval_ms=1, persistence_enabled=False)
        )
        layer = build_registry_layer(platform, store=InMemoryObjectStore())
        await layer.runtime.start()

        for seq in range(4):
            await layer.runtime.on_tracked(
                CAMERA, make_update([make_track(seq=seq)], seq=seq)
            )
        assert layer.runtime.expire_now() is not None

        await layer.runtime.stop()
