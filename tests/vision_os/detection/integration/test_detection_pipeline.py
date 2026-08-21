"""The Detection Engine, batch scheduler, worker, manager and runtime, wired.

These are the tests only an assembled pipeline can catch: cross-camera batching,
the lease protocol under eviction, every failure degrading rather than escalating,
the plugin gate refusing a broken adapter, and detection never taking the Vision
Runtime down with it.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.acquisition import FrameBuffer
from vision_os.adapters.detection import ReferenceDetector, ScriptedDetection
from vision_os.core.errors import (
    ConformanceFailedError,
    DetectionError,
    DetectionQueueFullError,
    TaxonomyError,
)
from vision_os.core.model.camera import SourceSemantics
from vision_os.core.model.frame import FrameDimensions, PrivacyState
from vision_os.core.model.ids import CameraId, ClassId, FrameRef, FrameSeq, StreamEpoch
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import (
    ClockQuality,
    Duration,
    FrameTime,
)
from vision_os.core.ports.detection import DetectionRequest, FrameView
from vision_os.core.ports.scheduling import Fidelity
from vision_os.kernel.config.schema import (
    DetectorDeclaration,
    MappingEntryDeclaration,
)
from vision_os.kernel.metrics import MetricName
from vision_os.kernel.plugins import PluginManager
from vision_os.perception.detection import (
    DetectionEngine,
    DetectionManager,
    DetectionRuntime,
    DetectionScheduler,
    DetectorRegistration,
    DeviceWorker,
)

from ..conftest import CAMERA, MODEL_ID, frame_ref, register_reference_model

WIDTH, HEIGHT = 8, 4
FRAME_BYTES = WIDTH * HEIGHT * 3
DIMENSIONS = FrameDimensions(width=WIDTH, height=HEIGHT)


def _publish(buffer: FrameBuffer, clock, seq: int, camera: CameraId = CAMERA):
    slot = buffer.acquire_slot(camera, SourceSemantics.REALTIME)
    slot.memory()[:FRAME_BYTES] = bytes([(seq % 250) + 1]) * FRAME_BYTES
    now = clock.now()
    return buffer.publish(
        slot,
        frame_ref=FrameRef(camera, StreamEpoch(1), FrameSeq(seq)),
        time=FrameTime(
            pts=seq * 40,
            t_capture=now,
            t_capture_uncertainty=Duration.from_millis(10),
            t_ingest=now,
            t_decoded=now,
            clock_quality=ClockQuality.NTP_SYNCED,
        ),
        dimensions=DIMENSIONS,
        privacy_state=PrivacyState.MASKED,
        bytes_written=FRAME_BYTES,
    )


def _view(seq: int = 0, camera: CameraId = CAMERA) -> FrameView:
    return FrameView(
        frame_ref=FrameRef(camera, StreamEpoch(1), FrameSeq(seq)),
        dimensions=DIMENSIONS,
        pixels=memoryview(bytearray(FRAME_BYTES)).toreadonly(),
    )


@pytest.fixture
def engine(
    clock,
    bus,
    metrics,
    buffer,
    camera_manager,
    taxonomy,
    binding,
    detection_scheduler,
    worker,
    detection_config,
    camera,
):
    camera_manager.provision(camera)
    buffer.register_camera(CAMERA)
    return DetectionEngine(
        clock=clock,
        bus=bus,
        metrics=metrics,
        buffer=buffer,
        camera_manager=camera_manager,
        taxonomy=taxonomy,
        binding=binding,
        scheduler=detection_scheduler,
        worker=worker,
        config=detection_config,
        config_revision="cfg-test",
    )


class TestEngineHappyPath:
    async def test_produces_standardized_detections(self, engine, buffer, clock) -> None:
        """The whole point of Flow 2: pixels in, standardized detections out."""
        frame = _publish(buffer, clock, 0)
        outcome = await engine.detect(frame.frame_ref)

        assert not outcome.failed
        assert outcome.count == 2, "the third scripted detection is below threshold"
        for detection in outcome.detections:
            assert detection.frame_ref == frame.frame_ref
            assert detection.coordinate_space == "normalized"
            assert detection.spatial.bbox.is_within_unit()
            assert detection.confidence.semantics.value == "detection_presence"
            assert detection.provenance.model_artifact_hash
            assert detection.provenance.config_revision == "cfg-test"
            assert detection.evidence.input_hash.startswith("blake2b:")
            assert detection.taxonomy_version
            assert detection.timing.inference_ms >= 0.0

    async def test_empty_result_is_not_a_failure(
        self, clock, bus, metrics, buffer, camera_manager, taxonomy, binding,
        detection_config, camera, worker, detection_scheduler
    ) -> None:
        """"Nothing there" and "could not look" are different facts (D5)."""
        camera_manager.provision(camera)
        buffer.register_camera(CAMERA)
        engine = DetectionEngine(
            clock=clock, bus=bus, metrics=metrics, buffer=buffer,
            camera_manager=camera_manager, taxonomy=taxonomy, binding=binding,
            scheduler=detection_scheduler, worker=worker,
            config=DetectionSection_with(detection_config, confidence_threshold=0.999),
            config_revision="cfg-test",
        )
        frame = _publish(buffer, clock, 0)
        outcome = await engine.detect(frame.frame_ref)

        assert not outcome.failed
        assert outcome.count == 0

    async def test_publishes_a_completion_event(self, engine, buffer, clock, bus) -> None:
        subscription = bus.subscribe(["detection.completed"])
        frame = _publish(buffer, clock, 0)
        await engine.detect(frame.frame_ref)

        events = subscription.drain()
        assert events
        assert events[0].detection_count == 2
        assert events[0].frame_ref == str(frame.frame_ref)

    async def test_records_metrics(self, engine, buffer, clock, metrics) -> None:
        frame = _publish(buffer, clock, 0)
        await engine.detect(frame.frame_ref)

        snapshot = metrics.snapshot()
        assert snapshot.counter_value(
            MetricName.DETECTION_FRAMES_PROCESSED, camera_id=str(CAMERA)
        ) == 1
        assert snapshot.counter_value(
            MetricName.DETECTIONS_EMITTED, camera_id=str(CAMERA)
        ) == 2
        assert snapshot.histogram_values(
            MetricName.DETECTION_INFERENCE_MS, camera_id=str(CAMERA)
        )
        # The low-confidence detection never reaches the platform's rejection
        # path: the adapter pre-filtered it on ``min_confidence``, which is the
        # point of that hint. Platform-side rejection is covered in the
        # normalizer's unit tests, where the adapter does not pre-filter.

    async def test_releases_the_lease(self, engine, buffer, clock) -> None:
        """A leaked lease would exhaust the pool for every camera."""
        frame = _publish(buffer, clock, 0)
        await engine.detect(frame.frame_ref)
        assert buffer.stats().leases_active == 0

    async def test_capability_gap_is_reported(self, engine) -> None:
        gap = engine.capability_gap((ClassId("person"), ClassId("furniture.bed")))
        assert ClassId("furniture.bed") in gap
        assert ClassId("person") not in gap

    async def test_detect_batch_returns_independent_outcomes(
        self, engine, buffer, clock
    ) -> None:
        refs = [_publish(buffer, clock, i).frame_ref for i in range(3)]
        outcomes = await engine.detect_batch(refs)
        assert len(outcomes) == 3
        assert all(not o.failed for o in outcomes.values())


class TestEngineFailureDegradation:
    """Detection failure must never terminate the Vision Runtime (invariant V9)."""

    async def test_evicted_frame_degrades_cleanly(self, engine) -> None:
        """Normal: the frame went between admission and detection."""
        outcome = await engine.detect(frame_ref(999))
        assert outcome.failed
        assert outcome.reason == "frame_unavailable"
        assert outcome.count == 0

    async def test_detector_failure_is_absorbed(
        self, clock, bus, metrics, buffer, camera_manager, taxonomy, binding,
        detection_config, camera, scripted_detections
    ) -> None:
        camera_manager.provision(camera)
        buffer.register_camera(CAMERA)
        failing = ReferenceDetector(
            clock=clock,
            producible_classes=(ClassId("person"),),
            script=scripted_detections,
            fail_on_call=1,
        )
        worker = DeviceWorker(clock=clock, detector=failing, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        engine = DetectionEngine(
            clock=clock, bus=bus, metrics=metrics, buffer=buffer,
            camera_manager=camera_manager, taxonomy=taxonomy,
            binding=binding,
            scheduler=DetectionScheduler(
                clock=clock, executor=execute, max_batch_size=1,
                max_wait=Duration(0), queue_capacity=8,
                inference_timeout=Duration.from_millis(500),
            ),
            worker=worker, config=detection_config, config_revision="cfg",
        )
        subscription = bus.subscribe(["detection.failed"])
        frame = _publish(buffer, clock, 0)
        outcome = await engine.detect(frame.frame_ref)

        assert outcome.failed
        assert subscription.drain(), "a failure must be published, never silent"

    async def test_contract_violation_is_rejected_not_propagated(
        self, clock, bus, metrics, buffer, camera_manager, taxonomy, binding,
        detection_config, camera
    ) -> None:
        """A native label reaching state would be undetectable downstream."""
        camera_manager.provision(camera)
        buffer.register_camera(CAMERA)
        leaky = ReferenceDetector(
            clock=clock,
            producible_classes=(ClassId("person"),),
            script=(ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),),
            emit_native_label="pedestrian",
        )
        worker = DeviceWorker(clock=clock, detector=leaky, device_id="cpu")

        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        engine = DetectionEngine(
            clock=clock, bus=bus, metrics=metrics, buffer=buffer,
            camera_manager=camera_manager, taxonomy=taxonomy, binding=binding,
            scheduler=DetectionScheduler(
                clock=clock, executor=execute, max_batch_size=1,
                max_wait=Duration(0), queue_capacity=8,
                inference_timeout=Duration.from_millis(500),
            ),
            worker=worker, config=detection_config, config_revision="cfg",
        )
        frame = _publish(buffer, clock, 0)
        outcome = await engine.detect(frame.frame_ref)

        assert outcome.failed
        assert outcome.reason.startswith("contract_violation") or "taxonomy" in outcome.reason

    async def test_timeout_degrades(
        self, clock, bus, metrics, buffer, camera_manager, taxonomy, binding,
        detection_config, camera
    ) -> None:
        camera_manager.provision(camera)
        buffer.register_camera(CAMERA)

        async def slow(key, batch):
            await asyncio.sleep(0.5)
            return []

        engine = DetectionEngine(
            clock=clock, bus=bus, metrics=metrics, buffer=buffer,
            camera_manager=camera_manager, taxonomy=taxonomy, binding=binding,
            scheduler=DetectionScheduler(
                clock=clock, executor=slow, max_batch_size=1,
                max_wait=Duration(0), queue_capacity=8,
                inference_timeout=Duration.from_millis(30),
            ),
            worker=DeviceWorker(clock=clock, detector=binding.detector, device_id="cpu"),
            config=detection_config, config_revision="cfg",
        )
        frame = _publish(buffer, clock, 0)
        outcome = await engine.detect(frame.frame_ref)

        assert outcome.failed
        assert outcome.reason.startswith("timeout") or "exceeded" in outcome.reason

    async def test_unknown_camera_degrades(self, engine, buffer, clock) -> None:
        other = CameraId("cam-unregistered")
        buffer.register_camera(other)
        frame = _publish(buffer, clock, 0, camera=other)
        outcome = await engine.detect(frame.frame_ref)
        assert outcome.failed
        assert outcome.reason.startswith("camera_unknown") or "unknown camera" in outcome.reason


class TestBatchScheduler:
    async def test_batches_across_cameras(self, clock, worker) -> None:
        """The critical inversion: one shared model, many camera flows."""
        seen: list[int] = []

        async def execute(key, batch):
            seen.append(len(batch))
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=4,
            max_wait=Duration.from_millis(5), queue_capacity=32,
            inference_timeout=Duration.from_millis(1_000),
        )
        key = _batch_key()
        results = await asyncio.gather(
            *(
                scheduler.submit(
                    key=key,
                    frame_ref=_view(i, CameraId(f"cam-{i:02d}")).frame_ref,
                    camera_id=CameraId(f"cam-{i:02d}"),
                    view=_view(i, CameraId(f"cam-{i:02d}")),
                    request=DetectionRequest(),
                )
                for i in range(4)
            )
        )
        assert len(results) == 4
        assert max(seen) > 1, "frames from different cameras must share a batch"

    async def test_batch_full_triggers_immediately(self, clock, worker) -> None:
        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=2,
            max_wait=Duration.from_millis(60_000), queue_capacity=8,
            inference_timeout=Duration.from_millis(1_000),
        )
        key = _batch_key()
        results = await asyncio.gather(
            *(
                scheduler.submit(
                    key=key, frame_ref=_view(i).frame_ref, camera_id=CAMERA,
                    view=_view(i), request=DetectionRequest(),
                )
                for i in range(2)
            )
        )
        assert len(results) == 2, (
            "a full batch must flush without waiting for the timeout, or a busy "
            "deployment would stall on a wait it never needed"
        )

    async def test_partial_batch_flushes_with_zero_wait(self, clock, worker) -> None:
        """Deterministic mode: composition must not depend on arrival timing."""
        async def execute(key, batch):
            return await worker.execute([i.view for i in batch], batch[0].request)

        scheduler = DetectionScheduler(
            clock=clock, executor=execute, max_batch_size=16,
            max_wait=Duration(0), queue_capacity=8,
            inference_timeout=Duration.from_millis(1_000),
        )
        result = await scheduler.submit(
            key=_batch_key(), frame_ref=_view(0).frame_ref, camera_id=CAMERA,
            view=_view(0), request=DetectionRequest(),
        )
        assert result.frame_ref == _view(0).frame_ref

    async def test_queue_is_bounded(self, clock) -> None:
        """An unbounded inference queue is a memory leak with a delayed fuse."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking(key, batch):
            started.set()
            await release.wait()
            return [
                _result_for(item.view) for item in batch
            ]

        scheduler = DetectionScheduler(
            clock=clock, executor=blocking, max_batch_size=1,
            max_wait=Duration(0), queue_capacity=2,
            inference_timeout=Duration.from_millis(2_000),
        )
        key = _batch_key()
        tasks = [
            asyncio.create_task(
                scheduler.submit(
                    key=key, frame_ref=_view(i).frame_ref, camera_id=CAMERA,
                    view=_view(i), request=DetectionRequest(),
                )
            )
            for i in range(3)
        ]
        await started.wait()

        # Fired concurrently: awaiting each in turn would block on the first
        # accepted submission rather than reaching the capacity boundary.
        overflow = [
            asyncio.create_task(
                scheduler.submit(
                    key=key, frame_ref=_view(90 + i).frame_ref, camera_id=CAMERA,
                    view=_view(90 + i), request=DetectionRequest(),
                )
            )
            for i in range(8)
        ]
        outcomes = await asyncio.gather(*overflow, return_exceptions=True)
        assert any(
            isinstance(outcome, DetectionQueueFullError) for outcome in outcomes
        ), "the queue must shed rather than grow without bound"

        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await scheduler.close()

    async def test_executor_failure_fails_only_its_batch(self, clock) -> None:
        async def exploding(key, batch):
            raise RuntimeError("device fault")

        scheduler = DetectionScheduler(
            clock=clock, executor=exploding, max_batch_size=1,
            max_wait=Duration(0), queue_capacity=8,
            inference_timeout=Duration.from_millis(500),
        )
        with pytest.raises(RuntimeError, match="device fault"):
            await scheduler.submit(
                key=_batch_key(), frame_ref=_view(0).frame_ref, camera_id=CAMERA,
                view=_view(0), request=DetectionRequest(),
            )
        assert scheduler.depth == 0, "a failed batch must not leak its queue slot"

    async def test_mismatched_result_count_is_caught(self, clock) -> None:
        """Obligation D6, enforced at the executor boundary too."""
        async def wrong_length(key, batch):
            return []

        scheduler = DetectionScheduler(
            clock=clock, executor=wrong_length, max_batch_size=1,
            max_wait=Duration(0), queue_capacity=8,
            inference_timeout=Duration.from_millis(500),
        )
        from vision_os.core.errors import DetectorContractError

        with pytest.raises(DetectorContractError, match="1:1 and in order"):
            await scheduler.submit(
                key=_batch_key(), frame_ref=_view(0).frame_ref, camera_id=CAMERA,
                view=_view(0), request=DetectionRequest(),
            )

    async def test_rejects_invalid_construction(self, clock) -> None:
        async def noop(key, batch):
            return []

        with pytest.raises(ValueError, match="max_batch_size"):
            DetectionScheduler(clock=clock, executor=noop, max_batch_size=0)
        with pytest.raises(ValueError, match="queue_capacity"):
            DetectionScheduler(clock=clock, executor=noop, queue_capacity=0)


class TestDeviceWorker:
    async def test_normalises_adapter_exceptions(self, clock, scripted_detections) -> None:
        """Every framework failure becomes one typed error at this boundary."""
        class Exploding(ReferenceDetector):
            def detect(self, frames, request):
                raise RuntimeError("cuda kernel launch failed")

        worker = DeviceWorker(
            clock=clock,
            detector=Exploding(clock=clock, producible_classes=(ClassId("person"),)),
            device_id="cuda:0",
        )
        from vision_os.core.errors import DetectionFailedError

        with pytest.raises(DetectionFailedError, match="cuda:0"):
            await worker.execute([_view(0)], DetectionRequest())
        assert worker.stats.failures == 1

    async def test_batch_length_mismatch_is_caught(self, clock) -> None:
        class ShortBatch(ReferenceDetector):
            def detect(self, frames, request):
                return list(super().detect(frames, request))[:1]

        worker = DeviceWorker(
            clock=clock,
            detector=ShortBatch(clock=clock, producible_classes=(ClassId("person"),)),
            device_id="cpu",
        )
        from vision_os.core.errors import DetectorContractError

        with pytest.raises(DetectorContractError, match="1:1 and in order"):
            await worker.execute([_view(0), _view(1)], DetectionRequest())

    async def test_health_survives_a_raising_adapter(self, clock) -> None:
        """A component that cannot report its health is itself unhealthy."""
        class BadHealth(ReferenceDetector):
            def health(self):
                raise RuntimeError("health probe failed")

        worker = DeviceWorker(
            clock=clock,
            detector=BadHealth(clock=clock, producible_classes=(ClassId("person"),)),
            device_id="cpu",
        )
        report = worker.health()
        assert report.state.value == "degraded"

    async def test_warm_runs_off_the_event_loop(self, clock, detector) -> None:
        worker = DeviceWorker(clock=clock, detector=detector, device_id="cpu")
        await worker.warm()
        assert detector.health().state.value == "healthy"


class TestDetectionManagerGates:
    def _declaration(self, **overrides) -> DetectorDeclaration:
        defaults = {
            "detector_id": "reference-primary",
            "adapter_id": "detector.reference",
            "model_id": str(MODEL_ID),
            "model_version": "1.0.0",
            "artifact_uri": "mem://reference-1.0.0.bin",
            "artifact_hash": "",
            "mappings": (
                MappingEntryDeclaration("person", "person"),
                MappingEntryDeclaration("forklift", "vehicle.forklift"),
            ),
        }
        defaults.update(overrides)
        return DetectorDeclaration(**defaults)

    def _manager(self, clock, bus, metrics, models, taxonomy) -> DetectionManager:
        from vision_os.conformance import platform_registry

        plugins = PluginManager(
            clock=clock, bus=bus, metrics=metrics, conformance=platform_registry()
        )
        return DetectionManager(
            clock=clock, bus=bus, metrics=metrics, plugins=plugins,
            models=models, taxonomy=taxonomy,
        )

    def test_activation_runs_every_gate(
        self, clock, bus, metrics, models, taxonomy, artifacts, scripted_detections
    ) -> None:
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        subscription = bus.subscribe(["detection.detector_loaded"])

        manager.register(
            DetectorRegistration(
                declaration=self._declaration(artifact_hash=spec.artifact.expected_hash),
                factory=lambda d: ReferenceDetector(
                    clock=clock,
                    producible_classes=(ClassId("person"), ClassId("vehicle.forklift")),
                    script=_declared_script(),
                ),
            )
        )
        binding = manager.activate("reference-primary")

        assert binding.coverage.valid
        assert binding.model_handle.artifact_hash == spec.artifact.expected_hash
        assert subscription.drain()
        assert manager.binding().adapter_id == "detector.reference"

    def test_non_conforming_adapter_is_refused(
        self, clock, bus, metrics, models, taxonomy, artifacts
    ) -> None:
        """The gate that makes "swap any detector" a guarantee (invariant V3).

        The adapter emits boxes outside normalized space — the failure the kit
        exists to catch before a single real frame is processed.
        """
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(artifact_hash=spec.artifact.expected_hash),
                factory=lambda d: ReferenceDetector(
                    clock=clock,
                    producible_classes=(ClassId("person"), ClassId("vehicle.forklift")),
                    script=(
                        ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9),
                    ),
                    emit_out_of_range=True,
                ),
            )
        )
        with pytest.raises(ConformanceFailedError):
            manager.activate("reference-primary")

    def test_mapping_to_an_undefined_class_fails_at_load(
        self, clock, bus, metrics, models, taxonomy, artifacts
    ) -> None:
        """Not at first frame."""
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(
                    artifact_hash=spec.artifact.expected_hash,
                    mappings=(MappingEntryDeclaration("thing", "nonexistent"),),
                ),
                factory=lambda d: ReferenceDetector(
                    clock=clock, producible_classes=(ClassId("person"),)
                ),
            )
        )
        with pytest.raises(TaxonomyError, match="does not define"):
            manager.activate("reference-primary")

    def test_declaring_a_capability_the_mapping_cannot_yield_is_rejected(
        self, clock, bus, metrics, models, taxonomy, artifacts, scripted_detections
    ) -> None:
        """A capability the platform cannot deliver is worse than an absent one.

        The adapter is internally consistent — it emits only what it declares —
        so it passes conformance. What it cannot do is *translate* to
        ``container.tray``, because its mapping has no entry yielding one, and a
        consumer demanding that class would wait forever (invariant V8).
        """
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(
                    artifact_hash=spec.artifact.expected_hash,
                    mappings=(MappingEntryDeclaration("person", "person"),),
                ),
                factory=lambda d: ReferenceDetector(
                    clock=clock,
                    producible_classes=(ClassId("person"), ClassId("container.tray")),
                    script=(
                        ScriptedDetection(
                            ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.9
                        ),
                    ),
                ),
            )
        )
        with pytest.raises(TaxonomyError, match="cannot deliver|yields none"):
            manager.activate("reference-primary")

    def test_missing_mapping_is_rejected(
        self, clock, bus, metrics, models, taxonomy, artifacts
    ) -> None:
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(
                    artifact_hash=spec.artifact.expected_hash, mappings=()
                ),
                factory=lambda d: ReferenceDetector(
                    clock=clock, producible_classes=(ClassId("person"),)
                ),
            )
        )
        with pytest.raises(TaxonomyError, match="no taxonomy mapping"):
            manager.activate("reference-primary")

    def test_unregistered_detector_is_typed(
        self, clock, bus, metrics, models, taxonomy
    ) -> None:
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        with pytest.raises(DetectionError, match="not registered"):
            manager.activate("ghost")

    def test_disabled_detector_is_refused(
        self, clock, bus, metrics, models, taxonomy, artifacts
    ) -> None:
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(
                    artifact_hash=spec.artifact.expected_hash, enabled=False
                ),
                factory=lambda d: ReferenceDetector(
                    clock=clock, producible_classes=(ClassId("person"),)
                ),
            )
        )
        with pytest.raises(DetectionError, match="disabled"):
            manager.activate("reference-primary")

    def test_deactivate_publishes_and_releases(
        self, clock, bus, metrics, models, taxonomy, artifacts, scripted_detections
    ) -> None:
        spec = register_reference_model(models, artifacts)
        manager = self._manager(clock, bus, metrics, models, taxonomy)
        manager.register(
            DetectorRegistration(
                declaration=self._declaration(artifact_hash=spec.artifact.expected_hash),
                factory=lambda d: ReferenceDetector(
                    clock=clock,
                    producible_classes=(ClassId("person"), ClassId("vehicle.forklift")),
                    script=_declared_script(),
                ),
            )
        )
        manager.activate("reference-primary")
        subscription = bus.subscribe(["detection.detector_unloaded"])
        manager.deactivate("primary_detector")

        assert subscription.drain()
        with pytest.raises(DetectionError, match="no detector is bound"):
            manager.binding()


class TestDetectionRuntimeSeam:
    """The runtime is a firewall: nothing downstream may stop acquisition."""

    async def test_consumes_admitted_frames(self, clock, bus, metrics, health, engine, buffer) -> None:
        runtime = DetectionRuntime(
            clock=clock, bus=bus, metrics=metrics, health=health, engine=engine
        )
        await runtime.start()
        frame = _publish(buffer, clock, 0)
        await runtime.on_admitted(frame.frame_ref, Fidelity(640, 640))

        assert runtime.stats.frames_consumed == 1
        assert runtime.stats.detections_emitted == 2

    async def test_never_raises_even_when_the_engine_does(
        self, clock, bus, metrics, health
    ) -> None:
        """The contract says a consumer must not raise; this proves the guard."""
        class ExplodingEngine:
            async def detect(self, frame_ref, fidelity=None, **kwargs):
                raise RuntimeError("catastrophic")

            async def warm(self) -> None: ...

            def health(self):
                from vision_os.core.model.health import ComponentHealth, HealthState
                from vision_os.core.model.ids import ModuleId

                return ComponentHealth(
                    ModuleId("x"), HealthState.HEALTHY, clock.now()
                )

        runtime = DetectionRuntime(
            clock=clock, bus=bus, metrics=metrics, health=health,
            engine=ExplodingEngine(),
        )
        await runtime.start()
        await runtime.on_admitted(frame_ref(0), Fidelity(640, 640))
        assert runtime.stats.frames_failed == 1

    async def test_ignores_frames_before_start(
        self, clock, bus, metrics, health, engine
    ) -> None:
        runtime = DetectionRuntime(
            clock=clock, bus=bus, metrics=metrics, health=health, engine=engine
        )
        await runtime.on_admitted(frame_ref(0), Fidelity(640, 640))
        assert runtime.stats.frames_consumed == 0

    async def test_a_bad_sink_does_not_break_detection(
        self, clock, bus, metrics, health, engine, buffer
    ) -> None:
        def exploding_sink(detections):
            raise RuntimeError("consumer fault")

        runtime = DetectionRuntime(
            clock=clock, bus=bus, metrics=metrics, health=health,
            engine=engine, sink=exploding_sink,
        )
        await runtime.start()
        frame = _publish(buffer, clock, 0)
        await runtime.on_admitted(frame.frame_ref, Fidelity(640, 640))
        assert runtime.stats.frames_detected == 1


def _declared_script():
    """A script emitting only classes the detector declares it can produce."""
    return (
        ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.92),
        ScriptedDetection(ClassId("vehicle.forklift"), Box(0.5, 0.4, 0.8, 0.9), 0.71),
    )


def _batch_key():
    from vision_os.perception.detection import BatchKey

    return BatchKey(
        model_id=str(MODEL_ID),
        model_version="1.0.0",
        precision="fp32",
        inference_width=640,
        inference_height=640,
    )


def _result_for(view: FrameView):
    from vision_os.core.model.provenance import InferenceTiming, ModelMeta
    from vision_os.core.ports.detection import DetectionResult

    return DetectionResult(
        frame_ref=view.frame_ref,
        detections=(),
        model_meta=ModelMeta(
            model_id=MODEL_ID, model_version="1.0.0", artifact_hash="blake2b:x"
        ),
        timing=InferenceTiming(batch_size=1),
    )


def DetectionSection_with(base, **overrides):
    """Copy a config section with overrides, keeping it frozen."""
    from dataclasses import replace

    return replace(base, **overrides)
