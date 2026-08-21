"""Fixtures for the Flow 2 detection suite.

Everything here builds real modules wired to dependency-free reference adapters,
so the whole detection pipeline runs in CI without a GPU, a model file, or a
network — while still exercising production code paths rather than stand-ins.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.detection import ReferenceDetector, ScriptedDetection
from vision_os.adapters.models import (
    CpuDeviceProvider,
    InMemoryArtifactStore,
    ScriptedRuntime,
    ScriptedSession,
    StaticDeviceProvider,
)
from vision_os.core.model.ids import (
    AdapterId,
    CameraId,
    ClassId,
    FrameRef,
    FrameSeq,
    ModelId,
    StreamEpoch,
)
from vision_os.core.model.space import Box
from vision_os.core.model.taxonomy import (
    GeometryKind,
    MappingEntry,
    TaxonomyClass,
    TaxonomyMapping,
    UnmappedPolicy,
)
from vision_os.core.ports.models import ArtifactRef, DeviceInfo, DeviceKind
from vision_os.kernel.config.schema import DetectionSection
from vision_os.kernel.models import (
    CalibrationRegistry,
    DeviceBroker,
    ModelManager,
    ModelSpec,
)
from vision_os.perception.detection import (
    DetectionScheduler,
    DetectorBinding,
    DeviceWorker,
)
from vision_os.taxonomy import TaxonomyRegistry

MODEL_ID = ModelId("reference-detector")
ADAPTER_ID = AdapterId("detector.reference")
CAMERA = CameraId("cam-01")


def frame_ref(seq: int = 0, camera: CameraId = CAMERA) -> FrameRef:
    return FrameRef(camera, StreamEpoch(1), FrameSeq(seq))


@pytest.fixture
def taxonomy() -> TaxonomyRegistry:
    """A small, deliberately domain-neutral taxonomy.

    ``person`` and ``vehicle.forklift`` are visual kinds any observer would name.
    No role, no judgment — those would be rejected at registration (invariant V1).
    """
    registry = TaxonomyRegistry()
    registry.register_classes(
        (
            TaxonomyClass(ClassId("person"), registry.version),
            TaxonomyClass(ClassId("vehicle"), registry.version),
            TaxonomyClass(ClassId("vehicle.forklift"), registry.version),
            TaxonomyClass(
                ClassId("container"),
                registry.version,
                geometry_kinds=(GeometryKind.BOX, GeometryKind.MASK),
            ),
            TaxonomyClass(ClassId("container.tray"), registry.version),
        )
    )
    return registry


@pytest.fixture
def mapping() -> TaxonomyMapping:
    return TaxonomyMapping(
        adapter_id=ADAPTER_ID,
        model_id=MODEL_ID,
        entries=(
            MappingEntry("person", ClassId("person")),
            MappingEntry("forklift", ClassId("vehicle.forklift")),
            MappingEntry("tray", ClassId("container.tray")),
        ),
        unmapped_policy=UnmappedPolicy.DROP,
        native_label_space="coco",
    )


@pytest.fixture
def detection_config() -> DetectionSection:
    """Batch wait of zero: deterministic mode requires fixed batch composition."""
    return DetectionSection(
        enabled=True,
        confidence_threshold=0.25,
        max_detections_per_frame=50,
        max_batch_size=4,
        batch_max_wait_ms=0,
        inference_timeout_ms=1_000,
        queue_capacity=16,
        apply_platform_nms=True,
    )


@pytest.fixture
def scripted_detections() -> tuple[ScriptedDetection, ...]:
    return (
        ScriptedDetection(ClassId("person"), Box(0.1, 0.1, 0.3, 0.6), 0.92),
        ScriptedDetection(ClassId("vehicle.forklift"), Box(0.5, 0.4, 0.8, 0.9), 0.71),
        ScriptedDetection(ClassId("container.tray"), Box(0.2, 0.7, 0.35, 0.8), 0.10),
    )


@pytest.fixture
def detector(clock, scripted_detections) -> ReferenceDetector:
    return ReferenceDetector(
        clock=clock,
        producible_classes=(
            ClassId("person"),
            ClassId("vehicle.forklift"),
            ClassId("container.tray"),
        ),
        script=scripted_detections,
        model_id=MODEL_ID,
    )


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    return ArtifactRef(uri="mem://reference.bin", expected_hash="")


@pytest.fixture
def artifacts() -> InMemoryArtifactStore:
    return InMemoryArtifactStore()


@pytest.fixture
def cpu_devices() -> CpuDeviceProvider:
    return CpuDeviceProvider()


@pytest.fixture
def gpu_devices() -> StaticDeviceProvider:
    """Two synthetic accelerators, removable mid-test."""
    return StaticDeviceProvider(
        (
            DeviceInfo("cuda:0", DeviceKind.CUDA, 0, 8 * 1024**3, "synthetic-0"),
            DeviceInfo("cuda:1", DeviceKind.CUDA, 1, 8 * 1024**3, "synthetic-1"),
        )
    )


@pytest.fixture
def broker(cpu_devices, gpu_devices) -> DeviceBroker:
    return DeviceBroker((cpu_devices, gpu_devices), allow_cpu_fallback=True)


@pytest.fixture
def cpu_only_broker(cpu_devices) -> DeviceBroker:
    return DeviceBroker((cpu_devices,), allow_cpu_fallback=True)


@pytest.fixture
def runtime_adapter() -> ScriptedRuntime:
    return ScriptedRuntime(vram_bytes=1024**3)


@pytest.fixture
def calibration() -> CalibrationRegistry:
    return CalibrationRegistry()


@pytest.fixture
def models(clock, bus, metrics, broker, artifacts, runtime_adapter, calibration):
    return ModelManager(
        clock=clock,
        bus=bus,
        metrics=metrics,
        broker=broker,
        artifacts=artifacts,
        runtimes=(runtime_adapter,),
        calibration=calibration,
    )


def register_reference_model(
    manager: ModelManager,
    artifacts: InMemoryArtifactStore,
    *,
    version: str = "1.0.0",
    vram_bytes: int = 1024**3,
    device_kind: str = "cpu",
    permitted_contexts: tuple[str, ...] = (),
) -> ModelSpec:
    digest = artifacts.put(f"mem://reference-{version}.bin", b"reference-weights")
    spec = ModelSpec(
        model_id=MODEL_ID,
        version=version,
        artifact=ArtifactRef(
            uri=f"mem://reference-{version}.bin", expected_hash=digest
        ),
        vram_bytes=vram_bytes,
        device_kind=device_kind,
        permitted_contexts=permitted_contexts,
    )
    manager.register(spec)
    return spec


@pytest.fixture
def binding(clock, detector, mapping, taxonomy, models, artifacts) -> DetectorBinding:
    register_reference_model(models, artifacts)
    coverage = taxonomy.register_mapping(mapping)
    handle = models.acquire(MODEL_ID, "1.0.0", owner="test")
    return DetectorBinding(
        adapter_id=ADAPTER_ID,
        adapter_version="1.0.0",
        detector=detector,
        capabilities=detector.capabilities(),
        model_handle=handle,
        mapping=mapping,
        coverage=coverage,
    )


@pytest.fixture
def worker(clock, detector) -> DeviceWorker:
    return DeviceWorker(clock=clock, detector=detector, device_id="cpu")


@pytest.fixture
def detection_scheduler(clock, worker, detection_config) -> DetectionScheduler:
    from vision_os.core.model.timebase import Duration

    async def execute(key, batch):
        return await worker.execute([item.view for item in batch], batch[0].request)

    return DetectionScheduler(
        clock=clock,
        executor=execute,
        max_batch_size=detection_config.max_batch_size,
        max_wait=Duration.from_millis(detection_config.batch_max_wait_ms),
        queue_capacity=detection_config.queue_capacity,
        inference_timeout=Duration.from_millis(detection_config.inference_timeout_ms),
    )


@pytest.fixture
def scripted_session() -> ScriptedSession:
    return ScriptedSession(names=("person", "car", "forklift"))
