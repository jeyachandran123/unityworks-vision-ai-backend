"""Composition root for Flow 2 — assemble the detection layer.

Kept separate from ``bootstrap`` so that Flow 1 stays buildable, testable and
shippable with no knowledge that detection exists. A deployment that declares no
detectors never imports this module, and the platform behaves exactly as it did
before Flow 2 was written.

This is the **only** place where a concrete detector adapter is named. Everything
above it holds ports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.detection import YoloDetector
from .adapters.models import (
    CpuDeviceProvider,
    CudaDeviceProvider,
    LocalArtifactStore,
    ScriptedRuntime,
    UltralyticsRuntime,
)
from .bootstrap import VisionPlatform
from .conformance import ConformanceRegistry
from .core.errors import DetectionError
from .core.model.ids import AdapterId, ClassId, ModelId
from .core.model.taxonomy import (
    ClassStatus,
    GeometryKind,
    MappingEntry,
    TaxonomyClass,
    TaxonomyMapping,
    UnmappedPolicy,
)
from .core.model.timebase import Duration
from .core.ports.detection import DetectorPort
from .core.ports.models import ArtifactStorePort, DevicePort, ModelRuntimePort
from .kernel.config.schema import DetectorDeclaration
from .kernel.models import CalibrationRegistry, DeviceBroker, ModelManager
from .kernel.plugins import PortCatalogue
from .perception.detection import (
    DetectionEngine,
    DetectionManager,
    DetectionRuntime,
    DetectionScheduler,
    DetectorRegistration,
    DeviceWorker,
)
from .taxonomy import TaxonomyRegistry


@dataclass(slots=True)
class DetectionLayer:
    """Every constructed detection collaborator, exposed for test and operation."""

    taxonomy: TaxonomyRegistry
    broker: DeviceBroker
    models: ModelManager
    manager: DetectionManager
    scheduler: DetectionScheduler
    worker: DeviceWorker
    engine: DetectionEngine
    runtime: DetectionRuntime

    async def start(self) -> None:
        await self.runtime.start()

    async def stop(self) -> None:
        await self.runtime.stop()
        self.manager.close()
        self.models.close()


def build_taxonomy(platform: VisionPlatform) -> TaxonomyRegistry:
    """Load the declared Visual Taxonomy.

    Registered in ancestry order so declaration order in configuration does not
    matter — an orphaned ``vehicle.forklift`` would otherwise break every
    hierarchical query for ``vehicle``.
    """
    registry = TaxonomyRegistry()
    declarations = platform.config.taxonomy()
    registry.register_classes(
        tuple(
            TaxonomyClass(
                class_id=ClassId(declaration.class_id),
                taxonomy_version=registry.version,
                geometry_kinds=tuple(
                    GeometryKind(kind) for kind in declaration.geometry_kinds
                ),
                status=ClassStatus(declaration.status),
                superseded_by=(
                    ClassId(declaration.superseded_by)
                    if declaration.superseded_by
                    else None
                ),
                description=declaration.description,
            )
            for declaration in declarations
        )
    )
    return registry


def build_model_manager(
    platform: VisionPlatform,
    *,
    device_providers: Sequence[DevicePort] | None = None,
    artifacts: ArtifactStorePort | None = None,
    runtimes: Sequence[ModelRuntimePort] | None = None,
    calibration: CalibrationRegistry | None = None,
) -> tuple[DeviceBroker, ModelManager]:
    """Construct M18 with its device, artifact and runtime adapters.

    CUDA is offered but never required: a node without it enumerates CPU only and
    the broker falls back, so an accelerator-less edge box starts normally.
    """
    settings = platform.config.models()
    providers = device_providers or (CpuDeviceProvider(), CudaDeviceProvider())
    broker = DeviceBroker(
        providers,
        allow_cpu_fallback=settings.allow_cpu_fallback,
        headroom_fraction=settings.vram_headroom_fraction,
    )
    manager = ModelManager(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        broker=broker,
        artifacts=artifacts or LocalArtifactStore(settings.artifact_cache_dir),
        runtimes=runtimes or (UltralyticsRuntime(), ScriptedRuntime()),
        calibration=calibration,
        deployment_context=settings.deployment_context,
        warmup_enabled=settings.warmup_enabled,
    )
    return broker, manager


def yolo_factory(
    platform: VisionPlatform,
    models: ModelManager,
) -> Any:
    """Build a ``YoloDetector`` from a declaration.

    The only function in the codebase that names YOLO. Swapping in RT-DETR means
    writing a sibling factory and pointing configuration at it — no platform
    module changes (invariant V3).
    """

    def factory(declaration: DetectorDeclaration) -> DetectorPort:
        handle = models.acquire(
            ModelId(declaration.model_id),
            declaration.model_version,
            owner=f"detector-factory:{declaration.detector_id}",
            device_hint=(
                declaration.device_kind if declaration.device_kind != "cpu" else None
            ),
        )
        mapping = TaxonomyMapping(
            adapter_id=AdapterId(declaration.adapter_id),
            model_id=ModelId(declaration.model_id),
            entries=tuple(
                MappingEntry(
                    native_label=entry.native_label,
                    class_id=ClassId(entry.class_id),
                    mapping_confidence=entry.mapping_confidence,
                    notes=entry.notes,
                )
                for entry in declaration.mappings
            ),
            unmapped_policy=UnmappedPolicy(declaration.unmapped_policy),
            native_label_space=declaration.native_label_space,
        )
        return YoloDetector(
            clock=platform.clock,
            session=handle.session,  # type: ignore[arg-type]
            mapping=mapping,
            model_id=ModelId(declaration.model_id),
            model_version=declaration.model_version,
            artifact_hash=declaration.artifact_hash,
            device_id=handle.device_id,
            precision=declaration.precision,
            deterministic=True,
        )

    return factory


def build_detection_layer(
    platform: VisionPlatform,
    *,
    detector_factory,
    device_providers: Sequence[DevicePort] | None = None,
    artifacts: ArtifactStorePort | None = None,
    runtimes: Sequence[ModelRuntimePort] | None = None,
    calibration: CalibrationRegistry | None = None,
    conformance: ConformanceRegistry | None = None,
    taxonomy: TaxonomyRegistry | None = None,
    detection_sink=None,
    detection_consumer=None,
) -> DetectionLayer:
    """Assemble Flow 2 against an already-built Flow 1 platform.

    Raises:
        DetectionError: no detector is declared or enabled. Detection is opt-in;
            a site that declares none runs Flow 1 exactly as before.
    """
    settings = platform.config.detection()
    registry = taxonomy or build_taxonomy(platform)
    broker, models = build_model_manager(
        platform,
        device_providers=device_providers,
        artifacts=artifacts,
        runtimes=runtimes,
        calibration=calibration,
    )

    # The Plugin Manager refuses to activate an adapter for a port with no kit.
    # Fail here, loudly, rather than at activation with a confusing message.
    kits = conformance or platform.conformance
    if kits.get(PortCatalogue.DETECTOR) is None:
        raise DetectionError(
            "no conformance kit is registered for the detector port; an adapter "
            "cannot be activated without one (invariant V3). Build the platform "
            "with conformance=platform_registry()."
        )

    manager = DetectionManager(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        plugins=platform.plugins,
        models=models,
        taxonomy=registry,
    )
    declarations = [d for d in platform.config.detectors() if d.enabled]
    if not declarations:
        raise DetectionError(
            "detection is enabled but no enabled detector is declared; a site with "
            "no detector should leave detection.enabled false rather than run a "
            "layer that can produce nothing"
        )
    manager.register_all(
        DetectorRegistration(declaration=declaration, factory=detector_factory)
        for declaration in declarations
    )

    binding = manager.activate(declarations[0].detector_id)
    worker = DeviceWorker(
        clock=platform.clock,
        detector=binding.detector,
        device_id=binding.model_handle.device_id,
    )
    scheduler = DetectionScheduler(
        clock=platform.clock,
        executor=_make_executor(worker),
        max_batch_size=settings.max_batch_size,
        max_wait=Duration.from_millis(settings.batch_max_wait_ms),
        queue_capacity=settings.queue_capacity,
        inference_timeout=Duration.from_millis(settings.inference_timeout_ms),
    )
    engine = DetectionEngine(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        buffer=platform.buffer,
        camera_manager=platform.cameras,
        taxonomy=registry,
        binding=binding,
        scheduler=scheduler,
        worker=worker,
        config=settings,
        config_revision=str(platform.config.revision()),
        deterministic=platform.config.platform().deterministic,
    )
    runtime = DetectionRuntime(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        health=platform.health,
        engine=engine,
        sink=detection_sink,
        consumer=detection_consumer,
    )
    return DetectionLayer(
        taxonomy=registry,
        broker=broker,
        models=models,
        manager=manager,
        scheduler=scheduler,
        worker=worker,
        engine=engine,
        runtime=runtime,
    )


def _make_executor(worker: DeviceWorker):
    """Bridge the batch scheduler to a device worker.

    The scheduler knows nothing about devices and the worker knows nothing about
    batching; this closure is the only place the two meet.
    """

    async def execute(key, batch):
        views = [item.view for item in batch]
        request = batch[0].request
        return await worker.execute(views, request)

    return execute


__all__ = [
    "DetectionLayer",
    "build_detection_layer",
    "build_model_manager",
    "build_taxonomy",
    "yolo_factory",
]
