"""The Detection Manager — detector lifecycle.

Single responsibility: *load, validate, activate and retire detectors. Run no
inference.*

Every activation passes four gates before a single frame reaches the adapter:

1. **Compatibility** — port and platform version ranges (Plugin Manager).
2. **Conformance** — ``kit.detector``'s fast subset. An adapter that gets a
   coordinate convention or a fail-closed path wrong is rejected here rather than
   discovered months later in production data (invariant V3).
3. **Taxonomy validation** — a mapping naming a class the taxonomy does not
   define fails at load, not at first frame.
4. **Capability declaration** — what the detector can produce is published, so a
   consumer asking for something it cannot produce gets an explicit gap rather
   than silence (invariant V8).

Only then does the Model Manager grant a handle and the detector become bindable.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...core.errors import DetectionError, PluginError, TaxonomyError
from ...core.model.ids import AdapterId, ClassId, ModelId, PluginId
from ...core.model.taxonomy import (
    MappingEntry,
    TaxonomyMapping,
    UnmappedPolicy,
)
from ...core.ports.clock import Clock
from ...core.ports.detection import DetectorPort
from ...core.ports.models import ArtifactRef
from ...kernel.config.schema import DetectorDeclaration
from ...kernel.events import DetectorLoaded, DetectorUnloaded, EventBus
from ...kernel.metrics import MetricName, MetricsEngine
from ...kernel.models.manager import ModelManager, ModelSpec
from ...kernel.plugins import PluginDescriptor, PluginManager, PluginManifest, PortCatalogue
from ...kernel.plugins.manifest import VersionRange
from ...taxonomy import TaxonomyRegistry
from .binding import DetectorBinding

DetectorFactory = Callable[[DetectorDeclaration], DetectorPort]
"""Constructs an adapter from its declaration.

Dependency injection at the composition root: the manager never imports a
concrete adapter, so no platform module knows YOLO exists.
"""


@dataclass(frozen=True, slots=True)
class DetectorRegistration:
    """A declared detector plus the factory that builds its adapter."""

    declaration: DetectorDeclaration
    factory: DetectorFactory


class DetectionManager:
    """Owns which detectors exist and which is bound to each role."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        plugins: PluginManager,
        models: ModelManager,
        taxonomy: TaxonomyRegistry,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._plugins = plugins
        self._models = models
        self._taxonomy = taxonomy
        self._lock = threading.RLock()
        self._registrations: dict[str, DetectorRegistration] = {}
        self._bindings: dict[str, DetectorBinding] = {}

    # --- registration ------------------------------------------------------------ #

    def register(self, registration: DetectorRegistration) -> None:
        """Declare a detector. Does not load, validate, or activate it."""
        with self._lock:
            self._registrations[registration.declaration.detector_id] = registration

    def register_all(self, registrations: Sequence[DetectorRegistration]) -> None:
        for registration in registrations:
            self.register(registration)

    # --- activation --------------------------------------------------------------- #

    def activate(self, detector_id: str) -> DetectorBinding:
        """Run every gate, then bind the detector to its role.

        Raises:
            DetectionError: the detector is not registered or is disabled.
            ConformanceFailedError: the adapter failed ``kit.detector``. It is
                not activated, and no frame ever reaches it.
            TaxonomyError: the mapping names an undefined class.
            ModelUnavailableError / ModelLoadError: the artifact will not load.
        """
        with self._lock:
            registration = self._registrations.get(detector_id)
        if registration is None:
            raise DetectionError(
                f"detector '{detector_id}' is not registered", detector_id=detector_id
            )
        declaration = registration.declaration
        if not declaration.enabled:
            raise DetectionError(
                f"detector '{detector_id}' is disabled in configuration",
                detector_id=detector_id,
            )

        mapping = self._build_mapping(declaration)
        coverage = self._taxonomy.register_mapping(mapping)

        model_id = self._register_model(declaration)
        plugin_id = self._load_plugin(registration)
        detector = self._plugins.activate(plugin_id)

        capabilities = detector.capabilities()
        self._verify_capability_against_mapping(declaration, capabilities, coverage)

        handle = self._models.acquire(
            model_id,
            declaration.model_version,
            owner=f"detector:{detector_id}",
            device_hint=(
                declaration.device_kind if declaration.device_kind != "cpu" else None
            ),
        )

        binding = DetectorBinding(
            adapter_id=AdapterId(declaration.adapter_id),
            adapter_version=declaration.model_version,
            detector=detector,
            capabilities=capabilities,
            model_handle=handle,
            mapping=mapping,
            coverage=coverage,
            role=declaration.role,
            calibration=self._models.calibration(model_id, declaration.model_version),
        )

        with self._lock:
            self._bindings[declaration.role] = binding
        self._models.pin(declaration.role, model_id, declaration.model_version)

        self._metrics.counter(
            MetricName.DETECTORS_ACTIVE, adapter_id=declaration.adapter_id
        ).increment()
        self._bus.publish(
            DetectorLoaded(
                occurred_at=self._clock.now(),
                partition_key=declaration.role,
                adapter_id=declaration.adapter_id,
                model_id=declaration.model_id,
                producible_classes=len(capabilities.producible_classes),
            )
        )
        return binding

    def deactivate(self, role: str, reason: str = "requested") -> None:
        """Release a role's detector and its model handle."""
        with self._lock:
            binding = self._bindings.pop(role, None)
        if binding is None:
            return
        self._models.release(binding.model_handle)
        self._metrics.counter(
            MetricName.DETECTORS_ACTIVE, adapter_id=str(binding.adapter_id)
        ).increment(-1)
        self._bus.publish(
            DetectorUnloaded(
                occurred_at=self._clock.now(),
                partition_key=role,
                adapter_id=str(binding.adapter_id),
                reason=reason,
            )
        )

    def swap(self, role: str, detector_id: str) -> DetectorBinding:
        """Replace a role's detector.

        On failure the incumbent stays bound: a half-applied swap is worse than
        an outdated detector, because the platform would be left with no detector
        at all for a role that had a working one.
        """
        with self._lock:
            incumbent = self._bindings.get(role)
        try:
            binding = self.activate(detector_id)
        except Exception:
            if incumbent is not None:
                with self._lock:
                    self._bindings[role] = incumbent
            raise
        if incumbent is not None and incumbent is not binding:
            self._models.release(incumbent.model_handle)
        return binding

    # --- lookup -------------------------------------------------------------------- #

    def binding(self, role: str = "primary_detector") -> DetectorBinding:
        with self._lock:
            binding = self._bindings.get(role)
        if binding is None:
            raise DetectionError(f"no detector is bound to role '{role}'", role=role)
        return binding

    def try_binding(self, role: str = "primary_detector") -> DetectorBinding | None:
        with self._lock:
            return self._bindings.get(role)

    def bindings(self) -> tuple[DetectorBinding, ...]:
        with self._lock:
            return tuple(self._bindings.values())

    def producible_classes(self) -> tuple[ClassId, ...]:
        """Every class any bound detector can produce, for gap reporting."""
        produced: list[ClassId] = []
        for binding in self.bindings():
            for class_id in binding.capabilities.producible_classes:
                if class_id not in produced:
                    produced.append(class_id)
        return tuple(produced)

    def capability_gap(self, requested: Sequence[ClassId]) -> tuple[ClassId, ...]:
        producible = self.producible_classes()
        return tuple(
            class_id
            for class_id in requested
            if not any(
                produced == class_id or produced.startswith(f"{class_id}.")
                for produced in producible
            )
        )

    def close(self) -> None:
        for role in tuple(self._bindings):
            self.deactivate(role, reason="shutdown")

    # --- internals ------------------------------------------------------------------ #

    def _build_mapping(self, declaration: DetectorDeclaration) -> TaxonomyMapping:
        if not declaration.mappings:
            raise TaxonomyError(
                f"detector '{declaration.detector_id}' declares no taxonomy mapping; "
                f"without one its native labels could not be translated and would "
                f"leak into the platform (obligation D2)",
                detector_id=declaration.detector_id,
            )
        return TaxonomyMapping(
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

    def _register_model(self, declaration: DetectorDeclaration) -> ModelId:
        spec = ModelSpec(
            model_id=ModelId(declaration.model_id),
            version=declaration.model_version,
            artifact=ArtifactRef(
                uri=declaration.artifact_uri, expected_hash=declaration.artifact_hash
            ),
            precision=declaration.precision,
            device_kind=declaration.device_kind,
            vram_bytes=declaration.vram_bytes,
            licence=declaration.licence,
            permitted_contexts=declaration.permitted_contexts,
            calibration_id=declaration.calibration_id,
            runtime_options=dict(declaration.runtime_options),
        )
        return self._models.register(spec)

    def _load_plugin(self, registration: DetectorRegistration) -> PluginId:
        """Register and load the adapter through the Plugin Manager.

        Routed through the kernel rather than instantiated directly so the
        conformance gate cannot be bypassed — that gate is the whole reason
        "swap any detector" is a guarantee rather than a hope.
        """
        declaration = registration.declaration
        plugin_id = PluginId(declaration.detector_id)
        manifest = PluginManifest(
            plugin_id=plugin_id,
            version=declaration.model_version,
            port_id=PortCatalogue.DETECTOR,
            port_version_range=VersionRange.parse(">=1.0 <2.0"),
            platform_range=VersionRange.parse(">=1.0 <2.0"),
            capabilities={
                "adapter_id": declaration.adapter_id,
                "model_id": declaration.model_id,
                "precision": declaration.precision,
            },
        )
        self._plugins.register(
            PluginDescriptor(
                manifest=manifest,
                factory=lambda: registration.factory(declaration),
            )
        )
        try:
            self._plugins.load(plugin_id)
        except PluginError:
            self._metrics.counter(
                MetricName.PLUGINS_REJECTED, port=str(PortCatalogue.DETECTOR)
            ).increment()
            raise
        return plugin_id

    def _verify_capability_against_mapping(
        self,
        declaration: DetectorDeclaration,
        capabilities,
        coverage,
    ) -> None:
        """The adapter's declared classes must be a subset of its mapping's.

        An adapter claiming to produce a class its own mapping cannot yield would
        make the published capability surface a lie, and a consumer would wait
        forever for data that can never arrive (invariant V8).
        """
        undeclared = [
            class_id
            for class_id in capabilities.producible_classes
            if class_id not in coverage.producible
        ]
        if undeclared:
            raise TaxonomyError(
                f"detector '{declaration.detector_id}' declares it can produce "
                f"{sorted(undeclared)}, but its taxonomy mapping yields none of them. "
                f"A capability the platform cannot deliver is worse than an absent "
                f"one (invariant V8).",
                detector_id=declaration.detector_id,
            )
