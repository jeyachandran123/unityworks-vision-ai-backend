"""Composition root for Flow 4 — the Object Registry.

Assembles M7 against an already-built platform. The registry holds
``ObjectStorePort`` and (optionally) ``IdentityResolverPort``; which
implementation satisfies either is decided only here.

**No identity resolver is constructible.** ``15_ROADMAP`` section 3 states P11
has no implementations in Phase 1, so there is no factory table to select from —
the parameter exists for a Phase 2 adapter and defaults to ``None``, which is the
supported configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters.registry import FileObjectStore, InMemoryObjectStore
from .bootstrap import VisionPlatform
from .core.errors import RegistryError
from .core.model.ids import ConfigRevision, ModuleId, SiteId, TenantId
from .core.model.provenance import Provenance
from .core.model.timebase import Duration
from .core.ports.registry import IdentityResolverPort, ObjectStorePort
from .kernel.plugins.manifest import PortCatalogue
from .perception.registry import (
    AttributeRegistry,
    BindingPolicy,
    LifecyclePolicy,
    ObjectRegistry,
    RegistryRuntime,
)

REGISTRY_MODULE = ModuleId("object_registry")
REGISTRY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RegistryLayer:
    """Everything Flow 4 assembled, for tests and operators to reach into."""

    registry: ObjectRegistry
    runtime: RegistryRuntime
    store: ObjectStorePort

    @property
    def store_id(self) -> str:
        return self.store.store_id


def build_lifecycle_policy(platform: VisionPlatform) -> LifecyclePolicy:
    """Translate configuration into lifecycle horizons. Pure mapping."""
    settings = platform.config.registry()
    return LifecyclePolicy(
        min_observations_to_confirm=settings.min_observations_to_confirm,
        provisional_horizon=Duration.from_millis(settings.provisional_horizon_ms),
        occlusion_horizon=Duration.from_millis(settings.occlusion_horizon_ms),
        dormant_horizon=Duration.from_millis(settings.dormant_horizon_ms),
        retention_horizon=Duration.from_millis(settings.retention_horizon_ms),
        max_objects_per_camera=settings.max_objects_per_camera,
    )


def build_binding_policy(platform: VisionPlatform) -> BindingPolicy:
    settings = platform.config.registry()
    return BindingPolicy(
        max_reentry_distance=settings.max_reentry_distance,
        max_reentry_gap=Duration.from_millis(settings.max_reentry_gap_ms),
        ambiguity_margin=settings.ambiguity_margin,
        min_binding_confidence=settings.min_binding_confidence,
        epoch_rebind_penalty=settings.epoch_rebind_penalty,
        class_must_match=settings.class_must_match,
    )


def build_object_store(platform: VisionPlatform) -> ObjectStorePort:
    """Select a durable store.

    A file store when persistence is enabled and a cache directory exists; an
    in-memory one otherwise. The in-memory store is not a fallback for failure —
    it is the honest implementation for a deployment that accepts session-scoped
    identity, and it says so through ``store_id``.
    """
    settings = platform.config.registry()
    if not settings.persistence_enabled:
        return InMemoryObjectStore()
    models = platform.config.models()
    return FileObjectStore(Path(models.artifact_cache_dir).parent / "objects")


def build_registry_layer(
    platform: VisionPlatform,
    *,
    store: ObjectStorePort | None = None,
    resolver: IdentityResolverPort | None = None,
    attributes: AttributeRegistry | None = None,
    registry_sink=None,
) -> RegistryLayer:
    """Assemble Flow 4 against an already-built platform.

    Args:
        resolver: An optional ``IdentityResolverPort``. **No implementation ships
            in Phase 1**; the parameter exists so a Phase 2 adapter attaches
            without touching this function.

    Raises:
        RegistryError: the registry is disabled, or no conformance kit is
            registered for the object store. An ungated store is never used.
    """
    settings = platform.config.registry()

    if not settings.enabled:
        raise RegistryError(
            "registry.enabled is false; a site that does not want canonical "
            "objects should not build the layer rather than build one that owns "
            "nothing"
        )

    if platform.conformance.get(PortCatalogue.STATE_STORE) is None:
        raise RegistryError(
            "no conformance kit is registered for the object store port; an "
            "adapter cannot be activated without one (invariant V3). Build the "
            "platform with conformance=platform_registry()."
        )

    if resolver is not None:
        # Refuse rather than silently accept: P11 has no implementations in
        # Phase 1, and a resolver appearing here means either a Phase 2 adapter
        # arrived early or something is being passed that should not be.
        raise RegistryError(
            f"an IdentityResolverPort ('{resolver.resolver_id}') was supplied, but "
            f"P11 has no implementations in Phase 1 (15_ROADMAP section 3). "
            f"Cross-camera identity is classified C2 and policy-gated."
        )

    platform_config = platform.config.platform()
    tenant_id = TenantId(getattr(platform_config, "tenant_id", "") or _first_tenant(platform))
    site_id = SiteId(getattr(platform_config, "site_id", "") or _first_site(platform))

    provenance = Provenance(
        producer_module=REGISTRY_MODULE,
        producer_version=REGISTRY_VERSION,
        config_revision=ConfigRevision(str(platform.config.revision())),
        deterministic=platform_config.deterministic,
    )

    selected_store = store or build_object_store(platform)
    kit = platform.conformance.get(PortCatalogue.STATE_STORE)
    report = kit.run(selected_store, fast_only=True)
    if not report.passed:
        raise RegistryError(
            f"object store failed conformance: {'; '.join(report.failures)}"
        )

    registry = ObjectRegistry(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        config=settings,
        tenant_id=tenant_id,
        site_id=site_id,
        provenance=provenance,
        lifecycle=build_lifecycle_policy(platform),
        binding=build_binding_policy(platform),
        attributes=attributes or AttributeRegistry(),
        resolver=None,
    )
    runtime = RegistryRuntime(
        clock=platform.clock,
        metrics=platform.metrics,
        health=platform.health,
        registry=registry,
        config=settings,
        store=selected_store,
        sink=registry_sink,
    )
    return RegistryLayer(registry=registry, runtime=runtime, store=selected_store)


def _first_tenant(platform: VisionPlatform) -> str:
    """Tenancy comes from the camera declarations, not from a global setting.

    A node can serve several tenants; the registry stamps each object with the
    tenancy of the camera that produced it. This is only the partition-level
    default for a deployment that declared exactly one.
    """
    cameras = platform.config.cameras()
    return cameras[0].tenant_id if cameras else "unset"


def _first_site(platform: VisionPlatform) -> str:
    cameras = platform.config.cameras()
    return cameras[0].site_id if cameras else "unset"
