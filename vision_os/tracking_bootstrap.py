"""Composition root for Flow 3 — the only module that names a tracker.

Every other module in the platform holds ``TrackerPort``. The mapping from a
configured name to a concrete implementation exists here and nowhere else, which
is what makes the claim *"the platform does not know ByteTrack exists"* checkable
rather than aspirational — and it is checked, by an architecture test that scans
every other module's AST for tracker identifiers.

Swapping ByteTrack for a transformer tracker is a change to one config value and
one entry in ``TRACKER_FACTORIES``. No platform module changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.tracking import TRACKER_FACTORIES, build_iou_tracker
from .bootstrap import VisionPlatform
from .core.errors import ConfigurationError, TrackingError
from .core.ports.tracking import TrackerPort
from .kernel.plugins.manifest import PortCatalogue
from .perception.tracking import (
    AssociationPolicy,
    LifecyclePolicy,
    TrackingEngine,
    TrackingManager,
    TrackingRuntime,
)


@dataclass(frozen=True, slots=True)
class TrackingLayer:
    """Everything Flow 3 assembled, for tests and operators to reach into."""

    manager: TrackingManager
    engine: TrackingEngine
    runtime: TrackingRuntime

    @property
    def tracker_id(self) -> str:
        binding = self.manager.binding
        return binding.tracker_id if binding else ""

    @property
    def is_fallback(self) -> bool:
        return self.manager.is_fallback


def build_lifecycle_policy(platform: VisionPlatform) -> LifecyclePolicy:
    """Translate configuration into the lifecycle bounds. Pure mapping."""
    settings = platform.config.tracking()
    return LifecyclePolicy(
        min_hits_to_confirm=settings.min_hits_to_confirm,
        max_coast_frames=settings.max_coast_frames,
        max_lost_frames=settings.max_lost_frames,
        max_age_frames=settings.max_age_frames,
        max_tracks_per_camera=settings.max_tracks_per_camera,
    )


def build_association_policy(platform: VisionPlatform) -> AssociationPolicy:
    settings = platform.config.tracking()
    return AssociationPolicy(
        iou_weight=settings.iou_weight,
        distance_weight=settings.distance_weight,
        scale_weight=settings.scale_weight,
        max_cost=settings.max_association_cost,
        min_iou=settings.iou_threshold,
        gate_multiplier=settings.gate_multiplier,
        ambiguity_margin=settings.ambiguity_margin,
    )


def tracker_factory(platform: VisionPlatform, tracker_id: str) -> TrackerPort:
    """Build the named tracker. **The only function that resolves a name.**

    Raises:
        ConfigurationError: the name is unknown. Failing at boot with the list of
            valid names beats failing at the first frame with a lookup error.
    """
    factory = TRACKER_FACTORIES.get(tracker_id)
    if factory is None:
        raise ConfigurationError(
            f"unknown tracker '{tracker_id}'; available: "
            f"{', '.join(sorted(TRACKER_FACTORIES))}"
        )
    settings = platform.config.tracking()
    return factory(
        lifecycle=build_lifecycle_policy(platform),
        association=build_association_policy(platform),
        config_revision=str(platform.config.revision()),
        history_length=settings.history_length,
    )


def build_tracking_layer(
    platform: VisionPlatform,
    *,
    tracker: TrackerPort | None = None,
    tracking_sink=None,
) -> TrackingLayer:
    """Assemble Flow 3 against an already-built platform.

    The configured tracker is gated by the conformance kit before activation. If
    it fails to build or fails the kit, the platform falls back to the
    always-available geometric tracker rather than starting without tracking —
    degrade, never die (invariant V9).

    Raises:
        TrackingError: tracking is not enabled, or no conformance kit is
            registered for P9. An ungated tracker is never activated, so a
            missing kit is fatal rather than a warning.
        ConfigurationError: appearance is enabled but no provider exists.
    """
    settings = platform.config.tracking()

    if not settings.enabled:
        raise TrackingError(
            "tracking.enabled is false; a site that does not want tracking should "
            "not build the layer rather than build one that produces nothing"
        )

    if platform.conformance.get(PortCatalogue.TRACKER) is None:
        raise TrackingError(
            "no conformance kit is registered for the tracker port; an adapter "
            "cannot be activated without one (invariant V3). Build the platform "
            "with conformance=platform_registry()."
        )

    if settings.appearance_enabled:
        # Loud rather than silent. Appearance embeddings are C2 biometric data
        # and no provider ships (12_SECURITY section 4.3); quietly running on
        # geometry instead would hide the gap from the operator who asked for it.
        raise ConfigurationError(
            "tracking.appearance_enabled is true but no EmbeddingPort provider "
            "is available. Appearance embeddings are C2 biometric data, disabled "
            "by default, and no provider ships with the platform."
        )

    manager = TrackingManager(
        metrics=platform.metrics,
        conformance=platform.conformance,
        fallback_factory=lambda: build_iou_tracker(
            config_revision=str(platform.config.revision()),
            history_length=settings.history_length,
        ),
        appearance_available=settings.appearance_enabled,
        require_deterministic=settings.require_deterministic,
    )

    # Construction is inside the guard as well as activation: an unknown tracker
    # name is exactly as survivable as one that fails its kit, and a deployment
    # should degrade rather than refuse to start over a typo.
    try:
        manager.load(tracker or tracker_factory(platform, settings.tracker_id))
    except Exception as exc:  # noqa: BLE001 - a bad tracker degrades, never blocks boot
        # 10_RELIABILITY section 7.3: the IoU fallback needs no weights and no
        # device, so this path cannot itself fail.
        manager.fall_back(f"{type(exc).__name__}: {exc}")

    engine = TrackingEngine(
        clock=platform.clock,
        bus=platform.bus,
        metrics=platform.metrics,
        manager=manager,
        config=settings,
    )
    runtime = TrackingRuntime(
        clock=platform.clock,
        metrics=platform.metrics,
        health=platform.health,
        engine=engine,
        config=settings,
        sink=tracking_sink,
    )
    return TrackingLayer(manager=manager, engine=engine, runtime=runtime)
