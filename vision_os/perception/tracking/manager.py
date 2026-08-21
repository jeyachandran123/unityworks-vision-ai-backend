"""Tracker binding, gating, and degradation.

> **Single responsibility:** *Decide which tracker is active and guarantee one
> always is. Track nothing yourself.*

Two gates run before any tracker becomes reachable, in cost order:

1. **Compatibility** — does the adapter satisfy ``TrackerPort``, and does it
   declare capabilities the deployment can honour? A tracker requiring
   embeddings with no provider configured fails here rather than degrading
   silently to geometry, because a silent downgrade makes a capability gap
   invisible (invariant V8).
2. **Conformance** — the fast subset of ``TRACKER_KIT`` runs against the live
   adapter. **An adapter failing the kit never becomes reachable**: it is not
   loaded in a degraded mode, the binding is simply never installed.

And one guarantee holds afterwards: **there is always a working tracker.** The
geometric IoU fallback needs no weights and no device, so ``fall_back()`` cannot
itself fail (10_RELIABILITY section 7.3). Tracking degrades in accuracy, never
in availability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...conformance.kit import ConformanceRegistry
from ...core.errors import (
    ConformanceFailedError,
    EmbeddingUnavailableError,
    PortIncompatibleError,
)
from ...core.model.ids import ModuleId
from ...core.ports.tracking import TrackerCapabilities, TrackerPort
from ...kernel.metrics import MetricName, MetricsEngine
from ...kernel.plugins.manifest import PortCatalogue

TRACKING_MANAGER_ID = ModuleId("tracking_manager")

TrackerFactory = Callable[[], TrackerPort]


@dataclass(frozen=True, slots=True)
class TrackerBinding:
    """An activated tracker and what it declared about itself."""

    tracker: TrackerPort
    capabilities: TrackerCapabilities
    is_fallback: bool = False

    @property
    def tracker_id(self) -> str:
        return self.capabilities.tracker_id


class TrackingManager:
    """Owns the active tracker binding for the platform."""

    def __init__(
        self,
        *,
        metrics: MetricsEngine,
        conformance: ConformanceRegistry,
        fallback_factory: TrackerFactory,
        appearance_available: bool = False,
        require_deterministic: bool = False,
    ) -> None:
        self._metrics = metrics
        self._conformance = conformance
        self._fallback_factory = fallback_factory
        self._appearance_available = appearance_available
        self._require_deterministic = require_deterministic

        self._binding: TrackerBinding | None = None
        self._fallback_reason = ""

    # --- activation ----------------------------------------------------------- #

    def load(self, tracker: TrackerPort) -> TrackerBinding:
        """Gate and activate a tracker.

        Raises:
            PortIncompatibleError: the adapter does not satisfy the port, or
                declares capabilities the deployment cannot honour.
            EmbeddingUnavailableError: it requires embeddings and none exist.
            ConformanceFailedError: it failed the kit. **Never activated.**
        """
        capabilities = self._check_compatibility(tracker)
        self._check_conformance(tracker)

        binding = TrackerBinding(tracker=tracker, capabilities=capabilities)
        self._binding = binding
        self._fallback_reason = ""
        self._metrics.gauge(MetricName.TRACKERS_ACTIVE).set(1.0)
        return binding

    def _check_compatibility(self, tracker: TrackerPort) -> TrackerCapabilities:
        if not isinstance(tracker, TrackerPort):
            raise PortIncompatibleError(
                f"{type(tracker).__name__} does not satisfy TrackerPort",
                port_id=str(PortCatalogue.TRACKER),
            )
        capabilities = tracker.capabilities()

        if capabilities.requires_embeddings and not self._appearance_available:
            raise EmbeddingUnavailableError(
                f"tracker '{capabilities.tracker_id}' requires appearance embeddings "
                f"but no provider is configured. Refusing rather than silently "
                f"degrading to geometry: a capability gap must be visible, not "
                f"inferred from worse results (invariant V8). Note that appearance "
                f"embeddings are C2 biometric data, disabled by default.",
                tracker_id=capabilities.tracker_id,
            )

        if self._require_deterministic and not capabilities.deterministic:
            raise PortIncompatibleError(
                f"tracker '{capabilities.tracker_id}' declares itself "
                f"non-deterministic and deterministic mode is required "
                f"(invariant V13)",
                port_id=str(PortCatalogue.TRACKER),
            )
        return capabilities

    def _check_conformance(self, tracker: TrackerPort) -> None:
        kit = self._conformance.get(PortCatalogue.TRACKER)
        if kit is None:
            # Fail loudly rather than activating an ungated adapter. A missing
            # kit is a wiring bug, and treating it as "no checks required" is
            # how the gate quietly stops being a gate.
            raise ConformanceFailedError(
                "no conformance kit registered for P9.TrackerPort; refusing to "
                "activate an ungated tracker",
                port_id=str(PortCatalogue.TRACKER),
            )
        report = kit.run(tracker, fast_only=True)
        if not report.passed:
            self._metrics.counter(MetricName.CONFORMANCE_FAILURES).increment()
            raise ConformanceFailedError(
                f"tracker failed conformance: {'; '.join(report.failures)}",
                port_id=str(PortCatalogue.TRACKER),
                failures=report.failures,
            )

    # --- degradation ---------------------------------------------------------- #

    def fall_back(self, reason: str) -> TrackerBinding:
        """Replace the active tracker with the always-available fallback.

        Idempotent: repeated failures from an already-degraded tracker do not
        churn the binding. The fallback is pure geometry with no weights and no
        device, so this path cannot itself fail — which is what makes "degrade,
        never die" true rather than aspirational (V9).
        """
        if self._binding is not None and self._binding.is_fallback:
            return self._binding

        tracker = self._fallback_factory()
        binding = TrackerBinding(
            tracker=tracker, capabilities=tracker.capabilities(), is_fallback=True
        )
        self._binding = binding
        self._fallback_reason = reason
        self._metrics.counter(MetricName.TRACKER_FALLBACKS).increment()
        return binding

    def unload(self) -> None:
        self._binding = None
        self._metrics.gauge(MetricName.TRACKERS_ACTIVE).set(0.0)

    # --- access ---------------------------------------------------------------- #

    @property
    def tracker(self) -> TrackerPort:
        if self._binding is None:
            raise PortIncompatibleError(
                "no tracker is bound; the tracking layer was used before load()",
                port_id=str(PortCatalogue.TRACKER),
            )
        return self._binding.tracker

    @property
    def binding(self) -> TrackerBinding | None:
        return self._binding

    @property
    def capabilities(self) -> TrackerCapabilities | None:
        return self._binding.capabilities if self._binding else None

    @property
    def is_loaded(self) -> bool:
        return self._binding is not None

    @property
    def is_fallback(self) -> bool:
        return self._binding is not None and self._binding.is_fallback

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason
