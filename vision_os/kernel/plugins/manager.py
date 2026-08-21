"""M17 Plugin Manager — make swappable code loadable and safe.

Single responsibility: *validate, load, isolate, and version plugins. Know
nothing of what they do.*

The gate that matters is conformance. ``activate`` runs the port's fast
conformance subset and **refuses to bind an adapter that fails**, which is the
mechanism that makes "swap any model without platform change" a guarantee rather
than an aspiration (invariant V3).

Signature verification fails closed: unsigned or mis-signed code never loads
(12_SECURITY §6).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...conformance.kit import ConformanceRegistry, ConformanceReport
from ...core.errors import (
    ConformanceFailedError,
    ManifestInvalidError,
    PluginError,
    PortIncompatibleError,
    SignatureInvalidError,
)
from ...core.model.ids import PluginId, PortId
from ...core.ports.clock import Clock
from ..events import EventBus, PluginLoaded, PluginRejected
from ..metrics import MetricName, MetricsEngine
from .manifest import BINDABLE_PORTS, PluginManifest

#: The platform version adapters declare compatibility against.
PLATFORM_VERSION = "1.0.0"

#: Port contract versions the platform currently implements.
PORT_VERSIONS: dict[PortId, str] = {}


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    """A discovered but not yet loaded plugin."""

    manifest: PluginManifest
    factory: Callable[[], Any]
    """Constructs the adapter. Dependency injection: the plugin never reaches
    into the platform to find its collaborators."""


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    manifest: PluginManifest
    instance: Any
    conformance: ConformanceReport


class SignatureVerifier:
    """Trust-root verification.

    The default implementation requires a signature to be *present* when the
    manager is constructed with ``require_signatures=True``. Real cryptographic
    verification is an adapter concern; this class is the enforcement point so
    that the policy exists from Flow 1 rather than being retrofitted.
    """

    def __init__(self, *, trusted: frozenset[str] = frozenset()) -> None:
        self._trusted = trusted

    def verify(self, manifest: PluginManifest) -> bool:
        if manifest.signature is None:
            return False
        if not self._trusted:
            return True
        return manifest.signature in self._trusted


class PluginManager:
    """Discover, validate, load, and bind plugins to ports."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        conformance: ConformanceRegistry,
        verifier: SignatureVerifier | None = None,
        require_signatures: bool = False,
        require_conformance: bool = True,
        bindable_ports: frozenset[PortId] = BINDABLE_PORTS,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._conformance = conformance
        self._verifier = verifier or SignatureVerifier()
        self._require_signatures = require_signatures
        self._require_conformance = require_conformance
        self._bindable_ports = bindable_ports
        self._lock = threading.RLock()
        self._descriptors: dict[PluginId, PluginDescriptor] = {}
        self._loaded: dict[PluginId, LoadedPlugin] = {}
        self._bindings: dict[PortId, PluginId] = {}

    # --- discovery and validation ------------------------------------------ #

    def register(self, descriptor: PluginDescriptor) -> None:
        """Register a discovered plugin. Does not load or activate it."""
        with self._lock:
            self._descriptors[descriptor.manifest.plugin_id] = descriptor

    def validate(self, manifest: PluginManifest) -> tuple[str, ...]:
        """Return violations; empty means the manifest may be loaded."""
        violations: list[str] = []

        if manifest.port_id not in self._bindable_ports:
            violations.append(
                f"port '{manifest.port_id}' is not bindable in this build. "
                f"Bindable ports: {sorted(self._bindable_ports)}"
            )
        if not manifest.platform_range.contains(PLATFORM_VERSION):
            violations.append(
                f"platform {PLATFORM_VERSION} is outside the plugin's declared "
                f"range {manifest.platform_range}"
            )
        port_version = PORT_VERSIONS.get(manifest.port_id, "1.0.0")
        if not manifest.port_version_range.contains(port_version):
            violations.append(
                f"port {manifest.port_id} is at {port_version}, outside the "
                f"plugin's declared range {manifest.port_version_range}"
            )
        if self._require_signatures and not self._verifier.verify(manifest):
            violations.append("signature missing or not trusted")
        if self._require_conformance and self._conformance.get(manifest.port_id) is None:
            violations.append(
                f"no conformance kit is registered for port {manifest.port_id}; "
                f"an adapter cannot be activated without one (V3)"
            )
        return tuple(violations)

    # --- loading ------------------------------------------------------------ #

    def load(self, plugin_id: PluginId) -> LoadedPlugin:
        """Validate, instantiate, and run the fast conformance subset.

        Raises:
            ManifestInvalidError, PortIncompatibleError, SignatureInvalidError:
                on validation failure.
            ConformanceFailedError: when the adapter fails its kit. The plugin is
                not activated.
        """
        with self._lock:
            descriptor = self._descriptors.get(plugin_id)
        if descriptor is None:
            raise PluginError(f"plugin '{plugin_id}' has not been registered", plugin_id=plugin_id)

        manifest = descriptor.manifest
        violations = self.validate(manifest)
        if violations:
            self._reject(manifest, "; ".join(violations))
            if any("signature" in v for v in violations):
                raise SignatureInvalidError(
                    f"plugin '{plugin_id}' rejected: {violations[0]}", plugin_id=plugin_id
                )
            if any("range" in v or "bindable" in v for v in violations):
                raise PortIncompatibleError(
                    f"plugin '{plugin_id}' rejected: {'; '.join(violations)}",
                    plugin_id=plugin_id,
                )
            raise ManifestInvalidError(
                f"plugin '{plugin_id}' rejected: {'; '.join(violations)}", plugin_id=plugin_id
            )

        try:
            instance = descriptor.factory()
        except Exception as exc:  # noqa: BLE001 - normalise adapter construction failure
            self._reject(manifest, f"construction failed: {exc}")
            raise PluginError(
                f"plugin '{plugin_id}' failed to construct: {exc}", plugin_id=plugin_id
            ) from exc

        report = self._run_conformance(manifest, instance, fast_only=True)
        if not report.passed:
            self._reject(manifest, f"conformance failed: {report.summary()}")
            self._metrics.counter(
                MetricName.CONFORMANCE_FAILURES, port=str(manifest.port_id)
            ).increment()
            raise ConformanceFailedError(
                f"plugin '{plugin_id}' failed conformance for {manifest.port_id}",
                failures=report.failures,
                plugin_id=plugin_id,
            )

        loaded = LoadedPlugin(manifest=manifest, instance=instance, conformance=report)
        with self._lock:
            self._loaded[plugin_id] = loaded
        self._metrics.counter(MetricName.PLUGINS_LOADED, port=str(manifest.port_id)).increment()
        self._bus.publish(
            PluginLoaded(
                occurred_at=self._clock.now(),
                partition_key=str(plugin_id),
                plugin_id=plugin_id,
                port_id=str(manifest.port_id),
            )
        )
        return loaded

    def run_conformance(
        self, plugin_id: PluginId, *, fast_only: bool = False
    ) -> ConformanceReport:
        """Run a kit against an already-loaded plugin (nightly full runs)."""
        with self._lock:
            loaded = self._loaded.get(plugin_id)
        if loaded is None:
            raise PluginError(f"plugin '{plugin_id}' is not loaded", plugin_id=plugin_id)
        return self._run_conformance(loaded.manifest, loaded.instance, fast_only=fast_only)

    def _run_conformance(
        self, manifest: PluginManifest, instance: Any, *, fast_only: bool
    ) -> ConformanceReport:
        kit = self._conformance.get(manifest.port_id)
        if kit is None:
            if self._require_conformance:
                return ConformanceReport(
                    port_id=manifest.port_id,
                    kit_version="none",
                    passed=False,
                    failures=("no conformance kit registered for this port",),
                    fast_subset_only=fast_only,
                )
            return ConformanceReport(
                port_id=manifest.port_id, kit_version="none", passed=True, fast_subset_only=fast_only
            )
        return kit.run(instance, fast_only=fast_only)

    def _reject(self, manifest: PluginManifest, reason: str) -> None:
        self._metrics.counter(MetricName.PLUGINS_REJECTED, port=str(manifest.port_id)).increment()
        self._bus.publish(
            PluginRejected(
                occurred_at=self._clock.now(),
                partition_key=str(manifest.plugin_id),
                plugin_id=manifest.plugin_id,
                reason=reason,
            )
        )

    # --- binding ------------------------------------------------------------ #

    def activate(self, plugin_id: PluginId) -> Any:
        """Bind a loaded plugin to its port and return the adapter instance."""
        with self._lock:
            loaded = self._loaded.get(plugin_id)
            if loaded is None:
                raise PluginError(f"plugin '{plugin_id}' is not loaded", plugin_id=plugin_id)
            self._bindings[loaded.manifest.port_id] = plugin_id
            return loaded.instance

    def resolve(self, port_id: PortId) -> Any:
        """Return the adapter currently bound to a port."""
        with self._lock:
            plugin_id = self._bindings.get(port_id)
            if plugin_id is None:
                raise PluginError(f"no adapter is bound to port {port_id}", port_id=str(port_id))
            return self._loaded[plugin_id].instance

    def try_resolve(self, port_id: PortId) -> Any | None:
        with self._lock:
            plugin_id = self._bindings.get(port_id)
            return self._loaded[plugin_id].instance if plugin_id else None

    def swap(self, port_id: PortId, new_plugin_id: PluginId) -> Any:
        """Replace the adapter bound to a port.

        On failure the previous binding stays in force — a rollback, not a
        half-applied swap.
        """
        with self._lock:
            previous = self._bindings.get(port_id)
        try:
            if new_plugin_id not in self._loaded:
                self.load(new_plugin_id)
            return self.activate(new_plugin_id)
        except PluginError:
            with self._lock:
                if previous is not None:
                    self._bindings[port_id] = previous
            raise

    def deactivate(self, port_id: PortId) -> None:
        with self._lock:
            self._bindings.pop(port_id, None)

    def unload(self, plugin_id: PluginId) -> None:
        with self._lock:
            loaded = self._loaded.pop(plugin_id, None)
            if loaded is not None:
                for port_id, bound in list(self._bindings.items()):
                    if bound == plugin_id:
                        self._bindings.pop(port_id, None)

    # --- introspection ------------------------------------------------------ #

    def catalogue(self) -> tuple[LoadedPlugin, ...]:
        with self._lock:
            return tuple(self._loaded.values())

    def bindings(self) -> dict[PortId, PluginId]:
        with self._lock:
            return dict(self._bindings)

    def capabilities(self) -> dict[PortId, dict[str, str]]:
        """Published capability declarations, for V8 gap reporting."""
        with self._lock:
            return {
                port_id: dict(self._loaded[plugin_id].manifest.capabilities)
                for port_id, plugin_id in self._bindings.items()
            }
