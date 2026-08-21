"""The Vision OS lifecycle owner.

One object holds the assembled platform for the process, and every route reaches
it through here. It exists so that "is the platform running?" has exactly one
answer, and so that the answer is never confused with "did the platform see
anything?".

### Why nothing is assembled in Phase 1 by default

`VISION_AUTOSTART` defaults to false. A platform needs a `SourcePort` to read
from, and Phase 1 binds none — the streaming source is Phase 3 work, and the
file-replay source arrives with it. Booting a platform with no source would
produce a system that reports itself healthy and observes nothing forever, which
is precisely the state invariant V8 exists to make impossible to misread.

So the runtime starts *not started*, says so, and every observation route returns
`VISION_UNAVAILABLE` rather than an empty result. `assemble()` is fully
implemented and exercised by tests; what is missing is the source, not the
composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.configuration.settings import Settings
from app.errors import ConfigurationInvalidError
from app.vision.composition import (
    VisionComposition,
    assert_shared_attribute_registry,
    build_attribute_registry,
    declared_keys,
    describe_composition,
    load_policies,
)


@dataclass(frozen=True, slots=True)
class VisionStatus:
    """What the application can honestly say about the platform right now."""

    assembled: bool
    reason: str = ""
    attributes: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return {
            "assembled": self.assembled,
            "reason": self.reason,
            "attributes": list(self.attributes),
            "policies": list(self.policies),
        }


class VisionRuntime:
    """Owns the platform for the process lifetime."""

    __slots__ = ("_composition", "_reason", "_settings")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._composition: VisionComposition | None = None
        self._reason = "not started"

    @property
    def composition(self) -> VisionComposition | None:
        return self._composition

    @property
    def assembled(self) -> bool:
        return self._composition is not None

    def status(self) -> VisionStatus:
        if self._composition is None:
            return VisionStatus(assembled=False, reason=self._reason)
        described = describe_composition(self._composition)
        return VisionStatus(
            assembled=True,
            attributes=tuple(described["attributes"]),
            policies=tuple(
                f"{p['policy_id']}@{p['version']}" for p in described["policies"]
            ),
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Assemble the platform if configured to. Returns whether it is running.

        Never raises during startup: a platform that fails to assemble is
        reported, not fatal, because the authentication and administration
        surfaces remain useful while perception is down. The distinction is
        recorded in `status().reason` so nothing has to guess.
        """
        if not self._settings.vision_autostart:
            self._reason = (
                "VISION_AUTOSTART is false; no source adapter is bound in this "
                "phase, and a platform with no source would report health while "
                "observing nothing"
            )
            logger.info("Vision OS not started: {}", self._reason)
            return False

        try:
            self._composition = self.assemble()
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self._reason = f"{type(exc).__name__}: {exc}"
            logger.error("Vision OS failed to assemble: {}", self._reason)
            return False

        logger.info(
            "Vision OS assembled — {} attribute(s) declared",
            len(self._composition.declared_attributes),
        )
        return True

    async def stop(self) -> None:
        self._composition = None
        self._reason = "stopped"

    # ── assembly ─────────────────────────────────────────────────────────────

    def assemble(self, *, bindings_factory: Any = None) -> VisionComposition:
        """Build the platform from its own composition roots.

        This function *calls* Vision OS's bootstraps. It does not reimplement
        them, and it makes no adapter choice the platform's own configuration
        layer could make instead.

        Args:
            bindings_factory: The P1/P2 source bindings. Required to acquire
                frames; a caller may omit it to assemble the semantic layers
                alone, which is what registry and freshness tests do.
        """
        from vision_os.registry_bootstrap import build_registry_layer

        policies = load_policies(self._settings.vision_semantic_policy)

        # ── THE canonical registry. Built once, here, and shared. ────────────
        attributes = build_attribute_registry(policies)

        platform = self._build_platform(bindings_factory)

        registry_layer = build_registry_layer(
            platform,
            store=self._object_store(),
            attributes=attributes,          # ← the Phase 6.9 parameter
        )

        understanding = None  # bound in Phase 3, with the source

        # Fails assembly rather than degrading. A composition with two registries
        # runs perfectly and is silently wrong, which is the worst failure mode
        # available to it.
        assert_shared_attribute_registry(registry_layer, understanding, attributes)

        return VisionComposition(
            system=None,
            api=None,
            platform=platform,
            attributes=attributes,
            registry_layer=registry_layer,
            policies=policies,
        )

    def _build_platform(self, bindings_factory: Any) -> Any:
        from vision_os.adapters.configuration import InMemoryConfigSource
        from vision_os.bootstrap import build_platform
        from vision_os.conformance import platform_registry
        from vision_os.kernel.clock import VirtualClock
        from vision_os.kernel.config import ConfigLayer

        document = self._config_document()

        def _no_sources(camera):
            raise ConfigurationInvalidError(
                "no source bindings are configured; frame acquisition arrives in "
                "Phase 3 with the streaming RTSP and file adapters"
            )

        return build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
            bindings_factory=bindings_factory or _no_sources,
            clock=VirtualClock(),
            conformance=platform_registry(),
        )

    def _object_store(self) -> Any:
        from vision_os.adapters.registry import InMemoryObjectStore

        return InMemoryObjectStore()

    def _config_document(self) -> dict[str, Any]:
        """A minimal valid platform configuration with **no cameras**.

        Cameras are a Phase 4 domain entity and a Phase 3 acquisition concern.
        An empty camera list is the honest representation of "this deployment has
        not been given any yet" — and unlike an empty *scope*, it carries no
        wildcard meaning anywhere in the platform.
        """
        return {
            "platform": {"deployment_profile": "embedded", "clock_mode": "virtual"},
            "buffer": {
                "slots_per_camera": 8,
                "bytes_per_slot": 1920 * 1080 * 3,
                "lease_deadline_ms": 1000,
                "history_window_ms": 5000,
            },
            "scheduler": {"global_budget_fps": 60.0, "drop_alarm_window_ms": 1000},
            "source": {
                "reconnect_backoff_initial_ms": 500,
                "reconnect_backoff_max_ms": 30_000,
                "stall_watchdog_ms": 10_000,
            },
            "health": {"aggregation_interval_ms": 1000, "report_timeout_ms": 5000},
            "runtime": {"attach_stagger_ms": 0, "drain_timeout_ms": 5000},
            # The object registry is where attributes live, so it is enabled even
            # though nothing acquires frames yet: it is the layer this phase
            # must prove holds the one canonical AttributeRegistry.
            #
            # `persistence_enabled` stays false. A durable object store is a
            # Phase 5 adapter, and binding an in-memory one while claiming
            # persistence would be worse than not persisting at all.
            "registry": {
                "enabled": True,
                "min_observations_to_confirm": 2,
                "persistence_enabled": False,
            },
            "profiles": [
                {
                    "profile_id": "standard",
                    "target_fps": 4.0,
                    "max_in_flight": 4,
                    "inference_width": 640,
                    "inference_height": 640,
                }
            ],
            "regions": [],
            "cameras": [],
        }


__all__ = ["VisionRuntime", "VisionStatus", "declared_keys"]
