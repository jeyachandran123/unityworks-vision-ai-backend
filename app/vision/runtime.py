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
from pathlib import Path
from typing import Any

from loguru import logger

from app.configuration.settings import Settings
from app.errors import ConfigurationInvalidError
from app.vision.understanding import UnderstandingComposition, build_understanding
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

    def assemble(
        self,
        *,
        bindings_factory: Any = None,
        bind_understanding: bool = True,
        bind_perception: bool = False,
        provider: str | None = None,
        static_value: str | None = None,
    ) -> VisionComposition:
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
        document = self._config_document()

        # ── THE canonical registry. Built once, here, and shared. ────────────
        attributes = build_attribute_registry(policies)

        # The detector declaration is written into the document **before** the
        # platform is built, because `build_platform` reads configuration once at
        # construction. Mutating the document afterwards would leave the running
        # platform bound to a detector list it never saw.
        bound_detector = None
        if bind_perception:
            from vision_os.adapters.configuration.detector_providers import build_detector
            from vision_os.kernel.clock import VirtualClock

            bound_detector = build_detector(clock=VirtualClock())
            document["detectors"] = [
                bound_detector.declaration(
                    detector_id="primary",
                    artifact_uri=_ARTIFACT_URI,
                )
            ]

        # After the detector, because the taxonomy must cover what it can name.
        document["taxonomy"] = _taxonomy_from(policies, bound_detector)

        platform = self._build_platform(bindings_factory, document=document)

        registry_layer = build_registry_layer(
            platform,
            store=self._object_store(),
            attributes=attributes,          # ← the Phase 6.9 parameter
        )

        # ── Flow 5 + 6, bound in Phase 4 ────────────────────────────────────
        #
        # Optional, and reported rather than fatal. A platform whose provider is
        # unconfigured still detects, tracks, registers and serves observations;
        # 10_RELIABILITY §4.3 step 5 is explicit that with no understanding
        # available *"attributes stop; presence/spatial CONTINUE"*. Refusing to
        # assemble would take down more than the missing capability.
        composition: UnderstandingComposition | None = None
        if bind_understanding:
            try:
                composition = build_understanding(
                    platform,
                    registry_layer,
                    attributes,
                    policies=policies,
                    provider=provider,
                    static_value=static_value,
                )
            except ConfigurationInvalidError as exc:
                logger.warning("understanding not bound: {}", exc)

        understanding = composition.understanding if composition else None

        # ── Flow 2/3: find and follow things ────────────────────────────────
        #
        # Built after understanding so the crop layer exists to consume from.
        # `detection_consumer=tracking.runtime` and the tracking → registry
        # bridge are the platform's own declared seams; this wires them and
        # implements neither.
        detection = tracking = None
        if bind_perception:
            detection, tracking = self._build_perception(
                platform, registry_layer, bound_detector
            )

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
            cropping=composition.cropping if composition else None,
            understanding=understanding,
            detection=detection,
            tracking=tracking,
            policies=policies,
            understanding_composition=composition,
        )

    def _build_perception(self, platform: Any, registry_layer: Any, bound: Any):
        """Detection and tracking, wired to each other and to the registry.

        The detector is whatever `VISION_DETECTOR_PROVIDER` names — this method
        contains no model name and no inference. The two seams it connects,
        detection → tracking and tracking → registry, are both declared by the
        platform; wiring them is composition, and reimplementing either would be
        a second pipeline.
        """
        from vision_os.detection_bootstrap import build_detection_layer
        from vision_os.tracking_bootstrap import build_tracking_layer

        tracking = build_tracking_layer(platform, tracking_sink=None)

        # The model manager fetches weights through an ArtifactStorePort. The
        # provider already resolved where they are; this puts them somewhere the
        # manager can fetch from, which is the store's whole job.
        from vision_os.adapters.models import InMemoryArtifactStore

        artifacts = InMemoryArtifactStore()
        artifacts.put(_ARTIFACT_URI, _artifact_bytes(bound))

        detection = build_detection_layer(
            platform,
            detector_factory=lambda *_args, **_kwargs: bound.detector,
            artifacts=artifacts,
            detection_consumer=tracking.runtime,
        )

        # Tracking → registry. The registry runtime is the declared consumer;
        # attaching it here is the Flow 3/4 seam and nothing more.
        tracking.runtime._sink = registry_layer.runtime  # noqa: SLF001
        return detection, tracking

    def _build_platform(self, bindings_factory: Any, *, document: Any = None) -> Any:
        from vision_os.adapters.configuration import InMemoryConfigSource
        from vision_os.bootstrap import build_platform
        from vision_os.conformance import platform_registry
        from vision_os.kernel.clock import VirtualClock
        from vision_os.kernel.config import ConfigLayer

        document = document if document is not None else self._config_document()

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
            # ── Attention (M8) ──────────────────────────────────────────────
            #
            # The cost gate. `part_focused` spends the canonical crop on the
            # region a question is about rather than on the black bars either
            # side of a letterboxed standing person; with no regions declared it
            # plans the whole box, exactly as a padded strategy would.
            #
            # The per-attribute resolutions (head 448, hands 224) come from the
            # policy document, not from here — Phase 4.2 measured them and the
            # geometry is a domain decision.
            # ── Perception (M5/M6) ──────────────────────────────────────────
            "detection": {"enabled": True},
            # The platform has no opinion about which tracker suits a site,
            # so the site must name one. IoU is the shipped default.
            "tracking": {"enabled": True, "tracker_id": "tracker.iou"},
            "cropping": {
                "enabled": True,
                "crop_strategy": "crop.part_focused",
                "understanding_calls_per_hour": 3_600.0,
            },
            # ── Understanding (M9) ──────────────────────────────────────────
            "understanding": {
                "enabled": True,
            },
            # ── Synthesis + State, so observations reach the API ────────────
            "synthesis": {"enabled": True, "suppression_policy": "suppression.exact"},
            "state": {"enabled": True, "max_objects_per_partition": 64},
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
            # The classes the platform may name. A detector mapping to an
            # undeclared class is refused at load, not at the first frame —
            # which is why this is here rather than discovered in production.
            #
            # `person` comes from the active policy's scope, not from a
            # hardcoded vertical: see `_taxonomy_from(policies)`.
            "taxonomy": [],
        }


#: Where the detector's weights are published for the model manager to fetch.
#: One name, used by both the declaration and the store, so they cannot disagree.
_ARTIFACT_URI = "mem://detector.bin"


def _artifact_bytes(bound: Any) -> bytes:
    """The detector's weights, as bytes the artifact store can hold.

    A real weights file is read from disk; the scripted reference detector has
    no file and supplies a marker instead. Either way the manager fetches
    through the port rather than reaching for a path.
    """
    path = getattr(bound, "artifact_path", "") or ""
    if path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.read_bytes()

    # The scripted reference detector has no file on disk. `_build_reference` in
    # `detector_providers.py` declares `blake2b(b"reference")`, so the store must
    # hold exactly those bytes — the model manager verifies the hash on fetch and
    # a mismatch is an integrity failure, not a warning.
    #
    # Not `b"reference-weights"`: that constant belongs to the platform's own
    # test modules, which publish under a different URI and never meet this code.
    return b"reference"


def _taxonomy_from(policies: tuple[Any, ...], bound: Any = None) -> list[dict[str, Any]]:
    """Every class the platform may name — from policy *and* from the detector.

    Both sources are required, for different reasons:

    * **Policy** declares the classes its attributes apply to. `person` is a
      domain fact and belongs in the document that owns the domain; hardcoding it
      here would put a vertical's vocabulary into the composition root.
    * **The detector** declares what it can emit. A mapping to an undeclared
      class is refused at load — correctly — so a COCO detector that can name 80
      classes needs all 80 declared, or its mappings must be narrowed first with
      `VISION_DETECTOR_CLASSES`.

    Taking the union means any detector composes without editing this function.
    Narrowing what the detector emits is a configuration decision, made in
    `VISION_DETECTOR_CLASSES`, not a decision made here by omission.
    """
    classes: set[str] = set()

    for policy in policies:
        # `SemanticPolicy.object_classes` is the policy's declared scope.
        for name in getattr(policy, "object_classes", ()) or ():
            classes.add(str(name))

    if bound is not None:
        for entry in getattr(bound, "mappings", ()) or ():
            classes.add(str(getattr(entry, "class_id", "")))
        for name in getattr(bound, "classes", ()) or ():
            classes.add(str(name))

    return [
        {"class_id": name, "geometry_kinds": ("box",)}
        for name in sorted(c for c in classes if c)
    ]


__all__ = ["VisionRuntime", "VisionStatus", "declared_keys"]
