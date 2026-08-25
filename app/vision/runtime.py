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
from app.vision.bridges import TrackingToRegistryBridge
from app.vision.composition import (
    VisionComposition,
    assert_shared_attribute_registry,
    build_attribute_registry,
    declared_keys,
    describe_composition,
    load_policies,
)
from app.vision.understanding import UnderstandingComposition, build_understanding


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

    __slots__ = ("_bridge", "_composition", "_reason", "_settings")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._composition: VisionComposition | None = None
        #: The L3→L4 adapter, kept so DevTools can report what crossed the seam.
        self._bridge: TrackingToRegistryBridge | None = None
        self._reason = "not started"

    @property
    def composition(self) -> VisionComposition | None:
        return self._composition

    @property
    def bridge(self) -> TrackingToRegistryBridge | None:
        """The L3→L4 adapter, for DevTools. `None` when perception is unbound."""
        return self._bridge

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
            policies=tuple(f"{p['policy_id']}@{p['version']}" for p in described["policies"]),
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
            self._composition = self.assemble(
                bind_perception=self._settings.vision_bind_perception,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self._reason = f"{type(exc).__name__}: {exc}"
            logger.error("Vision OS failed to assemble: {}", self._reason)
            return False

        started = await self._boot_layers(self._composition)

        logger.info(
            "Vision OS assembled — {} attribute(s) declared, perception {}, " "layers started: {}",
            len(self._composition.declared_attributes),
            "bound" if self._composition.detection is not None else "NOT bound",
            ", ".join(started) or "none",
        )
        return True

    async def _boot_layers(self, composition: VisionComposition) -> tuple[str, ...]:
        """Start each layer runtime that has one.

        Assembly wires the layers; it does not run them. `DetectionRuntime`
        refuses work until `start()` has warmed the detector — `on_admitted`
        returns immediately while `_started` is false — so a stack that is
        assembled and never booted accepts every frame and does nothing with any
        of them, silently. That is the exact failure this method exists to
        prevent, and it is why the started list is logged.

        This mirrors `VisionSystem.boot()` for the layers it owns, and
        deliberately does **not** call `platform.boot()`: that attaches cameras
        through the bindings factory, and acquisition in this application belongs
        to `LiveRuntime`, which owns the sources. Booting both would be two
        acquisition paths for the same cameras.

        Warm-up order is the pipeline's own: a downstream stage must be ready
        before the one that feeds it, or the first frames are handed to a
        consumer that is not yet listening.
        """
        started: list[str] = []
        for name in (
            "synthesis",
            "state",
            "understanding",
            "cropping",
            "registry_layer",
            "tracking",
            "detection",
        ):
            layer = getattr(composition, name, None)
            runtime = getattr(layer, "runtime", None)
            starter = getattr(runtime, "start", None)
            if not callable(starter):
                continue
            try:
                await starter()
                started.append(name)
            except Exception as exc:  # noqa: BLE001 - one layer, not the process
                # Reported and named. A layer that failed to start is a
                # capability this deployment does not have, and it must be
                # visible rather than inferred later from an empty result.
                logger.error(
                    "Vision OS layer '{}' failed to start: {}: {}",
                    name,
                    type(exc).__name__,
                    exc,
                )
        return tuple(started)

    async def stop(self) -> None:
        if self._composition is not None:
            for name in (
                "detection",
                "tracking",
                "registry_layer",
                "cropping",
                "understanding",
                "state",
                "synthesis",
            ):
                runtime = getattr(getattr(self._composition, name, None), "runtime", None)
                stopper = getattr(runtime, "stop", None)
                if callable(stopper):
                    try:
                        await stopper()
                    except Exception as exc:  # noqa: BLE001 - shutdown continues
                        logger.warning(
                            "layer '{}' did not stop cleanly: {}: {}",
                            name,
                            type(exc).__name__,
                            exc,
                        )
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
        from vision_os.exposure_bootstrap import build_exposure_layer
        from vision_os.registry_bootstrap import build_registry_layer
        from vision_os.synthesis_bootstrap import build_synthesis_layer

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
            attributes=attributes,  # ← the Phase 6.9 parameter
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
                    # An explicit argument wins; otherwise the deployment's
                    # configured provider, which the platform cannot read for
                    # itself because `.env` never reaches `os.environ`.
                    provider=provider or (self._settings.vision_understander_provider or None),
                    static_value=static_value,
                    api_key=self._settings.vision_understander_api_key.get_secret_value(),
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
            detection, tracking = self._build_perception(platform, registry_layer, bound_detector)

        # ── Flow 7: publish what M7 holds ───────────────────────────────────
        #
        # Never composed before this phase. Attributes reached M7 and stopped
        # there: `synthesis.built = 0`, `state.appended = 0`, and the Observation
        # API served nothing — a system that had done all the expensive work and
        # could not show any of it.
        #
        # `attach=True` wires the runtime to the registry's own declared sink, so
        # a registry result becomes an observation without passing through
        # understanding. That is the platform's dotted edge, not a new path.
        synthesis = None
        try:
            synthesis = build_synthesis_layer(
                platform,
                registry_layer,
                # The canonical instance again. A second registry here would put
                # attributes the schema considers illegal into the permanent
                # record — the one place they can never be taken back out.
                attributes=attributes,
                attach=True,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            logger.error(
                "synthesis not bound: {}: {}. Attributes will reach M7 and no "
                "observation will be published.",
                type(exc).__name__,
                exc,
            )

        # ── The second synthesis seam: M9 results → attribute observations ──
        #
        # `attach=True` above wires only the *registry* seam, which carries
        # presence and spatial facts. Attribute facts arrive on a different
        # seam, `attach_understanding`, and it had never been called — so
        # Phase 7 measured 180 attributes written to M7, `synthesis.built`
        # counting presence and spatial only, and
        # `synthesis.attributes_published = 0`. Every confirmed person the
        # Observation API returned carried `attrs = 0`, because state is
        # projected from observations (07_STATE §1.1 makes the observation the
        # only write path) and no attribute observation was ever published.
        #
        # This does not make a second copy of the attribute and does not
        # change what M7 holds: the platform's own helper *chains* onto the
        # existing sink, so `RegistryWriteBackSink` still writes to M7 exactly
        # as before and synthesis additionally publishes the observation that
        # carries the fact into Vision State.
        if synthesis is not None and understanding is not None:
            try:
                from vision_os.synthesis_bootstrap import attach_understanding

                attach_understanding(understanding, synthesis.runtime)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                logger.error(
                    "understanding not attached to synthesis: {}: {}. PPE "
                    "attributes will reach M7 and never appear on an object.",
                    type(exc).__name__,
                    exc,
                )

        # ── Flow 8: expose what Vision State holds ──────────────────────────
        #
        # Also never composed. `VisionComposition.api` has been `None` since
        # Phase 1, so the platform's own `ObservationApi` — with its authorizer,
        # scoping and audit trail already built — was unreachable, and DevTools
        # served a fixture instead of the live camera.
        #
        # The state manager is the API's **only** window onto the platform,
        # which is what makes the L6/L7 split real rather than nominal. Passing
        # `synthesis.state` here is the whole wiring.
        exposure = None
        if synthesis is not None:
            try:
                exposure = build_exposure_layer(platform, synthesis.state)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                logger.error(
                    "exposure not bound: {}: {}. Vision State will hold "
                    "observations that no API can serve.",
                    type(exc).__name__,
                    exc,
                )

        # Fails assembly rather than degrading. A composition with two registries
        # runs perfectly and is silently wrong, which is the worst failure mode
        # available to it.
        assert_shared_attribute_registry(registry_layer, understanding, attributes)

        return VisionComposition(
            system=None,
            api=exposure.api if exposure is not None else None,
            platform=platform,
            attributes=attributes,
            registry_layer=registry_layer,
            cropping=composition.cropping if composition else None,
            understanding=understanding,
            detection=detection,
            tracking=tracking,
            policies=policies,
            synthesis=synthesis,
            exposure=exposure,
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
        # The model manager fetches weights through an ArtifactStorePort. The
        # provider already resolved where they are; this puts them somewhere the
        # manager can fetch from, which is the store's whole job.
        from vision_os.adapters.models import InMemoryArtifactStore
        from vision_os.detection_bootstrap import build_detection_layer
        from vision_os.tracking_bootstrap import build_tracking_layer

        artifacts = InMemoryArtifactStore()
        artifacts.put(_ARTIFACT_URI, _artifact_bytes(bound))

        # Tracking first: detection needs it as its declared consumer.
        #
        # Tracking → registry goes through an adapter rather than by assigning a
        # private attribute. `TrackingRuntime` calls its sink as `sink(outcome)`;
        # `RegistryRuntime` exposes `await on_tracked(camera_id, update)`. The
        # two do not match, and because tracking guards its sink (invariant V9)
        # the mismatch was counted as `sink_failures` and never surfaced —
        # detection and tracking ran while registry, cropping, understanding,
        # synthesis and state all stayed at zero. `TrackingToRegistryBridge`
        # translates the shape and counts what crosses.
        bridge = TrackingToRegistryBridge(registry_layer.runtime)
        tracking = build_tracking_layer(platform, tracking_sink=bridge)
        self._bridge = bridge

        detection = build_detection_layer(
            platform,
            detector_factory=lambda *_args, **_kwargs: bound.detector,
            artifacts=artifacts,
            detection_consumer=tracking.runtime,
        )

        return detection, tracking

    def _build_platform(self, bindings_factory: Any, *, document: Any = None) -> Any:
        from vision_os.adapters.configuration import InMemoryConfigSource
        from vision_os.bootstrap import build_platform
        from vision_os.conformance import platform_registry
        from vision_os.kernel.clock import SystemClock, VirtualClock
        from vision_os.kernel.config import ConfigLayer

        document = document if document is not None else self._config_document()

        # A virtual clock does not advance on its own. Under one, the detection
        # scheduler's inference budget expires the instant it is set, every frame
        # fails `timeout`, and — because detection publishes a *transient*
        # failure rather than raising — the pipeline reports itself healthy while
        # observing nothing. That is invariant V8's exact failure mode, and it is
        # why the clock is a named deployment decision.
        deterministic = self._settings.vision_deterministic_clock
        document["platform"] = {
            **document.get("platform", {}),
            "clock_mode": "virtual" if deterministic else "system",
        }

        def _no_sources(camera):
            raise ConfigurationInvalidError(
                "no source bindings are configured; frame acquisition arrives in "
                "Phase 3 with the streaming RTSP and file adapters"
            )

        return build_platform(
            config_sources={ConfigLayer.SITE: InMemoryConfigSource(document)},
            bindings_factory=bindings_factory or _no_sources,
            clock=VirtualClock() if deterministic else SystemClock(),
            conformance=platform_registry(),
            allocator=self._frame_pool(document),
        )

    def _frame_pool(self, document: dict[str, Any]) -> Any:
        """A frame pool sized for the cameras this deployment actually feeds.

        The platform's own default multiplies `slots_per_camera` by
        `len(cameras)`, and this application declares no cameras on purpose
        (see `_config_document`) — so the default arrives sized for one camera
        however many are running. Four live cameras against that pool
        exhausted it on essentially every frame: `PoolExhaustedError` on
        publish, nothing reaching detection, and sessions still reporting
        `running` because a publish failure is recorded rather than raised.

        Sized here instead, from the fan-in the deployment states and the
        history window the document asks the buffer to hold.
        """
        from vision_os.adapters.memory import HostMemoryPool

        buffer = document.get("buffer", {})
        cameras = max(1, int(self._settings.vision_analysis_cameras))
        slots = max(1, int(buffer.get("slots_per_camera", 8))) * cameras
        # The same 1.5 headroom the platform's own default applies, for frames
        # pinned by a reader while the next one arrives.
        slots = int(slots * 1.5)
        return HostMemoryPool(
            slots=slots, bytes_per_slot=int(buffer.get("bytes_per_slot", 1920 * 1080 * 3))
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
            # `clock_mode` is overwritten by `_build_platform` from settings;
            # stated here so the document is valid on its own.
            "platform": {"deployment_profile": "embedded", "clock_mode": "system"},
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
            # ── Exposure (M14) ──────────────────────────────────────────────
            #
            # Enabled so Vision State has a reader. `authz.tenant_reads` rather
            # than the shipped `authz.deny_all` default: this application has
            # already authenticated the caller and resolved their tenant and
            # camera scope at the HTTP edge, and the platform's authorizer is
            # the second gate that confines the query to that tenant.
            #
            # Not `authz.static`, which would hand every caller one fixed grant
            # — that is right for the DevTools fixture and wrong for a live
            # multi-tenant API.
            "api": {
                "enabled": True,
                "authorizer": "authz.tenant_reads",
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
        {"class_id": name, "geometry_kinds": ("box",)} for name in sorted(c for c in classes if c)
    ]


__all__ = ["VisionRuntime", "VisionStatus", "declared_keys"]
