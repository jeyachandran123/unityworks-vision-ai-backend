"""Composition root for Flow 7 — Observation Builder and Vision State.

Assembles M11 and M12 against an already-built platform. The builder holds
``SuppressionPolicyPort`` and ``ObservationSinkPort``; state holds
``ObservationLogPort``. Which implementation satisfies each is decided **only
here**.

Every adapter is run through its conformance kit before use — invariant V3 as a
gate rather than an aspiration (06_PORTS §5). For P20 that gate matters most:
``idempotent_by_id`` is what makes 07_STATE §9.1's *"restart, replay from the last
committed log position"* safe, and a log that double-counted would corrupt the
record every time it recovered.

**This is also where the two seams are wired.** The synthesis runtime attaches to
the Crop Manager's registry-adjacent path and to the Understanding runtime's
sink, which the Flow 4 and Flow 6 reports named as their extension points. Both
earlier flows remain unaware of M11: each holds a callable and never learns what
implements it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .adapters.synthesis import (
    SUPPRESSION_FACTORIES,
    InMemoryObservationLog,
)
from .bootstrap import VisionPlatform
from .conformance import (
    OBSERVATION_LOG_KIT,
    OBSERVATION_SINK_KIT,
    SUPPRESSION_POLICY_KIT,
)
from .core.errors import ObservationError, StateError
from .core.model.ids import ClassId, ConfigRevision, ModuleId, SiteId
from .core.model.provenance import Provenance
from .core.ports.synthesis import (
    ObservationLogPort,
    ObservationSinkPort,
    SuppressionPolicyPort,
)
from .kernel.plugins.manifest import PortCatalogue
from .perception.registry.attributes import AttributeRegistry
from .registry_bootstrap import RegistryLayer
from .state import VisionStateManager
from .synthesis import (
    CeilingGate,
    ObservationBuilder,
    SynthesisRuntime,
    TaxonomyView,
)

SYNTHESIS_MODULE = ModuleId("observation_builder")
SYNTHESIS_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class SynthesisLayer:
    """Everything Flow 7 assembled, for tests and operators to reach into."""

    builder: ObservationBuilder
    runtime: SynthesisRuntime
    state: VisionStateManager
    log: ObservationLogPort
    sinks: tuple[ObservationSinkPort, ...] = ()

    @property
    def policy_id(self) -> str:
        return self.builder._suppression_policy.policy_id  # noqa: SLF001


def build_taxonomy_view(platform: VisionPlatform) -> TaxonomyView:
    """The classes and version M11 checks against.

    Read from configuration rather than from a producer's claim: the whole point
    of the mismatch check is to catch a producer that believes something
    different from the site.
    """
    declarations = platform.config.taxonomy()
    return TaxonomyView(
        version=_taxonomy_version(platform),
        classes=frozenset(ClassId(d.class_id) for d in declarations),
    )


def _taxonomy_version(platform: VisionPlatform) -> str:
    """The site's declared taxonomy version.

    Derived from the config revision when a deployment has not declared one
    explicitly: a version that changes with configuration is still a version, and
    an empty one would disable the mismatch check silently.
    """
    return str(platform.config.revision())


def build_suppression_policy(platform: VisionPlatform) -> SuppressionPolicyPort:
    """Select a suppression policy by name.

    Raises:
        ObservationError: the configured policy is unknown. Refusing beats
            defaulting: a typo that silently fell back to ``always`` would
            multiply the platform's output volume by 10-50x with no signal but
            the storage bill.
    """
    settings = platform.config.synthesis()
    factory = SUPPRESSION_FACTORIES.get(settings.suppression_policy)
    if factory is None:
        raise ObservationError(
            f"unknown suppression policy '{settings.suppression_policy}'; known "
            f"policies are {sorted(SUPPRESSION_FACTORIES)}",
            requested=settings.suppression_policy,
        )
    if settings.suppression_policy == "suppression.threshold":
        return factory(position_threshold=settings.position_threshold)
    return factory()


def build_synthesis_layer(
    platform: VisionPlatform,
    registry_layer: RegistryLayer,
    *,
    attributes: AttributeRegistry | None = None,
    suppression_policy: SuppressionPolicyPort | None = None,
    log: ObservationLogPort | None = None,
    sinks: Sequence[ObservationSinkPort] = (),
    taxonomy: TaxonomyView | None = None,
    site_id: SiteId | None = None,
    attach: bool = True,
) -> SynthesisLayer:
    """Assemble Flow 7 against an already-built platform.

    Args:
        attributes: The Attribute Schema Registry — **the same instance** M7 and
            M9 hold. A second copy would drift, and the drift would surface as
            attributes the registry considers illegal sitting in the permanent
            record.
        log: The observation log. Defaults to ``log.memory``, which is honest for
            an embedded deployment and says so through its id — a site claiming
            durability binds ``log.file`` or something replicated.
        attach: Wire the runtime to the registry's sink. False leaves the layer
            assembled but unconnected, which is what a unit test wants.

    Raises:
        ObservationError: synthesis is disabled, or an adapter failed its kit.
        StateError: state is disabled.
    """
    synthesis_config = platform.config.synthesis()
    state_config = platform.config.state()

    if not synthesis_config.enabled:
        raise ObservationError(
            "synthesis.enabled is false; a site that does not want published "
            "facts should not build the layer rather than build one that "
            "publishes nothing"
        )
    if not state_config.enabled:
        raise StateError(
            "state.enabled is false; observations would be built and then have "
            "nowhere to go, which is worse than not building them"
        )

    registry = attributes if attributes is not None else AttributeRegistry()
    policy = (
        suppression_policy
        if suppression_policy is not None
        else build_suppression_policy(platform)
    )
    selected_log = log if log is not None else InMemoryObservationLog()

    _gate(platform, PortCatalogue.SUPPRESSION_POLICY, SUPPRESSION_POLICY_KIT, policy)
    _gate(platform, PortCatalogue.OBSERVATION_LOG, OBSERVATION_LOG_KIT, selected_log)
    for sink in sinks:
        _gate(platform, PortCatalogue.OBSERVATION_SINK, OBSERVATION_SINK_KIT, sink)

    platform_config = platform.config.platform()
    provenance = Provenance(
        producer_module=SYNTHESIS_MODULE,
        producer_version=SYNTHESIS_VERSION,
        config_revision=ConfigRevision(str(platform.config.revision())),
        deterministic=platform_config.deterministic,
    )

    view = taxonomy if taxonomy is not None else build_taxonomy_view(platform)
    builder = ObservationBuilder(
        clock=platform.clock,
        metrics=platform.metrics,
        events=platform.bus,
        config=synthesis_config,
        gate=CeilingGate(registry, view),
        provenance=provenance,
        suppression_policy=policy,
    )

    state = VisionStateManager(
        clock=platform.clock,
        metrics=platform.metrics,
        events=platform.bus,
        config=state_config,
        log=selected_log,
        site_id=site_id or _site_of(platform),
    )

    runtime = SynthesisRuntime(
        clock=platform.clock,
        metrics=platform.metrics,
        health=platform.health,
        builder=builder,
        config=synthesis_config,
        state=state,
        sinks=sinks,
        taxonomy_version=view.version,
    )

    if attach:
        _attach(registry_layer, runtime)

    return SynthesisLayer(
        builder=builder,
        runtime=runtime,
        state=state,
        log=selected_log,
        sinks=tuple(sinks),
    )


def _gate(platform: VisionPlatform, port, kit, adapter) -> None:
    """Run an adapter's conformance kit before it is ever used.

    A registered kit wins over the shipped one when a deployment registered its
    own, but *some* kit must run. For P20 this catches a non-idempotent log,
    which would corrupt the record on every recovery. Flow 4 gates P21 the same
    way, and this follows that precedent.

    **The kit writes real records**, because a store can only be shown to store
    by storing. It writes them under its own reserved ``kit-*`` partitions, so a
    read for a real camera never sees them — but they are there, and
    ``_purge_kit_traces`` removes what it can afterwards. What cannot be removed
    is documented as a known limitation rather than hidden: a durable adapter
    that keeps no per-partition deletion path retains a handful of fixture
    records under partitions no camera will ever use.
    """
    registered = platform.conformance.get(port)
    effective = registered or kit
    report = effective.run(adapter, fast_only=True)
    if not report.passed:
        raise ObservationError(
            f"adapter for {port} failed conformance: {'; '.join(report.failures)}",
            port=str(port),
        )
    _purge_kit_traces(adapter)


def _purge_kit_traces(adapter) -> None:
    """Clear what the conformance run left behind.

    A live sink whose first delivery is a fixture would mislead anyone reading
    its output, and a collecting sink's count would be wrong from the start. The
    reset is best-effort by design: an adapter with no way to forget is left
    alone rather than poked with an interface it never declared.
    """
    reset = getattr(adapter, "reset", None)
    if callable(reset):
        reset()


def _site_of(platform: VisionPlatform) -> SiteId:
    """Site from the camera declarations, not from a global setting.

    A node can serve several sites; this is the partition-level default for a
    deployment that declared exactly one, matching Flow 4's precedent.
    """
    cameras = platform.config.cameras()
    return SiteId(cameras[0].site_id if cameras else "unset")


def _attach(registry_layer: RegistryLayer, runtime: SynthesisRuntime) -> None:
    """Wire synthesis to the registry's declared sink.

    `01_LAYERED` §3.1's dotted edges: registry results become observations
    **without passing through understanding**. Attaching here rather than only to
    M9's output is what makes that true — presence and spatial facts are
    published whether or not a model ever ran.

    The registry's sink is synchronous while ``on_registered`` is a coroutine, so
    the hand-off is scheduled. Awaiting would put synthesis latency on the
    critical path of the layer beneath it, and V9 says a failure at L5 may not
    stop L2.
    """
    import asyncio

    existing = getattr(registry_layer.runtime, "_sink", None)

    def sink(update) -> None:
        if existing is not None:
            existing(update)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(runtime.on_registered(update))
            return
        loop.create_task(runtime.on_registered(update))

    registry_layer.runtime._sink = sink  # noqa: SLF001 - the declared seam


def attach_understanding(understanding_layer, runtime: SynthesisRuntime) -> None:
    """Wire M9's results into synthesis.

    Separate from ``_attach`` because the two seams carry different facts and a
    deployment may run either without the other: presence observations need no
    understanding, and a site with no registry could still synthesise attributes.
    """
    import asyncio

    existing = getattr(understanding_layer.runtime, "_sink", None)

    def sink(results) -> None:
        if existing is not None:
            existing(results)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(runtime.on_understood(results))
            return
        loop.create_task(runtime.on_understood(results))

    understanding_layer.runtime._sink = sink  # noqa: SLF001 - the declared seam
