"""Composition root for Flow 6 — the Understanding Engine.

Assembles M9 against an already-built platform and Crop Manager. The engine holds
``UnderstanderPort``, ``OutputCoercionPort`` and the ``PromptProvider`` seam;
which implementation satisfies each is decided **only here**.

Every adapter is run through its conformance kit before use — invariant V3 as a
gate rather than an aspiration (06_PORTS §5). For P15 that gate is the one that
matters most: ``no_fabrication_on_failure`` is the only thing standing between an
adapter that invents attributes under load and a production observation log.

**No understander is constructible by name.** There is no factory table mapping
``"vlm.qwen2_5vl"`` to a class, because binding a real model needs weights, a
runtime and a device — M18's concern and a deployment's choice. Adapters are
passed in. The reference set exists so the platform is testable, not so a
deployment can get a VLM by editing a string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .adapters.understanding import (
    COERCION_FACTORIES,
    PromptTemplate,
    StaticPromptProvider,
)
from .bootstrap import VisionPlatform
from .conformance import OUTPUT_COERCION_KIT, UNDERSTANDER_KIT
from .core.errors import UnderstandingError
from .core.model.ids import ConfigRevision, ModuleId
from .core.model.provenance import Provenance
from .core.model.timebase import Duration
from .core.ports.understanding import PromptProvider, UnderstanderPort
from .cropping_bootstrap import CroppingLayer
from .kernel.plugins.manifest import PortCatalogue
from .perception.registry.attributes import AttributeRegistry
from .perception.understanding import (
    BoundUnderstander,
    CapabilityRouter,
    ResponseCache,
    RoutingPolicy,
    UnderstandingEngine,
    UnderstandingRuntime,
)

UNDERSTANDING_MODULE = ModuleId("understanding_engine")
UNDERSTANDING_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class UnderstandingLayer:
    """Everything Flow 6 assembled, for tests and operators to reach into."""

    engine: UnderstandingEngine
    runtime: UnderstandingRuntime
    router: CapabilityRouter
    prompts: PromptProvider

    @property
    def understanders(self) -> tuple[str, ...]:
        return tuple(bound.adapter_id for bound in self.router.bound)


def build_routing_policy(platform: VisionPlatform) -> RoutingPolicy:
    """Translate configuration into a routing policy. Pure mapping."""
    settings = platform.config.understanding()
    return RoutingPolicy(
        prefer_local=settings.prefer_local_models,
        prefer_deterministic=settings.prefer_deterministic_models,
        prefer_coverage=settings.prefer_coverage,
        max_fallback_depth=settings.fallback_depth,
    )


def build_coercion(platform: VisionPlatform):
    """Select a coercion strategy by name.

    Raises:
        UnderstandingError: the configured strategy is unknown. Refusing beats
            defaulting: a typo that silently fell back to passthrough would stop
            every attribute in the deployment from ever parsing, and the symptom
            would be an empty platform rather than an error.
    """
    settings = platform.config.understanding()
    factory = COERCION_FACTORIES.get(settings.coercion_strategy)
    if factory is None:
        raise UnderstandingError(
            f"unknown coercion strategy '{settings.coercion_strategy}'; known "
            f"strategies are {sorted(COERCION_FACTORIES)}",
            requested=settings.coercion_strategy,
        )
    return factory()


def build_prompt_provider(
    templates: Sequence[PromptTemplate] = (),
    *,
    attributes: AttributeRegistry | None = None,
) -> StaticPromptProvider:
    """Build the M10 stand-in and validate its declarations.

    Validation here is a narrowed form of the ceiling's second gate: a prompt
    declaring an unregistered key is refused **at load**, so a broken prompt
    fails before it costs a model call rather than producing rejections forever.
    M10 will do this more thoroughly, on packs, with the neutrality check.

    Raises:
        UnderstandingError: a declared output key is not registered.
    """
    provider = StaticPromptProvider(templates)
    if attributes is not None:
        violations = provider.validate_against(attributes)
        if violations:
            raise UnderstandingError(
                "prompt validation failed: "
                + "; ".join(violations)
                + ". Attributes must pass the neutrality gate before a prompt may "
                "declare them (00_CHARTER section 4.3 gate 2).",
                violations=violations,
            )
    return provider


def build_understanding_layer(
    platform: VisionPlatform,
    cropping_layer: CroppingLayer,
    *,
    understanders: Sequence[UnderstanderPort] = (),
    fallbacks: Sequence[UnderstanderPort] = (),
    prompts: PromptProvider | None = None,
    prompt_templates: Sequence[PromptTemplate] = (),
    attributes: AttributeRegistry | None = None,
    coercion=None,
    understanding_sink=None,
    attach: bool = True,
) -> UnderstandingLayer:
    """Assemble Flow 6 against an already-built platform and Crop Manager.

    Args:
        understanders: Primary P15 adapters. **Empty is legal** and is the
            shipping default: 10_RELIABILITY §4.3 step 5 says that with no
            understanding available *"attributes stop; presence/spatial
            CONTINUE"*. The layer builds, reports the gap, and the rest of the
            platform is unaffected.
        fallbacks: Adapters reachable only through a fallback chain. Kept apart
            so a cheaper fallback can never win a primary route — otherwise the
            platform quietly runs on its worst model forever.
        attach: Wire the runtime to the Crop Manager's sink. False leaves the
            layer assembled but unconnected, which is what a unit test wants.

    Raises:
        UnderstandingError: understanding is disabled, an adapter failed its
            conformance kit, or a remote adapter was bound at a site that
            forbids remote residency.
    """
    settings = platform.config.understanding()

    if not settings.enabled:
        raise UnderstandingError(
            "understanding.enabled is false; a site that does not want attributes "
            "should not build the layer rather than build one that understands "
            "nothing"
        )

    registry = attributes if attributes is not None else AttributeRegistry()
    selected_coercion = coercion if coercion is not None else build_coercion(platform)
    _gate_adapter(platform, PortCatalogue.OUTPUT_COERCION, OUTPUT_COERCION_KIT, selected_coercion)

    router = CapabilityRouter(policy=build_routing_policy(platform))
    for adapter in understanders:
        router.bind(_bind(platform, adapter, settings, is_fallback=False))
    for adapter in fallbacks:
        router.bind(_bind(platform, adapter, settings, is_fallback=True))

    provider = (
        prompts
        if prompts is not None
        else build_prompt_provider(prompt_templates, attributes=registry)
    )

    platform_config = platform.config.platform()
    provenance = Provenance(
        producer_module=UNDERSTANDING_MODULE,
        producer_version=UNDERSTANDING_VERSION,
        config_revision=ConfigRevision(str(platform.config.revision())),
        deterministic=platform_config.deterministic,
    )

    engine = UnderstandingEngine(
        clock=platform.clock,
        metrics=platform.metrics,
        events=platform.bus,
        config=settings,
        router=router,
        prompts=provider,
        coercion=selected_coercion,
        attributes=registry,
        provenance=provenance,
        cache=ResponseCache(
            capacity=settings.cache_capacity,
            ttl=Duration.from_millis(settings.cache_ttl_ms),
        ),
    )

    runtime = UnderstandingRuntime(
        clock=platform.clock,
        metrics=platform.metrics,
        health=platform.health,
        engine=engine,
        config=settings,
        sink=understanding_sink,
    )

    if attach:
        _attach(cropping_layer, runtime)

    return UnderstandingLayer(
        engine=engine, runtime=runtime, router=router, prompts=provider
    )


def _bind(
    platform: VisionPlatform, adapter: UnderstanderPort, settings, *, is_fallback: bool
) -> BoundUnderstander:
    """Gate one understander and capture its capabilities.

    Capabilities are read **once**, here, rather than per request: a capability
    that changed under a running route would make two identical requests take
    different paths, which V13 forbids and no consumer could explain.
    """
    capabilities = adapter.capabilities()
    if capabilities.is_remote and not settings.allow_remote_understanders:
        raise UnderstandingError(
            f"understander '{adapter.adapter_id}' declares residency "
            f"'{capabilities.data_residency}' but this site forbids remote "
            f"understanders; a residency policy must be enforced at binding, not "
            f"discovered in an export audit (12_SECURITY)",
            adapter_id=adapter.adapter_id,
        )
    _gate_adapter(platform, PortCatalogue.UNDERSTANDER, UNDERSTANDER_KIT, adapter)
    return BoundUnderstander(
        adapter=adapter, capabilities=capabilities, is_fallback=is_fallback
    )


def _gate_adapter(platform: VisionPlatform, port, kit, adapter) -> None:
    """Run an adapter's conformance kit before it is ever used.

    The registered kit wins over the shipped one when a deployment registered its
    own, but *some* kit must run. For P15 this is the gate that catches
    fabrication-on-failure, which is otherwise undetectable downstream.
    """
    registered = platform.conformance.get(port)
    effective = registered or kit
    report = effective.run(adapter, fast_only=True)
    if not report.passed:
        raise UnderstandingError(
            f"adapter for {port} failed conformance: {'; '.join(report.failures)}",
            port=str(port),
        )


def _attach(cropping_layer: CroppingLayer, runtime: UnderstandingRuntime) -> None:
    """Wire the understanding runtime to the Crop Manager's declared sink.

    M8's sink is synchronous while ``on_crops`` is a coroutine, so the hand-off
    is scheduled rather than awaited. That is the point: 08_RUNTIME §5.2 gives
    this edge `drop_oldest` because *"losing an enrichment is acceptable"*, and
    awaiting would put a 2-second VLM call on the critical path of a layer whose
    budget is measured in microseconds.
    """
    import asyncio

    existing = getattr(cropping_layer.runtime, "_sink", None)

    def sink(result, crops) -> None:
        if existing is not None:
            existing(result, crops)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop — a synchronous harness. Run to completion so the crops are
            # not silently dropped.
            asyncio.run(runtime.on_crops(result, crops))
            return
        loop.create_task(runtime.on_crops(result, crops))

    cropping_layer.runtime._sink = sink  # noqa: SLF001 - the declared seam
