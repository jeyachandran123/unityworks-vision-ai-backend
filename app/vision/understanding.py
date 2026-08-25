"""Binding the understanding layer — the capability Phase 3 left open.

### The composition, in order

    policies
        ↓
    canonical AttributeRegistry ─────┐
        ↓                            │  the SAME object
    registry layer (M7) ◄────────────┤
        ↓                            │
    cropping layer (M8)              │
        ↓                            │
    understanding layer (M9) ◄───────┘
        ↓
    RegistryWriteBackSink → RegistryEngine.apply_attribute → M7

The registry is built once and passed to both M7 and M9. `assert_shared_registry`
checks it by **identity** at the end of assembly and raises rather than
continuing, because a composition with two registries runs perfectly and is
silently wrong — which is exactly how Phase 6 lost nine sub-phases.

### Provider selection

Through the platform's own `build_understander`, which reads
`VISION_UNDERSTANDER_PROVIDER`. This module names no model, no URL and no API
key, and contains no `if provider == "nvidia"`. Adding a provider is a change to
the platform's provider registry and nothing here.

### What this module does not do

It performs no inference, holds no prompt text, and knows no domain vocabulary.
`head_covering` and `hairnet` appear in policy documents, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.errors import ConfigurationInvalidError
from app.vision.composition import registry_of
from app.vision.writeback import RegistryWriteBackSink


class SharedRegistryViolation(RuntimeError):
    """M7 and M9 hold different AttributeRegistry instances. Fails assembly.

    Not repaired by constructing a third registry — a composition that cannot
    say which registry is canonical has a wiring defect, and papering over it
    would restore the exact silence Phase 6 spent nine sub-phases breaking.
    """


@dataclass(frozen=True, slots=True)
class UnderstandingComposition:
    """The assembled Flow 5/6 pair, plus the sink that closes M9 → M7."""

    cropping: Any
    understanding: Any
    sink: RegistryWriteBackSink
    provider_id: str
    provider_note: str
    producible: tuple[str, ...]

    def to_wire(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "provider_note": self.provider_note,
            "producible_attributes": list(self.producible),
            "understanders": list(getattr(self.understanding, "understanders", ())),
            "writeback": self.sink.audit.to_wire(),
        }


def build_understanding(
    platform: Any,
    registry_layer: Any,
    attributes: Any,
    *,
    policies: tuple[Any, ...] = (),
    provider: str | None = None,
    static_value: str | None = None,
    api_key: str = "",
) -> UnderstandingComposition:
    """Assemble cropping and understanding over an existing platform and M7.

    Args:
        attributes: The **canonical** registry. The same object M7 was built
            with; passing a different one raises at the end of this function.
        provider: Overrides `VISION_UNDERSTANDER_PROVIDER`. For tests that need
            a specific adapter without touching process environment.

    Raises:
        ConfigurationInvalidError: no attribute is declared, or the provider
            could not be built.
        SharedRegistryViolation: M7 and M9 ended up with different registries.
    """
    from vision_os.adapters.configuration.understander_providers import (
        ProviderConfigurationError,
        build_understander,
    )
    from vision_os.cropping_bootstrap import build_cropping_layer
    from vision_os.perception.cropping import CapabilityView
    from vision_os.understanding_bootstrap import build_understanding_layer

    producible = _declared_keys(attributes)
    if not producible:
        # An understander that can produce nothing can never be routed to. This
        # is a configuration error, not a degraded mode, and it is reported at
        # assembly rather than discovered as permanent silence at runtime.
        raise ConfigurationInvalidError(
            "no attribute is declared; load a semantic policy before binding "
            "understanding, or the layer can never be routed to"
        )

    # ── the adapter ──────────────────────────────────────────────────────────
    #
    # The platform decides which one from its own configuration. The static head
    # needs a value inside the attribute's registered domain, and the registry is
    # not visible from the provider module — so the caller supplies it as a
    # default the environment can still override.
    defaults: dict[str, str] = {}
    if static_value:
        defaults["VISION_STATIC_VALUE"] = static_value
    if api_key:
        # Named for the platform's own key, so this composition root does not
        # decide which provider the credential belongs to.
        defaults["VISION_NVIDIA_API_KEY"] = api_key
    try:
        adapter, note = build_understander(
            producible=tuple(_attribute_keys(attributes)),
            provider=provider,
            defaults=defaults or None,
        )
    except ProviderConfigurationError as exc:
        # Names the missing configuration without quoting a credential.
        raise ConfigurationInvalidError(
            f"the understanding provider could not be built: {exc}"
        ) from exc

    # ── Flow 5: what deserves an expensive look ──────────────────────────────
    #
    # Capability is declared from what the bound adapter can actually produce.
    # Without it every demand is admitted and immediately marked unsatisfiable
    # with NO_CAPABLE_MODEL — accepted and left waiting forever.
    cropping = build_cropping_layer(
        platform,
        registry_layer,
        capabilities=_capability_view(CapabilityView, adapter, attributes),
        evidence_regions=_evidence_regions(policies),
        output_sizes=_output_sizes(policies),
    )

    # ── Flow 6: extract meaning, and hold it in M7 ───────────────────────────
    #
    # `prompt_templates` is the seam that was left empty. Without it the prompt
    # provider holds nothing, and M9 refuses every request with
    # `PROMPT_UNAVAILABLE` → `UnderstandingOutcome.UNSUPPORTED` *before* any
    # model call. On real CCTV that presented as 120 crops taken, 120 results
    # produced, 120 failed, zero attributes and zero write-backs — with no
    # error anywhere, because refusing to ask is the correct behaviour when
    # there is no declared question.
    templates = _prompt_templates(policies)
    sink = RegistryWriteBackSink(registry_layer.registry)
    understanding = build_understanding_layer(
        platform,
        cropping,
        understanders=(adapter,),
        prompt_templates=templates,
        attributes=attributes,  # ← the canonical instance
        understanding_sink=sink,
    )

    assert_shared_registry(registry_layer, understanding, attributes)

    logger.info(
        "understanding bound — provider={} producible={} ({})",
        getattr(adapter, "adapter_id", "?"),
        len(producible),
        note,
    )

    return UnderstandingComposition(
        cropping=cropping,
        understanding=understanding,
        sink=sink,
        provider_id=str(getattr(adapter, "adapter_id", "?")),
        provider_note=note,
        producible=producible,
    )


def assert_shared_registry(registry_layer: Any, understanding: Any, attributes: Any) -> None:
    """Verify by **identity** that one registry reached both M7 and M9.

    Identity, not equality: two registries built from the same documents compare
    equal and drift the moment one side reloads a policy — and the drift surfaces
    as an `AttributeRejectedError` for an attribute the operator can see declared
    in their own policy file.
    """
    m7 = registry_of(registry_layer)
    if m7 is None:
        raise SharedRegistryViolation(
            "M7 holds no AttributeRegistry; it has nothing to validate a "
            "write-back against and will refuse every attribute"
        )
    if m7 is not attributes:
        raise SharedRegistryViolation(
            "M7 holds a different AttributeRegistry instance than the one this "
            "composition built — every M9 write-back will be rejected and "
            "FRESH_ENOUGH will never fire"
        )

    m9 = registry_of(understanding)
    if m9 is not None and m9 is not attributes:
        raise SharedRegistryViolation(
            "the understanding layer holds a different AttributeRegistry " "instance than M7"
        )


# ── helpers ──────────────────────────────────────────────────────────────────


def _attribute_keys(attributes: Any) -> list[Any]:
    schemas = getattr(attributes, "schemas", None)
    return list(schemas.keys()) if isinstance(schemas, dict) else []


def _prompt_templates(policies: tuple[Any, ...]) -> tuple[Any, ...]:
    """The prompt each policy declares, as the provider's own template type.

    **The policy owns the wording; this owns nothing.** `SemanticPolicy` already
    knows how to render itself into a `PromptTemplate` — preamble, per-attribute
    question, output keys, applicable classes and token ceiling all come from the
    document. This function calls that and collects the results.

    So no question text, no attribute name and no domain vocabulary occurs in
    this function. Two policies from entirely different domains register by the
    same path, and adding a third needs no change here (§6) — a property a test
    enforces by reading this source.

    A policy that declares no prompt is skipped rather than given a generated
    one. An improvised question would produce a confident answer to something
    nobody wrote down, which is worse than the honest `PROMPT_UNAVAILABLE` that
    M9 raises instead (§9).
    """
    templates = []
    for policy in policies or ():
        build = getattr(policy, "build_prompt_template", None)
        if not callable(build):
            continue
        try:
            template = build()
        except Exception as exc:  # noqa: BLE001 - one policy, not the assembly
            # Named, never swallowed: a policy whose prompt will not build is a
            # capability this deployment silently lacks, and the symptom
            # downstream is a rule that is permanently UNKNOWN.
            logger.error(
                "policy '{}' declares a prompt that could not be built: {}: {}",
                getattr(policy, "policy_id", "?"),
                type(exc).__name__,
                exc,
            )
            continue
        if template is not None:
            templates.append(template)
    return tuple(templates)


def _declared_keys(attributes: Any) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in _attribute_keys(attributes)))


def _capability_view(capability_view: Any, adapter: Any, attributes: Any) -> Any:
    """What the platform can actually produce, from the bound adapter.

    Two sets, and the difference between them is the useful part:

    * `registered_attributes` — everything policy declared.
    * `producible_attributes` — the subset the *bound adapter* can answer.

    Asked of the adapter rather than assumed from policy, because a policy may
    declare an attribute no bound model can produce. The honest outcome then is
    `NO_CAPABLE_MODEL` at demand time rather than a demand that waits forever,
    and a deployment learns at startup that a rule can never reach a verdict.
    """
    from vision_os.core.model.ids import ClassId

    registered = frozenset(_attribute_keys(attributes))
    capabilities = adapter.capabilities()
    producible = frozenset(key for key in registered if capabilities.can_produce((key,)))

    # The classes policy scopes its attributes to. Read from the schemas rather
    # than named here — `person` is a domain fact and belongs in the document.
    classes: set[Any] = set()
    schemas = getattr(attributes, "schemas", {})
    if isinstance(schemas, dict):
        for schema in schemas.values():
            classes.update(ClassId(str(c)) for c in getattr(schema, "applies_to", ()))

    return capability_view(
        registered_attributes=registered,
        producible_attributes=producible,
        producible_classes=frozenset(classes),
    )


def _evidence_regions(policies: tuple[Any, ...]) -> dict[Any, Any] | None:
    """Per-attribute crop geometry, from the policy documents.

    The head band and the hand band come from `kitchen-safety.example.json`.
    Not one region is written here — the geometry is a domain decision and lives
    in the document that owns the domain.
    """
    regions: dict[Any, Any] = {}
    for policy in policies:
        found = getattr(policy, "evidence_regions", None)
        if isinstance(found, dict):
            regions.update(found)
    return regions or None


def _output_sizes(policies: tuple[Any, ...]) -> dict[Any, Any] | None:
    """Per-attribute crop resolution — head 448, hands 224.

    Measured in Phase 4.2: head accuracy 23.3% → 74.4%, FALSE ABSENT 20 → 4.
    Read from policy, never chosen here.
    """
    sizes: dict[Any, Any] = {}
    for policy in policies:
        found = getattr(policy, "output_sizes", None)
        if isinstance(found, dict):
            sizes.update(found)
    return sizes or None


__all__ = [
    "SharedRegistryViolation",
    "UnderstandingComposition",
    "assert_shared_registry",
    "build_understanding",
]
