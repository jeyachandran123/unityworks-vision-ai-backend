"""The Vision OS composition root.

**This module is composition, not logic.** It calls the platform's own bootstrap
functions and wires their outputs together. It contains no detection, tracking,
cropping, understanding or compliance code, and a boundary test
(`tests/app/test_vision_boundary.py`) reads this package's source to keep it so.

### The one wiring detail that matters more than the rest

    attributes     = build_attribute_registry(policies)
    registry_layer = build_registry_layer(platform, attributes=attributes)
    understanding  = build_understanding_layer(..., attributes=attributes)
                                                    ^^^^^^^^^^^^^^^^^^^^^
                                          the SAME object, not an equal one

Phase 6 spent nine sub-phases and five discarded hypotheses discovering that
`build_registry_layer` was not receiving `attributes=`. Understanding and the
object registry therefore held *different* `AttributeRegistry` instances, M7
refused **308 of 308** attributes M9 had successfully produced, and
`SkipReason.FRESH_ENOUGH` had never once fired in the platform's life. Nothing
downstream could tell: understanding reported zero failures, the sink reported
zero failures, and the platform silently re-asked the VLM for an answer it
already had, on every frame, forever.

`assert_shared_attribute_registry()` re-checks it at assembly time by object
**identity**, and a regression test asserts the same property. Equality is not
enough — two registries built from the same documents compare equal and drift
the moment one side reloads a policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import ConfigurationInvalidError, VisionUnavailableError


@dataclass(frozen=True, slots=True)
class VisionComposition:
    """An assembled platform, plus the objects the application must hold onto."""

    system: Any
    api: Any
    platform: Any
    attributes: Any
    registry_layer: Any
    cropping: Any = None
    understanding: Any = None
    detection: Any = None
    tracking: Any = None
    policies: tuple[Any, ...] = ()
    #: Flow 7: the observation builder, the state manager and the log. Without
    #: it M7 holds attributes that no consumer can read — `synthesis.built` and
    #: `state.appended` stay at zero and the Observation API has nothing to
    #: serve, which is indistinguishable from a camera that saw nothing.
    synthesis: Any = None
    #: Flow 8: the ObservationApi, its authorizer, hub and audit trail. `None`
    #: means Vision State holds observations that no consumer can reach.
    exposure: Any = None
    #: Flow 5/6 and the M9 → M7 sink, when a provider was configured.
    understanding_composition: Any = None

    @property
    def declared_attributes(self) -> tuple[str, ...]:
        return declared_keys(self.attributes)


class SharedRegistryViolation(RuntimeError):
    """Two AttributeRegistry instances reached the pipeline. Raised at assembly.

    At assembly rather than at runtime, because the symptom of this bug is
    silence: write-backs are rejected, freshness never fires, and the only
    visible effect is a VLM bill that scales with frames instead of with change.
    """


def registry_of(layer: Any) -> Any | None:
    """The ``AttributeRegistry`` a layer validates against.

    The platform's layers hold it at different depths and under different names —
    ``RegistryLayer`` carries an ``ObjectRegistry`` which holds
    ``_attribute_registry``; other layers expose it directly. Probing rather than
    reaching for one fixed attribute keeps this working across the layers without
    the application asserting a private shape that the platform never promised.

    Returns ``None`` when the layer holds no registry, which is a fact worth
    reporting rather than an error.
    """
    candidates = (
        layer,
        getattr(layer, "engine", None),
        getattr(layer, "registry", None),
        getattr(layer, "runtime", None),
    )
    for holder in candidates:
        if holder is None:
            continue
        for name in ("_attribute_registry", "attributes", "attribute_registry"):
            found = getattr(holder, name, None)
            # `require` is the registry's validation entry point; matching on it
            # avoids mistaking a plain dict of attribute values for the registry.
            if found is not None and hasattr(found, "require"):
                return found
    return None


def declared_keys(attributes: Any) -> tuple[str, ...]:
    """Attribute keys this registry has granted, sorted.

    Empty is a valid answer and a working configuration: no attributes means no
    demand, no crops and no model calls, while detection, tracking, the registry
    and the Observation API run exactly as before.

    ``AttributeRegistry.schemas`` is a ``dict[AttributeKey, AttributeSchema]`` —
    a field, not a method. Reading it directly rather than probing, because this
    is the platform's declared shape and a silent fallback would report "no
    attributes" for a registry that holds several, which is exactly the kind of
    quiet wrong answer Phase 6 spent nine sub-phases chasing.
    """
    schemas = getattr(attributes, "schemas", None)
    if isinstance(schemas, dict):
        return tuple(sorted(str(key) for key in schemas))
    return ()


def assert_shared_attribute_registry(
    registry_layer: Any, understanding: Any, attributes: Any
) -> None:
    """Verify by **identity** that one registry reached both M7 and M9.

    Raises:
        SharedRegistryViolation: the instances differ, or M7 holds none.
    """
    m7 = registry_of(registry_layer)
    if m7 is None:
        raise SharedRegistryViolation(
            "the registry layer holds no AttributeRegistry; M7 has nothing to "
            "validate a write-back against and will refuse every attribute"
        )
    if m7 is not attributes:
        raise SharedRegistryViolation(
            "M7 holds a different AttributeRegistry instance than the one built "
            "for this composition — every M9 write-back will be rejected and "
            "FRESH_ENOUGH will never fire"
        )

    if understanding is not None:
        m9 = registry_of(understanding)
        if m9 is not None and m9 is not attributes:
            raise SharedRegistryViolation(
                "the understanding layer holds a different AttributeRegistry " "instance than M7"
            )


def load_policies(policy_paths: str) -> tuple[Any, ...]:
    """Load semantic policy documents from a comma-separated path list.

    An empty setting is valid and means *no policy*: the platform then declares
    no attributes, demands nothing, spends no model calls, and says so through
    its capability summary. Detection, tracking, the registry, Vision State and
    the Observation API all run exactly as before.

    A path that is *named and missing* is different, and raises. Silently
    skipping it would leave a deployment believing a policy is in force.
    """
    from vision_os.adapters.configuration.semantic_policy import SemanticPolicy

    raw = (policy_paths or "").strip()
    if not raw:
        return ()

    policies = []
    for chunk in raw.split(","):
        candidate = chunk.strip()
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            raise ConfigurationInvalidError(
                f"semantic policy document not found: {path}",
                details={"path": str(path)},
            )
        policies.append(SemanticPolicy.from_file(str(path)))
    return tuple(policies)


def build_attribute_registry(policies: tuple[Any, ...] = ()) -> Any:
    """The one canonical registry for this composition. Built exactly once.

    Registration goes through the platform's own ``AttributeRegistry``, so every
    entry passes the neutrality gate: a policy may *ask* for a concept, and only
    the registry can *grant* it. An attribute whose name encodes a verdict is
    refused here, which is what keeps the Semantic Ceiling intact no matter what
    a policy document asks for.
    """
    from vision_os.perception.registry.attributes import AttributeRegistry

    registry = AttributeRegistry()
    for policy in policies:
        if policy is not None:
            policy.register_attributes(registry)
    return registry


def describe_composition(composition: VisionComposition) -> dict[str, Any]:
    """A safe, structured summary for diagnostics.

    Holds no imagery, no credentials and no principal. It answers "what is this
    platform configured to observe", which is the question an operator debugging
    a silent camera actually has.
    """
    return {
        "attributes": list(composition.declared_attributes),
        "policies": [
            {
                "policy_id": str(getattr(p, "policy_id", "")),
                "version": str(getattr(p, "version", "")),
            }
            for p in composition.policies
        ],
        "shared_registry": registry_of(composition.registry_layer) is composition.attributes,
    }


def require_composition(composition: VisionComposition | None) -> VisionComposition:
    """Unwrap an optional composition, or fail with the right error.

    ``VisionUnavailableError`` rather than a generic 500, because "the platform
    is not running" and "the platform observed nothing" must never look the same
    to a consumer (invariant V8).
    """
    if composition is None:
        raise VisionUnavailableError(
            "Vision OS is not assembled in this process; no observation can be "
            "served, and that is not the same as observing nothing"
        )
    return composition


__all__ = [
    "SharedRegistryViolation",
    "VisionComposition",
    "assert_shared_attribute_registry",
    "build_attribute_registry",
    "declared_keys",
    "describe_composition",
    "load_policies",
    "registry_of",
    "require_composition",
]
