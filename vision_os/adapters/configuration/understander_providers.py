"""Which understander a deployment binds — resolved here and nowhere else.

`adapters/cropping` and `adapters/exposure` already establish the pattern this
follows: a **closed factory table**, keyed by a configuration string. *"A
deployment names a strategy; it does not import one."* ``UNDERSTANDER_FACTORIES``
is the same idea for P15.

### Why resolution lives in this file and not in the layer

``build_understanding_layer`` takes ``understanders: Sequence[UnderstanderPort]``
— adapters are *passed in*, never named by string. That is deliberate: the
moment the platform resolves a provider name, it knows which providers exist,
and P15's whole claim (that the platform is indifferent to what answered)
becomes false. So the string never reaches the layer. It is resolved here, in
the adapter package, and a bound object is handed over.

### Why this file lives in `adapters/configuration` and not beside the adapters

It was written next to them first, and the architecture suite rejected it:

    adapters/understanding/providers.py imports os
    adapters/understanding/providers.py imports pathlib

§M9 requires that the understanding adapters own **no world state** — *"every
call is a pure function of (crop, prompt, model)"* — and the test enforces it by
forbidding `os`, `pathlib` and every other ambient-state import anywhere under
that package. A module that reads the environment is ambient state by
definition, so it cannot live there no matter how convenient the import would be.

This package is where that already belongs: P23 config sources and P24 secret
providers read files and environments here, and nothing in the understanding
path does.

### Why the environment is read here and not in an adapter

An adapter that read ``os.environ`` could not be constructed twice with
different settings, could not be tested without mutating the process, and would
make credentials its business. Every understanding adapter takes explicit
constructor arguments. Configuration is this module's job — it is the seam
between a deployment's environment and the platform's contracts.

Nothing here imports ``app.config``. Vision OS must stay runnable without the
Atlas backend around it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...core.model.ids import AttributeKey
from ...core.ports.understanding import UnderstanderPort

#: Primary selector for CCTV/object understanding.
PROVIDER_ENV = "VISION_UNDERSTANDER_PROVIDER"

#: Consulted only when the primary is unset.
#:
#: The Document Platform's switch, honoured for backward compatibility with
#: deployments configured before vision understanding had its own. They are
#: **different ports** — ``DocumentVLMPort`` turns a page into an invoice, P15
#: turns a crop into attributes — so a site that eventually wants a local model
#: reading crops and a hosted one reading invoices can set the primary and stop
#: sharing a switch. Until then, one variable keeps working.
LEGACY_PROVIDER_ENV = "DOCUMENT_VLM_PROVIDER"

DEFAULT_PROVIDER = "static"


class ProviderConfigurationError(ValueError):
    """A provider was named that cannot be built. Raised at composition time.

    Never at request time: a deployment that cannot bind an understander must
    fail while someone is watching the logs, not on the first crop of a live
    session.
    """


# --- environment resolution ------------------------------------------------------ #


def resolve_provider_name(env: Mapping[str, str] | None = None) -> str:
    """Which provider this deployment asked for, lower-cased.

    Primary, then legacy, then the static head — which is the only default that
    cannot surprise anyone: it needs no key, no network and no weights, and it
    announces itself as a constant rather than pretending to be a model.
    """
    source = os.environ if env is None else env
    for key in (PROVIDER_ENV, LEGACY_PROVIDER_ENV):
        value = (source.get(key) or "").strip().lower()
        if value:
            return value
    return DEFAULT_PROVIDER


def read_env_file(path: Path | str) -> dict[str, str]:
    """Parse a ``.env`` file into a plain mapping. Absent file → empty mapping.

    Exists so a deployment can keep one copy of a credential that several
    services share, rather than duplicating a key per component and discovering
    at rotation time that one copy was missed.
    """
    values: dict[str, str] = {}
    file = Path(path)
    if not file.is_file():
        return values
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        values[name.strip()] = raw.split("#", 1)[0].strip().strip("'\"")
    return values


def _setting(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return default


# --- factories ------------------------------------------------------------------- #


def _build_nvidia(
    *, producible: Sequence[AttributeKey], env: Mapping[str, str], **_: Any
) -> UnderstanderPort:
    from ..understanding.nvidia_vl import DEFAULT_BASE_URL, DEFAULT_MODEL, NvidiaVisionUnderstander

    api_key = _setting(env, "VISION_NVIDIA_API_KEY", "NVIDIA_API_KEY")
    if not api_key:
        raise ProviderConfigurationError(
            "provider 'nvidia' needs an API key. Set NVIDIA_API_KEY (or "
            "VISION_NVIDIA_API_KEY to give vision understanding its own). No key "
            "is read from source, and none is defaulted."
        )
    return NvidiaVisionUnderstander(
        producible=producible,
        api_key=api_key,
        base_url=_setting(env, "VISION_NVIDIA_BASE_URL", "NVIDIA_BASE_URL",
                          default=DEFAULT_BASE_URL),
        model=_setting(env, "VISION_NVIDIA_MODEL", "NVIDIA_MODEL", default=DEFAULT_MODEL),
        timeout_s=float(_setting(env, "VISION_UNDERSTANDER_TIMEOUT_S", default="60")),
    )


def _build_ollama(
    *, producible: Sequence[AttributeKey], env: Mapping[str, str], **_: Any
) -> UnderstanderPort:
    from ..understanding.ollama_vl import DEFAULT_ENDPOINT, DEFAULT_MODEL, OllamaVisionUnderstander

    return OllamaVisionUnderstander(
        producible=producible,
        endpoint=_setting(env, "VISION_OLLAMA_BASE_URL", "OLLAMA_BASE_URL",
                          default=DEFAULT_ENDPOINT),
        model=_setting(env, "VISION_OLLAMA_MODEL", "VISION_MODEL", default=DEFAULT_MODEL),
        timeout_s=float(_setting(env, "VISION_UNDERSTANDER_TIMEOUT_S", default="180")),
    )


def _build_static(
    *, producible: Sequence[AttributeKey], env: Mapping[str, str], **_: Any
) -> UnderstanderPort:
    """The constant-answer head. Needs nothing, and says so.

    Bound to the first producible attribute because a head produces exactly one.

    **The value must be a member of that attribute's registered domain**, and
    nothing here can check that — the registry lives on the other side of the
    port. This first defaulted to ``"unknown"``, which is not in any enum a
    deployment is likely to register: the head answered, the validator rejected
    every field as out-of-domain, and the pipeline produced no attributes at all
    while appearing to work. The platform was right and the default was wrong.

    So the composition root supplies it through ``defaults``, because the
    composition root is the only place that knows both the vocabulary and the
    head. Absent that, the value is empty and the validator rejects it loudly —
    which is the correct failure, and much easier to diagnose than a plausible
    constant that silently never validates.
    """
    from ..understanding.understanders import StaticAttributeHead

    return StaticAttributeHead(
        attribute=producible[0],
        value=_setting(env, "VISION_STATIC_VALUE"),
    )


#: Providers selectable by configuration. A closed table, like the tracker,
#: crop-strategy and coercion factories.
UNDERSTANDER_FACTORIES: Mapping[str, Any] = {
    "nvidia": _build_nvidia,
    "ollama": _build_ollama,
    "static": _build_static,
}


def build_understander(
    *,
    producible: Sequence[AttributeKey],
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
    env_file: Path | str | None = None,
    defaults: Mapping[str, str] | None = None,
) -> tuple[UnderstanderPort, str]:
    """Build the configured understander. Returns the adapter and a note.

    The note travels to the UI so an operator can see *which* adapter answered
    and why it was chosen — 10_RELIABILITY §7.2's rule that a binding decision is
    never silent.

    Args:
        defaults: Deployment defaults from the composition root, overridable by
            the file and the environment. This is how a caller supplies a value
            only it can know — the constant head's answer has to be a member of
            an attribute's registered domain, and the registry is not visible
            from here.

    Raises:
        ProviderConfigurationError: the named provider is unknown, or its
            configuration is incomplete. Both are composition-time failures.
    """
    if not producible:
        raise ProviderConfigurationError(
            "no producible attributes were supplied; an understander that can "
            "produce nothing can never be routed to and would never be selected"
        )

    # The file supplies **credentials**; the environment supplies **selection**.
    #
    # These are resolved from different places on purpose. `env_file` is
    # typically a neighbouring service's `.env`, read so one credential does not
    # have to be copied per component. But that file also contains *its* provider
    # setting, and merging it before resolving the name let a document-platform
    # variable silently choose the CCTV understander — a deployment picking up a
    # decision from a file it does not own. Selection stays where the operator
    # set it.
    selection = os.environ if env is None else env
    name = (provider or resolve_provider_name(selection)).strip().lower()

    merged: dict[str, str] = dict(defaults or {})
    if env_file is not None:
        merged.update(read_env_file(env_file))
    merged.update(selection)
    factory = UNDERSTANDER_FACTORIES.get(name)
    if factory is None:
        raise ProviderConfigurationError(
            f"{PROVIDER_ENV}='{name}' is not a registered understander provider. "
            f"Available: {', '.join(sorted(UNDERSTANDER_FACTORIES))}."
        )

    adapter = factory(producible=tuple(producible), env=merged)
    return adapter, f"{name} ({adapter.adapter_id})"


__all__ = [
    "DEFAULT_PROVIDER",
    "LEGACY_PROVIDER_ENV",
    "PROVIDER_ENV",
    "UNDERSTANDER_FACTORIES",
    "ProviderConfigurationError",
    "build_understander",
    "read_env_file",
    "resolve_provider_name",
]
