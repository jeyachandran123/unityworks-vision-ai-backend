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


#: Settings of the ``region_head`` provider. Paths may be relative to the
#: working directory or to the repository that vendors this package.
REGION_HEAD_WEIGHTS_ENV = "VISION_REGION_HEAD_WEIGHTS"
REGION_HEAD_LABELS_ENV = "VISION_REGION_HEAD_LABELS"
REGION_HEAD_MIN_SCORE_ENV = "VISION_REGION_HEAD_MIN_SCORE"
REGION_HEAD_MARGIN_ENV = "VISION_REGION_HEAD_MARGIN"
REGION_HEAD_INPUT_ENV = "VISION_REGION_HEAD_INPUT"


def _resolve(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        # `<repository root>/value`: configuration → adapters → vision_os → root.
        path = Path(__file__).resolve().parents[3] / value
    return path


def _graph_input_size(weights: Path, setting: str) -> int:
    """The square input the graph was exported for, read from the graph.

    Taken from the file rather than a setting because the two disagreeing is a
    silent accuracy loss, not an error: a 448 crop fitted into a 640 graph is
    enlarged by nearest neighbour, and on kitchen-01 that cost the head band 15
    points (73.2 % against 87.8 % for the same model exported at 448). A
    setting is accepted only when it agrees, or when the graph's input is
    dynamic and something has to choose.
    """
    import onnxruntime  # noqa: PLC0415 - optional, adapter-scoped

    shape = onnxruntime.InferenceSession(
        str(weights), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].shape
    static = shape[-1] if len(shape) == 4 and isinstance(shape[-1], int) else None
    if static is not None:
        if setting and int(setting) != static:
            raise ProviderConfigurationError(
                f"{REGION_HEAD_INPUT_ENV}={setting} contradicts '{weights.name}', which "
                f"was exported for {static}x{static}. Leave it unset."
            )
        return int(static)
    return int(setting or 640)


def _build_region_head(
    *, producible: Sequence[AttributeKey], env: Mapping[str, str], **_: Any
) -> UnderstanderPort:
    """A local model that labels regions of a crop, as an ``attr.*`` head.

    The weights are an ONNX graph in the usual ``(1, 4 + classes, anchors)``
    layout, opened here with the same session the detector adapter uses, so the
    head inherits its letterbox and decoding. The label mapping is a JSON file
    naming, per attribute, which model label means which registered value —
    the vocabulary is data, and this module only checks that every label it
    names is one the model can actually report.
    """
    import hashlib
    import json

    from ..detection.letterbox import compute_transform, invert_to_normalized
    from ..models.runtimes import LetterboxedImage, open_onnx_detector_session
    from ..understanding.region_head import RegionLabelHead

    weights_setting = _setting(env, REGION_HEAD_WEIGHTS_ENV)
    labels_setting = _setting(env, REGION_HEAD_LABELS_ENV)
    if not weights_setting or not labels_setting:
        raise ProviderConfigurationError(
            f"provider 'region_head' needs {REGION_HEAD_WEIGHTS_ENV} (an ONNX graph) "
            f"and {REGION_HEAD_LABELS_ENV} (its label mapping)."
        )
    weights, labels_path = _resolve(weights_setting), _resolve(labels_setting)
    if not weights.is_file():
        raise ProviderConfigurationError(f"region head weights not found at '{weights}'")
    if not labels_path.is_file():
        raise ProviderConfigurationError(f"region head label mapping not found at '{labels_path}'")

    try:
        mapping = json.loads(labels_path.read_text(encoding="utf-8"))
        label_values = {
            str(attribute): {str(label): str(value) for label, value in labels.items()}
            for attribute, labels in mapping["attributes"].items()
        }
    except (ValueError, KeyError, AttributeError) as exc:
        raise ProviderConfigurationError(
            f"region head label mapping '{labels_path}' is not valid: {exc}"
        ) from exc

    min_score = float(_setting(env, REGION_HEAD_MIN_SCORE_ENV, default="0.4"))
    size = _graph_input_size(weights, _setting(env, REGION_HEAD_INPUT_ENV))
    session = open_onnx_detector_session(str(weights), conf_threshold=min_score)
    names = list(session.class_names())
    unknown = sorted(
        {label for labels in label_values.values() for label in labels} - set(names)
    )
    if unknown:
        raise ProviderConfigurationError(
            f"region head mapping names label(s) {unknown} that '{weights.name}' cannot "
            f"report; it reports {names}. A misspelt label would never match and "
            f"every answer would read as not visible."
        )

    def score_regions(pixels, width, height, colour_space):
        import numpy as np  # noqa: PLC0415 - optional, adapter-scoped

        image = np.frombuffer(pixels, dtype=np.uint8)[: width * height * 3].reshape(height, width, 3)
        if str(colour_space).lower().startswith("rgb"):
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)
        transform = compute_transform(
            source_width=width, source_height=height, target_width=size, target_height=size
        )
        raw = session.infer(
            [LetterboxedImage(pixels=memoryview(image.tobytes()), width=width,
                              height=height, transform=transform)]
        )[0]
        regions = []
        for item in raw:
            box = invert_to_normalized(transform, item.x1, item.y1, item.x2, item.y2)
            regions.append(
                (names[item.class_index], float(item.score), box.x1, box.y1, box.x2, box.y2)
            )
        return regions

    digest = hashlib.blake2b(weights.read_bytes(), digest_size=16).hexdigest()
    return RegionLabelHead(
        score_regions=score_regions,
        label_values=label_values,
        producible=producible,
        model_id=str(mapping.get("model") or weights.stem),
        artifact_hash=f"blake2b:{digest}",
        min_score=min_score,
        centre_margin=float(_setting(env, REGION_HEAD_MARGIN_ENV, default="0.15")),
        unseen_value=str(mapping.get("unseen_value") or "not_visible"),
    )


#: Providers selectable by configuration. A closed table, like the tracker,
#: crop-strategy and coercion factories.
UNDERSTANDER_FACTORIES: Mapping[str, Any] = {
    "nvidia": _build_nvidia,
    "ollama": _build_ollama,
    "region_head": _build_region_head,
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
