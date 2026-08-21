"""Provider selection: configuration in, a bound P15 adapter out.

The tests that matter here are the negative ones. A provider table is easy to
get right in the happy case and easy to get wrong in the ways that hurt: a
typo'd name silently falling back to something that answers, a missing key
producing an adapter that fails on the first crop instead of at boot, or the
selection mechanism quietly changing what the port does.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.understanding import (
    NvidiaVisionUnderstander,
    OllamaVisionUnderstander,
    StaticAttributeHead,
)
from vision_os.adapters.configuration.understander_providers import (
    DEFAULT_PROVIDER,
    LEGACY_PROVIDER_ENV,
    PROVIDER_ENV,
    UNDERSTANDER_FACTORIES,
    ProviderConfigurationError,
    build_understander,
    read_env_file,
    resolve_provider_name,
)
from vision_os.conformance.understanding_kits import UNDERSTANDER_KIT
from vision_os.core.model.ids import AttributeKey
from vision_os.core.ports.understanding import UnderstanderPort

PRODUCIBLE = (AttributeKey("posture"), AttributeKey("colour"))
KEYED = {"NVIDIA_API_KEY": "nvapi-test"}


# --- name resolution ------------------------------------------------------------ #


def test_primary_env_selects_the_provider():
    assert resolve_provider_name({PROVIDER_ENV: "nvidia"}) == "nvidia"


def test_falls_back_to_the_document_platform_switch():
    """Backward compatibility for sites configured before vision had its own."""
    assert resolve_provider_name({LEGACY_PROVIDER_ENV: "ollama"}) == "ollama"


def test_primary_wins_over_legacy():
    """Otherwise a site could not move vision off the document provider."""
    resolved = resolve_provider_name(
        {PROVIDER_ENV: "ollama", LEGACY_PROVIDER_ENV: "nvidia"}
    )
    assert resolved == "ollama"


def test_defaults_to_static_when_nothing_is_configured():
    """The only default that cannot surprise anyone: no key, no network, and it
    announces itself as a constant rather than pretending to be a model."""
    assert resolve_provider_name({}) == DEFAULT_PROVIDER == "static"


def test_resolution_is_case_and_whitespace_insensitive():
    assert resolve_provider_name({PROVIDER_ENV: "  NVIDIA \n"}) == "nvidia"


# --- building -------------------------------------------------------------------- #


def test_nvidia_resolves_to_the_nvidia_adapter():
    adapter, note = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia", **KEYED}
    )
    assert isinstance(adapter, NvidiaVisionUnderstander)
    assert adapter.adapter_id == "understander.nvidia_vl"
    assert "nvidia" in note


def test_ollama_resolves_to_the_ollama_adapter():
    adapter, note = build_understander(producible=PRODUCIBLE, env={PROVIDER_ENV: "ollama"})
    assert isinstance(adapter, OllamaVisionUnderstander)
    assert adapter.adapter_id == "understander.ollama_vl"
    assert "ollama" in note


def test_static_resolves_to_the_static_head():
    adapter, _ = build_understander(producible=PRODUCIBLE, env={PROVIDER_ENV: "static"})
    assert isinstance(adapter, StaticAttributeHead)


def test_legacy_switch_builds_the_same_adapter():
    adapter, _ = build_understander(
        producible=PRODUCIBLE, env={LEGACY_PROVIDER_ENV: "nvidia", **KEYED}
    )
    assert isinstance(adapter, NvidiaVisionUnderstander)


def test_explicit_provider_argument_overrides_the_environment():
    """Composition roots that already know what they want should not have to
    mutate the process environment to say so."""
    adapter, _ = build_understander(
        producible=PRODUCIBLE, provider="static", env={PROVIDER_ENV: "nvidia", **KEYED}
    )
    assert isinstance(adapter, StaticAttributeHead)


# --- configuration failures ------------------------------------------------------ #


def test_unknown_provider_is_a_clear_composition_time_error():
    with pytest.raises(ProviderConfigurationError) as exc:
        build_understander(producible=PRODUCIBLE, env={PROVIDER_ENV: "gpt5-vision"})

    message = str(exc.value)
    assert "gpt5-vision" in message
    # The message must name what *is* available, or an operator is left guessing.
    for known in UNDERSTANDER_FACTORIES:
        assert known in message


def test_nvidia_without_a_key_fails_at_binding_not_at_first_crop():
    with pytest.raises(ProviderConfigurationError) as exc:
        build_understander(producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia"})
    assert "NVIDIA_API_KEY" in str(exc.value)


def test_no_producible_attributes_is_refused():
    with pytest.raises(ProviderConfigurationError):
        build_understander(producible=(), env={PROVIDER_ENV: "static"})


def test_no_api_key_is_ever_defaulted_in_source():
    """A key that appeared from a default would be a key nobody could rotate."""
    import inspect

    from vision_os.adapters.configuration import understander_providers as providers

    source = inspect.getsource(providers)
    assert "nvapi-" not in source


# --- credentials ----------------------------------------------------------------- #


def test_credentials_can_come_from_an_env_file(tmp_path):
    """One copy of a shared credential, rather than one per component."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\nNVIDIA_API_KEY=nvapi-from-file  # trailing\nUNRELATED=x\n",
        encoding="utf-8",
    )
    assert read_env_file(env_file)["NVIDIA_API_KEY"] == "nvapi-from-file"


def test_missing_env_file_is_not_an_error(tmp_path):
    assert read_env_file(tmp_path / "absent.env") == {}


def test_a_credentials_file_cannot_choose_the_provider(tmp_path):
    """The file supplies credentials; the environment supplies selection.

    Regression: `env_file` is typically a neighbouring service's `.env`, and it
    carries *that* service's provider setting. Merging it before resolving the
    name let `DOCUMENT_VLM_PROVIDER=nvidia` in the Atlas backend's file silently
    select the CCTV understander — a deployment inheriting a decision from a file
    it does not own.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DOCUMENT_VLM_PROVIDER=nvidia\nNVIDIA_API_KEY=nvapi-from-file\n", encoding="utf-8"
    )

    adapter, _ = build_understander(producible=PRODUCIBLE, env={}, env_file=env_file)
    assert isinstance(adapter, StaticAttributeHead), (
        "a provider named only in a credentials file must not be selected"
    )


def test_credentials_still_come_from_the_file_once_selected(tmp_path):
    """The other half: having refused the file's *selection*, still use its key."""
    env_file = tmp_path / ".env"
    env_file.write_text("NVIDIA_API_KEY=nvapi-from-file\n", encoding="utf-8")

    adapter, _ = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia"}, env_file=env_file
    )
    assert isinstance(adapter, NvidiaVisionUnderstander)


# --- the port is unchanged by how it was selected -------------------------------- #


@pytest.mark.parametrize(
    ("provider", "env"),
    [("nvidia", KEYED), ("ollama", {}), ("static", {})],
)
def test_every_provider_satisfies_the_port(provider, env):
    adapter, _ = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: provider, **env}
    )
    assert isinstance(adapter, UnderstanderPort)
    capabilities = adapter.capabilities()
    assert capabilities.producible_attributes
    assert capabilities.cost_class >= 0.0


def test_residency_is_declared_honestly_per_provider():
    """The field ``_bind`` gates on. A site forbidding remote understanders must
    be able to refuse the hosted one and keep the local one."""
    hosted, _ = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia", **KEYED}
    )
    local, _ = build_understander(producible=PRODUCIBLE, env={PROVIDER_ENV: "ollama"})

    assert hosted.capabilities().is_remote is True
    assert local.capabilities().is_remote is False


def test_selection_does_not_alter_port_behaviour():
    """Two adapters built through different routes must behave identically."""
    from_env, _ = build_understander(
        producible=PRODUCIBLE, env={PROVIDER_ENV: "nvidia", **KEYED}
    )
    from_arg, _ = build_understander(
        producible=PRODUCIBLE, provider="nvidia", env=KEYED
    )

    assert from_env.adapter_id == from_arg.adapter_id
    assert from_env.capabilities() == from_arg.capabilities()


# --- conformance still holds after selection ------------------------------------- #


class _StubbedNvidia(NvidiaVisionUnderstander):
    __slots__ = ("_reply",)

    def __init__(self, reply, **kw):
        super().__init__(**kw)
        self._reply = reply

    def _chat(self, prompt, image_b64, *, max_tokens, temperature, timeout):
        if isinstance(self._reply, Exception):
            raise self._reply
        return {
            "choices": [{"message": {"content": self._reply}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _StubbedOllama(OllamaVisionUnderstander):
    __slots__ = ("_reply",)

    def __init__(self, reply, **kw):
        super().__init__(**kw)
        self._reply = reply

    def _post(self, path, payload, *, timeout):
        if isinstance(self._reply, Exception):
            raise self._reply
        return {"response": self._reply, "prompt_eval_count": 10, "eval_count": 5}


ANSWER = '{"posture": "standing", "colour": "red"}'


def test_nvidia_adapter_passes_the_p15_kit():
    report = UNDERSTANDER_KIT.run(
        _StubbedNvidia(ANSWER, producible=PRODUCIBLE, api_key="nvapi-test")
    )
    assert report.passed, report.failures
    assert len(report.executed) == 9


def test_ollama_adapter_passes_the_p15_kit():
    report = UNDERSTANDER_KIT.run(_StubbedOllama(ANSWER, producible=PRODUCIBLE))
    assert report.passed, report.failures
    assert len(report.executed) == 9
