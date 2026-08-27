"""Replacing a retired hosted VLM, and never being blind to that again.

On **2026-08-26T09:00:00Z** NVIDIA retired
``nvidia/llama-3.1-nemotron-nano-vl-8b-v1``. Every call returned ``410 Gone``,
every crop became a refusal, no attribute was produced, and the Alerts page
stayed empty for eighteen hours. Cameras, detection, tracking, cropping,
compliance and incident creation were all healthy the entire time.

Two things made that outage possible, and both are covered here:

1. **The model could only be named in code.** `nvidia_vl.DEFAULT_MODEL` was the
   sole source, and `.env` could not override it because the provider factory
   resolves against `os.environ`, which pydantic-settings never writes to. There
   was no configuration change that could fix a retired model.
2. **Nothing told anyone.** A dead analysis and a quiet kitchen render the same
   empty page, and the product could not distinguish them.

No test here needs the network.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from app.configuration.settings import Settings
from vision_os.adapters.configuration.understander_providers import (
    PROVIDER_ENV,
    build_understander,
)
from vision_os.adapters.understanding.nvidia_vl import (
    DEFAULT_MODEL,
    MODEL_RETIRED_STATUSES,
    ModelRetiredError,
    NvidiaVisionUnderstander,
)
from vision_os.adapters.understanding.payload import extract_json
from vision_os.core.model.ids import AttributeKey

PRODUCIBLE = (AttributeKey("head_covering"), AttributeKey("hand_covering"))
KEYED = {"NVIDIA_API_KEY": "nvapi-test"}
RETIRED = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


# --- 1. configuration ------------------------------------------------------------ #


def test_the_deployment_can_name_the_model():
    """The fix for the outage, stated as a test.

    Without this the only way to replace a retired model is to edit Python and
    redeploy — during an incident, on a safety product.
    """
    adapter, _ = build_understander(
        producible=PRODUCIBLE,
        env={PROVIDER_ENV: "nvidia", "VISION_NVIDIA_MODEL": "meta/llama-3.2-11b-vision-instruct",
             **KEYED},
    )
    assert adapter._model == "meta/llama-3.2-11b-vision-instruct"


def test_settings_carry_the_model_from_file_to_the_factory():
    """`.env` reaches the settings object, never `os.environ`. This is the bridge."""
    options = Settings(
        VISION_NVIDIA_MODEL="meta/llama-3.2-11b-vision-instruct",
        VISION_NVIDIA_BASE_URL="https://nim.internal/v1",
    ).understander_options()

    adapter, _ = build_understander(
        producible=PRODUCIBLE, provider="nvidia", env=KEYED, defaults=options
    )
    assert adapter._model == "meta/llama-3.2-11b-vision-instruct"
    assert adapter._base == "https://nim.internal/v1"


def test_a_real_environment_variable_outranks_the_file():
    """`defaults -> file -> environment`. An operator naming a replacement for one
    run must not be overruled by a stale file."""
    options = Settings(VISION_NVIDIA_MODEL="from-file").understander_options()
    adapter, _ = build_understander(
        producible=PRODUCIBLE, provider="nvidia",
        env={"VISION_NVIDIA_MODEL": "from-environment", **KEYED}, defaults=options,
    )
    assert adapter._model == "from-environment"


def test_an_unset_setting_does_not_blank_a_working_default():
    """Empty must mean "the adapter default applies", not "the empty string" —
    otherwise an unset base_url points the adapter at nowhere."""
    assert "VISION_NVIDIA_BASE_URL" not in Settings(
        VISION_NVIDIA_BASE_URL=""
    ).understander_options()


def test_no_credential_travels_in_the_options():
    """The key is a SecretStr on its own path. Nothing printable carries it."""
    options = Settings(
        VISION_NVIDIA_API_KEY="nvapi-secret", VISION_NVIDIA_MODEL="m"
    ).understander_options()
    assert not any("nvapi" in value for value in options.values())


# --- 2. model selection ---------------------------------------------------------- #


def test_the_default_model_is_not_the_retired_one():
    """The constant is the last-resort default layer. Pointing it at a model that
    no longer exists means a deployment naming nothing gets a dead adapter."""
    assert DEFAULT_MODEL != RETIRED


def test_the_committed_template_documents_the_model_variable():
    """A hosted identifier is deployment configuration, so the template a new
    deployment copies has to say so. Asserted against `.env.example` rather than
    `.env`, which is git-ignored and absent in CI."""
    import pathlib

    template = (pathlib.Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )
    assert "VISION_NVIDIA_MODEL=" in template
    assert RETIRED in template, "the retired model should be named as a warning"


def test_the_committed_template_carries_no_credential():
    """`.env.example` is committed. A real key in it is a key in git history."""
    import pathlib
    import re

    template = (pathlib.Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )
    assert not re.search(r"nvapi-[A-Za-z0-9_\-]{10,}", template)


# --- 3. availability and the retirement itself ----------------------------------- #


def _adapter(**kw) -> NvidiaVisionUnderstander:
    return NvidiaVisionUnderstander(producible=PRODUCIBLE, api_key="nvapi-test", **kw)


@pytest.mark.parametrize("status", sorted(MODEL_RETIRED_STATUSES))
def test_a_retirement_status_is_raised_under_its_own_name(monkeypatch, status):
    """410 is not "one more failed call". No retry fixes it, and treating it as
    transient is what turned a retirement into eighteen silent hours."""
    adapter = _adapter()

    def boom(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://x/v1/chat/completions", status, "Gone", {},
            _Body(json.dumps({"detail": "end of life"})),
        )

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ModelRetiredError) as caught:
        adapter._chat("prompt", "aGk=", max_tokens=8, temperature=0.0, timeout=5)
    assert caught.value.status == status


def test_a_retirement_becomes_a_refusal_and_never_an_answer(monkeypatch):
    """**The safety line.** A dead model must not produce a value that any rule
    could compare. It refuses, and the refusal says why."""
    adapter = _adapter()
    _retire(monkeypatch)
    response = adapter.understand(_request())

    assert response.refused is True
    assert dict(response.structured) == {}
    assert "model retired" in (response.refusal_reason or "")


def test_health_reports_a_retirement_so_it_is_not_read_as_no_violations(monkeypatch):
    """The field that distinguishes "nothing to report" from "nothing is working"."""
    adapter = _adapter()
    assert adapter.health()["available"] is True

    _retire(monkeypatch)
    adapter.understand(_request())

    health = adapter.health()
    assert health["available"] is False
    assert health["state"] == "model_retired"
    assert "no longer available" in health["reason"]


def test_probe_latches_a_model_the_endpoint_does_not_list(monkeypatch):
    """Caught at binding, where someone is watching, rather than on the first
    crop of a live shift."""
    adapter = _adapter(model="gone/model")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Ctx(json.dumps({"data": [{"id": "other/model"}]})),
    )
    result = adapter.probe()

    assert result["model_listed"] is False
    assert adapter.health()["available"] is False


def test_a_transient_failure_is_not_reported_as_a_retirement(monkeypatch):
    """A 503 passes; a retirement does not. Conflating them would either spam
    operators or hide the one that matters."""
    adapter = _adapter()

    def boom(*_a, **_k):
        raise urllib.error.HTTPError("https://x", 503, "Busy", {}, _Body("overloaded"))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError) as caught:
        adapter._chat("p", "aGk=", max_tokens=8, temperature=0.0, timeout=5)
    assert not isinstance(caught.value, ModelRetiredError)
    assert adapter.health()["available"] is True


# --- 4. the replacement model's answer shape ------------------------------------- #

#: Verbatim from `meta/llama-3.2-11b-vision-instruct` against the shipped kitchen
#: prompt. It answers the object **body** with no braces and no commas — every
#: value correct and in-domain, and the whole answer was being discarded.
BRACELESS = (
    '"head_covering": "cap"\n'
    '"face_covering": "none"\n'
    '"hand_covering": "not_visible"'
)


def test_the_replacement_models_answer_is_recovered():
    assert extract_json(BRACELESS) == {
        "head_covering": "cap",
        "face_covering": "none",
        "hand_covering": "not_visible",
    }


def test_well_formed_json_still_parses():
    """The retired model wrapped its object. Recovery must not cost that."""
    assert extract_json('{"head_covering": "none"}') == {"head_covering": "none"}
    assert extract_json('```json\n{"head_covering": "hood"}\n```') == {"head_covering": "hood"}


@pytest.mark.parametrize(
    "text",
    [
        "I cannot determine the head covering from this image.",
        "Sorry, I am unable to help with that.",
        '"head_covering": ',
        "",
        "none",
    ],
)
def test_a_refusal_or_a_fragment_never_becomes_an_answer(text):
    """**U2.** Recovery is legitimate; invention is not. Anything that is not
    plainly an object body stays unparseable with the original preserved — and
    critically, never resolves to a value a rule could read as compliant."""
    assert extract_json(text) is None


def _retire(monkeypatch) -> None:
    """Make the transport answer 410, as NVIDIA did on 2026-08-26.

    Driven through `urllib` rather than by replacing `_chat`: the adapter uses
    `__slots__`, so an instance attribute cannot be patched — and going through
    the real transport exercises the status-to-error mapping too.
    """

    def gone(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://x/v1/chat/completions", 410, "Gone", {},
            _Body(json.dumps({"detail": "end of life on 2026-08-26T09:00:00Z"})),
        )

    monkeypatch.setattr("urllib.request.urlopen", gone)


def _request():
    from vision_os.core.model.ids import CropId, PromptId, RequestId
    from vision_os.core.ports.understanding import (
        CropView,
        OutputSchema,
        RenderedPrompt,
        UnderstandingPortRequest,
    )

    schema = OutputSchema(fields=PRODUCIBLE)
    return UnderstandingPortRequest(
        request_id=RequestId("r"),
        crops=(CropView(crop_id=CropId("c"), pixels=memoryview(bytes(8 * 8 * 3)),
                        width=8, height=8),),
        prompt=RenderedPrompt(prompt_id=PromptId("p"), version="1.0.0", text="?",
                              output_schema=schema),
        output_schema=schema,
    )


class _Body:
    """Minimal file-like for HTTPError, which reads its body once."""

    def __init__(self, text: str) -> None:
        self._text = text.encode()

    def read(self) -> bytes:
        return self._text


class _Ctx:
    """Minimal context manager standing in for urlopen."""

    def __init__(self, text: str) -> None:
        self._text = text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._text
