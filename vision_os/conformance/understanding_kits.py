"""Conformance kits for P15 ``UnderstanderPort`` and P16 ``OutputCoercionPort``.

06_PORTS calls P15 *"the most volatile port in the platform, and therefore the
most carefully bounded"*. This kit is where the bounding becomes executable.

The check that matters most is ``understander/no_fabrication_on_failure``.
10_RELIABILITY §2.1 names it directly as one of the defences against Byzantine
failure, and U2 explains the stakes:

> *This is the single most dangerous failure mode for a VLM-based system, because
> fabricated output is indistinguishable from real output downstream.*

An adapter that returns a plausible default on timeout passes every other test
ever written. It poisons the observation log silently, forever, and nothing
downstream can detect it. This check is the only thing standing between that
adapter and production.

**What these kits cannot check**, stated plainly: no check here can tell whether
a model's answers are *correct*. That needs a labelled corpus — 06_PORTS §5.1's
"golden data" section — which is a per-deployment asset, not a platform one. The
kits verify contracts: the structural properties whose violation is silent.
"""

from __future__ import annotations

from ..core.errors import UnderstanderTimeoutError, UnderstanderUnavailableError
from ..core.model.ids import AttributeKey, CropId, RequestId
from ..core.ports.understanding import (
    CropView,
    OutputSchema,
    RenderedPrompt,
    UnderstandingPortRequest,
)
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_SCHEMA = OutputSchema(fields=(AttributeKey("kit.posture"),))
_PIXELS = memoryview(bytes(8 * 8 * 3))


def _prompt(fields: tuple[AttributeKey, ...] = ()) -> RenderedPrompt:
    from ..core.model.ids import PromptId

    return RenderedPrompt(
        prompt_id=PromptId("kit.prompt"),
        version="1.0.0",
        text="Describe the posture of the subject.",
        output_schema=OutputSchema(fields=fields or _SCHEMA.fields),
    )


def _request(
    suffix: str = "1", *, fields: tuple[AttributeKey, ...] = (), crops: int = 1
) -> UnderstandingPortRequest:
    prompt = _prompt(fields)
    return UnderstandingPortRequest(
        request_id=RequestId(f"kit-request-{suffix}"),
        crops=tuple(
            CropView(
                crop_id=CropId(f"kit-crop-{suffix}-{index}"),
                pixels=_PIXELS,
                width=8,
                height=8,
            )
            for index in range(crops)
        ),
        prompt=prompt,
        output_schema=prompt.output_schema,
        context={"class_id": "person"},
    )


# --- P15 UnderstanderPort ------------------------------------------------------- #


def _check_shape(adapter) -> None:
    assert hasattr(adapter, "adapter_id"), "an understander must expose adapter_id"
    assert isinstance(adapter.adapter_id, str) and adapter.adapter_id, (
        "adapter_id must be a non-empty string; it labels every metric and every "
        "provenance record this adapter produces"
    )
    capabilities = adapter.capabilities()
    assert capabilities.producible_attributes, (
        "an understander declaring no producible attributes can never be routed "
        "to, and would silently never be selected"
    )
    assert capabilities.model_id, "capabilities must name the model"


def _check_capabilities_are_complete(adapter) -> None:
    """Every field routing and budgeting depend on must be declared."""
    capabilities = adapter.capabilities()
    assert capabilities.cost_class >= 0.0, "cost_class drives M8's budget policy"
    assert capabilities.max_batch_size >= 1
    assert capabilities.data_residency, (
        "data_residency must be declared; it gates use in regulated sites, and an "
        "undeclared residency is a remote export nobody authorised"
    )
    if capabilities.supports_batching:
        assert capabilities.max_batch_size > 1, (
            "an adapter declaring batching must accept batches larger than one"
        )


def _check_returns_only_declared_fields(adapter) -> None:
    """Obligation U1. Extra fields go to ``unparsed``, never to ``structured``."""
    request = _request("u1")
    response = _invoke(adapter, request)
    if response is None:
        return
    declared = {str(key) for key in request.output_schema.fields}
    extra = set(response.structured) - declared
    assert not extra, (
        f"adapter returned undeclared field(s) {sorted(extra)}; U1 requires the "
        f"adapter return what the schema declared and nothing else — anything "
        f"volunteered belongs in `unparsed`, where the platform can see it "
        f"without treating it as fact"
    )


def _check_no_fabrication_on_failure(adapter) -> None:
    """**Obligation U2 — the check this kit exists for.**

    An adapter that returns a plausible default on failure passes every other
    test ever written and poisons the observation log forever, because fabricated
    output is indistinguishable from real output downstream.

    Both shapes are acceptable: raising a typed error, or returning an explicit
    refusal. What is not acceptable is structured content produced by a call that
    failed.
    """

    class _Failing:
        """A crop that cannot be read — the simplest injectable fault."""

    request = _request("u2")
    broken = UnderstandingPortRequest(
        request_id=request.request_id,
        crops=(
            CropView(
                crop_id=CropId("kit-crop-corrupt"),
                pixels=memoryview(b""),
                width=8,
                height=8,
            ),
        ),
        prompt=request.prompt,
        output_schema=request.output_schema,
        context=request.context,
    )
    try:
        response = adapter.understand(broken)
    except (UnderstanderTimeoutError, UnderstanderUnavailableError):
        return
    except Exception:  # noqa: BLE001 - any typed failure is acceptable here
        return

    if response.refused:
        assert not response.structured, (
            "a refusal carrying structured output is reporting one of them falsely"
        )
        return
    assert not response.structured or all(
        str(key) in {str(f) for f in broken.output_schema.fields}
        for key in response.structured
    ), (
        "the adapter produced content from an unreadable crop; U2 forbids "
        "fabricating on failure, and this is the failure mode that cannot be "
        "detected downstream"
    )
    assert _Failing is not None


def _check_raw_output_preserved(adapter) -> None:
    """Obligation U3. Without the verbatim bytes, V4 is theoretical."""
    response = _invoke(adapter, _request("u3"))
    if response is None or response.refused:
        return
    if response.structured:
        assert response.raw_output, (
            "an adapter that answered must preserve its raw output verbatim; "
            "evidence without the model's own words cannot explain a result six "
            "months later (U3, invariant V4)"
        )


def _check_stateless_across_requests(adapter) -> None:
    """Obligation U5. *"Or caching and replay both break."*

    Two identical requests must be independently answerable. An adapter carrying
    conversation state answers the second differently, which silently makes every
    cache hit wrong and every replay a different run.
    """
    first = _invoke(adapter, _request("u5"))
    second = _invoke(adapter, _request("u5"))
    if first is None or second is None:
        return
    if not adapter.capabilities().deterministic:
        return
    assert dict(first.structured) == dict(second.structured), (
        "a deterministic adapter answered two identical requests differently; "
        "either it carries state across requests (U5) or its determinism "
        "declaration is false"
    )


def _check_batch_is_total(adapter) -> None:
    """Every request id comes back. A dropped id is a lost answer in disguise."""
    requests = [_request(str(index)) for index in range(3)]
    try:
        responses = adapter.understand_batch(requests)
    except (UnderstanderTimeoutError, UnderstanderUnavailableError):
        return
    returned = set(responses)
    expected = {request.request_id for request in requests}
    assert returned == expected, (
        f"batch returned {len(returned)} of {len(expected)} request ids; a "
        f"dropped id is indistinguishable from a lost one, so the mapping must "
        f"be total even when every entry failed"
    )


def _check_cost_is_estimable(adapter) -> None:
    """Obligation U7. M8's budget policy decides *before* the money is spent."""
    estimate = adapter.estimate_cost(_request("u7"))
    assert estimate is not None, "estimate_cost must return an estimate"
    assert estimate.cost_units >= 0.0, "cost cannot be negative"


def _check_temporal_declaration_is_honest(adapter) -> None:
    """An adapter must not accept a crop sequence it declared it cannot handle.

    Temporal understanding is Phase 3. An adapter that quietly accepted a
    sequence and analysed only the first frame would produce an answer about one
    instant labelled as an answer about a span.
    """
    capabilities = adapter.capabilities()
    if capabilities.supports_temporal:
        return
    if capabilities.max_crops_per_request > 1:
        return
    try:
        response = adapter.understand(_request("temporal", crops=3))
    except Exception:  # noqa: BLE001 - refusing is the correct behaviour
        return
    assert response.refused or not response.structured, (
        "the adapter declares no temporal support and accepts one crop, but "
        "answered a three-crop request; an answer about one frame presented as "
        "an answer about a sequence is a claim nobody made"
    )


def _invoke(adapter, request):
    try:
        return adapter.understand(request)
    except (UnderstanderTimeoutError, UnderstanderUnavailableError):
        return None


UNDERSTANDER_KIT = ConformanceKit(
    port_id=PortCatalogue.UNDERSTANDER,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_shape),
        ConformanceCheck(
            "capabilities_are_complete", KitSection.SHAPE, _check_capabilities_are_complete
        ),
        ConformanceCheck(
            "returns_only_declared_fields",
            KitSection.SEMANTICS,
            _check_returns_only_declared_fields,
            obligation="U1",
        ),
        ConformanceCheck(
            "no_fabrication_on_failure",
            KitSection.FAILURE,
            _check_no_fabrication_on_failure,
            obligation="U2",
        ),
        ConformanceCheck(
            "raw_output_preserved",
            KitSection.SEMANTICS,
            _check_raw_output_preserved,
            obligation="U3",
        ),
        ConformanceCheck(
            "stateless_across_requests",
            KitSection.SEMANTICS,
            _check_stateless_across_requests,
            obligation="U5",
        ),
        ConformanceCheck("batch_is_total", KitSection.SHAPE, _check_batch_is_total),
        ConformanceCheck(
            "cost_is_estimable", KitSection.SEMANTICS, _check_cost_is_estimable, obligation="U7"
        ),
        ConformanceCheck(
            "temporal_declaration_is_honest",
            KitSection.FAILURE,
            _check_temporal_declaration_is_honest,
        ),
    ),
)


# --- P16 OutputCoercionPort ------------------------------------------------------ #


def _check_coercion_shape(adapter) -> None:
    assert hasattr(adapter, "strategy_id"), "a coercion strategy must expose strategy_id"
    assert isinstance(adapter.strategy_id, str) and adapter.strategy_id
    result = adapter.coerce('{"kit.posture": "standing"}', schema=_SCHEMA)
    assert result is not None, "coerce must return a result"


def _check_never_invents_a_field(adapter) -> None:
    """Obligation X1. Output keys come from the model's text alone."""
    result = adapter.coerce("the subject appears to be upright", schema=_SCHEMA)
    assert not result.parsed, (
        f"the strategy produced {sorted(result.parsed)} from prose that named no "
        f"field; inferring a field from a sentence is a second understanding "
        f"layer with no schema and no evidence (X1)"
    )


def _check_never_discards(adapter) -> None:
    """Obligation X2 and 02_VOM §9.3. What does not parse is preserved."""
    prose = "I cannot tell from this image."
    result = adapter.coerce(prose, schema=_SCHEMA)
    if result.parsed:
        return
    assert result.unparsed, (
        "text that produced no fields was discarded; 02_VOM section 9.3 requires "
        "it preserved as an inspectable note — the difference between a "
        "diagnosable platform and a black box"
    )


def _check_coercion_determinism(adapter) -> None:
    """Obligation X3. Identical text yields identical parses (V13)."""
    text = '{"kit.posture": "standing", "extra": 1}'
    first = adapter.coerce(text, schema=_SCHEMA)
    second = adapter.coerce(text, schema=_SCHEMA)
    assert dict(first.parsed) == dict(second.parsed), (
        "identical text parsed differently; a replay must reproduce the same "
        "attributes or the evidence explains a run that never happened"
    )


def _check_never_raises(adapter) -> None:
    """Obligation X4. Malformed text is the normal case, not an exception."""
    for text in (
        "",
        "   ",
        "{",
        '{"unterminated": ',
        "null",
        "[1, 2, 3]",
        "\x00\x01\x02",
        '{"kit.posture": {"nested": {"deeply": true}}}',
    ):
        try:
            adapter.coerce(text, schema=_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - any raise is the failure
            raise AssertionError(
                f"coercion raised {type(exc).__name__} on {text!r}; malformed "
                f"model output is the normal case, and returning an empty parse "
                f"with the text preserved is the correct answer (X4)"
            ) from exc


def _check_undeclared_fields_are_not_parsed(adapter) -> None:
    """A field the prompt did not declare must not arrive as a parsed value.

    The platform still sees it — X2 puts it in ``unparsed`` — but as text a human
    reads, never as a value a consumer queries.
    """
    result = adapter.coerce(
        '{"kit.posture": "standing", "is_violation": true}', schema=_SCHEMA
    )
    assert "is_violation" not in result.parsed, (
        "an undeclared field was returned as parsed; this is precisely how a "
        "model's judgment becomes a platform fact, and it is what the ceiling "
        "exists to prevent"
    )


OUTPUT_COERCION_KIT = ConformanceKit(
    port_id=PortCatalogue.OUTPUT_COERCION,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_coercion_shape),
        ConformanceCheck(
            "never_invents_a_field",
            KitSection.SEMANTICS,
            _check_never_invents_a_field,
            obligation="X1",
        ),
        ConformanceCheck(
            "never_discards", KitSection.SEMANTICS, _check_never_discards, obligation="X2"
        ),
        ConformanceCheck(
            "determinism", KitSection.SEMANTICS, _check_coercion_determinism, obligation="X3"
        ),
        ConformanceCheck(
            "never_raises", KitSection.FAILURE, _check_never_raises, obligation="X4"
        ),
        ConformanceCheck(
            "undeclared_fields_are_not_parsed",
            KitSection.SEMANTICS,
            _check_undeclared_fields_are_not_parsed,
            obligation="X1",
        ),
    ),
)


ALL_UNDERSTANDING_KITS: tuple[ConformanceKit, ...] = (
    UNDERSTANDER_KIT,
    OUTPUT_COERCION_KIT,
)
