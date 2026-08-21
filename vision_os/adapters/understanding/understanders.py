"""Reference understanders — P15.

**No VLM ships here.** 06_PORTS lists `vlm.qwen2_5vl`, `vlm.gpt41_vision` and the
rest as *adapter examples*; binding one requires weights, a runtime and a device,
which are M18's concern and a deployment's choice. What ships is the machinery
that proves the port is at the right altitude:

``understander.scripted``
    Answers from a fixed script. The reference implementation, and the one every
    conformance kit and integration test runs against — deterministic, free, and
    incapable of hallucinating, which makes it the only honest way to test the
    *platform's* behaviour rather than a model's.

``attr.static_classifier``
    A **specialized head**, not a VLM: it produces exactly one attribute at a
    cost class two orders of magnitude below a generalist. 06_PORTS calls the
    `attr.*` adapters *"the point of this port's design"* — they prove the
    platform *"is genuinely indifferent to whether a 7-billion-parameter
    generalist or a 2-megabyte specialist answered."*

``understander.unavailable``
    Always fails. Not a test double: 10_RELIABILITY §7.2 requires a fallback chain
    to terminate in **explicit unavailability, never a guess**, and a deployment
    that has lost its model needs something that says so rather than something
    that silently produces nothing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ...core.errors import UnderstanderTimeoutError, UnderstanderUnavailableError
from ...core.model.ids import AttributeKey, ModelId, RequestId
from ...core.model.understanding import CostEstimate, ModelMeta, Timing
from ...core.ports.understanding import (
    UnderstanderCapabilities,
    UnderstandingPortRequest,
    UnderstandingPortResponse,
)

#: Model ids for the reference adapters.
#:
#: Module-level constants rather than calls in a default argument: a call there
#: is evaluated once at import anyway, and naming it makes the shared identity
#: obvious to anyone binding two of these in one deployment.
SCRIPTED_MODEL = ModelId("scripted-understander")
STATIC_HEAD_MODEL = ModelId("static-attribute-head")
UNAVAILABLE_MODEL = ModelId("unavailable")


@dataclass(frozen=True, slots=True)
class ScriptedAnswer:
    """One canned response, keyed by nothing — returned in order.

    Order rather than content matching, deliberately: a script that matched on
    the crop would be a tiny model, and a test that passes because the double got
    the answer right tests the double.
    """

    fields: Mapping[str, object] = field(default_factory=dict)
    unparsed: str | None = None
    confidence: Mapping[str, float] | None = None
    refused: bool = False
    refusal_reason: str = ""
    raise_timeout: bool = False
    raise_unavailable: bool = False


class ScriptedUnderstander:
    """Answers from a script. Deterministic, free, and never hallucinating.

    Every conformance check and every integration test runs against this, because
    the property under test is *"does the platform handle what a model said"* and
    a real model would make that non-deterministic for no benefit.
    """

    __slots__ = ("_answers", "_calls", "_capabilities", "_cursor", "_id")

    def __init__(
        self,
        *,
        adapter_id: str = "understander.scripted",
        model_id: ModelId = SCRIPTED_MODEL,
        producible: Sequence[AttributeKey] = (),
        answers: Sequence[ScriptedAnswer] = (),
        supports_batching: bool = True,
        max_batch_size: int = 8,
        cost_class: float = 1.0,
        deterministic: bool = True,
        data_residency: str = "local",
        supports_structured_output: bool = True,
    ) -> None:
        if not producible:
            raise ValueError(
                "a scripted understander must declare what it can produce; an "
                "understander producing nothing can never be routed to"
            )
        self._id = adapter_id
        self._answers = list(answers)
        self._cursor = 0
        self._calls = 0
        self._capabilities = UnderstanderCapabilities(
            producible_attributes=tuple(producible),
            model_id=model_id,
            max_crops_per_request=1,
            supports_structured_output=supports_structured_output,
            supports_temporal=False,
            supports_batching=supports_batching,
            max_batch_size=max_batch_size if supports_batching else 1,
            cost_class=cost_class,
            deterministic=deterministic,
            data_residency=data_residency,
        )

    @property
    def adapter_id(self) -> str:
        return self._id

    @property
    def calls(self) -> int:
        return self._calls

    def capabilities(self) -> UnderstanderCapabilities:
        return self._capabilities

    def understand(
        self, request: UnderstandingPortRequest
    ) -> UnderstandingPortResponse:
        self._calls += 1
        answer = self._next()

        if answer.raise_timeout:
            raise UnderstanderTimeoutError(
                f"scripted timeout for request '{request.request_id}'",
                adapter_id=self._id,
            )
        if answer.raise_unavailable:
            raise UnderstanderUnavailableError(
                f"scripted unavailability for request '{request.request_id}'",
                adapter_id=self._id,
            )

        if answer.refused:
            return UnderstandingPortResponse(
                refused=True,
                refusal_reason=answer.refusal_reason or "scripted refusal",
                raw_output=b"",
                model_meta=self._meta(),
                timing=Timing(inference_ms=1.0, total_ms=1.0),
            )

        # U1: return only what the schema declared. Anything else is the model
        # volunteering, and it goes to ``unparsed`` where a human can see it.
        declared = {str(key) for key in request.output_schema.fields}
        kept = {k: v for k, v in answer.fields.items() if k in declared}
        extra = {k: v for k, v in answer.fields.items() if k not in declared}

        unparsed = answer.unparsed
        if extra:
            volunteered = json.dumps(extra, sort_keys=True, default=str)
            unparsed = f"{unparsed}\n{volunteered}" if unparsed else volunteered

        return UnderstandingPortResponse(
            structured=kept,
            unparsed=unparsed,
            field_confidence=answer.confidence,
            raw_output=json.dumps(dict(answer.fields), sort_keys=True, default=str).encode(),
            model_meta=self._meta(),
            timing=Timing(inference_ms=1.0, total_ms=1.0, batch_size=len(request.crops)),
        )

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[RequestId, UnderstandingPortResponse]:
        """Every request id appears in the result, mapped or refused.

        A dropped id is an answer nobody can distinguish from a lost one, so a
        failure becomes an explicit refusal rather than an absence.
        """
        out: dict[RequestId, UnderstandingPortResponse] = {}
        for request in requests:
            try:
                out[request.request_id] = self.understand(request)
            except (UnderstanderTimeoutError, UnderstanderUnavailableError) as exc:
                out[request.request_id] = UnderstandingPortResponse(
                    refused=True,
                    refusal_reason=exc.message,
                    model_meta=self._meta(),
                )
        return out

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        return CostEstimate(
            cost_units=self._capabilities.cost_class * len(request.crops),
            model_id=self._capabilities.model_id,
            attributes_covered=tuple(request.output_schema.fields),
        )

    def _next(self) -> ScriptedAnswer:
        """The next scripted answer, or an empty one once the script runs out.

        Empty rather than wrapping: a test that ran past its script should see
        *no answer*, not the first answer again, which would silently make a
        multi-call test pass for the wrong reason.
        """
        if self._cursor >= len(self._answers):
            return ScriptedAnswer()
        answer = self._answers[self._cursor]
        self._cursor += 1
        return answer

    def _meta(self) -> ModelMeta:
        return ModelMeta(
            model_id=self._capabilities.model_id,
            model_version="1.0.0",
            artifact_hash="scripted:no-weights",
            adapter_id=self._id,
            deterministic=self._capabilities.deterministic,
        )


class StaticAttributeHead:
    """A **specialized head** producing exactly one attribute, very cheaply.

    06_PORTS on why this matters more than it looks:

    > *They are not VLMs at all, and they prove the abstraction is at the right
    > altitude.*

    11_PERFORMANCE §7's migration — VLM discovers an attribute, its evidence
    trains a head, the head takes over in production — is a routing change only
    because this and a VLM satisfy the same port and produce the same registered
    attribute. Nothing downstream can tell which answered, except by reading the
    provenance that says so.
    """

    __slots__ = ("_calls", "_capabilities", "_id", "_key", "_value")

    def __init__(
        self,
        *,
        attribute: AttributeKey,
        value: object,
        adapter_id: str = "attr.static_head",
        model_id: ModelId = STATIC_HEAD_MODEL,
        cost_class: float = 0.01,
    ) -> None:
        self._id = adapter_id
        self._key = attribute
        self._value = value
        self._calls = 0
        self._capabilities = UnderstanderCapabilities(
            producible_attributes=(attribute,),
            model_id=model_id,
            supports_structured_output=True,
            supports_batching=True,
            max_batch_size=32,
            cost_class=cost_class,
            deterministic=True,
            data_residency="local",
        )

    @property
    def adapter_id(self) -> str:
        return self._id

    @property
    def calls(self) -> int:
        return self._calls

    def capabilities(self) -> UnderstanderCapabilities:
        return self._capabilities

    def understand(
        self, request: UnderstandingPortRequest
    ) -> UnderstandingPortResponse:
        self._calls += 1
        if not request.output_schema.declares(self._key):
            # The head was asked something it does not answer. It says nothing
            # rather than answering the wrong question (U1).
            return UnderstandingPortResponse(
                model_meta=self._meta(),
                timing=Timing(inference_ms=0.1, total_ms=0.1),
            )
        return UnderstandingPortResponse(
            structured={str(self._key): self._value},
            field_confidence={str(self._key): 0.95},
            raw_output=json.dumps({str(self._key): self._value}, default=str).encode(),
            model_meta=self._meta(),
            timing=Timing(inference_ms=0.1, total_ms=0.1, batch_size=len(request.crops)),
        )

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[RequestId, UnderstandingPortResponse]:
        return {request.request_id: self.understand(request) for request in requests}

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        return CostEstimate(
            cost_units=self._capabilities.cost_class,
            model_id=self._capabilities.model_id,
            attributes_covered=(self._key,),
        )

    def _meta(self) -> ModelMeta:
        return ModelMeta(
            model_id=self._capabilities.model_id,
            model_version="1.0.0",
            artifact_hash="static:no-weights",
            adapter_id=self._id,
            deterministic=True,
        )


class UnavailableUnderstander:
    """Always unavailable. The honest terminal state of a fallback chain.

    10_RELIABILITY §7.2 rule 2: *"The last link is always explicit unavailability,
    never a guess. No chain terminates in a fabricated default."* A deployment
    whose model is gone binds this so the platform reports the gap rather than
    reporting nothing at all.
    """

    __slots__ = ("_capabilities", "_id", "_reason")

    def __init__(
        self,
        *,
        producible: Sequence[AttributeKey],
        adapter_id: str = "understander.unavailable",
        model_id: ModelId = UNAVAILABLE_MODEL,
        reason: str = "no understander is available at this site",
    ) -> None:
        self._id = adapter_id
        self._reason = reason
        self._capabilities = UnderstanderCapabilities(
            producible_attributes=tuple(producible),
            model_id=model_id,
            cost_class=0.0,
            deterministic=True,
        )

    @property
    def adapter_id(self) -> str:
        return self._id

    def capabilities(self) -> UnderstanderCapabilities:
        return self._capabilities

    def understand(
        self, request: UnderstandingPortRequest
    ) -> UnderstandingPortResponse:
        raise UnderstanderUnavailableError(self._reason, adapter_id=self._id)

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[RequestId, UnderstandingPortResponse]:
        raise UnderstanderUnavailableError(self._reason, adapter_id=self._id)

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        return CostEstimate(
            cost_units=0.0,
            attributes_uncovered=tuple(request.output_schema.fields),
        )
