"""P15 ``UnderstanderPort``, P16 ``OutputCoercionPort``, and the M10 seam.

> 06_PORTS §P15: *"The most volatile port in the platform, and therefore the most
> carefully bounded."*

Volatile because the model behind it changes every few months, and bounded
because everything downstream depends on it not changing. The seven obligations
U1–U7 are what make a 7-billion-parameter generalist and a 2-megabyte specialist
interchangeable — 06_PORTS calls the `attr.*` adapters *"the point of this port's
design"*:

> *They are not VLMs at all, and they prove the abstraction is at the right
> altitude: the platform asks for registered attributes with evidence, and is
> genuinely indifferent to whether a 7-billion-parameter generalist or a
> 2-megabyte specialist answered.*

**On the Prompt Manager seam.** ``PromptProvider`` below is **not a port**. The
catalogue's P17 ``PromptSourcePort`` belongs to M10 and describes how M10 loads
prompt *assets*; the relationship between M9 and M10 is a plain module dependency
(04_MODULES §M9 Dependencies). ``PromptProvider`` names exactly the three M10
calls M9 makes, so M9 can be written and tested before M10 exists and needs no
change when it does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..model.ids import (
    AttributeKey,
    ClassId,
    CropId,
    ModelId,
    PromptId,
    RequestId,
)
from ..model.timebase import Duration
from ..model.understanding import (
    CostEstimate,
    ModelMeta,
    Timing,
)

UNDERSTANDER_PORT_VERSION = "1.0.0"
OUTPUT_COERCION_PORT_VERSION = "1.0.0"


# --- the M10 seam -------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """What the platform will accept back from a model.

    Declared by the prompt (04_MODULES §M10: prompts are *"bound to a declared
    output schema referencing registered attribute keys"*), enforced by M9.

    ``fields`` is deliberately a plain key list rather than a full type
    specification: the **types live in the Attribute Schema Registry**, which is
    the ceiling's first gate. A schema that restated them would be a second
    source of truth, and the two would drift.
    """

    fields: tuple[AttributeKey, ...]
    strict: bool = True
    """When true, a field outside ``fields`` is rejected rather than ignored
    (port obligation U1). False exists only for a shadow-evaluation variant that
    wants to observe what a model volunteers."""

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError(
                "an output schema with no fields would accept nothing and cost a "
                "model call to discover it"
            )
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("an output schema must not declare a field twice")

    def declares(self, key: AttributeKey) -> bool:
        return key in self.fields


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One instruction, rendered and pinned to its version.

    Immutable, and carrying its own identity. 04_MODULES §M10: *"Provide immutable
    `prompt_id@version` for provenance on every observation."*
    """

    prompt_id: PromptId
    version: str
    text: str
    output_schema: OutputSchema
    content_hash: str = ""
    model_family: str = ""
    """Which family this rendering targets. The same logical prompt is phrased
    differently for different model families, and the *rendering* records which."""

    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.version:
            raise ValueError("a rendered prompt must carry its id and version")
        if not self.text.strip():
            raise ValueError(
                f"prompt '{self.prompt_id}@{self.version}' rendered to empty text; "
                f"a model asked nothing answers nothing"
            )

    @property
    def pinned(self) -> str:
        return f"{self.prompt_id}@{self.version}"


@runtime_checkable
class PromptProvider(Protocol):
    """The three M10 calls M9 makes. **Not a port** — a module seam.

    M9 consumes prompts and never creates them. Hard-coding a prompt string in the
    engine would produce a prompt with no version, no declared output schema and
    no load-time validation — bypassing the second of the ceiling's three
    enforcement points (00_CHARTER §4.3).
    """

    def resolve(
        self, attributes: Sequence[AttributeKey], *, class_id: ClassId, model_family: str
    ) -> tuple[PromptId, str] | None:
        """Which prompt covers this attribute set. ``None`` means none does.

        ``None`` rather than an exception because 04_MODULES §M10 makes
        ``NoSuitablePrompt`` a **normal** outcome that M8 turns into a capability
        gap — the consumer learns the demand is unsatisfiable (V8).
        """
        ...

    def render(
        self, prompt_id: PromptId, version: str, context: Mapping[str, Any]
    ) -> RenderedPrompt:
        """Render a pinned prompt with context. Raises on an unknown prompt."""
        ...

    def schema_of(self, prompt_id: PromptId, version: str) -> OutputSchema:
        """The declared output schema for a pinned prompt."""
        ...


# --- P15 UnderstanderPort ------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CropView:
    """One crop, as an adapter sees it.

    Carries pixels *and* the metadata an adapter needs to interpret them. Note
    what is absent: no object id, no track id, no tenant. An adapter is handed
    pixels and a question, never a subject — which is what keeps a model from
    being able to accumulate anything about anyone (12_SECURITY).
    """

    crop_id: CropId
    pixels: memoryview
    width: int
    height: int
    colour_space: str = "bgr24"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"degenerate crop view {self.width}x{self.height}")


@dataclass(frozen=True, slots=True)
class UnderstandingPortRequest:
    """06_PORTS §P15's ``UnderstandingRequest``, implemented exactly.

    ``crops`` is a sequence: one for single-frame, N for temporal. 15_ROADMAP §4
    confirms Phase 3 temporal understanding needs *"no contract change"* — the
    shape already admits it and single-frame is the degenerate case.
    """

    request_id: RequestId
    crops: tuple[CropView, ...]
    prompt: RenderedPrompt
    output_schema: OutputSchema
    context: Mapping[str, Any] = field(default_factory=dict)
    """``{class_id, prior_attributes?, quality}``. Interpretation context, never
    business context."""

    max_tokens: int = 512
    timeout: Duration | None = None
    temperature: float = 0.0
    """Zero by default: a deterministic answer is worth more to this platform
    than a fluent one, and V13 wants replay to reproduce."""

    def __post_init__(self) -> None:
        if not self.crops:
            raise ValueError("an understanding request requires at least one crop")

    @property
    def is_temporal(self) -> bool:
        return len(self.crops) > 1


@dataclass(frozen=True, slots=True)
class UnderstandingPortResponse:
    """06_PORTS §P15's ``UnderstandingResponse``, implemented exactly.

    Note what an adapter returns and what it does **not**: it returns parsed
    fields, whatever did not parse, optional per-field confidence, the verbatim
    bytes, its own metadata and its timing. It does not return an ``Attribute``,
    because an ``Attribute`` is a platform object with a registry-validated key
    and a confidence *semantics* — and deciding those is M9's job, not an
    adapter's.
    """

    structured: Mapping[str, Any] = field(default_factory=dict)
    unparsed: str | None = None
    """What did not fit — **preserved as evidence** (U3, 02_VOM §9.3)."""

    field_confidence: Mapping[str, float] | None = None
    """Self-reported, if the model provides it. Labelled ``SELF_REPORTED`` by M9
    and never presented as a calibrated probability (U4)."""

    raw_output: bytes = b""
    model_meta: ModelMeta | None = None
    timing: Timing = Timing()
    refused: bool = False
    """The model declined — safety filter, content policy. An **explicit** result
    recorded as evidence, never an empty success (U2)."""

    refusal_reason: str = ""

    def __post_init__(self) -> None:
        if self.refused and self.structured:
            raise ValueError(
                "a refusal cannot carry structured output; a model that both "
                "refused and answered is reporting one of them falsely"
            )


@dataclass(frozen=True, slots=True)
class UnderstanderCapabilities:
    """06_PORTS §P15's ``UnderstanderCapabilities``, implemented exactly.

    ``producible_attributes`` is the field invariant V8 rests on: it is how a
    capability gap becomes *visible* rather than being discovered by a consumer
    waiting for an attribute that will never arrive.
    """

    producible_attributes: tuple[AttributeKey, ...]
    model_id: ModelId
    max_crops_per_request: int = 1
    min_resolution: tuple[int, int] = (1, 1)
    max_resolution: tuple[int, int] = (4096, 4096)
    colour_space: str = "bgr24"
    supports_structured_output: bool = False
    """Constrained decoding available. When true, coercion is nearly free and
    schema conformance is guaranteed rather than checked."""

    supports_temporal: bool = False
    """Crop sequences. **False for every adapter shipped in Phase 1** — temporal
    understanding is Phase 3 (15_ROADMAP §4)."""

    supports_batching: bool = False
    max_batch_size: int = 1
    max_output_tokens: int = 512
    cost_class: float = 1.0
    """Relative cost unit. A specialized head is ~0.01 against a VLM's 1.0, which
    is what makes 11_PERFORMANCE §7's migration measurable."""

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    deterministic: bool = False
    data_residency: str = "local"
    """``local`` | ``remote(region)``. **Gates use in regulated sites** — a site
    with a residency policy refuses a remote adapter at binding time rather than
    discovering the export in an audit."""

    def __post_init__(self) -> None:
        if not self.producible_attributes:
            raise ValueError(
                f"understander '{self.model_id}' declares no producible "
                f"attributes; an understander that can produce nothing cannot be "
                f"routed to and would silently never be selected"
            )
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.supports_batching and self.max_batch_size < 2:
            raise ValueError(
                "an adapter declaring batching support must accept batches > 1"
            )
        if self.cost_class < 0:
            raise ValueError("cost_class must be non-negative")

    def can_produce(self, attributes: Sequence[AttributeKey]) -> bool:
        producible = set(self.producible_attributes)
        return all(key in producible for key in attributes)

    def covers(self, attributes: Sequence[AttributeKey]) -> tuple[AttributeKey, ...]:
        producible = set(self.producible_attributes)
        return tuple(key for key in attributes if key in producible)

    @property
    def is_remote(self) -> bool:
        return not self.data_residency.startswith("local")


@runtime_checkable
class UnderstanderPort(Protocol):
    """P15 — turn pixels and a question into structured fields.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **U1** | Returns **fields declared in ``output_schema`` and nothing else**. Extra fields go to ``unparsed``. The adapter never invents a schema. |
    | **U2** | **Never fabricates on failure.** A refusal, timeout or unparseable response is an explicit result, never a plausible default. *The single most dangerous failure mode for a VLM-based system, because fabricated output is indistinguishable from real output downstream.* |
    | **U3** | ``raw_output`` is preserved verbatim (V4). |
    | **U4** | Self-reported confidence is labelled ``SELF_REPORTED`` and never presented as a calibrated probability. |
    | **U5** | **Stateless across requests** — no conversation history, no accumulated context. Two identical requests must be independently answerable, *"or caching and replay both break."* |
    | **U6** | **Never applies business interpretation.** It answers the prompt it was given. |
    | **U7** | Cost is estimable **before** invocation, so M8's budget policy can decide. |
    """

    @property
    def adapter_id(self) -> str:
        ...

    def capabilities(self) -> UnderstanderCapabilities:
        """What this understander can produce. Called at binding and for routing."""
        ...

    def understand(
        self, request: UnderstandingPortRequest
    ) -> UnderstandingPortResponse:
        """Answer one request. Raises a typed error; never fabricates."""
        ...

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[RequestId, UnderstandingPortResponse]:
        """Answer many. **Every request id appears in the result**, mapped to a
        response or to an explicit refusal — a dropped id is an answer nobody can
        distinguish from a lost one."""
        ...

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        """What this would cost, before spending it (U7)."""
        ...


# --- P16 OutputCoercionPort ------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class CoercionResult:
    """What a coercion strategy made of raw model output.

    ``parsed`` and ``unparsed`` together must account for everything the model
    said. A strategy that returned neither for some fragment would be discarding
    output, which 02_VOM §9.3 forbids.
    """

    parsed: Mapping[str, Any] = field(default_factory=dict)
    unparsed: str | None = None
    strategy_used: str = ""
    reparsed: bool = False
    """Whether a second parse attempt was needed. Recorded because a model that
    routinely needs re-parsing is a prompt problem, and the statistic is how
    anyone finds out."""

    @property
    def is_empty(self) -> bool:
        return not self.parsed


@runtime_checkable
class OutputCoercionPort(Protocol):
    """P16 — turn model text into fields, or admit it could not.

    Separated from P15 because the *parsing* strategy and the *model* vary
    independently: JSON-schema constrained decoding, grammar-constrained
    generation, regex extraction and logit-biased classification all apply to many
    models, and a model that gains structured-output support should not require a
    new adapter to benefit.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **X1** | Never invents a field. Output keys come from the model's text alone. |
    | **X2** | Never discards. Anything not parsed appears in ``unparsed`` (02_VOM §9.3). |
    | **X3** | Deterministic: identical text yields identical parses (V13). |
    | **X4** | Never raises on malformed input — malformed text is the normal case, and returning an empty parse with the text preserved is the correct answer. |
    """

    @property
    def strategy_id(self) -> str:
        ...

    def coerce(self, raw: str, *, schema: OutputSchema) -> CoercionResult:
        """Parse raw model text against the declared schema. Never raises."""
        ...
