"""The Understanding model (04_MODULES §M9, 02_VOM §9.3 and §10.9).

An ``UnderstandingResult`` is a **claim with its receipt**. The claim is the
attribute list; the receipt is everything needed to defend it six months later —
which model, which prompt version, which crop, what the model actually said, and
what path the engine took to get there.

Three fields carry the weight, and each exists because omitting it produces a
failure nobody detects:

``rejected_fields``
    The model said it; the schema refused it. Recording the refusal is what turns
    *"the VLM never produces posture"* into a statistic with a named cause. §M9's
    failure table makes this the ceiling's producer-side gate, and notes why it is
    a **schema property rather than a review process**: *"it cannot be forgotten
    under deadline pressure."*

``unstructured_note``
    02_VOM §9.3: preserved, **inspectable but never promoted**. Not queryable as
    fact, never entering Vision State, never filterable by the API. It exists so a
    human debugging an odd result can see exactly what the model said — the
    practical difference between a diagnosable platform and a black box.

``decision_path``
    02_VOM §10.9: *"records that the primary model timed out, the fallback ran,
    the quality gate passed at marginal, and the result was accepted with reduced
    confidence. Six months later, that is the difference between explaining a
    result and guessing at it."*

**Nothing here is an observation.** M9 produces candidate claims; M11 decides
whether they become published facts. That boundary is why `01_LAYERED` §1.2 keeps
L4 and L5 apart.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .confidence import Confidence, ConfidenceSemantics
from .crop import TriggerReason
from .ids import (
    AttributeKey,
    BlobRef,
    CameraId,
    ClassId,
    CropId,
    EvidenceId,
    FrameRef,
    ModelId,
    ObjectId,
    PromptId,
    RequestId,
    SiteId,
    TenantId,
)
from .provenance import Provenance
from .timebase import Duration, Instant
from .visual_object import Attribute

UNDERSTANDING_SCHEMA_VERSION = "1.0.0"

#: Cap on preserved free text, in characters.
#:
#: 02_VOM §9.3 says the note is *"bounded"*. A model that returns three pages of
#: prose must not turn an enrichment into a memory problem — and the first
#: paragraph is where the diagnostic value is.
MAX_UNSTRUCTURED_NOTE_CHARS = 4_096


class UnderstandingOutcome(enum.Enum):
    """How an understanding attempt ended.

    Deliberately not a boolean. *"We asked and the model said nothing fits"* and
    *"we never got to ask"* are different facts, and a consumer that cannot tell
    them apart will read both as "no attributes" (invariant V8).
    """

    SUCCEEDED = "succeeded"
    """At least one attribute passed the schema gate."""

    NO_ATTRIBUTES = "no_attributes"
    """The model answered and nothing survived coercion. A real answer."""

    REFUSED = "refused"
    """The model declined — safety filter, content policy. Recorded as evidence."""

    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    """No capable model could be reached, after the fallback chain was exhausted."""

    UNSUPPORTED = "unsupported"
    """No bound understander declares the requested attributes. A **capability
    gap**, not a failure: the consumer learns the demand cannot be served here
    rather than waiting forever (V8)."""

    @property
    def produced_a_model_call(self) -> bool:
        """Whether a model was actually invoked. Drives cost accounting."""
        return self in (
            UnderstandingOutcome.SUCCEEDED,
            UnderstandingOutcome.NO_ATTRIBUTES,
            UnderstandingOutcome.REFUSED,
        )

    @property
    def is_failure(self) -> bool:
        return self in (
            UnderstandingOutcome.TIMED_OUT,
            UnderstandingOutcome.UNAVAILABLE,
            UnderstandingOutcome.UNSUPPORTED,
        )


class RejectionReason(enum.Enum):
    """Why one field the model produced was refused.

    Every value is a *schema* fact, not a judgment about content. That is the
    point of §M9's failure table: a model volunteering ``is_violation`` is
    rejected by exactly the mechanism that rejects a typo, because to the platform
    they are the same thing — an unregistered key.
    """

    UNREGISTERED_KEY = "unregistered_key"
    """**The ceiling violation.** The key is absent from the Attribute Schema
    Registry. If the rate is sustained, a prompt has drifted beyond its declared
    schema and it is alarmed."""

    NOT_IN_OUTPUT_SCHEMA = "not_in_output_schema"
    """Registered, but the prompt did not declare it. The model volunteered
    something nobody asked for (port obligation U1)."""

    WRONG_TYPE = "wrong_type"
    OUT_OF_DOMAIN = "out_of_domain"
    """A value outside the schema's declared allowed set or range."""

    CARDINALITY_VIOLATION = "cardinality_violation"
    CLASS_NOT_APPLICABLE = "class_not_applicable"
    """The attribute is registered but does not apply to this object's class."""

    DEPRECATED_KEY = "deprecated_key"
    UNPARSEABLE_VALUE = "unparseable_value"


class UnderstandingStep(enum.Enum):
    """One step on the decision path (02_VOM §10.9).

    The vocabulary is deliberately about *mechanism* — which model, which retry,
    which gate — and never about content. A step that said "decided the person
    looks suspicious" would be the ceiling leaking into the audit trail.
    """

    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    PROMPT_RESOLVED = "prompt_resolved"
    PROMPT_UNAVAILABLE = "prompt_unavailable"
    MODEL_SELECTED = "model_selected"
    NO_CAPABLE_MODEL = "no_capable_model"
    BATCHED = "batched"
    INVOKED = "invoked"
    TIMED_OUT = "timed_out"
    RETRIED = "retried"
    FALLBACK_MODEL_USED = "fallback_model_used"
    """`10_RELIABILITY` §7.2: *"A fallback is never silent."* This step is how a
    consumer sees the change, and how a drift canary alarms on it."""

    CIRCUIT_OPEN = "circuit_open"
    STRUCTURED_PARSE = "structured_parse"
    REPARSE_ATTEMPTED = "reparse_attempted"
    CONSTRAINED_REASK = "constrained_reask"
    COERCED = "coerced"
    QUARANTINED = "quarantined"
    """Nothing coerced; the output went to ``unstructured_note`` and **zero**
    attributes were emitted."""

    SCHEMA_REJECTED = "schema_rejected"
    MODEL_REFUSED = "model_refused"
    CONFIDENCE_MARKED_SELF_REPORTED = "confidence_marked_self_reported"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One step, with when it happened and any detail worth keeping."""

    step: UnderstandingStep
    detail: str = ""
    at_ms: float = 0.0
    """Milliseconds from the start of the request. Relative rather than absolute
    so a replay produces an identical path (V13)."""

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{self.step.value}{f'({self.detail})' if self.detail else ''}"


@dataclass(frozen=True, slots=True)
class Timing:
    """02_VOM §10.9's ``Timing``, implemented exactly.

    Every field is measured rather than estimated. ``batch_size`` and
    ``model_load_state`` are here because a p99 latency that is really a cold
    start is a different problem from one that is really a queue.
    """

    queued_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    batch_size: int = 1
    device_id: str = ""
    model_load_state: str = "warm"
    """``warm`` | ``cold``."""

    def __post_init__(self) -> None:
        for name in (
            "queued_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @property
    def overhead_ms(self) -> float:
        """Time not spent in the model. The number that says whether the platform
        or the model is the bottleneck."""
        return max(0.0, self.total_ms - self.inference_ms)


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Which weights answered, exactly (invariant V4).

    ``artifact_hash`` is mandatory alongside ``model_id``: a version string is a
    label a human chose, and the hash is what actually ran.
    """

    model_id: ModelId
    model_version: str
    artifact_hash: str
    adapter_id: str = ""
    device_id: str = ""
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model metadata requires a model_id")
        if not self.artifact_hash:
            raise ValueError(
                f"model '{self.model_id}' named without its artifact hash; the "
                f"exact weights that produced a result are mandatory (V4)"
            )

    @property
    def pinned(self) -> str:
        return f"{self.model_id}@{self.model_version}"


@dataclass(frozen=True, slots=True)
class PromptMeta:
    """Which instruction was given, exactly.

    04_MODULES §M10: published versions are immutable, because *"provenance is
    worthless if `prompt@3.2.0` means different things on different days."*
    """

    prompt_id: PromptId
    version: str
    content_hash: str = ""
    """Hash of the rendered text. Detects an asset store that mutated a published
    version — the failure M10's table calls out by name."""

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt metadata requires a prompt_id")
        if not self.version:
            raise ValueError(
                f"prompt '{self.prompt_id}' named without a version; an "
                f"unversioned prompt makes every result it produced unexplainable"
            )

    @property
    def pinned(self) -> str:
        return f"{self.prompt_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class RejectedField:
    """A field the model produced that the schema refused.

    Kept rather than dropped, because §M9's whole argument is that a rejection is
    *information*: a sustained rate means a prompt has drifted, and that is only
    visible if the rejections are counted with their reasons.
    """

    field_name: str
    reason: RejectionReason
    raw_value: str = ""
    """The offending value, stringified and bounded. Preserved so a human can see
    what the model actually said, never parsed into a fact."""

    detail: str = ""

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("a rejection must name the field it refused")

    @property
    def is_ceiling_violation(self) -> bool:
        """Whether this rejection is the Semantic Ceiling doing its job."""
        return self.reason is RejectionReason.UNREGISTERED_KEY


@dataclass(frozen=True, slots=True)
class UnderstandingEvidence:
    """02_VOM §10.9's ``Evidence``, minus ``observation_id``.

    **The omission is deliberate and load-bearing.** §10.9 declares
    ``observation_id`` on the completed record, but M9 creates no observations and
    may not mint an id for an object it is forbidden to create. M11 stamps it when
    it assembles the observation this evidence explains. Everything else — the
    trigger reason, the input hash, the refs, the note, the path, the timing — is
    produced here, because only here is it known.
    """

    evidence_id: EvidenceId
    trigger_reason: TriggerReason
    """**Why this was computed at all.** Inherited from the crop, because the
    reason the platform looked is part of the reason the answer exists."""

    input_hash: str
    """Hash of the exact model input — crop content plus rendered prompt. Two
    results with the same input hash and different answers means the model is
    non-deterministic, which is a fact worth being able to prove."""

    frame_ref: FrameRef
    crop_ref: CropId | None = None
    raw_output_ref: BlobRef | None = None
    """Content-addressed reference to the verbatim model output."""

    unstructured_note: str | None = None
    """02_VOM §9.3. Inspectable, never promoted, bounded."""

    decision_path: tuple[DecisionRecord, ...] = ()
    timing: Timing = Timing()
    retention: str = "evidence"

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence requires an id")
        if not self.input_hash:
            raise ValueError(
                "evidence requires an input hash; without it a result cannot be "
                "shown to be reproducible (invariant V4)"
            )
        if (
            self.unstructured_note is not None
            and len(self.unstructured_note) > MAX_UNSTRUCTURED_NOTE_CHARS
        ):
            raise ValueError(
                f"unstructured_note is {len(self.unstructured_note)} chars; "
                f"02_VOM section 9.3 requires it bounded at "
                f"{MAX_UNSTRUCTURED_NOTE_CHARS}"
            )

    def steps(self) -> tuple[UnderstandingStep, ...]:
        return tuple(record.step for record in self.decision_path)

    def took(self, step: UnderstandingStep) -> bool:
        return any(record.step is step for record in self.decision_path)

    @property
    def used_fallback(self) -> bool:
        """`10_RELIABILITY` §7.2 rule 1 made queryable."""
        return self.took(UnderstandingStep.FALLBACK_MODEL_USED)


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    """04_MODULES §M9's ``UnderstandingResult``, implemented exactly.

    Immutable. A revision is a **new** result, never an edit (V5).

    The invariant that matters: ``attributes`` contains only schema-conformant,
    registered keys, and **everything the model said that is not in it appears in
    ``rejected_fields`` or ``unstructured_note``**. Nothing the model produced is
    ever silently discarded.
    """

    request_id: RequestId
    tenant_id: TenantId
    site_id: SiteId
    camera_id: CameraId
    object_id: ObjectId | None
    """``None`` only for a crop taken without an object — reserved for future
    whole-frame understanding. Every result M9 produces today names its object."""

    class_id: ClassId
    outcome: UnderstandingOutcome
    evidence: UnderstandingEvidence
    provenance: Provenance

    attributes: tuple[Attribute, ...] = ()
    """**Schema-conformant only.** Every key is registered, applies to this
    object's class, and carries a value inside its declared domain."""

    rejected_fields: tuple[RejectedField, ...] = ()
    model_used: ModelMeta | None = None
    """``None`` when no model was reached — unsupported capability, open circuit,
    or an exhausted fallback chain."""

    prompt_used: PromptMeta | None = None
    requested_attributes: tuple[AttributeKey, ...] = ()
    """What was asked for. The difference between this and ``attributes`` is the
    coverage gap, and a consumer needs both to compute it (V8)."""

    demand_ids: tuple[str, ...] = ()
    """Which consumer demands this work served.

    Carried on the *result* rather than passed alongside it, because M11 receives
    only the result: 02_VOM section 11's envelope has ``demand_ids``, and a
    result that could not name its demands would force the handoff to carry two
    things that must be kept in sync."""

    cache_hit: bool = False
    cost_units: float = 0.0
    """Relative cost of this call, in the adapter's declared units."""

    raw_output: bytes | None = None
    """Verbatim model output (port obligation U3). **Data plane** — present on the
    producing node, absent once the result reference travels (V12)."""

    schema_version: str = UNDERSTANDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("an understanding result requires a request_id")
        if self.evidence.crop_ref is None and self.object_id is not None:
            raise ValueError(
                f"result for object '{self.object_id}' carries no crop reference; "
                f"an attribute claim that cannot name the pixels behind it is not "
                f"evidence (02_VOM section 10.9)"
            )
        if self.outcome is UnderstandingOutcome.SUCCEEDED and not self.attributes:
            raise ValueError(
                "outcome is 'succeeded' but no attribute was produced; "
                "'the model answered and nothing fit' is NO_ATTRIBUTES, and "
                "conflating them hides a schema drift problem"
            )
        if self.attributes and self.outcome.is_failure:
            raise ValueError(
                f"outcome '{self.outcome.value}' cannot carry attributes; a "
                f"failed understanding that produced values would be fabrication "
                f"(port obligation U2)"
            )
        if self.outcome.produced_a_model_call and self.model_used is None:
            raise ValueError(
                f"outcome '{self.outcome.value}' implies a model ran, but no "
                f"model metadata was recorded; a result whose model is unknown is "
                f"not reproducible (invariant V4)"
            )

    # --- derived, all pure --------------------------------------------------- #

    @property
    def produced_attributes(self) -> bool:
        return bool(self.attributes)

    @property
    def attribute_keys(self) -> tuple[AttributeKey, ...]:
        return tuple(attribute.key for attribute in self.attributes)

    @property
    def unsatisfied(self) -> tuple[AttributeKey, ...]:
        """Requested attributes that did not arrive. **The V8 field.**

        A consumer reads this to tell "the platform looked and found nothing" from
        "the platform never answered", without having to diff two lists itself.
        """
        produced = set(self.attribute_keys)
        return tuple(key for key in self.requested_attributes if key not in produced)

    @property
    def ceiling_violations(self) -> tuple[RejectedField, ...]:
        """Rejections that were the Semantic Ceiling working.

        Counted separately from ordinary schema mismatches because a sustained
        rate means a **prompt has drifted**, which is a different operator
        response from a model that formats badly.
        """
        return tuple(field for field in self.rejected_fields if field.is_ceiling_violation)

    @property
    def has_unstructured_note(self) -> bool:
        return bool(self.evidence.unstructured_note)

    def without_raw_output(self) -> UnderstandingResult:
        """The control-plane form: everything except the verbatim bytes.

        Invariant V12 made mechanical, and `01_LAYERED` §3.2 sizes the
        understanding-to-builder edge at ~3 KB — *"structured claims + raw output
        reference"*. The reference travels; the payload does not.
        """
        from dataclasses import replace

        return replace(self, raw_output=None)

    def attribute(self, key: AttributeKey) -> Attribute | None:
        for candidate in self.attributes:
            if candidate.key == key:
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class UnderstandingRequest:
    """One unit of work for the engine.

    ``crops`` is a **sequence**, matching P15's ``crops: CropView[]``. Single-frame
    is the degenerate case; `15_ROADMAP` §4 confirms temporal understanding needs
    *"no contract change"* and no module change — only new adapters. Flow 6 ships
    the shape and binds nothing temporal.
    """

    request_id: RequestId
    tenant_id: TenantId
    site_id: SiteId
    camera_id: CameraId
    object_id: ObjectId | None
    class_id: ClassId
    crop_ids: tuple[CropId, ...]
    frame_ref: FrameRef
    requested_attributes: tuple[AttributeKey, ...]
    trigger_reason: TriggerReason
    t_capture: Instant

    prior_attributes: Mapping[AttributeKey, object] = field(default_factory=dict)
    """Context for the prompt. §M10 renders with *"prior values"*; a model told
    what was previously believed can answer "has this changed" cheaply."""

    quality: object = None
    """The crop's ``QualityGrades``. Passed as context so a prompt can hedge on a
    marginal input, never used by M9 to decide whether to proceed — that decision
    was M8's and is already made."""

    priority_class: str = ""
    demand_ids: tuple[str, ...] = ()
    deadline: Duration | None = None

    def __post_init__(self) -> None:
        if not self.crop_ids:
            raise ValueError(
                "an understanding request requires at least one crop; there is "
                "nothing to understand without pixels"
            )
        if not self.requested_attributes:
            raise ValueError(
                "an understanding request with no requested attributes would "
                "spend a model call on nothing"
            )
        if self.frame_ref.camera_id != self.camera_id:
            raise ValueError("a request must name the frame's own camera")

    @property
    def is_temporal(self) -> bool:
        """Whether this is a crop *sequence*. Phase 3's degenerate case today."""
        return len(self.crop_ids) > 1

    @property
    def primary_crop(self) -> CropId:
        return self.crop_ids[0]


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What a request would cost, before it is made (port obligation U7).

    Exists so M8's budget policy can decide. §M9 is explicit that understanding is
    *"governed by M8 rather than by itself"* — M9 answers the question and never
    acts on the answer.
    """

    cost_units: float
    model_id: ModelId | None = None
    estimated_latency: Duration | None = None
    attributes_covered: tuple[AttributeKey, ...] = ()
    attributes_uncovered: tuple[AttributeKey, ...] = ()
    """Requested attributes no bound model can produce. A **capability gap**
    reported before any money is spent."""

    def __post_init__(self) -> None:
        if self.cost_units < 0:
            raise ValueError("cost_units must be non-negative")

    @property
    def fully_covered(self) -> bool:
        return not self.attributes_uncovered


def self_reported(value: float) -> Confidence:
    """A VLM's opinion about itself, labelled honestly.

    02_VOM §7.2 rule 3 and port obligation U4: `SELF_REPORTED` is *"a language
    model's opinion about itself and is not a probability"*. It may be surfaced;
    it may never be the sole basis for a platform decision. Marking it at the
    point of creation is what stops it being compared against a calibrated score
    three modules downstream.
    """
    return Confidence(
        value=max(0.0, min(1.0, value)),
        semantics=ConfidenceSemantics.SELF_REPORTED,
        calibrated=False,
        raw_score=value if 0.0 <= value <= 1.0 else None,
    )


def bounded_note(text: str | None) -> str | None:
    """Truncate free text to the documented bound, marking the truncation.

    Marked rather than silent: a note that was cut must not read like a note the
    model finished.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= MAX_UNSTRUCTURED_NOTE_CHARS:
        return stripped
    suffix = " …[truncated]"
    return stripped[: MAX_UNSTRUCTURED_NOTE_CHARS - len(suffix)] + suffix


def summarize(results: Sequence[UnderstandingResult]) -> dict[str, int]:
    """Outcome counts for a batch. Used by metrics and by operators."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome.value] = counts.get(result.outcome.value, 0) + 1
    return counts
