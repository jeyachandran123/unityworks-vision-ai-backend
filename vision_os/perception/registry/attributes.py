"""The Attribute Schema Registry and its neutrality gate (02_VOM section 9).

> **Single responsibility:** *Decide whether an attribute may exist. Produce
> none, infer nothing.*

``14_TESTING`` section 6 names this the **registry gate**: *"Attempting to
register a judgment-bearing attribute is rejected."* It is one of four
independent gates enforcing the Semantic Ceiling, and the only one inside M7.

The gate is not documentation. ``neutrality_justification`` is required, and a
proposed attribute must name the **visible evidence** that supports it:

| Proposed | Verdict |
|---|---|
| `headwear_present: bool` — "head region shows a covering" | registered |
| `posture: enum` — "body configuration is directly visible" | registered |
| `queue_position: count` — "ordinal position along a region's axis" | registered (pure geometry) |
| `is_employee: bool` — "uniform implies employment" | **rejected** — role, not appearance |
| `is_compliant: bool` — "missing helmet is a violation" | **rejected** — policy |
| `wait_time_excessive: bool` — "dwell exceeds threshold" | **rejected** — threshold is business |

Note the pattern in every rejection: **the rejected attribute is the accepted one
plus a business premise.** The registry always has a neutral counterpart to
offer, which is why enforcing the gate does not block delivery — it relocates a
line of logic to where it belongs.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

from ...core.errors import AttributeRejectedError
from ...core.model.ids import AttributeKey, ClassId
from ...core.model.timebase import Duration


class AttributeValueType(enum.Enum):
    """02_VOM section 9 value types. Closed set."""

    ENUM = "enum"
    BOOL = "bool"
    SCALAR = "scalar"
    VECTOR = "vector"
    TEXT = "text"
    RELATION = "relation"
    COUNT = "count"


class Cardinality(enum.Enum):
    SINGLE = "single"
    MULTI = "multi"


class EvidenceRequirement(enum.Enum):
    CROP = "crop"
    FRAME = "frame"
    SEQUENCE = "sequence"


class SchemaStatus(enum.Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


#: Key tokens that name a judgment rather than an appearance.
#:
#: Two shapes dominate: a **role** ("employee", "customer", "intruder") and a
#: **verdict** ("compliant", "violation", "excessive", "suspicious"). Both encode
#: a premise the platform does not hold.
_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        # roles — who someone is, not what is visible
        "employee", "staff", "customer", "shopper", "visitor", "patient",
        "nurse", "doctor", "guard", "intruder", "trespasser", "suspect",
        "owner", "driver", "operator", "worker", "manager", "vip",
        # verdicts — conclusions requiring a policy
        "compliant", "compliance", "violation", "violating", "unauthorized",
        "authorised", "authorized", "permitted", "forbidden", "illegal",
        "suspicious", "abnormal", "anomalous", "anomaly", "threat", "danger",
        "dangerous", "unsafe", "risky", "alert", "incident",
        # thresholds — a number only a consumer owns
        "excessive", "insufficient", "overcrowded", "crowded", "congested",
        "busy", "idle", "loitering", "queueing", "waiting", "delayed", "late",
        "overdue", "understaffed", "overstaffed",
        # business nouns
        "order", "invoice", "sku", "shift", "appointment", "booking",
    }
)

#: Prefixes that almost always introduce a judgment.
#:
#: ``is_`` and ``has_`` are not forbidden outright — ``has_headwear`` is a
#: perfectly neutral claim about pixels. They are forbidden *in combination with*
#: a judgment token, which the token set above already catches. What is caught
#: here is the residue: keys asserting a verdict with no noun at all.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {"is_ok", "is_good", "is_bad", "is_normal", "is_valid", "is_correct", "status"}
)

#: A justification must name visible evidence. These words indicate the author
#: reached for a premise instead.
_WEAK_JUSTIFICATION: frozenset[str] = frozenset(
    {"implies", "means", "indicates that they are", "should", "must", "policy",
     "rule", "requirement", "because they are", "obviously"}
)

_MIN_JUSTIFICATION_CHARS = 12


@dataclass(frozen=True, slots=True)
class AttributeSchema:
    """One registered attribute definition (02_VOM section 9)."""

    key: AttributeKey
    value_type: AttributeValueType
    neutrality_justification: str
    """**Required.** What visible evidence supports this attribute."""

    applies_to: tuple[ClassId, ...] = ()
    domain: tuple[str, ...] = ()
    """Allowed values for ``ENUM``; unit or range description otherwise."""

    cardinality: Cardinality = Cardinality.SINGLE
    validity: Duration | None = None
    """Default staleness horizon. ``None`` means the value does not expire."""

    evidence_requirement: EvidenceRequirement = EvidenceRequirement.CROP
    version: str = "1.0.0"
    status: SchemaStatus = SchemaStatus.ACTIVE

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("an attribute schema requires a key")
        if self.value_type is AttributeValueType.ENUM and not self.domain:
            raise ValueError(
                f"attribute '{self.key}' is an enum but declares no domain; an "
                f"unconstrained enum is a text field wearing a type"
            )

    @property
    def is_active(self) -> bool:
        return self.status is SchemaStatus.ACTIVE

    def accepts(self, value: object) -> str | None:
        """Whether a value conforms. Returns ``None`` on success, else the reason.

        Added in Flow 6, where M9 must *"coerce raw model output into the declared
        schema, and quarantine what does not fit"* (04_MODULES section M9
        responsibility 4). The check lives here, on the schema, rather than in M9
        — a second implementation of "fits" would drift from this one, and the
        drift would show up as attributes the registry considers illegal sitting
        in the observation log.

        A **string** rather than a boolean, because the caller records *why* a
        field was refused: 04_MODULES section M9 makes a sustained rejection rate
        the signal that a prompt has drifted, and that is only diagnosable if the
        reason is attributed.
        """
        # ``VECTOR`` is the one type whose *single* value is itself a sequence, so
        # the cardinality check cannot be applied to it structurally. Without this
        # branch a vector attribute at the default SINGLE cardinality could never
        # be accepted at all — the value would always look like a multi-value.
        if self.value_type is AttributeValueType.VECTOR:
            if self.cardinality is Cardinality.MULTI:
                if not isinstance(value, list | tuple) or not all(
                    isinstance(item, list | tuple) for item in value
                ):
                    return "expected a sequence of vectors"
                for item in value:
                    reason = self._accepts_one(item)
                    if reason is not None:
                        return reason
                return None
            return self._accepts_one(value)

        if self.cardinality is Cardinality.MULTI:
            if not isinstance(value, list | tuple):
                return "expected a sequence for a multi-valued attribute"
            for item in value:
                reason = self._accepts_one(item)
                if reason is not None:
                    return reason
            return None
        if isinstance(value, list | tuple):
            return "expected a single value, got a sequence"
        return self._accepts_one(value)

    def _accepts_one(self, value: object) -> str | None:
        kind = self.value_type

        if kind is AttributeValueType.BOOL:
            # ``bool`` before ``int``: in Python ``True`` is an ``int``, and a
            # model returning 1 for a boolean is guessing at the encoding rather
            # than answering the question.
            return None if isinstance(value, bool) else "expected a boolean"

        if kind is AttributeValueType.ENUM:
            if not isinstance(value, str):
                return "expected a string for an enum attribute"
            if value not in self.domain:
                return f"'{value}' is outside the declared domain {list(self.domain)}"
            return None

        if kind is AttributeValueType.SCALAR:
            if isinstance(value, bool) or not isinstance(value, int | float):
                return "expected a number"
            return self._within_range(float(value))

        if kind is AttributeValueType.COUNT:
            if isinstance(value, bool) or not isinstance(value, int):
                return "expected an integer count"
            if value < 0:
                return f"count must be non-negative, got {value}"
            return self._within_range(float(value))

        if kind is AttributeValueType.TEXT:
            return None if isinstance(value, str) else "expected text"

        if kind is AttributeValueType.RELATION:
            if not isinstance(value, str) or not value:
                return "expected a referent identifier"
            if self.domain and value not in self.domain:
                return (
                    f"'{value}' is outside the declared domain "
                    f"{list(self.domain)} of referent classes"
                )
            return None

        if kind is AttributeValueType.VECTOR:
            if not isinstance(value, list | tuple) or not value:
                return "expected a non-empty numeric vector"
            if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
                return "vector components must be numbers"
            return None

        return f"unhandled value type {kind.value}"  # pragma: no cover - closed enum

    def _within_range(self, value: float) -> str | None:
        """Apply a ``min:max`` domain when one is declared.

        The domain is text for every type except ``ENUM``, so a numeric range is
        expressed as a single ``"0:1"`` entry. An unparseable range is treated as
        *no constraint* rather than as a failure: a malformed schema must not
        silently reject every value a model produces.
        """
        if len(self.domain) != 1 or ":" not in self.domain[0]:
            return None
        low_text, _, high_text = self.domain[0].partition(":")
        try:
            low, high = float(low_text), float(high_text)
        except ValueError:
            return None
        if not low <= value <= high:
            return f"{value} is outside the declared range [{low}, {high}]"
        return None


def _tokens(key: str) -> set[str]:
    return {part for part in re.split(r"[_.\-]", key.lower()) if part}


def check_neutrality(key: AttributeKey, justification: str) -> None:
    """Raise unless the attribute names visible evidence.

    Raises:
        AttributeRejectedError: the key encodes a role, a verdict, or a
            threshold, or the justification appeals to a premise rather than to
            pixels. The message names the neutral counterpart to register
            instead, because the gate exists to relocate a line of logic, not to
            block delivery.
    """
    lowered = key.lower()
    offending = _tokens(lowered) & _FORBIDDEN_TOKENS

    if offending:
        token = sorted(offending)[0]
        raise AttributeRejectedError(
            f"attribute '{key}' is rejected: '{token}' names a role, a verdict or "
            f"a threshold, none of which are visible in pixels. Register what can "
            f"be seen instead — 'uniform_present' rather than 'is_employee', "
            f"'helmet_present' rather than 'is_compliant', 'dwell_duration' "
            f"rather than 'wait_time_excessive' (02_VOM section 9.1, invariant V1)",
            attribute_key=str(key),
            token=token,
        )

    if lowered in _FORBIDDEN_KEYS:
        raise AttributeRejectedError(
            f"attribute '{key}' is rejected: it asserts a verdict with no visual "
            f"referent. Name the observable property instead (invariant V1)",
            attribute_key=str(key),
        )

    if len(justification.strip()) < _MIN_JUSTIFICATION_CHARS:
        raise AttributeRejectedError(
            f"attribute '{key}' is rejected: neutrality_justification is required "
            f"and must name the visible evidence supporting the attribute. It is "
            f"the registration gate, not documentation (02_VOM section 9.1)",
            attribute_key=str(key),
        )

    weak = [phrase for phrase in _WEAK_JUSTIFICATION if phrase in justification.lower()]
    if weak:
        raise AttributeRejectedError(
            f"attribute '{key}' is rejected: its justification appeals to "
            f"'{weak[0]}' rather than to visible evidence. A justification must "
            f"say what the pixels show, not what it would imply (invariant V1)",
            attribute_key=str(key),
            phrase=weak[0],
        )


@dataclass(slots=True)
class AttributeRegistry:
    """The registered attribute vocabulary for a deployment.

    Injected into the Object Registry, which consults it before holding any
    attribute value. An unregistered attribute is refused: the gate is worthless
    if a producer can bypass it by simply not asking.
    """

    schemas: dict[AttributeKey, AttributeSchema] = field(default_factory=dict)

    def register(self, schema: AttributeSchema) -> AttributeSchema:
        """Admit an attribute after the neutrality gate.

        Raises:
            AttributeRejectedError: the attribute is judgment-bearing.
        """
        check_neutrality(schema.key, schema.neutrality_justification)
        self.schemas[schema.key] = schema
        return schema

    def get(self, key: AttributeKey) -> AttributeSchema | None:
        return self.schemas.get(key)

    def require(self, key: AttributeKey) -> AttributeSchema:
        schema = self.schemas.get(key)
        if schema is None:
            raise AttributeRejectedError(
                f"attribute '{key}' is not registered; the registry holds only "
                f"attributes that have passed the neutrality gate",
                attribute_key=str(key),
            )
        if not schema.is_active:
            raise AttributeRejectedError(
                f"attribute '{key}' is deprecated and no longer accepted",
                attribute_key=str(key),
            )
        return schema

    def applies(self, key: AttributeKey, class_id: ClassId) -> bool:
        """Whether this attribute may be carried by this taxonomy class.

        An empty ``applies_to`` means "any class" — a deliberate degenerate case
        rather than an omission, since some attributes (``truncation``) are
        genuinely class-independent.
        """
        schema = self.schemas.get(key)
        if schema is None:
            return False
        if not schema.applies_to:
            return True
        return any(
            class_id == allowed or class_id.startswith(f"{allowed}.")
            for allowed in schema.applies_to
        )

    def __contains__(self, key: object) -> bool:
        return key in self.schemas

    def __len__(self) -> int:
        return len(self.schemas)
