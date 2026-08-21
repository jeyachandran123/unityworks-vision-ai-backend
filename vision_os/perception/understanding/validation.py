"""The schema gate — where unbounded model output meets a typed contract.

> **Single responsibility:** *Turn what a model said into registered attributes,
> and account for everything that did not fit.*

This module is the producer-side expression of the Semantic Ceiling. 04_MODULES
§M9 explains why it is a schema property rather than a review process:

> *Model emits a judgment ("this is a violation") — rejected by the same
> mechanism; it is simply an unregistered key. **This is why the ceiling is a
> schema property rather than a review process** — it cannot be forgotten under
> deadline pressure.*

A judgment and a typo are the same event here, and that is the design working.

**The accounting invariant.** Every field the model produced ends in exactly one
of three places: an accepted `Attribute`, a `RejectedField` with a named reason,
or — for output that never parsed at all — the `unstructured_note`. Nothing is
silently discarded (02_VOM §9.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...core.model.confidence import Confidence, ConfidenceSemantics
from ...core.model.ids import AttributeKey, ClassId
from ...core.model.provenance import Provenance
from ...core.model.timebase import Duration, Instant
from ...core.model.understanding import (
    RejectedField,
    RejectionReason,
    self_reported,
)
from ...core.model.visual_object import Attribute
from ...core.ports.understanding import OutputSchema
from ..registry.attributes import AttributeRegistry, SchemaStatus

#: Longest raw value kept on a rejection record, in characters.
#:
#: Enough to see what the model said, bounded so a pathological response cannot
#: turn a rejection log into a memory problem.
MAX_REJECTED_VALUE_CHARS = 256

#: Confidence assigned when a model offers none.
#:
#: Deliberately **not** 1.0. A model that did not say how sure it is has not said
#: it is certain, and defaulting to certainty would manufacture confidence the
#: platform never received — which is the fabrication U2 forbids, wearing a
#: different hat.
DEFAULT_SELF_REPORTED_CONFIDENCE = 0.5


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The result of putting one model response through the gate.

    ``accepted`` and ``rejected`` partition the model's fields exactly. A caller
    that wanted to know "did everything the model said get accounted for" can add
    the two lengths and compare — which is what the engine's own test does.
    """

    accepted: tuple[Attribute, ...] = ()
    rejected: tuple[RejectedField, ...] = ()

    @property
    def total_fields(self) -> int:
        return len(self.accepted) + len(self.rejected)

    @property
    def ceiling_violations(self) -> tuple[RejectedField, ...]:
        return tuple(f for f in self.rejected if f.is_ceiling_violation)

    @property
    def produced_nothing(self) -> bool:
        return not self.accepted


class AttributeValidator:
    """Applies the Attribute Schema Registry to model output.

    Holds the registry rather than a copy of it: the registry is the ceiling's
    first gate (00_CHARTER §4.3) and a second copy would be a second, unenforced
    vocabulary.

    Stateless and pure — the same response, schema and class always produce the
    same outcome, which is what makes a rejection reproducible from a replay.
    """

    __slots__ = ("_registry",)

    def __init__(self, registry: AttributeRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> AttributeRegistry:
        return self._registry

    def validate(
        self,
        structured: Mapping[str, object],
        *,
        schema: OutputSchema,
        class_id: ClassId,
        observed_at: Instant,
        producer: Provenance,
        evidence_ref: str | None = None,
        field_confidence: Mapping[str, float] | None = None,
    ) -> ValidationOutcome:
        """Put one model response through the gate.

        Checks run in a fixed order, cheapest and most consequential first, so a
        field is attributed to the *first* thing wrong with it. An unregistered
        key is reported as a ceiling violation even if its value is also
        malformed, because those call for different responses: one means a prompt
        drifted, the other means a model formats badly.
        """
        accepted: list[Attribute] = []
        rejected: list[RejectedField] = []
        confidences = field_confidence or {}

        # Sorted so the output order is a property of the data rather than of
        # dict insertion — two runs over the same response must agree (V13).
        for field_name in sorted(structured):
            value = structured[field_name]
            reason, detail = self._reject_reason(field_name, value, schema, class_id)
            if reason is not None:
                rejected.append(
                    RejectedField(
                        field_name=field_name,
                        reason=reason,
                        raw_value=_bounded(value),
                        detail=detail,
                    )
                )
                continue

            key = AttributeKey(field_name)
            attribute_schema = self._registry.require(key)
            accepted.append(
                Attribute(
                    key=key,
                    schema_version=attribute_schema.version,
                    value=value,
                    confidence=self._confidence_for(field_name, confidences),
                    observed_at=observed_at,
                    producer=producer,
                    valid_until=_valid_until(observed_at, attribute_schema.validity),
                    evidence_ref=evidence_ref,
                )
            )

        return ValidationOutcome(accepted=tuple(accepted), rejected=tuple(rejected))

    # --- the gate, one field at a time ------------------------------------------ #

    def _reject_reason(
        self,
        field_name: str,
        value: object,
        schema: OutputSchema,
        class_id: ClassId,
    ) -> tuple[RejectionReason | None, str]:
        key = AttributeKey(field_name)

        # 1. The ceiling. An unregistered key is refused whether it is a typo or
        #    a verdict — to the platform they are the same event.
        registered = self._registry.get(key)
        if registered is None:
            return (
                RejectionReason.UNREGISTERED_KEY,
                "not in the Attribute Schema Registry; a key the registry does "
                "not hold has not passed the neutrality gate",
            )

        if registered.status is not SchemaStatus.ACTIVE:
            return (
                RejectionReason.DEPRECATED_KEY,
                f"attribute '{key}' is deprecated and no longer accepted",
            )

        # 2. The prompt's own declaration. A registered key the prompt did not
        #    ask for is a model volunteering (port obligation U1).
        if schema.strict and not schema.declares(key):
            return (
                RejectionReason.NOT_IN_OUTPUT_SCHEMA,
                f"the prompt did not declare '{key}'; the model volunteered it",
            )

        # 3. Applicability. A registered attribute on the wrong class is a claim
        #    the schema explicitly scoped away from.
        if not self._registry.applies(key, class_id):
            return (
                RejectionReason.CLASS_NOT_APPLICABLE,
                f"'{key}' does not apply to class '{class_id}'",
            )

        # 4. The value itself, judged by the schema that declared it.
        problem = registered.accepts(value)
        if problem is not None:
            return (_value_reason(problem), problem)

        return (None, "")

    def _confidence_for(
        self, field_name: str, confidences: Mapping[str, float]
    ) -> Confidence:
        """Attribute confidence, always honestly labelled.

        02_VOM §7.2 rule 3 and port obligation U4: a VLM's number about itself is
        ``SELF_REPORTED`` — *"a language model's opinion about itself and is not a
        probability"*. Labelling it here, at the moment of creation, is what stops
        it being compared against a calibrated detector score three modules later.
        """
        raw = confidences.get(field_name)
        if raw is None:
            return self_reported(DEFAULT_SELF_REPORTED_CONFIDENCE)
        return self_reported(float(raw))


def _value_reason(problem: str) -> RejectionReason:
    """Map the schema's complaint onto a rejection reason.

    Three buckets, because they are three different operator responses: a wrong
    *type* is usually a prompt phrasing problem, an out-of-*domain* value is
    usually a model that knows a richer vocabulary than the registry, and a
    *cardinality* mismatch is usually a schema that should have been multi.
    """
    lowered = problem.lower()
    if "outside the declared domain" in lowered or "outside the declared range" in lowered:
        return RejectionReason.OUT_OF_DOMAIN
    if "sequence" in lowered:
        return RejectionReason.CARDINALITY_VIOLATION
    if lowered.startswith("expected"):
        return RejectionReason.WRONG_TYPE
    return RejectionReason.UNPARSEABLE_VALUE


def _valid_until(observed_at: Instant, validity: Duration | None) -> Instant | None:
    """The staleness horizon from the schema.

    ``None`` means the value does not expire on a schedule — a deliberate case
    (02_VOM §9), not an omission. Stamping it here means a consumer never has to
    look up the schema to know whether what it is reading is still current (V8).
    """
    if validity is None:
        return None
    return Instant(observed_at.ns + validity.ns)


def _bounded(value: object) -> str:
    """Stringify a rejected value, bounded.

    Kept so a human can see what the model actually said. Never parsed back into
    anything: the whole point of the rejection is that this value is not a fact.
    """
    text = str(value)
    if len(text) <= MAX_REJECTED_VALUE_CHARS:
        return text
    return text[: MAX_REJECTED_VALUE_CHARS - 1] + "…"


def unsatisfied(
    requested: Sequence[AttributeKey], accepted: Sequence[Attribute]
) -> tuple[AttributeKey, ...]:
    """Requested attributes that did not arrive. The V8 computation.

    Extracted so both the engine and its tests use one definition of "missing",
    rather than each computing a set difference and disagreeing about order.
    """
    produced = {attribute.key for attribute in accepted}
    return tuple(key for key in requested if key not in produced)


def is_self_reported(attribute: Attribute) -> bool:
    """Whether this claim rests on a model's opinion of itself."""
    return attribute.confidence.semantics is ConfidenceSemantics.SELF_REPORTED
