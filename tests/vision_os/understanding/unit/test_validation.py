"""The schema gate — where unbounded model output meets a typed contract.

The most important test in this file is
``test_a_judgment_is_rejected_by_the_same_mechanism_as_a_typo``. 04_MODULES §M9
explains why that equivalence is the design rather than a coincidence:

> *Model emits a judgment ("this is a violation") — rejected by the same
> mechanism; it is simply an unregistered key. **This is why the ceiling is a
> schema property rather than a review process** — it cannot be forgotten under
> deadline pressure.*

The second most important is the accounting invariant: **every field the model
produced ends in exactly one of accepted, rejected, or the unstructured note.**
Nothing is silently discarded (02_VOM §9.3).
"""

from __future__ import annotations

import pytest

from vision_os.core.model.confidence import ConfidenceSemantics
from vision_os.core.model.ids import AttributeKey, ConfigRevision, ModuleId
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.timebase import Duration
from vision_os.core.model.understanding import RejectionReason
from vision_os.core.ports.understanding import OutputSchema
from vision_os.perception.registry.attributes import (
    AttributeSchema,
    AttributeValueType,
    Cardinality,
    SchemaStatus,
)
from vision_os.perception.understanding import AttributeValidator, unsatisfied

from ..conftest import (
    CARRYING,
    HEADWEAR,
    HEIGHT,
    PERSON,
    POSTURE,
    UNREGISTERED,
    VEHICLE,
    at,
    build_registry,
)

PRODUCER = Provenance(
    producer_module=ModuleId("understanding_engine"),
    producer_version="1.0.0",
    config_revision=ConfigRevision("test"),
)

FULL_SCHEMA = OutputSchema(fields=(POSTURE, HEADWEAR, HEIGHT, CARRYING))


@pytest.fixture
def validator(attribute_registry) -> AttributeValidator:
    return AttributeValidator(attribute_registry)


def run(validator, fields, *, schema=FULL_SCHEMA, class_id=PERSON, confidence=None):
    return validator.validate(
        fields,
        schema=schema,
        class_id=class_id,
        observed_at=at(3),
        producer=PRODUCER,
        evidence_ref="crop-1",
        field_confidence=confidence,
    )


class TestTheCeiling:
    def test_a_judgment_is_rejected_by_the_same_mechanism_as_a_typo(
        self, validator
    ) -> None:
        """**The design, in one assertion.**

        A model volunteering ``is_violation`` and a model misspelling ``postur``
        produce the *identical* rejection reason, because to the platform they
        are the identical event: a key the registry does not hold. That is what
        makes the ceiling impossible to forget under deadline pressure — there is
        no human step to skip.
        """
        outcome = run(
            validator,
            {str(UNREGISTERED): True, "postur": "standing"},
        )
        assert not outcome.accepted
        assert len(outcome.rejected) == 2
        assert {field.reason for field in outcome.rejected} == {
            RejectionReason.UNREGISTERED_KEY
        }

    def test_a_ceiling_violation_is_counted_separately(self, validator) -> None:
        """A sustained rate means a prompt drifted — a different response from a
        model that formats badly."""
        outcome = run(
            validator, {str(UNREGISTERED): True, str(POSTURE): "levitating"}
        )
        assert len(outcome.ceiling_violations) == 1
        assert outcome.ceiling_violations[0].field_name == str(UNREGISTERED)
        assert len(outcome.rejected) == 2, "the bad enum is rejected too, differently"

    def test_the_rejected_value_is_preserved(self, validator) -> None:
        """So a human can see what the model actually said.

        Preserved, never parsed back: the whole point of the rejection is that
        this value is not a fact.
        """
        outcome = run(validator, {str(UNREGISTERED): "the worker is unsafe"})
        assert outcome.rejected[0].raw_value == "the worker is unsafe"

    def test_a_deprecated_attribute_is_refused(self) -> None:
        registry = build_registry()
        registry.schemas[POSTURE] = AttributeSchema(
            key=POSTURE,
            value_type=AttributeValueType.ENUM,
            domain=("standing",),
            neutrality_justification="Body configuration is directly visible",
            status=SchemaStatus.DEPRECATED,
        )
        outcome = run(AttributeValidator(registry), {str(POSTURE): "standing"})
        assert outcome.rejected[0].reason is RejectionReason.DEPRECATED_KEY

    def test_a_registered_key_the_prompt_did_not_declare_is_refused(
        self, validator
    ) -> None:
        """Port obligation U1. The model volunteered something nobody asked for."""
        outcome = run(
            validator,
            {str(POSTURE): "standing", str(HEADWEAR): True},
            schema=OutputSchema(fields=(POSTURE,)),
        )
        assert len(outcome.accepted) == 1
        assert outcome.rejected[0].reason is RejectionReason.NOT_IN_OUTPUT_SCHEMA

    def test_a_lenient_schema_admits_undeclared_registered_keys(
        self, validator
    ) -> None:
        """The shadow-evaluation case, and the only reason ``strict`` exists."""
        outcome = run(
            validator,
            {str(POSTURE): "standing", str(HEADWEAR): True},
            schema=OutputSchema(fields=(POSTURE,), strict=False),
        )
        assert len(outcome.accepted) == 2


class TestValueValidation:
    def test_an_enum_outside_its_domain_is_refused(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): "levitating"})
        assert outcome.rejected[0].reason is RejectionReason.OUT_OF_DOMAIN
        assert "levitating" in outcome.rejected[0].detail

    def test_a_valid_enum_is_accepted(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): "sitting"})
        assert outcome.accepted[0].value == "sitting"

    def test_a_string_for_a_boolean_is_refused(self, validator) -> None:
        """``"yes"`` is not ``True``. Coercing it would hide a prompt bug forever.

        The right fix is a prompt that says "answer true or false", and the
        rejection is what tells someone to make it.
        """
        outcome = run(validator, {str(HEADWEAR): "yes"})
        assert outcome.rejected[0].reason is RejectionReason.WRONG_TYPE

    def test_an_integer_for_a_boolean_is_refused(self, validator) -> None:
        """``1`` is an ``int`` that Python calls truthy; the schema does not.

        A model returning 1 for a boolean is guessing at the encoding rather than
        answering the question.
        """
        outcome = run(validator, {str(HEADWEAR): 1})
        assert outcome.rejected[0].reason is RejectionReason.WRONG_TYPE

    def test_a_scalar_outside_its_range_is_refused(self, validator) -> None:
        outcome = run(validator, {str(HEIGHT): 1.4})
        assert outcome.rejected[0].reason is RejectionReason.OUT_OF_DOMAIN

    def test_a_scalar_inside_its_range_is_accepted(self, validator) -> None:
        outcome = run(validator, {str(HEIGHT): 0.62})
        assert outcome.accepted[0].value == pytest.approx(0.62)

    def test_a_multi_valued_attribute_accepts_a_list(self, validator) -> None:
        outcome = run(validator, {str(CARRYING): ["bag", "tool"]})
        assert outcome.accepted[0].value == ["bag", "tool"]

    def test_a_multi_valued_attribute_refuses_a_scalar(self, validator) -> None:
        outcome = run(validator, {str(CARRYING): "bag"})
        assert outcome.rejected[0].reason is RejectionReason.CARDINALITY_VIOLATION

    def test_a_single_valued_attribute_refuses_a_list(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): ["standing", "sitting"]})
        assert outcome.rejected[0].reason is RejectionReason.CARDINALITY_VIOLATION

    def test_a_relation_outside_its_referents_is_refused(self, validator) -> None:
        outcome = run(validator, {str(CARRYING): ["spaceship"]})
        assert outcome.rejected[0].reason is RejectionReason.OUT_OF_DOMAIN


class TestApplicability:
    def test_an_attribute_scoped_to_another_class_is_refused(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): "standing"}, class_id=VEHICLE)
        assert outcome.rejected[0].reason is RejectionReason.CLASS_NOT_APPLICABLE

    def test_a_class_independent_attribute_applies_everywhere(self, validator) -> None:
        """An empty ``applies_to`` is a deliberate degenerate case, not an omission."""
        outcome = run(validator, {str(HEIGHT): 0.5}, class_id=VEHICLE)
        assert outcome.accepted[0].key == HEIGHT

    def test_a_subclass_inherits_applicability(self, validator) -> None:
        from vision_os.core.model.ids import ClassId

        outcome = run(
            validator, {str(POSTURE): "standing"}, class_id=ClassId("person.child")
        )
        assert outcome.accepted, "dotted taxonomy paths must inherit applicability"


class TestConfidence:
    def test_model_confidence_is_labelled_self_reported(self, validator) -> None:
        """02_VOM §7.2 rule 3 and obligation U4.

        A VLM's number about itself *"is a language model's opinion about itself
        and is not a probability"*. Labelling it at creation is what stops it
        being compared against a calibrated detector score three modules later.
        """
        outcome = run(
            validator, {str(POSTURE): "standing"}, confidence={str(POSTURE): 0.87}
        )
        confidence = outcome.accepted[0].confidence
        assert confidence.semantics is ConfidenceSemantics.SELF_REPORTED
        assert not confidence.calibrated
        assert confidence.value == pytest.approx(0.87)
        assert confidence.raw_score == pytest.approx(0.87)

    def test_absent_confidence_does_not_become_certainty(self, validator) -> None:
        """A model that did not say how sure it is has not said it is certain.

        Defaulting to 1.0 would manufacture confidence the platform never
        received — fabrication wearing a different hat.
        """
        outcome = run(validator, {str(POSTURE): "standing"})
        assert outcome.accepted[0].confidence.value < 1.0
        assert outcome.accepted[0].confidence.semantics is ConfidenceSemantics.SELF_REPORTED

    def test_out_of_range_confidence_is_clamped_not_rejected(self, validator) -> None:
        """A model reporting 1.7 is wrong about the scale, not about the answer."""
        outcome = run(
            validator, {str(POSTURE): "standing"}, confidence={str(POSTURE): 1.7}
        )
        assert outcome.accepted[0].confidence.value == 1.0
        assert outcome.accepted[0].confidence.raw_score is None, (
            "a raw score outside [0,1] is not preserved as if it were valid"
        )


class TestStalenessHorizon:
    def test_validity_becomes_valid_until(self, validator) -> None:
        """Stamped here so a consumer never looks up a schema to know if a value
        is current (V8)."""
        outcome = run(validator, {str(POSTURE): "standing"})
        assert outcome.accepted[0].valid_until is not None
        assert outcome.accepted[0].valid_until.ns == at(3).ns + Duration.from_millis(
            60_000
        ).ns

    def test_no_validity_means_no_expiry(self, validator) -> None:
        """A deliberate case, not an omission (02_VOM §9)."""
        outcome = run(validator, {str(HEADWEAR): True})
        assert outcome.accepted[0].valid_until is None


class TestTheAccountingInvariant:
    def test_every_field_is_accepted_or_rejected(self, validator) -> None:
        """Nothing the model produced is silently discarded."""
        fields = {
            str(POSTURE): "standing",
            str(HEADWEAR): True,
            str(HEIGHT): 9.0,
            str(UNREGISTERED): True,
            "gibberish": [1, 2],
        }
        outcome = run(validator, fields)
        assert outcome.total_fields == len(fields)

    def test_output_order_is_stable(self, validator) -> None:
        """Two runs over the same response must agree (V13).

        Dict iteration order is insertion order in Python, so a response whose
        keys arrived differently would otherwise produce a differently-ordered
        attribute list — and a replay that differs in order is a replay that
        differs.
        """
        forward = run(validator, {str(POSTURE): "standing", str(HEADWEAR): True})
        backward = run(validator, {str(HEADWEAR): True, str(POSTURE): "standing"})
        assert [a.key for a in forward.accepted] == [a.key for a in backward.accepted]

    def test_validation_is_pure(self, validator) -> None:
        fields = {str(POSTURE): "standing", str(UNREGISTERED): True}
        first = run(validator, fields)
        second = run(validator, fields)
        assert [a.key for a in first.accepted] == [a.key for a in second.accepted]
        assert [r.reason for r in first.rejected] == [r.reason for r in second.rejected]

    def test_an_empty_response_produces_nothing_and_says_so(self, validator) -> None:
        outcome = run(validator, {})
        assert outcome.produced_nothing
        assert outcome.total_fields == 0

    def test_unsatisfied_reports_the_coverage_gap(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): "standing"})
        missing = unsatisfied((POSTURE, HEADWEAR, CARRYING), outcome.accepted)
        assert missing == (HEADWEAR, CARRYING)


class TestProvenanceOnEveryAttribute:
    def test_each_attribute_carries_its_producer(self, validator) -> None:
        outcome = run(validator, {str(POSTURE): "standing"})
        assert outcome.accepted[0].producer.config_revision == ConfigRevision("test")

    def test_each_attribute_carries_its_evidence_reference(self, validator) -> None:
        """A claim that cannot name the pixels behind it is not evidence."""
        outcome = run(validator, {str(POSTURE): "standing"})
        assert outcome.accepted[0].evidence_ref == "crop-1"

    def test_each_attribute_carries_its_schema_version(self, validator) -> None:
        """So a value stays interpretable when the schema evolves."""
        outcome = run(validator, {str(POSTURE): "standing"})
        assert outcome.accepted[0].schema_version == "1.0.0"


class TestSchemaAcceptsDirectly:
    """The ``accepts`` method itself — one definition of "fits"."""

    def test_a_bool_schema_takes_only_bools(self) -> None:
        schema = AttributeSchema(
            key=AttributeKey("k"),
            value_type=AttributeValueType.BOOL,
            neutrality_justification="A visible covering is present or not",
        )
        assert schema.accepts(True) is None
        assert schema.accepts(1) is not None
        assert schema.accepts("true") is not None

    def test_a_count_refuses_negatives(self) -> None:
        schema = AttributeSchema(
            key=AttributeKey("k"),
            value_type=AttributeValueType.COUNT,
            neutrality_justification="Countable items are visible in the region",
        )
        assert schema.accepts(3) is None
        assert schema.accepts(-1) is not None
        assert schema.accepts(1.5) is not None

    def test_a_vector_requires_numbers(self) -> None:
        schema = AttributeSchema(
            key=AttributeKey("k"),
            value_type=AttributeValueType.VECTOR,
            neutrality_justification="A measured multi-component quantity",
        )
        assert schema.accepts([1.0, 2.0]) is None
        assert schema.accepts([]) is not None
        assert schema.accepts(["a"]) is not None

    def test_a_malformed_range_constrains_nothing(self) -> None:
        """A broken schema must not silently reject every value a model produces.

        Failing open here is deliberate: the alternative is an attribute that
        never arrives and a rejection reason that blames the model.
        """
        schema = AttributeSchema(
            key=AttributeKey("k"),
            value_type=AttributeValueType.SCALAR,
            domain=("not:a:range",),
            neutrality_justification="A measured ratio visible in the frame",
        )
        assert schema.accepts(999.0) is None

    def test_multi_cardinality_validates_each_item(self) -> None:
        schema = AttributeSchema(
            key=AttributeKey("k"),
            value_type=AttributeValueType.ENUM,
            domain=("a", "b"),
            cardinality=Cardinality.MULTI,
            neutrality_justification="Multiple visible markers may co-occur",
        )
        assert schema.accepts(["a", "b"]) is None
        assert schema.accepts(["a", "z"]) is not None
