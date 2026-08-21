"""The final constitutional enforcement point (00_CHARTER §4.3, §M11).

The ceiling has three gates: registration, declared output, and this one. The
first two can be bypassed — a plugin can construct an ``Attribute`` directly, a
non-conformant adapter can return whatever it likes — which is precisely why the
last one exists and why it must re-check what the earlier ones checked.

The brief: *"Nothing reaches Vision State without passing this gate."*

Every test here attacks the gate rather than exercising it. A gate tested only
with valid input is a gate that has never been shown to close.
"""

from __future__ import annotations

import dataclasses

import pytest

from vision_os.core.errors import AttributeRejectedError, ValidationFailedError
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import AttributeKey, ClassId
from vision_os.core.model.observation import ViolationKind
from vision_os.kernel.events import SchemaViolationSpike

from .conftest import (
    HEADWEAR,
    HEIGHT,
    POSTURE,
    UNREGISTERED,
    VEHICLE,
    attribute,
    context,
    make_object,
    presence_of,
    synthesis_config,
    understanding,
)


class TestUnregisteredAttributesAreDropped:
    """§M11: *"drop the attribute, keep the rest of the observation."*"""

    def test_an_unregistered_attribute_never_reaches_the_observation(
        self, builder
    ) -> None:
        """The attack the gate exists for.

        A model that volunteered ``is_authorized`` past M9's schema gate — a
        non-conformant adapter, a plugin building an Attribute by hand — still
        cannot get it into the permanent record.
        """
        result = understanding(
            attributes=(
                attribute(POSTURE, "standing"),
                attribute(UNREGISTERED, True),
            )
        )
        published = builder.build_attribute(make_object(), result, context())

        assert len(published) == 1
        keys = {a.key for a in published[0].attributes}
        assert POSTURE in keys
        assert UNREGISTERED not in keys

    def test_the_rest_of_the_observation_survives(self, builder) -> None:
        """Dropping one attribute must not discard a valid measurement.

        The alternative — rejecting the envelope — would mean one creative model
        response silently deletes the presence and spatial facts that came with
        it.
        """
        result = understanding(
            attributes=(attribute(POSTURE, "sitting"), attribute(UNREGISTERED, True))
        )
        published = builder.build_attribute(make_object(), result, context())
        assert published
        assert published[0].object_id
        assert published[0].provenance is not None

    def test_the_drop_is_recorded_as_a_violation_not_silently(self, builder) -> None:
        """A silent drop is indistinguishable from the model never saying it.

        The record lives on the ``ValidationResult`` and the metrics, not on the
        envelope: 02_VOM §11 declares no violations field, and the observation is
        the record of *facts*. Adding one would be inventing architecture to make
        a test easier.
        """
        from vision_os.core.model.observation import Observation

        result = understanding(
            attributes=(attribute(POSTURE, "standing"), attribute(UNREGISTERED, True))
        )
        candidate = builder.build_attribute(make_object(), result, context())[0]
        assert "violations" not in Observation.__dataclass_fields__

        checked = builder.gate.validate(
            dataclasses.replace(candidate, attributes=result.attributes)
        )
        assert UNREGISTERED in checked.dropped_attributes
        assert ViolationKind.UNREGISTERED_ATTRIBUTE in {
            v.kind for v in checked.violations
        }

    def test_an_observation_of_only_unregistered_attributes_publishes_nothing(
        self, builder
    ) -> None:
        """And returns empty rather than raising.

        The ceiling refusing everything a model said is the ceiling *working*.
        Raising would report a platform failure at the moment the platform
        correctly declined to record something it was never allowed to record —
        and it would conflate the two responses §M11 keeps apart.
        """
        result = understanding(attributes=(attribute(UNREGISTERED, True),))
        assert builder.build_attribute(make_object(), result, context()) == []


class TestAttributeOwnershipIsChecked:
    """An attribute declared for ``person`` may not be asserted about a vehicle."""

    def test_an_attribute_outside_its_declared_class_is_dropped(self, builder) -> None:
        """``posture`` ``applies_to=(person,)``.

        A vehicle with a posture is not a schema error the earlier gates
        necessarily caught — M9 checks the class it was asked about, and a
        mis-routed result carries the wrong one.
        """
        result = understanding(class_id=VEHICLE, attributes=(attribute(POSTURE),))
        assert builder.build_attribute(
            make_object(class_id=VEHICLE), result, context()
        ) == []

    def test_an_unscoped_attribute_applies_to_any_class(self, builder) -> None:
        """``apparent_height_ratio`` declares no ``applies_to`` — it is universal."""
        result = understanding(
            class_id=VEHICLE, attributes=(attribute(HEIGHT, 0.4),)
        )
        published = builder.build_attribute(
            make_object(class_id=VEHICLE), result, context()
        )
        assert published
        assert HEIGHT in {a.key for a in published[0].attributes}


class TestValueDomainsAreRechecked:
    def test_a_value_outside_its_declared_domain_is_dropped(self, builder) -> None:
        """``posture`` declares four values; ``loitering`` is not one of them.

        And could not be: it is an interpretation, not a body configuration.
        """
        result = understanding(attributes=(attribute(POSTURE, "loitering"),))
        assert builder.build_attribute(make_object(), result, context()) == []

    def test_a_scalar_outside_its_range_is_dropped(self, builder) -> None:
        result = understanding(attributes=(attribute(HEIGHT, 4.2),))
        assert builder.build_attribute(make_object(), result, context()) == []

    def test_a_wrongly_typed_value_is_dropped(self, builder) -> None:
        """``headwear_present`` is a BOOL. ``"maybe"`` is not a bool."""
        result = understanding(attributes=(attribute(HEADWEAR, "maybe"),))
        assert builder.build_attribute(make_object(), result, context()) == []


class TestConfidenceValidity:
    def test_an_attribute_carrying_the_wrong_confidence_semantics_is_refused(
        self,
    ) -> None:
        """Enforced at construction — the type refuses before the gate sees it.

        Two probabilities under one name is the defect; catching it in the model
        rather than the gate means no code path anywhere can produce one.
        """
        from vision_os.core.model.visual_object import Attribute

        from .conftest import at, object_provenance

        with pytest.raises(ValueError, match="ATTRIBUTE or SELF_REPORTED"):
            Attribute(
                key=POSTURE,
                schema_version="1.0.0",
                value="standing",
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.DETECTION_PRESENCE
                ),
                observed_at=at(3),
                producer=object_provenance(),
            )

    def test_confidence_is_never_fabricated_upward(self, builder) -> None:
        """M11 stamps what it was given; it does not round a claim up."""
        result = understanding(attributes=(attribute(POSTURE, "standing", confidence=0.31),))
        published = builder.build_attribute(make_object(), result, context())
        assert published[0].attributes[0].confidence.value == pytest.approx(0.31)


class TestEnvelopeFailuresRejectEntirely:
    """§M11: *"reject the observation entirely and alarm."*"""

    def test_a_taxonomy_mismatch_rejects_the_observation(self, builder) -> None:
        """A producer on a different taxonomy version is asserting about a
        vocabulary the site does not have. Keeping the record would mean storing
        a fact whose class means something else here.
        """
        with pytest.raises(ValidationFailedError):
            builder.build_presence(
                make_object(), context(taxonomy_version="taxonomy-99")
            )

    def test_an_unregistered_class_rejects_the_observation(self, builder) -> None:
        """02_VOM: the class ontology is closed. A class nobody declared cannot
        be reasoned about, replayed, or removed later.
        """
        with pytest.raises(ValidationFailedError):
            builder.build_presence(make_object(class_id=ClassId("drone")), context())

    def test_a_rejected_observation_is_never_returned(self, builder) -> None:
        """It raises rather than returning ``None``.

        ``None`` means *suppressed* — a success. Conflating the two would let a
        caller treat a constitutional failure as a routine no-op.
        """
        with pytest.raises(ValidationFailedError):
            builder.build_presence(make_object(class_id=ClassId("drone")), context())
        assert builder.rejected >= 1


class TestSustainedDriftAlarms:
    """§M11: *"count, alarm on sustained rate."*"""

    def test_one_unregistered_attribute_does_not_alarm(self, builder, bus) -> None:
        subscription = bus.subscribe(["synthesis.schema_violation_spike"])
        result = understanding(attributes=(attribute(UNREGISTERED, True),))
        builder.build_attribute(make_object(), result, context())
        assert not subscription.drain()

    def test_a_sustained_rate_alarms(
        self, clock, metrics, bus, attribute_registry, taxonomy
    ) -> None:
        """A drifted producer — a new prompt, a new model, a partial deploy — is a
        deploy-time problem surfacing at publication time, and it must be loud.
        """
        from vision_os.adapters.synthesis import AlwaysPublish
        from vision_os.synthesis import CeilingGate, ObservationBuilder

        from .conftest import builder_provenance

        builder = ObservationBuilder(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=synthesis_config(
                suppression_policy="suppression.always",
                rejection_window=4,
                rejection_alarm_rate=0.5,
            ),
            gate=CeilingGate(attribute_registry, taxonomy),
            provenance=builder_provenance(),
            suppression_policy=AlwaysPublish(),
        )
        subscription = bus.subscribe(["synthesis.schema_violation_spike"])

        for i in range(6):
            result = understanding(
                object_id=f"obj-{i}",
                attributes=(attribute(UNREGISTERED, True),),
            )
            builder.build_attribute(make_object(object_id=f"obj-{i}"), result, context())

        events = subscription.drain()
        assert events, "a sustained violation rate must alarm"
        assert isinstance(events[0], SchemaViolationSpike)
        assert events[0].violation_rate >= 0.5


class TestTheGateIsPure:
    def test_validate_publishes_no_events_and_moves_no_counters(
        self, builder, bus
    ) -> None:
        """§M11's ``validate`` is exposed so a caller can check without publishing.

        A dry run that emitted events would make an operator's check
        indistinguishable from real traffic.
        """
        observation = presence_of(builder)
        subscription = bus.subscribe(["synthesis.observation_rejected"])
        before = builder.rejected

        broken = dataclasses.replace(observation, taxonomy_version="taxonomy-99")
        result = builder.validate(broken)

        assert result.rejected
        assert builder.rejected == before
        assert not subscription.drain()

    def test_validating_the_same_observation_twice_gives_the_same_answer(
        self, builder
    ) -> None:
        """V13. A gate whose answer drifts cannot be replayed."""
        observation = presence_of(builder)
        assert builder.validate(observation) == builder.validate(observation)


class TestTheCeilingHasNoBusinessVocabulary:
    def test_the_registry_refuses_an_interpretive_attribute(
        self, attribute_registry
    ) -> None:
        """The first gate, re-asserted here because M11 depends on it holding.

        ``is_authorized`` is not visible. No camera can see authorisation; it can
        only see a person and a place, and the conclusion belongs to a consumer.
        """
        from vision_os.perception.registry.attributes import (
            AttributeSchema,
            AttributeValueType,
        )

        with pytest.raises(AttributeRejectedError):
            attribute_registry.register(
                AttributeSchema(
                    key=AttributeKey("is_authorized"),
                    value_type=AttributeValueType.BOOL,
                    neutrality_justification="Person is allowed to be here",
                )
            )
