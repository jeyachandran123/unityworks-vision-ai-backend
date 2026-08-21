"""The attribute neutrality gate — the Semantic Ceiling inside M7.

``14_TESTING`` section 6 names this the **registry gate**: *"Attempting to
register a judgment-bearing attribute is rejected."*

The table in 02_VOM section 9.1 is read back directly below, accepted and
rejected rows alike. The rejections matter more: a gate that admits everything
is a gate nobody notices is missing.

Note the pattern in every rejection and its repair — *the rejected attribute is
the accepted one plus a business premise*. The tests assert the rejection message
names that repair, because a gate that only says "no" gets removed by the first
engineer under deadline.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import AttributeRejectedError
from vision_os.core.model.ids import AttributeKey, ClassId
from vision_os.core.model.timebase import Duration
from vision_os.perception.registry.attributes import (
    AttributeRegistry,
    AttributeSchema,
    AttributeValueType,
    Cardinality,
    EvidenceRequirement,
    SchemaStatus,
    check_neutrality,
)


def schema(key: str, justification: str, **overrides) -> AttributeSchema:
    defaults = dict(
        key=AttributeKey(key),
        value_type=AttributeValueType.BOOL,
        neutrality_justification=justification,
    )
    defaults.update(overrides)
    return AttributeSchema(**defaults)


class TestAcceptedAttributes:
    """Every row 02_VOM section 9.1 marks registered."""

    @pytest.mark.parametrize(
        ("key", "justification"),
        [
            ("headwear_present", "Head region shows a covering"),
            ("posture", "Body configuration is directly visible"),
            ("carrying", "An object is visibly supported by the person"),
            (
                "hi_vis_present",
                "Torso region shows high-visibility colouring or retroreflection",
            ),
            ("queue_position", "Ordinal position along a region's principal axis"),
        ],
    )
    def test_a_visually_grounded_attribute_registers(
        self, key: str, justification: str
    ) -> None:
        check_neutrality(AttributeKey(key), justification)

    def test_the_repaired_counterparts_all_register(self) -> None:
        """Each rejection has a neutral counterpart the registry can offer."""
        for key, justification in (
            ("uniform_present", "Torso shows a garment matching a known pattern"),
            ("helmet_present", "Head region shows rigid protective covering"),
            ("dwell_duration", "Elapsed presence within a region, measured"),
        ):
            check_neutrality(AttributeKey(key), justification)


class TestRejectedAttributes:
    """Every row 02_VOM section 9.1 marks rejected."""

    def test_a_role_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="role"):
            check_neutrality(
                AttributeKey("is_employee"), "The uniform indicates employment"
            )

    def test_the_role_rejection_names_its_repair(self) -> None:
        with pytest.raises(AttributeRejectedError) as caught:
            check_neutrality(AttributeKey("is_employee"), "Uniform is visible")
        assert "uniform_present" in str(caught.value), (
            "a gate that only says 'no' is one the next engineer removes"
        )

    def test_a_policy_verdict_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="verdict"):
            check_neutrality(
                AttributeKey("is_compliant"), "A missing helmet is a safety breach"
            )

    def test_a_threshold_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="threshold"):
            check_neutrality(
                AttributeKey("wait_time_excessive"), "Dwell is longer than usual"
            )

    def test_the_threshold_rejection_offers_the_measurement(self) -> None:
        with pytest.raises(AttributeRejectedError) as caught:
            check_neutrality(AttributeKey("wait_time_excessive"), "Dwell is long")
        assert "dwell_duration" in str(caught.value)

    @pytest.mark.parametrize(
        "key",
        [
            "is_customer",
            "is_intruder",
            "is_suspicious",
            "is_unauthorized",
            "queueing_detected",
            "loitering_flag",
            "overcrowded",
            "shift_assigned",
        ],
    )
    def test_judgment_bearing_keys_are_rejected(self, key: str) -> None:
        with pytest.raises(AttributeRejectedError):
            check_neutrality(AttributeKey(key), "Something is visible in the frame")

    def test_a_bare_verdict_with_no_referent_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="visual referent"):
            check_neutrality(AttributeKey("is_ok"), "It looks fine in the image")


class TestJustificationIsTheGate:
    """``neutrality_justification`` is not documentation (02_VOM section 9.1)."""

    def test_an_empty_justification_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="required"):
            check_neutrality(AttributeKey("posture"), "")

    def test_a_trivial_justification_is_rejected(self) -> None:
        with pytest.raises(AttributeRejectedError, match="required"):
            check_neutrality(AttributeKey("posture"), "yes")

    @pytest.mark.parametrize(
        "justification",
        [
            "The uniform implies they work here",
            "It means the area is unsafe",
            "Policy says this must be worn",
            "They should not be standing there",
        ],
    )
    def test_a_justification_appealing_to_a_premise_is_rejected(
        self, justification: str
    ) -> None:
        """A justification must say what the pixels show, not what it implies."""
        with pytest.raises(AttributeRejectedError, match="visible evidence"):
            check_neutrality(AttributeKey("headwear_present"), justification)

    def test_a_justification_naming_pixels_is_accepted(self) -> None:
        check_neutrality(
            AttributeKey("headwear_present"),
            "The head region shows a covering distinct from hair",
        )


class TestSchemaValidation:
    def test_a_valid_schema_constructs(self) -> None:
        assert schema("posture", "Body configuration is directly visible")

    def test_a_keyless_schema_is_refused(self) -> None:
        with pytest.raises(ValueError, match="key"):
            AttributeSchema(
                key=AttributeKey(""),
                value_type=AttributeValueType.BOOL,
                neutrality_justification="Something is visible",
            )

    def test_an_enum_without_a_domain_is_refused(self) -> None:
        """An unconstrained enum is a text field wearing a type."""
        with pytest.raises(ValueError, match="domain"):
            schema(
                "posture",
                "Body configuration is directly visible",
                value_type=AttributeValueType.ENUM,
            )

    def test_an_enum_with_a_domain_is_accepted(self) -> None:
        assert schema(
            "posture",
            "Body configuration is directly visible",
            value_type=AttributeValueType.ENUM,
            domain=("standing", "sitting", "lying", "crouching"),
        )

    def test_the_value_types_are_the_documented_seven(self) -> None:
        assert {t.value for t in AttributeValueType} == {
            "enum", "bool", "scalar", "vector", "text", "relation", "count",
        }

    def test_evidence_requirements_are_the_documented_three(self) -> None:
        assert {e.value for e in EvidenceRequirement} == {"crop", "frame", "sequence"}

    def test_cardinality_is_single_or_multi(self) -> None:
        assert {c.value for c in Cardinality} == {"single", "multi"}


class TestRegistry:
    @pytest.fixture
    def registry(self) -> AttributeRegistry:
        return AttributeRegistry()

    def test_registering_admits_the_attribute(self, registry) -> None:
        registry.register(schema("posture", "Body configuration is directly visible"))
        assert AttributeKey("posture") in registry
        assert len(registry) == 1

    def test_registering_a_judgment_is_refused(self, registry) -> None:
        with pytest.raises(AttributeRejectedError):
            registry.register(schema("is_employee", "The uniform indicates a job"))
        assert len(registry) == 0, "a rejected attribute must not be admitted"

    def test_requiring_an_unregistered_attribute_is_refused(self, registry) -> None:
        """The gate is worthless if a producer can bypass it by not asking."""
        with pytest.raises(AttributeRejectedError, match="not registered"):
            registry.require(AttributeKey("posture"))

    def test_a_deprecated_attribute_is_refused(self, registry) -> None:
        registry.register(
            schema(
                "posture",
                "Body configuration is directly visible",
                status=SchemaStatus.DEPRECATED,
            )
        )
        with pytest.raises(AttributeRejectedError, match="deprecated"):
            registry.require(AttributeKey("posture"))

    def test_applies_to_restricts_by_class(self, registry) -> None:
        registry.register(
            schema(
                "headwear_present",
                "Head region shows a covering",
                applies_to=(ClassId("person"),),
            )
        )
        assert registry.applies(AttributeKey("headwear_present"), ClassId("person"))
        assert not registry.applies(
            AttributeKey("headwear_present"), ClassId("vehicle")
        )

    def test_applies_to_is_hierarchical(self, registry) -> None:
        registry.register(
            schema(
                "headwear_present",
                "Head region shows a covering",
                applies_to=(ClassId("person"),),
            )
        )
        assert registry.applies(
            AttributeKey("headwear_present"), ClassId("person.child")
        )

    def test_an_empty_applies_to_means_any_class(self, registry) -> None:
        """A deliberate degenerate case: some attributes are class-independent."""
        registry.register(schema("truncation", "Fraction of the object off-frame"))
        assert registry.applies(AttributeKey("truncation"), ClassId("anything"))

    def test_an_unregistered_attribute_applies_to_nothing(self, registry) -> None:
        assert not registry.applies(AttributeKey("posture"), ClassId("person"))

    def test_a_validity_horizon_is_carried(self, registry) -> None:
        registered = registry.register(
            schema(
                "posture",
                "Body configuration is directly visible",
                validity=Duration.from_millis(30_000),
            )
        )
        assert registered.validity is not None
        assert registered.validity.millis == pytest.approx(30_000)

    def test_re_registering_replaces_the_schema(self, registry) -> None:
        registry.register(schema("posture", "Body configuration is directly visible"))
        registry.register(
            schema("posture", "Body configuration is directly visible", version="2.0.0")
        )
        assert len(registry) == 1
        assert registry.get(AttributeKey("posture")).version == "2.0.0"
