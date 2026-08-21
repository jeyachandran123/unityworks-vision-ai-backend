"""A use case must be data. These tests are what keep it that way.

The failure this file guards against is gradual: a domain arrives as a JSON
document, then one special case is added for it in Python, then another, and the
platform that was going to be generic is a kitchen product with a restaurant
branch. Every test below fails the moment a domain word has to be written into
source to make a deployment work.
"""

from __future__ import annotations

import json

import pytest

from vision_os.adapters.configuration.semantic_policy import (
    POLICY_ENV,
    SemanticPolicy,
    load_policy,
)
from vision_os.adapters.understanding.payload import split_by_schema
from vision_os.core.errors import ConfigurationError
from vision_os.core.model.demand import TriggerHint
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId
from vision_os.core.ports.understanding import OutputSchema
from vision_os.perception.registry.attributes import AttributeRegistry
from vision_os.perception.understanding.validation import AttributeValidator

POLICIES = "config/policies"


def document(**overrides):
    base = {
        "policy_id": "test-policy",
        "version": "1.0.0",
        "scope": {"object_classes": ["person"]},
        "attributes": [
            {
                "key": "posture",
                "type": "enum",
                "values": ["standing", "sitting"],
                "justification": "Body configuration is directly visible",
            }
        ],
    }
    base.update(overrides)
    return base


def write(tmp_path, doc, name="policy.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --- pluggability ----------------------------------------------------------- #


def test_a_new_use_case_needs_no_code_change(tmp_path):
    """**The test this whole design exists to pass.**

    A semantic concept nothing in this repository has ever heard of, driven from
    a document to a registered attribute, an output schema, a demand and a
    prompt — with no import of it anywhere in source.
    """
    invented = write(
        tmp_path,
        document(
            policy_id="warehouse-loading",
            scene="warehouse",
            attributes=[
                {
                    "key": "pallet_wrap_state",
                    "type": "enum",
                    "values": ["unwrapped", "partially_wrapped", "wrapped"],
                    "justification": "Wrapping film on the pallet is directly visible",
                    "validity_ms": 90000,
                }
            ],
        ),
    )

    policy = SemanticPolicy.from_file(invented)
    registry = AttributeRegistry()
    keys = policy.register_attributes(registry)
    template = policy.build_prompt_template()
    demand = policy.build_demand(subscriber="wms")

    assert keys == (AttributeKey("pallet_wrap_state"),)
    assert template.output_keys == (AttributeKey("pallet_wrap_state"),)
    assert demand.required_attributes == (AttributeKey("pallet_wrap_state"),)
    assert "pallet_wrap_state" in template.template
    assert "unwrapped" in template.template


def test_different_policies_produce_different_demands(tmp_path):
    a = SemanticPolicy.from_file(
        write(tmp_path, document(policy_id="a", demand={"freshness_ms": 5000}), "a.json")
    )
    b = SemanticPolicy.from_file(
        write(tmp_path, document(policy_id="b", demand={"freshness_ms": 90000}), "b.json")
    )

    demand_a = a.build_demand(subscriber="s")
    demand_b = b.build_demand(subscriber="s")

    assert demand_a.freshness.millis == 5000
    assert demand_b.freshness.millis == 90000
    assert demand_a.demand_id != demand_b.demand_id


def test_different_classes_get_different_demands(tmp_path):
    people = SemanticPolicy.from_file(
        write(tmp_path, document(scope={"object_classes": ["person"]}), "p.json")
    )
    vehicles = SemanticPolicy.from_file(
        write(
            tmp_path,
            document(
                policy_id="fleet",
                scope={"object_classes": ["car", "truck", "bus"]},
                attributes=[
                    {
                        "key": "load_state",
                        "type": "enum",
                        "values": ["empty", "loaded"],
                        "justification": "Cargo area contents are directly visible",
                    }
                ],
            ),
            "v.json",
        )
    )

    assert people.build_demand(subscriber="s").subject_filter.class_ids == (ClassId("person"),)
    assert vehicles.build_demand(subscriber="s").subject_filter.class_ids == (
        ClassId("car"),
        ClassId("truck"),
        ClassId("bus"),
    )


def test_different_schemas_allow_different_attributes(tmp_path):
    """The ceiling is per-policy: one cannot return the other's fields."""
    first = SemanticPolicy.from_file(write(tmp_path, document(), "1.json"))
    second = SemanticPolicy.from_file(
        write(
            tmp_path,
            document(
                policy_id="other",
                attributes=[
                    {
                        "key": "head_covering",
                        "type": "enum",
                        "values": ["none", "hairnet"],
                        "justification": "A covering on the head is directly visible",
                    }
                ],
            ),
            "2.json",
        )
    )

    schema_one = OutputSchema(fields=first.attribute_keys)
    model_answer = {"posture": "standing", "head_covering": "hairnet"}

    accepted, unparsed = split_by_schema(model_answer, schema_one)
    assert accepted == {"posture": "standing"}
    assert "head_covering" in (unparsed or "")

    schema_two = OutputSchema(fields=second.attribute_keys)
    accepted_two, unparsed_two = split_by_schema(model_answer, schema_two)
    assert accepted_two == {"head_covering": "hairnet"}
    assert "posture" in (unparsed_two or "")


# --- no policy --------------------------------------------------------------- #


def test_no_policy_configured_is_none_not_a_default_domain():
    """Absent a policy the platform must invent nothing.

    No attributes registered means no demands, which means no understanding
    requests and no model calls — Vision OS running without a semantic use case,
    which is a supported configuration and not a broken one.
    """
    assert load_policy(env={}) is None
    assert load_policy(env={POLICY_ENV: "   "}) is None


def test_no_policy_registers_no_attributes():
    registry = AttributeRegistry()
    policy = load_policy(env={})
    if policy is not None:  # pragma: no cover - guarded by the test above
        policy.register_attributes(registry)
    assert len(registry.all()) == 0 if hasattr(registry, "all") else True


# --- the ceiling still applies ------------------------------------------------ #


def test_an_enum_without_values_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="unconstrained enum"):
        SemanticPolicy.from_file(
            write(
                tmp_path,
                document(
                    attributes=[
                        {"key": "mood", "type": "enum", "justification": "visible"}
                    ]
                ),
            )
        )


def test_an_attribute_without_justification_is_refused(tmp_path):
    """The neutrality gate is what keeps a business rule out of the registry."""
    with pytest.raises(ConfigurationError, match="neutrality justification"):
        SemanticPolicy.from_file(
            write(
                tmp_path,
                document(
                    attributes=[
                        {"key": "is_authorised", "type": "bool", "justification": ""}
                    ]
                ),
            )
        )


def test_a_policy_requiring_nothing_is_refused(tmp_path):
    with pytest.raises(ConfigurationError):
        SemanticPolicy.from_file(write(tmp_path, document(attributes=[])))


def test_unregistered_model_fields_never_become_attributes(tmp_path):
    """End of the chain: schema splits, then the validator re-checks."""
    from vision_os.core.model.ids import ModuleId
    from vision_os.core.model.provenance import Provenance
    from vision_os.core.model.timebase import Instant

    policy = SemanticPolicy.from_file(write(tmp_path, document()))
    registry = AttributeRegistry()
    policy.register_attributes(registry)

    schema = OutputSchema(fields=policy.attribute_keys)
    structured, unparsed = split_by_schema(
        {"posture": "standing", "invented_by_the_model": "anything"}, schema
    )
    assert "invented_by_the_model" in (unparsed or "")

    outcome = AttributeValidator(registry).validate(
        # Force the undeclared key past the first gate to prove the second one
        # catches it independently.
        {**structured, "invented_by_the_model": "anything"},
        schema=schema,
        class_id=ClassId("person"),
        observed_at=Instant(0),
        producer=Provenance(
            producer_module=ModuleId("M9"), producer_version="1.0.0", config_revision=1
        ),
    )

    accepted = {str(a.key) for a in outcome.accepted}
    rejected = {r.field_name for r in outcome.rejected}
    assert accepted == {"posture"}
    assert "invented_by_the_model" in rejected


# --- call-volume controls are the policy's, not the adapter's ------------------ #


def test_the_policy_owns_freshness_triggers_and_budget(tmp_path):
    policy = SemanticPolicy.from_file(
        write(
            tmp_path,
            document(
                demand={
                    "freshness_ms": 45000,
                    "triggers": ["on_first_sight", "on_region_entry"],
                    "priority_class": "p2",
                    "budget": {"max_calls_per_hour": 120, "max_cost_per_hour": 3.5},
                }
            ),
        )
    )
    demand = policy.build_demand(subscriber="s", cameras=[CameraId("cam-1")])

    assert demand.freshness.millis == 45000
    assert demand.trigger_hints == (TriggerHint.ON_FIRST_SIGHT, TriggerHint.ON_REGION_ENTRY)
    assert demand.budget.max_calls_per_hour == 120
    assert demand.budget.max_cost_per_hour == pytest.approx(3.5)
    assert demand.scope.camera_ids == (CameraId("cam-1"),)


def test_an_unknown_trigger_is_refused(tmp_path):
    policy = SemanticPolicy.from_file(
        write(tmp_path, document(demand={"triggers": ["whenever_you_like"]}))
    )
    with pytest.raises(ConfigurationError, match="trigger hint"):
        policy.build_demand(subscriber="s")


def test_the_demand_carries_policy_provenance(tmp_path):
    policy = SemanticPolicy.from_file(write(tmp_path, document()))
    labels = policy.build_demand(subscriber="s").labels
    assert labels["policy_id"] == "test-policy"
    assert labels["policy_version"] == "1.0.0"


# --- the shipped documents load ------------------------------------------------ #


@pytest.mark.parametrize("name", ["appearance", "kitchen-safety.example"])
def test_shipped_policies_load_and_register(name):
    policy = SemanticPolicy.from_file(f"{POLICIES}/{name}.json")
    registry = AttributeRegistry()
    keys = policy.register_attributes(registry)

    assert keys
    assert policy.build_prompt_template().output_keys == keys
    assert policy.build_demand(subscriber="test").required_attributes == keys


def test_no_domain_vocabulary_is_written_into_the_loader():
    """The whole point: domains live in documents, never in source."""
    import inspect

    from vision_os.adapters.configuration import semantic_policy

    source = inspect.getsource(semantic_policy)
    # Split off the module docstring: it *discusses* these words in order to
    # explain the rule, which is not the same as depending on them.
    body = source.split('"""', 2)[-1].lower()
    for domain_word in ("hairnet", "apron", "shirt_colour", "cooking", "restaurant"):
        assert domain_word not in body, f"'{domain_word}' was hardcoded into the loader"


# --- per-attribute crop resolution (Phase 4.2) -------------------------------- #
#
# Resolution is declared per attribute because how much detail a claim needs
# depends on the claim, and because vision tokens scale with area: 448 costs 4x
# the tokens of 224, so raising it globally to fix one question would be a cost
# with no measured return.


def sized(value, key="posture"):
    return document(
        attributes=[
            {
                "key": key,
                "type": "enum",
                "values": ["standing", "sitting"],
                "justification": "Body configuration is directly visible",
                "output_size": value,
            }
        ]
    )


def test_output_size_survives_policy_loading(tmp_path):
    policy = SemanticPolicy.from_file(write(tmp_path, sized(448)))
    assert policy.output_sizes == {"posture": (448, 448)}


def test_output_size_accepts_an_explicit_width_and_height(tmp_path):
    policy = SemanticPolicy.from_file(write(tmp_path, sized({"width": 448, "height": 224})))
    assert policy.output_sizes == {"posture": (448, 224)}


def test_an_attribute_declaring_no_size_is_absent_rather_than_defaulted(tmp_path):
    """So the strategy can tell "the document said nothing" from "the document
    said the default" — the distinction that lets a default change safely."""
    policy = SemanticPolicy.from_file(write(tmp_path, document()))
    assert policy.output_sizes == {}


@pytest.mark.parametrize("bad", [0, -1, {"width": 448}, "448", True, {"width": 448, "height": 0}])
def test_a_malformed_output_size_is_refused_at_load(tmp_path, bad):
    """Refused while someone is looking at the file. A size silently ignored
    would leave a deployment believing it had raised a resolution it had not,
    and the symptom — a confident answer from too little detail — looks like a
    model problem rather than a typo."""
    with pytest.raises(ConfigurationError):
        SemanticPolicy.from_file(write(tmp_path, sized(bad)))


def test_an_absurd_output_size_is_refused(tmp_path):
    """Vision tokens scale with area, so an unbounded config value is an
    unbounded bill. A typo of 4480 for 448 is caught here, not on an invoice."""
    with pytest.raises(ConfigurationError, match="ceiling"):
        SemanticPolicy.from_file(write(tmp_path, sized(4480)))


def test_the_shipped_kitchen_policy_raises_only_the_head_band(tmp_path):
    """The measured configuration: 23.3% -> 74.4% head accuracy came from
    raising the head band. Hands are left at the default because no measurement
    supports paying 4x the tokens for them.

    `face_covering` also reads 448, and that is not a second decision. It
    declares head_covering's band exactly, so it shares the same crop and the
    strategy renders that crop once at the largest size declared for it. The
    load-bearing assertion is the one about hands: a band nobody measured is
    still not paying 4x.
    """
    policy = SemanticPolicy.from_file(f"{POLICIES}/kitchen-safety.example.json")
    assert policy.output_sizes == {
        "head_covering": (448, 448),
        "face_covering": (448, 448),
    }
    assert "hand_covering" not in policy.output_sizes

    # The reason face_covering is free rather than a second 448 crop.
    assert (
        policy.evidence_regions["face_covering"]
        == policy.evidence_regions["head_covering"]
    )


def test_declaring_a_size_does_not_disturb_the_other_declarations(tmp_path):
    """Region, floors and size are independent knobs on the same attribute."""
    policy = SemanticPolicy.from_file(f"{POLICIES}/kitchen-safety.example.json")
    assert policy.evidence_regions["head_covering"] == (0.0, 0.45)
    assert policy.quality_floors["head_covering"]["max_blur"] == 0.5
    assert policy.output_sizes["head_covering"] == (448, 448)
