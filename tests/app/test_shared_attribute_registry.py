"""M7 and M9 must share one AttributeRegistry instance. **Mandatory regression.**

The Semantic Ceiling is only canonical if there is exactly one of it. Phase 6.8
measured what happens when there is not: the composition root registered every
policy attribute into a registry handed only to Understanding, built the object
registry with a default empty one, and M7 then refused **308 of 308** attributes
M9 had successfully produced —

    AttributeRejectedError: attribute 'head_covering' is not registered

Nothing downstream could tell. The understanding layer reported zero failures,
the sink reported zero failures, and the platform silently re-asked the VLM for
an answer it already held, on every frame, forever. Freshness, staleness,
confidence refresh and quality refresh were unreachable for the whole of Phases
6.1 through 6.8 because of it.

Passing `attributes=` to `build_registry_layer` produced `FRESH_ENOUGH = 522` on
the very first run — the first time that skip reason had fired in the platform's
life.

These tests fail if the two layers are ever given separate registries again.
"""

from __future__ import annotations

import pytest

from app.configuration.settings import Settings
from app.vision.composition import (
    SharedRegistryViolation,
    assert_shared_attribute_registry,
    build_attribute_registry,
    load_policies,
    registry_of,
)
from app.vision.runtime import VisionRuntime

POLICIES = (
    "config/policies/kitchen-safety.example.json,"
    "config/policies/object-identity.example.json"
)


@pytest.fixture(scope="module")
def composition():
    settings = Settings(vision_semantic_policy=POLICIES)
    return VisionRuntime(settings).assemble()


class TestSharedRegistry:
    def test_m7_holds_an_attribute_registry_at_all(self, composition) -> None:
        """Without one, M7 has nothing to validate a write-back against."""
        assert registry_of(composition.registry_layer) is not None

    def test_m7_holds_the_canonical_instance(self, composition) -> None:
        """Identity, not equality.

        Two registries holding equal definitions compare equal and drift the
        moment one side reloads a policy — and the drift surfaces as an
        `AttributeRejectedError` for an attribute the operator can see declared
        in their own policy file.
        """
        assert registry_of(composition.registry_layer) is composition.attributes

    def test_the_assembly_assertion_rejects_a_second_registry(self) -> None:
        """The guard must actually fail when handed two registries.

        A guard that cannot fail is a comment.
        """
        policies = load_policies(POLICIES)
        canonical = build_attribute_registry(policies)
        impostor = build_attribute_registry(policies)

        assert canonical is not impostor

        class FakeLayer:
            def __init__(self, registry):
                self.attributes = registry

        with pytest.raises(SharedRegistryViolation):
            assert_shared_attribute_registry(FakeLayer(impostor), None, canonical)

    def test_the_assembly_assertion_rejects_a_layer_with_no_registry(self) -> None:
        class Bare:
            pass

        with pytest.raises(SharedRegistryViolation):
            assert_shared_attribute_registry(Bare(), None, build_attribute_registry(()))


class TestDeclaredAttributes:
    """M7 must accept what policy declared, and refuse what it did not."""

    def test_policy_attributes_reach_the_registry(self, composition) -> None:
        declared = set(composition.declared_attributes)
        assert {"head_covering", "hand_covering", "face_covering"} <= declared

    def test_m7_accepts_a_declared_attribute(self, composition) -> None:
        from vision_os.core.model.ids import AttributeKey

        schema = composition.attributes.require(AttributeKey("head_covering"))
        assert schema is not None
        assert "not_visible" in schema.domain, (
            "the refusal value must survive migration; without it a model that "
            "could not see a body part reports a decided state instead"
        )

    def test_m7_refuses_an_undeclared_attribute(self, composition) -> None:
        """An attribute nobody declared cannot be written back.

        This is the gate that made the shared-registry bug visible at all — and
        it must keep working, because it is also what stops a model inventing a
        vocabulary the policy never granted.
        """
        from vision_os.core.errors import AttributeRejectedError
        from vision_os.core.model.ids import AttributeKey

        with pytest.raises(AttributeRejectedError):
            composition.attributes.require(AttributeKey("wearing_a_hat_probably"))


class TestNeutralityGateSurvivedMigration:
    """The Semantic Ceiling is enforced, not merely documented."""

    @pytest.mark.parametrize(
        "key", ["is_compliant", "ppe_violation", "raise_alert", "violation_state"]
    )
    def test_a_verdict_shaped_attribute_is_refused(self, key: str) -> None:
        """A business verdict cannot become an observation.

        Vision OS reports what is visible. "The head covering is not visible" is
        an observation; "this person is non-compliant" is a judgment, and it
        belongs to `compliance/`, outside the platform.
        """
        from vision_os.perception.registry.attributes import (
            AttributeRegistry,
            AttributeSchema,
            AttributeValueType,
        )
        from vision_os.core.model.ids import AttributeKey

        registry = AttributeRegistry()
        with pytest.raises(Exception) as caught:
            registry.register(
                AttributeSchema(
                    key=AttributeKey(key),
                    value_type=AttributeValueType.ENUM,
                    neutrality_justification="directly visible in the crop",
                    applies_to=("person",),
                    domain=("yes", "no"),
                )
            )
        assert "neutral" in str(caught.value).lower() or "reject" in str(caught.value).lower()
