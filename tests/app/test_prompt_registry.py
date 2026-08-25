"""Phase 6A.1: policy prompt material must reach the prompt provider.

The defect these tests lock down produced no error anywhere. Detection ran,
tracking ran, crops were taken and quality-gated, the real NVIDIA adapter was
bound and routing succeeded — and every request was refused with
`PROMPT_UNAVAILABLE` because the provider held nothing. On live CCTV that read
as 120 crops, 120 results, 120 failures, zero attributes, zero write-backs.

So the tests here assert three separable things:

1. templates declared by a policy are **registered**;
2. an attribute **nobody declared** still resolves to nothing — the honest
   refusal is a feature, not a gap to paper over;
3. the registration code is **generic** — a policy about vehicles works exactly
   as well as one about kitchens, with no code change.
"""

from __future__ import annotations

import json

import pytest

from app.vision.understanding import _prompt_templates
from vision_os.adapters.configuration.semantic_policy import SemanticPolicy
from vision_os.core.model.ids import AttributeKey, ClassId
from vision_os.understanding_bootstrap import build_prompt_provider

MODEL_FAMILY = "nvidia"


def policy_document(
    *,
    policy_id: str,
    classes: list[str],
    attributes: list[dict],
    prompt_id: str,
) -> dict:
    """A minimal, valid semantic policy. Deliberately domain-free."""
    return {
        "policy_id": policy_id,
        "version": "1.0.0",
        "scene": "test",
        "scope": {"object_classes": classes, "lifecycle": ["active"], "min_confidence": 0.4},
        "attributes": attributes,
        "demand": {"freshness_ms": 60_000, "triggers": ["on_change"]},
        "prompt": {
            "prompt_id": prompt_id,
            "version": "1.0.0",
            "max_output_tokens": 128,
            "preamble": "Report only what is plainly visible.",
        },
    }


def attribute(key: str, values: list[str], question: str) -> dict:
    return {
        "key": key,
        "type": "enum",
        "values": values,
        "justification": f"{key} is directly visible in the crop",
        "validity_ms": 120_000,
        "question": question,
    }


@pytest.fixture
def vehicle_policy(tmp_path) -> SemanticPolicy:
    """A policy about vehicles. Nothing in the product's domain.

    If any registration code were kitchen-aware, this fixture would expose it.
    """
    document = policy_document(
        policy_id="fleet.vehicle",
        classes=["car", "truck"],
        attributes=[
            attribute("vehicle_colour", ["red", "blue", "white", "not_visible"],
                      "State the dominant body colour, or not_visible."),
            attribute("load_secured", ["yes", "no", "not_visible"],
                      "State whether the load is strapped, or not_visible."),
        ],
        prompt_id="fleet.vehicle.body",
    )
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return SemanticPolicy.from_file(path)


@pytest.fixture
def kitchen_policy() -> SemanticPolicy:
    """The real shipped policy, loaded from the repository."""
    return SemanticPolicy.from_file("./config/policies/kitchen-safety.example.json")


class TestRegistration:
    def test_a_policy_declaring_a_prompt_produces_a_template(self, kitchen_policy):
        """The whole defect in one assertion: zero templates was the bug."""
        templates = _prompt_templates((kitchen_policy,))
        assert len(templates) == 1
        assert templates[0].output_keys, "a template with no output keys cannot be validated"

    def test_every_declared_attribute_appears_in_the_template(self, kitchen_policy):
        template = _prompt_templates((kitchen_policy,))[0]
        assert set(template.output_keys) == set(kitchen_policy.attribute_keys)

    def test_the_wording_comes_from_the_policy_not_from_code(self, kitchen_policy):
        """The template must be the policy's own text, not something generated."""
        template = _prompt_templates((kitchen_policy,))[0]
        assert kitchen_policy.prompt_preamble in template.template
        for requirement in kitchen_policy.requirements:
            question = getattr(requirement, "question", "")
            if question:
                assert question in template.template

    def test_several_policies_each_register(self, kitchen_policy, vehicle_policy):
        templates = _prompt_templates((kitchen_policy, vehicle_policy))
        assert {str(t.prompt_id) for t in templates} == {
            "kitchen-safety.person",
            "fleet.vehicle.body",
        }

    def test_a_policy_that_cannot_build_a_prompt_is_reported_not_fatal(self, kitchen_policy):
        """One broken policy must not take the other policies down with it."""

        class Broken:
            policy_id = "broken"

            def build_prompt_template(self):
                raise ValueError("declares an unregistered key")

        templates = _prompt_templates((Broken(), kitchen_policy))
        assert len(templates) == 1, "the healthy policy must still register"

    def test_an_object_with_no_prompt_support_is_skipped(self, kitchen_policy):
        templates = _prompt_templates((object(), kitchen_policy))
        assert len(templates) == 1


class TestLookup:
    def _provider(self, *policies):
        return build_prompt_provider(_prompt_templates(policies))

    def test_each_demanded_attribute_resolves(self, kitchen_policy):
        provider = self._provider(kitchen_policy)
        for key in kitchen_policy.attribute_keys:
            found = provider.resolve(
                (key,), class_id=ClassId("person"), model_family=MODEL_FAMILY
            )
            assert found is not None, f"{key} demanded but no prompt resolves"

    def test_one_prompt_covers_every_attribute_at_once(self, kitchen_policy):
        """The cost property. Three questions on one crop is one model call.

        If each attribute needed its own prompt, the same crop would be sent
        three times and the bill would triple for identical pixels.
        """
        provider = self._provider(kitchen_policy)
        found = provider.resolve(
            tuple(kitchen_policy.attribute_keys),
            class_id=ClassId("person"),
            model_family=MODEL_FAMILY,
        )
        assert found is not None

    def test_an_undeclared_attribute_resolves_to_nothing(self, kitchen_policy):
        """§9: no declared prompt means the model cannot be asked.

        The temptation when fixing an empty registry is to add a generic
        fallback. That would turn "nobody wrote this question down" into a
        confident answer about something nobody specified, which is worse than
        the refusal it replaces.
        """
        provider = self._provider(kitchen_policy)
        assert (
            provider.resolve(
                (AttributeKey("vehicle_colour"),),
                class_id=ClassId("person"),
                model_family=MODEL_FAMILY,
            )
            is None
        )

    def test_a_prompt_does_not_apply_to_a_class_it_never_named(self, kitchen_policy):
        provider = self._provider(kitchen_policy)
        assert (
            provider.resolve(
                (AttributeKey("head_covering"),),
                class_id=ClassId("chair"),
                model_family=MODEL_FAMILY,
            )
            is None
        )

    def test_a_partially_covered_request_does_not_resolve(self, kitchen_policy):
        """Half a prompt is not a prompt.

        Asking one model call to answer two attributes when the template only
        declares one would produce a result whose schema cannot be validated.
        """
        provider = self._provider(kitchen_policy)
        assert (
            provider.resolve(
                (AttributeKey("head_covering"), AttributeKey("vehicle_colour")),
                class_id=ClassId("person"),
                model_family=MODEL_FAMILY,
            )
            is None
        )


class TestGeneric:
    """§6: the registration path must know nothing about any domain."""

    def test_a_vehicle_policy_registers_with_no_code_change(self, vehicle_policy):
        provider = build_prompt_provider(_prompt_templates((vehicle_policy,)))
        found = provider.resolve(
            (AttributeKey("vehicle_colour"),),
            class_id=ClassId("car"),
            model_family=MODEL_FAMILY,
        )
        assert found is not None

    def test_the_registration_source_names_no_domain_word(self):
        """A grep, as a test. The cheapest guard against the easy mistake.

        Whole words only. An earlier version matched `ppe` inside `appears` and
        failed on its own prose — the sort of false alarm that gets a useful
        test deleted rather than fixed.
        """
        import re
        from pathlib import Path

        source = Path("app/vision/understanding.py").read_text(encoding="utf-8")
        body = source[source.index("def _prompt_templates") :]
        body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body

        forbidden = (
            "hairnet", "hairnets", "glove", "gloves", "mask", "masks", "chef",
            "kitchen", "ppe", "restaurant",
            "head_covering", "hand_covering", "face_covering",
        )
        found = sorted({w for w in forbidden if re.search(rf"\b{w}\b", body, re.IGNORECASE)})
        assert not found, f"domain vocabulary in generic registration code: {found}"
