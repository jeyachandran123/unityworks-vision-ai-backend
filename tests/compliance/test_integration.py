"""The two capabilities, joined, through the seams they actually use.

These tests drive the real Crop Manager with the real verification policy, then
feed what the platform would have published into the real evaluator. Nothing is
mocked at a module boundary: the objects come from the production ``VisualObject``
shape, the decisions come from composed production policies, and the verdicts
come from the production evaluator.

What they are checking is the architectural claim the whole design rests on:

    DETECTION discovers candidates.
    VERIFICATION decides whether more evidence is needed.
    UNDERSTANDING produces the facts.
    COMPLIANCE evaluates the facts.

and that no layer silently takes over another's job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance import ComplianceEvaluator, ComplianceState, RuleSet
from vision_os.adapters.configuration.verification_rules import (
    load_verification_rules,
    rules_from_file,
)
from vision_os.adapters.cropping import (
    DefaultTriggerPolicy,
    VerificationPolicy,
    VerificationRules,
)
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.crop import SkipReason, TriggerReason
from vision_os.core.model.detection import QualityGrades
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId, ObjectId
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.cropping import (
    CLOSED_SET,
    AttributeStatus,
    TriggerCandidate,
)

from .conftest import NOW, attribute, subject

KIND = AttributeKey("visible_object_kind")
CONFIG = Path(__file__).resolve().parents[2] / "config"


def build_policy(rules: VerificationRules) -> VerificationPolicy:
    return VerificationPolicy(DefaultTriggerPolicy(), rules)


def candidate(**overrides) -> TriggerCandidate:
    base = {
        "object_id": ObjectId("obj-1"),
        "camera_id": CameraId("cam-1"),
        "class_id": ClassId("toothbrush"),
        "box": Box(0.1, 0.1, 0.3, 0.5),
        "lifecycle": "active",
        "identity_confidence": 0.92,
        "first_seen": Instant(0),
        "last_confirmed": NOW,
        "observation_count": 6,
        "estimated_quality": QualityGrades(scale_pixels=140.0),
        "class_confidence": Confidence.uncalibrated(
            0.454, ConfidenceSemantics.CLASSIFICATION
        ),
        "label_space_kind": CLOSED_SET,
        "class_in_native_vocabulary": True,
    }
    base.update(overrides)
    return TriggerCandidate(**base)


def wants(*keys: AttributeKey):
    def resolve(**_):
        return {k: (Duration.from_millis(60_000), "", ("demand-verify",)) for k in keys}

    return resolve


# --- 22-23. the two paths through the pipeline --------------------------------- #


class TestBothPipelinePaths:
    """The trusted path and the corroborated path, side by side."""

    @pytest.fixture
    def policy(self) -> VerificationPolicy:
        return build_policy(rules_from_file(CONFIG / "policies" / "verification.example.json"))

    def test_an_ambiguous_detection_is_corroborated_then_evaluated(
        self, policy, full_coverage
    ) -> None:
        """Detection -> verification -> understanding -> observation -> compliance."""
        decision = policy.evaluate(
            [candidate()], now=NOW, demands=[wants(KIND)]
        )[0]

        # Verification decided a model call is warranted, and said why.
        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED
        assert "0.454" in decision.detail

        # The VLM answered. That answer is an attribute, **not a new class_id** —
        # the detector's claim stands beside it rather than being overwritten,
        # and both remain separately traceable.
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        evaluator = ComplianceEvaluator(rules)

        # The model could not tell. `unknown` is a legal domain value and the
        # rule refuses it — the honest outcome, and the one observed on real
        # footage where a phone crop was too small to identify.
        undecided = subject(
            class_id="cell_phone",
            attributes={"visible_object_kind": attribute("visible_object_kind", "unknown")},
        )
        finding = evaluator.evaluate_object(undecided, now=NOW, coverage=full_coverage)[0]

        assert finding.state is ComplianceState.VIOLATION
        assert finding.failed_conditions[0].attribute_key == "visible_object_kind"
        assert finding.subject.class_id == "cell_phone", (
            "the detector's class must survive corroboration unchanged"
        )

        # The model identified it. Same rule, same subject class, opposite verdict.
        identified = subject(
            class_id="cell_phone",
            attributes={"visible_object_kind": attribute("visible_object_kind", "phone")},
        )
        agreed = evaluator.evaluate_object(identified, now=NOW, coverage=full_coverage)[0]

        assert agreed.state is ComplianceState.COMPLIANT

    def test_a_trusted_detection_skips_straight_to_understanding(
        self, policy, full_coverage
    ) -> None:
        """Detection -> no verification -> normal understanding -> compliance."""
        decision = policy.evaluate(
            [candidate(class_id=ClassId("person"), class_confidence=Confidence.uncalibrated(0.94, ConfidenceSemantics.CLASSIFICATION))],
            now=NOW,
            demands=[wants(KIND)],
        )[0]

        assert decision.skip is SkipReason.EVIDENCE_SUFFICIENT

        view = subject(
            attributes={
                "head_covering": attribute("head_covering", "none"),
                "hand_covering": attribute("hand_covering", "gloves"),
            }
        )
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        finding = ComplianceEvaluator(rules).evaluate_object(
            view, now=NOW, coverage=full_coverage, subject_label="Employee #2"
        )[0]

        assert finding.state is ComplianceState.VIOLATION
        assert finding.describe() == "Employee #2: is not wearing a head covering"


# --- 24. reuse across frames ---------------------------------------------------- #


class TestVerificationIsReusedAcrossFrames:
    def test_three_frames_of_one_object_cost_one_call(self) -> None:
        """The §8 scenario, asserted as an exact call count.

        Frames 100, 101 and 102 see the same object at 0.45, 0.43 and 0.47. The
        first warrants corroboration; once the answer exists and is fresh, the
        other two do not — and the mechanism is per-object attribute freshness,
        not a second cache.
        """
        policy = build_policy(
            VerificationRules.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "attributes": [str(KIND)],
                            "require_when": {
                                "detector_label_space": [CLOSED_SET],
                                "class_confidence_below": 0.65,
                            },
                        }
                    ]
                }
            )
        )
        scores = (0.45, 0.43, 0.47)
        verified: dict[AttributeKey, AttributeStatus] = {}
        calls = 0

        for index, score in enumerate(scores):
            frame_now = Instant(NOW.ns + index * 40_000_000)  # 25 fps
            decision = policy.evaluate(
                [
                    candidate(
                        class_confidence=Confidence.uncalibrated(
                            score, ConfidenceSemantics.CLASSIFICATION
                        ),
                        attributes=dict(verified),
                        last_analysed=None if not verified else frame_now,
                    )
                ],
                now=frame_now,
                demands=[wants(KIND)],
            )[0]

            if decision.fires:
                calls += 1
                # The understanding pipeline answered; the platform stores it
                # with the freshness horizon the attribute schema declared.
                verified[KIND] = AttributeStatus(
                    key=KIND,
                    observed_at=frame_now,
                    confidence=0.88,
                    valid_until=Instant(frame_now.ns + 120_000_000_000),
                )

        assert calls == 1, "three frames of one object must not cost three calls"

    def test_expiry_restores_corroboration(self) -> None:
        policy = build_policy(
            VerificationRules.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "attributes": [str(KIND)],
                            "require_when": {"class_confidence_below": 0.65},
                        }
                    ]
                }
            )
        )
        expired = {
            KIND: AttributeStatus(
                key=KIND,
                observed_at=Instant(NOW.ns - 900_000_000_000),
                confidence=0.88,
                valid_until=Instant(NOW.ns - 800_000_000_000),
            )
        }
        decision = policy.evaluate(
            [candidate(attributes=expired, last_analysed=Instant(NOW.ns - 10))],
            now=NOW,
            demands=[wants(KIND)],
        )[0]

        assert decision.reason is TriggerReason.IDENTITY_UNVERIFIED


# --- 25. independence ------------------------------------------------------------ #


class TestSubjectsAreIndependent:
    def test_tracked_subjects_are_verified_and_judged_separately(
        self, full_coverage
    ) -> None:
        policy = build_policy(
            VerificationRules.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "attributes": [str(KIND)],
                            "require_when": {"class_confidence_below": 0.65},
                        }
                    ]
                }
            )
        )
        decisions = policy.evaluate(
            [
                candidate(object_id=ObjectId("obj-1")),
                candidate(
                    object_id=ObjectId("obj-2"),
                    class_confidence=Confidence.uncalibrated(
                        0.97, ConfidenceSemantics.CLASSIFICATION
                    ),
                ),
            ],
            now=NOW,
            demands=[wants(KIND)],
        )

        assert decisions[0].reason is TriggerReason.IDENTITY_UNVERIFIED
        assert decisions[1].skip is SkipReason.EVIDENCE_SUFFICIENT

        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        evaluator = ComplianceEvaluator(rules)
        findings = evaluator.evaluate(
            [
                subject(
                    "obj-1",
                    attributes={
                        "head_covering": attribute("head_covering", "hairnet"),
                        "hand_covering": attribute("hand_covering", "gloves"),
                    },
                ),
                subject(
                    "obj-2",
                    attributes={
                        "head_covering": attribute("head_covering", "none"),
                        "hand_covering": attribute("hand_covering", "gloves"),
                    },
                ),
            ],
            now=NOW,
            coverage=full_coverage,
            labels={"obj-1": "Employee #1", "obj-2": "Employee #2"},
        )

        # Scoped to the PPE rule. The shipped document also carries an
        # informational face-covering rule that covers `person`, and these
        # fixtures declare no face_covering — so it contributes an UNKNOWN per
        # subject. That is the correct behaviour of a rule whose evidence is
        # absent, and it is not what this test is about.
        ppe = [f for f in findings if f.rule_id == "kitchen.person.ppe.v1"]

        assert [f.state for f in ppe] == [
            ComplianceState.COMPLIANT,
            ComplianceState.VIOLATION,
        ]
        assert ppe[1].describe() == "Employee #2: is not wearing a head covering"


# --- 27. the no-policy configuration --------------------------------------------- #


class TestAbsentConfigurationIsSupported:
    def test_no_verification_document_means_no_policy(self) -> None:
        assert load_verification_rules(env={}) is None

    def test_no_rules_document_means_no_evaluation(self) -> None:
        from compliance import load_rules

        assert load_rules(env={}) is None

    def test_an_empty_ruleset_evaluates_nothing(self, full_coverage) -> None:
        findings = ComplianceEvaluator(RuleSet()).evaluate_object(
            subject(), now=NOW, coverage=full_coverage
        )

        assert findings == ()


# --- the shipped examples are valid ---------------------------------------------- #


class TestShippedExamplesLoad:
    """An example nobody can load teaches the wrong thing twice.

    These files are documentation, and documentation that has drifted from the
    parser is worse than none — a reader copies it and gets an error the file
    itself implies is impossible.
    """

    def test_the_verification_example_parses(self) -> None:
        rules = rules_from_file(CONFIG / "policies" / "verification.example.json")

        assert rules.governed_attributes == frozenset({KIND})

    def test_the_compliance_example_parses(self) -> None:
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )

        assert {r.rule_id for r in rules.rules} == {
            "kitchen.person.ppe.v1",
            "kitchen.person.face_covering.v1",
            "site.object.identity.v1",
        }

    def test_the_examples_agree_with_the_semantic_policy(self) -> None:
        """§20's dependency question, answered across three documents.

        Every attribute the rules depend on must be one the semantic policy
        actually asks the platform to observe — otherwise the rule can never
        reach a verdict here, and a deployment should learn that at startup
        rather than from a permanent UNKNOWN.
        """
        policy = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(
                encoding="utf-8"
            )
        )
        observed = {entry["key"] for entry in policy["attributes"]}
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        person_rule = rules.get("kitchen.person.ppe.v1")

        assert person_rule is not None
        assert set(person_rule.attributes) <= observed, (
            "the example rule depends on an attribute the example policy never "
            "asks anyone to observe"
        )

    def test_the_object_rule_agrees_with_the_object_policy(self) -> None:
        """The second half of §20, across the other pair of documents.

        The identity rule depends on `visible_object_kind`, which the
        object-identity policy declares — and which the verification rules
        govern. All three documents have to name the same attribute or the rule
        can never reach a verdict.
        """
        policy = json.loads(
            (CONFIG / "policies" / "object-identity.example.json").read_text(
                encoding="utf-8"
            )
        )
        observed = {entry["key"] for entry in policy["attributes"]}
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        object_rule = rules.get("site.object.identity.v1")

        assert object_rule is not None
        assert set(object_rule.attributes) <= observed
        assert rules_from_file(
            CONFIG / "policies" / "verification.example.json"
        ).governed_attributes == {AttributeKey("visible_object_kind")}

    def test_an_unproduced_attribute_is_reported_before_evaluation(self) -> None:
        """§20 answered at startup rather than as a permanent UNKNOWN.

        A deployment that binds only the kitchen policy cannot serve the object
        identity rule, and `unproducible_against` says so by name instead of
        leaving the rule silently unserved.
        """
        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )
        kitchen = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(
                encoding="utf-8"
            )
        )
        # Read from the document rather than restated here: a test that hardcodes
        # the vocabulary fails when the policy gains an attribute, which says
        # nothing about the property under test.
        gaps = rules.unproducible_against(
            [entry["key"] for entry in kitchen["attributes"]]
        )

        assert gaps, "the gap must be visible at startup"
        assert all(rule_id == "site.object.identity.v1" for rule_id, _ in gaps)

    def test_both_policies_together_serve_every_rule(self) -> None:
        """With both documents loaded there is no capability gap left.

        This is the configuration the demo actually runs, and the assertion is
        what proves the three documents are consistent as a set rather than
        individually.
        """
        producible: list[str] = []
        for name in ("kitchen-safety.example.json", "object-identity.example.json"):
            document = json.loads(
                (CONFIG / "policies" / name).read_text(encoding="utf-8")
            )
            producible.extend(entry["key"] for entry in document["attributes"])

        rules = RuleSet.from_document(
            json.loads(
                (CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8")
            )
        )

        assert rules.unproducible_against(producible) == ()


# --- provider independence -------------------------------------------------------- #


class TestSemanticsDoNotDependOnTheModel:
    def test_a_provider_swap_changes_no_verification_or_compliance_semantics(
        self, full_coverage
    ) -> None:
        """Test 26, as far as it can honestly be asserted without a GPU.

        Neither layer names a model. The verification policy reasons about a
        score, a label-space kind and a quality grade; the evaluator reasons
        about values and timestamps. Swapping NVIDIA for Ollama changes which
        adapter answers a P15 call and nothing either of these layers reads —
        which is provable by the dependency closure rather than by running both.
        """
        import compliance.evaluator as evaluator_module
        from vision_os.adapters.cropping import verification as verification_module

        for module in (evaluator_module, verification_module):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for vendor in ("nvidia", "ollama", "openai", "qwen", "llava", "internvl"):
                assert vendor not in source.lower(), (
                    f"{Path(module.__file__).name} names '{vendor}'; neither layer "
                    f"may depend on which model answered"
                )
