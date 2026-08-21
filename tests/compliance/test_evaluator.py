"""Three-valued rule evaluation — the tests that keep bad evidence out of a verdict.

The most important tests in this file are the ones about ``UNKNOWN``. A rule
engine that turns missing evidence into ``false`` manufactures violations out of
blind spots, and one that turns it into ``true`` hides real ones. Both failures
are silent, and both are what the architecture principle names:

    WRONG OBSERVATION -> WRONG COMPLIANCE -> WRONG BUSINESS DECISION
"""

from __future__ import annotations

import pytest

from compliance import (
    ComplianceEvaluator,
    ComplianceState,
    RuleDocumentError,
    RuleSet,
    UnknownReason,
)
from vision_os.core.model.api import CoverageSummary
from vision_os.core.model.timebase import Instant

from .conftest import NOW, RECENT, attribute, subject


def only(findings) -> object:
    assert len(findings) == 1, f"expected one finding, got {len(findings)}"
    return findings[0]


# --- 13-15. the decided verdicts --------------------------------------------- #


class TestDecidedVerdicts:
    def test_all_conditions_true_is_compliant(
        self, evaluator, compliant_subject, full_coverage
    ) -> None:
        finding = only(
            evaluator.evaluate_object(
                compliant_subject, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.COMPLIANT
        assert finding.failed_conditions == ()
        assert len(finding.satisfied_conditions) == 3

    def test_one_false_condition_is_a_violation(self, evaluator, full_coverage) -> None:
        view = subject(
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.state is ComplianceState.VIOLATION
        assert len(finding.failed_conditions) == 1
        assert finding.failed_conditions[0].attribute_key == "headwear_present"

    def test_every_failure_is_recorded_not_just_the_first(
        self, evaluator, full_coverage
    ) -> None:
        """No short-circuit. A subject failing three requirements reports three."""
        view = subject(
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", False),
                "face_covering_present": attribute("face_covering_present", False),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.state is ComplianceState.VIOLATION
        assert {c.attribute_key for c in finding.failed_conditions} == {
            "headwear_present",
            "gloves_present",
            "face_covering_present",
        }

    def test_a_failure_outranks_an_unknown(self, evaluator, full_coverage) -> None:
        """An observed failure is a violation even beside unmeasurable evidence.

        The reverse precedence would let one blind spot mask every real failure
        next to it.
        """
        view = subject(
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.state is ComplianceState.VIOLATION
        assert finding.unresolved_conditions  # the mask was never established


# --- 16-17. the undecidable ones ---------------------------------------------- #


class TestUnknownIsNeverCoerced:
    def test_a_missing_attribute_is_unknown_not_false(
        self, evaluator, full_coverage
    ) -> None:
        """The single most dangerous coercion in the system."""
        view = subject(
            attributes={
                "headwear_present": attribute("headwear_present", True),
                "gloves_present": attribute("gloves_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.state is ComplianceState.UNKNOWN
        assert finding.state is not ComplianceState.VIOLATION
        assert UnknownReason.ATTRIBUTE_ABSENT in finding.unknown_reasons

    def test_an_expired_value_is_unknown(self, evaluator, full_coverage) -> None:
        """The platform stopped vouching for it; the rule stops relying on it."""
        view = subject(
            attributes={
                "headwear_present": attribute(
                    "headwear_present",
                    True,
                    observed_at=Instant(1_000_000_000),
                    valid_until=Instant(2_000_000_000),
                ),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.ATTRIBUTE_STALE in finding.unknown_reasons

    def test_a_rule_may_insist_on_something_fresher_than_the_platform_does(
        self, full_coverage
    ) -> None:
        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "r",
                        "version": "1.0.0",
                        "require": [
                            {"attribute": "headwear_present", "operator": "eq", "value": True}
                        ],
                        "evidence": {"max_staleness_ms": 100},
                    }
                ],
            }
        )
        view = subject(attributes={"headwear_present": attribute("headwear_present", True)})
        finding = only(
            ComplianceEvaluator(rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.ATTRIBUTE_STALE in finding.unknown_reasons

    def test_uncorroborated_evidence_is_unknown_when_the_rule_requires_it(
        self, full_coverage
    ) -> None:
        """§15's protected evaluation.

        The rule declared that its conclusion needs corroboration. Without it the
        evaluator declines rather than trusting a claim the rule itself called
        insufficient — and it does **not** go and fetch one.
        """
        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "r",
                        "version": "1.0.0",
                        "require": [
                            {"attribute": "surface_category", "operator": "eq", "value": "type_b"}
                        ],
                        "evidence": {"requires": ["visible_object_kind"]},
                    }
                ],
            }
        )
        view = subject(
            class_id="container",
            attributes={"surface_category": attribute("surface_category", "type_b")},
        )
        finding = only(
            ComplianceEvaluator(rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.EVIDENCE_UNVERIFIED in finding.unknown_reasons

    def test_corroborated_evidence_reaches_a_verdict(self, full_coverage) -> None:
        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "r",
                        "version": "1.0.0",
                        "require": [
                            {"attribute": "surface_category", "operator": "eq", "value": "type_b"}
                        ],
                        "evidence": {"requires": ["visible_object_kind"]},
                    }
                ],
            }
        )
        view = subject(
            class_id="container",
            attributes={
                "surface_category": attribute("surface_category", "type_a"),
                "visible_object_kind": attribute("visible_object_kind", "tray"),
            },
        )
        finding = only(
            ComplianceEvaluator(rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.VIOLATION

    def test_a_capability_gap_is_distinguishable_from_a_missing_value(
        self, evaluator, full_coverage
    ) -> None:
        """Waiting will not help. The operator response differs from every other
        unknown reason, so the reason must differ too."""
        finding = only(
            evaluator.evaluate_object(
                subject(),
                now=NOW,
                coverage=full_coverage,
                capability_gaps=("headwear_present",),
            )
        )

        assert UnknownReason.CAPABILITY_GAP in finding.unknown_reasons

    def test_a_coverage_gap_blocks_a_rule_that_asked_it_to(self) -> None:
        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "r",
                        "version": "1.0.0",
                        "require": [
                            {"attribute": "headwear_present", "operator": "eq", "value": True}
                        ],
                        "evidence": {"require_full_coverage": True},
                    }
                ],
            }
        )
        view = subject(attributes={"headwear_present": attribute("headwear_present", True)})
        finding = only(
            ComplianceEvaluator(rules).evaluate_object(
                view, now=NOW, coverage=CoverageSummary(observable_fraction=0.4)
            )
        )

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.COVERAGE_GAP in finding.unknown_reasons

    def test_coverage_travels_on_every_finding_not_only_uncertain_ones(
        self, evaluator, compliant_subject
    ) -> None:
        """A compliant verdict under 40% coverage is a different claim from one
        under full coverage, and a reviewer cannot tell without the number."""
        finding = only(
            evaluator.evaluate_object(
                compliant_subject,
                now=NOW,
                coverage=CoverageSummary(observable_fraction=0.4),
            )
        )

        assert finding.state is ComplianceState.COMPLIANT
        assert finding.coverage_fraction == 0.4

    def test_a_type_mismatch_is_unknown_not_a_violation(self, full_coverage) -> None:
        """A broken rule must not be presented as a real violation."""
        rules = RuleSet.from_document(
            {
                "version": "1",
                "rules": [
                    {
                        "rule_id": "r",
                        "version": "1.0.0",
                        "require": [
                            {"attribute": "headwear_present", "operator": "gt", "value": 3}
                        ],
                    }
                ],
            }
        )
        view = subject(attributes={"headwear_present": attribute("headwear_present", True)})
        finding = only(
            ComplianceEvaluator(rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.VALUE_UNPARSEABLE in finding.unknown_reasons


# --- the conditional shape ---------------------------------------------------- #


class TestGuardedRules:
    def test_a_matching_guard_reaches_the_requirement(
        self, surface_rules, full_coverage
    ) -> None:
        view = subject(
            class_id="container",
            attributes={
                "contents_category": attribute("contents_category", "type_b"),
                "surface_category": attribute("surface_category", "type_a"),
            },
        )
        finding = only(
            ComplianceEvaluator(surface_rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.VIOLATION
        assert finding.failed_conditions[0].attribute_key == "surface_category"

    def test_a_non_matching_guard_is_not_applicable_not_compliant(
        self, surface_rules, full_coverage
    ) -> None:
        """Reporting it compliant would inflate a pass rate with subjects the
        rule never examined."""
        view = subject(
            class_id="container",
            attributes={
                "contents_category": attribute("contents_category", "type_a"),
                "surface_category": attribute("surface_category", "type_a"),
            },
        )
        finding = only(
            ComplianceEvaluator(surface_rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.NOT_APPLICABLE
        assert finding.state is not ComplianceState.COMPLIANT

    def test_an_unknown_guard_is_unknown(self, surface_rules, full_coverage) -> None:
        view = subject(
            class_id="container",
            attributes={"surface_category": attribute("surface_category", "type_a")},
        )
        finding = only(
            ComplianceEvaluator(surface_rules).evaluate_object(
                view, now=NOW, coverage=full_coverage
            )
        )

        assert finding.state is ComplianceState.UNKNOWN


# --- 20-21. determinism and traceability -------------------------------------- #


class TestDeterminismAndTraceability:
    def test_the_same_inputs_produce_an_identical_finding(
        self, evaluator, compliant_subject, full_coverage
    ) -> None:
        first = evaluator.evaluate_object(
            compliant_subject, now=NOW, coverage=full_coverage
        )
        second = evaluator.evaluate_object(
            compliant_subject, now=NOW, coverage=full_coverage
        )

        assert first == second
        assert first[0].finding_id == second[0].finding_id

    def test_a_finding_names_its_rule_and_version(
        self, evaluator, compliant_subject, full_coverage
    ) -> None:
        finding = only(
            evaluator.evaluate_object(
                compliant_subject, now=NOW, coverage=full_coverage
            )
        )

        assert finding.pinned_rule == "site.subject.ppe.v1@1.0.0"
        assert finding.ruleset_version == "2026.1"

    def test_every_condition_carries_its_evidence_handle(
        self, evaluator, full_coverage
    ) -> None:
        """The first hop of the chain: finding -> condition -> evidence id."""
        view = subject(
            attributes={
                "headwear_present": attribute(
                    "headwear_present", False, evidence_ref="ev-headwear-7"
                ),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))

        assert finding.failed_conditions[0].evidence_ref == "ev-headwear-7"
        assert "ev-headwear-7" in finding.evidence_refs

    def test_a_condition_records_what_it_observed(self, evaluator, full_coverage) -> None:
        view = subject(
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            }
        )
        finding = only(evaluator.evaluate_object(view, now=NOW, coverage=full_coverage))
        failed = finding.failed_conditions[0]

        assert failed.expected is True
        assert failed.observed is False
        assert failed.observed_at == RECENT

    def test_a_violation_must_name_what_failed(self) -> None:
        """Enforced at construction: an unappealable verdict cannot be built."""
        from compliance.finding import Finding, SubjectRef

        with pytest.raises(ValueError, match="no failed condition"):
            Finding(
                finding_id="f",
                rule_id="r",
                rule_version="1.0.0",
                ruleset_version="1",
                state=ComplianceState.VIOLATION,
                subject=SubjectRef(object_id="o", class_id="person", camera_id="cam"),
                evaluated_at=NOW,
            )

    def test_multiple_subjects_are_evaluated_independently(
        self, evaluator, full_coverage
    ) -> None:
        """One subject's missing evidence must not change another's verdict."""
        good = subject(
            "obj-1",
            attributes={
                "headwear_present": attribute("headwear_present", True),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            },
        )
        bad = subject(
            "obj-2",
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            },
        )
        findings = evaluator.evaluate([good, bad], now=NOW, coverage=full_coverage)

        assert [f.state for f in findings] == [
            ComplianceState.COMPLIANT,
            ComplianceState.VIOLATION,
        ]


# --- presentation -------------------------------------------------------------- #


class TestPresentationIsAssembledNotGenerated:
    def test_the_end_user_sentence_comes_from_the_document(
        self, evaluator, full_coverage
    ) -> None:
        """§17's closing line, and it is rendered from structured data.

        Every word is from the rule document or from an id, so a stored finding
        regenerates the identical sentence six months later without asking a
        model anything.
        """
        view = subject(
            "obj-2",
            attributes={
                "headwear_present": attribute("headwear_present", False),
                "gloves_present": attribute("gloves_present", True),
                "face_covering_present": attribute("face_covering_present", True),
            },
        )
        finding = only(
            evaluator.evaluate_object(
                view, now=NOW, coverage=full_coverage, subject_label="Employee #2"
            )
        )

        assert finding.describe() == "Employee #2: is not wearing a hairnet"

    def test_an_unknown_says_why_rather_than_asserting_anything(
        self, evaluator, full_coverage
    ) -> None:
        finding = only(
            evaluator.evaluate_object(
                subject(), now=NOW, coverage=full_coverage, subject_label="Employee #4"
            )
        )

        assert "cannot be assessed" in finding.describe()
        assert "attribute_absent" in finding.describe()


# --- document validation -------------------------------------------------------- #


class TestDocumentValidation:
    def test_an_unversioned_rule_is_refused(self) -> None:
        with pytest.raises(RuleDocumentError, match="carries no version"):
            RuleSet.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "require": [{"attribute": "a", "operator": "eq", "value": 1}],
                        }
                    ]
                }
            )

    def test_a_rule_that_cannot_fail_is_refused(self) -> None:
        with pytest.raises(RuleDocumentError, match="requires nothing"):
            RuleSet.from_document(
                {"rules": [{"rule_id": "r", "version": "1.0.0", "require": []}]}
            )

    def test_a_duplicate_rule_id_is_refused(self) -> None:
        entry = {
            "rule_id": "r",
            "version": "1.0.0",
            "require": [{"attribute": "a", "operator": "eq", "value": 1}],
        }
        with pytest.raises(RuleDocumentError, match="declared twice"):
            RuleSet.from_document({"rules": [entry, entry]})

    def test_an_unknown_operator_is_refused(self) -> None:
        with pytest.raises(RuleDocumentError, match="not a supported operator"):
            RuleSet.from_document(
                {
                    "rules": [
                        {
                            "rule_id": "r",
                            "version": "1.0.0",
                            "require": [
                                {"attribute": "a", "operator": "approximately", "value": 1}
                            ],
                        }
                    ]
                }
            )

    def test_unproducible_attributes_are_reported_before_evaluation(
        self, ppe_rules
    ) -> None:
        """§20 answerable at startup, rather than as a permanent ``UNKNOWN``."""
        gaps = ppe_rules.unproducible_against(["headwear_present", "gloves_present"])

        assert gaps == (("site.subject.ppe.v1", "face_covering_present"),)

    def test_a_ruleset_declares_everything_it_depends_on(self, ppe_rules) -> None:
        assert set(ppe_rules.required_attributes) == {
            "headwear_present",
            "gloves_present",
            "face_covering_present",
        }
