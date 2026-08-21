"""A refusal must never become a violation.

These are the §19 acceptance cases, written against the failure that produced
them. Reviewing real kitchen CCTV on 2026-08-13, five of five examined people
received a confident ``hand_covering`` value while their hands were inside a pot,
behind a body, or out of frame. Two became VIOLATIONs. Nothing was broken: the
model answered, the value was inside its registered domain, the rule compared it
correctly, and the finding was wrong.

The gap was that ``not_visible`` — the honest answer — is a legal enum member,
so ``hand_covering == "gloves"`` failed on it exactly as a bare hand would.
Absence of evidence became evidence of absence at the one line where the platform
had no way to tell them apart.

``unknown_values`` is that line. A rule declares which members of a domain are
refusals; the evaluator turns those into UNKNOWN before any comparison happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance import (
    ComplianceEvaluator,
    ComplianceState,
    RuleSet,
    UnknownReason,
)
from vision_os.core.model.api import CoverageSummary

from .conftest import NOW, attribute, subject

CONFIG = Path(__file__).resolve().parents[2] / "config"

#: The shipped kitchen rule, loaded rather than restated. A test asserting on a
#: copy of a document proves the copy works.
KITCHEN = RuleSet.from_document(
    json.loads((CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8"))
)


def verdict(head: str | None = None, hand: str | None = None):
    """One person through the real kitchen rule."""
    held = {}
    if head is not None:
        held["head_covering"] = attribute("head_covering", head)
    if hand is not None:
        held["hand_covering"] = attribute("hand_covering", hand)
    findings = ComplianceEvaluator(KITCHEN).evaluate_object(
        subject(attributes=held), now=NOW, coverage=CoverageSummary()
    )
    return next(f for f in findings if f.rule_id == "kitchen.person.ppe.v1")


class TestTheDecidedCases:
    """A and B, D and E — evidence present, verdict reached."""

    def test_covered_head_and_gloved_hands_is_compliant(self) -> None:
        assert verdict("hairnet", "gloves").state is ComplianceState.COMPLIANT

    def test_bare_head_is_a_violation(self) -> None:
        finding = verdict("none", "gloves")

        assert finding.state is ComplianceState.VIOLATION
        assert [c.attribute_key for c in finding.failed_conditions] == ["head_covering"]

    def test_bare_hands_are_a_violation(self) -> None:
        finding = verdict("hairnet", "none")

        assert finding.state is ComplianceState.VIOLATION
        assert [c.attribute_key for c in finding.failed_conditions] == ["hand_covering"]


class TestARefusalIsNeverAViolation:
    """C, F, G, H — the cases that produced the false findings."""

    def test_invisible_hands_do_not_become_bare_hands(self) -> None:
        """The exact 2026-08-13 failure, now locked.

        A chef with a hairnet and both hands inside a pot must be UNKNOWN. It was
        a VIOLATION, and the sentence read "is not wearing gloves" about hands
        nobody had seen.
        """
        finding = verdict("hairnet", "not_visible")

        assert finding.state is ComplianceState.UNKNOWN
        assert finding.state is not ComplianceState.VIOLATION
        assert UnknownReason.NOT_OBSERVABLE in finding.unknown_reasons
        assert finding.failed_conditions == ()

    def test_invisible_head_does_not_become_a_bare_head(self) -> None:
        finding = verdict("not_visible", "gloves")

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.NOT_OBSERVABLE in finding.unknown_reasons

    def test_neither_visible_is_unknown(self) -> None:
        finding = verdict("not_visible", "not_visible")

        assert finding.state is ComplianceState.UNKNOWN
        assert len(finding.unresolved_conditions) == 2

    def test_a_missing_attribute_is_still_unknown(self) -> None:
        """The other refusal: nothing was ever recorded, as opposed to recorded
        as unobservable. Different reason, same refusal to guess."""
        finding = verdict("hairnet", None)

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.ATTRIBUTE_ABSENT in finding.unknown_reasons

    def test_the_two_refusals_stay_distinguishable(self) -> None:
        """``NOT_OBSERVABLE`` and ``ATTRIBUTE_ABSENT`` mean different things to
        an operator: one is a camera angle, the other is a pipeline that never
        ran. Collapsing them would hide whichever is actually happening."""
        looked = verdict("hairnet", "not_visible").unknown_reasons
        never_asked = verdict("hairnet", None).unknown_reasons

        assert looked != never_asked


class TestPartialEvidenceReportsWhatItKnows:
    """§10's last case, and §11's requirement that partial evidence not spread."""

    def test_a_real_failure_survives_an_unknown_beside_it(self) -> None:
        """Bare head plus invisible hands is a VIOLATION **for the head**.

        The head failure is established and must be reported; the hand condition
        stays unresolved on the same finding rather than being swept into the
        verdict or suppressed by it.
        """
        finding = verdict("none", "not_visible")

        assert finding.state is ComplianceState.VIOLATION
        assert [c.attribute_key for c in finding.failed_conditions] == ["head_covering"]
        assert [c.attribute_key for c in finding.unresolved_conditions] == ["hand_covering"]
        assert "not wearing a head covering" in finding.describe()
        assert "gloves" not in finding.describe()

    def test_the_sentence_never_names_an_unresolved_condition(self) -> None:
        """A reviewer reading the finding must not see a claim about a body part
        the platform could not see."""
        assert "gloves" not in verdict("none", "not_visible").describe()


class TestTheShippedDocumentsAgree:
    def test_the_rule_declares_both_refusals(self) -> None:
        """A domain member that means "could not tell" is useless unless the rule
        that consumes it says so."""
        rule = KITCHEN.get("kitchen.person.ppe.v1")

        assert rule is not None
        for condition in rule.require:
            assert "not_visible" in condition.unknown_values, (
                f"'{condition.attribute}' can return not_visible and the rule does "
                f"not list it; it would be compared as though it were a fact"
            )

    def test_the_policy_offers_the_refusal_it_needs(self) -> None:
        """§8 requires head_covering to be able to answer UNKNOWN. Before 2.0.0
        its domain had no such member, so the model could not be honest even when
        it wanted to be."""
        policy = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(encoding="utf-8")
        )
        for entry in policy["attributes"]:
            assert "not_visible" in entry["values"], (
                f"'{entry['key']}' has no way to say the region was not visible"
            )

    def test_the_policy_scope_is_the_three_ppe_questions(self) -> None:
        """head, face and hands — in that order, and no others.

        `face_covering` arrived in policy 2.1.0. It is third question but not a
        third model call: it declares head_covering's band, so it rides that
        crop and costs one extra key in a response already being parsed.
        """
        policy = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(encoding="utf-8")
        )

        assert [e["key"] for e in policy["attributes"]] == [
            "head_covering",
            "face_covering",
            "hand_covering",
        ]

    @pytest.mark.parametrize(
        "key", ["head_covering", "face_covering", "hand_covering"]
    )
    def test_every_attribute_declares_where_it_is_visible(self, key: str) -> None:
        """The geometry the part-focused strategy reads. Data, not code."""
        policy = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(encoding="utf-8")
        )
        entry = next(e for e in policy["attributes"] if e["key"] == key)
        region = entry["evidence_region"]

        assert 0.0 <= region["top"] <= 1.0
        assert 0.0 < region["height"] <= 1.0
        assert region["top"] + region["height"] <= 1.0001
