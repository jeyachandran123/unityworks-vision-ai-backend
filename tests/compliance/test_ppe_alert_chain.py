"""The §15 acceptance cases, against the shipped kitchen documents.

`test_ppe_uncertainty.py` already pins the head and hand cases (A, B and the
refusal cases). This file adds what the live-hardening review of 2026-08-31 asked
for and did not find covered:

* the **face/mask** rule (cases C and D), which arrived in policy 2.1.0 and had
  no acceptance test of its own;
* **case G** — an unavailable model must produce no PPE verdict at all, rather
  than a compliant one;
* cases **E and F**, the per-hand ones, which the shipped policy **cannot
  express**. That is recorded here as an executable limitation rather than a
  sentence in a report, because a limitation nobody can run is one that gets
  forgotten. See `TestPerHandStateIsNotRepresentable`.

Nothing here invents policy. Both documents are loaded from `config/`, as the
neighbouring suites do — a test asserting against a restated copy of a rule
proves only that the copy works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance import ComplianceEvaluator, ComplianceState, RuleSet, UnknownReason
from vision_os.core.model.api import CoverageSummary

from .conftest import NOW, attribute, subject

CONFIG = Path(__file__).resolve().parents[2] / "config"

KITCHEN = RuleSet.from_document(
    json.loads((CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8"))
)
POLICY = json.loads(
    (CONFIG / "policies" / "kitchen-safety.example.json").read_text(encoding="utf-8")
)

PPE_RULE = "kitchen.person.ppe.v1"
FACE_RULE = "kitchen.person.face_covering.v1"


def findings(**held: str):
    """One person through the real kitchen documents, keyed by rule id."""
    attributes = {key: attribute(key, value) for key, value in held.items()}
    produced = ComplianceEvaluator(KITCHEN).evaluate_object(
        subject(attributes=attributes), now=NOW, coverage=CoverageSummary()
    )
    return {f.rule_id: f for f in produced}


class TestCaseCAndD_Face:
    """visible face + mask PRESENT -> no violation; + mask ABSENT -> violation."""

    def test_case_c_a_worn_mask_is_compliant(self) -> None:
        assert findings(face_covering="mask")[FACE_RULE].state is ComplianceState.COMPLIANT

    def test_case_d_a_bare_face_is_a_violation(self) -> None:
        finding = findings(face_covering="none")[FACE_RULE]

        assert finding.state is ComplianceState.VIOLATION
        assert "is not wearing a mask over the nose and mouth" in finding.describe()

    def test_a_chin_mask_is_a_violation_and_says_so_truthfully(self) -> None:
        """`mask_below_nose` is a DECIDED state — the model saw a mask and saw
        where it sat. It fails the requirement without becoming 'we could not
        tell', which is the distinction the five-value domain exists to keep."""
        finding = findings(face_covering="mask_below_nose")[FACE_RULE]

        assert finding.state is ComplianceState.VIOLATION
        assert UnknownReason.NOT_OBSERVABLE not in finding.unknown_reasons

    def test_an_unreadable_face_is_unknown_not_a_violation(self) -> None:
        finding = findings(face_covering="not_visible")[FACE_RULE]

        assert finding.state is ComplianceState.UNKNOWN
        assert UnknownReason.NOT_OBSERVABLE in finding.unknown_reasons

    def test_a_profile_is_answerable_by_policy(self) -> None:
        """§7: a side-facing face must not be converted into NOT_VISIBLE. The
        policy says so in the question, and that wording is the only thing
        standing between a turned head and a lost mask violation."""
        entry = next(a for a in POLICY["attributes"] if a["key"] == "face_covering")
        assert "A face seen from the side IS visible" in entry["question"]

    def test_the_face_rule_stays_off_the_high_severity_rule(self) -> None:
        """Folding an unmeasured attribute into the validated PPE rule would let
        it raise HIGH-severity violations on evidence nobody has scored."""
        ppe = KITCHEN.get(PPE_RULE)
        assert [c.attribute for c in ppe.require] == ["head_covering", "hand_covering"]
        assert KITCHEN.get(FACE_RULE).severity == "informational"


class TestCaseG_ModelUnavailable:
    """An inference failure must produce NO verdict — never a compliant one."""

    def test_no_attribute_at_all_is_unknown(self) -> None:
        """What a 429 storm actually looks like downstream: the crop was cut,
        the call was made, the refusal produced nothing, so the registry holds
        no value and the rule has nothing to compare."""
        finding = findings()[PPE_RULE]

        assert finding.state is ComplianceState.UNKNOWN
        assert finding.state is not ComplianceState.COMPLIANT
        assert UnknownReason.ATTRIBUTE_ABSENT in finding.unknown_reasons

    def test_every_rule_refuses_together(self) -> None:
        produced = findings()
        assert {f.state for f in produced.values()} == {ComplianceState.UNKNOWN}

    def test_a_partial_answer_still_reports_what_it_knows(self) -> None:
        """The head answered before the quota ran out; the hand call refused.

        The head violation must survive. Suppressing it because the other half
        was unavailable would discard a fact the platform actually has.
        """
        finding = findings(head_covering="none")[PPE_RULE]

        assert finding.state is ComplianceState.VIOLATION
        assert [c.attribute_key for c in finding.failed_conditions] == ["head_covering"]
        assert [c.attribute_key for c in finding.unresolved_conditions] == ["hand_covering"]

    def test_the_unknown_reasons_stay_distinguishable(self) -> None:
        """'the model never answered' and 'the model looked and could not see'
        are different operator problems. NO_ANALYSIS must not read as
        REGION_NOT_VISIBLE (§17)."""
        never_ran = findings(head_covering="hairnet")[PPE_RULE].unknown_reasons
        looked = findings(head_covering="hairnet", hand_covering="not_visible")[
            PPE_RULE
        ].unknown_reasons

        assert never_ran != looked
        assert UnknownReason.ATTRIBUTE_ABSENT in never_ran
        assert UnknownReason.NOT_OBSERVABLE in looked


class TestPerHandStateIsNotRepresentable:
    """Cases E and F cannot be expressed by the shipped policy. **Recorded, not
    fixed.**

    §6 of the live-hardening brief requires ``LEFT_HAND`` and ``RIGHT_HAND`` to
    be evaluated independently, so that a bare left hand beside a gloved right
    hand is still a potential violation. The shipped policy declares a single
    ``hand_covering`` over one crop band, with the domain
    ``none | gloves | not_visible`` — there is no place to put a second hand.

    The consequence is a real, currently-unmeasured false-negative path: the
    question says *"Answer 'gloves' ONLY if you can actually see a covering on a
    hand"* — **a** hand, singular — so a person with one gloved and one bare
    hand can legitimately be answered ``gloves``, and the rule then reads
    COMPLIANT for both.

    Closing it means changing the attribute domain, the question, the crop
    geometry and the rule together, and re-measuring against annotated hands.
    That is a policy change, and this phase was explicitly scoped not to invent
    one. These tests therefore assert the limitation exactly as it is, so that
    it is visible in the suite and so that any future per-hand work has to come
    past them deliberately.
    """

    def test_the_policy_declares_one_hand_attribute(self) -> None:
        keys = [a["key"] for a in POLICY["attributes"]]
        assert keys == ["head_covering", "face_covering", "hand_covering"]
        assert "left_hand_covering" not in keys
        assert "right_hand_covering" not in keys

    def test_the_domain_cannot_name_a_side(self) -> None:
        entry = next(a for a in POLICY["attributes"] if a["key"] == "hand_covering")
        assert set(entry["values"]) == {"none", "gloves", "not_visible"}

    def test_the_rule_reads_the_single_attribute(self) -> None:
        conditions = [c.attribute for c in KITCHEN.get(PPE_RULE).require]
        assert "hand_covering" in conditions
        assert not any("left" in c or "right" in c for c in conditions)

    def test_case_e_one_bare_hand_cannot_be_distinguished(self) -> None:
        """The false negative, stated as a fact about today's system.

        There is exactly one value to record, so 'left bare, right gloved'
        reaches the rule as whichever single value the model chose. If that is
        ``gloves`` the finding is COMPLIANT — for a person with a bare hand.
        """
        assert findings(
            head_covering="hairnet", hand_covering="gloves"
        )[PPE_RULE].state is ComplianceState.COMPLIANT

    def test_case_f_one_unseen_hand_cannot_be_distinguished(self) -> None:
        """And the same shape in the other direction: 'left not visible, right
        gloved' is representable only as ``gloves`` or ``not_visible``. The
        second is the honest one and yields UNKNOWN, which is correct — but the
        policy cannot *require* it, so the choice rests entirely on the prompt's
        wording rather than on anything structural."""
        assert findings(
            head_covering="hairnet", hand_covering="not_visible"
        )[PPE_RULE].state is ComplianceState.UNKNOWN

    def test_the_question_still_refuses_rather_than_guesses(self) -> None:
        """The one guard that *is* in place, and the reason this limitation has
        not produced a flood of false compliance: the question spends most of
        its words on when to answer ``not_visible``."""
        entry = next(a for a in POLICY["attributes"] if a["key"] == "hand_covering")
        question = entry["question"]
        assert "not_visible" in question
        assert 'Answer "gloves" ONLY if' in question
        assert "A visible forearm, sleeve or cuff is NOT a visible hand" in question


class TestTheThreeAttributesAreAllAsked:
    @pytest.mark.parametrize(
        "key,rule",
        [("head_covering", PPE_RULE), ("hand_covering", PPE_RULE),
         ("face_covering", FACE_RULE)],
    )
    def test_each_ppe_attribute_reaches_a_rule(self, key: str, rule: str) -> None:
        """§13: a rule depending on an attribute nobody observes can never reach
        a verdict, and an attribute no rule reads can never raise one."""
        assert key in {a["key"] for a in POLICY["attributes"]}
        assert key in {c.attribute for c in KITCHEN.get(rule).require}
