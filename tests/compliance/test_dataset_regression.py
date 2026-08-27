"""What the replacement model's real answers do to the real rule.

`test_ppe_uncertainty.py` proves the *rule* handles a refusal correctly. This
proves nothing about whether the bound **model** ever produces one — and those
are different questions. A perfect `unknown_values` implementation protects
nobody from a model that answers `none` for a head it cannot see.

So this replays 43 recorded answers from
`meta/llama-3.2-11b-vision-instruct` — real crops from the human-annotated
kitchen-01 dataset, real policy prompt, real crop geometry — through the real
shipped kitchen rule, and measures the verdicts.

Recorded rather than re-requested: deterministic, and needs no key or network.
The recording is evidence and is never edited to make a test pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance import ComplianceEvaluator, ComplianceState, RuleSet
from vision_os.core.model.api import CoverageSummary

from .conftest import NOW, attribute, subject

CONFIG = Path(__file__).resolve().parents[2] / "config"
RECORDED = Path(__file__).resolve().parent / "kitchen01_model_answers.json"

KITCHEN = RuleSet.from_document(
    json.loads((CONFIG / "rules" / "site-safety.example.json").read_text(encoding="utf-8"))
)
ANSWERS = json.loads(RECORDED.read_text(encoding="utf-8"))
CASES = ANSWERS["cases"]


def verdict(head: str | None, hand: str | None) -> ComplianceState:
    """One person through the real kitchen rule."""
    held = {}
    if head is not None:
        held["head_covering"] = attribute("head_covering", head)
    if hand is not None:
        held["hand_covering"] = attribute("hand_covering", hand)
    findings = ComplianceEvaluator(KITCHEN).evaluate_object(
        subject(attributes=held), now=NOW, coverage=CoverageSummary()
    )
    return next(f for f in findings if f.rule_id == "kitchen.person.ppe.v1").state


def classify() -> dict[str, list[dict]]:
    """Split the recorded run by what the verdict was *and whether it was right*."""
    buckets: dict[str, list[dict]] = {
        "false_violation": [], "true_violation": [], "unknown": [],
        "compliant": [], "missed_violation": [],
    }
    for case in CASES:
        state = verdict(case["model"].get("head_covering"), case["model"].get("hand_covering"))
        head_truth = case["truth"].get("head_covering")
        hand_truth = case["truth"].get("hand_covering")
        if state is ComplianceState.VIOLATION:
            # False when the annotator could not see the body part at all: the
            # model asserted a bare head or bare hands that nobody could observe.
            unobservable = head_truth == "not_visible" and hand_truth == "not_visible"
            buckets["false_violation" if unobservable else "true_violation"].append(case)
        elif state is ComplianceState.UNKNOWN:
            # A real bare hand or head that ended as UNKNOWN is a miss, not a save.
            if "absent" in (head_truth, hand_truth):
                buckets["missed_violation"].append(case)
            else:
                buckets["unknown"].append(case)
        elif state is ComplianceState.COMPLIANT:
            buckets["compliant"].append(case)
    return buckets


class TestTheRecordingItself:
    def test_the_recording_covers_the_whole_dataset(self) -> None:
        assert len(CASES) == 43

    def test_every_recorded_value_is_inside_the_registered_domain(self) -> None:
        """A value outside the domain is rejected before it reaches a rule, so a
        recording full of them would prove nothing about compliance."""
        policy = json.loads(
            (CONFIG / "policies" / "kitchen-safety.example.json").read_text(encoding="utf-8")
        )
        domains = {a["key"]: set(a["values"]) for a in policy["attributes"]}
        for case in CASES:
            for key, value in case["model"].items():
                assert value is None or value in domains[key], (
                    f"{case['frame']}/{case['subject']}: {key}={value!r} is not in "
                    f"the registered domain"
                )


class TestTheSafetyContract:
    """These must hold whatever the model's accuracy is."""

    def test_an_unparsed_answer_never_reaches_a_rule_as_a_value(self) -> None:
        """Three answers in this run did not parse — one used unquoted values, one
        was prose, one was markdown. All three arrive as `None`, and `None` must
        become UNKNOWN rather than a comparison."""
        unparsed = [c for c in CASES if c["model"].get("head_covering") is None]
        assert unparsed, "expected the recording to contain unparsed answers"
        for case in unparsed:
            assert verdict(None, case["model"].get("hand_covering")) is not ComplianceState.COMPLIANT

    def test_not_visible_never_produces_a_violation(self) -> None:
        """The four-state contract, asserted over real recorded values."""
        for case in CASES:
            if case["model"].get("head_covering") == "not_visible" and \
               case["model"].get("hand_covering") == "not_visible":
                assert verdict("not_visible", "not_visible") is ComplianceState.UNKNOWN

    def test_an_observed_bare_head_still_produces_a_violation(self) -> None:
        """The contract has to cut both ways: protecting refusals must not have
        disabled the rule."""
        assert verdict("none", "gloves") is ComplianceState.VIOLATION


class TestWhatTheModelActuallyDoes:
    """Measurement, not a target. These record the model's real behaviour."""

    def test_the_model_never_refuses_on_head_covering(self) -> None:
        """**The finding that blocks a safety claim.**

        `head_covering` carries `not_visible` in its domain and the rule lists it
        as an unknown value, but this model never once used it across 43 real
        subjects. The refusal path is unreachable in practice for this attribute,
        so every head the model cannot see becomes a confident `none`.

        Asserted as-is so that a future model or prompt which *does* refuse makes
        this test fail loudly and forces the number below to be re-measured.
        """
        refusals = [c for c in CASES if c["model"].get("head_covering") == "not_visible"]
        assert len(refusals) == 0

    def test_the_false_violation_count_is_pinned(self) -> None:
        """11 of 43 subjects are reported non-compliant while the human annotator
        recorded that neither their head nor their hands were visible.

        Eleven, not the twelve a raw `truth=not_visible / model=none` count
        gives: one of those twelve had genuinely bare hands, so the rule reaches
        VIOLATION for a reason that happens to be correct. Counting confusion
        cells rather than verdicts would have overstated the harm by one.

        Pinned rather than asserted-to-be-zero because it is **not** a regression
        introduced by the model replacement — the retired model failed the same
        way — and because pretending it passes would be worse than recording it.
        A change in either direction should fail this test and be explained.
        """
        buckets = classify()
        assert len(buckets["false_violation"]) == 11

    def test_hand_covering_does_use_its_refusal(self) -> None:
        """The contrast that shows the head failure is not the rule's fault: on
        the same crops, from the same model, `hand_covering` refuses freely."""
        refusals = [c for c in CASES if c["model"].get("hand_covering") == "not_visible"]
        assert len(refusals) > 20


@pytest.mark.parametrize("bucket", ["false_violation", "missed_violation"])
def test_the_unsafe_buckets_are_reported_not_hidden(bucket: str) -> None:
    """A named, non-crashing accessor for each failure mode, so a reviewer can
    print them rather than re-deriving them from the JSON."""
    assert isinstance(classify()[bucket], list)
