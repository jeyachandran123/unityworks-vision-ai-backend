"""One person, one alert, one sentence.

Rules 2026.2 read hairnet, face mask, apron and gloves as four conditions of
one PPE rule, so an alert names every item the person was seen without —
"Employee #2 is not wearing a hairnet, a face mask or gloves" — instead of
`describe()`'s "Employee #2: is not wearing a hairnet; is not wearing …".
"""

from __future__ import annotations

import json
from pathlib import Path

from app.vision.compliance_driver import _summary, join_items
from compliance import ComplianceEvaluator, RuleSet
from tests.compliance.conftest import NOW, attribute, subject
from vision_os.core.model.api import CoverageSummary

RULES = RuleSet.from_document(
    json.loads(
        (
            Path(__file__).resolve().parents[2] / "config" / "rules" / "site-safety.example.json"
        ).read_text(encoding="utf-8")
    )
)


def ppe_summary(**held: str) -> str:
    finding = next(
        f
        for f in ComplianceEvaluator(RULES).evaluate_object(
            subject(attributes={k: attribute(k, v) for k, v in held.items()}),
            now=NOW,
            coverage=CoverageSummary(),
            subject_label="Employee #2",
        )
        if f.rule_id == "kitchen.person.ppe.v1"
    )
    return _summary(finding)


def test_every_item_seen_missing_is_one_sentence() -> None:
    assert (
        ppe_summary(head_covering="none", face_covering="none", apron="none", hand_covering="none")
        == "Employee #2 is not wearing a hairnet, a face mask, an apron or gloves"
    )


def test_an_item_that_was_not_visible_is_not_said_to_be_missing() -> None:
    assert (
        ppe_summary(
            head_covering="none", face_covering="not_visible", apron="apron", hand_covering="none"
        )
        == "Employee #2 is not wearing a hairnet or gloves"
    )


def test_one_item_reads_as_a_plain_sentence() -> None:
    assert ppe_summary(face_covering="none") == "Employee #2 is not wearing a face mask"


def test_join_items() -> None:
    assert join_items([]) == ""
    assert join_items(["gloves"]) == "gloves"
    assert join_items(["a hairnet", "gloves"]) == "a hairnet or gloves"
