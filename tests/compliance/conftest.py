"""Fixtures for the compliance suite.

Views are hand-built rather than driven through a running platform, and that is
the point: the evaluator is a pure function, so a test can hand it exactly the
facts it wants to reason about without booting eight layers. The integration
suite covers the other direction.

Every attribute here carries ``SELF_REPORTED`` confidence, because that is what a
VLM-produced attribute actually carries in this platform. A fixture using
``ATTRIBUTE`` semantics would be testing against a shape the pipeline does not
produce.
"""

from __future__ import annotations

import pytest

from compliance import ComplianceEvaluator, RuleSet
from vision_os.core.model.api import AttributeView, CoverageSummary, ObjectView
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId, ObjectId
from vision_os.core.model.timebase import Instant
from vision_os.core.model.visual_object import LifecycleState

NOW = Instant(10_000_000_000)
RECENT = Instant(9_000_000_000)


def attribute(
    key: str,
    value: object,
    *,
    observed_at: Instant = RECENT,
    valid_until: Instant | None = None,
    evidence_ref: str | None = None,
) -> AttributeView:
    return AttributeView(
        key=AttributeKey(key),
        value=value,
        confidence=Confidence(
            value=0.9, semantics=ConfidenceSemantics.SELF_REPORTED, raw_score=0.9
        ),
        observed_at=observed_at,
        valid_until=valid_until,
        evidence_ref=evidence_ref or f"evidence-{key}",
    )


def subject(
    object_id: str = "obj-1",
    *,
    class_id: str = "person",
    camera_id: str = "cam-1",
    attributes: dict[str, AttributeView] | None = None,
) -> ObjectView:
    return ObjectView(
        object_id=ObjectId(object_id),
        class_id=ClassId(class_id),
        class_confidence=Confidence(
            value=0.93, semantics=ConfidenceSemantics.IDENTITY, raw_score=0.93
        ),
        lifecycle=LifecycleState.ACTIVE,
        camera_id=CameraId(camera_id),
        first_seen=Instant(0),
        last_seen=NOW,
        last_confirmed=NOW,
        attributes={AttributeKey(k): v for k, v in (attributes or {}).items()},
        observation_count=12,
    )


PPE_DOCUMENT = {
    "version": "2026.1",
    "rules": [
        {
            "rule_id": "site.subject.ppe.v1",
            "version": "1.0.0",
            "severity": "high",
            "subject_classes": ["person"],
            "require": [
                {
                    "attribute": "headwear_present",
                    "operator": "eq",
                    "value": True,
                    "message": "is not wearing a hairnet",
                },
                {
                    "attribute": "gloves_present",
                    "operator": "eq",
                    "value": True,
                    "message": "is not wearing gloves",
                },
                {
                    "attribute": "face_covering_present",
                    "operator": "eq",
                    "value": True,
                    "message": "is not wearing a mask",
                },
            ],
        }
    ],
}

#: The conditional shape: a guard plus a requirement. A subject whose guard does
#: not hold is ``NOT_APPLICABLE``, never compliant.
SURFACE_DOCUMENT = {
    "version": "2026.1",
    "rules": [
        {
            "rule_id": "site.surface.pairing.v1",
            "version": "1.0.0",
            "severity": "critical",
            "subject_classes": ["container"],
            "when": [
                {"attribute": "contents_category", "operator": "eq", "value": "type_b"}
            ],
            "require": [
                {
                    "attribute": "surface_category",
                    "operator": "eq",
                    "value": "type_b",
                    "message": "is on a surface designated for a different category",
                }
            ],
        }
    ],
}


@pytest.fixture
def ppe_rules() -> RuleSet:
    return RuleSet.from_document(PPE_DOCUMENT)


@pytest.fixture
def surface_rules() -> RuleSet:
    return RuleSet.from_document(SURFACE_DOCUMENT)


@pytest.fixture
def evaluator(ppe_rules: RuleSet) -> ComplianceEvaluator:
    return ComplianceEvaluator(ppe_rules)


@pytest.fixture
def full_coverage() -> CoverageSummary:
    return CoverageSummary(observable_fraction=1.0, cameras_observing=1)


@pytest.fixture
def compliant_subject() -> ObjectView:
    return subject(
        attributes={
            "headwear_present": attribute("headwear_present", True),
            "gloves_present": attribute("gloves_present", True),
            "face_covering_present": attribute("face_covering_present", True),
        }
    )
