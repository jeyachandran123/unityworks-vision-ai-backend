"""Phase 8: the application chain from a compliance finding to an Incident.

The rule engine itself is `compliance/`, and it has its own suite — including
`test_ppe_uncertainty.py`, which owns the four-state semantics. **These tests
deliberately do not re-test that.** What had never existed anywhere is the step
after the verdict: a violation becoming a durable Incident, staying *one*
incident while it continues, and closing only on evidence.

The safety invariant is re-asserted here anyway, but at the incident layer
rather than the verdict layer: it is not enough that UNKNOWN is not a violation
— UNKNOWN must also never open an incident, and must never close one.
"""

from __future__ import annotations

import pytest

from app.vision.compliance_driver import RAISES_INCIDENTS, _finding_wire, _observed_at
from compliance import ComplianceState

RULE_DOCUMENT = {
    "version": "test-2026.1",
    "rules": [
        {
            "rule_id": "kitchen.person.ppe.v1",
            "version": "1.0.0",
            "severity": "high",
            "subject_classes": ["person"],
            "require": [
                {
                    "attribute": "head_covering",
                    "operator": "ne",
                    "value": "none",
                    "unknown_values": ["not_visible"],
                    "message": "is not wearing a head covering",
                }
            ],
        }
    ],
}


def _rules():
    from compliance import RuleSet

    return RuleSet.from_document(RULE_DOCUMENT)


def _finding(state: ComplianceState, *, object_id: str = "obj-1", camera: str = "cam-12"):
    """A finding shaped like the real one, without needing a live platform."""
    from compliance.finding import ConditionOutcome, Finding, SubjectRef, UnknownReason
    from vision_os.core.model.ids import CameraId, ObjectId
    from vision_os.core.model.timebase import Instant

    subject = SubjectRef(
        object_id=ObjectId(object_id), camera_id=CameraId(camera), class_id="person"
    )
    if state is ComplianceState.VIOLATION:
        condition = ConditionOutcome(
            attribute_key="head_covering", operator="ne", expected="none",
            observed="none", satisfied=False, message="is not wearing a head covering",
        )
    elif state is ComplianceState.COMPLIANT:
        condition = ConditionOutcome(
            attribute_key="head_covering", operator="ne", expected="none",
            observed="hairnet", satisfied=True,
        )
    else:
        condition = ConditionOutcome(
            attribute_key="head_covering", operator="ne", expected="none",
            observed="not_visible", unknown_reason=UnknownReason.NOT_OBSERVABLE,
        )

    return Finding(
        finding_id=f"f-{object_id}-{state.value}",
        rule_id="kitchen.person.ppe.v1",
        rule_version="1.0.0",
        ruleset_version="test-2026.1",
        state=state,
        subject=subject,
        evaluated_at=Instant(1_787_000_000_000_000_000),
        conditions=(condition,),
        severity="high",
    )


class _Settings:
    default_tenant_id = "org-test"


async def apply(app, findings, cameras=None):
    """Run the driver's real persistence step against a real database.

    `ComplianceDriver.apply` is the code under test — reading Vision State is a
    separate concern with its own failure modes, and stubbing it here keeps
    these tests about incidents rather than about the platform.
    """
    from app.vision.compliance_driver import ComplianceDriver

    driver = ComplianceDriver(
        settings=_Settings(),
        vision=None,
        database=app.state.database,
        rules=_rules(),
    )
    return await driver.apply(findings, cameras=cameras or {"cam-12": "rest-01"})


async def _incidents(app, organization_id: str = "org-test"):
    from app.domain.incidents import IncidentService

    async with app.state.database.session_scope() as session:
        rows = await IncidentService(session).list(organization_id=organization_id)
        return list(rows[0]) if isinstance(rows, tuple) else list(rows)


class TestViolationBecomesIncident:
    @pytest.mark.asyncio
    async def test_a_violation_opens_an_incident_that_persists(self, app):
        run = await apply(app, [_finding(ComplianceState.VIOLATION)])
        assert run.incidents_opened == 1

        rows = await _incidents(app)
        assert len(rows) == 1
        incident = rows[0]
        assert incident.rule_id == "kitchen.person.ppe.v1"
        assert incident.camera_key == "cam-12"
        assert incident.object_id == "obj-1"
        assert incident.severity == "high"
        assert incident.status == "active"
        assert incident.ruleset_version == "test-2026.1"
        assert incident.restaurant_id == "rest-01"
        # The reasoning is frozen with the incident, so a later rule change
        # cannot rewrite what was decided about this person.
        assert "head_covering" in incident.finding_snapshot

    @pytest.mark.asyncio
    async def test_a_continuing_violation_stays_one_incident(self, app):
        """§11. A chef stays hatless for minutes; that is one event, not 400."""
        first = await apply(app, [_finding(ComplianceState.VIOLATION)])
        assert (first.incidents_opened, first.incidents_updated) == (1, 0)

        for _ in range(5):
            again = await apply(app, [_finding(ComplianceState.VIOLATION)])
            assert (again.incidents_opened, again.incidents_updated) == (0, 1)

        assert len(await _incidents(app)) == 1

    @pytest.mark.asyncio
    async def test_two_people_get_their_own_incidents(self, app):
        """Deduplication is per subject, not per camera."""
        await apply(app, [
            _finding(ComplianceState.VIOLATION, object_id="obj-1"),
            _finding(ComplianceState.VIOLATION, object_id="obj-2"),
        ])
        assert len({i.object_id for i in await _incidents(app)}) == 2


class TestSafetyAtTheIncidentLayer:
    @pytest.mark.asyncio
    async def test_unknown_never_opens_an_incident(self, app):
        run = await apply(app, [_finding(ComplianceState.UNKNOWN)])
        assert (run.incidents_opened, run.incidents_resolved) == (0, 0)
        assert await _incidents(app) == []

    @pytest.mark.asyncio
    async def test_unknown_does_not_close_a_real_violation(self, app):
        """A person who walked out of frame has not put a hairnet on.

        Closing on "we can no longer see the violation" is how a safety system
        learns to lie.
        """
        await apply(app, [_finding(ComplianceState.VIOLATION)])
        run = await apply(app, [_finding(ComplianceState.UNKNOWN)])

        assert run.incidents_resolved == 0
        assert (await _incidents(app))[0].status == "active"


class TestResolution:
    @pytest.mark.asyncio
    async def test_a_later_compliant_observation_resolves_it(self, app):
        """§13. Closed by evidence, not by a timer."""
        await apply(app, [_finding(ComplianceState.VIOLATION)])
        run = await apply(app, [_finding(ComplianceState.COMPLIANT)])

        assert run.incidents_resolved == 1
        resolved = (await _incidents(app))[0]
        assert resolved.status == "resolved"
        assert resolved.resolution_kind == "observation"

    @pytest.mark.asyncio
    async def test_acknowledge_then_resolve_moves_the_lifecycle(self, app):
        """§12. ACTIVE → ACKNOWLEDGED → RESOLVED, on the existing domain."""
        from app.domain.incidents import IncidentService

        await apply(app, [_finding(ComplianceState.VIOLATION)])
        incident_id = (await _incidents(app))[0].id

        async with app.state.database.session_scope() as session:
            await IncidentService(session).acknowledge(
                organization_id="org-test", incident_id=incident_id, actor="sup@example.com"
            )
        assert (await _incidents(app))[0].status == "acknowledged"

        run = await apply(app, [_finding(ComplianceState.COMPLIANT)])
        assert run.incidents_resolved == 1
        assert (await _incidents(app))[0].status == "resolved"


class TestSeverityGate:
    def test_informational_severity_does_not_raise_incidents(self):
        """The shipped face rule is `informational` because its accuracy is
        unmeasured. Findings accrue; nobody is paged on unscored evidence."""
        assert "informational" not in RAISES_INCIDENTS
        assert {"low", "medium", "high", "critical"} <= RAISES_INCIDENTS

    @pytest.mark.asyncio
    async def test_an_informational_violation_is_not_persisted_as_an_incident(self, app):
        import dataclasses

        finding = dataclasses.replace(
            _finding(ComplianceState.VIOLATION), severity="informational"
        )
        run = await apply(app, [finding])
        assert run.incidents_opened == 0
        assert await _incidents(app) == []


class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_a_violation_is_invisible_to_another_tenant(self, app):
        """§20."""
        await apply(app, [_finding(ComplianceState.VIOLATION)])
        assert len(await _incidents(app, "org-test")) == 1
        assert await _incidents(app, "org-other") == []


class TestFindingSnapshot:
    def test_the_snapshot_keeps_the_reasoning_not_just_the_verdict(self):
        wire = _finding_wire(_finding(ComplianceState.VIOLATION))
        assert wire["state"] == "violation"
        assert wire["ruleset_version"] == "test-2026.1"
        condition = wire["conditions"][0]
        assert condition["attribute"] == "head_covering"
        assert condition["observed"] == "none"
        assert condition["message"] == "is not wearing a head covering"
        # A verdict reached under partial coverage is a different claim from one
        # reached under full coverage, and a reviewer cannot tell without this.
        assert "coverage_fraction" in wire

    def test_the_incident_is_stamped_with_the_observation_time_not_now(self):
        """A backlog must not look like a burst of fresh violations."""
        finding = _finding(ComplianceState.VIOLATION)
        assert _observed_at(finding).timestamp() == pytest.approx(
            finding.evaluated_at.ns / 1_000_000_000
        )


class TestPassCounters:
    """The counter bug that only appeared once a real violation existed.

    `record` used `setattr(self, finding.state.value, ...)`, which resolved for
    `compliant` and `unknown` and raised `AttributeError` for `violation` —
    because the field is `violations`. Every pass over a compliant or
    unobservable kitchen worked perfectly; the first real violation 500'd the
    route. All three states are asserted here so no future rename can
    reintroduce it silently.
    """

    def test_every_state_increments_its_own_counter(self):
        from app.vision.compliance_driver import CompliancePass

        run = CompliancePass()
        run.record(_finding(ComplianceState.VIOLATION))
        run.record(_finding(ComplianceState.COMPLIANT))
        run.record(_finding(ComplianceState.UNKNOWN))
        run.record(_finding(ComplianceState.VIOLATION, object_id="obj-2"))

        assert (run.violations, run.compliant, run.unknown) == (2, 1, 1)
        assert run.findings == 4
        assert run.by_rule["kitchen.person.ppe.v1"] == {
            "compliant": 1, "violation": 2, "unknown": 1
        }

    def test_the_wire_form_carries_every_counter(self):
        from app.vision.compliance_driver import CompliancePass

        run = CompliancePass()
        run.record(_finding(ComplianceState.VIOLATION))
        wire = run.to_wire()
        assert wire["violations"] == 1
        assert wire["by_rule"]["kitchen.person.ppe.v1"]["violation"] == 1

    def test_the_snapshot_distinguishes_failed_from_unresolved(self):
        """`satisfied` is tri-state and must not flatten to a boolean.

        `false` (positively observed and wrong) and `null` (could not be
        established) are the difference between a violation and a refusal, and
        JSON makes them easy to confuse.
        """
        wire = _finding_wire(_finding(ComplianceState.VIOLATION))
        assert wire["conditions"][0]["outcome"] == "failed"

        wire = _finding_wire(_finding(ComplianceState.UNKNOWN))
        condition = wire["conditions"][0]
        assert condition["outcome"] == "unresolved"
        assert condition["unknown_reason"] == "not_observable"

        wire = _finding_wire(_finding(ComplianceState.COMPLIANT))
        assert wire["conditions"][0]["outcome"] == "held"

    def test_the_snapshot_carries_the_evidence_handle_never_imagery(self):
        wire = _finding_wire(_finding(ComplianceState.VIOLATION))
        condition = wire["conditions"][0]
        assert "evidence_ref" in condition
        blob = str(wire)
        for forbidden in ("jpeg", "jpg", "base64", "\xff\xd8"):
            assert forbidden not in blob.lower()
