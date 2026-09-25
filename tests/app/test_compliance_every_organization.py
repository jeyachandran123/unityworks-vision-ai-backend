"""The compliance pass serves every organization, each on its own.

### The defect this closes

The pass read one organization — `DEFAULT_TENANT_ID` — and asked Vision OS for
its cameras by bare key (`cam-11`). Observations have been stored under the
runtime identity (`org-unityworks:cam-11`) since 2026-09-03, so every read came
back empty: no finding, no incident, no alert, for three weeks, while
perception ran normally. And an organization other than the configured one —
PCC1, PCC2, every future one — could never raise an alert at all.

### What must hold

- Every **active** organization is evaluated, under its **own** tenant, with
  the camera ids its observations are actually stored under.
- A finding becomes an incident in the organization that owns the camera, on
  the camera's own key and zone — and never in another organization's queue.
- One organization failing does not stop the others.
- No organization is fixed in code or configuration.

The platform read and the rule evaluation are replaced at their two seams
(`_observation_reader`, `evaluate`); the rules themselves are covered in
`compliance/` and `test_compliance_incidents.py` and are not changed here.
"""

from __future__ import annotations

import inspect

import pytest
from sqlalchemy import select

from app.domain.models import Camera, Zone
from app.users.models import Organization
from app.vision import compliance_driver as module
from app.vision.compliance_driver import ComplianceDriver, CompliancePass
from compliance import ComplianceState
from tests.app.conftest import make_recorder
from tests.app.test_compliance_incidents import _finding, _rules

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def estate(seeded):
    """Two active organizations with a `cam-12` each, one suspended one."""
    app = seeded
    async with app.state.database.session_scope() as session:
        session.add(
            Organization(id="org-paused", name="Paused", slug="paused-slug", status="suspended")
        )
        await session.flush()
        session.add(make_recorder("org-paused"))
        for org, zone in (
            ("org-test", "zone-a"),
            ("org-other", "zone-b"),
            ("org-paused", "zone-p"),
        ):
            session.add(Zone(id=zone, organization_id=org, name=f"Kitchen {org}"))
        await session.flush()
        for org, key, zone, enabled in (
            ("org-test", "cam-12", "zone-a", True),
            ("org-test", "cam-13", "zone-a", False),
            ("org-other", "cam-12", "zone-b", True),
            ("org-paused", "cam-12", "zone-p", True),
        ):
            session.add(
                Camera(
                    id=f"{org}-{key}",
                    organization_id=org,
                    zone_id=zone,
                    recorder_id=f"rec-{org}",
                    camera_key=key,
                    name=f"{org} {key}",
                    channel=int(key[-2:]),
                    enabled=enabled,
                )
            )
    return app


def _driver(app) -> ComplianceDriver:
    return ComplianceDriver(
        settings=app.state.settings, vision=None, database=app.state.database, rules=_rules()
    )


async def _incidents(app, organization_id: str) -> list:
    from app.domain.incidents import IncidentService

    async with app.state.database.session_scope() as session:
        rows = await IncidentService(session).list(organization_id=organization_id)
        return list(rows[0]) if isinstance(rows, tuple) else list(rows)


class _Reader:
    """Stands in for `ObservationReader`, recording what it was asked."""

    def __init__(self, organization_id: str, asked: list) -> None:
        self.organization_id = organization_id
        self.asked = asked

    def read(self, scope):
        self.asked.append(
            (self.organization_id, str(scope.tenant_id), tuple(str(c) for c in scope.camera_ids))
        )
        return None


def _violations_for_each_organization(snapshot):
    """`evaluate`, replaced: one violation on the organization's own cam-12."""
    run = CompliancePass()
    finding = _finding(
        ComplianceState.VIOLATION,
        object_id=f"person-{snapshot}",
        camera=f"{snapshot}:cam-12",
    )
    run.record(finding)
    return run, (finding,)


# ── which organizations, which cameras ───────────────────────────────────────


async def test_every_active_organization_is_evaluated_with_its_switched_on_cameras(estate):
    assert await _driver(estate).estate() == {
        "org-test": {"cam-12": "zone-a"},
        "org-other": {"cam-12": "zone-b"},
    }


async def test_the_platform_is_asked_under_each_organizations_own_tenant_and_ids(
    estate, monkeypatch
):
    """The ids observations are stored under: `org-other:cam-12`, not `cam-12`."""
    driver = _driver(estate)
    asked: list = []
    monkeypatch.setattr(
        ComplianceDriver, "_observation_reader", lambda self, org: _Reader(org, asked)
    )

    driver.snapshot(("cam-12",), organization_id="org-other")

    assert asked == [("org-other", "org-other", ("org-other:cam-12",))]


# ── findings become each organization's own incidents ────────────────────────


async def test_each_organizations_violation_becomes_its_own_incident(estate, monkeypatch):
    driver = _driver(estate)
    monkeypatch.setattr(
        ComplianceDriver, "snapshot", lambda self, keys, *, organization_id: organization_id
    )
    monkeypatch.setattr(
        ComplianceDriver,
        "evaluate",
        lambda self, snapshot: _violations_for_each_organization(snapshot),
    )

    total = await driver.run_once()

    mine = await _incidents(estate, "org-test")
    theirs = await _incidents(estate, "org-other")
    assert [(i.camera_key, i.zone_id, i.object_id) for i in mine] == [
        ("cam-12", "zone-a", "person-org-test")
    ]
    assert [(i.camera_key, i.zone_id, i.object_id) for i in theirs] == [
        ("cam-12", "zone-b", "person-org-other")
    ]
    assert await _incidents(estate, "org-paused") == []
    assert total.incidents_opened == 2
    assert driver.last_pass_for("org-test").incidents_opened == 1
    assert driver.last_pass_for("org-other").incidents_opened == 1


async def test_a_compliant_observation_resolves_the_incident_in_its_own_organization(
    estate, monkeypatch
):
    driver = _driver(estate)
    violation = _finding(ComplianceState.VIOLATION, object_id="p1", camera="org-other:cam-12")
    compliant = _finding(ComplianceState.COMPLIANT, object_id="p1", camera="org-other:cam-12")

    await driver.apply([violation], cameras={"cam-12": "zone-b"}, organization_id="org-other")
    run = await driver.apply([compliant], cameras={"cam-12": "zone-b"}, organization_id="org-other")

    assert run.incidents_resolved == 1
    [incident] = await _incidents(estate, "org-other")
    assert incident.status == "resolved"


async def test_one_organization_failing_does_not_stop_the_others(estate, monkeypatch):
    driver = _driver(estate)
    monkeypatch.setattr(
        ComplianceDriver, "snapshot", lambda self, keys, *, organization_id: organization_id
    )

    def evaluate(self, snapshot):
        if snapshot == "org-other":
            raise RuntimeError("platform read failed for this tenant")
        return _violations_for_each_organization(snapshot)

    monkeypatch.setattr(ComplianceDriver, "evaluate", evaluate)
    await driver.run_once()

    assert len(await _incidents(estate, "org-test")) == 1
    assert driver.last_pass_for("org-other").errors == 1


async def test_a_finding_on_another_organizations_camera_is_never_filed_here(estate):
    """Defence in depth: a runtime id carries its owner, and a pass for one
    organization refuses a finding that names another's camera."""
    driver = _driver(estate)
    stray = _finding(ComplianceState.VIOLATION, object_id="p9", camera="org-other:cam-12")

    run = await driver.apply([stray], cameras={"cam-12": "zone-a"}, organization_id="org-test")

    assert await _incidents(estate, "org-test") == []
    assert await _incidents(estate, "org-other") == []
    assert run.errors == 1


# ── nothing is fixed to one organization ─────────────────────────────────────


def test_no_organization_is_fixed_in_the_compliance_pass():
    source = inspect.getsource(module)
    assert "default_tenant_id" not in source


async def test_the_evidence_and_the_alert_carry_the_owning_organization(
    estate, monkeypatch, tmp_path
):
    """The frame is stored for the camera's organization, under its own key."""
    from app.domain.models import EvidenceRecord
    from app.vision.decision_frames import DECISION_FRAMES

    monkeypatch.setattr(estate.state.settings, "evidence_capture", True)
    monkeypatch.setattr(estate.state.settings, "evidence_path", str(tmp_path / "evidence"))
    DECISION_FRAMES.clear()
    SECOND = 1_000_000_000
    ref = "org-other:cam-12/e1/f1"
    # Filed under the id the platform analysed with, as ingest files it.
    DECISION_FRAMES.remember(
        camera_id="org-other:cam-12",
        frame_ref=ref,
        captured_at_ns=1_787_000_000 * SECOND - SECOND,
        width=64,
        height=48,
        jpeg=b"\xff\xd8\xffscene",
    )
    DECISION_FRAMES.attach_subject(
        camera_id="org-other:cam-12",
        frame_ref=ref,
        object_id="p1",
        box=(0.1, 0.2, 0.4, 0.9),
        crop_jpeg=b"\xff\xd8\xffp1",
        crop_id="crop-p1",
        sent_to_model=True,
        object_class="person",
    )
    try:
        driver = _driver(estate)
        finding = _finding(ComplianceState.VIOLATION, object_id="p1", camera="org-other:cam-12")
        run = await driver.apply(
            [finding], cameras={"cam-12": "zone-b"}, organization_id="org-other"
        )
    finally:
        DECISION_FRAMES.clear()

    assert run.evidence_captured == 1
    async with estate.state.database.session_scope() as session:
        records = (await session.execute(select(EvidenceRecord))).scalars().all()
    assert {(r.organization_id, r.camera_key) for r in records} == {("org-other", "cam-12")}
