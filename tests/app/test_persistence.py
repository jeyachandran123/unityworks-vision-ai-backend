"""Phase 5: durable state.

The suite is organised around the claims Phase 5 makes, and the last class is
the one that matters most: **restart recovery**. Everything before it could be
satisfied by an in-memory store that looks durable until the process ends. Only
`TestRestartRecovery` builds a second application against the same file-backed
database and asserts that what the first one wrote is still there.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from app.authorization.model import Permission, Role, permissions_for
from app.configuration.settings import Settings
from app.domain.audit import REDACTED, AuditAction, AuditTrail
from app.domain.audit import to_wire as audit_to_wire
from app.domain.cameras import CameraService, FrameService, to_rtsp_config
from app.domain.cameras import to_wire as camera_to_wire
from app.domain.evidence import EvidenceStore
from app.domain.evidence import to_wire as evidence_to_wire
from app.domain.incidents import IncidentService
from app.domain.models import EvidenceState, IncidentStatus
from app.domain.retention import RetentionService
from app.errors import ConflictError, EvidenceForbiddenError, NotFoundError, ValidationError
from app.infrastructure.database import create_all_for_tests
from app.main import create_app
from tests.app.conftest import bearer, make_user

ORG = "org-test"
OTHER = "org-other"


@pytest_asyncio.fixture
async def session(app):
    """A session against the suite's in-memory database."""
    async with app.state.database.session_scope() as active:
        yield active


@pytest_asyncio.fixture
async def admin_headers(seeded, client):
    """An `org_admin` inside the test tenant.

    `developer` deliberately does not hold `MANAGE_CAMERAS` — DevTools access is
    not operator authority — so camera administration needs its own caller.
    """
    async with seeded.state.database.session_scope() as session:
        _, admin = make_user(
            email="admin@example.com",
            roles=("org_admin",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(admin)
    return await bearer(client, "admin@example.com")


# ── cameras ──────────────────────────────────────────────────────────────────


class TestCameraConfiguration:
    @pytest.mark.asyncio
    async def test_a_new_camera_is_disabled_until_somebody_enables_it(self, session):
        """Adding a camera must not silently begin processing video of people."""
        service = CameraService(session)
        camera = await service.create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-09",
            name="Prep bench",
            channel=9,
            host="10.0.0.5",
        )
        assert camera.enabled is False

        running = await service.enabled_for_runtime(organization_id=ORG)
        assert running == []

    @pytest.mark.asyncio
    async def test_only_enabled_cameras_reach_the_runtime(self, session):
        """Sixteen DVR channels must not become sixteen pipelines."""
        service = CameraService(session)
        for channel in range(1, 5):
            await service.create(
                organization_id=ORG,
                restaurant_id="rest-01",
                camera_key=f"cam-{channel:02d}",
                name=f"Camera {channel}",
                channel=channel,
                host="10.0.0.5",
            )
        await session.flush()
        await service.set_enabled(organization_id=ORG, camera_key="cam-02", enabled=True)

        running = await service.enabled_for_runtime(organization_id=ORG)
        assert [c.camera_key for c in running] == ["cam-02"]

    @pytest.mark.asyncio
    async def test_a_password_cannot_be_stored_in_the_camera_row(self, session):
        """A database dump must not be a credential dump."""
        with pytest.raises(ValidationError):
            await CameraService(session).create(
                organization_id=ORG,
                restaurant_id="rest-01",
                camera_key="cam-99",
                name="Bad",
                channel=99,
                host="10.0.0.5",
                credential_ref="hunter2",  # a value, not a reference
            )

    @pytest.mark.asyncio
    async def test_a_reference_is_accepted_and_never_resolved_here(self, session):
        camera = await CameraService(session).create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-10",
            name="Wash",
            channel=10,
            host="10.0.0.5",
            username="admin",
            credential_ref="env:CCTV_PASSWORD",
        )
        assert camera.credential_ref == "env:CCTV_PASSWORD"

        wire = camera_to_wire(camera)
        assert wire["credential_configured"] is True
        assert "hunter2" not in str(wire)
        # The URI is diagnosable and carries no credential.
        assert wire["uri"] == "rtsp://***:***@10.0.0.5:554/cam/realmonitor?channel=10&subtype=1"

    @pytest.mark.asyncio
    async def test_the_row_translates_to_runtime_config_without_resolving(self, session):
        camera = await CameraService(session).create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-11",
            name="Line",
            channel=11,
            host="10.0.0.5",
            username="admin",
            credential_ref="env:CCTV_PASSWORD",
            analysis_fps=2.0,
        )
        config = to_rtsp_config(camera)
        assert config.camera_id == "cam-11"
        assert config.channel == 11
        assert config.analysis_fps == 2.0
        assert config.credential_ref == "env:CCTV_PASSWORD"
        assert "***" in config.redacted_uri()

    @pytest.mark.asyncio
    async def test_duplicate_keys_are_refused(self, session):
        service = CameraService(session)
        await service.create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-12",
            name="A",
            channel=12,
            host="10.0.0.5",
        )
        await session.flush()
        with pytest.raises(ConflictError):
            await service.create(
                organization_id=ORG,
                restaurant_id="rest-01",
                camera_key="cam-12",
                name="B",
                channel=12,
                host="10.0.0.5",
            )

    @pytest.mark.asyncio
    async def test_an_empty_camera_grant_selects_nothing(self, session):
        """`()` is none. It must never read as a wildcard."""
        service = CameraService(session)
        await service.create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-13",
            name="A",
            channel=13,
            host="10.0.0.5",
        )
        await session.flush()

        assert await service.list(organization_id=ORG, camera_keys=()) == []
        assert len(await service.list(organization_id=ORG, camera_keys=None)) == 1

    @pytest.mark.asyncio
    async def test_a_camera_is_invisible_across_tenants(self, session):
        service = CameraService(session)
        await service.create(
            organization_id=ORG,
            restaurant_id="rest-01",
            camera_key="cam-14",
            name="A",
            channel=14,
            host="10.0.0.5",
        )
        await session.flush()
        with pytest.raises(NotFoundError):
            await service.get(organization_id=OTHER, camera_key="cam-14")


# ── evidence ─────────────────────────────────────────────────────────────────


class TestEvidenceLifecycle:
    @pytest.mark.asyncio
    async def test_bytes_survive_and_are_integrity_checked(self, session, tmp_path):
        store = EvidenceStore(session, root=tmp_path)
        await store.put(
            organization_id=ORG,
            evidence_ref="ev-1",
            camera_key="cam-01",
            payload=b"not-really-a-jpeg",
            captured_at=datetime.now(UTC),
            purpose="head covering finding",
            retention_days=30,
        )
        await session.flush()

        record, payload = await store.fetch(organization_id=ORG, evidence_ref="ev-1")
        assert payload == b"not-really-a-jpeg"
        assert record.content_hash.startswith("blake2b:")

    @pytest.mark.asyncio
    async def test_corrupted_bytes_are_refused_rather_than_served(self, session, tmp_path):
        """The hash is the integrity check, not decoration."""
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-2",
            camera_key="cam-01",
            payload=b"original",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        (tmp_path / record.storage_ref).write_bytes(b"tampered")

        with pytest.raises(NotFoundError):
            await store.fetch(organization_id=ORG, evidence_ref="ev-2")

    @pytest.mark.asyncio
    async def test_expired_evidence_is_refused_even_while_the_bytes_remain(self, session, tmp_path):
        """Retention is a promise about what is served, not only what is erased."""
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-3",
            camera_key="cam-01",
            payload=b"aged",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        record.expires_at = datetime.now(UTC) - timedelta(days=1)

        expired = await store.expire_due(organization_id=ORG)
        assert expired == ["ev-3"]
        assert record.state == EvidenceState.EXPIRED.value
        # Still on disk, and still refused.
        assert (tmp_path / record.storage_ref).is_file()
        with pytest.raises(EvidenceForbiddenError):
            await store.fetch(organization_id=ORG, evidence_ref="ev-3")

    @pytest.mark.asyncio
    async def test_deletion_erases_the_bytes_and_keeps_a_tombstone(self, session, tmp_path):
        """An erasure request has to be provable afterwards."""
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-4",
            camera_key="cam-01",
            payload=b"to-erase",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        path = tmp_path / record.storage_ref

        deleted = await store.delete(
            organization_id=ORG,
            evidence_ref="ev-4",
            actor="officer@example.com",
            reason="subject access request",
        )
        assert not path.exists()
        assert deleted.state == EvidenceState.DELETED.value
        assert deleted.deleted_by == "officer@example.com"
        assert deleted.deletion_reason == "subject access request"
        # The row survived: that is what makes the deletion provable.
        assert (
            await store.metadata(organization_id=ORG, evidence_ref="ev-4")
        ).deleted_at is not None

    @pytest.mark.asyncio
    async def test_metadata_never_carries_a_storage_path(self, session, tmp_path):
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-5",
            camera_key="cam-01",
            payload=b"x",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        wire = evidence_to_wire(record)
        assert "storage_ref" not in wire
        assert str(tmp_path) not in str(wire)

    @pytest.mark.asyncio
    async def test_evidence_is_invisible_across_tenants(self, session, tmp_path):
        store = EvidenceStore(session, root=tmp_path)
        await store.put(
            organization_id=ORG,
            evidence_ref="ev-6",
            camera_key="cam-01",
            payload=b"x",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        with pytest.raises(NotFoundError):
            await store.metadata(organization_id=OTHER, evidence_ref="ev-6")


# ── incidents ────────────────────────────────────────────────────────────────


def _finding() -> dict:
    return {
        "attribute": "head_covering",
        "state": "ABSENT",
        "confidence": 0.82,
        "evaluated_at": "2026-08-21T09:00:00+00:00",
    }


class TestIncidentLifecycle:
    @pytest.mark.asyncio
    async def test_a_repeat_finding_updates_rather_than_raising_a_second_incident(self, session):
        """Four minutes in frame is one incident, not two hundred."""
        service = IncidentService(session)
        first, created = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-7",
            observed_at=datetime(2026, 8, 21, 9, 0, tzinfo=UTC),
            finding=_finding(),
            evidence_refs=("ev-a",),
        )
        assert created is True
        await session.flush()

        again, created_again = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-7",
            observed_at=datetime(2026, 8, 21, 9, 4, tzinfo=UTC),
            finding=_finding(),
            evidence_refs=("ev-b",),
        )
        assert created_again is False
        assert again.id == first.id
        # The repeat is not lost: the incident shows it is ongoing.
        assert again.observed_at.replace(tzinfo=UTC) == datetime(2026, 8, 21, 9, 4, tzinfo=UTC)
        assert set(again.evidence_refs.split(",")) == {"ev-a", "ev-b"}

    @pytest.mark.asyncio
    async def test_nothing_but_an_observation_or_an_operator_closes_an_incident(self, session):
        service = IncidentService(session)
        incident, _ = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-8",
            observed_at=datetime.now(UTC),
            finding=_finding(),
        )
        await session.flush()

        for invented in ("auto", "timeout", "viewed", ""):
            with pytest.raises(ValidationError):
                await service.resolve(
                    organization_id=ORG,
                    incident_id=incident.id,
                    actor="someone",
                    kind=invented,
                )

    @pytest.mark.asyncio
    async def test_an_operator_resolution_requires_a_reason(self, session):
        service = IncidentService(session)
        incident, _ = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-9",
            observed_at=datetime.now(UTC),
            finding=_finding(),
        )
        await session.flush()
        with pytest.raises(ValidationError):
            await service.resolve(
                organization_id=ORG,
                incident_id=incident.id,
                actor="manager@example.com",
                kind="operator",
                note="   ",
            )

    @pytest.mark.asyncio
    async def test_a_later_observation_closes_it_and_says_so(self, session):
        """`resolution_kind` keeps "the system saw it fixed" distinct from
        "a manager said it was fixed"."""
        service = IncidentService(session)
        await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-10",
            observed_at=datetime.now(UTC),
            finding=_finding(),
        )
        await session.flush()

        resolved = await service.resolve_by_observation(
            organization_id=ORG,
            camera_key="cam-01",
            object_id="person-10",
            rule_id="rule.head_covering",
        )
        assert resolved is not None
        assert resolved.status == IncidentStatus.RESOLVED.value
        assert resolved.resolution_kind == "observation"

        # Nothing open any more, so a second call is a no-op rather than an error.
        assert (
            await service.resolve_by_observation(
                organization_id=ORG,
                camera_key="cam-01",
                object_id="person-10",
                rule_id="rule.head_covering",
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_acknowledging_does_not_close_it(self, session):
        service = IncidentService(session)
        incident, _ = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-11",
            observed_at=datetime.now(UTC),
            finding=_finding(),
        )
        await session.flush()

        acked = await service.acknowledge(
            organization_id=ORG, incident_id=incident.id, actor="supervisor@example.com"
        )
        assert acked.status == IncidentStatus.ACKNOWLEDGED.value
        assert acked.resolved_at is None

    @pytest.mark.asyncio
    async def test_the_finding_is_frozen_not_recomputed(self, session):
        """An incident must stay explicable after the rules change."""
        service = IncidentService(session)
        incident, _ = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-12",
            observed_at=datetime.now(UTC),
            ruleset_version="2026.08.1",
            finding=_finding(),
        )
        await session.flush()

        from app.domain.incidents import to_wire

        wire = to_wire(incident)
        assert wire["finding"]["state"] == "ABSENT"
        assert wire["finding"]["confidence"] == 0.82
        assert wire["ruleset_version"] == "2026.08.1"


# ── audit ────────────────────────────────────────────────────────────────────


class TestAuditTrail:
    @pytest.mark.asyncio
    async def test_secrets_never_reach_an_audit_row(self, session):
        """The longest-retained table in the system is the worst place to leak."""
        trail = AuditTrail(session)
        event = await trail.record(
            action=AuditAction.CAMERA_UPDATED,
            organization_id=ORG,
            actor="admin@example.com",
            resource_type="camera",
            resource_id="cam-01",
            detail={
                "password": "hunter2",
                "api_key": "nvapi-abcdefghijklmnop",
                "nested": {"refresh_token": "eyJhbGciOi.payload.signature"},
                "uri": "rtsp://admin:hunter2@10.0.0.5:554/cam",
                "note": "a plain note survives",
            },
        )
        wire = audit_to_wire(event)
        detail = wire["detail"]

        assert detail["password"] == REDACTED
        assert detail["api_key"] == REDACTED
        assert detail["nested"]["refresh_token"] == REDACTED
        assert "hunter2" not in str(detail)
        assert "nvapi-" not in str(detail)
        assert detail["note"] == "a plain note survives"

    @pytest.mark.asyncio
    async def test_evidence_bytes_are_reduced_to_a_size(self, session):
        event = await AuditTrail(session).record(
            action=AuditAction.EVIDENCE_READ,
            organization_id=ORG,
            actor="officer@example.com",
            detail={"payload": b"\xff\xd8\xff" * 100},
        )
        assert audit_to_wire(event)["detail"]["payload"] == REDACTED

    @pytest.mark.asyncio
    async def test_the_trail_offers_no_way_to_edit_a_row(self):
        """A trail that can be edited is not one."""
        assert not hasattr(AuditTrail, "update")
        assert not hasattr(AuditTrail, "delete")
        assert not hasattr(AuditTrail, "amend")

    @pytest.mark.asyncio
    async def test_the_trail_is_tenant_scoped(self, session):
        trail = AuditTrail(session)
        await trail.record(action=AuditAction.LOGIN, organization_id=ORG, actor="a@example.com")
        await trail.record(action=AuditAction.LOGIN, organization_id=OTHER, actor="b@example.com")
        await session.flush()

        ours = await trail.query(organization_id=ORG)
        assert [e.actor for e in ours] == ["a@example.com"]


# ── retention ────────────────────────────────────────────────────────────────


class TestRetention:
    def _service(self, session, tmp_path, **days):
        return RetentionService(
            session,
            root=tmp_path,
            evidence_days=days.get("evidence", 30),
            incident_days=days.get("incident", 365),
            audit_days=days.get("audit", 730),
        )

    @pytest.mark.asyncio
    async def test_a_sweep_that_cannot_erase_destroys_nothing(self, session, tmp_path):
        """A process that starts should not begin deleting because it started."""
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-r1",
            camera_key="cam-01",
            payload=b"aged",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        record.expires_at = datetime.now(UTC) - timedelta(days=1)
        path = tmp_path / record.storage_ref

        report = await self._service(session, tmp_path).sweep(erase=False)
        assert report.dry_run is True
        assert report.evidence_expired == 1
        assert report.evidence_erased == 0
        assert path.is_file()  # marked, not erased

    @pytest.mark.asyncio
    async def test_erasure_removes_bytes_and_leaves_the_row(self, session, tmp_path):
        store = EvidenceStore(session, root=tmp_path)
        record = await store.put(
            organization_id=ORG,
            evidence_ref="ev-r2",
            camera_key="cam-01",
            payload=b"aged",
            captured_at=datetime.now(UTC),
            purpose="finding",
            retention_days=30,
        )
        await session.flush()
        record.expires_at = datetime.now(UTC) - timedelta(days=1)
        path = tmp_path / record.storage_ref

        report = await self._service(session, tmp_path).sweep(erase=True)
        assert report.evidence_erased == 1
        assert not path.exists()

        surviving = await store.metadata(organization_id=ORG, evidence_ref="ev-r2")
        assert surviving.state == EvidenceState.DELETED.value
        assert surviving.deleted_by == "retention"

    @pytest.mark.asyncio
    async def test_an_unresolved_incident_is_never_pruned_however_old(self, session, tmp_path):
        """Age is not closure. A violation nobody dealt with must not vanish."""
        service = IncidentService(session)
        incident, _ = await service.open(
            organization_id=ORG,
            camera_key="cam-01",
            rule_id="rule.head_covering",
            object_id="person-old",
            observed_at=datetime.now(UTC) - timedelta(days=5000),
            finding=_finding(),
        )
        await session.flush()
        incident.created_at = datetime.now(UTC) - timedelta(days=5000)

        report = await self._service(session, tmp_path, incident=1).sweep(erase=True)
        assert report.incidents_pruned == 0
        assert (
            await service.get(organization_id=ORG, incident_id=incident.id)
        ).status == IncidentStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_evidence_expires_before_audit_does(self):
        """Imagery outliving the record of who looked at it would be backwards."""
        settings = Settings(
            app_env="test",
            secret_key="test-only-secret-value-not-for-any-deployment",
            database_url_override="sqlite+aiosqlite:///:memory:",
        )
        assert settings.evidence_retention_days < settings.incident_retention_days
        assert settings.incident_retention_days <= settings.audit_retention_days
        # And nothing is destroyed unless a deployment asked for it.
        assert settings.retention_sweep_enabled is False


# ── permissions ──────────────────────────────────────────────────────────────


class TestPermissionSeparation:
    def test_reading_an_observation_does_not_grant_the_image(self):
        """ "A person was here" and "here is their picture" are different acts."""
        for role in Role:
            granted = permissions_for(frozenset({role}))
            if Permission.VIEW_OBSERVATIONS in granted:
                # Not asserted as never-together — some roles legitimately hold
                # both. Asserted as *not implied*: there is at least one role
                # that reads observations and may not see imagery.
                pass
        supervisor = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
        assert Permission.VIEW_OBSERVATIONS in supervisor
        assert Permission.VIEW_EVIDENCE not in supervisor

    def test_seeing_evidence_does_not_grant_destroying_it(self):
        for role in Role:
            granted = permissions_for(frozenset({role}))
            if Permission.DELETE_EVIDENCE in granted:
                assert (
                    Permission.VIEW_EVIDENCE in granted
                ), f"{role.value} can delete evidence it cannot see"
        officer = permissions_for(frozenset({Role.HYGIENE_OFFICER}))
        assert Permission.VIEW_EVIDENCE in officer
        assert Permission.DELETE_EVIDENCE not in officer

    def test_the_audit_trail_has_its_own_permission(self):
        supervisor = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
        assert Permission.VIEW_AUDIT not in supervisor
        assert Permission.VIEW_AUDIT in permissions_for(frozenset({Role.AUDITOR}))

    def test_viewing_cameras_does_not_grant_configuring_them(self):
        supervisor = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
        assert Permission.VIEW_CAMERAS in supervisor
        assert Permission.MANAGE_CAMERAS not in supervisor


# ── the API surface ──────────────────────────────────────────────────────────


class TestProductApi:
    @pytest.mark.asyncio
    async def test_a_login_leaves_an_audit_row(self, seeded, client):
        await bearer(client, "manager@example.com")

        async with seeded.state.database.session_scope() as session:
            events = await AuditTrail(session).query(organization_id=ORG, action=AuditAction.LOGIN)
        assert [e.actor for e in events] == ["manager@example.com"]

    @pytest.mark.asyncio
    async def test_a_failed_login_is_recorded_even_though_the_request_failed(self, seeded, client):
        """The row that shows a password-spraying attempt must survive the rollback."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "manager@example.com", "password": "wrong-password"},
        )
        assert response.status_code == 401

        async with seeded.state.database.session_scope() as session:
            events = await AuditTrail(session).query(
                organization_id="org-test", action=AuditAction.LOGIN_FAILED
            )
        assert len(events) == 1
        assert "wrong-password" not in str(audit_to_wire(events[0]))

    @pytest.mark.asyncio
    async def test_a_supervisor_may_list_cameras_but_not_create_one(self, seeded, client):
        headers = await bearer(client, "supervisor@example.com")

        assert (await client.get("/api/v1/cameras", headers=headers)).status_code == 200
        created = await client.post(
            "/api/v1/cameras",
            headers=headers,
            json={"camera_key": "cam-20", "name": "X", "channel": 20, "restaurant_id": "r"},
        )
        assert created.status_code == 403

    @pytest.mark.asyncio
    async def test_evidence_imagery_is_refused_when_the_deployment_disabled_it(
        self, seeded, client
    ):
        """Two gates: the permission, and a deployment decision that defaults off."""
        headers = await bearer(client, "developer@example.com")
        response = await client.get("/api/v1/evidence/whatever/image", headers=headers)
        assert response.status_code == 403
        assert "ALLOW_EVIDENCE" in response.text

        async with seeded.state.database.session_scope() as session:
            denied = await AuditTrail(session).query(
                organization_id=ORG, action=AuditAction.EVIDENCE_DENIED
            )
        # The refusal is recorded even though the request failed.
        assert len(denied) == 1

    @pytest.mark.asyncio
    async def test_the_audit_endpoint_needs_its_own_permission(self, seeded, client):
        supervisor = await bearer(client, "supervisor@example.com")
        assert (await client.get("/api/v1/audit", headers=supervisor)).status_code == 403

    @pytest.mark.asyncio
    async def test_a_camera_created_through_the_api_starts_disabled(
        self, seeded, client, admin_headers
    ):
        headers = admin_headers
        response = await client.post(
            "/api/v1/cameras",
            headers=headers,
            json={
                "camera_key": "cam-21",
                "name": "New",
                "channel": 21,
                "restaurant_id": "rest-01",
                "host": "10.0.0.5",
                "credential_ref": "env:CCTV_PASSWORD",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["enabled"] is False
        assert "hunter2" not in response.text

    @pytest.mark.asyncio
    async def test_enabling_a_camera_gets_its_own_audit_action(self, seeded, client, admin_headers):
        headers = admin_headers
        await client.post(
            "/api/v1/cameras",
            headers=headers,
            json={
                "camera_key": "cam-22",
                "name": "New",
                "channel": 22,
                "restaurant_id": "rest-01",
                "host": "10.0.0.5",
            },
        )
        patched = await client.patch(
            "/api/v1/cameras/cam-22", headers=headers, json={"enabled": True}
        )
        assert patched.status_code == 200
        assert patched.json()["enabled"] is True

        async with seeded.state.database.session_scope() as session:
            events = await AuditTrail(session).query(
                organization_id=ORG, action=AuditAction.CAMERA_ENABLED
            )
        assert [e.resource_id for e in events] == ["cam-22"]

    @pytest.mark.asyncio
    async def test_a_frame_listing_carries_no_pixels(self, seeded, client):
        async with seeded.state.database.session_scope() as session:
            await FrameService(session).record(
                organization_id=ORG,
                camera_key="cam-01",
                sequence=1,
                epoch=1,
                captured_at=datetime.now(UTC),
                received_at=datetime.now(UTC),
                width=640,
                height=360,
                source_kind="replay",
                observation_count=3,
            )

        headers = await bearer(client, "manager@example.com")
        response = await client.get("/api/v1/cameras/cam-01/frames", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        frame = body["frames"][0]
        assert frame["observation_count"] == 3
        assert frame["source_kind"] == "replay"
        assert "payload" not in frame and "image" not in frame

    @pytest.mark.asyncio
    async def test_a_camera_outside_the_grant_reads_as_empty_not_as_forbidden(self, seeded, client):
        """Confirming a camera exists is itself a disclosure."""
        headers = await bearer(client, "manager@example.com")  # granted cam-01, cam-02
        response = await client.get("/api/v1/cameras/cam-16/frames", headers=headers)
        assert response.status_code == 200
        assert response.json()["frames"] == []


# ── restart recovery ─────────────────────────────────────────────────────────


class TestRestartRecovery:
    """The claim that separates Phase 5 from Phase 4.

    Two applications, one file-backed database, and a full shutdown between
    them. Nothing here would pass against the in-memory stores Phase 4 used.
    """

    @staticmethod
    def _settings(tmp_path) -> Settings:
        return Settings(
            app_env="test",
            secret_key="test-only-secret-value-not-for-any-deployment",
            database_url_override=f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}",
            redis_enabled=False,
            vision_autostart=False,
            feature_devtools=True,
            metrics_enabled=False,
            evidence_path=str(tmp_path / "evidence"),
        )

    @pytest.mark.asyncio
    async def test_cameras_incidents_evidence_and_audit_all_survive_a_restart(self, tmp_path):
        settings = self._settings(tmp_path)

        # ── first process ────────────────────────────────────────────────────
        first = create_app(settings)
        first.state.database.connect()
        await create_all_for_tests(first.state.database)

        async with first.state.database.session_scope() as session:
            org, user = make_user(email="manager@example.com")
            session.add(org)
            session.add(user)

        async with first.state.database.session_scope() as session:
            cameras = CameraService(session)
            await cameras.create(
                organization_id=ORG,
                restaurant_id="rest-01",
                camera_key="cam-01",
                name="Prep",
                channel=1,
                host="10.0.0.5",
                credential_ref="env:CCTV_PASSWORD",
            )
            await cameras.create(
                organization_id=ORG,
                restaurant_id="rest-01",
                camera_key="cam-02",
                name="Wash",
                channel=2,
                host="10.0.0.5",
            )
            await session.flush()
            await cameras.set_enabled(organization_id=ORG, camera_key="cam-01", enabled=True)

            incident, _ = await IncidentService(session).open(
                organization_id=ORG,
                camera_key="cam-01",
                rule_id="rule.head_covering",
                object_id="person-1",
                observed_at=datetime.now(UTC),
                finding=_finding(),
                evidence_refs=("ev-restart",),
            )
            incident_id = incident.id

            await EvidenceStore(session, root=settings.evidence_path).put(
                organization_id=ORG,
                evidence_ref="ev-restart",
                camera_key="cam-01",
                payload=b"evidence-bytes",
                captured_at=datetime.now(UTC),
                purpose="head covering finding",
                retention_days=30,
            )
            await AuditTrail(session).record(
                action=AuditAction.EVIDENCE_CREATED,
                organization_id=ORG,
                actor="vision-os",
                resource_type="evidence",
                resource_id="ev-restart",
            )

        # Full shutdown: the engine is disposed, exactly as at process exit.
        await first.state.database.disconnect()

        # ── second process ───────────────────────────────────────────────────
        second = create_app(self._settings(tmp_path))
        second.state.database.connect()

        async with second.state.database.session_scope() as session:
            cameras = CameraService(session)

            # The camera set the runtime would start is restored exactly.
            running = await cameras.enabled_for_runtime(organization_id=ORG)
            assert [c.camera_key for c in running] == ["cam-01"]
            # And the one left disabled is still disabled.
            assert (await cameras.get(organization_id=ORG, camera_key="cam-02")).enabled is False
            # The credential is still a reference, not a value.
            assert running[0].credential_ref == "env:CCTV_PASSWORD"

            # The incident is still open, and still explicable.
            incident = await IncidentService(session).get(
                organization_id=ORG, incident_id=incident_id
            )
            assert incident.status == IncidentStatus.ACTIVE.value
            from app.domain.incidents import to_wire as incident_to_wire

            assert incident_to_wire(incident)["finding"]["state"] == "ABSENT"
            assert incident_to_wire(incident)["evidence_refs"] == ["ev-restart"]

            # The evidence bytes survived, and still pass their integrity check.
            record, payload = await EvidenceStore(session, root=settings.evidence_path).fetch(
                organization_id=ORG, evidence_ref="ev-restart"
            )
            assert payload == b"evidence-bytes"
            assert record.state == EvidenceState.RETAINED.value

            # The audit trail survived.
            events = await AuditTrail(session).query(
                organization_id=ORG, action=AuditAction.EVIDENCE_CREATED
            )
            assert [e.resource_id for e in events] == ["ev-restart"]

        await second.state.database.disconnect()

    @pytest.mark.asyncio
    async def test_the_runtime_starts_the_cameras_the_database_names(self, tmp_path):
        """Recovery is not "some cameras came back". It is the same set."""
        settings = self._settings(tmp_path)
        app = create_app(settings)
        app.state.database.connect()
        await create_all_for_tests(app.state.database)

        async with app.state.database.session_scope() as session:
            service = CameraService(session)
            for channel in range(1, 5):
                await service.create(
                    organization_id=ORG,
                    restaurant_id="rest-01",
                    camera_key=f"cam-{channel:02d}",
                    name=f"Camera {channel}",
                    channel=channel,
                    host="10.0.0.5",
                )
            await session.flush()
            for key in ("cam-01", "cam-03"):
                await service.set_enabled(organization_id=ORG, camera_key=key, enabled=True)

        from app.main import _start_cameras_from_database

        # FEATURE_LIVE_CCTV is off by default, so nothing dials. The assertion is
        # about *selection*: the runtime is asked for exactly the enabled rows.
        assert settings.feature_live_cctv is False
        started = await _start_cameras_from_database(app)
        assert started == 0

        async with app.state.database.session_scope() as session:
            selected = await CameraService(session).enabled_for_runtime(organization_id=ORG)
        assert [c.camera_key for c in selected] == ["cam-01", "cam-03"]

        await app.state.database.disconnect()
