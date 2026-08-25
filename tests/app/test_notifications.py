"""Phase 9: notification, and the evidence capture that rides with it.

The properties that matter here are all about *restraint*: who does **not** get
told, and how often. A safety system that pages on uncertainty, or once per
frame, is one an operator learns to ignore — and an ignored alarm is worse than
no alarm, because it looks like coverage.
"""

from __future__ import annotations

import json

import pytest

from app.domain.notifications import (
    FileChannel,
    LogChannel,
    Notifier,
    NullChannel,
    build_notifier,
)
from compliance import ComplianceState
from tests.app.test_compliance_incidents import _finding, _incidents, _rules


class _Settings:
    default_tenant_id = "org-test"
    evidence_capture = False
    evidence_path = "./var/evidence-test"
    evidence_retention_days = 30
    notification_channel = "log"
    notification_file_path = "./var/test-notifications.jsonl"


class _RecordingChannel:
    channel_id = "recording"

    def __init__(self, *, succeed: bool = True, explode: bool = False) -> None:
        self.sent = []
        self._succeed = succeed
        self._explode = explode

    async def send(self, notice):
        if self._explode:
            raise RuntimeError("channel is down")
        self.sent.append(notice)
        return self._succeed


async def _run(app, findings, *, settings=None, notifier=None, wall=None):
    from app.vision.compliance_driver import ComplianceDriver

    driver = ComplianceDriver(
        settings=settings or _Settings(),
        vision=None,
        database=app.state.database,
        rules=_rules(),
        wall=wall,
        notifier=notifier,
    )
    return await driver.apply(findings, cameras={"cam-12": "rest-01"})


class TestChannelSelection:
    def test_the_default_channel_is_a_real_one(self):
        notifier = build_notifier(_Settings())
        assert isinstance(notifier, Notifier)
        assert notifier.channel_id == "log"

    def test_off_means_no_notifier_and_is_not_an_error(self):
        settings = _Settings()
        settings.notification_channel = "off"
        assert build_notifier(settings) is None

    def test_an_unknown_channel_is_refused_loudly(self):
        """A typo must not look like a deployment that happens to tell nobody."""
        settings = _Settings()
        settings.notification_channel = "slakc"
        with pytest.raises(ValueError, match="unknown notification channel"):
            build_notifier(settings)


class TestDispatch:
    @pytest.mark.asyncio
    async def test_a_new_violation_notifies_once(self, app):
        channel = _RecordingChannel()
        run = await _run(
            app, [_finding(ComplianceState.VIOLATION)], notifier=Notifier(channel)
        )
        assert run.incidents_opened == 1
        assert run.notifications_sent == 1
        assert len(channel.sent) == 1

        notice = channel.sent[0]
        assert notice.camera_key == "cam-12"
        assert notice.rule_id == "kitchen.person.ppe.v1"
        assert notice.severity == "high"
        assert notice.incident_id

    @pytest.mark.asyncio
    async def test_a_continuing_violation_does_not_notify_again(self, app):
        """§2. One active violation must not page somebody every five seconds."""
        channel = _RecordingChannel()
        notifier = Notifier(channel)

        await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=notifier)
        for _ in range(5):
            run = await _run(
                app, [_finding(ComplianceState.VIOLATION)], notifier=notifier
            )
            assert run.notifications_sent == 0

        assert len(channel.sent) == 1, "a running violation notified more than once"
        assert len(await _incidents(app)) == 1

    @pytest.mark.asyncio
    async def test_unknown_never_notifies(self, app):
        """The four-state design must not be undone at the last step."""
        channel = _RecordingChannel()
        run = await _run(
            app, [_finding(ComplianceState.UNKNOWN)], notifier=Notifier(channel)
        )
        assert run.notifications_sent == 0
        assert channel.sent == []

    @pytest.mark.asyncio
    async def test_compliant_never_notifies(self, app):
        channel = _RecordingChannel()
        await _run(
            app, [_finding(ComplianceState.COMPLIANT)], notifier=Notifier(channel)
        )
        assert channel.sent == []

    @pytest.mark.asyncio
    async def test_an_informational_finding_never_reaches_a_channel(self, app):
        import dataclasses

        channel = _RecordingChannel()
        finding = dataclasses.replace(
            _finding(ComplianceState.VIOLATION), severity="informational"
        )
        run = await _run(app, [finding], notifier=Notifier(channel))
        assert (run.incidents_opened, run.notifications_sent) == (0, 0)
        assert channel.sent == []


class TestDeliveryIsNotTheRecord:
    @pytest.mark.asyncio
    async def test_a_channel_failure_does_not_roll_back_the_incident(self, app):
        """A violation that happened is a fact. Whether anyone was told is a
        different fact, and losing the first because of the second would be the
        worse failure by far."""
        channel = _RecordingChannel(explode=True)
        notifier = Notifier(channel)

        run = await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=notifier)

        assert run.incidents_opened == 1
        assert len(await _incidents(app)) == 1, "the incident must survive"
        assert notifier.audit.failed == 1
        assert "channel is down" in notifier.audit.last_error

    @pytest.mark.asyncio
    async def test_a_suppressing_channel_is_counted_not_mistaken_for_success(self, app):
        notifier = Notifier(NullChannel())
        run = await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=notifier)
        assert run.notifications_sent == 0
        assert notifier.audit.suppressed == 1
        assert notifier.audit.sent == 0


class TestNoticeContents:
    @pytest.mark.asyncio
    async def test_the_notice_carries_no_imagery(self, app):
        """§6. Only the evidence *reference* leaves; bytes never do."""
        channel = _RecordingChannel()
        await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=Notifier(channel))

        blob = json.dumps(channel.sent[0].to_wire()).lower()
        for forbidden in ("jpeg", "base64", "\\xff\\xd8", "payload", "storage_ref"):
            assert forbidden not in blob

    @pytest.mark.asyncio
    async def test_the_notice_says_which_condition_failed(self, app):
        channel = _RecordingChannel()
        finding = _finding(ComplianceState.VIOLATION)
        await _run(app, [finding], notifier=Notifier(channel))

        reasons = " ".join(channel.sent[0].reasons)
        assert "head_covering" in reasons
        assert "none" in reasons

    @pytest.mark.asyncio
    async def test_the_wire_form_is_json_serialisable(self, app):
        channel = _RecordingChannel()
        await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=Notifier(channel))
        wire = channel.sent[0].to_wire()
        assert json.loads(json.dumps(wire))["type"] == "incident.opened"


class TestFileChannel:
    @pytest.mark.asyncio
    async def test_it_appends_one_json_object_per_line(self, app, tmp_path):
        target = tmp_path / "notifications.jsonl"
        notifier = Notifier(FileChannel(target))

        await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=notifier)
        await _run(
            app,
            [_finding(ComplianceState.VIOLATION, object_id="obj-2")],
            notifier=notifier,
        )

        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["type"] == "incident.opened"

    @pytest.mark.asyncio
    async def test_the_log_channel_reports_delivery(self, app):
        notifier = Notifier(LogChannel())
        run = await _run(app, [_finding(ComplianceState.VIOLATION)], notifier=notifier)
        assert run.notifications_sent == 1
        assert notifier.audit.sent == 1


class TestEvidenceCapture:
    class _Stream:
        def __init__(self, jpeg: bytes) -> None:
            self._jpeg = jpeg

        def latest(self, after_seq, timeout):
            return 1, self._jpeg

    class _Wall:
        def __init__(self, jpeg: bytes | None) -> None:
            self._jpeg = jpeg

        def get(self, camera_id):
            if self._jpeg is None:
                return None
            return TestEvidenceCapture._Stream(self._jpeg)

    @pytest.mark.asyncio
    async def test_capture_is_off_unless_the_deployment_enables_it(self, app):
        """§1. Storing images of identifiable people is never inherited."""
        settings = _Settings()
        assert settings.evidence_capture is False

        wall = self._Wall(b"\xff\xd8fake-jpeg\xff\xd9")
        run = await _run(
            app, [_finding(ComplianceState.VIOLATION)], settings=settings, wall=wall
        )
        assert run.evidence_captured == 0
        assert (await _incidents(app))[0].evidence_refs in ("", None)

    @pytest.mark.asyncio
    async def test_an_enabled_deployment_stores_a_frame_against_the_incident(
        self, app, tmp_path
    ):
        from app.domain.evidence import EvidenceStore

        settings = _Settings()
        settings.evidence_capture = True
        settings.evidence_path = str(tmp_path / "evidence")

        wall = self._Wall(b"\xff\xd8fake-jpeg-bytes\xff\xd9")
        finding = _finding(ComplianceState.VIOLATION)
        run = await _run(app, [finding], settings=settings, wall=wall)

        assert run.evidence_captured == 1
        incident = (await _incidents(app))[0]
        assert incident.evidence_refs, "the incident must carry the handle"

        async with app.state.database.session_scope() as session:
            record = await EvidenceStore(
                session, root=settings.evidence_path
            ).metadata(organization_id="org-test", evidence_ref=incident.evidence_refs)

        # Right camera, right subject, its own honest timestamp.
        assert record.camera_key == "cam-12"
        assert record.object_id == "obj-1"
        assert record.captured_at is not None
        assert record.content_hash.startswith("blake2b:")
        assert record.size_bytes > 0

    @pytest.mark.asyncio
    async def test_a_camera_with_no_frame_does_not_fail_the_incident(self, app):
        """An incident without a picture is still a violation to act on."""
        settings = _Settings()
        settings.evidence_capture = True

        run = await _run(
            app, [_finding(ComplianceState.VIOLATION)], settings=settings,
            wall=self._Wall(None),
        )
        assert run.incidents_opened == 1
        assert run.evidence_captured == 0

    @pytest.mark.asyncio
    async def test_a_continuing_violation_captures_only_once(self, app, tmp_path):
        """Otherwise a running violation writes a frame every five seconds."""
        settings = _Settings()
        settings.evidence_capture = True
        settings.evidence_path = str(tmp_path / "evidence")
        wall = self._Wall(b"\xff\xd8fake\xff\xd9")

        first = await _run(
            app, [_finding(ComplianceState.VIOLATION)], settings=settings, wall=wall
        )
        assert first.evidence_captured == 1

        for _ in range(3):
            again = await _run(
                app, [_finding(ComplianceState.VIOLATION)], settings=settings, wall=wall
            )
            assert again.evidence_captured == 0
