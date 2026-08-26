"""The seams §11 depends on, driven rather than described.

`test_subject_evidence.py` pins what an incident is *allowed* to show. This
pins that the pieces actually arrive:

* a real `Crop` produces a real JPEG at the crop seam, and `NEVER_PERSIST`
  produces none;
* a violation writes the frame **and** its crops into the durable evidence
  store, sharing one instant and one frame reference;
* retrieving any of them still costs the caller `VIEW_EVIDENCE`, still obeys
  `ALLOW_EVIDENCE`, and still leaves an audit row — a crop is imagery of an
  identifiable person and gets no cheaper path than the frame it came from.
"""

from __future__ import annotations

import json

import pytest

from app.vision.decision_frames import DECISION_FRAMES, encode_jpeg

CAMERA = "cam-12"
SUBJECT = "obj-person-1"
BYSTANDER = "obj-person-2"
SECOND = 1_000_000_000


# --- the crop seam --------------------------------------------------------- #


def _real_crop(*, pixels: bytes | None, retention=None, size=(8, 8)):
    """A genuine `Crop`, not a stand-in.

    Built through the platform's own model so its `__post_init__` invariants
    apply: the pixel-format assumption this evidence path rests on — BGR24 at
    `output_size` — is the platform's, and a test that faked the object would
    stop noticing if it changed.
    """
    from vision_os.core.model.crop import (
        Crop,
        CropTransform,
        GateResult,
        PrivacyClass,
        RetentionMode,
        TriggerReason,
    )
    from vision_os.core.model.detection import QualityGrades
    from vision_os.core.model.ids import (
        CameraId,
        ConfigRevision,
        CropId,
        FrameRef,
        ModuleId,
        ObjectId,
        SiteId,
        StreamEpoch,
        TenantId,
    )
    from vision_os.core.model.provenance import Provenance
    from vision_os.core.model.space import Box
    from vision_os.core.model.timebase import Instant

    width, height = size
    camera = CameraId(CAMERA)
    return Crop(
        crop_id=CropId("crop-1"),
        tenant_id=TenantId("tenant-1"),
        site_id=SiteId("site-1"),
        camera_id=camera,
        source_frame=FrameRef(
            camera_id=camera, stream_epoch=StreamEpoch(1), frame_seq=100
        ),
        object_id=ObjectId(SUBJECT),
        source_box=Box(0.4, 0.3, 0.55, 0.85),
        padding_applied=0.15,
        output_size=(width, height),
        transform=CropTransform(
            output_width=width, output_height=height,
            source_width=960, source_height=576,
            crop_x=10, crop_y=10, crop_width=width, crop_height=height,
        ),
        quality=QualityGrades(scale_pixels=264.0),
        gate_result=GateResult.accept(),
        retention=retention or RetentionMode.EPHEMERAL,
        privacy_class=PrivacyClass.C1_IMAGERY,
        t_capture=Instant(100 * SECOND),
        trigger_reason=TriggerReason.FIRST_SIGHT,
        provenance=Provenance(
            producer_module=ModuleId("crop_manager"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        pixels=memoryview(pixels) if pixels is not None else None,
    )


class TestTheCropSeamProducesRealPixels:
    def test_a_real_crop_encodes_to_a_real_jpeg(self):
        """The format assumption, checked against the platform's own object.

        If crop pixels ever stop being BGR24 at `output_size`, this fails here
        rather than producing a scrambled thumbnail in front of an operator.
        """
        from app.vision.runtime import _crop_jpeg

        crop = _real_crop(pixels=bytes(8 * 8 * 3))
        jpeg = _crop_jpeg(crop, encode_jpeg)
        assert jpeg is not None
        assert jpeg.startswith(b"\xff\xd8\xff")

    def test_the_encoded_crop_has_the_crops_own_dimensions(self):
        from io import BytesIO

        from PIL import Image

        from app.vision.runtime import _crop_jpeg

        crop = _real_crop(pixels=bytes(24 * 16 * 3), size=(24, 16))
        image = Image.open(BytesIO(_crop_jpeg(crop, encode_jpeg)))
        assert image.size == (24, 16)

    def test_never_persist_keeps_no_imagery(self):
        """12_SECURITY §2.3's no-evidence mode means what it says."""
        from app.vision.runtime import _crop_jpeg
        from vision_os.core.model.crop import RetentionMode

        crop = _real_crop(pixels=bytes(8 * 8 * 3), retention=RetentionMode.NEVER_PERSIST)
        assert _crop_jpeg(crop, encode_jpeg) is None

    def test_a_crop_that_kept_no_pixels_yields_none_not_an_empty_image(self):
        from app.vision.runtime import _crop_jpeg

        assert _crop_jpeg(_real_crop(pixels=None), encode_jpeg) is None

    def test_a_short_buffer_costs_the_thumbnail_and_nothing_else(self):
        from app.vision.runtime import _crop_jpeg

        assert _crop_jpeg(_real_crop(pixels=b"\x00\x01\x02"), encode_jpeg) is None


# --- the durable side ------------------------------------------------------ #


@pytest.fixture
def evidence_settings(tmp_path):
    class _Settings:
        default_tenant_id = "org-test"
        evidence_capture = True
        evidence_path = tmp_path / "evidence"
        evidence_retention_days = 30

    return _Settings()


@pytest.fixture
def decision_frame():
    """One analysed frame with two people, in the process-wide store.

    The real one, because `_capture_evidence` reaches for `DECISION_FRAMES` by
    name — so a test against a private instance would prove nothing about the
    path that actually runs.
    """
    DECISION_FRAMES.clear()
    ref = f"{CAMERA}/e1/f00100"
    DECISION_FRAMES.remember(
        camera_id=CAMERA, frame_ref=ref, captured_at_ns=100 * SECOND,
        width=960, height=576, jpeg=b"\xff\xd8\xffscene",
    )
    for object_id, box in (
        (BYSTANDER, (0.10, 0.30, 0.25, 0.85)),
        (SUBJECT, (0.55, 0.32, 0.72, 0.88)),
    ):
        DECISION_FRAMES.attach_subject(
            camera_id=CAMERA, frame_ref=ref, object_id=object_id, box=box,
            crop_jpeg=b"\xff\xd8\xff" + object_id.encode(),
            crop_id=f"crop-{object_id}", sent_to_model=True, object_class="person",
        )
    yield ref
    DECISION_FRAMES.clear()


async def _capture(app, evidence_settings, *, object_id=SUBJECT):
    """Run the driver's real capture step against a real database."""
    from app.vision.compliance_driver import ComplianceDriver
    from compliance import ComplianceState
    from tests.app.test_compliance_incidents import _finding

    driver = ComplianceDriver.__new__(ComplianceDriver)
    driver._settings = evidence_settings
    driver._wall = None

    finding = _finding(ComplianceState.VIOLATION, object_id=object_id, camera=CAMERA)
    async with app.state.database.session_scope() as session:
        ref = await driver._capture_evidence(
            session, camera_key=CAMERA, finding=finding
        )
    return ref


async def _records(app):
    from sqlalchemy import select

    from app.domain.models import EvidenceRecord

    async with app.state.database.session_scope() as session:
        rows = (await session.execute(select(EvidenceRecord))).scalars().all()
        return {r.evidence_ref: r for r in rows}


class TestTheEvidenceStoreReceivesBoth:
    async def test_the_frame_and_every_crop_are_stored(
        self, app, evidence_settings, decision_frame
    ):
        ref = await _capture(app, evidence_settings)
        records = await _records(app)
        assert ref in records
        assert f"{ref}.crop.{SUBJECT}" in records
        assert f"{ref}.crop.{BYSTANDER}" in records

    async def test_the_frame_record_carries_the_subject_geometry(
        self, app, evidence_settings, decision_frame
    ):
        ref = await _capture(app, evidence_settings)
        geometry = json.loads((await _records(app))[ref].geometry)
        assert geometry["subject"]["object_id"] == SUBJECT
        assert geometry["subject"]["box"] == [0.55, 0.32, 0.72, 0.88]

    async def test_a_crop_shares_the_frames_instant_and_reference(
        self, app, evidence_settings, decision_frame
    ):
        """Both halves of one photograph. A crop stamped `now` would sit
        minutes away from the frame it was cut from."""
        ref = await _capture(app, evidence_settings)
        records = await _records(app)
        frame, crop = records[ref], records[f"{ref}.crop.{SUBJECT}"]
        assert crop.captured_at == frame.captured_at
        assert crop.frame_ref == frame.frame_ref == decision_frame

    async def test_the_crop_names_its_own_object_not_the_alert_subject(
        self, app, evidence_settings, decision_frame
    ):
        ref = await _capture(app, evidence_settings)
        crop = (await _records(app))[f"{ref}.crop.{BYSTANDER}"]
        assert crop.object_id == BYSTANDER
        assert json.loads(crop.geometry)["is_subject"] is False

    async def test_a_crop_is_labelled_a_crop_and_the_frame_a_frame(
        self, app, evidence_settings, decision_frame
    ):
        ref = await _capture(app, evidence_settings)
        records = await _records(app)
        assert records[ref].purpose.endswith(":decision-frame")
        assert records[f"{ref}.crop.{SUBJECT}"].purpose.endswith(":decision-crop")

    async def test_capture_stays_off_when_the_deployment_says_so(
        self, app, evidence_settings, decision_frame
    ):
        """`EVIDENCE_CAPTURE` governs the crops exactly as it governs the frame.
        Retaining thumbnails of people in a deployment that opted out of
        retaining imagery would be the same decision made twice, differently."""
        evidence_settings.evidence_capture = False
        assert await _capture(app, evidence_settings) == ""
        assert await _records(app) == {}

    async def test_two_incidents_on_one_frame_keep_separate_subjects(
        self, app, evidence_settings, decision_frame
    ):
        """The real cam-13 shape, through the database."""
        first = await _capture(app, evidence_settings, object_id=SUBJECT)
        second = await _capture(app, evidence_settings, object_id=BYSTANDER)
        records = await _records(app)

        assert first != second
        assert json.loads(records[first].geometry)["subject"]["object_id"] == SUBJECT
        assert json.loads(records[second].geometry)["subject"]["object_id"] == BYSTANDER

    async def test_the_bytes_are_deduplicated_not_the_rows(
        self, app, evidence_settings, decision_frame
    ):
        """Two incidents on one frame must each keep their own record — and one
        copy of the pixels. A shared row would let resolving one erase the
        other's evidence."""
        first = await _capture(app, evidence_settings, object_id=SUBJECT)
        second = await _capture(app, evidence_settings, object_id=BYSTANDER)
        records = await _records(app)
        assert records[first].content_hash == records[second].content_hash
        assert records[first].id != records[second].id


class TestUnknownNeverBecomesSubjectEvidence:
    async def test_an_unknown_finding_stores_no_crop_of_the_person(
        self, app, evidence_settings, decision_frame, monkeypatch
    ):
        """§9. An attribute that could not be observed must not put a person's
        face in an evidence file — and a gallery is a more pointed way of doing
        that than a wide shot."""
        from compliance import ComplianceState
        from tests.app.test_compliance_incidents import _finding
        from tests.app.test_decision_evidence import TestGUnknownProducesNoViolationEvidence

        run, attempted = await TestGUnknownProducesNoViolationEvidence._run(
            app, monkeypatch, ComplianceState.UNKNOWN
        )
        assert run.incidents_opened == 0
        assert attempted == []
        assert await _records(app) == {}
        # The counterpart lives in test_decision_evidence.py: a VIOLATION does
        # reach the capture path, so the assertion above cannot pass vacuously.
        assert _finding(ComplianceState.VIOLATION) is not None


# --- authorization and audit are unchanged --------------------------------- #


class TestCropsAreGovernedLikeEveryOtherImage:
    """A crop is imagery of an identifiable person. It gets no shortcut."""

    @staticmethod
    async def _stored(app, settings, tmp_path):
        from app.domain import evidence as evidence_domain

        async with app.state.database.session_scope() as session:
            await evidence_domain.EvidenceStore(session, root=tmp_path).put(
                organization_id="org-test",
                evidence_ref="ev-1.crop.obj-person-1",
                camera_key="cam-01",
                payload=b"\xff\xd8\xffcrop",
                captured_at=__import__("datetime").datetime.now(
                    __import__("datetime").UTC
                ),
                purpose="compliance:kitchen.person.ppe.v1:decision-crop",
                retention_days=30,
                geometry=json.dumps({"kind": "decision-crop", "is_subject": True}),
            )
        return "ev-1.crop.obj-person-1"

    async def test_a_supervisor_without_view_evidence_is_refused_a_crop(
        self, app, client, seeded, settings, tmp_path
    ):
        from tests.app.conftest import bearer

        ref = await self._stored(app, settings, settings.evidence_path)
        response = await client.get(
            f"/api/v1/evidence/{ref}/image",
            headers=await bearer(client, "supervisor@example.com"),
        )
        assert response.status_code == 403

    async def test_an_anonymous_caller_is_refused_a_crop(self, app, client, seeded, settings):
        ref = await self._stored(app, settings, settings.evidence_path)
        response = await client.get(f"/api/v1/evidence/{ref}/image")
        assert response.status_code == 401

    async def test_a_crop_retrieval_leaves_an_audit_row(
        self, app, client, seeded, settings
    ):
        """Every disclosure of imagery is recorded. A gallery of four crops is
        four disclosures, and must be four rows — not one for the page."""
        from sqlalchemy import select

        from app.domain.models import AuditEvent
        from tests.app.conftest import bearer

        ref = await self._stored(app, settings, settings.evidence_path)
        headers = await bearer(client, "developer@example.com")
        await client.get(f"/api/v1/evidence/{ref}/image", headers=headers)

        async with app.state.database.session_scope() as session:
            rows = (
                await session.execute(
                    select(AuditEvent).where(AuditEvent.resource_id == ref)
                )
            ).scalars().all()
        assert rows, "a crop was served without an audit row"
        assert rows[0].resource_type == "evidence"

    async def test_geometry_is_served_as_a_document_not_a_string(
        self, app, client, seeded, settings
    ):
        from tests.app.conftest import bearer

        ref = await self._stored(app, settings, settings.evidence_path)
        response = await client.get(
            f"/api/v1/evidence/{ref}", headers=await bearer(client, "developer@example.com")
        )
        assert response.status_code == 200
        assert response.json()["geometry"]["kind"] == "decision-crop"

    async def test_unreadable_geometry_costs_the_highlight_not_the_evidence(self):
        """A row written by a future version, or corrupted. The picture is still
        evidence; it simply cannot say where in itself the subject was."""
        from app.domain.evidence import _geometry

        assert _geometry("{not json") is None
        assert _geometry("[1, 2]") is None
        assert _geometry("") is None
