"""An alert must point at the person it is about, and at nobody else.

The evidence path already showed the right *frame* (see
`test_decision_evidence.py`). It could not say **who in it**. On a real cam-13
incident that frame held two bare-headed chefs and two independent verdicts,
and the operator was shown one undifferentiated photograph of both — which is
one guess away from acting on the wrong person.

These tests pin the join one step further than the last suite did:

    subject + observation instant -> the frame that subject was seen in
                                  -> **the box that subject occupied in it**
                                  -> the crop that was sent to the model

Everything here comes from what was recorded at the crop seam at decision time.
Nothing recomputes a box from a stored image later: a detector run over the
JPEG would find *a* person, and "a person in the picture" is exactly the claim
this must never make.
"""

from __future__ import annotations

import json

import pytest

from app.vision.compliance_driver import MAX_CROPS_PER_INCIDENT, _exhibits
from app.vision.decision_frames import DecisionFrameStore

CAMERA = "cam-12"
SUBJECT = "obj-person-1"
BYSTANDER = "obj-person-2"
FURNITURE = "obj-chair-1"
SECOND = 1_000_000_000
REF = "ev-1"


def _jpeg(marker: bytes) -> bytes:
    return b"\xff\xd8\xff" + marker


def _frame(store, *, at_s: int = 100, camera: str = CAMERA):
    ref = f"{camera}/e1/f{at_s:05d}"
    store.remember(
        camera_id=camera, frame_ref=ref, captured_at_ns=at_s * SECOND,
        width=960, height=576, jpeg=_jpeg(b"scene"),
    )
    return ref


def _subject(store, frame_ref, object_id, *, box, crop=True,
             object_class="person", camera=CAMERA):
    store.attach_subject(
        camera_id=camera, frame_ref=frame_ref, object_id=object_id, box=box,
        crop_jpeg=_jpeg(object_id.encode()) if crop else None,
        crop_id=f"crop-{object_id}", sent_to_model=True,
        object_class=object_class,
    )


class _Finding:
    """The shape `_exhibits` reads. The real one carries far more."""

    def __init__(self, object_id: str = SUBJECT, class_id: str = "person") -> None:
        self.subject = type(
            "S", (), {"object_id": object_id, "class_id": class_id}
        )()


@pytest.fixture
def store() -> DecisionFrameStore:
    return DecisionFrameStore()


@pytest.fixture
def two_people(store):
    """One frame, two people, the alert about the one on the right."""
    ref = _frame(store)
    _subject(store, ref, BYSTANDER, box=(0.10, 0.30, 0.25, 0.85))
    _subject(store, ref, SUBJECT, box=(0.55, 0.32, 0.72, 0.88))
    return store.get(CAMERA, ref)


def _manifest(exhibits):
    return json.loads(exhibits.manifest)


# --- the subject's box comes from the decision, not from a later look ------- #


class TestSubjectBoxComesFromDecisionEvidence:
    def test_the_box_is_the_one_recorded_when_the_crop_was_cut(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        assert manifest["subject"]["object_id"] == SUBJECT
        assert manifest["subject"]["box"] == [0.55, 0.32, 0.72, 0.88]

    def test_the_box_belongs_to_the_finding_and_not_to_a_neighbour(self, two_people):
        """The failure mode this whole suite exists for: highlighting the wrong
        person is worse than highlighting nobody, because it is confident."""
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        assert manifest["subject"]["box"] != [0.10, 0.30, 0.25, 0.85]
        assert manifest["subject"]["is_subject"] is True

    def test_the_manifest_names_the_frame_the_box_is_measured_in(self, two_people):
        """A normalized box is meaningless without the image it indexes."""
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        assert manifest["frame"]["frame_ref"] == two_people.frame_ref
        assert manifest["frame"]["width"] == 960

    def test_a_subject_absent_from_the_frame_produces_no_evidence_at_all(self, store):
        """Rather than a gallery of the people who *were* there.

        If the finding's subject was not cut from this frame, nothing in it is
        evidence for this finding, and offering the bystanders would invite
        precisely the wrong reading.
        """
        ref = _frame(store)
        _subject(store, ref, BYSTANDER, box=(0.1, 0.3, 0.25, 0.85))
        exhibits = _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        assert exhibits.manifest == ""
        assert exhibits.crops == ()

    def test_a_frame_with_no_subjects_produces_no_evidence(self, store):
        ref = _frame(store)
        exhibits = _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        assert exhibits.manifest == ""

    def test_a_missing_decision_frame_is_survived(self):
        """The labelled context-frame fallback. No geometry, and no crash."""
        exhibits = _exhibits(None, finding=_Finding(), evidence_ref=REF)
        assert exhibits.manifest == ""
        assert exhibits.crops == ()


# --- the crop is the one the model was actually shown ---------------------- #


class TestCropComesFromTheSameDecision:
    def test_the_subject_crop_carries_the_pixels_recorded_at_the_seam(self, two_people):
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        subject_crop = next(c for c in exhibits.crops if c.object_id == SUBJECT)
        assert subject_crop.jpeg == _jpeg(SUBJECT.encode())

    def test_the_crop_carries_the_same_box_as_the_frame_manifest(self, two_people):
        """So the thumbnail and the highlight cannot describe different places."""
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        manifest = _manifest(exhibits)
        subject_crop = next(c for c in exhibits.crops if c.object_id == SUBJECT)
        assert json.loads(subject_crop.geometry)["box"] == manifest["subject"]["box"]

    def test_the_manifest_points_at_the_crop_it_stored(self, two_people):
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        manifest = _manifest(exhibits)
        assert manifest["subject"]["crop_ref"] == f"{REF}.crop.{SUBJECT}"
        assert any(c.evidence_ref == manifest["subject"]["crop_ref"] for c in exhibits.crops)

    def test_a_subject_whose_crop_was_not_retained_still_gets_its_box(self, store):
        """`NEVER_PERSIST`, or a frame past the per-frame crop ceiling. The
        highlight survives; only the thumbnail is lost."""
        ref = _frame(store)
        _subject(store, ref, SUBJECT, box=(0.5, 0.3, 0.7, 0.9), crop=False)
        exhibits = _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        manifest = _manifest(exhibits)
        assert manifest["subject"]["box"] == [0.5, 0.3, 0.7, 0.9]
        assert manifest["subject"]["crop_ref"] == ""
        assert exhibits.crops == ()

    def test_the_subjects_crop_is_kept_when_the_cap_bites(self, store):
        """The cap must never spend its budget on bystanders and drop the one
        image the alert is about — so the subject is stored first."""
        ref = _frame(store)
        for index in range(MAX_CROPS_PER_INCIDENT + 4):
            _subject(store, ref, f"obj-crowd-{index:02d}",
                     box=(0.01 * index, 0.3, 0.01 * index + 0.05, 0.8))
        # Deliberately rightmost, so ordering by position would drop it.
        _subject(store, ref, SUBJECT, box=(0.90, 0.3, 0.98, 0.9))

        exhibits = _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        assert len(exhibits.crops) == MAX_CROPS_PER_INCIDENT
        assert exhibits.crops[0].object_id == SUBJECT


# --- several people in one frame stay several people ----------------------- #


class TestMultiPersonSubjectIdentity:
    def test_each_person_is_labelled_separately(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        labels = {manifest["subject"]["label"]} | {c["label"] for c in manifest["context"]}
        assert labels == {"Person #1", "Person #2"}

    def test_labels_run_left_to_right_as_a_person_would_number_them(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        assert manifest["context"][0]["label"] == "Person #1"   # x1 = 0.10
        assert manifest["subject"]["label"] == "Person #2"      # x1 = 0.55

    def test_exactly_one_object_is_the_alert_subject(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        everyone = [manifest["subject"], *manifest["context"]]
        assert sum(1 for o in everyone if o["is_subject"]) == 1

    def test_two_findings_on_one_frame_each_name_their_own_subject(self, two_people):
        """The real cam-13 case: one frame, two bare-headed people, two
        incidents. They must not converge on the same highlight."""
        first = _manifest(_exhibits(two_people, finding=_Finding(SUBJECT), evidence_ref="ev-a"))
        second = _manifest(_exhibits(two_people, finding=_Finding(BYSTANDER), evidence_ref="ev-b"))

        assert first["subject"]["object_id"] == SUBJECT
        assert second["subject"]["object_id"] == BYSTANDER
        assert first["subject"]["box"] != second["subject"]["box"]
        assert first["subject"]["crop_ref"] != second["subject"]["crop_ref"]

    def test_each_persons_crop_is_their_own(self, two_people):
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        by_object = {c.object_id: c.jpeg for c in exhibits.crops}
        assert by_object[SUBJECT] == _jpeg(SUBJECT.encode())
        assert by_object[BYSTANDER] == _jpeg(BYSTANDER.encode())


# --- only what actually contributed to the verdict ------------------------- #


class TestUnrelatedObjectsAreNotDecisionEvidence:
    def test_a_chair_in_the_same_frame_is_not_presented_as_evidence(self, store):
        """§4. A PPE finding is about a person. Cropping a chair beside them and
        captioning it "decision evidence" misrepresents what was decided."""
        ref = _frame(store)
        _subject(store, ref, SUBJECT, box=(0.5, 0.3, 0.7, 0.9))
        _subject(store, ref, FURNITURE, box=(0.1, 0.5, 0.3, 0.9), object_class="chair")

        exhibits = _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        manifest = _manifest(exhibits)
        assert manifest["context"] == []
        assert [c.object_id for c in exhibits.crops] == [SUBJECT]

    def test_an_object_of_unrecorded_class_is_left_out_of_context(self, store):
        """It cannot be *asserted* to be a person, so it is not shown as one."""
        ref = _frame(store)
        _subject(store, ref, SUBJECT, box=(0.5, 0.3, 0.7, 0.9))
        _subject(store, ref, "obj-unknown", box=(0.1, 0.3, 0.3, 0.9), object_class="")

        manifest = _manifest(
            _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        )
        assert manifest["context"] == []

    def test_the_subject_itself_is_never_filtered_out_by_class(self, store):
        """Its class comes from the finding, which always carries one — so a
        subject whose crop-seam class went unrecorded still gets highlighted."""
        ref = _frame(store)
        _subject(store, ref, SUBJECT, box=(0.5, 0.3, 0.7, 0.9), object_class="")
        manifest = _manifest(
            _exhibits(store.get(CAMERA, ref), finding=_Finding(), evidence_ref=REF)
        )
        assert manifest["subject"]["object_id"] == SUBJECT
        assert manifest["subject"]["class"] == "person"

    def test_context_objects_come_only_from_the_analysed_frame(self, store):
        """A person cut from a *different* frame is not in this picture, and a
        box measured in one image drawn over another is fiction."""
        first = _frame(store, at_s=100)
        second = _frame(store, at_s=140)
        _subject(store, first, SUBJECT, box=(0.5, 0.3, 0.7, 0.9))
        _subject(store, second, BYSTANDER, box=(0.1, 0.3, 0.3, 0.9))

        manifest = _manifest(
            _exhibits(store.get(CAMERA, first), finding=_Finding(), evidence_ref=REF)
        )
        assert manifest["context"] == []


# --- the payload the gallery is built from --------------------------------- #


class TestGalleryPayload:
    def test_the_manifest_has_the_shape_the_ui_reads(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        assert manifest["kind"] == "decision-frame"
        assert set(manifest) == {"kind", "frame", "subject", "context"}
        assert set(manifest["subject"]) == {
            "object_id", "class", "label", "box", "is_subject", "sent_to_model",
            "crop_ref",
        }

    def test_each_crop_record_describes_itself(self, two_people):
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        geometry = json.loads(exhibits.crops[0].geometry)
        assert geometry["kind"] == "decision-crop"
        assert geometry["object_id"] == exhibits.crops[0].object_id
        assert geometry["is_subject"] is True

    def test_the_payload_carries_no_secret(self, two_people):
        """§5. Boxes, labels and handles — never a token, key or path."""
        exhibits = _exhibits(two_people, finding=_Finding(), evidence_ref=REF)
        blob = (exhibits.manifest + " ".join(c.geometry for c in exhibits.crops)).lower()
        for forbidden in ("password", "secret", "token", "api_key", "bearer",
                          "storage_ref", "c:\\", "/var/"):
            assert forbidden not in blob

    def test_a_box_is_four_finite_numbers(self, two_people):
        manifest = _manifest(_exhibits(two_people, finding=_Finding(), evidence_ref=REF))
        box = manifest["subject"]["box"]
        assert len(box) == 4
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in box)
