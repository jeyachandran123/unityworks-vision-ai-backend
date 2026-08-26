"""An alert's picture must be the picture the decision was made on.

Measured on the running product before this was fixed, camera 13:

    attribute observed at   14:04:40Z   hand_covering = none
    incident opened at      14:05:29Z
    evidence frame stamped  14:05:17Z   ← 37 s after the observation

`_capture_evidence` took the camera wall's *current* JPEG, so an operator was
shown an empty kitchen underneath a true statement about a person who had been
there a minute earlier. Everything needed to do better already existed —
`Crop` carries `source_frame`, `source_box` and `object_id`; `EvidenceStore.put`
already accepts `frame_ref` and `captured_at` — except the pixels, which every
holder had let go of by the time the incident opened.

These tests pin the join: **subject + observation instant → the frame that
subject was seen in**, never a later one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.vision.decision_frames import (
    MAX_FRAMES_PER_CAMERA,
    DecisionFrameStore,
)

CAMERA = "cam-12"
PERSON = "obj-person-1"
OTHER = "obj-person-2"
SECOND = 1_000_000_000


def _frame(store, *, at_s: int, subject: str | None = PERSON, camera: str = CAMERA,
           jpeg: bytes | None = None, box=(0.3, 0.4, 0.4, 0.8)) -> str:
    ref = f"{camera}/e1/f{at_s:05d}"
    store.remember(
        camera_id=camera,
        frame_ref=ref,
        captured_at_ns=at_s * SECOND,
        width=960,
        height=576,
        jpeg=jpeg if jpeg is not None else b"\xff\xd8\xff" + f"frame{at_s}".encode(),
    )
    if subject:
        store.attach_subject(
            camera_id=camera, frame_ref=ref, object_id=subject, box=box,
            sent_to_model=True,
        )
    return ref


@pytest.fixture
def store() -> DecisionFrameStore:
    return DecisionFrameStore()


class TestAObservationFrameIsRetained:
    def test_an_analysed_frame_is_kept_with_its_capture_time(self, store):
        ref = _frame(store, at_s=100)
        frame = store.get(CAMERA, ref)
        assert frame is not None
        assert frame.captured_at_ns == 100 * SECOND
        assert frame.jpeg.startswith(b"\xff\xd8\xff")

    def test_the_subject_box_is_kept_with_the_frame(self, store):
        """§11 needs this to highlight the right person; it also proves the
        crop and the frame agree about where the object was."""
        ref = _frame(store, at_s=100, box=(0.11, 0.22, 0.33, 0.44))
        frame = store.get(CAMERA, ref)
        assert frame.subjects[PERSON].box == (0.11, 0.22, 0.33, 0.44)
        assert frame.subjects[PERSON].sent_to_model is True

    def test_a_frame_nobody_was_cut_from_holds_no_subjects(self, store):
        ref = _frame(store, at_s=100, subject=None)
        assert store.get(CAMERA, ref).subjects == {}

    def test_attaching_to_an_unknown_frame_reports_failure(self, store):
        """Rather than inventing a frame to hang the subject on."""
        assert store.attach_subject(
            camera_id=CAMERA, frame_ref="never/seen", object_id=PERSON,
            box=(0, 0, 1, 1),
        ) is False


class TestBIncidentUsesObservationEvidence:
    def test_the_frame_returned_is_the_one_the_subject_was_seen_in(self, store):
        first = _frame(store, at_s=100)
        _frame(store, at_s=110, subject=OTHER)      # someone else, later
        chosen = store.nearest_before(
            CAMERA, PERSON, 105 * SECOND, tolerance_ns=30 * SECOND
        )
        assert chosen is not None
        assert chosen.frame_ref == first

    def test_a_different_persons_frame_is_never_returned(self, store):
        _frame(store, at_s=100, subject=OTHER)
        assert store.nearest_before(
            CAMERA, PERSON, 105 * SECOND, tolerance_ns=30 * SECOND
        ) is None

    def test_another_cameras_frame_is_never_returned(self, store):
        _frame(store, at_s=100, camera="cam-13")
        assert store.latest_for_object(CAMERA, PERSON) is None


class TestCDelayedIncidentDoesNotChangeTheImage:
    def test_a_later_frame_is_never_chosen(self, store):
        """The defect, exactly.

        The room keeps being photographed after the observation. None of those
        later frames may be the evidence, however close they are.
        """
        chosen_ref = _frame(store, at_s=100)
        for later in (101, 102, 130, 160):
            _frame(store, at_s=later)

        chosen = store.nearest_before(
            CAMERA, PERSON, 100 * SECOND, tolerance_ns=30 * SECOND
        )
        assert chosen.frame_ref == chosen_ref, (
            "evidence moved to a frame taken after the decision"
        )

    def test_the_answer_does_not_drift_as_the_incident_is_delayed(self, store):
        """Whether the pass runs 1 s or 25 s late, the picture is the same."""
        expected = _frame(store, at_s=100)
        for later in range(101, 126):
            _frame(store, at_s=later, subject=OTHER)

        answers = {
            store.nearest_before(
                CAMERA, PERSON, 100 * SECOND, tolerance_ns=delay * SECOND
            ).frame_ref
            for delay in (1, 5, 15, 25, 30)
        }
        assert answers == {expected}

    def test_a_frame_far_older_than_the_observation_is_refused(self, store):
        """Better no picture than one from a different visit."""
        _frame(store, at_s=10)
        assert store.nearest_before(
            CAMERA, PERSON, 100 * SECOND, tolerance_ns=30 * SECOND
        ) is None

    def test_the_driver_never_falls_back_to_a_later_frame(self, store, monkeypatch):
        """The bug this suite did not catch the first time.

        `_decision_frame` used to fall back to `latest_for_object` whenever no
        earlier frame was in tolerance, and `latest_for_object` has no time
        constraint. On a real cam-13 incident that returned a frame **25.8 s
        after** its own observation, stored under the `decision-frame` label —
        the exact defect, through the fallback door.
        """
        from app.vision import compliance_driver as module

        monkeypatch.setattr(module, "DECISION_FRAMES", store, raising=False)
        monkeypatch.setitem(
            __import__("sys").modules["app.vision.decision_frames"].__dict__,
            "DECISION_FRAMES", store,
        )

        # Only frames AFTER the observation are retained.
        for at_s in (140, 150, 160):
            _frame(store, at_s=at_s)

        driver = module.ComplianceDriver.__new__(module.ComplianceDriver)

        class _Condition:
            outcome = type("O", (), {"value": "failed"})()
            observed_at_ns = 100 * SECOND

        class _Finding:
            conditions = (_Condition(),)
            subject = type("S", (), {"object_id": PERSON})()

        assert driver._decision_frame(camera_key=CAMERA, finding=_Finding()) is None, (
            "the driver reached forward in time for a picture"
        )


class TestDEvidenceTimestampMatchesTheDecision:
    def test_captured_at_is_the_frame_time_not_now(self, store):
        _frame(store, at_s=100)
        frame = store.nearest_before(
            CAMERA, PERSON, 105 * SECOND, tolerance_ns=30 * SECOND
        )
        captured = datetime.fromtimestamp(frame.captured_at_ns / 1e9, tz=UTC)
        assert captured == datetime.fromtimestamp(100, tz=UTC)
        assert captured < datetime.now(UTC)

    def test_the_frame_reference_travels_with_it(self, store):
        """So the stored evidence can be joined back to the observation
        instead of merely being believed."""
        ref = _frame(store, at_s=100)
        assert store.latest_for_object(CAMERA, PERSON).frame_ref == ref


class TestFRetentionAndBounds:
    def test_the_ring_is_bounded_by_frame_count(self, store):
        for at_s in range(MAX_FRAMES_PER_CAMERA + 25):
            _frame(store, at_s=at_s)
        assert store.stats()["frames_retained"] == MAX_FRAMES_PER_CAMERA
        assert store.evictions >= 25

    def test_the_ring_is_bounded_by_bytes_as_well(self, store):
        """Frame count alone is not a memory bound."""
        from app.vision import decision_frames

        big = b"\xff\xd8\xff" + b"x" * (4 * 1024 * 1024)
        for at_s in range(20):
            _frame(store, at_s=at_s, jpeg=big)
        assert store.stats()["retained_bytes"] <= decision_frames.MAX_BYTES_PER_CAMERA

    def test_the_oldest_frame_is_the_one_dropped(self, store):
        for at_s in range(MAX_FRAMES_PER_CAMERA + 5):
            _frame(store, at_s=at_s)
        assert store.get(CAMERA, f"{CAMERA}/e1/f00000") is None
        newest = f"{CAMERA}/e1/f{MAX_FRAMES_PER_CAMERA + 4:05d}"
        assert store.get(CAMERA, newest) is not None

    def test_cameras_are_bounded_independently(self, store):
        for at_s in range(MAX_FRAMES_PER_CAMERA + 10):
            _frame(store, at_s=at_s, camera="cam-11")
        _frame(store, at_s=1, camera="cam-14")
        assert store.get("cam-14", "cam-14/e1/f00001") is not None

    def test_an_empty_payload_is_not_retained(self, store):
        store.remember(camera_id=CAMERA, frame_ref="r", captured_at_ns=0,
                       width=960, height=576, jpeg=b"")
        assert store.stats()["frames_retained"] == 0


class TestGUnknownProducesNoViolationEvidence:
    """§6-G. An unobservable attribute must not put a person in an evidence file.

    Driven through the real driver against a real database, watching the
    capture call itself rather than inferring from the incident count.
    """

    @staticmethod
    async def _run(app, monkeypatch, state):
        from app.vision import compliance_driver as module
        from compliance.finding import ComplianceState  # noqa: F401
        from tests.app.test_compliance_incidents import _finding, apply

        attempted: list = []
        original = module.ComplianceDriver._capture_evidence

        async def _spy(self, session, *, camera_key, finding):
            attempted.append(str(finding.finding_id))
            return await original(self, session, camera_key=camera_key, finding=finding)

        monkeypatch.setattr(module.ComplianceDriver, "_capture_evidence", _spy)
        run = await apply(app, [_finding(state)])
        return run, attempted

    async def test_unknown_never_reaches_the_capture_path(self, app, monkeypatch):
        from compliance.finding import ComplianceState

        run, attempted = await self._run(app, monkeypatch, ComplianceState.UNKNOWN)
        assert run.incidents_opened == 0
        assert attempted == [], (
            "an unobservable attribute reached for a photograph of the person "
            "it could not see"
        )

    async def test_a_real_violation_does_reach_it(self, app, monkeypatch):
        """The counterpart, so the test above cannot pass by never running."""
        from compliance.finding import ComplianceState

        run, attempted = await self._run(app, monkeypatch, ComplianceState.VIOLATION)
        assert run.incidents_opened == 1
        assert len(attempted) == 1


class TestEncodingIsSafe:
    def test_a_short_payload_does_not_raise(self):
        from app.vision.decision_frames import encode_jpeg

        assert encode_jpeg(b"\x00\x01", 960, 576) == b""

    def test_zero_dimensions_do_not_raise(self):
        from app.vision.decision_frames import encode_jpeg

        assert encode_jpeg(b"\x00" * 100, 0, 0) == b""

    def test_a_real_payload_encodes_to_a_jpeg(self):
        from app.vision.decision_frames import encode_jpeg

        payload = bytes(16 * 16 * 3)
        jpeg = encode_jpeg(payload, 16, 16)
        assert jpeg.startswith(b"\xff\xd8\xff")
