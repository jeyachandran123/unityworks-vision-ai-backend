"""P9 dataset tooling — the guards that make the corpus trustworthy.

These tests are the difference between a dataset that *claims* to prevent
leakage and one that *cannot* contain it. Each asserts a property the P9 report
depends on being true.
"""

from __future__ import annotations

import pytest

from tools.p9_dataset.manifest import (
    SplitRefused,
    assign_splits,
    build,
    digest_of,
    statistics,
    verify,
    write,
)
from tools.p9_dataset.migrate import migrate
from tools.p9_dataset.schema import (
    AttributeState,
    LabelProvenance,
    Observability,
    QualityStatus,
    Region,
    RegionAnnotation,
    Split,
    SubjectAnnotation,
)
from tools.p9_dataset.validate import errors_only, validate_manifest


def region(
    r=Region.HEAD,
    obs=Observability.VISIBLE,
    state=AttributeState.PRESENT,
    **kw,
) -> RegionAnnotation:
    return RegionAnnotation(region=r, observability=obs, state=state, **kw)


def subject(
    sample_id="s.1",
    subject_id="p1",
    session_id="sess1",
    regions=None,
    *,
    identity_verified=True,
    provenance=LabelProvenance.HUMAN_VERIFIED,
    quality=QualityStatus.ACCEPTED,
    split=Split.UNASSIGNED,
    frame_id="f1",
) -> SubjectAnnotation:
    return SubjectAnnotation(
        sample_id=sample_id,
        subject_id=subject_id,
        session_id=session_id,
        camera_id="cam-1",
        frame_id=frame_id,
        source_video="v.mp4",
        box=(0.1, 0.1, 0.5, 0.9),
        regions=regions or (region(),),
        label_provenance=provenance,
        box_provenance=LabelProvenance.HUMAN_VERIFIED,
        annotator="tester",
        annotated_at="2026-08-27",
        quality_status=quality,
        split=split,
        identity_verified=identity_verified,
    )


class TestTheCentralInvariant:
    """NOT_VISIBLE is not ABSENT — enforced at construction, not by review."""

    @pytest.mark.parametrize("state", [AttributeState.PRESENT, AttributeState.ABSENT])
    @pytest.mark.parametrize(
        "obs", [Observability.NOT_VISIBLE, Observability.UNCERTAIN]
    )
    def test_an_unobservable_region_cannot_carry_a_decided_state(self, obs, state):
        with pytest.raises(ValueError, match="cannot carry a decided attribute"):
            region(obs=obs, state=state)

    def test_an_observable_region_must_be_evaluated(self):
        """VISIBLE + NOT_EVALUATED is a skipped annotation, not an observation."""
        with pytest.raises(ValueError, match="must be evaluated"):
            region(obs=Observability.VISIBLE, state=AttributeState.NOT_EVALUATED)

    def test_the_legal_refusal_shape_is_accepted(self):
        got = region(obs=Observability.NOT_VISIBLE, state=AttributeState.NOT_EVALUATED)
        assert not got.is_ground_truth_violation
        assert not got.is_ground_truth_positive

    def test_a_violation_requires_having_seen_the_region(self):
        assert region(state=AttributeState.ABSENT).is_ground_truth_violation
        assert not region(
            obs=Observability.NOT_VISIBLE, state=AttributeState.NOT_EVALUATED
        ).is_ground_truth_violation


class TestHandsAreTwoRegions:
    def test_one_gloved_one_bare_is_expressible(self):
        """The example v1.0.0's single `hand_covering` could not describe."""
        got = subject(
            regions=(
                region(Region.LEFT_HAND, state=AttributeState.PRESENT),
                region(Region.RIGHT_HAND, state=AttributeState.ABSENT),
            )
        )
        assert got.region_of(Region.LEFT_HAND).state is AttributeState.PRESENT
        assert got.region_of(Region.RIGHT_HAND).is_ground_truth_violation

    def test_duplicate_regions_are_refused(self):
        with pytest.raises(ValueError, match="duplicate region"):
            subject(regions=(region(Region.HEAD), region(Region.HEAD)))


class TestLeakage:
    def test_a_group_cannot_span_two_splits(self):
        found = validate_manifest(
            [
                subject("a", "p1", "s1", split=Split.TRAIN),
                subject("b", "p1", "s1", split=Split.TEST, frame_id="f2"),
            ]
        )
        assert any(e.check == "group_leakage" for e in errors_only(found))

    def test_the_same_person_in_two_sessions_is_not_leakage(self):
        """Different days are genuinely different appearances."""
        found = validate_manifest(
            [
                subject("a", "p1", "s1", split=Split.TRAIN),
                subject("b", "p1", "s2", split=Split.TEST, frame_id="f2"),
            ]
        )
        assert not any(e.check == "group_leakage" for e in errors_only(found))

    def test_a_frame_cannot_span_two_splits(self):
        """Two people in one frame share lighting, camera and moment."""
        found = validate_manifest(
            [
                subject("a", "p1", "s1", split=Split.TRAIN, frame_id="shared"),
                subject("b", "p2", "s2", split=Split.TEST, frame_id="shared"),
            ]
        )
        assert any(e.check == "frame_leakage" for e in errors_only(found))

    def test_an_unverified_identity_collapses_the_session_into_one_group(self):
        """The pessimistic default that cannot produce silent leakage."""
        a = subject("a", "p1", "s1", identity_verified=False)
        b = subject("b", "p2", "s1", identity_verified=False)
        assert a.group_key == b.group_key
        verified = subject("c", "p1", "s1", identity_verified=True)
        assert verified.group_key != a.group_key


class TestGroundTruthDiscipline:
    def test_a_machine_label_cannot_enter_an_evaluation_split(self):
        found = validate_manifest(
            [subject(provenance=LabelProvenance.MACHINE_PROPOSED, split=Split.TEST)]
        )
        assert any(
            e.check == "machine_label_in_evaluation" for e in errors_only(found)
        )

    def test_a_machine_label_cannot_be_marked_accepted(self):
        found = validate_manifest(
            [
                subject(
                    provenance=LabelProvenance.MACHINE_PROPOSED,
                    quality=QualityStatus.ACCEPTED,
                )
            ]
        )
        assert any(e.check == "machine_label_accepted" for e in errors_only(found))

    def test_an_unresolved_disagreement_cannot_be_evaluated(self):
        """Adjudicate it or exclude it — never resolve it silently."""
        found = validate_manifest(
            [subject(quality=QualityStatus.DISPUTED, split=Split.TEST)]
        )
        assert any(e.check == "disputed_in_evaluation" for e in errors_only(found))

    def test_an_unattributable_label_is_refused(self):
        with pytest.raises(ValueError, match="names its annotator"):
            SubjectAnnotation(
                sample_id="x", subject_id="p", session_id="s", camera_id="c",
                frame_id="f", source_video="v", box=(0, 0, 1, 1),
                regions=(region(),),
                label_provenance=LabelProvenance.HUMAN_VERIFIED,
                box_provenance=LabelProvenance.HUMAN_VERIFIED,
                annotator="", annotated_at="2026-08-27",
            )


class TestSplitting:
    def test_it_refuses_a_split_it_cannot_make_honestly(self):
        with pytest.raises(SplitRefused, match="allocatable group"):
            assign_splits([subject("a", "p1", "s1"), subject("b", "p2", "s1",
                                                            identity_verified=False)])

    def test_allocation_is_whole_group_and_deterministic(self):
        people = [
            subject(f"s{i}", f"p{i}", f"sess{i}", frame_id=f"f{i}") for i in range(9)
        ]
        first = assign_splits(people, seed="fixed")
        second = assign_splits(people, seed="fixed")
        assert [s.split for s in first] == [s.split for s in second]
        assert not errors_only(validate_manifest(first))

    def test_hard_test_groups_are_chosen_not_sampled(self):
        people = [
            subject(f"s{i}", f"p{i}", f"sess{i}", frame_id=f"f{i}") for i in range(9)
        ]
        pinned = people[0].group_key
        out = assign_splits(people, hard_test_groups=[pinned])
        assert next(s for s in out if s.group_key == pinned).split is Split.HARD_TEST


class TestImmutability:
    def test_a_frozen_version_detects_mutation(self, tmp_path):
        people = [
            subject(f"s{i}", f"p{i}", f"sess{i}", frame_id=f"f{i}") for i in range(9)
        ]
        path = write(build(assign_splits(people), version="T-v1"), tmp_path / "m.json")
        ok, message = verify(path)
        assert ok, message

        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        document["samples"][0]["regions"][0]["state"] = "absent"
        path.write_text(json.dumps(document), encoding="utf-8")

        ok, message = verify(path)
        assert not ok
        assert "MUTATED" in message

    def test_the_digest_ignores_ordering(self):
        a = subject("a", "p1", "s1")
        b = subject("b", "p2", "s2", frame_id="f2")
        assert digest_of([a, b]) == digest_of([b, a])

    def test_a_dataset_that_fails_its_own_checks_gets_no_version(self):
        broken = [
            subject("a", "p1", "s1", split=Split.TRAIN),
            subject("b", "p1", "s1", split=Split.TEST, frame_id="f2"),
        ]
        with pytest.raises(ValueError, match="refusing to freeze"):
            build(broken, version="T-v1")


class TestKitchen01Migration:
    def test_every_subject_migrates(self):
        assert len(migrate()) == 43

    def test_head_labels_are_preserved_exactly(self):
        counts = {}
        for entry in migrate():
            head = entry.region_of(Region.HEAD)
            key = (head.observability.value, head.state.value)
            counts[key] = counts.get(key, 0) + 1
        assert counts[("visible", "present")] == 30
        assert counts[("not_visible", "not_evaluated")] == 13

    def test_it_records_that_the_boxes_came_from_a_detector(self):
        """The property that makes kitchen-01 unable to measure detection recall.

        v1.0.0 could only state it in prose; the schema now carries it per sample.
        """
        assert all(
            s.box_provenance is LabelProvenance.DETECTOR_DERIVED for s in migrate()
        )
        assert all(
            s.label_provenance is LabelProvenance.HUMAN_VERIFIED for s in migrate()
        )

    def test_hands_are_not_invented(self):
        """v1's single hand value cannot be split into left and right.

        Emitting no hand region is the honest outcome; stamping the joint value
        onto both hands would fabricate per-hand detail for the exact class the
        corpus is shortest of.
        """
        for entry in migrate():
            assert entry.region_of(Region.LEFT_HAND) is None
            assert entry.region_of(Region.RIGHT_HAND) is None
            if "hand_covering" in entry.note:
                assert "re-annotation required" in entry.note

    def test_identity_is_not_claimed(self):
        """`s0..s4` are detection-order slots, not people."""
        assert all(not s.identity_verified for s in migrate())

    def test_it_is_a_single_indivisible_group(self):
        assert len({s.group_key for s in migrate()}) == 1

    def test_it_still_contains_no_violation_example(self):
        """The finding P9 exists to fix, asserted so it cannot be lost."""
        assert statistics(migrate()).as_dict()["ground_truth_violations"] == 0


class TestSupportWarnings:
    def test_thin_classes_are_warned_not_hidden(self):
        found = validate_manifest([subject(split=Split.TEST)])
        thin = [e for e in found if e.check == "insufficient_support"]
        assert thin, "a class with one example must not be silently quotable"
        assert all(e.severity == "warning" for e in thin)
