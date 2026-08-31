"""P9.5 live-collection tooling — dedup, agreement, adjudication.

No network and no camera: the collector's *configuration* is tested, its
transport is not. A test that needs a DVR is a test that stops running.
"""

from __future__ import annotations

import pytest

from tools.p9_dataset.agreement import (
    Adjudication,
    agreement,
    apply_adjudications,
    disagreements,
    pair_annotations,
    summary,
)
from tools.p9_dataset.dedupe import FrameHash, dhash, find_duplicates, hamming
from tools.p9_dataset.schema import (
    AttributeState,
    LabelProvenance,
    Observability,
    QualityStatus,
    Region,
    RegionAnnotation,
    SubjectAnnotation,
)


def region(r=Region.HEAD, obs=Observability.VISIBLE, state=AttributeState.PRESENT):
    return RegionAnnotation(region=r, observability=obs, state=state)


def annot(sample_id, annotator, regions=None, camera="cam-11"):
    return SubjectAnnotation(
        sample_id=sample_id,
        subject_id="p1",
        session_id="s1",
        camera_id=camera,
        frame_id="f1",
        source_video="",
        box=(0.1, 0.1, 0.5, 0.9),
        regions=regions or (region(),),
        label_provenance=LabelProvenance.HUMAN_VERIFIED,
        box_provenance=LabelProvenance.HUMAN_VERIFIED,
        annotator=annotator,
        annotated_at="2026-08-27",
        quality_status=QualityStatus.ACCEPTED,
    )


class TestDeduplication:
    def test_identical_images_hash_identically(self, tmp_path):
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        Image.fromarray(pixels).save(a, quality=95)
        Image.fromarray(pixels).save(b, quality=95)
        assert hamming(dhash(a), dhash(b)) == 0

    def test_a_duplicate_is_only_a_duplicate_when_adjacent_in_time(self):
        """The rule that protects hard cases.

        Two identical hashes far apart in a session are two genuine observations
        of a recurring condition — an empty kitchen at 09:00 and at 14:00 — and
        collapsing them would understate how often the cameras see nothing.
        """
        near = [FrameHash(f"p{i}", "cam-11", "s1", i, 0b1010) for i in range(2)]
        assert len(find_duplicates(near)) == 1

        far = [
            FrameHash("p0", "cam-11", "s1", 0, 0b1010),
            FrameHash("p9", "cam-11", "s1", 99, 0b1010),
        ]
        assert find_duplicates(far) == {}

    def test_frames_from_different_cameras_are_never_duplicates(self):
        frames = [
            FrameHash("a", "cam-11", "s1", 0, 0b1010),
            FrameHash("b", "cam-12", "s1", 0, 0b1010),
        ]
        assert find_duplicates(frames) == {}

    def test_frames_from_different_sessions_are_never_duplicates(self):
        frames = [
            FrameHash("a", "cam-11", "s1", 0, 0b1010),
            FrameHash("b", "cam-11", "s2", 0, 0b1010),
        ]
        assert find_duplicates(frames) == {}

    def test_a_dissimilar_neighbour_survives(self):
        frames = [
            FrameHash("a", "cam-11", "s1", 0, 0x0000000000000000),
            FrameHash("b", "cam-11", "s1", 1, 0xFFFFFFFFFFFFFFFF),
        ]
        assert find_duplicates(frames) == {}


class TestIndependence:
    def test_two_passes_by_one_person_are_refused(self):
        """Self-consistency is not inter-annotator agreement."""
        with pytest.raises(ValueError, match="self-consistency"):
            pair_annotations([annot("x", "alice")], [annot("x", "alice")])

    def test_two_annotators_pair_normally(self):
        assert len(pair_annotations([annot("x", "alice")], [annot("x", "bob")])) == 1

    def test_unmatched_samples_are_dropped_not_guessed(self):
        assert pair_annotations([annot("x", "alice")], [annot("y", "bob")]) == []


class TestAgreement:
    def test_disagreement_is_recorded_per_field(self):
        a = annot("x", "alice", (region(state=AttributeState.PRESENT),))
        b = annot("x", "bob", (region(state=AttributeState.ABSENT),))
        found = disagreements([(a, b)])
        assert len(found) == 1
        assert found[0].field_name == "state"
        assert {found[0].value_a, found[0].value_b} == {"present", "absent"}

    def test_observability_and_state_disagree_separately(self):
        a = annot("x", "alice", (region(),))
        b = annot(
            "x",
            "bob",
            (
                region(
                    obs=Observability.NOT_VISIBLE, state=AttributeState.NOT_EVALUATED
                ),
            ),
        )
        assert {d.field_name for d in disagreements([(a, b)])} == {
            "observability",
            "state",
        }

    def test_a_rate_is_suppressed_below_support(self):
        """A percentage from four comparisons is noise wearing a decimal point."""
        head = agreement([(annot("x", "alice"), annot("x", "bob"))])["head.state"]
        assert head["rate"] is None
        assert "below 20" in head["rate_suppressed_reason"]

    def test_there_is_no_single_headline_number(self):
        report = agreement([(annot("x", "alice"), annot("x", "bob"))])
        assert set(report) == {
            f"{r.value}.{f}" for r in Region for f in ("observability", "state")
        }

    def test_summary_reports_unresolved_disagreements(self):
        a = annot("x", "alice", (region(state=AttributeState.PRESENT),))
        b = annot("x", "bob", (region(state=AttributeState.ABSENT),))
        out = summary([(a, b)])
        assert out["disagreements"] == 1
        assert out["unresolved"] == 1
        assert out["median_rate_where_reportable"] is None


class TestAdjudication:
    def _adj(self, **kw):
        base = {
            "sample_id": "x",
            "region": Region.HEAD,
            "annotator_a": "alice",
            "annotator_b": "bob",
            "observability_a": "visible",
            "observability_b": "visible",
            "state_a": "present",
            "state_b": "absent",
            "adjudicator": "carol",
            "resolved_observability": "visible",
            "resolved_state": "present",
            "reason": "hairnet visible under the light",
        }
        base.update(kw)
        return Adjudication(**base)

    def test_a_reason_is_mandatory(self):
        with pytest.raises(ValueError, match="without a reason"):
            self._adj(reason="")

    def test_an_original_annotator_cannot_adjudicate(self):
        """A tie cannot be broken by a player."""
        with pytest.raises(ValueError, match="cannot be broken by a player"):
            self._adj(adjudicator="alice")

    def test_both_originals_are_preserved(self):
        entry = self._adj()
        assert entry.state_a == "present"
        assert entry.state_b == "absent"

    def test_applying_it_marks_the_sample_adjudicated(self):
        sample = annot("x", "alice", (region(state=AttributeState.ABSENT),))
        out = apply_adjudications([sample], [self._adj()])[0]
        assert out.label_provenance is LabelProvenance.HUMAN_ADJUDICATED
        assert out.region_of(Region.HEAD).state is AttributeState.PRESENT
        assert "carol" in out.region_of(Region.HEAD).note

    def test_an_adjudication_cannot_create_an_illegal_state(self):
        """The schema invariant holds through adjudication too."""
        sample = annot("x", "alice")
        with pytest.raises(ValueError, match="cannot carry a decided attribute"):
            apply_adjudications(
                [sample],
                [
                    self._adj(
                        resolved_observability="not_visible", resolved_state="absent"
                    )
                ],
            )


class TestCollectorConfiguration:
    def test_the_uri_is_redacted(self):
        """No credential may appear in a manifest, a log or a session record."""
        from app.vision.sources.rtsp import RtspCameraConfig

        config = RtspCameraConfig(
            camera_id="cam-11",
            host="example.test",
            channel=11,
            username="operator",
            credential_ref="env:CCTV_PASSWORD",
        )
        redacted = config.redacted_uri()
        assert "operator" not in redacted
        assert "channel=11" in redacted
        assert "***" in redacted

    def test_the_collector_holds_no_password(self):
        """Structural: the config carries a REFERENCE, never a secret."""
        from dataclasses import fields

        from app.vision.sources.rtsp import RtspCameraConfig

        names = {f.name for f in fields(RtspCameraConfig)}
        assert "credential_ref" in names
        assert "password" not in names

    def test_it_cannot_reach_production_decision_code(self):
        """Read-only by construction.

        The collector imports no compliance engine, no registry and no alerting.
        A data acquisition tool that could change a verdict is not a data
        acquisition tool.
        """
        import ast
        from pathlib import Path

        source = Path("tools/p9_dataset/collect.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert "compliance" not in imported
        for banned in ("compliance_driver", "domain.incidents", "ComplianceEvaluator"):
            assert banned not in source
