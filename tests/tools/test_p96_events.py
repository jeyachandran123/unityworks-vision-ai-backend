"""P9.6 event-aware sampling — the sampler, the cooldown, the event-aware dedup.

No camera, no model file, no network. The `EventSampler` takes perception as two
callables, so every rule below is exercised against a scripted scene: a test that
needs a DVR is a test that stops running.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.p9_dataset.dedupe import (
    COUNT_TESTABLE_REASONS,
    PROTECTED_REASONS,
    FrameHash,
    Redundancy,
    audit_rescues,
    classification_summary,
    classify,
)
from tools.p9_dataset.events import (
    PRIORITY_OVERRIDES_COOLDOWN,
    EventSampler,
    SamplingConfig,
    SamplingReason,
    match_tracks,
)

PERSON = (0.10, 0.10, 0.30, 0.90)
OTHER = (0.60, 0.10, 0.80, 0.90)


def sampler(script, **overrides):
    """Drive the sampler from a list of `(hash, boxes)` per frame."""
    defaults = {
        "min_gap_seconds": 0.0,
        "hard_floor_seconds": 0.0,
        "heartbeat_seconds": 10_000.0,
        "detect_every_seconds": 0.0,
    }
    defaults.update(overrides)
    config = SamplingConfig(**defaults)
    return EventSampler(
        config,
        hash_of=lambda i: script[i][0],
        detect=lambda i: [(box, 0.9) for box in script[i][1]],
    )


def run(script, **overrides):
    engine = sampler(script, **overrides)
    return [engine.offer(i, float(i), i) for i in range(len(script))], engine


def reasons_of(decisions) -> list[str]:
    return [d.reason.value for d in decisions if d.accepted]


class TestSamplingReasons:
    def test_every_accepted_frame_carries_exactly_one_reason(self):
        """Phase 5's requirement, checked rather than assumed."""
        decisions, _ = run([(0, []), (0xFFFF, [PERSON]), (0xFFFF, [PERSON])])
        accepted = [d for d in decisions if d.accepted]
        assert accepted
        for decision in accepted:
            assert isinstance(decision.reason, SamplingReason)

    def test_a_rejected_frame_names_what_suppressed_it(self):
        """Phase 4 cannot measure the cooldown's cost without this."""
        decisions, _ = run(
            [(0, [PERSON])] * 6, min_gap_seconds=100.0, hard_floor_seconds=100.0
        )
        rejected = [d for d in decisions if not d.accepted]
        assert rejected
        assert all(d.suppressed_by for d in rejected)

    def test_priorities_are_unique_and_total(self):
        """Ties would make reason selection depend on iteration order."""
        priorities = [reason.priority for reason in SamplingReason]
        assert len(set(priorities)) == len(priorities)

    def test_the_reason_chosen_is_the_highest_priority_one_that_fired(self):
        decisions, _ = run([(0, []), (0xFFFF, [PERSON]), (0xFFFF, [PERSON])])
        for decision in decisions:
            if decision.accepted and len(decision.reasons) > 1:
                assert decision.reason is decision.reasons[0]
                assert decision.reason.priority == max(r.priority for r in decision.reasons)


class TestPPEBlindness:
    """Phase 3. The sampler detects CHANGE, never VIOLATION."""

    def test_no_reason_names_a_ppe_concept(self):
        vocabulary = " ".join(r.value for r in SamplingReason)
        for banned in ("hairnet", "glove", "mask", "cover", "ppe", "compliant", "violation"):
            assert banned not in vocabulary

    def test_perception_enters_through_exactly_two_callables(self):
        """Structural: there is no third argument a PPE signal could arrive on.

        A convention can be broken by the next caller; a constructor that accepts
        no such parameter cannot be.
        """
        import inspect

        parameters = inspect.signature(EventSampler.__init__).parameters
        assert set(parameters) == {"self", "config", "hash_of", "detect"}

    def test_the_module_references_no_ppe_signal(self):
        source = Path("tools/p9_dataset/events.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for banned in ("head_covering", "face_covering", "AttributeState", "Observability"):
            assert banned not in names


class TestHysteresis:
    """The defect the Phase 4 sweep exposed, pinned so it cannot return."""

    def test_detector_flicker_is_not_reported_as_traffic(self):
        """A box dropping out for one detection is not a person leaving.

        Measured on 226 s of recorded production footage, the naive version fired
        65 `PERSON_ENTERED` and 64 `PERSON_LEFT`. Nobody entered 65 times: the
        detector was stuttering, and because both events are high priority they
        punched through the cooldown and resampled an unchanged scene.
        """
        flicker = [(0xFF00, [PERSON]), (0xFF00, []), (0xFF00, [PERSON])] * 4
        _, engine = run(flicker)
        statistics = engine.statistics()
        assert statistics["by_reason"].get("person_left", 0) == 0

    def test_a_new_box_is_provisional_until_confirmed(self):
        decisions, _ = run([(0, []), (0, [PERSON]), (0, [PERSON])], track_min_hits=2)
        assert decisions[1].people == 0, "one sighting is not yet a person"
        assert decisions[2].people == 1

    def test_a_departure_is_only_reported_after_max_age(self):
        script = [(0, [PERSON])] * 3 + [(0, [])] * 4
        decisions, _ = run(script, track_min_hits=1, track_max_age=2)
        left = [i for i, d in enumerate(decisions) if d.accepted and d.reason is SamplingReason.PERSON_LEFT]
        assert left, "the person must eventually be reported as gone"
        assert left[0] >= 5, "and not on the first missed detection"

    def test_a_person_who_never_confirms_never_departs(self):
        """A single spurious detection must not generate an exit event."""
        decisions, _ = run([(0, []), (0, [PERSON]), (0, []), (0, []), (0, [])], track_min_hits=3)
        assert SamplingReason.PERSON_LEFT.value not in reasons_of(decisions)


class TestAssociation:
    def test_a_fast_walker_is_not_an_exit_and_an_entrance(self):
        """At 0.5 s between detections a walking person outruns their own box.

        Pure IoU association then reports one person leaving and another
        arriving. That does not change whether the frame is kept, but it
        corrupts the event distribution the report publishes.
        """
        far = (0.40, 0.10, 0.60, 0.90)
        current, matched, lost, _ = match_tracks(
            [_track(0, PERSON)],
            [(far, 0.9)],
            threshold=0.3,
            next_id=1,
            distance_diagonals=1.2,
        )
        assert matched and not lost
        assert current[0].track_id == 0

    def test_the_fallback_refuses_a_wildly_different_box(self):
        tiny = (0.40, 0.40, 0.42, 0.46)
        _, matched, lost, _ = match_tracks(
            [_track(0, PERSON)], [(tiny, 0.9)], threshold=0.3, next_id=1,
            distance_diagonals=1.2, area_ratio=2.0,
        )
        assert not matched and lost

    def test_the_fallback_can_be_disabled(self):
        far = (0.40, 0.10, 0.60, 0.90)
        _, matched, lost, _ = match_tracks(
            [_track(0, PERSON)], [(far, 0.9)], threshold=0.3, next_id=1,
            distance_diagonals=0.0,
        )
        assert not matched and lost

    def test_association_is_deterministic_under_tied_overlap(self):
        previous = [_track(0, (0.0, 0.0, 0.4, 1.0)), _track(1, (0.0, 0.0, 0.4, 1.0))]
        boxes = [((0.0, 0.0, 0.4, 1.0), 0.9), ((0.0, 0.0, 0.4, 1.0), 0.9)]
        first = match_tracks(previous, boxes, threshold=0.3, next_id=2)[0]
        second = match_tracks(previous, boxes, threshold=0.3, next_id=2)[0]
        assert [t.track_id for t in first] == [t.track_id for t in second]


def _track(track_id, box):
    from tools.p9_dataset.events import Track

    return Track(track_id=track_id, box=box, confidence=0.9, hits=5, confirmed=True)


class TestJitterSuppression:
    def test_a_boundary_crossing_needs_real_travel(self):
        """A person standing on a grid edge jitters across it every detection.

        Without a magnitude requirement that reads as repeated traversal of the
        kitchen, and the rescued-frame audit showed only 31 % of those rescues
        were corroborated.
        """
        left = (0.320, 0.10, 0.340, 0.90)
        right = (0.334, 0.10, 0.354, 0.90)
        script = [(0, [left]), (0, [right]), (0, [left]), (0, [right])]
        decisions, _ = run(script, track_min_hits=1, grid=3)
        assert SamplingReason.REGION_TRANSITION.value not in reasons_of(decisions)

    def test_a_real_traversal_still_fires(self):
        here = (0.05, 0.10, 0.25, 0.90)
        there = (0.70, 0.10, 0.90, 0.90)
        decisions, _ = run([(0, [here]), (0, [there])], track_min_hits=1, grid=3)
        assert SamplingReason.REGION_TRANSITION.value in reasons_of(decisions)

    def test_occlusion_uses_a_schmitt_trigger(self):
        """Two workers side by side must not emit an event per detection."""
        near = (0.28, 0.10, 0.48, 0.90)   # ~ engage threshold against PERSON
        apart = (0.30, 0.10, 0.50, 0.90)
        script = [
            (0, [PERSON, near]),
            (0, [PERSON, apart]),
            (0, [PERSON, near]),
            (0, [PERSON, apart]),
            (0, [PERSON, near]),
        ]
        _, engine = run(script, track_min_hits=1, edge_epsilon=0.0)
        fired = engine.statistics()["by_reason"].get("occlusion_changed", 0)
        assert fired <= 1, "the overlap boundary must not chatter"


class TestCooldown:
    def test_an_ordinary_event_waits_out_the_gap(self):
        script = [(0, [PERSON])] * 8
        decisions, _ = run(
            script, min_gap_seconds=4.0, hard_floor_seconds=4.0, track_min_hits=1
        )
        accepted = [d.timestamp for d in decisions if d.accepted]
        gaps = [b - a for a, b in zip(accepted, accepted[1:], strict=False)]
        assert all(gap >= 4.0 for gap in gaps)

    def test_a_high_priority_event_may_override_it(self):
        """A person walking into frame is the instant a cooldown must not eat."""
        script = [(0, []), (0, [PERSON]), (0, [PERSON, OTHER])]
        decisions, _ = run(
            script, min_gap_seconds=100.0, hard_floor_seconds=0.0, track_min_hits=1
        )
        assert decisions[2].accepted
        assert decisions[2].reason.priority >= PRIORITY_OVERRIDES_COOLDOWN

    def test_the_hard_floor_still_binds(self):
        script = [(0, []), (0, [PERSON]), (0, [PERSON, OTHER])]
        decisions, _ = run(
            script, min_gap_seconds=100.0, hard_floor_seconds=100.0, track_min_hits=1
        )
        assert not decisions[2].accepted
        assert decisions[2].suppressed_by == "cooldown"

    def test_suppression_is_counted(self):
        # The scene must actually change, or the frames are rejected as
        # `no_event` and never reach the cooldown at all.
        script = [(0xFFFF if i % 2 else 0x0000, [PERSON]) for i in range(8)]
        _, engine = run(
            script, min_gap_seconds=100.0, hard_floor_seconds=100.0, track_min_hits=1
        )
        assert engine.statistics()["suppressed"]["cooldown"] > 0


class TestCaps:
    def test_a_capped_reason_falls_through_to_a_lesser_one(self):
        """The cap stops one event type monopolising, not the frame being kept.

        Dropping the frame instead would discard evidence some *other* event
        also justified.
        """
        engine = sampler([(0, [PERSON])], max_per_reason=1)
        engine._by_reason[SamplingReason.PERSON_ENTERED] = 1
        decision = engine.offer(0, 0.0, 0)
        assert decision.accepted
        assert decision.reason is not SamplingReason.PERSON_ENTERED

    def test_all_reasons_capped_suppresses_the_frame(self):
        engine = sampler([(0, [PERSON])], max_per_reason=0)
        decision = engine.offer(0, 0.0, 0)
        assert not decision.accepted
        assert decision.suppressed_by == "reason_cap"

    def test_the_session_cap_is_absolute(self):
        _, engine = run([(0xFFFF * (i % 2), [PERSON]) for i in range(20)], max_samples=3)
        assert engine.accepted == 3
        assert engine.statistics()["suppressed"]["session_cap"] > 0

    def test_a_tighter_cap_buys_event_diversity(self):
        """The measured trade the sweep used to pick `max_per_reason`."""
        script = []
        for i in range(40):
            boxes = [PERSON] if i % 4 else [PERSON, OTHER]
            script.append((0xFFFF if i % 3 else 0, boxes))
        _, loose = run(script, max_per_reason=100, track_min_hits=1)
        _, tight = run(script, max_per_reason=2, track_min_hits=1)
        assert len(tight.statistics()["by_reason"]) >= len(loose.statistics()["by_reason"])


class TestBaseline:
    def test_the_session_opener_is_a_heartbeat_not_a_scene_change(self):
        """There is no earlier frame for the scene to have changed from."""
        decisions, _ = run([(0x1234, [])], heartbeat_seconds=45.0)
        assert decisions[0].accepted
        assert decisions[0].reason is SamplingReason.PERIODIC_HEARTBEAT

    def test_a_frozen_camera_still_produces_the_baseline(self):
        """cam-14's median consecutive distance is 0. It must not go silent."""
        frozen = [(0xABCD, []) for _ in range(40)]
        decisions, engine = run(frozen, heartbeat_seconds=5.0)
        assert engine.statistics()["baseline_triggered"] >= 5
        assert set(reasons_of(decisions)) == {SamplingReason.PERIODIC_HEARTBEAT.value}

    def test_the_heartbeat_stays_quiet_while_events_fire(self):
        """It measures from the last accepted sample, so a busy scene needs none."""
        busy = [(0xFFFF if i % 2 else 0x0000, [PERSON]) for i in range(30)]
        _, engine = run(busy, heartbeat_seconds=5.0, track_min_hits=1)
        statistics = engine.statistics()
        assert statistics["baseline_triggered"] < statistics["event_triggered"]

    def test_the_baseline_is_counted_separately(self):
        _, engine = run([(0xABCD, [])] * 20, heartbeat_seconds=3.0)
        statistics = engine.statistics()
        assert statistics["baseline_triggered"] + statistics["event_triggered"] == (
            statistics["frames_accepted"]
        )


class TestDeterminism:
    def test_the_same_stream_yields_the_same_decisions(self):
        script = [(0xFFFF if i % 3 else 0, [PERSON] if i % 2 else []) for i in range(60)]
        first, _ = run(script)
        second, _ = run(script)
        assert [(d.accepted, d.reason) for d in first] == [
            (d.accepted, d.reason) for d in second
        ]


class TestIdentityIsNeverInvented:
    """Phase 10. Tracker ids are an engineering aid, never a person."""

    def test_a_track_id_is_not_reused_across_a_gap(self):
        script = [(0, [PERSON])] * 3 + [(0, [])] * 6 + [(0, [PERSON])] * 3
        _, engine = run(script, track_min_hits=1, track_max_age=1)
        assert engine.statistics()["candidate_subject_tracks"] >= 2

    def test_the_statistic_is_named_a_candidate_track(self):
        _, engine = run([(0, [PERSON])] * 3)
        assert "candidate_subject_tracks" in engine.statistics()
        assert "subjects" not in engine.statistics()
        assert "people" not in engine.statistics()


class TestEventAwareDeduplication:
    """Phase 12. Similar and redundant are not the same thing."""

    def _frames(self, reasons, bits=0xAAAA):
        return [
            FrameHash(Path(f"f{i}.jpg"), "cam-11", "s1", i, bits, reason, people=1)
            for i, reason in enumerate(reasons)
        ]

    def test_a_protected_reason_survives_looking_identical(self):
        verdicts = classify(self._frames(["scene_changed", "person_entered"]))
        assert verdicts["f1.jpg"] is Redundancy.MEANINGFUL_CHANGE

    def test_an_unprotected_reason_is_removed(self):
        verdicts = classify(self._frames(["scene_changed", "person_moved"]))
        assert verdicts["f1.jpg"] is Redundancy.EXACT_DUPLICATE

    def test_exact_and_near_are_distinguished(self):
        frames = [
            FrameHash(Path("a.jpg"), "cam-11", "s1", 0, 0b1111, "scene_changed"),
            FrameHash(Path("b.jpg"), "cam-11", "s1", 1, 0b1111, "scene_changed"),
            FrameHash(Path("c.jpg"), "cam-11", "s1", 2, 0b1110, "scene_changed"),
        ]
        verdicts = classify(frames)
        assert verdicts["b.jpg"] is Redundancy.EXACT_DUPLICATE
        assert verdicts["c.jpg"] is Redundancy.NEAR_DUPLICATE

    def test_every_protected_reason_is_a_real_reason(self):
        vocabulary = {reason.value for reason in SamplingReason}
        assert PROTECTED_REASONS <= vocabulary
        assert COUNT_TESTABLE_REASONS <= PROTECTED_REASONS

    def test_the_p95_corpus_still_classifies_without_reasons(self):
        """Backwards compatibility: the wall-clock corpus carries no reasons."""
        frames = [
            FrameHash(Path(f"f{i}.jpg"), "cam-11", "s1", i, 0xAAAA) for i in range(3)
        ]
        verdicts = classify(frames)
        assert Redundancy.MEANINGFUL_CHANGE not in verdicts.values()

    def test_a_rescued_frame_remains_comparable(self):
        """It is a genuine observation, not an exemption from the rules."""
        verdicts = classify(
            self._frames(["scene_changed", "person_entered", "person_moved"])
        )
        assert verdicts["f2.jpg"] in (
            Redundancy.EXACT_DUPLICATE,
            Redundancy.NEAR_DUPLICATE,
        )

    def test_the_summary_accounts_for_every_frame(self):
        summary = classification_summary(
            classify(self._frames(["scene_changed", "person_entered", "person_moved"]))
        )
        assert (
            summary["unique"]
            + summary["exact_duplicates"]
            + summary["near_duplicates"]
            + summary["meaningful_change"]
            == summary["frames"]
        )
        assert summary["retained"] + summary["removed"] == summary["frames"]


class TestRescueAudit:
    """The protection rule is checked against evidence, not trusted."""

    def test_an_unchanged_count_refutes_a_population_claim(self):
        frames = [
            FrameHash(Path("a.jpg"), "cam-11", "s1", 0, 0xAAAA, "scene_changed", people=2),
            FrameHash(Path("b.jpg"), "cam-11", "s1", 1, 0xAAAA, "person_entered", people=2),
        ]
        audit = audit_rescues(frames)
        assert audit["corroborated_rate"] == 0.0

    def test_a_changed_count_corroborates_it(self):
        frames = [
            FrameHash(Path("a.jpg"), "cam-11", "s1", 0, 0xAAAA, "scene_changed", people=1),
            FrameHash(Path("b.jpg"), "cam-11", "s1", 1, 0xAAAA, "person_entered", people=2),
        ]
        assert audit_rescues(frames)["corroborated_rate"] == 1.0

    def test_occlusion_is_not_scored_by_person_count(self):
        """An unchanged count is what it PREDICTS; scoring it would manufacture
        a failure the instrument is not entitled to report."""
        frames = [
            FrameHash(Path("a.jpg"), "cam-11", "s1", 0, 0xAAAA, "scene_changed", people=2),
            FrameHash(Path("b.jpg"), "cam-11", "s1", 1, 0xAAAA, "occlusion_changed", people=2),
        ]
        audit = audit_rescues(frames)
        assert audit["not_count_testable"] == 1
        assert audit["corroborated_rate"] is None

    def test_an_unknown_count_is_never_scored_either_way(self):
        frames = [
            FrameHash(Path("a.jpg"), "cam-11", "s1", 0, 0xAAAA, "scene_changed"),
            FrameHash(Path("b.jpg"), "cam-11", "s1", 1, 0xAAAA, "person_entered"),
        ]
        audit = audit_rescues(frames)
        assert audit["count_unknown"] == 1
        assert audit["corroborated_rate"] is None


class TestCollectorContract:
    def test_a_sample_records_its_own_provenance(self):
        """Phase 19: camera, day, session, sequence, timestamp, reason."""
        from tools.p9_dataset.collect import CameraResult

        result = CameraResult(camera_id="cam-11", channel=11, redacted_uri="x")
        assert result.samples == []
        assert result.by_reason == {}
        assert result.candidate_subject_tracks == 0

    def test_the_collector_accepts_a_sampling_policy(self):
        import inspect

        from tools.p9_dataset.collect import collect, collect_camera

        assert "sampling" in inspect.signature(collect_camera).parameters
        assert "sampling" in inspect.signature(collect).parameters
        assert "period" in inspect.signature(collect).parameters

    def test_the_config_serialises_every_threshold(self):
        """A corpus must be traceable to the exact policy that produced it."""
        import dataclasses

        payload = SamplingConfig().as_dict()
        for field in dataclasses.fields(SamplingConfig):
            assert field.name in payload

    def test_the_interval_mode_survives(self):
        """A superseded strategy that still runs is evidence; one that only gets
        described is a claim."""
        import inspect

        from tools.p9_dataset.collect import main

        assert "interval" in inspect.getsource(main)


class TestFrameHandling:
    def test_the_frame_is_not_converted_until_something_needs_it(self):
        """Converting every decoded frame to RGB was the throughput bug.

        The sampler hashes every frame and detects on a fraction, so an eager
        1920x1080 colour-space conversion dominated the loop.
        """
        from tools.p9_dataset.collect import _LazyFrame

        class Counting:
            def __init__(self):
                self.calls = 0

            def to_image(self):
                self.calls += 1
                return _Convertible()

        source = Counting()
        lazy = _LazyFrame(source)
        assert source.calls == 0
        first, second = lazy.image, lazy.image
        assert first is second
        assert source.calls == 1, "and it is converted at most once"

    def test_both_hash_entry_points_agree(self, tmp_path):
        """`dhash_image` and `dhash` must not disagree about the same picture."""
        import numpy as np
        from PIL import Image

        from tools.p9_dataset.dedupe import dhash, dhash_image

        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        path = tmp_path / "frame.png"
        image = Image.fromarray(pixels)
        image.save(path)
        assert dhash_image(image) == dhash(path)


class _Convertible:
    def convert(self, _mode):
        return self


class TestSessionMetadata:
    def test_a_p95_session_yields_no_reasons_rather_than_an_error(self, tmp_path):
        """The corpus is heterogeneous by history; the loader reads both halves."""
        import json

        from tools.p9_dataset.dedupe import session_reasons

        (tmp_path / "session.json").write_text(
            json.dumps({"session_id": "s", "cameras": [{"camera_id": "cam-11"}]}),
            encoding="utf-8",
        )
        assert session_reasons(tmp_path) == {}

    def test_reasons_are_read_back_per_frame(self, tmp_path):
        import json

        from tools.p9_dataset.dedupe import session_reasons

        (tmp_path / "session.json").write_text(
            json.dumps(
                {
                    "session_id": "s",
                    "cameras": [
                        {
                            "camera_id": "cam-11",
                            "samples": [
                                {"file": "a.jpg", "sampling_reason": "person_entered"}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert session_reasons(tmp_path)["a.jpg"]["sampling_reason"] == "person_entered"

    def test_a_missing_session_record_is_not_fatal(self, tmp_path):
        from tools.p9_dataset.dedupe import session_reasons

        assert session_reasons(tmp_path) == {}


class TestDiversityFlags:
    def test_a_dominant_category_is_flagged_over(self):
        from tools.p9_dataset.quality import balance

        result = balance({"cam-12": 90, "cam-11": 4, "cam-13": 3, "cam-14": 3})
        assert result["categories"]["cam-12"]["flag"] == "OVERREPRESENTED"
        assert result["categories"]["cam-14"]["flag"] == "UNDERREPRESENTED"

    def test_an_even_spread_is_balanced(self):
        from tools.p9_dataset.quality import balance

        result = balance({"a": 25, "b": 25, "c": 25, "d": 25})
        assert {e["flag"] for e in result["categories"].values()} == {"BALANCED"}

    def test_an_empty_dimension_says_so(self):
        from tools.p9_dataset.quality import balance

        assert balance({})["total"] == 0

    def test_shares_sum_to_one(self):
        from tools.p9_dataset.quality import balance

        result = balance({"a": 7, "b": 11, "c": 2})
        assert abs(sum(e["share"] for e in result["categories"].values()) - 1.0) < 1e-6


class TestNoLabelsAreProduced:
    """Phase 16. P9.6 outputs unlabeled data, sampling metadata and provenance."""

    def test_the_review_queue_carries_no_ppe_state(self):
        """The strongest guard in the phase, checked against the real artefact.

        A candidate that acquired a `state` would be a machine proposal wearing
        the shape of ground truth, and every downstream metric computed from it
        would be the model scoring itself.
        """
        import json

        path = Path("datasets/p9-live/review_queue.json")
        if not path.exists():
            return
        queue = json.loads(path.read_text(encoding="utf-8"))
        for candidate in queue.get("candidates", []):
            assert "state" not in candidate
            assert "observability" not in candidate
            assert "head_covering" not in candidate
            assert "regions" not in candidate
            assert candidate["review_status"] == "awaiting_human_annotation"
            assert candidate["box_provenance"] == "detector_derived"

    def test_the_head_hint_is_geometry_and_says_so(self):
        import json

        path = Path("datasets/p9-live/review_queue.json")
        if not path.exists():
            return
        queue = json.loads(path.read_text(encoding="utf-8"))
        legal = {"located", "low_confidence", "not_located", "unsupported", "unknown"}
        for candidate in queue.get("candidates", []):
            assert candidate["head_observability_hint"] in legal
            assert candidate["hint_provenance"] == "machine_proposed"

    def test_every_candidate_carries_the_full_provenance_chain(self):
        """Phase 19: camera, day, session, sequence, timestamp."""
        import json

        path = Path("datasets/p9-live/review_queue.json")
        if not path.exists():
            return
        queue = json.loads(path.read_text(encoding="utf-8"))
        for candidate in queue.get("candidates", []):
            for field in (
                "camera_id",
                "session_id",
                "frame_id",
                "collection_day",
                "sequence",
                "offset_seconds",
                "sampling_reason",
            ):
                assert field in candidate, f"{candidate['candidate_id']} lacks {field}"


class TestImportBoundary:
    """Phase 20, widened from P9.5: the whole package, not one module."""

    def test_no_dataset_tool_can_reach_production_decision_code(self):
        offenders = {}
        for path in sorted(Path("tools/p9_dataset").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
            if "compliance" in imported:
                offenders[path.name] = imported
        assert offenders == {}

    def test_no_dataset_tool_names_a_verdict_symbol(self):
        banned = (
            "ComplianceEvaluator",
            "compliance_driver",
            "domain.incidents",
            "registry_bootstrap",
            "AlertPublisher",
            "PolicyDocument",
        )
        offenders = {}
        for path in sorted(Path("tools/p9_dataset").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            found = [name for name in banned if name in source]
            if found:
                offenders[path.name] = found
        assert offenders == {}

    def test_the_sampler_imports_nothing_from_the_platform_at_all(self):
        """`events.py` is pure policy. It cannot even reach the camera."""
        tree = ast.parse(Path("tools/p9_dataset/events.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert imported <= {"__future__", "enum", "dataclasses", "typing"}
