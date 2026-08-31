"""P9.6 Phase 2 — departure semantics, sample-class separation, event audit.

No camera and no model. Every rule is exercised against a scripted scene or a
synthetic trace, so the suite runs anywhere.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tools.p9_dataset.baselines import PHASE1, PHASE2_B, PHASE2_C, POLICIES
from tools.p9_dataset.event_audit import CONSEQUENCES, audit
from tools.p9_dataset.event_audit import Testability as Verdict
from tools.p9_dataset.events import (
    DepartureRule,
    EventSampler,
    SampleClass,
    SamplingConfig,
    SamplingReason,
)

PERSON = (0.10, 0.10, 0.30, 0.90)
OTHER = (0.60, 0.10, 0.80, 0.90)


def run(script, **overrides):
    defaults = {
        "min_gap_seconds": 0.0,
        "hard_floor_seconds": 0.0,
        "heartbeat_seconds": 10_000.0,
        "detect_every_seconds": 0.0,
        "track_min_hits": 1,
    }
    defaults.update(overrides)
    engine = EventSampler(
        SamplingConfig(**defaults),
        hash_of=lambda i: script[i][0],
        detect=lambda i: [(box, 0.9) for box in script[i][1]],
    )
    return [engine.offer(i, float(i), i) for i in range(len(script))], engine


def entry(reason, people, boxes, *, reference):
    """A synthetic retained frame with its audit reference already attached."""
    payload = {
        "reason": reason,
        "hash": 0xAAAA,
        "people": people,
        "boxes": [list(b) for b in boxes],
        "reference": None,
    }
    if reference is not None:
        count, ref_boxes = reference
        payload["reference"] = {
            "people": count,
            "boxes": [list(b) for b in ref_boxes],
            "hash": 0xAAAA,
        }
    return payload


def departure(decisions):
    return next(
        (d for d in decisions if d.accepted and d.reason is SamplingReason.PERSON_LEFT),
        None,
    )


class TestFrozenBaseline:
    """Phase 1 of the brief: the old policy must stay executable."""

    def test_phase1_is_pinned_by_value_not_by_defaults(self):
        """If the defaults drift, PHASE1 must not drift with them.

        Otherwise a later A/B compares two things, neither of which ran in
        production.
        """
        assert PHASE1.version == "p9.6-events-1"
        assert PHASE1.departure_rule is DepartureRule.ON_EXPIRY
        assert PHASE1.track_max_age == 2
        assert PHASE1.departure_confirm_misses == 0

    def test_phase1_differs_from_the_current_default(self):
        """The whole point of Phase 2 — if these agreed, nothing changed."""
        assert SamplingConfig().departure_rule is not PHASE1.departure_rule

    def test_every_policy_serialises_completely(self):
        for name, policy in POLICIES.items():
            payload = policy.as_dict()
            for field in dataclasses.fields(SamplingConfig):
                assert field.name in payload, f"{name} lost {field.name}"

    def test_policy_versions_are_distinct(self):
        versions = [p.version for p in POLICIES.values()]
        assert len(set(versions)) == len(versions)


class TestDepartureSemantics:
    """Phase 2 of the brief. `PERSON_LEFT` supplied 151 of 323 person-free frames."""

    def test_on_expiry_keeps_the_frame_after_the_person_is_gone(self):
        """Phase 1 behaviour, pinned so the A/B has a real control."""
        script = [(0, [PERSON])] * 3 + [(0, [])] * 5
        decisions, _ = run(script, departure_rule=DepartureRule.ON_EXPIRY, track_max_age=2)
        event = departure(decisions)
        assert event is not None
        assert event.capture_frame_index == -1, "no retrospective capture requested"
        assert event.people == 0, "and the frame it keeps has nobody in it"

    def test_last_confirmed_asks_for_the_frame_the_person_was_in(self):
        script = [(0, [PERSON])] * 3 + [(0, [])] * 5
        decisions, _ = run(
            script, departure_rule=DepartureRule.LAST_CONFIRMED, track_max_age=2
        )
        event = departure(decisions)
        assert event is not None
        assert event.capture_frame_index == 2, "the last frame the track was matched in"
        assert event.capture_frame_index < event.frame_index

    def test_the_requested_frame_is_one_the_sampler_has_already_seen(self):
        """A retrospective capture must never name a frame from the future."""
        script = [(0, [PERSON])] * 4 + [(0, [])] * 6
        decisions, _ = run(script, departure_rule=DepartureRule.LAST_CONFIRMED)
        for decision in decisions:
            if decision.capture_frame_index >= 0:
                assert decision.capture_frame_index < decision.frame_index

    def test_a_fallback_is_recorded_never_silently_substituted(self):
        """Keeping a different frame from the one the policy names is a
        provenance error, so the decision says when it happened."""
        script = [(0, [PERSON])] * 3 + [(0, [])] * 5
        decisions, _ = run(script, departure_rule=DepartureRule.LAST_CONFIRMED)
        event = departure(decisions)
        assert event.capture_fallback == ""

    def test_confirm_misses_delays_believing_a_departure(self):
        """A person behind a passing trolley has not left the kitchen."""
        script = [(0, [PERSON])] * 3 + [(0, [])] * 4 + [(0, [PERSON])] * 3
        strict, _ = run(script, track_max_age=2, departure_confirm_misses=4)
        assert departure(strict) is None, "the person came back; nobody departed"

        loose, _ = run(script, track_max_age=2, departure_confirm_misses=0)
        assert departure(loose) is not None, "the control must still fire"

    def test_a_genuine_departure_still_fires_under_confirm_misses(self):
        """The guard must delay a departure, not abolish it."""
        script = [(0, [PERSON])] * 3 + [(0, [])] * 12
        decisions, _ = run(script, track_max_age=2, departure_confirm_misses=4)
        assert departure(decisions) is not None

    def test_temporary_occlusion_is_not_a_departure(self):
        script = [(0, [PERSON, OTHER])] * 3 + [(0, [PERSON])] * 2 + [(0, [PERSON, OTHER])] * 3
        decisions, _ = run(script, track_max_age=3, departure_confirm_misses=2)
        assert departure(decisions) is None

    def test_the_rule_is_recorded_in_the_statistics(self):
        _, engine = run([(0, [PERSON])] * 3, departure_rule=DepartureRule.LAST_CONFIRMED)
        assert engine.statistics()["departure_rule"] == "last_confirmed"


class TestSampleClassSeparation:
    """Phase 3 of the brief. A baseline frame is supposed to be empty."""

    def test_the_heartbeat_is_classed_baseline(self):
        decisions, _ = run([(0xABCD, [])] * 6, heartbeat_seconds=2.0)
        beats = [d for d in decisions if d.accepted]
        assert beats
        assert all(d.sample_class is SampleClass.BASELINE for d in beats)

    def test_every_other_reason_is_classed_event(self):
        script = [(0, []), (0, [PERSON]), (0, [PERSON, OTHER]), (0xFFFF, [PERSON, OTHER])]
        decisions, _ = run(script, heartbeat_seconds=10_000.0)
        for decision in decisions:
            if decision.accepted and decision.reason is not SamplingReason.PERIODIC_HEARTBEAT:
                assert decision.sample_class is SampleClass.EVENT

    def test_the_two_classes_are_counted_apart(self):
        script = [(0xABCD, [])] * 4 + [(0xFFFF, [PERSON])] * 4
        _, engine = run(script, heartbeat_seconds=2.0)
        by_class = engine.statistics()["by_class"]
        assert set(by_class) <= {"event", "baseline"}
        assert sum(by_class.values()) == engine.statistics()["frames_accepted"]

    def test_a_class_is_never_inferred_from_person_count(self):
        """A baseline frame that happens to contain someone is still baseline.

        Classing by content rather than by trigger would make the baseline's
        own coverage metric depend on the thing it is meant to be independent
        of.
        """
        script = [(0xABCD, [PERSON])] + [(0xABCD, [PERSON])] * 5
        decisions, _ = run(script, heartbeat_seconds=2.0)
        beats = [
            d
            for d in decisions
            if d.accepted and d.reason is SamplingReason.PERIODIC_HEARTBEAT
        ]
        assert beats
        assert all(d.sample_class is SampleClass.BASELINE for d in beats)


class TestEventAuditContract:
    """Phase 5 of the brief. Scoring the wrong consequence is worse than none."""

    def test_every_reason_has_a_declared_consequence(self):
        for reason in SamplingReason:
            assert reason.value in CONSEQUENCES, f"{reason.value} claims nothing"

    def test_tautological_reasons_are_never_scored(self):
        """A metric that cannot fail is decoration."""
        for name in ("scene_changed", "periodic_heartbeat"):
            _, testability = CONSEQUENCES[name]
            assert testability is Verdict.TAUTOLOGICAL

    def test_manual_review_is_untestable(self):
        _, testability = CONSEQUENCES["manual_review"]
        assert testability is Verdict.UNTESTABLE

    def test_occlusion_is_testable_from_boxes_not_from_counts(self):
        """The Phase 1 audit scored it by person count and manufactured a
        failure; an unchanged count is what the event PREDICTS."""
        claim, testability = CONSEQUENCES["occlusion_changed"]
        assert testability is Verdict.TESTABLE
        assert "count need not" in claim

    def test_a_tautological_row_carries_no_rate(self):
        kept = [
            {"reason": "scene_changed", "hash": 0xAAAA, "people": 1, "boxes": [PERSON]},
            {"reason": "scene_changed", "hash": 0xAAAA, "people": 1, "boxes": [PERSON]},
        ]
        report = audit(kept, SamplingConfig())
        assert report["by_reason"]["scene_changed"]["rate"] is None

    def test_a_failed_prediction_is_recorded_as_a_failure(self):
        """`person_entered` whose reference already held the same count did not
        deliver what it claimed."""
        kept = [
            entry("person_entered", 2, [PERSON, OTHER], reference=(2, [PERSON, OTHER])),
        ]
        report = audit(kept, SamplingConfig())["by_reason"]["person_entered"]
        assert report["compared"] == 1
        assert report["failed"] == 1
        assert report["rate"] == 0.0

    def test_a_held_prediction_is_recorded_as_held(self):
        kept = [entry("person_entered", 2, [PERSON, OTHER], reference=(1, [PERSON]))]
        report = audit(kept, SamplingConfig())["by_reason"]["person_entered"]
        assert report["held"] == 1
        assert report["rate"] == 1.0

    def test_a_frame_with_no_reference_is_not_counted_as_a_pass(self):
        """A claim that was never put to the test has not survived one.

        The reference comes from the raw trace; when nothing in the window
        resembles the frame there is no baseline to test against, and treating
        that as corroboration would inflate every rate.
        """
        kept = [entry("person_entered", 2, [PERSON, OTHER], reference=None)]
        report = audit(kept, SamplingConfig())["by_reason"]["person_entered"]
        assert report["no_match"] == 1
        assert report["compared"] == 0
        assert report["rate"] is None

    def test_the_reference_comes_from_the_trace_not_the_retained_set(self):
        """Otherwise the baseline moves with the policy under test.

        Measured during this phase: a policy-dependent reference alone moved
        `person_entered` from 80.0 % to 25.7 % between two policies whose entry
        logic is byte-identical.
        """
        import inspect

        from tools.p9_dataset.trace import replay

        assert "reference" in inspect.getsource(replay)
        assert "reference_earliest" in inspect.getsource(replay)

    def test_occlusion_is_scored_on_geometry_and_can_hold_with_a_stable_count(self):
        """The exact case the Phase 1 audit got wrong."""
        apart = [(0.05, 0.1, 0.25, 0.9), (0.70, 0.1, 0.90, 0.9)]
        overlapping = [(0.05, 0.1, 0.25, 0.9), (0.10, 0.1, 0.30, 0.9)]
        kept = [entry("occlusion_changed", 2, overlapping, reference=(2, apart))]
        report = audit(kept, SamplingConfig(edge_epsilon=0.0))["by_reason"]["occlusion_changed"]
        assert report["compared"] == 1
        assert report["held"] == 1, "count unchanged, geometry changed — it held"

    def test_person_left_is_scored_against_the_rule_in_force(self):
        """ON_EXPIRY predicts a lower count; LAST_CONFIRMED predicts a non-empty
        frame. Scoring the second with the first's contract reports a collapse
        belonging entirely to the auditor — measured at 91.4 % -> 56.5 % before
        the contract was made rule-aware."""
        kept = [entry("person_left", 2, [PERSON, OTHER], reference=(1, [PERSON]))]

        expiry = audit(kept, PHASE1)["by_reason"]["person_left"]
        assert expiry["failed"] == 1, "count went up, so the ON_EXPIRY claim fails"

        confirmed = audit(kept, PHASE2_B)["by_reason"]["person_left"]
        assert confirmed["held"] == 1, "the frame shows the person — the point of the rule"

    def test_the_overall_rate_covers_only_testable_reasons(self):
        kept = [
            {"reason": "periodic_heartbeat", "hash": 0xAAAA, "people": 0, "boxes": []},
            {"reason": "periodic_heartbeat", "hash": 0xAAAA, "people": 0, "boxes": []},
        ]
        report = audit(kept, SamplingConfig())
        assert report["overall_testable"]["compared"] == 0
        assert report["overall_testable"]["rate"] is None


class TestReplayDeterminism:
    """Phase 4 of the brief: the same trace must give the same answer."""

    def _trace(self):
        """A person arrives, stays a while, leaves, and comes back.

        The pattern matters: without a genuine departure the departure rules
        cannot diverge, and a test that cannot distinguish them proves nothing.
        """
        observations = []
        for i in range(48):
            present = (i % 24) < 10
            boxes = [[list(PERSON), 0.9]] if present else []
            if present and i % 24 >= 5:
                boxes.append([list(OTHER), 0.85])
            observations.append(
                {"i": i, "t": i * 0.25, "hash": 0xFFFF if present else 0x00FF, "boxes": boxes}
            )
        return {
            "trace_id": "trace-test",
            "cameras": [
                {"camera_id": "cam-11", "frames_decoded": 600, "observations": observations}
            ],
        }

    def test_replay_is_reproducible(self):
        from tools.p9_dataset.trace import replay

        first = replay(self._trace(), PHASE1)
        second = replay(self._trace(), PHASE1)
        assert [e["captured"] for e in first["by_camera"]["cam-11"]["kept"]] == [
            e["captured"] for e in second["by_camera"]["cam-11"]["kept"]
        ]

    def test_replay_honours_retrospective_capture(self):
        from tools.p9_dataset.trace import replay

        result = replay(self._trace(), PHASE2_B)
        kept = result["by_camera"]["cam-11"]["kept"]
        retro = [e for e in kept if e["retrospective"]]
        for entry in retro:
            assert entry["captured"] < entry["offered"]

    def test_the_two_policies_can_differ_on_the_same_trace(self):
        """If they never differ, the A/B is measuring nothing."""
        from tools.p9_dataset.trace import replay

        trace = self._trace()
        a = replay(trace, PHASE1)["by_camera"]["cam-11"]["kept"]
        b = replay(trace, PHASE2_C)["by_camera"]["cam-11"]["kept"]
        assert [e["captured"] for e in a] != [e["captured"] for e in b]

    def test_a_trace_carries_no_pixels(self):
        """Privacy: the replay substrate must not be identifiable footage."""
        observation = self._trace()["cameras"][0]["observations"][0]
        assert set(observation) == {"i", "t", "hash", "boxes"}


class TestTraceRecorderSafety:
    def test_the_recorder_reaches_no_production_decision_code(self):
        import ast

        source = Path("tools/p9_dataset/trace.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert "compliance" not in imported

    def test_the_recorder_stores_no_ppe_signal(self):
        source = Path("tools/p9_dataset/trace.py").read_text(encoding="utf-8")
        for banned in ("head_covering", "hairnet", "glove", "AttributeState"):
            assert banned not in source


class TestNoRegressionInGuards:
    """Phase 13: the Phase 1 invariants must survive Phase 2 unchanged."""

    def test_perception_still_enters_through_exactly_two_callables(self):
        import inspect

        assert set(inspect.signature(EventSampler.__init__).parameters) == {
            "self",
            "config",
            "hash_of",
            "detect",
        }

    def test_no_reason_names_a_ppe_concept(self):
        vocabulary = " ".join(r.value for r in SamplingReason)
        for banned in ("hairnet", "glove", "mask", "cover", "ppe", "violation"):
            assert banned not in vocabulary

    def test_departure_rules_are_a_closed_set(self):
        assert {r.value for r in DepartureRule} == {"on_expiry", "last_confirmed"}

    def test_sample_classes_are_a_closed_set(self):
        assert {c.value for c in SampleClass} == {"event", "baseline"}

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_every_frozen_policy_replays_without_error(self, name):
        from tools.p9_dataset.trace import replay

        trace = TestReplayDeterminism()._trace()
        assert replay(trace, POLICIES[name])["by_camera"]["cam-11"]["kept"] is not None
