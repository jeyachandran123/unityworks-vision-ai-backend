"""What the pose observability gate does to the real rule's real verdicts.

`test_dataset_regression.py` measures the pipeline **without** the gate and pins
its false-violation count. This measures the same 43 recorded model answers with
the gate in front of them, and pins the difference.

Both replay recorded evidence — the VLM's answers and the pose producer's
verdicts — so neither needs a key, a network, an ONNX runtime or a 13 MB
artefact. A measurement that only runs on one machine stops being a measurement.

**The headline result, and the reason this phase shipped:**

```
                    without gate   with gate   delta
false_violation               11           1     -10
true_violation                 1           1      +0
compliant                      6           6      +0
missed_violation               2           2      +0
```

Ten of eleven false violations removed at **zero cost** to any correct verdict,
and 13 of 43 head-band model calls avoided.

The one survivor is `f00780/s2`, the known unsafe acceptance: one keypoint at
0.57, annotated head-not-located, and Phase 4.3's separate visual inspection
found a head visible in that frame. The ground truth there is genuinely
ambiguous. It was not relabelled — correcting ground truth to improve a result
the same investigation produced is the circularity this programme refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compliance import ComplianceState

from .test_dataset_regression import CASES, verdict

RECORDED = Path(__file__).resolve().parent / "kitchen01_pose_verdicts.json"
POSE = json.loads(RECORDED.read_text(encoding="utf-8"))
VERDICTS = {(c["frame"], c["subject"]): c for c in POSE["cases"]}

#: The states on which the gate withholds the crop, so no value reaches the rule.
REFUSING = frozenset({"not_located", "low_confidence"})


def _key(case: dict) -> tuple[str, str]:
    return case["frame"], case["subject"]


def _gated(case: dict) -> bool:
    return VERDICTS[_key(case)]["state"] in REFUSING


def classify(*, gate: bool) -> dict[str, list[dict]]:
    """The same buckets `test_dataset_regression.classify` uses, gate optional.

    A refused head means the attribute is never produced, which reaches the rule
    as `held is None` -> ATTRIBUTE_ABSENT -> UNKNOWN. That is the existing path,
    not a new one: the gate withholds evidence, it never invents a value.
    """
    buckets: dict[str, list[dict]] = {
        "false_violation": [], "true_violation": [], "unknown": [],
        "compliant": [], "missed_violation": [],
    }
    for case in CASES:
        head = case["model"].get("head_covering")
        if gate and _gated(case):
            head = None
        state = verdict(head, case["model"].get("hand_covering"))
        head_truth = case["truth"].get("head_covering")
        hand_truth = case["truth"].get("hand_covering")
        if state is ComplianceState.VIOLATION:
            unobservable = head_truth == "not_visible" and hand_truth == "not_visible"
            buckets["false_violation" if unobservable else "true_violation"].append(case)
        elif state is ComplianceState.UNKNOWN:
            if "absent" in (head_truth, hand_truth):
                buckets["missed_violation"].append(case)
            else:
                buckets["unknown"].append(case)
        elif state is ComplianceState.COMPLIANT:
            buckets["compliant"].append(case)
    return buckets


class TestTheRecordingItself:
    def test_it_covers_every_subject(self) -> None:
        assert len(POSE["cases"]) == 43
        assert {_key(c) for c in CASES} == set(VERDICTS)

    def test_it_names_the_artefact_it_came_from(self) -> None:
        """A recording nobody can trace to a model is not evidence."""
        assert POSE["model_sha256"].startswith("4abcdec00c4c9891")
        assert POSE["producer"] == "observability.pose.yolov8n"
        assert POSE["keypoint_confidence_floor"] == 0.5

    def test_every_state_is_one_the_port_defines(self) -> None:
        from vision_os.core.model.region_observability import RegionState

        legal = {state.value for state in RegionState}
        assert {c["state"] for c in POSE["cases"]} <= legal


class TestAccuracyAgainstHumanAnnotation:
    """Scored against the human head-location labels, never against the VLM.

    Using hairnet labels or compliance outcomes here would make the evaluation
    circular — the producer would be scored on the thing it is meant to improve.
    """

    def test_precision_and_recall_are_pinned(self) -> None:
        tp = fp = tn = fn = 0
        for case in POSE["cases"]:
            located = case["state"] == "located"
            truth = case["human_head_located"]
            if truth and located:
                tp += 1
            elif truth and not located:
                fn += 1
            elif not truth and located:
                fp += 1
            else:
                tn += 1

        assert (tp, fp, tn, fn) == (29, 1, 10, 3)
        assert pytest.approx(tp / (tp + fp), abs=0.001) == 0.967  # precision
        assert pytest.approx(tp / (tp + fn), abs=0.001) == 0.906  # recall

    def test_it_beats_the_measurement_that_justified_building_it(self) -> None:
        """Phase 4.4 published P=0.966 R=0.875 using PIL to resize.

        The shipped adapter carries no PIL dependency and area-averages instead,
        which is the correct downsample and measurably better: recall 0.875 ->
        0.906, false refusals 4 -> 3, with the unsafe acceptance unchanged at 1.
        """
        located = sum(1 for c in POSE["cases"] if c["state"] == "located")
        refused_but_visible = sum(
            1 for c in POSE["cases"]
            if c["human_head_located"] and c["state"] != "located"
        )
        assert located == 30
        assert refused_but_visible == 3, "phase 4.4's PIL path refused 4"

    def test_the_unsafe_acceptance_is_still_one_and_still_named(self) -> None:
        """Honesty about what this does not fix.

        One head the annotator could not read is still passed through. Pinned so
        that it cannot grow quietly.
        """
        unsafe = [
            c for c in POSE["cases"]
            if not c["human_head_located"] and c["state"] == "located"
        ]
        assert len(unsafe) == 1
        assert (unsafe[0]["frame"], unsafe[0]["subject"]) == ("f00780", "s2")


class TestTheEffectOnRealVerdicts:
    def test_false_violations_fall_from_eleven_to_one(self) -> None:
        """The measurement this phase exists for."""
        assert len(classify(gate=False)["false_violation"]) == 11
        assert len(classify(gate=True)["false_violation"]) == 1

    def test_no_true_violation_is_lost(self) -> None:
        """The cost side, and the one that would justify rejecting the gate.

        An earlier draft of the adapter downsampled by nearest neighbour and
        failed exactly this test: `f01500/s2`'s best head keypoint fell from 0.59
        to 0.46 and the corpus's only true violation became a refusal. Aggregate
        accuracy was unchanged, which is why the aggregate could not catch it.
        """
        before = classify(gate=False)
        after = classify(gate=True)
        assert len(after["true_violation"]) == len(before["true_violation"]) == 1

    def test_no_compliant_verdict_is_lost(self) -> None:
        assert len(classify(gate=True)["compliant"]) == 6

    def test_no_new_violation_is_missed(self) -> None:
        assert len(classify(gate=True)["missed_violation"]) == 2

    def test_the_surviving_false_violation_is_the_known_one(self) -> None:
        survivors = classify(gate=True)["false_violation"]
        assert [(c["frame"], c["subject"]) for c in survivors] == [("f00780", "s2")]

    def test_refused_evidence_becomes_unknown_never_compliant(self) -> None:
        """The safety property. A withheld crop must never read as 'we looked
        and it was fine' — that would trade a false alarm for a missed hazard,
        which is the worse of the two errors."""
        for case in CASES:
            if not _gated(case):
                continue
            state = verdict(None, case["model"].get("hand_covering"))
            assert state is not ComplianceState.COMPLIANT

    def test_every_verdict_is_accounted_for(self) -> None:
        for gate in (True, False):
            assert sum(len(v) for v in classify(gate=gate).values()) == 43


class TestCost:
    def test_the_gate_saves_model_calls_rather_than_adding_them(self) -> None:
        """A safety fix that raised the bill would be a harder argument."""
        avoided = sum(1 for c in CASES if _gated(c))
        assert avoided == 13
        assert avoided / len(CASES) > 0.25

    def test_the_producer_stays_within_the_detector_s_budget_class(self) -> None:
        """~71 ms/frame against the detector's measured 51.9 ms.

        Recorded rather than asserted tightly: this is a CPU timing on one
        machine, and pinning it hard would make the suite fail on a slower one.
        The check is that it is the same order of magnitude, not that it is fast.
        """
        assert 0 < POSE["median_frame_latency_ms"] < 250
