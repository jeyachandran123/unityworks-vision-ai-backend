"""The kitchen policy's confidence floor sits above a pot and below people.

PCC2, camera 13, 2026-10-16: a cooking pot with a clear lid in the bottom-left
corner was detected as ``person`` at **0.41** by yolo11s through the platform's
letterbox. The 0.40 floor admitted it, the model was asked about the head of a
pot, answered that no covering was visible on it, and the rule raised an alert.

Across the 32 PCC2 decision frames measured, only two person detections fell in
[0.40, 0.45): that pot (0.41) and one real cook (0.40). A real person moves and
is re-detected; a pot stays in its corner at the same score all day and would
alert for as long as it is there. At 0.45 the next real people measured were
0.47 and above.

Medium detectors (yolo11m, yolo26m) do not detect the pot at all; this floor is
the CPU-only measure until one of those can run. See `test_detector_default.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vision.composition import load_policies
from app.vision.demands import SUBSCRIBER

POLICIES = Path(__file__).resolve().parents[2] / "config" / "policies"


@pytest.fixture(params=["kitchen-safety.example.json", "kitchen-safety.head-variant.json"])
def subject_filter(request):
    (policy,) = load_policies(str(POLICIES / request.param))
    return policy.build_demand(subscriber=SUBSCRIBER, cameras=()).subject_filter


def test_the_pot_detected_as_a_person_is_not_asked_about(subject_filter) -> None:
    assert not subject_filter.matches_confidence(0.41)


@pytest.mark.parametrize("confidence", [0.45, 0.47, 0.84, 0.91])
def test_people_detected_with_ordinary_confidence_still_are(subject_filter, confidence) -> None:
    assert subject_filter.matches_confidence(confidence)
