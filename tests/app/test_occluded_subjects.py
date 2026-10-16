"""A person the detector did not find in a frame is never asked about in it.

Measured on PCC2 on 2026-10-15: of 30 open alerts, **11** were about a box in
which the same detector, run on the alert's own decision frame, found nobody
(best overlap below 0.3). Where the stored box did belong to a detection the
overlap was 0.9 or better, so the boxes themselves are drawn correctly — they
were simply drawn where somebody *used to be*.

The cause was the kitchen policy's scope. It asked about objects in
``active`` **and** ``occluded`` state. The registry moves an object to
``occluded`` on the first frame it is not measured, and its box stays where it
was last seen. The Crop Manager cuts the crop from the *current* frame at that
remembered box, so the model was shown an empty shelf, answered that no head
covering was visible on it, and the rule raised an alert about nobody.

``occluded`` means "believed present, merely unmeasured" — a fine thing to
remember about a person, and exactly the wrong thing to photograph. These pin
the policy to people measured in the frame being analysed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.vision.composition import load_policies
from app.vision.demands import SUBSCRIBER

POLICIES = Path(__file__).resolve().parents[2] / "config" / "policies"


@pytest.fixture(params=["kitchen-safety.example.json", "kitchen-safety.head-variant.json"])
def kitchen_demand(request):
    (policy,) = load_policies(str(POLICIES / request.param))
    return policy.build_demand(subscriber=SUBSCRIBER, cameras=())


def test_a_person_measured_in_this_frame_is_asked_about(kitchen_demand) -> None:
    assert kitchen_demand.subject_filter.matches_lifecycle("active")


def test_a_person_not_found_in_this_frame_is_not(kitchen_demand) -> None:
    """The defect itself: an occluded person's box is a remembered position."""
    assert not kitchen_demand.subject_filter.matches_lifecycle("occluded")


@pytest.mark.parametrize("state", ["dormant", "departed", "provisional"])
def test_nor_is_one_that_has_left_or_is_unconfirmed(kitchen_demand, state) -> None:
    assert not kitchen_demand.subject_filter.matches_lifecycle(state)
