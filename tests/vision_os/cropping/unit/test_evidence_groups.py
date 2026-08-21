"""Attribute evidence groups — one crop per region, not one per subject.

The failure this exists to remove. A person box is roughly 1:3 and the canonical
crop is square, so a whole-subject crop spends most of its pixels on black bar.
Narrowing helps only if the narrowing is *per question*: unioning "the head" with
"the hands" reproduces the whole-body crop it was meant to escape, and on real
kitchen CCTV that lost plainly visible head coverings.

So M8 now emits one ``CropRequest`` per **evidence group** — attributes sharing a
declared region travel together, attributes about different regions get their own
crop.

Three properties are load-bearing and each is asserted here:

* **grouping is by region, never by meaning.** M8 compares two pairs of floats
  supplied by configuration and never learns what an attribute is.
* **every extra crop is charged.** Two groups cost two budget units. A split that
  bought free model calls would move the cost model rather than fix accuracy.
* **no regions means no change.** A deployment that declared none gets exactly
  one group, which is the behaviour before this existed.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.crop import SkipReason
from vision_os.core.model.timebase import Duration, Instant
from vision_os.perception.cropping import CropManager, UnderstandingBudget

from ..conftest import COLOUR, GARMENT, frame_context, make_demand, make_object

#: Two attributes in different places, as a policy document would declare them.
SPLIT = {str(COLOUR): (0.0, 0.45), str(GARMENT): (0.15, 0.55)}

#: The same two attributes visible in one place.
SHARED = {str(COLOUR): (0.15, 0.55), str(GARMENT): (0.15, 0.55)}

BOTH = (COLOUR, GARMENT)


@pytest.fixture
def build(
    clock,
    metrics,
    bus,
    cropping_config,
    trigger_policy,
    estimator,
    strategy,
    extractor,
    cropping_provenance,
    demand_registry,
    gate,
):
    """A Crop Manager with a chosen region map and budget ceiling."""

    def make(*, regions=None, calls_per_hour: float | None = None) -> CropManager:
        budget = UnderstandingBudget(
            ceiling_per_hour=(
                cropping_config.understanding_calls_per_hour
                if calls_per_hour is None
                else calls_per_hour
            ),
            window=Duration.from_millis(cropping_config.budget_window_ms),
            now=clock.monotonic(),
        )
        return CropManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=cropping_config,
            policy=trigger_policy,
            estimator=estimator,
            strategy=strategy,
            extractor=extractor,
            provenance=cropping_provenance,
            demands=demand_registry,
            budget=budget,
            gate=gate,
            evidence_regions=regions,
        )

    return make


def served(manager):
    manager.register_demand(make_demand(attributes=BOTH))
    return manager.evaluate([make_object()], frame_context())


class TestGroupingIsByRegion:
    def test_different_regions_get_their_own_crop(self, build) -> None:
        assert build(regions=SPLIT)._evidence_groups(BOTH) == (
            (COLOUR,),
            (GARMENT,),
        )

    def test_a_shared_region_costs_one_crop(self, build) -> None:
        """Attributes visible in the same place have no reason to be asked twice."""
        assert build(regions=SHARED)._evidence_groups(BOTH) == ((COLOUR, GARMENT),)

    def test_no_declared_regions_is_a_single_group(self, build) -> None:
        """The behaviour before this existed, preserved exactly."""
        assert build(regions=None)._evidence_groups(BOTH) == ((COLOUR, GARMENT),)

    def test_grouping_is_deterministic(self, build) -> None:
        """V13: identical demands produce identical requests in identical order."""
        manager = build(regions=SPLIT)

        assert manager._evidence_groups(BOTH) == manager._evidence_groups(
            tuple(reversed(BOTH))
        )

    def test_no_attributes_produces_no_groups(self, build) -> None:
        assert build(regions=SPLIT)._evidence_groups(()) == ()


class TestOneCropPerGroupThroughEvaluate:
    """Driven through the real ``evaluate``, not the helper."""

    def test_a_subject_produces_one_request_per_region(
        self, build
    ) -> None:
        result = served(build(regions=SPLIT))

        assert len(result.requests) == 2
        assert {frozenset(r.required_attributes) for r in result.requests} == {
            frozenset({str(COLOUR)}),
            frozenset({str(GARMENT)}),
        }

    def test_every_request_names_the_same_subject_and_frame(
        self, build
    ) -> None:
        """Two crops, one subject. Provenance must not fork."""
        manager = build(regions=SPLIT)
        manager.register_demand(make_demand(attributes=BOTH))
        obj, frame = make_object(), frame_context()

        result = manager.evaluate([obj], frame)

        assert {r.object_id for r in result.requests} == {obj.object_id}
        assert {r.frame_ref for r in result.requests} == {frame.frame_ref}

    def test_without_regions_a_subject_produces_one_request(
        self, build
    ) -> None:
        result = served(build(regions=None))

        assert len(result.requests) == 1
        assert set(result.requests[0].required_attributes) == {
            str(COLOUR),
            str(GARMENT),
        }


class TestEveryCropIsCharged:
    """A split must not buy free model calls."""

    def test_two_groups_spend_two_budget_units(
        self, build
    ) -> None:
        manager = build(regions=SPLIT)
        served(manager)

        assert manager.budget_status().spent_in_window == 2

    def test_one_group_spends_one(self, build) -> None:
        manager = build(regions=None)
        served(manager)

        assert manager.budget_status().spent_in_window == 1

    def test_running_out_mid_subject_degrades_honestly(
        self, build
    ) -> None:
        """One group served, the other skipped with an attributed reason.

        The alternative — answering the second question from the first
        question's pixels — is the failure the split exists to prevent, and it
        would be invisible.
        """
        manager = build(regions=SPLIT, calls_per_hour=1.0)
        result = served(manager)

        assert len(result.requests) == 1
        starved = [s for s in result.skipped if s.reason is SkipReason.BUDGET_EXHAUSTED]
        assert len(starved) == 1
        assert len(starved[0].attribute_keys) == 1, (
            "a starved group must name only its own attributes"
        )


class TestNoCandidateGoesSilent:
    """Obligation G1 and invariant V8 survive the split.

    A subject may now appear as several requests, or as a request plus a skip —
    but never as nothing.
    """

    def test_a_starved_subject_is_still_mentioned(
        self, build
    ) -> None:
        manager = build(regions=SPLIT, calls_per_hour=1.0)
        obj = make_object()
        manager.register_demand(make_demand(attributes=BOTH))

        result = manager.evaluate([obj], frame_context())
        mentioned = {r.object_id for r in result.requests} | {
            s.object_id for s in result.skipped
        }

        assert obj.object_id in mentioned

    def test_an_undemanded_subject_is_skipped_once_not_per_group(
        self, build
    ) -> None:
        """No demand means there are no groups to consider at all."""
        result = build(regions=SPLIT).evaluate([make_object()], frame_context())

        assert result.requests == ()
        assert len(result.skipped) == 1
        assert result.skipped[0].reason is SkipReason.NO_DEMAND
