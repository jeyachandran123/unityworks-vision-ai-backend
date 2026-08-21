"""Region membership and dwell — pure geometry, and the ceiling that guards it.

``07_STATE`` section 3.3 names this the place where the Semantic Ceiling is most
tempting to breach:

> *`occupancy` is a count. `dwell_stats` are descriptive statistics over
> durations. There is no `is_crowded`, no `exceeds_capacity`, no `queue_forming`
> — each of those requires a threshold or a definition that only a consumer
> possesses (V1).*

Two properties are load-bearing and each has a class here: **dwell comes from
capture time** (V11, 14_TESTING section 4), and **membership uses a spatial
index** because section M7 requires the polygon tests "must not be naive at 100
objects x 20 regions".
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import ObjectId, RegionId
from vision_os.core.model.region import ContainmentMethod, MembershipState
from vision_os.core.model.space import Box, Point
from vision_os.perception.registry.regions import (
    RegionIndex,
    RegionOccupancy,
    RegionTracker,
    containment_point,
)

from ..conftest import at, make_region

OBJECT = ObjectId("01JB0000000000000000000001")
OTHER = ObjectId("01JB0000000000000000000002")
Z3 = RegionId("Z3")


def inside_box() -> Box:
    """A box whose bottom centre lands inside the default region."""
    return Box(0.45, 0.5, 0.55, 0.7)


def outside_box() -> Box:
    return Box(0.02, 0.02, 0.08, 0.10)


class TestSpatialIndex:
    def test_an_empty_index_contains_nothing(self) -> None:
        assert RegionIndex().containing(Point(0.5, 0.5)) == ()

    def test_a_point_inside_is_found(self) -> None:
        index = RegionIndex((make_region(),))
        assert len(index.containing(Point(0.5, 0.5))) == 1

    def test_a_point_outside_is_not_found(self) -> None:
        index = RegionIndex((make_region(),))
        assert index.containing(Point(0.05, 0.05)) == ()

    def test_the_bounds_check_does_not_change_the_answer(self) -> None:
        """Rejection by bounds is exact, not approximate: the result must equal
        testing every polygon."""
        regions = (
            make_region("A", box=(0.0, 0.0, 0.4, 0.4)),
            make_region("B", box=(0.6, 0.6, 1.0, 1.0)),
            make_region("C", box=(0.3, 0.3, 0.7, 0.7)),
        )
        index = RegionIndex(regions)
        for x in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
            for y in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
                point = Point(x, y)
                indexed = {r.region_id for r in index.containing(point)}
                naive = {r.region_id for r in regions if r.geometry.contains(point)}
                assert indexed == naive, f"index disagreed with naive at {point}"

    def test_overlapping_regions_both_match(self) -> None:
        index = RegionIndex(
            (
                make_region("A", box=(0.2, 0.2, 0.8, 0.8)),
                make_region("B", box=(0.4, 0.4, 1.0, 1.0)),
            )
        )
        assert len(index.containing(Point(0.5, 0.5))) == 2

    def test_rebuilding_advances_the_version(self) -> None:
        index = RegionIndex((make_region(),))
        version = index.version
        index.rebuild((make_region("Z4"),))
        assert index.version > version
        assert len(index) == 1


class TestContainmentPoint:
    def test_bbox_bottom_centre_is_the_ground_proxy(self) -> None:
        box = Box(0.2, 0.3, 0.6, 0.9)
        point = containment_point(box, ContainmentMethod.BBOX_BOTTOM_CENTRE)
        assert point.x == pytest.approx(0.4)
        assert point.y == pytest.approx(0.9)

    def test_mask_overlap_uses_the_centre(self) -> None:
        box = Box(0.2, 0.3, 0.6, 0.9)
        point = containment_point(box, ContainmentMethod.MASK_OVERLAP)
        assert point.y == pytest.approx(0.6)

    def test_the_method_travels_with_the_result(self) -> None:
        """Containment from a bottom edge and from a projected ground point
        disagree at range; a consumer comparing dwell deserves to know which."""
        tracker = RegionTracker(
            regions=(make_region(),), method=ContainmentMethod.MASK_OVERLAP
        )
        tracker.update(OBJECT, inside_box(), at=at(0))
        membership = tracker.membership(OBJECT)[Z3]
        assert membership.method is ContainmentMethod.MASK_OVERLAP


class TestMembership:
    @pytest.fixture
    def tracker(self) -> RegionTracker:
        return RegionTracker(regions=(make_region(),))

    def test_entering_produces_an_entry_transition(self, tracker) -> None:
        transitions = tracker.update(OBJECT, inside_box(), at=at(0))
        assert len(transitions) == 1
        assert transitions[0].entered
        assert transitions[0].region_id == Z3

    def test_staying_inside_produces_no_transition(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        assert tracker.update(OBJECT, inside_box(), at=at(1)) == ()

    def test_leaving_produces_an_exit_transition(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        transitions = tracker.update(OBJECT, outside_box(), at=at(5))
        assert len(transitions) == 1
        assert transitions[0].exited

    def test_membership_is_recorded_while_inside(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        membership = tracker.membership(OBJECT)
        assert Z3 in membership
        assert membership[Z3].state is MembershipState.INSIDE

    def test_membership_is_dropped_on_exit(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.update(OBJECT, outside_box(), at=at(5))
        assert tracker.membership(OBJECT) == {}

    def test_an_object_never_inside_has_no_membership(self, tracker) -> None:
        tracker.update(OBJECT, outside_box(), at=at(0))
        assert tracker.membership(OBJECT) == {}

    def test_forgetting_closes_open_memberships(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        transitions = tracker.forget(OBJECT, at=at(10))
        assert len(transitions) == 1
        assert transitions[0].exited
        assert tracker.membership(OBJECT) == {}

    def test_forgetting_an_unknown_object_is_safe(self, tracker) -> None:
        assert tracker.forget(OTHER, at=at(1)) == ()


class TestDwellUsesCaptureTime:
    """V11 and 14_TESTING section 4: dwell is computed from ``t_capture``."""

    @pytest.fixture
    def tracker(self) -> RegionTracker:
        return RegionTracker(regions=(make_region(),))

    def test_dwell_accrues_from_the_entry_instant(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        membership = tracker.membership(OBJECT)[Z3]
        assert membership.dwell(at(10)).millis == pytest.approx(2_000)

    def test_dwell_on_exit_measures_the_whole_stay(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        transitions = tracker.update(OBJECT, outside_box(), at=at(25))
        assert transitions[0].dwell.millis == pytest.approx(5_000)

    def test_dwell_is_independent_of_how_often_frames_arrive(self, tracker) -> None:
        """A dwell of 45 s means 45 s in the world, regardless of pipeline rate."""
        tracker.update(OBJECT, inside_box(), at=at(0))
        for seq in range(1, 30):
            tracker.update(OBJECT, inside_box(), at=at(seq))
        dense = tracker.membership(OBJECT)[Z3].dwell(at(30))

        sparse_tracker = RegionTracker(regions=(make_region(),))
        sparse_tracker.update(OBJECT, inside_box(), at=at(0))
        sparse_tracker.update(OBJECT, inside_box(), at=at(30))
        sparse = sparse_tracker.membership(OBJECT)[Z3].dwell(at(30))

        assert dense == sparse

    def test_entry_transition_reports_zero_dwell(self, tracker) -> None:
        transitions = tracker.update(OBJECT, inside_box(), at=at(0))
        assert transitions[0].dwell.ns == 0

    def test_re_entering_restarts_the_accumulator(self, tracker) -> None:
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.update(OBJECT, outside_box(), at=at(10))
        tracker.update(OBJECT, inside_box(), at=at(20))
        assert tracker.membership(OBJECT)[Z3].dwell(at(25)).millis == pytest.approx(1_000)


class TestGeometryVersioning:
    def test_changing_geometry_closes_open_accumulations(self) -> None:
        """Section M7: time in the old shape is never attributed to the new one."""
        tracker = RegionTracker(regions=(make_region(version="1.0.0"),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        closed = tracker.set_regions((make_region(version="2.0.0"),), now=at(20))

        assert len(closed) == 1
        assert closed[0].exited
        assert closed[0].geometry_version == "1.0.0"
        assert closed[0].dwell.millis == pytest.approx(4_000)

    def test_new_accumulations_open_against_the_new_version(self) -> None:
        tracker = RegionTracker(regions=(make_region(version="1.0.0"),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.set_regions((make_region(version="2.0.0"),), now=at(20))
        tracker.update(OBJECT, inside_box(), at=at(21))

        membership = tracker.membership(OBJECT)[Z3]
        assert membership.geometry_version == "2.0.0"
        assert membership.entered_at == at(21)

    def test_every_transition_carries_its_geometry_version(self) -> None:
        """Both sides of a change are published with their version."""
        tracker = RegionTracker(regions=(make_region(version="1.0.0"),))
        entry = tracker.update(OBJECT, inside_box(), at=at(0))
        assert entry[0].geometry_version == "1.0.0"

    def test_a_version_bump_mid_stay_closes_and_reopens(self) -> None:
        tracker = RegionTracker(regions=(make_region(version="1.0.0"),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        # Same tracker, region redefined in place with a new version.
        tracker.index.rebuild((make_region(version="2.0.0"),))
        transitions = tracker.update(OBJECT, inside_box(), at=at(10))
        assert any(t.exited and t.geometry_version == "1.0.0" for t in transitions)
        assert any(t.entered and t.geometry_version == "2.0.0" for t in transitions)


class TestOccupancyIsCountingOnly:
    """07_STATE section 3.3 — the ceiling's most tempting breach point."""

    def test_occupancy_counts_by_class(self) -> None:
        tracker = RegionTracker(regions=(make_region(),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.update(OTHER, Box(0.4, 0.5, 0.5, 0.75), at=at(0))

        reports = tracker.occupancy(
            classes={OBJECT: "person", OTHER: "vehicle"}, now=at(5)
        )
        assert len(reports) == 1
        assert reports[0].occupancy == {"person": 1, "vehicle": 1}
        assert reports[0].total == 2

    def test_dwell_statistics_are_descriptive(self) -> None:
        tracker = RegionTracker(regions=(make_region(),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.update(OTHER, Box(0.4, 0.5, 0.5, 0.75), at=at(10))

        report = tracker.occupancy(
            classes={OBJECT: "person", OTHER: "person"}, now=at(20)
        )[0]
        assert report.dwell_current_max.millis == pytest.approx(4_000)
        assert report.dwell_current_mean.millis == pytest.approx(3_000)

    def test_present_objects_are_listed_deterministically(self) -> None:
        tracker = RegionTracker(regions=(make_region(),))
        tracker.update(OTHER, Box(0.4, 0.5, 0.5, 0.75), at=at(0))
        tracker.update(OBJECT, inside_box(), at=at(0))
        report = tracker.occupancy(
            classes={OBJECT: "person", OTHER: "person"}, now=at(5)
        )[0]
        assert report.present_objects == tuple(sorted(report.present_objects))

    def test_an_empty_region_reports_zero(self) -> None:
        tracker = RegionTracker(regions=(make_region(),))
        report = tracker.occupancy(classes={}, now=at(0))[0]
        assert report.total == 0
        assert report.occupancy == {}
        assert report.dwell_current_max.ns == 0

    def test_no_judgment_field_exists(self) -> None:
        """No ``is_crowded``, no ``exceeds_capacity``, no ``queue_forming``."""
        fields = set(RegionOccupancy.__dataclass_fields__)
        for forbidden in (
            "is_crowded", "exceeds_capacity", "queue_forming", "over_capacity",
            "threshold", "alert", "status", "busy",
        ):
            assert forbidden not in fields

    def test_occupancy_has_no_capacity_to_compare_against(self) -> None:
        """A capacity would require a number only a consumer possesses (V1)."""
        assert "capacity" not in RegionOccupancy.__dataclass_fields__


class TestMultipleRegions:
    def test_an_object_can_be_in_several_regions(self) -> None:
        tracker = RegionTracker(
            regions=(
                make_region("A", box=(0.2, 0.2, 0.8, 0.9)),
                make_region("B", box=(0.4, 0.4, 1.0, 1.0)),
            )
        )
        transitions = tracker.update(OBJECT, inside_box(), at=at(0))
        assert len(transitions) == 2
        assert len(tracker.membership(OBJECT)) == 2

    def test_leaving_one_region_keeps_the_other(self) -> None:
        tracker = RegionTracker(
            regions=(
                make_region("A", box=(0.0, 0.0, 0.6, 1.0)),
                make_region("B", box=(0.4, 0.0, 1.0, 1.0)),
            )
        )
        tracker.update(OBJECT, Box(0.45, 0.5, 0.55, 0.7), at=at(0))
        assert len(tracker.membership(OBJECT)) == 2

        transitions = tracker.update(OBJECT, Box(0.7, 0.5, 0.8, 0.7), at=at(5))
        assert len(transitions) == 1
        assert transitions[0].exited
        assert set(tracker.membership(OBJECT)) == {RegionId("B")}

    def test_tracked_object_count_is_reported(self) -> None:
        tracker = RegionTracker(regions=(make_region(),))
        tracker.update(OBJECT, inside_box(), at=at(0))
        tracker.update(OTHER, Box(0.4, 0.5, 0.5, 0.75), at=at(0))
        assert tracker.tracked_objects == 2


class TestNoRegionsConfigured:
    def test_membership_is_empty_when_no_regions_exist(self) -> None:
        tracker = RegionTracker()
        assert tracker.update(OBJECT, inside_box(), at=at(0)) == ()
        assert tracker.membership(OBJECT) == {}

    def test_occupancy_is_empty_when_no_regions_exist(self) -> None:
        assert RegionTracker().occupancy(classes={}, now=at(0)) == ()
