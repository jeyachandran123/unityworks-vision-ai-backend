"""Projection tests — the pure fold at the heart of M12 (07_STATE §2, §5).

Vision State is a *materialized projection* of an immutable log, not a database.
Everything that follows from that claim is tested here: purity, determinism,
bounded growth, and the log's authority over the projection.

``project`` is a function of ``(partition, observation) -> partition`` with no
clock, no I/O and no hidden state, which is what makes a rebuild produce exactly
the world the live run produced.
"""

from __future__ import annotations

import dataclasses

import pytest

from vision_os.core.errors import ProjectionError
from vision_os.core.model.ids import CameraId, LogPosition, ObjectId
from vision_os.core.model.observation import (
    LifecycleTransition,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
)
from vision_os.core.model.vision_state import CameraPartition
from vision_os.core.model.visual_object import LifecycleState
from vision_os.state.projection import ProjectionBounds, project

from .conftest import (
    CAMERA,
    HEIGHT,
    OTHER_CAMERA,
    POSTURE,
    at,
    attribute,
    context,
    make_object,
    presence_of,
    spatial,
    understanding,
)

BOUNDS = ProjectionBounds(
    trajectory_points=8, attribute_history=4, class_history=4, max_objects=16
)


def empty(camera: CameraId = CAMERA) -> CameraPartition:
    return CameraPartition(camera_id=camera)


def fold(partition: CameraPartition, *observations, bounds=BOUNDS) -> CameraPartition:
    """Apply observations in order, returning the final partition."""
    for i, observation in enumerate(observations, start=1):
        partition = project(
            partition, observation, bounds=bounds, position=LogPosition(i)
        ).partition
    return partition


class TestProjectionIsPure:
    def test_the_input_partition_is_never_mutated(self, loud_builder) -> None:
        """Structural sharing depends on it.

        07_STATE §5.1 makes a snapshot O(1) by handing out the same immutable
        objects. A projection that mutated in place would change a snapshot a
        reader was already holding.
        """
        before = empty()
        after = fold(before, presence_of(loud_builder))
        assert before.objects == {}
        assert after.objects != {}
        assert before is not after

    def test_the_same_observations_always_produce_the_same_partition(
        self, loud_builder
    ) -> None:
        """V13. The property that makes a rebuild trustworthy."""
        observations = [
            presence_of(loud_builder, make_object(object_id="a"), seq=1),
            presence_of(loud_builder, make_object(object_id="b"), seq=2),
            presence_of(loud_builder, make_object(object_id="a", seq=3), seq=3),
        ]
        first = fold(empty(), *observations)
        second = fold(empty(), *observations)
        assert first.objects.keys() == second.objects.keys()
        for object_id in first.objects:
            assert first.objects[object_id] == second.objects[object_id]

    def test_projection_takes_no_clock(self) -> None:
        """A fold that read a clock would replay differently every time."""
        import inspect

        parameters = inspect.signature(project).parameters
        assert "clock" not in parameters
        assert set(parameters) == {"partition", "observation", "bounds", "position"}

    def test_an_unchanged_object_is_shared_rather_than_copied(
        self, loud_builder
    ) -> None:
        """The mechanism behind O(1) snapshots.

        Projecting an observation about ``b`` must leave ``a``'s state as the
        *same object*, not an equal copy — otherwise a 500-object partition
        copies 500 objects per observation.
        """
        first = fold(empty(), presence_of(loud_builder, make_object(object_id="a")))
        held = first.objects[ObjectId("a")]
        second = fold(
            first, presence_of(loud_builder, make_object(object_id="b"), seq=4)
        )
        assert second.objects[ObjectId("a")] is held


class TestTheObservationIsTheOnlyWritePath:
    def test_state_begins_at_the_first_observation_and_nowhere_else(
        self, loud_builder
    ) -> None:
        partition = empty()
        assert partition.objects == {}
        partition = fold(partition, presence_of(loud_builder))
        assert ObjectId("obj-1") in partition.objects

    def test_an_observation_for_another_camera_is_refused(self, loud_builder) -> None:
        """07_STATE §4.1: the camera *is* the partition.

        Accepting it would put one camera's facts in another's timeline, and
        every per-camera guarantee — ordering, recovery, isolation — would be
        silently false.
        """
        foreign = loud_builder.build_presence(
            make_object(camera=OTHER_CAMERA), context(camera=OTHER_CAMERA)
        )
        assert foreign is not None and foreign.camera_id == OTHER_CAMERA
        with pytest.raises(ProjectionError, match="camera"):
            project(empty(CAMERA), foreign, bounds=BOUNDS, position=LogPosition(1))

    def test_the_partition_version_advances_with_every_write(
        self, loud_builder
    ) -> None:
        partition = empty()
        versions = []
        for seq in range(1, 4):
            partition = fold(
                partition,
                presence_of(loud_builder, make_object(object_id=f"o{seq}"), seq=seq),
            )
            versions.append(partition.version)
        assert versions == sorted(versions)
        assert len(set(versions)) == 3

    def test_the_log_position_watermark_advances(self, loud_builder) -> None:
        """§9.1: recovery replays *"from the last committed log position"*.

        A watermark that did not advance would replay the whole log every time.
        """
        partition = fold(empty(), presence_of(loud_builder))
        assert partition.log_position == LogPosition(1)


class TestObjectStateProjection:
    def test_presence_records_position_and_confirms_the_sighting(
        self, loud_builder
    ) -> None:
        partition = fold(empty(), presence_of(loud_builder))
        state = partition.objects[ObjectId("obj-1")]
        assert state.last_seen == at(3)
        assert state.last_confirmed == at(3)
        assert state.observation_count == 1

    def test_a_predicted_position_advances_last_seen_but_not_last_confirmed(
        self, loud_builder
    ) -> None:
        """The object-level expression of V8.

        A consumer must be able to ask "is this still true, or are we just
        assuming?" — and a projection that confirmed on a guess would erase the
        only field that answers it.
        """
        partition = fold(empty(), presence_of(loud_builder, seq=3))
        believed = loud_builder.build_presence(
            make_object(seq=6), context(seq=6), basis=MeasurementBasis.PREDICTED
        )
        assert believed is not None
        partition = fold(partition, believed)

        state = partition.objects[ObjectId("obj-1")]
        assert state.last_seen == at(6)
        assert state.last_confirmed == at(3)

    def test_attributes_land_as_current_values(self, loud_builder) -> None:
        published = loud_builder.build_attribute(
            make_object(), understanding(attributes=(attribute(POSTURE, "sitting"),)),
            context(),
        )
        partition = fold(empty(), *published)
        state = partition.objects[ObjectId("obj-1")]
        assert state.attributes[POSTURE].value == "sitting"

    def test_a_later_attribute_supersedes_an_earlier_one(self, loud_builder) -> None:
        first = loud_builder.build_attribute(
            make_object(),
            understanding(attributes=(attribute(POSTURE, "standing", observed_at=at(3)),)),
            context(seq=3),
        )
        second = loud_builder.build_attribute(
            make_object(seq=8),
            understanding(
                request_id="req-2",
                seq=8,
                attributes=(attribute(POSTURE, "sitting", observed_at=at(8)),),
            ),
            context(seq=8),
        )
        partition = fold(empty(), *first, *second)
        assert partition.objects[ObjectId("obj-1")].attributes[POSTURE].value == "sitting"

    def test_an_out_of_order_attribute_does_not_overwrite_a_newer_one(
        self, loud_builder
    ) -> None:
        """A late arrival is not a correction.

        Networks reorder. Applying an older measurement over a newer one would
        make state depend on delivery order rather than on when things happened,
        and V13's replay would then disagree with the live run.
        """
        recent = loud_builder.build_attribute(
            make_object(seq=8),
            understanding(seq=8, attributes=(attribute(POSTURE, "sitting", observed_at=at(8)),)),
            context(seq=8),
        )
        stale = loud_builder.build_attribute(
            make_object(seq=2),
            understanding(
                request_id="req-old",
                seq=2,
                attributes=(attribute(POSTURE, "standing", observed_at=at(2)),),
            ),
            context(seq=2),
        )
        partition = fold(empty(), *recent, *stale)
        assert partition.objects[ObjectId("obj-1")].attributes[POSTURE].value == "sitting"

    def test_a_lifecycle_observation_moves_the_state_machine(
        self, loud_builder
    ) -> None:
        transition = loud_builder.build_lifecycle(
            make_object(),
            LifecycleTransition(LifecycleState.ACTIVE, LifecycleState.OCCLUDED),
            context(),
        )
        assert transition is not None
        partition = fold(empty(), presence_of(loud_builder), transition)
        assert partition.objects[ObjectId("obj-1")].lifecycle is LifecycleState.OCCLUDED


class TestBoundedGrowth:
    """07_STATE §6.3: *"bounded by both count and time... a structural property
    of the ring buffers rather than a tunable that might be misconfigured to
    infinity."*"""

    def test_the_trajectory_ring_never_exceeds_its_bound(self, loud_builder) -> None:
        partition = empty()
        for seq in range(40):
            observation = loud_builder.build_spatial(
                make_object(seq=seq, position=spatial(0.1 + seq * 0.01, 0.2)),
                context(seq=seq),
            )
            if observation is not None:
                partition = fold(partition, observation)
        assert len(partition.objects[ObjectId("obj-1")].trajectory) <= 8

    def test_the_attribute_history_ring_never_exceeds_its_bound(
        self, loud_builder
    ) -> None:
        partition = empty()
        for seq in range(20):
            published = loud_builder.build_attribute(
                make_object(seq=seq),
                understanding(
                    request_id=f"req-{seq}",
                    seq=seq,
                    attributes=(
                        attribute(HEIGHT, 0.1 + seq * 0.01, observed_at=at(seq)),
                    ),
                ),
                context(seq=seq),
            )
            partition = fold(partition, *published)
        # The ring lives on the attribute, not on the object: 07_STATE §3.1 keeps
        # a short per-attribute history so a consumer can see that a value
        # flipped twice in four seconds. The object holds current values only.
        history = partition.objects[ObjectId("obj-1")].attributes[HEIGHT].previous
        assert len(history) <= 4
        assert len(history) > 1, "some history must survive, or the ring is a no-op"

    def test_the_object_population_is_shed_rather_than_grown(
        self, loud_builder
    ) -> None:
        """A partition that grew without limit is a memory leak with a fuse."""
        partition = empty()
        for i in range(64):
            partition = fold(
                partition,
                presence_of(loud_builder, make_object(object_id=f"obj-{i}"), seq=i),
            )
        assert len(partition.objects) <= 16

    def test_shedding_drops_provisional_objects_first(self, loud_builder) -> None:
        """A confirmed object is worth more than a maybe.

        Dropping the confirmed one and keeping a provisional would discard the
        better evidence, which is the opposite of what a bounded budget should
        buy.
        """
        partition = empty()
        confirmed = presence_of(
            loud_builder,
            make_object(object_id="confirmed", observation_count=50),
            seq=1,
        )
        partition = fold(partition, confirmed)
        for i in range(40):
            provisional = presence_of(
                loud_builder,
                make_object(
                    object_id=f"maybe-{i}", lifecycle=LifecycleState.PROVISIONAL
                ),
                seq=i + 2,
            )
            partition = fold(partition, provisional)
        assert ObjectId("confirmed") in partition.objects


class TestCoverageProjection:
    def test_a_coverage_observation_updates_partition_status(
        self, loud_builder
    ) -> None:
        blind = loud_builder.build_coverage(
            context(),
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(1),
            effective_rate=0.0,
        )
        partition = fold(empty(), blind)
        assert partition.observability.status is ObservabilityStatus.BLIND
        assert partition.observability.reason is ObservabilityReason.STREAM_DISCONNECTED

    def test_a_coverage_observation_needs_no_object(self, loud_builder) -> None:
        """It is about the camera, not about a thing in front of it."""
        observation = loud_builder.build_coverage(
            context(),
            status=ObservabilityStatus.DEGRADED,
            reason=ObservabilityReason.SCENE_OBSCURED,
            since=at(1),
            effective_rate=0.5,
        )
        assert observation.object_id is None
        assert fold(empty(), observation).observability.effective_rate == 0.5

    def test_recovery_is_projected_as_well_as_failure(self, loud_builder) -> None:
        """A platform that recorded going blind but not coming back would report
        a permanent outage after a transient one.
        """
        partition = fold(
            empty(),
            loud_builder.build_coverage(
                context(seq=1),
                status=ObservabilityStatus.BLIND,
                reason=ObservabilityReason.STREAM_DISCONNECTED,
                since=at(1),
                effective_rate=0.0,
            ),
            loud_builder.build_coverage(
                context(seq=5),
                status=ObservabilityStatus.OBSERVING,
                reason=ObservabilityReason.NORMAL,
                since=at(5),
                effective_rate=1.0,
            ),
        )
        assert partition.observability.status is ObservabilityStatus.OBSERVING


class TestPoisonIsQuarantinedNotFatal:
    """§M12: *"quarantine that observation, continue the projection, alarm."*"""

    def test_an_objectless_presence_cannot_even_be_constructed(
        self, loud_builder
    ) -> None:
        """Stronger than quarantine: this poison never exists.

        The projection defends against it anyway — a decoded log record could
        arrive from a future schema — but the type refuses first, so no code path
        in this process can produce one.
        """
        with pytest.raises(ValueError, match="must name its object"):
            dataclasses.replace(presence_of(loud_builder), object_id=None)

    def test_a_projection_error_names_the_observation(self, loud_builder) -> None:
        """An operator needs the id to find the record, not a description of it."""
        foreign = loud_builder.build_presence(
            make_object(camera=OTHER_CAMERA), context(camera=OTHER_CAMERA)
        )
        with pytest.raises(ProjectionError) as caught:
            project(empty(CAMERA), foreign, bounds=BOUNDS, position=LogPosition(1))
        assert caught.value.context["observation_id"]


class TestNoBusinessMeaningInState:
    def test_region_state_counts_and_does_not_judge(self) -> None:
        """07_STATE §10: *"would this field mean the same thing in a hospital, a
        warehouse, and a city street?"*

        ``occupancy`` passes. ``is_crowded`` does not — crowded is a threshold
        somebody chose, and the platform may not hold it.
        """
        from vision_os.core.model.vision_state import RegionState

        fields = set(RegionState.__dataclass_fields__)
        assert "occupancy" in fields
        for forbidden in ("is_crowded", "is_violation", "alert_level", "risk"):
            assert forbidden not in fields

    def test_object_state_holds_no_verdict(self) -> None:
        from vision_os.core.model.vision_state import ObjectState

        fields = set(ObjectState.__dataclass_fields__)
        for forbidden in ("is_authorized", "threat_level", "compliance", "person_name"):
            assert forbidden not in fields
