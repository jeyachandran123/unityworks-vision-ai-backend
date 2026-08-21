"""The camera partition — sole id minting, single writer, bounded everything.

Three properties carry the design and each has its own class below:

**Ids are minted here and nowhere else** (01_LAYERED section 8). Diffusing that is
how ID chaos begins, and the guard is that no other module can produce one.

**Objects are frozen; every mutation produces a new instance.** That is what makes
"single writer, multiple readers" true without readers taking locks.

**Every bound is finite.** The population is capped, spatial history is a ring,
class history is a ring. Section M7 calls unbounded history here *"the most
likely long-run memory leak in the entire platform"*.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import (
    IdentityConflictError,
    ObjectNotFoundError,
    RegistryCapacityError,
)
from vision_os.core.model.ids import ClassId, ObjectId, ulid_timestamp_ms
from vision_os.core.model.space import Box
from vision_os.core.model.visual_object import BindingMethod, LifecycleState
from vision_os.perception.registry.lifecycle import LifecyclePolicy
from vision_os.perception.registry.partition import (
    ClassDistribution,
    RegistryPartition,
    spatial_distance,
)

from ..conftest import CAMERA, PERSON, SITE, TENANT, at, spatial, track_id


def mint(partition: RegistryPartition, *, x: float = 0.3, seq: int = 0, class_id=PERSON):
    return partition.mint(
        class_id=class_id,
        confidence=0.9,
        spatial=spatial(Box(x, 0.4, x + 0.1, 0.8)),
        now=at(seq),
        class_confidence=0.9,
    )


class TestIdMinting:
    def test_ids_are_unique(self, partition) -> None:
        ids = {mint(partition, x=0.1 * i).object_id for i in range(1, 6)}
        assert len(ids) == 5

    def test_ids_are_ulids_encoding_the_injected_clock(self, partition) -> None:
        """V13: identity generation must not reintroduce hidden time."""
        record = mint(partition, seq=50)
        assert len(record.object_id) == 26
        assert ulid_timestamp_ms(record.object_id) == at(50).ns // 1_000_000

    def test_ids_sort_by_creation_time(self, partition) -> None:
        """02_VOM section 4.1: time-sortable for efficient range queries."""
        first = mint(partition, x=0.1, seq=0).object_id
        second = mint(partition, x=0.5, seq=100).object_id
        assert first < second

    def test_minting_counts_toward_the_sequence(self, partition) -> None:
        assert partition.sequence == 0
        mint(partition)
        mint(partition, x=0.6)
        assert partition.sequence == 2

    def test_a_removed_id_is_never_reissued(self, partition) -> None:
        first = mint(partition)
        partition.evict(first.object_id)
        second = mint(partition, x=0.6)
        assert second.object_id != first.object_id


class TestCapacity:
    def test_minting_past_capacity_is_refused(self, registry_provenance) -> None:
        """A runaway registry is a memory leak with a face (section M7)."""
        partition = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(max_objects_per_camera=3),
            provenance=registry_provenance,
        )
        for i in range(3):
            mint(partition, x=0.1 + i * 0.2)
        with pytest.raises(RegistryCapacityError, match="refusing to grow"):
            mint(partition, x=0.9)

    def test_capacity_is_reported(self, partition) -> None:
        assert not partition.at_capacity
        assert partition.stats().capacity == 32

    def test_saturation_tracks_the_population(self, partition) -> None:
        for i in range(8):
            mint(partition, x=0.05 + i * 0.1)
        assert partition.stats().saturation == pytest.approx(0.25)


class TestShedding:
    def test_only_provisional_objects_are_shed_candidates(self, partition) -> None:
        confirmed = mint(partition, x=0.1)
        partition.set_lifecycle(confirmed, LifecycleState.ACTIVE)
        provisional = mint(partition, x=0.6)

        candidates = partition.shed_candidates()
        assert [c.object_id for c in candidates] == [provisional.object_id]

    def test_shed_candidates_are_oldest_first(self, partition) -> None:
        first = mint(partition, x=0.1, seq=0)
        second = mint(partition, x=0.5, seq=10)
        assert [c.object_id for c in partition.shed_candidates()] == [
            first.object_id,
            second.object_id,
        ]

    def test_shedding_order_is_deterministic(self, partition) -> None:
        for i in range(5):
            mint(partition, x=0.05 + i * 0.15, seq=i)
        runs = {
            tuple(c.object_id for c in partition.shed_candidates()) for _ in range(20)
        }
        assert len(runs) == 1


class TestBoundedHistory:
    def test_spatial_history_is_a_ring(self, registry_provenance) -> None:
        partition = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(),
            provenance=registry_provenance,
            spatial_history=4,
        )
        record = mint(partition)
        for seq in range(1, 40):
            partition.observe(
                record,
                class_id=PERSON,
                class_confidence=0.9,
                spatial=spatial(Box(0.3, 0.4, 0.5, 0.8)),
                now=at(seq),
                measured=True,
            )
        assert len(record.spatial_history) == 4

    def test_class_history_is_a_ring(self, registry_provenance) -> None:
        partition = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(),
            provenance=registry_provenance,
            class_history=3,
        )
        record = mint(partition)
        for seq in range(1, 30):
            partition.observe(
                record,
                class_id=PERSON,
                class_confidence=0.9,
                spatial=spatial(Box(0.3, 0.4, 0.5, 0.8)),
                now=at(seq),
                measured=True,
            )
        assert len(record.class_history) == 3

    def test_zero_history_is_refused(self, registry_provenance) -> None:
        with pytest.raises(ValueError, match="history bounds"):
            RegistryPartition(
                CAMERA,
                tenant_id=TENANT,
                site_id=SITE,
                policy=LifecyclePolicy(),
                provenance=registry_provenance,
                spatial_history=0,
            )


class TestMeasuredVersusBelieved:
    def test_a_measured_sighting_advances_last_confirmed(self, partition) -> None:
        record = mint(partition, seq=0)
        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.9,
            spatial=spatial(Box(0.4, 0.4, 0.6, 0.8)),
            now=at(5),
            measured=True,
        )
        assert record.last_confirmed == at(5)
        assert record.last_seen == at(5)

    def test_an_unmeasured_frame_advances_only_last_seen(self, partition) -> None:
        """The object-level expression of V8 — believed, not measured."""
        record = mint(partition, seq=0)
        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.0,
            spatial=record.spatial,
            now=at(5),
            measured=False,
        )
        assert record.last_seen == at(5)
        assert record.last_confirmed == at(0)
        assert record.unmeasured_frames == 1

    def test_an_unmeasured_frame_does_not_extend_spatial_history(
        self, partition
    ) -> None:
        record = mint(partition, seq=0)
        before = len(record.spatial_history)
        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.0,
            spatial=record.spatial,
            now=at(5),
            measured=False,
        )
        assert len(record.spatial_history) == before

    def test_a_measurement_resets_the_unmeasured_counter(self, partition) -> None:
        record = mint(partition, seq=0)
        for seq in (1, 2, 3):
            partition.observe(
                record,
                class_id=PERSON,
                class_confidence=0.0,
                spatial=record.spatial,
                now=at(seq),
                measured=False,
            )
        assert record.unmeasured_frames == 3
        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.9,
            spatial=record.spatial,
            now=at(4),
            measured=True,
        )
        assert record.unmeasured_frames == 0


class TestClassResolution:
    def test_the_published_class_follows_the_distribution(self, partition) -> None:
        """Section M7 responsibility 4 — resolve flapping from the distribution."""
        record = mint(partition, class_id=PERSON)
        for seq in range(1, 6):
            partition.observe(
                record,
                class_id=PERSON,
                class_confidence=0.9,
                spatial=record.spatial,
                now=at(seq),
                measured=True,
            )
        # One dissenting frame must not flip the published class.
        partition.observe(
            record,
            class_id=ClassId("person.child"),
            class_confidence=0.4,
            spatial=record.spatial,
            now=at(6),
            measured=True,
        )
        assert record.class_id == PERSON

    def test_sustained_evidence_does_change_the_class(self, partition) -> None:
        record = mint(partition, class_id=PERSON)
        for seq in range(1, 12):
            partition.observe(
                record,
                class_id=ClassId("person.child"),
                class_confidence=0.95,
                spatial=record.spatial,
                now=at(seq),
                measured=True,
            )
        assert record.class_id == ClassId("person.child")

    def test_class_history_retains_what_was_seen(self, partition) -> None:
        """Section M7: never silently rewrite past class assertions."""
        record = mint(partition, class_id=PERSON)
        partition.observe(
            record,
            class_id=ClassId("person.child"),
            class_confidence=0.4,
            spatial=record.spatial,
            now=at(1),
            measured=True,
        )
        seen = [c.class_id for c in record.class_history]
        assert ClassId("person.child") in seen, (
            "the dissenting observation must survive even though it did not win"
        )


class TestClassDistribution:
    def test_the_heaviest_class_wins(self) -> None:
        distribution = ClassDistribution()
        distribution.observe(PERSON, 0.9)
        distribution.observe(PERSON, 0.8)
        distribution.observe(ClassId("vehicle"), 0.5)
        best = distribution.best()
        assert best is not None
        assert best[0] == PERSON

    def test_share_reports_the_evidence_fraction(self) -> None:
        distribution = ClassDistribution()
        distribution.observe(PERSON, 3.0)
        distribution.observe(ClassId("vehicle"), 1.0)
        assert distribution.share(PERSON) == pytest.approx(0.75)

    def test_an_empty_distribution_has_no_best(self) -> None:
        assert ClassDistribution().best() is None

    def test_ties_break_deterministically(self) -> None:
        """An arbitrary tie-break would make the published class depend on dict
        ordering, which is exactly the kind of non-determinism V13 forbids."""
        results = set()
        for _ in range(20):
            distribution = ClassDistribution()
            distribution.observe(ClassId("vehicle"), 1.0)
            distribution.observe(PERSON, 1.0)
            results.add(distribution.best()[0])
        assert len(results) == 1

    def test_negative_confidence_does_not_reduce_evidence(self) -> None:
        distribution = ClassDistribution()
        distribution.observe(PERSON, 1.0)
        distribution.observe(PERSON, -5.0)
        assert distribution.weights[PERSON] == pytest.approx(1.0)


class TestBindings:
    def test_opening_a_binding_records_the_track(self, partition) -> None:
        record = mint(partition)
        binding = partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        assert record.bound_track == track_id(7)
        assert binding.is_open

    def test_a_new_binding_closes_the_previous_one(self, partition) -> None:
        record = mint(partition)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        partition.open_binding(
            record,
            track_id=track_id(8),
            method=BindingMethod.SPATIO_TEMPORAL,
            confidence=0.7,
            now=at(5),
        )
        assert len(record.bindings) == 2
        assert record.bindings[0].bound_to == at(5)
        assert record.bound_track == track_id(8)

    def test_closed_bindings_are_retained_not_deleted(self, partition) -> None:
        """V5: which track contributed which interval must stay reconstructible."""
        record = mint(partition)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        partition.close_bindings(record, now=at(5))
        assert len(record.bindings) == 1
        assert record.bindings[0].bound_to == at(5)
        assert record.bound_track is None

    def test_superseding_marks_the_original(self, partition) -> None:
        record = mint(partition)
        first = partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        partition.open_binding(
            record,
            track_id=track_id(8),
            method=BindingMethod.SPATIO_TEMPORAL,
            confidence=0.7,
            now=at(5),
            supersedes=first.binding_id,
        )
        superseded = [b for b in record.bindings if b.binding_id == first.binding_id][0]
        assert superseded.superseded_by is not None

    def test_by_track_finds_the_bound_object(self, partition) -> None:
        record = mint(partition)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        assert partition.by_track(track_id(7)) is record
        assert partition.by_track(track_id(99)) is None

    def test_a_terminal_object_is_not_found_by_track(self, partition) -> None:
        record = mint(partition)
        partition.open_binding(
            record,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        partition.set_lifecycle(record, LifecycleState.EXPIRED)
        assert partition.by_track(track_id(7)) is None


class TestMerge:
    def test_merging_points_source_at_target(self, partition) -> None:
        source = mint(partition, x=0.1)
        target = mint(partition, x=0.6)
        partition.merge(source=source, target=target, now=at(10))
        assert source.lifecycle is LifecycleState.MERGED_INTO
        assert source.merged_into == target.object_id

    def test_merging_preserves_the_source_record(self, partition) -> None:
        """V5: observations referencing the old id stay resolvable."""
        source = mint(partition, x=0.1)
        target = mint(partition, x=0.6)
        partition.merge(source=source, target=target, now=at(10))
        assert source.object_id in partition
        assert partition.find(source.object_id) is not None

    def test_resolve_follows_the_merge(self, partition) -> None:
        source = mint(partition, x=0.1)
        target = mint(partition, x=0.6)
        partition.merge(source=source, target=target, now=at(10))
        assert partition.resolve(source.object_id).object_id == target.object_id

    def test_resolve_follows_a_merge_chain(self, partition) -> None:
        first = mint(partition, x=0.1)
        second = mint(partition, x=0.4)
        third = mint(partition, x=0.7)
        partition.merge(source=first, target=second, now=at(10))
        partition.merge(source=second, target=third, now=at(11))
        assert partition.resolve(first.object_id).object_id == third.object_id

    def test_the_target_inherits_lineage_and_evidence(self, partition) -> None:
        source = mint(partition, x=0.1, seq=0)
        target = mint(partition, x=0.6, seq=20)
        source.observation_count = 7
        partition.merge(source=source, target=target, now=at(30))
        assert source.object_id in target.lineage
        assert target.observation_count >= 8
        assert target.first_seen == at(0), (
            "the survivor inherits the earlier first_seen; the object existed then"
        )

    def test_merging_into_itself_is_refused(self, partition) -> None:
        record = mint(partition)
        with pytest.raises(IdentityConflictError, match="into itself"):
            partition.merge(source=record, target=record, now=at(10))

    def test_merging_into_an_already_merged_target_is_refused(self, partition) -> None:
        """The surviving identity would be ambiguous."""
        first = mint(partition, x=0.1)
        second = mint(partition, x=0.4)
        third = mint(partition, x=0.7)
        partition.merge(source=second, target=third, now=at(10))
        with pytest.raises(IdentityConflictError, match="ambiguous"):
            partition.merge(source=first, target=second, now=at(11))

    def test_a_cross_partition_merge_is_refused(self, registry_provenance) -> None:
        """Taking a lock across camera partitions reintroduces global contention."""
        from vision_os.core.model.ids import CameraId

        first = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(),
            provenance=registry_provenance,
        )
        second = RegistryPartition(
            CameraId("cam-02"),
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(),
            provenance=registry_provenance,
        )
        source = mint(first)
        target = mint(second)
        with pytest.raises(IdentityConflictError, match="two-phase"):
            first.merge(source=source, target=target, now=at(10))

    def test_merging_closes_the_source_bindings(self, partition) -> None:
        source = mint(partition, x=0.1)
        target = mint(partition, x=0.6)
        partition.open_binding(
            source,
            track_id=track_id(7),
            method=BindingMethod.FIRST_SIGHT,
            confidence=0.9,
            now=at(1),
        )
        partition.merge(source=source, target=target, now=at(10))
        assert source.bound_track is None


class TestReadsAndProjection:
    def test_get_raises_for_an_unknown_object(self, partition) -> None:
        with pytest.raises(ObjectNotFoundError, match="not in partition"):
            partition.get(ObjectId("01JB0000000000000000000099"))

    def test_find_returns_none_for_an_unknown_object(self, partition) -> None:
        assert partition.find(ObjectId("01JB0000000000000000000099")) is None

    def test_projection_is_immutable(self, partition) -> None:
        record = mint(partition)
        projected = partition.get(record.object_id)
        with pytest.raises(AttributeError):
            projected.class_id = ClassId("vehicle")  # type: ignore[misc]

    def test_projection_carries_the_injected_provenance(
        self, partition, registry_provenance
    ) -> None:
        record = mint(partition)
        assert partition.get(record.object_id).provenance == registry_provenance

    def test_two_reads_give_equal_but_independent_snapshots(self, partition) -> None:
        record = mint(partition)
        first = partition.get(record.object_id)
        second = partition.get(record.object_id)
        assert first == second

    def test_a_snapshot_does_not_drift_when_the_record_changes(self, partition) -> None:
        """Single writer, multiple readers — a reader's view cannot move."""
        record = mint(partition, seq=0)
        snapshot = partition.get(record.object_id)
        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.9,
            spatial=spatial(Box(0.7, 0.4, 0.9, 0.8)),
            now=at(9),
            measured=True,
        )
        assert snapshot.last_confirmed == at(0)
        assert partition.get(record.object_id).last_confirmed == at(9)

    def test_active_excludes_departed_and_terminal_objects(self, partition) -> None:
        present = mint(partition, x=0.1)
        partition.set_lifecycle(present, LifecycleState.ACTIVE)
        departed = mint(partition, x=0.4)
        partition.set_lifecycle(departed, LifecycleState.DEPARTED)
        merged = mint(partition, x=0.7)
        partition.set_lifecycle(merged, LifecycleState.ACTIVE)
        partition.merge(source=merged, target=present, now=at(5))

        active_ids = {o.object_id for o in partition.active()}
        assert active_ids == {present.object_id}

    def test_objects_come_back_in_stable_id_order(self, partition) -> None:
        for i in range(5):
            mint(partition, x=0.05 + i * 0.15, seq=i)
        first = [o.object_id for o in partition.objects()]
        assert first == sorted(first)


class TestVersioning:
    def test_the_version_advances_on_every_mutation(self, partition) -> None:
        start = partition.version
        record = mint(partition)
        after_mint = partition.version
        assert after_mint > start

        partition.observe(
            record,
            class_id=PERSON,
            class_confidence=0.9,
            spatial=record.spatial,
            now=at(1),
            measured=True,
        )
        assert partition.version > after_mint

    def test_reads_do_not_advance_the_version(self, partition) -> None:
        record = mint(partition)
        version = partition.version
        partition.get(record.object_id)
        partition.objects()
        partition.active()
        assert partition.version == version

    def test_closing_no_bindings_does_not_advance_the_version(self, partition) -> None:
        record = mint(partition)
        version = partition.version
        partition.close_bindings(record, now=at(5))
        assert partition.version == version


class TestAdoption:
    def test_adopting_a_duplicate_is_refused(self, partition) -> None:
        record = mint(partition)
        with pytest.raises(IdentityConflictError, match="already exists"):
            partition.adopt(record)

    def test_adopting_past_capacity_is_refused(self, registry_provenance) -> None:
        small = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(max_objects_per_camera=1),
            provenance=registry_provenance,
        )
        other = RegistryPartition(
            CAMERA,
            tenant_id=TENANT,
            site_id=SITE,
            policy=LifecyclePolicy(),
            provenance=registry_provenance,
        )
        mint(small)
        stranger = mint(other, x=0.8)
        with pytest.raises(RegistryCapacityError):
            small.adopt(stranger)


class TestGeometryHelper:
    def test_identical_spatial_claims_are_zero_apart(self) -> None:
        info = spatial(Box(0.3, 0.4, 0.5, 0.8))
        assert spatial_distance(info, info) == 0.0

    def test_a_missing_box_is_maximally_distant(self) -> None:
        """'Cannot compare' must not read as 'identical'."""
        from vision_os.core.model.space import FrameOfReference, SpatialInfo

        empty = SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED)
        assert spatial_distance(empty, spatial(Box(0.3, 0.4, 0.5, 0.8))) == 1.0

    def test_distance_is_normalized(self) -> None:
        near = spatial(Box(0.0, 0.0, 0.1, 0.1))
        far = spatial(Box(0.9, 0.9, 1.0, 1.0))
        assert 0.0 < spatial_distance(near, far) <= 1.0


class TestStats:
    def test_stats_break_down_by_lifecycle(self, partition) -> None:
        first = mint(partition, x=0.1)
        partition.set_lifecycle(first, LifecycleState.ACTIVE)
        mint(partition, x=0.6)
        stats = partition.stats()
        assert stats.by_state[LifecycleState.ACTIVE] == 1
        assert stats.by_state[LifecycleState.PROVISIONAL] == 1
        assert stats.total == 2

    def test_present_counts_only_believed_present_objects(self, partition) -> None:
        active = mint(partition, x=0.1)
        partition.set_lifecycle(active, LifecycleState.ACTIVE)
        departed = mint(partition, x=0.6)
        partition.set_lifecycle(departed, LifecycleState.DEPARTED)
        assert partition.stats().present == 1

    def test_ids_minted_is_reported(self, partition) -> None:
        mint(partition, x=0.1)
        mint(partition, x=0.6)
        assert partition.stats().ids_minted == 2
