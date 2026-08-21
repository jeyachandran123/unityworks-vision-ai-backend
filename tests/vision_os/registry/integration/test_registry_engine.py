"""The Object Registry engine — the M7 public API end to end.

``ingest`` is a firewall: it never raises, because a registry failure may not
stop tracking, which may not stop detection, which may not stop acquisition (V9).
Everything else raises on misuse, because those are direct API calls where a
caller can and should handle the error.

The tests that matter most are the ones asserting **refusals**: merge without
deletion, ambiguity without a guess, attributes without production, and the
registry as the only writer.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import (
    AttributeRejectedError,
    IdentityConflictError,
    ObjectNotFoundError,
)
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import (
    AttributeKey,
    ClassId,
    ConfigRevision,
    ModuleId,
    ObjectId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box
from vision_os.core.model.visual_object import (
    Attribute,
    BindingMethod,
    LifecycleState,
)
from vision_os.kernel.events import (
    IdentityAsserted,
    IdentityRevised,
    ObjectCreated,
    ObjectLifecycleChanged,
    ObjectPopulationCapped,
    RegionTransition,
)
from vision_os.kernel.metrics import MetricName
from vision_os.perception.registry.attributes import (
    AttributeSchema,
    AttributeValueType,
)

from ..conftest import (
    CAMERA,
    OTHER_CAMERA,
    PERSON,
    age,
    at,
    coast,
    drive,
    make_region,
    make_track,
    make_update,
    walking,
)


def ingest_one(registry, *, local=0, box=None, seq=0, camera=CAMERA, epoch=0, measured=True):
    track = make_track(
        local=local, box=box or walking(seq), seq=seq, camera=camera,
        epoch=epoch, measured=measured,
    )
    return registry.ingest(camera, make_update([track], seq=seq, camera=camera, epoch=epoch))


class TestIngestion:
    def test_a_track_becomes_an_object(self, registry) -> None:
        result = ingest_one(registry)
        assert not result.failed
        assert result.count == 1
        assert len(result.created) == 1

    def test_the_object_starts_provisional(self, registry) -> None:
        result = ingest_one(registry)
        assert result.objects[0].lifecycle is LifecycleState.PROVISIONAL

    def test_it_confirms_after_the_observation_threshold(self, registry) -> None:
        results = drive(registry, 5)
        assert results[-1].objects[0].lifecycle is LifecycleState.ACTIVE

    def test_one_track_produces_exactly_one_object(self, registry) -> None:
        results = drive(registry, 12)
        assert sum(len(r.created) for r in results) == 1
        assert len(registry.active(CAMERA)) == 1

    def test_the_object_is_bound_to_its_track(self, registry) -> None:
        drive(registry, 4)
        obj = registry.active(CAMERA)[0]
        assert obj.bound_track is not None
        assert obj.open_binding.method is BindingMethod.FIRST_SIGHT

    def test_observation_count_accumulates(self, registry) -> None:
        drive(registry, 7)
        assert registry.active(CAMERA)[0].observation_count == 7

    def test_two_separated_tracks_produce_two_objects(self, registry) -> None:
        for seq in range(5):
            tracks = [
                make_track(local=0, box=Box(0.1, 0.4, 0.2, 0.8), seq=seq),
                make_track(local=1, box=Box(0.7, 0.4, 0.8, 0.8), seq=seq),
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
        assert len(registry.active(CAMERA)) == 2

    def test_an_empty_frame_is_not_a_failure(self, registry) -> None:
        result = registry.ingest(CAMERA, make_update([], seq=0))
        assert not result.failed
        assert result.count == 0

    def test_a_predicted_track_does_not_advance_last_confirmed(self, registry) -> None:
        """V8 at object scale — believed is not measured."""
        drive(registry, 5)
        confirmed = registry.active(CAMERA)[0].last_confirmed
        ingest_one(registry, seq=10, measured=False)
        obj = registry.active(CAMERA)[0]
        assert obj.last_confirmed == confirmed
        assert obj.last_seen.ns > confirmed.ns


class TestLifecycleProgression:
    def test_an_unmeasured_object_becomes_occluded(self, registry) -> None:
        drive(registry, 5)
        coast(registry, 1, start=10)
        assert registry.objects(CAMERA)[0].lifecycle is LifecycleState.OCCLUDED

    def test_it_becomes_dormant_past_the_occlusion_horizon(self, registry) -> None:
        drive(registry, 5)
        coast(registry, 1, start=10)
        age(registry, 19)
        assert registry.objects(CAMERA)[0].lifecycle in (
            LifecycleState.DORMANT,
            LifecycleState.DEPARTED,
        )

    def test_it_departs_and_then_expires(self, registry) -> None:
        drive(registry, 5)
        coast(registry, 1, start=10)
        age(registry, 200)
        assert registry.objects(CAMERA) == (), (
            "an object past its retention horizon must leave the population"
        )

    def test_an_object_leaving_the_frame_goes_dormant_not_occluded(
        self, registry
    ) -> None:
        """Different claims; the diagram has separate edges for a reason."""
        for seq in range(5):
            registry.ingest(
                CAMERA,
                make_update(
                    [make_track(box=Box(0.90, 0.4, 0.99, 0.8), seq=seq)], seq=seq
                ),
            )
        coast(registry, 1, start=10)
        assert registry.objects(CAMERA)[0].lifecycle is LifecycleState.DORMANT

    def test_every_transition_is_published(self, registry, bus) -> None:
        subscription = bus.subscribe([ObjectLifecycleChanged])
        drive(registry, 5)
        events = subscription.drain()
        assert events
        assert all(e.previous and e.current for e in events)
        assert all(e.trigger for e in events), "a transition without a trigger is unexplained"


class TestReEntry:
    def test_a_dormant_object_is_re_bound_by_proximity(self, registry) -> None:
        drive(registry, 5)
        original = registry.active(CAMERA)[0].object_id
        coast(registry, 1, start=10)
        age(registry, 19)
        assert registry.objects(CAMERA)[0].lifecycle is LifecycleState.DORMANT

        # A new track appears where the object was last seen.
        last_box = registry.objects(CAMERA)[0].current_spatial.bbox
        result = registry.ingest(
            CAMERA,
            make_update([make_track(local=9, box=last_box, seq=20)], seq=20),
        )
        assert not result.failed
        assert result.created == (), "an existing object must be re-bound, not duplicated"
        assert registry.active(CAMERA)[0].object_id == original

    def test_a_re_bind_is_published_as_an_assertion(self, registry, bus) -> None:
        drive(registry, 5)
        coast(registry, 1, start=10)
        age(registry, 19)
        subscription = bus.subscribe([IdentityAsserted])
        last_box = registry.objects(CAMERA)[0].current_spatial.bbox
        registry.ingest(
            CAMERA, make_update([make_track(local=9, box=last_box, seq=20)], seq=20)
        )
        events = subscription.drain()
        assert events
        assert events[0].method == BindingMethod.SPATIO_TEMPORAL.value

    def test_a_distant_new_track_mints_a_new_object(self, registry) -> None:
        drive(registry, 5)
        coast(registry, 1, start=10)
        age(registry, 19)
        result = registry.ingest(
            CAMERA,
            make_update(
                [make_track(local=9, box=Box(0.85, 0.05, 0.95, 0.25), seq=41)], seq=41
            ),
        )
        assert len(result.created) == 1


class TestAmbiguityIsNeverGuessed:
    """Section M7: create a new object and publish the alternatives."""

    def _two_dormant_neighbours(self, registry):
        for seq in range(5):
            tracks = [
                make_track(local=0, box=Box(0.40, 0.4, 0.50, 0.8), seq=seq),
                make_track(local=1, box=Box(0.42, 0.4, 0.52, 0.8), seq=seq),
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
        coast(registry, 1, start=10)
        age(registry, 19)

    def test_an_ambiguous_re_entry_mints_a_new_object(self, registry) -> None:
        self._two_dormant_neighbours(registry)
        result = registry.ingest(
            CAMERA,
            make_update(
                [make_track(local=9, box=Box(0.41, 0.4, 0.51, 0.8), seq=23)], seq=23
            ),
        )
        assert len(result.created) == 1, (
            "an ambiguous re-entry must mint a new object rather than guess"
        )

    def test_the_alternatives_are_published(self, registry) -> None:
        self._two_dormant_neighbours(registry)
        result = registry.ingest(
            CAMERA,
            make_update(
                [make_track(local=9, box=Box(0.41, 0.4, 0.51, 0.8), seq=23)], seq=23
            ),
        )
        ambiguous = result.ambiguous_assertions
        assert ambiguous, "the candidates must survive the decision"
        assert len(ambiguous[0].alternatives) >= 2

    def test_ambiguity_is_counted(self, registry, metrics) -> None:
        self._two_dormant_neighbours(registry)
        registry.ingest(
            CAMERA,
            make_update(
                [make_track(local=9, box=Box(0.41, 0.4, 0.51, 0.8), seq=23)], seq=23
            ),
        )
        assert metrics.snapshot().counters_matching(MetricName.IDENTITY_AMBIGUITIES)

    def test_the_ambiguous_assertion_carries_low_confidence(self, registry) -> None:
        """A consumer counting unique visitors filters it out; one drawing an
        overlay keeps it. The platform does not choose (V1)."""
        self._two_dormant_neighbours(registry)
        result = registry.ingest(
            CAMERA,
            make_update(
                [make_track(local=9, box=Box(0.41, 0.4, 0.51, 0.8), seq=23)], seq=23
            ),
        )
        assertion = result.ambiguous_assertions[0]
        assert assertion.confidence.value < 1.0
        assert assertion.confidence.semantics is ConfidenceSemantics.IDENTITY


class TestMergePreservesHistory:
    """V5, and 14_TESTING section 4 names it an M7 invariant."""

    def _two_objects(self, registry):
        for seq in range(5):
            tracks = [
                make_track(local=0, box=Box(0.1, 0.4, 0.2, 0.8), seq=seq),
                make_track(local=1, box=Box(0.7, 0.4, 0.8, 0.8), seq=seq),
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
        objects = registry.active(CAMERA)
        return objects[0].object_id, objects[1].object_id

    def test_merging_returns_the_survivor(self, registry) -> None:
        source, target = self._two_objects(registry)
        assert registry.merge(source, target, evidence="test") == target

    def test_the_source_is_not_deleted(self, registry) -> None:
        source, target = self._two_objects(registry)
        registry.merge(source, target)
        survivor = registry.get(source)
        assert survivor.lifecycle is LifecycleState.MERGED_INTO
        assert survivor.merged_into == target

    def test_the_source_resolves_to_the_target(self, registry) -> None:
        """An observation referencing the old id remains answerable."""
        source, target = self._two_objects(registry)
        registry.merge(source, target)
        assert registry.resolve(source).object_id == target

    def test_the_target_records_the_lineage(self, registry) -> None:
        source, target = self._two_objects(registry)
        registry.merge(source, target)
        assert source in registry.get(target).lineage

    def test_the_merged_object_leaves_the_active_set(self, registry) -> None:
        source, target = self._two_objects(registry)
        registry.merge(source, target)
        active = {o.object_id for o in registry.active(CAMERA)}
        assert source not in active
        assert target in active

    def test_a_merge_is_published_as_a_revision(self, registry, bus) -> None:
        subscription = bus.subscribe([IdentityRevised])
        source, target = self._two_objects(registry)
        registry.merge(source, target, evidence="operator review")
        events = subscription.drain()
        assert events
        assert events[0].reason == "merge"
        assert events[0].evidence == "operator review"

    def test_a_cross_partition_merge_is_refused(self, registry) -> None:
        """Two-phase and eventually consistent, never a synchronous write."""
        drive(registry, 5, camera=CAMERA)
        drive(registry, 5, camera=OTHER_CAMERA)
        first = registry.active(CAMERA)[0].object_id
        second = registry.active(OTHER_CAMERA)[0].object_id
        with pytest.raises(IdentityConflictError, match="two-phase"):
            registry.merge(first, second)

    def test_merging_an_unknown_object_is_refused(self, registry) -> None:
        source, _ = self._two_objects(registry)
        with pytest.raises(ObjectNotFoundError):
            registry.merge(source, ObjectId("01JB0000000000000000000099"))

    def test_merging_is_counted(self, registry, metrics) -> None:
        source, target = self._two_objects(registry)
        registry.merge(source, target)
        assert metrics.snapshot().counters_matching(MetricName.OBJECTS_MERGED)


class TestSplit:
    def test_splitting_produces_two_ids(self, registry) -> None:
        drive(registry, 10)
        original = registry.active(CAMERA)[0].object_id
        kept, successor = registry.split(original, at=at(5), evidence="test")
        assert kept == original
        assert successor != original

    def test_the_successor_records_its_parent(self, registry) -> None:
        drive(registry, 10)
        original = registry.active(CAMERA)[0].object_id
        _, successor = registry.split(original, at=at(5))
        assert original in registry.get(successor).lineage

    def test_the_original_keeps_the_earlier_interval(self, registry) -> None:
        drive(registry, 10)
        original = registry.active(CAMERA)[0].object_id
        registry.split(original, at=at(5))
        assert registry.get(original).last_seen.ns <= at(5).ns

    def test_splitting_outside_the_lifetime_is_refused(self, registry) -> None:
        """One side would be empty; a rename pretending to be a correction."""
        drive(registry, 10)
        original = registry.active(CAMERA)[0].object_id
        with pytest.raises(IdentityConflictError, match="outside"):
            registry.split(original, at=at(500))

    def test_a_split_is_published_as_a_revision(self, registry, bus) -> None:
        subscription = bus.subscribe([IdentityRevised])
        drive(registry, 10)
        original = registry.active(CAMERA)[0].object_id
        registry.split(original, at=at(5))
        events = subscription.drain()
        assert events
        assert events[0].reason == "split"


class TestAttributesAreHeldNotProduced:
    def _register(self, registry, key="posture"):
        registry.attribute_registry.register(
            AttributeSchema(
                key=AttributeKey(key),
                value_type=AttributeValueType.ENUM,
                domain=("standing", "sitting"),
                neutrality_justification="Body configuration is directly visible",
                applies_to=(PERSON,),
            )
        )

    def _attribute(self, key="posture", value="standing"):
        return Attribute(
            key=AttributeKey(key),
            schema_version="1.0.0",
            value=value,
            confidence=Confidence.uncalibrated(0.8, ConfidenceSemantics.ATTRIBUTE),
            observed_at=at(5),
            producer=Provenance(
                producer_module=ModuleId("understanding_engine"),
                producer_version="1.0.0",
                config_revision=ConfigRevision("test"),
            ),
        )

    def test_a_registered_attribute_is_held(self, registry) -> None:
        drive(registry, 5)
        self._register(registry)
        obj_id = registry.active(CAMERA)[0].object_id
        registry.apply_attribute(obj_id, self._attribute())
        assert registry.get(obj_id).attribute(AttributeKey("posture")) is not None

    def test_an_unregistered_attribute_is_refused(self, registry) -> None:
        """The gate is worthless if a producer can bypass it by not asking."""
        drive(registry, 5)
        obj_id = registry.active(CAMERA)[0].object_id
        with pytest.raises(AttributeRejectedError, match="not registered"):
            registry.apply_attribute(obj_id, self._attribute())

    def test_an_attribute_for_the_wrong_class_is_refused(self, registry) -> None:
        drive(registry, 5)
        registry.attribute_registry.register(
            AttributeSchema(
                key=AttributeKey("wheel_count"),
                value_type=AttributeValueType.COUNT,
                neutrality_justification="Wheels are directly visible on the vehicle",
                applies_to=(ClassId("vehicle"),),
            )
        )
        obj_id = registry.active(CAMERA)[0].object_id
        with pytest.raises(AttributeRejectedError, match="does not apply"):
            registry.apply_attribute(obj_id, self._attribute(key="wheel_count", value=4))

    def test_applying_to_an_unknown_object_is_refused(self, registry) -> None:
        self._register(registry)
        with pytest.raises(ObjectNotFoundError):
            registry.apply_attribute(
                ObjectId("01JB0000000000000000000099"), self._attribute()
            )

    def test_the_registry_produces_no_attributes_itself(self, registry) -> None:
        """M7 holds; M9 produces. Holding is storage, producing is inference."""
        drive(registry, 8)
        assert registry.active(CAMERA)[0].attributes == {}

    def test_applying_is_counted(self, registry, metrics) -> None:
        drive(registry, 5)
        self._register(registry)
        obj_id = registry.active(CAMERA)[0].object_id
        registry.apply_attribute(obj_id, self._attribute())
        assert metrics.snapshot().counters_matching(MetricName.ATTRIBUTES_APPLIED)


class TestRegions:
    def test_entering_a_region_is_published(self, registry, bus) -> None:
        registry.set_regions(CAMERA, (make_region(),))
        subscription = bus.subscribe([RegionTransition])
        for seq in range(3):
            registry.ingest(
                CAMERA,
                make_update([make_track(box=Box(0.45, 0.5, 0.55, 0.7), seq=seq)], seq=seq),
            )
        events = subscription.drain()
        assert events
        assert events[0].entered
        assert events[0].region_id == "Z3"

    def test_leaving_publishes_the_dwell(self, registry, bus) -> None:
        registry.set_regions(CAMERA, (make_region(),))
        subscription = bus.subscribe([RegionTransition])
        for seq in range(5):
            registry.ingest(
                CAMERA,
                make_update([make_track(box=Box(0.45, 0.5, 0.55, 0.7), seq=seq)], seq=seq),
            )
        registry.ingest(
            CAMERA,
            make_update([make_track(box=Box(0.02, 0.02, 0.08, 0.10), seq=10)], seq=10),
        )
        exits = [e for e in subscription.drain() if not e.entered]
        assert exits
        assert exits[0].dwell_ms > 0

    def test_region_transitions_are_counted(self, registry, metrics) -> None:
        registry.set_regions(CAMERA, (make_region(),))
        for seq in range(3):
            registry.ingest(
                CAMERA,
                make_update([make_track(box=Box(0.45, 0.5, 0.55, 0.7), seq=seq)], seq=seq),
            )
        assert metrics.snapshot().counters_matching(MetricName.REGION_TRANSITIONS)

    def test_changing_geometry_closes_accumulations(self, registry, bus) -> None:
        registry.set_regions(CAMERA, (make_region(version="1.0.0"),))
        for seq in range(5):
            registry.ingest(
                CAMERA,
                make_update([make_track(box=Box(0.45, 0.5, 0.55, 0.7), seq=seq)], seq=seq),
            )
        subscription = bus.subscribe([RegionTransition])
        registry.set_regions(CAMERA, (make_region(version="2.0.0"),))
        closed = subscription.drain()
        assert closed
        assert closed[0].geometry_version == "1.0.0"


class TestCapacityAndShedding:
    def _small_registry(self, clock, bus, metrics, registry_config, registry_provenance):
        from vision_os.perception.registry import LifecyclePolicy, ObjectRegistry

        from ..conftest import SITE, TENANT

        return ObjectRegistry(
            clock=clock,
            bus=bus,
            metrics=metrics,
            config=registry_config,
            tenant_id=TENANT,
            site_id=SITE,
            provenance=registry_provenance,
            lifecycle=LifecyclePolicy(
                min_observations_to_confirm=3, max_objects_per_camera=3
            ),
        )

    def test_provisional_objects_are_shed_first(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = self._small_registry(
            clock, bus, metrics, registry_config, registry_provenance
        )
        tracks = [
            make_track(local=i, box=Box(0.05 + i * 0.2, 0.4, 0.13 + i * 0.2, 0.8), seq=0)
            for i in range(4)
        ]
        result = registry.ingest(CAMERA, make_update(tracks, seq=0))
        assert not result.failed
        assert len(registry.objects(CAMERA)) <= 3

    def test_the_cap_is_alarmed(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        """Section M7 requires an alarm — a runaway registry is a memory leak."""
        registry = self._small_registry(
            clock, bus, metrics, registry_config, registry_provenance
        )
        subscription = bus.subscribe([ObjectPopulationCapped])
        tracks = [
            make_track(local=i, box=Box(0.05 + i * 0.2, 0.4, 0.13 + i * 0.2, 0.8), seq=0)
            for i in range(5)
        ]
        registry.ingest(CAMERA, make_update(tracks, seq=0))
        assert subscription.drain()

    def test_the_population_never_exceeds_the_cap(
        self, clock, bus, metrics, registry_config, registry_provenance
    ) -> None:
        registry = self._small_registry(
            clock, bus, metrics, registry_config, registry_provenance
        )
        for seq in range(20):
            tracks = [
                make_track(
                    local=seq * 10 + i,
                    box=Box(0.05 + i * 0.2, 0.4, 0.13 + i * 0.2, 0.8),
                    seq=seq,
                )
                for i in range(4)
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
            assert len(registry.objects(CAMERA)) <= 3


class TestNeverRaises:
    """V9 — a registry failure may not stop the pipeline."""

    def test_an_exploding_partition_does_not_raise(self, registry, monkeypatch) -> None:
        drive(registry, 3)

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(registry, "_partition", explode)
        result = ingest_one(registry, seq=9)
        assert result.failed
        assert "RuntimeError" in result.reason

    def test_a_failure_is_counted(self, registry, metrics, monkeypatch) -> None:
        monkeypatch.setattr(
            registry, "_partition", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        ingest_one(registry)
        assert metrics.snapshot().counters_matching(MetricName.REGISTRY_FAILURES)

    def test_a_failure_degrades_health(self, registry, monkeypatch) -> None:
        monkeypatch.setattr(
            registry, "_partition", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
        )
        ingest_one(registry)
        assert registry.health().state.value == "degraded"

    def test_the_registry_recovers_after_a_transient_failure(
        self, registry, monkeypatch
    ) -> None:
        calls = {"n": 0}
        original = registry._partition  # noqa: SLF001

        def flaky(camera_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return original(camera_id)

        monkeypatch.setattr(registry, "_partition", flaky)
        assert ingest_one(registry, seq=0).failed
        assert not ingest_one(registry, seq=1).failed


class TestReadsAndScope:
    def test_get_raises_for_an_unknown_object(self, registry) -> None:
        with pytest.raises(ObjectNotFoundError):
            registry.get(ObjectId("01JB0000000000000000000099"))

    def test_active_is_scoped_by_camera(self, registry) -> None:
        drive(registry, 5, camera=CAMERA)
        drive(registry, 5, camera=OTHER_CAMERA)
        assert len(registry.active(CAMERA)) == 1
        assert len(registry.active(OTHER_CAMERA)) == 1
        assert len(registry.active()) == 2

    def test_active_of_an_unknown_camera_is_empty(self, registry) -> None:
        from vision_os.core.model.ids import CameraId

        assert registry.active(CameraId("never-seen")) == ()

    def test_objects_includes_terminal_ones(self, registry) -> None:
        for seq in range(5):
            tracks = [
                make_track(local=0, box=Box(0.1, 0.4, 0.2, 0.8), seq=seq),
                make_track(local=1, box=Box(0.7, 0.4, 0.8, 0.8), seq=seq),
            ]
            registry.ingest(CAMERA, make_update(tracks, seq=seq))
        objects = registry.active(CAMERA)
        registry.merge(objects[0].object_id, objects[1].object_id)
        assert len(registry.objects(CAMERA)) == 2
        assert len(registry.active(CAMERA)) == 1

    def test_partitions_are_reported(self, registry) -> None:
        drive(registry, 2, camera=CAMERA)
        drive(registry, 2, camera=OTHER_CAMERA)
        assert registry.partitions == (CAMERA, OTHER_CAMERA)


class TestExpiry:
    def test_expire_stale_advances_horizons(self, registry) -> None:
        drive(registry, 5)
        registry.expire_stale(at(1_000))
        assert registry.objects(CAMERA) == () or registry.objects(CAMERA)[
            0
        ].lifecycle in (LifecycleState.DEPARTED, LifecycleState.EXPIRED)

    def test_expiry_removes_objects_past_retention(self, registry) -> None:
        drive(registry, 5)
        registry.expire_stale(at(10_000))
        assert registry.objects(CAMERA) == ()

    def test_expiry_is_published(self, registry, bus) -> None:
        drive(registry, 5)
        subscription = bus.subscribe([ObjectLifecycleChanged])
        registry.expire_stale(at(10_000))
        currents = {e.current for e in subscription.drain()}
        assert "expired" in currents

    def test_expiry_of_an_empty_registry_is_safe(self, registry) -> None:
        assert registry.expire_stale(at(100)) == ()


class TestEvents:
    def test_object_creation_is_published(self, registry, bus) -> None:
        subscription = bus.subscribe([ObjectCreated])
        drive(registry, 3)
        events = subscription.drain()
        assert len(events) == 1
        assert events[0].camera_id == CAMERA
        assert events[0].object_id

    def test_identity_assertions_are_published(self, registry, bus) -> None:
        subscription = bus.subscribe([IdentityAsserted])
        drive(registry, 3)
        events = subscription.drain()
        assert events
        assert events[0].method == BindingMethod.FIRST_SIGHT.value
        assert events[0].confidence > 0

    def test_no_semantic_event_type_exists(self) -> None:
        """The registry publishes what happened, never what it means."""
        from vision_os.kernel.events import ALL_EVENT_TYPES

        registry_events = {
            t.event_type for t in ALL_EVENT_TYPES if t.event_type.startswith("registry.")
        }
        assert registry_events == {
            "registry.object_created",
            "registry.lifecycle_changed",
            "registry.identity_asserted",
            "registry.identity_revised",
            "registry.region_transition",
            "registry.population_capped",
        }


class TestMetrics:
    def test_active_objects_are_gauged(self, registry, metrics) -> None:
        drive(registry, 5)
        assert metrics.snapshot().gauge_value(
            MetricName.OBJECTS_ACTIVE, camera_id=str(CAMERA)
        ) == 1.0

    def test_creation_is_counted(self, registry, metrics) -> None:
        drive(registry, 5)
        assert metrics.snapshot().counters_matching(MetricName.OBJECTS_CREATED)

    def test_confirmation_is_counted(self, registry, metrics) -> None:
        drive(registry, 5)
        assert metrics.snapshot().counters_matching(MetricName.OBJECTS_CONFIRMED)

    def test_latency_is_recorded(self, registry, metrics) -> None:
        drive(registry, 3)
        assert metrics.snapshot().histogram_values(
            MetricName.REGISTRY_LATENCY_MS, camera_id=str(CAMERA)
        )

    def test_saturation_is_gauged(self, registry, metrics) -> None:
        drive(registry, 3)
        assert metrics.snapshot().gauge_value(
            MetricName.REGISTRY_SATURATION, camera_id=str(CAMERA)
        ) >= 0.0

    def test_no_metric_claims_a_wrong_identity(self) -> None:
        """Whether a binding was wrong needs ground truth the platform lacks."""
        for attribute in dir(MetricName):
            if attribute.startswith("_"):
                continue
            value = getattr(MetricName, attribute)
            if not isinstance(value, str) or "registry" not in value:
                continue
            for forbidden in ("false", "wrong", "incorrect", "error_rate", "accuracy"):
                assert forbidden not in value, f"{attribute} claims a judgment (V1)"
