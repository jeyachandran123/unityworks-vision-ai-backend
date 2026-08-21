"""Replay, performance, concurrency and stress.

V13 is the invariant that makes an observation log worth its cost: *"the same
inputs produce the same outputs"*, so a rebuild six months later reconstructs the
world rather than a plausible neighbour of it.

The load tests assert **shape**, not wall-clock: an absolute millisecond budget
on a shared CI box measures the box. What matters architecturally is that cost
stays flat as the partition grows, and that bounded things stay bounded.
"""

from __future__ import annotations

import asyncio
import time

from vision_os.core.model.ids import CameraId, ObjectId
from vision_os.core.model.observation import (
    ObservabilityReason,
    ObservabilityStatus,
)
from vision_os.core.model.timebase import Duration

from .conftest import (
    CAMERA,
    POSTURE,
    at,
    attribute,
    context,
    make_builder,
    make_object,
    state_config,
    understanding,
)


def build_run(*, count: int = 20, policy_name: str = "suppression.exact"):
    """One deterministic run of the builder over a fixed script."""
    from vision_os.adapters.synthesis import SUPPRESSION_FACTORIES

    builder = make_builder(
        policy=SUPPRESSION_FACTORIES[policy_name](),
        suppression_policy=policy_name,
    )
    published = []
    for seq in range(count):
        presence = builder.build_presence(
            make_object(object_id=f"obj-{seq % 4}", seq=seq), context(seq=seq)
        )
        if presence is not None:
            published.append(presence)
        attributes = builder.build_attribute(
            make_object(object_id=f"obj-{seq % 4}", seq=seq),
            understanding(
                request_id=f"req-{seq}",
                object_id=f"obj-{seq % 4}",
                seq=seq,
                attributes=(attribute(POSTURE, "standing", observed_at=at(seq)),),
            ),
            context(seq=seq),
        )
        published.extend(attributes)
    return published


class TestReplayDeterminism:
    """V13. Two runs over the same script must agree on everything but ids."""

    def test_the_same_script_publishes_the_same_observations(self) -> None:
        first = build_run()
        second = build_run()
        assert len(first) == len(second)
        assert [o.observation_type for o in first] == [
            o.observation_type for o in second
        ]
        assert [o.t_capture.ns for o in first] == [o.t_capture.ns for o in second]

    def test_the_same_script_suppresses_the_same_observations(self) -> None:
        """Suppression is part of the replayed behaviour, not noise around it.

        A policy whose decisions drifted would make a rebuild produce a *denser*
        or *sparser* log than the live run, and the two worlds would diverge
        without either being wrong on its own terms.
        """
        counts = [len(build_run()) for _ in range(4)]
        assert len(set(counts)) == 1

    def test_attribute_ordering_is_stable(self) -> None:
        """Two runs must drop and keep in the same order.

        Iteration over an unordered set would satisfy every other assertion here
        while making the log's byte content unreproducible.
        """
        first = [tuple(a.key for a in o.attributes) for o in build_run() if o.attributes]
        second = [tuple(a.key for a in o.attributes) for o in build_run() if o.attributes]
        assert first == second

    def test_observation_ids_are_time_sortable(self, clock) -> None:
        """A ULID orders by mint time, so a log sorts without a separate index.

        Random ids would force every ordered read to sort by timestamp, and two
        observations in the same millisecond would have no defined order at all.

        The clock must advance for this to mean anything: under a frozen clock
        every id shares a timestamp prefix and sorts by its random tail, which
        proves nothing either way.
        """
        builder = make_builder(clock=clock, policy=_always())
        ids = []
        for seq in range(8):
            clock.advance(Duration.from_millis(50))
            published = builder.build_presence(
                make_object(object_id=f"o{seq}", seq=seq), context(seq=seq)
            )
            ids.append(str(published.observation_id))
        assert ids == sorted(ids)

    def test_an_id_is_minted_from_platform_time_not_the_wall(self, clock) -> None:
        """V13. A wall-clock id would differ on every replay of the same log."""
        first = make_builder(clock=clock, policy=_always()).build_presence(
            make_object(), context()
        )
        second = make_builder(clock=clock, policy=_always()).build_presence(
            make_object(), context()
        )
        # Same platform instant, so the same timestamp prefix — only the random
        # tail differs, which is what a ULID promises.
        assert str(first.observation_id)[:10] == str(second.observation_id)[:10]

    def test_a_rebuild_reproduces_the_projection_exactly(self, state) -> None:
        published = build_run(count=12)
        state.append(published)
        before = state.snapshot().partitions[CAMERA]
        state.rebuild(CAMERA)
        after = state.snapshot().partitions[CAMERA]

        assert after.objects.keys() == before.objects.keys()
        for object_id in before.objects:
            old, new = before.objects[object_id], after.objects[object_id]
            assert new.last_seen == old.last_seen
            assert new.last_confirmed == old.last_confirmed
            assert new.observation_count == old.observation_count

    def test_projecting_in_a_different_batch_shape_gives_the_same_state(
        self, clock, metrics, bus
    ) -> None:
        """Batching is a transport detail, not a semantic one.

        One batch of twelve and twelve batches of one must land identically, or
        a recovery that re-batches differently would rebuild a different world.
        """
        from vision_os.adapters.synthesis import InMemoryObservationLog
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        published = build_run(count=12)

        def project_with(batches):
            manager = VisionStateManager(
                clock=clock,
                metrics=metrics,
                events=bus,
                config=state_config(),
                log=InMemoryObservationLog(),
                site_id=SITE,
            )
            for batch in batches:
                manager.append(batch)
            return manager.snapshot().partitions[CAMERA]

        whole = project_with([published])
        piecemeal = project_with([[o] for o in published])

        assert whole.objects.keys() == piecemeal.objects.keys()
        for object_id in whole.objects:
            assert (
                whole.objects[object_id].last_seen
                == piecemeal.objects[object_id].last_seen
            )


class TestPerformanceShape:
    """Cost must not grow with the partition."""

    def test_a_snapshot_is_constant_time(self, state) -> None:
        """07_STATE §5.1's O(1) claim, measured as a ratio rather than a budget.

        A snapshot that copied would grow linearly, and the whole read/write
        separation would be a comforting fiction.
        """
        def snapshot_cost(objects: int) -> float:
            state.forget(CAMERA)
            builder = make_builder(policy=_always())
            state.append([
                builder.build_presence(
                    make_object(object_id=f"o{i}", seq=i), context(seq=i)
                )
                for i in range(objects)
            ])
            start = time.perf_counter()
            for _ in range(200):
                state.snapshot()
            return time.perf_counter() - start

        small = snapshot_cost(4)
        large = snapshot_cost(256)
        assert large < small * 8, (
            f"snapshot cost grew {large / max(small, 1e-9):.1f}x for 64x the "
            f"objects; a snapshot is meant to be a pointer, not a copy"
        )

    def test_suppression_keeps_the_common_case_cheap(self) -> None:
        """§M11 puts the reduction at 10-50x. The stationary case is the one
        that matters: a parked car at 5fps is 18,000 identical facts an hour.
        """
        builder = make_builder()
        obj = make_object()
        builder.build_presence(obj, context(seq=0))
        published = sum(
            builder.build_presence(obj, context(seq=seq)) is not None
            for seq in range(1, 101)
        )
        assert published <= 10, f"{published}/100 republished; suppression is not working"

    def test_the_builder_does_not_grow_without_bound(self) -> None:
        """Steady-state memory must be calculable before deployment."""
        builder = make_builder(suppression_capacity=32)
        for i in range(1_000):
            builder.build_presence(
                make_object(object_id=f"o{i}", seq=i), context(seq=i)
            )
        assert builder.suppression.partition(CAMERA).tracked <= 32

    def test_projection_cost_does_not_grow_with_history(self, state) -> None:
        """Bounded rings mean the thousandth observation costs what the tenth did."""
        builder = make_builder(policy=_always())

        def append_cost(start: int, count: int) -> float:
            batch = [
                builder.build_presence(
                    make_object(object_id="obj-1", seq=seq), context(seq=seq)
                )
                for seq in range(start, start + count)
            ]
            began = time.perf_counter()
            state.append([o for o in batch if o is not None])
            return time.perf_counter() - began

        early = append_cost(0, 100)
        late = append_cost(5_000, 100)
        assert late < max(early, 1e-6) * 10


class TestConcurrency:
    """07_STATE §4: one writer per partition; partitions never synchronise."""

    async def test_concurrent_seams_for_one_camera_serialize(self, clock, metrics, health, state) -> None:
        """Two coroutines feeding one camera must not interleave a build.

        The runtime holds a per-camera lock. Without it, two builds could read
        the same suppression signature and both publish, or both suppress.
        """
        from vision_os.perception.registry.engine import RegistryUpdate
        from vision_os.synthesis import SynthesisRuntime

        from .conftest import TAXONOMY_VERSION, frame_ref

        runtime = SynthesisRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            builder=make_builder(clock=clock, metrics=metrics, policy=_always()),
            config=_config(),
            state=state,
            taxonomy_version=TAXONOMY_VERSION,
        )
        await runtime.start()

        async def feed(start: int):
            for i in range(start, start + 10):
                await runtime.on_registered(
                    RegistryUpdate(
                        camera_id=CAMERA,
                        frame_ref=frame_ref(i),
                        objects=(make_object(object_id=f"o{i}", seq=i),),
                    )
                )

        await asyncio.gather(feed(0), feed(100), feed(200))

        # Two observations per object — a presence fact and a spatial one, which
        # 02_VOM §11.2 keeps distinct because "something is here" and "it is
        # here" answer different questions.
        assert runtime.stats.observations_built == 60
        assert len(state.snapshot().partitions[CAMERA].objects) == 30

    async def test_two_cameras_never_wait_on_each_other(
        self, clock, metrics, health, state
    ) -> None:
        """§4.4: *"neither takes a cross-partition lock"*.

        The property that lets a 200-camera node degrade one stream without
        stalling the other 199.
        """
        from vision_os.perception.registry.engine import RegistryUpdate
        from vision_os.synthesis import SynthesisRuntime

        from .conftest import OTHER_CAMERA, TAXONOMY_VERSION, frame_ref

        runtime = SynthesisRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            builder=make_builder(clock=clock, metrics=metrics, policy=_always()),
            config=_config(),
            state=state,
            taxonomy_version=TAXONOMY_VERSION,
        )
        await runtime.start()

        async def feed(camera: CameraId, tag: str):
            for i in range(10):
                await runtime.on_registered(
                    RegistryUpdate(
                        camera_id=camera,
                        frame_ref=frame_ref(i, camera=camera),
                        objects=(
                            make_object(object_id=f"{tag}{i}", camera=camera, seq=i),
                        ),
                    )
                )

        await asyncio.gather(feed(CAMERA, "a"), feed(OTHER_CAMERA, "b"))
        snapshot = state.snapshot()
        assert len(snapshot.partitions[CAMERA].objects) == 10
        assert len(snapshot.partitions[OTHER_CAMERA].objects) == 10

    def test_a_reader_during_a_write_sees_a_whole_world(self, state) -> None:
        """Snapshot isolation without a lock, because the value is frozen.

        A reader holding a snapshot across a hundred appends must see none of
        them, not some of them.
        """
        builder = make_builder(policy=_always())
        state.append([
            builder.build_presence(make_object(object_id="o0", seq=0), context(seq=0))
        ])
        held = state.snapshot()

        for i in range(1, 100):
            published = builder.build_presence(
                make_object(object_id=f"o{i}", seq=i), context(seq=i)
            )
            if published is not None:
                state.append([published])

        assert len(held.partitions[CAMERA].objects) == 1


class TestStress:
    def test_a_long_run_stays_bounded(self, state) -> None:
        """The shape of a camera running for hours: many objects, bounded memory."""
        builder = make_builder(policy=_always())
        for i in range(2_000):
            published = builder.build_presence(
                make_object(object_id=f"o{i % 200}", seq=i), context(seq=i)
            )
            if published is not None:
                state.append([published])

        partition = state.snapshot().partitions[CAMERA]
        assert len(partition.objects) <= state_config().max_objects_per_partition
        for held in partition.objects.values():
            assert len(held.trajectory) <= state_config().trajectory_points

    def test_a_burst_of_coverage_changes_is_never_dropped(self, state) -> None:
        """V8's hardest case: the platform going blind repeatedly.

        Every one of these must land. A coverage observation the platform
        decided not to bother with is the platform hiding its own failure.
        """
        builder = make_builder()
        published = []
        for i in range(200):
            blind = i % 2 == 0
            published.append(
                builder.build_coverage(
                    context(seq=i),
                    status=(
                        ObservabilityStatus.BLIND
                        if blind
                        else ObservabilityStatus.OBSERVING
                    ),
                    reason=(
                        ObservabilityReason.STREAM_DISCONNECTED
                        if blind
                        else ObservabilityReason.NORMAL
                    ),
                    since=at(i),
                    effective_rate=0.0 if blind else 1.0,
                )
            )
        assert len(published) == 200
        assert state.append(published).accepted == 200

    def test_many_cameras_on_one_node(self, clock, metrics, bus) -> None:
        """Each partition independent, none aware of the others."""
        from vision_os.adapters.synthesis import InMemoryObservationLog
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        manager = VisionStateManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=state_config(),
            log=InMemoryObservationLog(),
            site_id=SITE,
        )
        builder = make_builder(policy=_always())
        batch = []
        for camera_index in range(64):
            camera = CameraId(f"cam-{camera_index:02d}")
            batch.append(
                builder.build_presence(
                    make_object(object_id=f"o{camera_index}", camera=camera),
                    context(camera=camera),
                )
            )
        manager.append([o for o in batch if o is not None])
        assert len(manager.partitions) == 64

    def test_history_stays_answerable_under_load(self, state) -> None:
        builder = make_builder(policy=_always())
        for i in range(500):
            published = builder.build_presence(
                make_object(object_id="obj-1", seq=i), context(seq=i)
            )
            if published is not None:
                state.append([published])
        history = state.history(ObjectId("obj-1"), limit=50)
        assert len(history) <= 50
        assert history == tuple(sorted(history, key=lambda o: o.t_capture.ns))


# --- helpers -------------------------------------------------------------------- #


def _always():
    from vision_os.adapters.synthesis import AlwaysPublish

    return AlwaysPublish()


def _config():
    from .conftest import synthesis_config

    return synthesis_config(suppression_policy="suppression.always")
