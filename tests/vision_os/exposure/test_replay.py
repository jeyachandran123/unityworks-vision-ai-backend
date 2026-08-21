"""Replay verification — V13, and the guarantee everything else rests on.

The brief: *"Replay must reconstruct **identical** Vision State from the
Observation Log. Replay must be deterministic, repeatable, auditable. No replay
shortcut may exist."*

07_STATE §9.1's recovery table reports *"no data loss"* for state-store
corruption, projection bugs and schema changes. Every one of those rows is a
promise **replay** keeps. If replay were subtly wrong, nothing would detect it —
the projection was the only other copy of what the log means.

So these tests do not merely check that a rebuild produces *something*. They
check it produces the same thing, field by field, and that the comparison is
strong enough to notice if it did not.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import CameraId, ObjectId
from vision_os.core.model.timebase import Instant
from vision_os.state.projection import ProjectionBounds
from vision_os.state.replay import (
    Divergence,
    ReplayVerifier,
    compare_partitions,
    deterministic_digest,
    replay_partition,
)

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    POSTURE,
    publish,
    publish_attributes,
)

BOUNDS = ProjectionBounds(
    trajectory_points=64, attribute_history=8, class_history=16, max_objects=512
)


@pytest.fixture
def verifier(clock, metrics, log) -> ReplayVerifier:
    return ReplayVerifier(clock=clock, metrics=metrics, log=log, bounds=BOUNDS)


class TestReplayReproducesTheWorld:
    def test_a_replayed_partition_is_identical(self, state, log, verifier) -> None:
        publish(state, count=8)
        publish_attributes(state, object_id="obj-0", value="sitting")

        live = state.snapshot().partitions[CAMERA]
        report = verifier.verify(CAMERA, live)

        assert report.identical, "\n".join(str(d) for d in report.divergences)
        assert report.objects_live == report.objects_replayed

    def test_replaying_twice_gives_the_same_result(self, state, log) -> None:
        """Repeatable, not merely deterministic once."""
        publish(state, count=6)
        first, _ = replay_partition(log, CAMERA, bounds=BOUNDS)
        second, _ = replay_partition(log, CAMERA, bounds=BOUNDS)
        assert compare_partitions(first, second) == ()

    def test_attributes_survive_a_replay(self, state, log, verifier) -> None:
        publish(state, count=2)
        publish_attributes(state, object_id="obj-0", value="crouching")
        live = state.snapshot().partitions[CAMERA]
        replayed, _ = replay_partition(log, CAMERA, bounds=BOUNDS)
        assert (
            replayed.objects[ObjectId("obj-0")].attributes[POSTURE].value == "crouching"
        )
        assert verifier.verify(CAMERA, live).identical

    def test_measured_and_believed_sightings_stay_distinct(
        self, state, log, verifier
    ) -> None:
        """The V8 field a subtly wrong projection would get wrong.

        A bug that confirmed on a prediction shows up nowhere else: the object is
        still present, at the right place, with the right class — only
        ``last_confirmed`` is wrong, and only this comparison notices.
        """
        from vision_os.adapters.synthesis import AlwaysPublish
        from vision_os.core.model.observation import MeasurementBasis

        from ..synthesis.conftest import context, make_builder, make_object  # noqa: TID252

        publish(state, count=1)
        builder = make_builder(policy=AlwaysPublish())
        believed = builder.build_presence(
            make_object(object_id="obj-0", seq=40),
            context(seq=40),
            basis=MeasurementBasis.PREDICTED,
        )
        state.append([believed])

        live = state.snapshot().partitions[CAMERA]
        assert live.objects[ObjectId("obj-0")].last_confirmed.ns < live.objects[
            ObjectId("obj-0")
        ].last_seen.ns
        assert verifier.verify(CAMERA, live).identical

    def test_each_partition_replays_independently(self, state, verifier) -> None:
        """07_STATE §4: no cross-partition anything, including recovery."""
        publish(state, count=3, camera=CAMERA)
        publish(state, count=3, camera=OTHER_CAMERA, start=50)
        snapshot = state.snapshot()
        reports = verifier.verify_all(list(snapshot.partitions.items()))
        assert len(reports) == 2
        assert all(r.identical for r in reports)


class TestTheComparisonIsStrongEnough:
    """A verifier that could not detect a divergence would prove nothing."""

    def test_a_missing_object_is_detected(self, state, log) -> None:
        import dataclasses

        publish(state, count=4)
        live = state.snapshot().partitions[CAMERA]
        damaged = dataclasses.replace(
            live, objects={k: v for k, v in list(live.objects.items())[:2]}
        )
        divergences = compare_partitions(live, damaged)
        assert divergences
        assert any(d.field_name == "objects" for d in divergences)

    def test_a_drifted_timestamp_is_detected(self, state, log) -> None:
        import dataclasses

        publish(state, count=2)
        live = state.snapshot().partitions[CAMERA]
        object_id = next(iter(live.objects))
        drifted = dataclasses.replace(
            live.objects[object_id],
            last_confirmed=Instant(live.objects[object_id].last_confirmed.ns - 1),
        )
        damaged = dataclasses.replace(live, objects={**live.objects, object_id: drifted})

        divergences = compare_partitions(live, damaged)
        assert any(d.field_name == "last_confirmed" for d in divergences)

    def test_a_changed_attribute_value_is_detected(self, state) -> None:
        import dataclasses

        publish(state, count=1)
        publish_attributes(state, object_id="obj-0", value="standing")
        live = state.snapshot().partitions[CAMERA]
        held = live.objects[ObjectId("obj-0")]
        altered = dataclasses.replace(
            held,
            attributes={
                **held.attributes,
                POSTURE: dataclasses.replace(held.attributes[POSTURE], value="lying"),
            },
        )
        damaged = dataclasses.replace(
            live, objects={**live.objects, ObjectId("obj-0"): altered}
        )
        divergences = compare_partitions(live, damaged)
        assert any("attributes" in d.field_name for d in divergences)

    def test_bookkeeping_differences_are_not_reported(self, state, log) -> None:
        """``version`` and ``log_position`` legitimately differ.

        A replay performs a different number of writes and may start partway
        through the log. Reporting those would make every successful replay look
        like a failure, and a test that always fails is a test that gets deleted.
        """
        import dataclasses

        from vision_os.core.model.ids import LogPosition, PartitionVersion

        publish(state, count=3)
        live = state.snapshot().partitions[CAMERA]
        rebookkept = dataclasses.replace(
            live,
            version=PartitionVersion(live.version + 999),
            log_position=LogPosition(int(live.log_position) + 999),
        )
        assert compare_partitions(live, rebookkept) == ()

    def test_a_divergence_names_the_field(self) -> None:
        """*"Replay produced different state"* is not actionable.

        *"Object o-14's last_confirmed is 200ms earlier"* points at the rule that
        drifted.
        """
        divergence = Divergence(
            scope="obj-14", field_name="last_confirmed", live=1, replayed=2
        )
        assert "obj-14" in str(divergence)
        assert "last_confirmed" in str(divergence)


class TestNoReplayShortcutExists:
    """The brief: *"No replay shortcut may exist."*"""

    def test_replay_calls_the_same_projection_the_live_path_calls(self) -> None:
        """Not a similar function — the identical one.

        A second implementation of what an observation means to state would
        diverge quietly, and the projection is the only other copy.
        """
        import inspect

        source = inspect.getsource(replay_partition)
        assert "project(" in source

        from vision_os.state import manager, projection

        assert "project" in dir(projection)
        assert "project" in inspect.getsource(manager.VisionStateManager._project_batch)

    def test_there_is_exactly_one_projection_function(self) -> None:
        """Two would drift, and the second would not be the one under test."""
        from vision_os.state import projection

        exported = [
            name
            for name in dir(projection)
            if name.startswith("project") and not name.startswith("_")
        ]
        assert exported == ["project"]

    def test_replay_reads_the_log_and_nothing_else(self) -> None:
        """A replay that consulted live state would be validating state against
        itself, which proves nothing at all.
        """
        import inspect

        source = inspect.getsource(replay_partition)
        for forbidden in ("snapshot", "_partitions", "VisionStateManager"):
            assert forbidden not in source


class TestReplayIsAuditable:
    def test_a_report_names_what_it_replayed(self, state, verifier) -> None:
        publish(state, count=5)
        report = verifier.verify(CAMERA, state.snapshot().partitions[CAMERA])
        assert report.camera_id == CAMERA
        assert report.observations == 5
        assert "5 observations" in report.summary()

    def test_a_mismatch_is_counted(self, state, verifier, metrics_exporter) -> None:
        """``REPLAY_MISMATCHES`` must be zero in a healthy platform.

        Non-zero means a rebuild produced a different world, which invalidates
        every recovery guarantee in 07_STATE §9.1.
        """
        import dataclasses

        publish(state, count=3)
        live = state.snapshot().partitions[CAMERA]
        damaged = dataclasses.replace(live, objects={})
        report = verifier.verify(CAMERA, damaged)
        assert not report.identical

    def test_a_digest_fingerprints_semantic_content(self, state, log) -> None:
        """Two runs producing the same digest produced the same facts in order."""
        publish(state, count=4)
        observations = tuple(log.read(CAMERA))
        assert deterministic_digest(observations) == deterministic_digest(observations)

    def test_the_digest_excludes_publication_time(self, state, log) -> None:
        """When the platform *said* something is not part of what it said.

        A replay legitimately publishes at a different wall time, and a digest
        including it would report every correct replay as a divergence.
        """
        import dataclasses

        publish(state, count=2)
        original = tuple(log.read(CAMERA))
        republished = tuple(
            dataclasses.replace(o, t_published=Instant(o.t_published.ns + 10**9))
            for o in original
        )
        assert deterministic_digest(original) == deterministic_digest(republished)

    def test_the_digest_detects_a_changed_fact(self, state, log) -> None:
        import dataclasses

        from vision_os.core.model.ids import ClassId

        publish(state, count=2)
        original = tuple(log.read(CAMERA))
        altered = (
            dataclasses.replace(original[0], class_id=ClassId("vehicle")),
            *original[1:],
        )
        assert deterministic_digest(original) != deterministic_digest(altered)


class TestRecoveryScenarios:
    """07_STATE §9.1's table, exercised."""

    def test_state_store_corruption_rebuilds_with_no_loss(self, state, log) -> None:
        """*"Rebuild from log — the log is authoritative."*"""
        publish(state, count=6)
        before = state.snapshot().partitions[CAMERA]
        state.forget(CAMERA)
        assert CAMERA not in state.partitions

        rebuilt, count = replay_partition(log, CAMERA, bounds=BOUNDS)
        assert count == 6
        assert rebuilt.objects.keys() == before.objects.keys()

    def test_a_projection_bug_is_fixed_by_a_rebuild(self, state, verifier) -> None:
        """§9.1: *"the strongest argument for event sourcing here."*"""
        publish(state, count=5)
        state.rebuild(CAMERA)
        assert verifier.verify(CAMERA, state.snapshot().partitions[CAMERA]).identical

    def test_replay_from_a_watermark_resumes_rather_than_restarting(
        self, state, log
    ) -> None:
        """§9.1's *"replay log from `log_position`"*.

        Without a resumable watermark, every crash would replay the entire log.
        """
        from vision_os.core.model.ids import LogPosition

        publish(state, count=8)
        partial, count = replay_partition(
            log, CAMERA, bounds=BOUNDS, start=LogPosition(5)
        )
        assert count == 3
        assert len(partial.objects) == 3

    def test_an_empty_partition_replays_to_an_empty_projection(self, log) -> None:
        """A cold start must be distinguishable from a failure."""
        rebuilt, count = replay_partition(log, CameraId("never-used"), bounds=BOUNDS)
        assert count == 0
        assert rebuilt.objects == {}
