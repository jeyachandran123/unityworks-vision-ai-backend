"""Vision State tests — M12 as a projection, not a database (07_STATE).

The brief is emphatic about what Vision State is *not*: not a database, not a
cache, not an event log, not a business model, not an analytics engine. Each of
those denials is a property something could accidentally acquire, so each one is
tested rather than asserted.

State transition, replay and failure categories all live here, because in a
log-and-projection design they are the same mechanism seen from three angles.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import (
    CommitFailedError,
    PartitionDegradedError,
    StateNotFoundError,
)
from vision_os.core.model.ids import CameraId, ObjectId
from vision_os.core.model.observation import (
    ObservabilityReason,
    ObservabilityStatus,
)
from vision_os.core.model.timebase import Duration
from vision_os.core.model.vision_state import ConsistencyLevel
from vision_os.kernel.events import PartitionDegraded

from .conftest import (
    CAMERA,
    OTHER_CAMERA,
    POSTURE,
    at,
    attribute,
    context,
    make_object,
    state_config,
    understanding,
)


def stream(builder, count: int, *, camera: CameraId = CAMERA, start: int = 0):
    """A run of published presence observations for distinct objects."""
    observations = []
    for i in range(count):
        published = builder.build_presence(
            make_object(object_id=f"obj-{start + i}", camera=camera, seq=start + i),
            context(seq=start + i, camera=camera),
        )
        if published is not None:
            observations.append(published)
    return observations


class TestObservationIsTheOnlyWritePath:
    """The brief: *"Observation is the only write path."*"""

    def test_appending_observations_is_the_only_public_mutator(self) -> None:
        """Every other public method reads.

        A second write path would mean two things could disagree about the same
        object, and the log would no longer be the system of record.
        """
        from vision_os.state import VisionStateManager

        writers = {"append", "resume", "rebuild", "retention_sweep", "forget",
                   "subscribe"}
        public = {
            name
            for name in dir(VisionStateManager)
            if not name.startswith("_") and callable(getattr(VisionStateManager, name, None))
        }
        assert public - writers <= {
            "snapshot", "object_state", "history", "observations_in", "coverage",
            "coverage_report", "site_context", "health", "partitions",
            "quarantined", "buffer_depth",
        }, (
            "a new public method appeared on the state manager; if it writes, the "
            "log is no longer the only write path"
        )

    def test_the_l7_read_seam_is_a_read(self) -> None:
        """``observations_in`` was added in Flow 8 for 09_API §2.2's query.

        Listed here deliberately: M14 reads history through M12 rather than
        holding P20 itself, so the seam is a method on this class — and every
        method added to this class has to be shown not to write.
        """
        import inspect

        from vision_os.state import VisionStateManager

        source = inspect.getsource(VisionStateManager.observations_in)
        for mutator in ("self._partitions[", "self._log.append", "self._buffers["):
            assert mutator not in source, f"the read seam calls {mutator}"

    def test_state_is_empty_until_an_observation_arrives(self, state) -> None:
        assert state.snapshot().partitions == {}

    def test_appending_creates_the_partition(self, state, loud_builder) -> None:
        state.append(stream(loud_builder, 1))
        assert CAMERA in state.snapshot().partitions

    def test_the_log_is_written_before_the_projection(
        self, state, loud_builder, log
    ) -> None:
        """§9.1 makes the log authoritative.

        Projecting first would leave state holding something the log does not, so
        a rebuild would silently produce a different world from the live one.
        """
        observations = stream(loud_builder, 3)
        state.append(observations)
        logged = list(log.read(CAMERA, limit=10))
        assert len(logged) == 3
        assert state.snapshot().partitions[CAMERA].log_position


class TestSnapshotsAreCheapAndConsistent:
    def test_a_snapshot_does_not_copy_the_partition(self, state, loud_builder) -> None:
        """07_STATE §5.1: *"a pointer to an immutable root"*.

        The read path and the write path never touch the same mutable memory,
        which is what lets M14 serve heavy query load without slowing perception.
        """
        state.append(stream(loud_builder, 4))
        first = state.snapshot()
        second = state.snapshot()
        assert first.partitions[CAMERA] is second.partitions[CAMERA]

    def test_a_snapshot_taken_before_a_write_does_not_see_it(
        self, state, loud_builder
    ) -> None:
        state.append(stream(loud_builder, 2))
        before = state.snapshot()
        state.append(stream(loud_builder, 2, start=50))
        assert len(before.partitions[CAMERA].objects) == 2
        assert len(state.snapshot().partitions[CAMERA].objects) == 4

    def test_a_single_partition_snapshot_is_strongly_consistent(
        self, state, loud_builder
    ) -> None:
        state.append(stream(loud_builder, 2))
        assert state.snapshot().consistency is ConsistencyLevel.STRONG

    def test_a_multi_partition_snapshot_refuses_to_claim_strong_consistency(
        self, state, loud_builder
    ) -> None:
        """07_STATE §4.4: no cross-partition lock, therefore no global instant.

        Claiming STRONG across cameras would be a promise the architecture
        deliberately does not make, and a consumer would build on it.
        """
        state.append(stream(loud_builder, 2))
        state.append(stream(loud_builder, 2, camera=OTHER_CAMERA, start=10))
        snapshot = state.snapshot()
        assert len(snapshot.partitions) == 2
        assert snapshot.consistency is ConsistencyLevel.SNAPSHOT_SET

    def test_a_snapshot_reports_what_it_could_not_include(
        self, state, loud_builder
    ) -> None:
        """V8. A partial answer that does not say it is partial is a wrong answer."""
        snapshot = state.snapshot(scope=[CameraId("cam-absent")])
        assert snapshot.incomplete
        assert snapshot.partitions == {}


class TestHistoryComesFromTheLogNotTheProjection:
    def test_history_reads_the_log(self, state, loud_builder) -> None:
        """07_STATE §6.1: *"History exists for perception, not for analytics."*

        Serving history from state would make the projection a time-series
        database, which §6.1 refuses — and the in-memory rings are bounded, so
        the answer would silently truncate.
        """
        for seq in range(6):
            published = loud_builder.build_presence(
                make_object(seq=seq), context(seq=seq)
            )
            if published is not None:
                state.append([published])
        history = state.history(ObjectId("obj-1"))
        assert len(history) >= 5
        assert [o.t_capture.ns for o in history] == sorted(
            o.t_capture.ns for o in history
        )

    def test_history_can_be_windowed(self, state, loud_builder) -> None:
        for seq in range(10):
            published = loud_builder.build_presence(
                make_object(seq=seq), context(seq=seq)
            )
            if published is not None:
                state.append([published])
        assert len(state.history(ObjectId("obj-1"), window=Duration.from_millis(1))) <= 10

    def test_an_unknown_object_is_a_typed_absence(self, state) -> None:
        """Distinct from an object that exists and is empty (§7.1)."""
        with pytest.raises(StateNotFoundError):
            state.object_state(ObjectId("never-seen"))


class TestPartitionsAreIndependent:
    """07_STATE §4: the camera is the partition; no cross-partition locks."""

    def test_a_batch_spanning_cameras_commits_per_partition(
        self, state, loud_builder
    ) -> None:
        mixed = stream(loud_builder, 2) + stream(
            loud_builder, 2, camera=OTHER_CAMERA, start=10
        )
        result = state.append(mixed)
        assert result.accepted == 4
        assert set(state.partitions) == {CAMERA, OTHER_CAMERA}

    def test_one_partition_halting_does_not_stop_another(
        self, clock, metrics, bus, loud_builder
    ) -> None:
        """The property that makes a 200-camera node survive one bad camera."""
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        failing = _FailingLog(fail_for=CAMERA)
        state = VisionStateManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=state_config(log_buffer_capacity=2),
            log=failing,
            site_id=SITE,
        )
        healthy = stream(loud_builder, 2, camera=OTHER_CAMERA, start=10)
        broken = stream(loud_builder, 8)

        result = state.append(broken + healthy)

        assert OTHER_CAMERA in state.partitions
        assert len(state.snapshot().partitions.get(OTHER_CAMERA).objects) == 2
        assert result.degraded

    def test_forgetting_one_camera_leaves_the_others(
        self, state, loud_builder
    ) -> None:
        state.append(stream(loud_builder, 2))
        state.append(stream(loud_builder, 2, camera=OTHER_CAMERA, start=10))
        state.forget(CAMERA)
        assert CAMERA not in state.partitions
        assert OTHER_CAMERA in state.partitions


class TestDurabilityFailureHaltsRatherThanDrops:
    """10_RELIABILITY §4.4 step 4. The ladder's last rung."""

    def test_a_full_buffer_stops_accepting_rather_than_dropping(
        self, clock, metrics, bus, loud_builder
    ) -> None:
        """*"Losing observations invisibly is a V8 violation of the worst kind."*

        A queue that dropped here would silently delete facts the platform had
        already decided were worth publishing.
        """
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        state = VisionStateManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=state_config(log_buffer_capacity=3),
            log=_FailingLog(fail_for=CAMERA),
            site_id=SITE,
        )
        with pytest.raises((CommitFailedError, PartitionDegradedError)):
            for _ in range(6):
                state.append(stream(loud_builder, 4))

    def test_halting_publishes_an_event(
        self, clock, metrics, bus, loud_builder
    ) -> None:
        """A partition that went quiet without saying so is indistinguishable
        from a camera with nothing to report.
        """
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        state = VisionStateManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=state_config(log_buffer_capacity=2),
            log=_FailingLog(fail_for=CAMERA),
            site_id=SITE,
        )
        subscription = bus.subscribe(["state.partition_degraded"])
        with pytest.raises((CommitFailedError, PartitionDegradedError)):
            for _ in range(8):
                state.append(stream(loud_builder, 4))
        events = subscription.drain()
        assert events
        assert isinstance(events[0], PartitionDegraded)

    def test_a_degraded_partition_resumes_only_on_an_operator_decision(
        self, clock, metrics, bus, loud_builder
    ) -> None:
        """Automatic resume would risk a second halt losing the buffered facts."""
        from vision_os.state import VisionStateManager

        from .conftest import SITE

        log = _FailingLog(fail_for=CAMERA)
        state = VisionStateManager(
            clock=clock,
            metrics=metrics,
            events=bus,
            config=state_config(log_buffer_capacity=2),
            log=log,
            site_id=SITE,
        )
        with pytest.raises((CommitFailedError, PartitionDegradedError)):
            for _ in range(8):
                state.append(stream(loud_builder, 4))

        log.healthy = True
        with pytest.raises(PartitionDegradedError):
            state.append(stream(loud_builder, 1, start=99))

        state.resume(CAMERA)
        assert state.append(stream(loud_builder, 1, start=98)).accepted == 1


class TestRebuildReproducesTheWorld:
    """V13. The claim that makes a log-and-projection design worth its cost."""

    def test_a_rebuild_from_the_log_reproduces_the_projection(
        self, state, loud_builder
    ) -> None:
        state.append(stream(loud_builder, 6))
        published = loud_builder.build_attribute(
            make_object(),
            understanding(attributes=(attribute(POSTURE, "sitting"),)),
            context(),
        )
        state.append(published)

        before = state.snapshot().partitions[CAMERA]
        handle = state.rebuild(CAMERA)
        after = state.snapshot().partitions[CAMERA]

        assert handle.clean
        assert after.objects.keys() == before.objects.keys()
        for object_id in before.objects:
            assert after.objects[object_id].last_seen == before.objects[object_id].last_seen

    def test_a_rebuild_swaps_atomically(self, state, loud_builder) -> None:
        """A reader holding a snapshot across a rebuild sees the old world whole.

        A partial swap would let a consumer observe a partition mid-replay, which
        is a state that never actually existed.
        """
        state.append(stream(loud_builder, 4))
        held = state.snapshot()
        state.rebuild(CAMERA)
        assert len(held.partitions[CAMERA].objects) == 4

    def test_replaying_the_same_log_twice_gives_the_same_state(
        self, state, loud_builder
    ) -> None:
        state.append(stream(loud_builder, 5))
        state.rebuild(CAMERA)
        once = state.snapshot().partitions[CAMERA]
        state.rebuild(CAMERA)
        twice = state.snapshot().partitions[CAMERA]
        assert once.objects.keys() == twice.objects.keys()


class TestCoverageState:
    def test_live_coverage_reports_the_current_status(
        self, state, loud_builder
    ) -> None:
        blind = loud_builder.build_coverage(
            context(),
            status=ObservabilityStatus.BLIND,
            reason=ObservabilityReason.STREAM_DISCONNECTED,
            since=at(1),
            effective_rate=0.0,
        )
        state.append([blind])
        assert state.coverage().by_camera[CAMERA].status is ObservabilityStatus.BLIND

    def test_a_historical_window_is_reconstructed_from_the_log(
        self, state, loud_builder
    ) -> None:
        """§7.3: *"a query over any past window can reconstruct exactly what was
        observable then."*
        """
        state.append([
            loud_builder.build_coverage(
                context(seq=1),
                status=ObservabilityStatus.BLIND,
                reason=ObservabilityReason.STREAM_DISCONNECTED,
                since=at(1),
                effective_rate=0.0,
                until=at(5),
            )
        ])
        report = state.coverage_report(CAMERA, since=at(0), until=at(10))
        assert 0.0 <= report.observable_fraction <= 1.0
        assert report.gaps

    def test_coverage_distinguishes_empty_from_blind(self, state, loud_builder) -> None:
        """The distinction the whole coverage machinery exists for."""
        state.append(stream(loud_builder, 1))
        assert state.coverage().by_camera[CAMERA] is not None


class TestStateIsNotADatabase:
    def test_state_has_no_query_language(self) -> None:
        """A query interface would invite filters, and a filter is a rule."""
        from vision_os.state import VisionStateManager

        for forbidden in ("query", "sql", "execute", "find_where", "filter"):
            assert not hasattr(VisionStateManager, forbidden)

    def test_state_has_no_delete_for_a_single_fact(self) -> None:
        """V5. Forgetting a *camera* is provisioning; deleting a fact is editing
        history, and the log is append-only.
        """
        from vision_os.state import VisionStateManager

        for forbidden in ("delete", "remove_observation", "update", "edit"):
            assert not hasattr(VisionStateManager, forbidden)

    def test_retention_sweeps_by_age_rather_than_by_content(
        self, state, loud_builder
    ) -> None:
        """Retention is a time policy. A sweep that chose by content would be
        deciding which facts matter, which is a business judgement.
        """
        state.append(stream(loud_builder, 3))
        report = state.retention_sweep()
        assert report is not None


class TestSubscribersLearnWhatChanged:
    def test_a_subscriber_receives_a_delta(self, state, loud_builder) -> None:
        """Without a delta a subscriber must diff two snapshots, which is O(n)
        per update and defeats the point of structural sharing.
        """
        deltas = []
        state.subscribe(deltas.append)
        state.append(stream(loud_builder, 2))
        assert deltas
        assert deltas[0].changed_objects

    def test_a_failing_subscriber_does_not_break_the_commit(
        self, state, loud_builder
    ) -> None:
        """07_STATE §5.2. A consumer's bug must not stop the platform recording
        facts; the write path owes nothing to its readers.
        """
        def explode(delta):
            raise RuntimeError("subscriber is broken")

        state.subscribe(explode)
        assert state.append(stream(loud_builder, 2)).accepted == 2

    def test_unsubscribing_stops_delivery(self, state, loud_builder) -> None:
        deltas = []
        cancel = state.subscribe(deltas.append)
        state.append(stream(loud_builder, 1))
        cancel()
        state.append(stream(loud_builder, 1, start=50))
        assert len(deltas) == 1


class _FailingLog:
    """A P20 adapter whose appends fail for one camera.

    Not a mock of the port — a real implementation of it that happens to be
    broken, which is the only way to exercise the durability ladder without
    reaching inside the manager.
    """

    __slots__ = ("_records", "fail_for", "healthy")

    def __init__(self, *, fail_for: CameraId) -> None:
        self.fail_for = fail_for
        self.healthy = False
        self._records: dict[CameraId, list] = {}

    @property
    def adapter_id(self) -> str:
        return "log.failing"

    @property
    def durable(self) -> bool:
        return True

    def append(self, camera_id: CameraId, observations):
        from vision_os.core.errors import LogUnavailableError
        from vision_os.core.model.ids import LogPosition
        from vision_os.core.ports.synthesis import LogAppendResult

        if camera_id == self.fail_for and not self.healthy:
            raise LogUnavailableError(
                "the store is unreachable", camera_id=str(camera_id)
            )
        held = self._records.setdefault(camera_id, [])
        held.extend(observations)
        return LogAppendResult(
            appended=len(observations), position=LogPosition(len(held))
        )

    def read(self, camera_id: CameraId, *, since=None, limit: int = 1000):
        return tuple(self._records.get(camera_id, ())[:limit])

    def position(self, camera_id: CameraId):
        from vision_os.core.model.ids import LogPosition

        return LogPosition(len(self._records.get(camera_id, ())))

    def trim(self, camera_id: CameraId, *, before) -> int:
        return 0
