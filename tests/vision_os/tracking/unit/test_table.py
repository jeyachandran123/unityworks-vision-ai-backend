"""The bounded track table — id discipline and memory bounds (T3, T8).

Two properties here are the difference between a tracker that runs for a month
and one that runs for a day: ids are never reused within an epoch, and the table
refuses to grow past its declared bound.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.tracking.motion import LinearPredictor
from vision_os.core.errors import TrackerCapacityError
from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    FrameRef,
    FrameSeq,
    StreamEpoch,
    TrackerEpoch,
)
from vision_os.core.model.space import Box
from vision_os.core.model.timebase import Instant
from vision_os.core.model.track import TrackState
from vision_os.perception.tracking.table import TrackTable

from ..conftest import SITE, TENANT

CAMERA = CameraId("cam-01")
PERSON = ClassId("person")


def frame_ref(seq: int = 0) -> FrameRef:
    return FrameRef(CAMERA, StreamEpoch(1), FrameSeq(seq))


def create(table: TrackTable, *, x: float = 0.1, seq: int = 0):
    return table.create(
        class_id=PERSON,
        box=Box(x, 0.4, x + 0.1, 0.8),
        predictor=LinearPredictor(),
        now=Instant(seq * 200_000_000),
        frame_ref=frame_ref(seq),
        tenant_id=TENANT,
        site_id=SITE,
    )


@pytest.fixture
def table() -> TrackTable:
    return TrackTable(CAMERA, max_tracks=8, history_length=4)


class TestConstruction:
    def test_rejects_a_zero_capacity(self) -> None:
        with pytest.raises(ValueError, match="max_tracks"):
            TrackTable(CAMERA, max_tracks=0)

    def test_rejects_a_zero_history(self) -> None:
        with pytest.raises(ValueError, match="history_length"):
            TrackTable(CAMERA, history_length=0)

    def test_starts_empty_at_epoch_zero(self, table: TrackTable) -> None:
        assert len(table) == 0
        assert table.epoch == 0


class TestIdDiscipline:
    def test_ids_increment(self, table: TrackTable) -> None:
        first = create(table, x=0.1)
        second = create(table, x=0.5)
        assert first.track_id.local_id == 0
        assert second.track_id.local_id == 1

    def test_ids_carry_the_camera_and_epoch(self, table: TrackTable) -> None:
        record = create(table)
        assert record.track_id.camera_id == CAMERA
        assert record.track_id.tracker_epoch == table.epoch

    def test_a_removed_id_is_never_reissued(self, table: TrackTable) -> None:
        """T3. Reuse lets a consumer join two unrelated objects into one history:
        invisible downstream and unrecoverable afterwards."""
        first = create(table)
        table.remove(first.track_id)
        second = create(table)
        assert second.track_id != first.track_id
        assert second.track_id.local_id > first.track_id.local_id

    def test_ids_stay_unique_across_heavy_churn(self, table: TrackTable) -> None:
        seen = set()
        for cycle in range(50):
            record = create(table, x=0.1 + (cycle % 5) * 0.1, seq=cycle)
            assert record.track_id not in seen
            seen.add(record.track_id)
            table.remove(record.track_id)
        assert len(seen) == 50

    def test_retired_ids_are_remembered(self, table: TrackTable) -> None:
        record = create(table)
        table.remove(record.track_id)
        assert table.was_retired(record.track_id)


class TestCapacity:
    def test_creating_past_capacity_is_refused(self) -> None:
        """A crowd degrades by refusing new tracks, never by growing (T8)."""
        table = TrackTable(CAMERA, max_tracks=3)
        for i in range(3):
            create(table, x=0.1 + i * 0.2)
        with pytest.raises(TrackerCapacityError, match="refusing new tracks"):
            create(table, x=0.9)

    def test_the_refusal_names_the_camera_and_bound(self) -> None:
        table = TrackTable(CAMERA, max_tracks=1)
        create(table)
        with pytest.raises(TrackerCapacityError) as caught:
            create(table, x=0.5)
        assert str(CAMERA) in str(caught.value)

    def test_capacity_is_reported(self, table: TrackTable) -> None:
        assert table.capacity == 8


class TestEviction:
    def test_a_tentative_track_is_evicted_before_a_confirmed_one(self) -> None:
        """A tentative track has asserted nothing yet, so it costs least."""
        table = TrackTable(CAMERA, max_tracks=4)
        confirmed = create(table, x=0.1)
        confirmed.state = TrackState.CONFIRMED
        tentative = create(table, x=0.5)

        evicted = table.evict_weakest()
        assert evicted is not None
        assert evicted.track_id == tentative.track_id
        assert confirmed.track_id in table

    def test_the_longest_coasting_confirmed_track_goes_first(self) -> None:
        table = TrackTable(CAMERA, max_tracks=4)
        fresh = create(table, x=0.1)
        fresh.state = TrackState.CONFIRMED
        stale = create(table, x=0.5)
        stale.state = TrackState.CONFIRMED
        stale.coast_frames = 4

        evicted = table.evict_weakest()
        assert evicted is not None
        assert evicted.track_id == stale.track_id

    def test_eviction_is_deterministic(self) -> None:
        results = set()
        for _ in range(20):
            table = TrackTable(CAMERA, max_tracks=4)
            for i in range(3):
                record = create(table, x=0.1 + i * 0.2)
                record.state = TrackState.CONFIRMED
                record.association_confidence = 0.5
            evicted = table.evict_weakest()
            results.add(evicted.track_id.local_id)
        assert len(results) == 1

    def test_evicting_an_empty_table_returns_none(self, table: TrackTable) -> None:
        assert table.evict_weakest() is None

    def test_an_evicted_id_is_retired(self) -> None:
        table = TrackTable(CAMERA, max_tracks=2)
        record = create(table)
        table.evict_weakest()
        assert table.was_retired(record.track_id)


class TestHistoryIsBounded:
    def test_history_is_capped_at_the_configured_length(self) -> None:
        """An hour-long track holds the same memory as a one-second track."""
        table = TrackTable(CAMERA, history_length=4)
        record = create(table)
        for seq in range(1, 50):
            record.history.append(frame_ref(seq))
        assert len(record.history) == 4

    def test_history_keeps_the_most_recent_frames(self) -> None:
        table = TrackTable(CAMERA, history_length=3)
        record = create(table, seq=0)
        for seq in range(1, 10):
            record.history.append(frame_ref(seq))
        assert [f.frame_seq for f in record.history] == [7, 8, 9]

    def test_history_holds_references_not_detections(self) -> None:
        """Copying detections into tracks makes memory grow with lifetime."""
        table = TrackTable(CAMERA)
        record = create(table)
        assert all(isinstance(item, FrameRef) for item in record.history)


class TestReset:
    def test_reset_discards_every_track(self, table: TrackTable) -> None:
        create(table, x=0.1)
        create(table, x=0.5)
        discarded = table.reset(TrackerEpoch(1))
        assert len(discarded) == 2
        assert len(table) == 0

    def test_reset_adopts_the_new_epoch(self, table: TrackTable) -> None:
        table.reset(TrackerEpoch(7))
        assert table.epoch == 7
        assert create(table).track_id.tracker_epoch == 7

    def test_reset_restarts_local_ids(self, table: TrackTable) -> None:
        """Safe only because the epoch changed — the composite id stays unique."""
        first = create(table)
        table.reset(TrackerEpoch(1))
        second = create(table)
        assert second.track_id.local_id == first.track_id.local_id
        assert second.track_id != first.track_id

    def test_reset_returns_the_discarded_records_for_publication(
        self, table: TrackTable
    ) -> None:
        """A silent reset looks downstream like every object vanishing at once."""
        created = [create(table, x=0.1 + i * 0.2) for i in range(3)]
        discarded = table.reset(TrackerEpoch(1))
        assert {r.track_id for r in discarded} == {r.track_id for r in created}


class TestOrdering:
    def test_records_come_back_in_stable_id_order(self, table: TrackTable) -> None:
        """Association indexes into this; a reordering would change tie-breaks."""
        for i in range(5):
            create(table, x=0.1 + i * 0.15)
        first = [r.track_id for r in table.records()]
        second = [r.track_id for r in table.records()]
        assert first == second == sorted(first)

    def test_order_survives_a_removal(self, table: TrackTable) -> None:
        records = [create(table, x=0.05 + i * 0.15) for i in range(5)]
        table.remove(records[2].track_id)
        remaining = [r.track_id for r in table.records()]
        assert remaining == sorted(remaining)
        assert records[2].track_id not in remaining


class TestStats:
    def test_stats_report_live_and_capacity(self, table: TrackTable) -> None:
        create(table)
        stats = table.stats()
        assert stats.live == 1
        assert stats.capacity == 8
        assert stats.saturation == pytest.approx(0.125)

    def test_stats_break_down_by_state(self, table: TrackTable) -> None:
        first = create(table, x=0.1)
        first.state = TrackState.CONFIRMED
        create(table, x=0.5)
        stats = table.stats()
        assert stats.by_state[TrackState.CONFIRMED] == 1
        assert stats.by_state[TrackState.TENTATIVE] == 1

    def test_stats_count_issued_and_retired_ids(self, table: TrackTable) -> None:
        record = create(table)
        table.remove(record.track_id)
        create(table, x=0.5)
        stats = table.stats()
        assert stats.ids_issued == 2
        assert stats.retired == 1

    def test_saturation_of_an_empty_table_is_zero(self, table: TrackTable) -> None:
        assert table.stats().saturation == 0.0


class TestMembership:
    def test_contains_finds_a_live_track(self, table: TrackTable) -> None:
        record = create(table)
        assert record.track_id in table

    def test_contains_is_false_after_removal(self, table: TrackTable) -> None:
        record = create(table)
        table.remove(record.track_id)
        assert record.track_id not in table

    def test_get_returns_none_for_an_unknown_id(self, table: TrackTable) -> None:
        other = TrackTable(CameraId("cam-99"))
        stranger = create(other)
        assert table.get(stranger.track_id) is None

    def test_removing_an_unknown_id_is_safe(self, table: TrackTable) -> None:
        other = TrackTable(CameraId("cam-99"))
        assert table.remove(create(other).track_id) is None
