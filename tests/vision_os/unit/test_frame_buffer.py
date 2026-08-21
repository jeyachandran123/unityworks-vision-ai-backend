"""M4 Frame Buffer — leases, pins, eviction, and bounded memory.

The properties defended here are the ones that fail silently and slowly: a pinned
frame freed under pressure, a leaked lease exhausting a shared pool, and a pool
that grows imperceptibly until a node dies on day 26.
"""

from __future__ import annotations

import pytest

from vision_os.acquisition import FrameBuffer
from vision_os.adapters.memory import HostMemoryPool
from vision_os.core.errors import FrameUnavailableError, PoolExhaustedError
from vision_os.core.model.camera import SourceSemantics
from vision_os.core.model.frame import FrameDimensions, PrivacyState
from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
from vision_os.core.model.timebase import (
    ClockQuality,
    Duration,
    FrameTime,
)
from vision_os.kernel.clock import VirtualClock

from ..conftest import CAMERA, FRAME_BYTES


@pytest.fixture
def buffer_config(narrow_buffer_config):
    """This module tests eviction and backpressure — use the shallow ring."""
    return narrow_buffer_config


def _time(clock: VirtualClock) -> FrameTime:
    now = clock.now()
    return FrameTime(
        pts=0,
        t_capture=now,
        t_capture_uncertainty=Duration.from_millis(10),
        t_ingest=now,
        t_decoded=now,
        clock_quality=ClockQuality.NTP_SYNCED,
    )


def _publish(
    buffer: FrameBuffer,
    clock: VirtualClock,
    dimensions: FrameDimensions,
    seq: int,
    *,
    camera_id: CameraId = CAMERA,
    semantics: SourceSemantics = SourceSemantics.REALTIME,
    fill: int = 1,
):
    slot = buffer.acquire_slot(camera_id, semantics)
    slot.memory()[:FRAME_BYTES] = bytes([fill]) * FRAME_BYTES
    return buffer.publish(
        slot,
        frame_ref=FrameRef(camera_id, StreamEpoch(0), FrameSeq(seq)),
        time=_time(clock),
        dimensions=dimensions,
        privacy_state=PrivacyState.MASKED,
        bytes_written=FRAME_BYTES,
    )


class TestPublishAndLease:
    def test_published_frame_is_leasable(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        lease = buffer.acquire(frame.frame_ref, holder_id="detector")
        assert lease.frame.frame_ref == frame.frame_ref
        lease.release()

    def test_lease_pixels_are_read_only(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """No stage may mutate shared pixels (01_LAYERED §4.3)."""
        frame = _publish(buffer, clock, dimensions, 0)
        with buffer.acquire(frame.frame_ref, "detector") as lease:
            assert lease.pixels().readonly
            with pytest.raises(TypeError):
                lease.pixels()[0] = 9

    def test_multiple_holders_share_one_frame_without_copying(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        first = buffer.acquire(frame.frame_ref, "detector")
        second = buffer.acquire(frame.frame_ref, "crop")
        assert bytes(first.pixels()) == bytes(second.pixels())
        first.release()
        second.release()

    def test_release_is_idempotent(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        lease = buffer.acquire(frame.frame_ref, "detector")
        lease.release()
        lease.release()
        assert buffer.stats().leases_active == 0

    def test_acquire_on_absent_frame_is_typed(self, buffer: FrameBuffer) -> None:
        """A normal, expected outcome — callers degrade and count."""
        with pytest.raises(FrameUnavailableError):
            buffer.acquire(FrameRef(CAMERA, StreamEpoch(0), FrameSeq(999)), "crop")

    def test_try_acquire_returns_none_rather_than_raising(
        self, buffer: FrameBuffer
    ) -> None:
        assert buffer.try_acquire(FrameRef(CAMERA, StreamEpoch(0), FrameSeq(9)), "crop") is None

    def test_refuses_to_publish_a_mask_failed_frame(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Fail-closed enforced at the buffer as well as the type (12_SECURITY)."""
        slot = buffer.acquire_slot(CAMERA)
        with pytest.raises(ValueError, match="refusing to publish"):
            buffer.publish(
                slot,
                frame_ref=FrameRef(CAMERA, StreamEpoch(0), FrameSeq(0)),
                time=_time(clock),
                dimensions=dimensions,
                privacy_state=PrivacyState.MASK_FAILED,
                bytes_written=FRAME_BYTES,
            )


class TestEviction:
    def test_realtime_evicts_oldest_when_ring_is_full(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Latency is protected over completeness for live sources."""
        frames = [_publish(buffer, clock, dimensions, i) for i in range(3)]
        assert buffer.is_resident(frames[0].frame_ref)

        _publish(buffer, clock, dimensions, 3)
        assert not buffer.is_resident(frames[0].frame_ref)
        assert buffer.stats().evictions == 1

    def test_archival_blocks_rather_than_dropping(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """Completeness is protected over latency for recorded sources."""
        for i in range(3):
            _publish(buffer, clock, dimensions, i, semantics=SourceSemantics.ARCHIVAL)
        with pytest.raises(PoolExhaustedError):
            buffer.acquire_slot(CAMERA, SourceSemantics.ARCHIVAL)

    def test_pinned_frame_is_never_evicted(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """The Crop Manager's guarantee that a frame survives past detection."""
        first = _publish(buffer, clock, dimensions, 0)
        buffer.pin(first.frame_ref, Duration.from_millis(60_000), reason="crop")

        for i in range(1, 4):
            _publish(buffer, clock, dimensions, i)

        assert buffer.is_resident(first.frame_ref)

    def test_leased_frame_is_never_evicted(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        first = _publish(buffer, clock, dimensions, 0)
        lease = buffer.acquire(first.frame_ref, "slow-stage")
        for i in range(1, 4):
            _publish(buffer, clock, dimensions, i)
        assert buffer.is_resident(first.frame_ref)
        lease.release()

    def test_unpin_allows_eviction_again(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        first = _publish(buffer, clock, dimensions, 0)
        handle = buffer.pin(first.frame_ref, Duration.from_millis(60_000), reason="crop")
        buffer.unpin(handle)
        for i in range(1, 5):
            _publish(buffer, clock, dimensions, i)
        assert not buffer.is_resident(first.frame_ref)


class TestLeaseDeadlines:
    def test_expired_lease_is_force_broken(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """One stuck stage must not exhaust the pool for every camera (V9)."""
        frame = _publish(buffer, clock, dimensions, 0)
        lease = buffer.acquire(frame.frame_ref, "stuck-stage")

        clock.advance(Duration.from_millis(5_000))
        buffer.sweep()

        assert lease.released
        with pytest.raises(FrameUnavailableError):
            _ = lease.frame
        assert buffer.stats().lease_leaks == 1

    def test_lease_leak_is_attributed_to_its_holder(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions, bus
    ) -> None:
        subscription = bus.subscribe(["buffer.lease_leaked"])
        frame = _publish(buffer, clock, dimensions, 0)
        buffer.acquire(frame.frame_ref, "the-culprit")
        clock.advance(Duration.from_millis(5_000))
        buffer.sweep()

        events = subscription.drain()
        assert events and events[0].holder_id == "the-culprit"

    def test_pin_ttl_expires(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        buffer.pin(frame.frame_ref, Duration.from_millis(100), reason="crop")
        clock.advance(Duration.from_millis(500))
        buffer.sweep()
        assert buffer.stats().pins_active == 0


class TestBoundedMemory:
    def test_history_window_reclaims_old_frames(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        clock.advance(Duration.from_millis(10_000))
        assert buffer.sweep() >= 1
        assert not buffer.is_resident(frame.frame_ref)

    def test_pool_returns_to_baseline_after_cycling(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions, pool
    ) -> None:
        """The soak-failure guard: no steady-state growth."""
        baseline = pool.stats().in_use
        for cycle in range(200):
            frame = _publish(buffer, clock, dimensions, cycle)
            lease = buffer.acquire(frame.frame_ref, "detector")
            lease.release()
            clock.advance(Duration.from_millis(10_000))
            buffer.sweep()
        assert pool.stats().in_use == baseline

    def test_forget_camera_releases_everything(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions, pool
    ) -> None:
        for i in range(3):
            _publish(buffer, clock, dimensions, i)
        buffer.forget_camera(CAMERA)
        assert pool.stats().in_use == 0

    def test_per_camera_rings_are_independent(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        """No cross-camera write contention, and no cross-camera eviction."""
        other = CameraId("cam-02")
        first = _publish(buffer, clock, dimensions, 0, camera_id=CAMERA)
        for i in range(4):
            _publish(buffer, clock, dimensions, i, camera_id=other)
        assert buffer.is_resident(first.frame_ref)


class TestPoolAdapter:
    def test_oversized_request_is_rejected(self) -> None:
        pool = HostMemoryPool(slots=2, bytes_per_slot=16)
        with pytest.raises(PoolExhaustedError, match="exceeds slot size"):
            pool.allocate(1_000)

    def test_exhaustion_is_typed(self) -> None:
        pool = HostMemoryPool(slots=1, bytes_per_slot=16)
        pool.allocate(16)
        with pytest.raises(PoolExhaustedError, match="exhausted"):
            pool.allocate(16)

    def test_double_release_does_not_corrupt_occupancy(self) -> None:
        pool = HostMemoryPool(slots=2, bytes_per_slot=16)
        allocation = pool.allocate(16)
        pool.release(allocation)
        pool.release(allocation)
        assert pool.stats().in_use == 0

    def test_rejects_invalid_construction(self) -> None:
        with pytest.raises(ValueError, match="slots"):
            HostMemoryPool(slots=0, bytes_per_slot=16)
        with pytest.raises(ValueError, match="bytes_per_slot"):
            HostMemoryPool(slots=1, bytes_per_slot=0)


class TestStats:
    def test_stats_reflect_activity(
        self, buffer: FrameBuffer, clock: VirtualClock, dimensions: FrameDimensions
    ) -> None:
        frame = _publish(buffer, clock, dimensions, 0)
        lease = buffer.acquire(frame.frame_ref, "detector")
        buffer.pin(frame.frame_ref, Duration.from_millis(1_000), "crop")

        stats = buffer.stats()
        assert stats.slots_in_use == 1
        assert stats.leases_active == 1
        assert stats.pins_active == 1
        lease.release()
        assert buffer.stats().leases_active == 0
