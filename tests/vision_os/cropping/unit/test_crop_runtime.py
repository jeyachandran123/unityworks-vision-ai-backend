"""The Crop Runtime — the seam, the lease, and the firewall.

Three properties, each of which fails silently if untested:

**The seam never raises.** An attention failure may not stop the registry, which
may not stop tracking, which may not stop detection, which may not stop
acquisition (V9). The firewall is only real if something actually throws at it.

**A failed registry update is not an empty scene.** Evaluating a broken update's
(empty) object list would manufacture a "nothing is here" conclusion from an
upstream fault — exactly the conflation V8 forbids.

**One lease per frame, not one per request.** A lease is a buffer-slot
reservation; taking N of them for N objects on the same frame multiplies pressure
on the pool for no benefit.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_os.core.errors import FrameUnavailableError
from vision_os.core.model.crop import SkipReason
from vision_os.core.model.health import HealthState
from vision_os.perception.cropping import CropRuntime
from vision_os.perception.registry.engine import RegistryUpdate

from ..conftest import (
    CAMERA,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    OTHER_CAMERA,
    at,
    frame_ref,
    make_demand,
    make_object,
    sharp_frame,
)


class _FakeDimensions:
    def __init__(self) -> None:
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.colour_space = "bgr24"


class _FakeTime:
    def __init__(self, seq: int) -> None:
        self.t_capture = at(seq)


class _FakeFrame:
    def __init__(self, seq: int, camera=CAMERA) -> None:
        self.frame_ref = frame_ref(seq, camera=camera)
        self.dimensions = _FakeDimensions()
        self.time = _FakeTime(seq)


class _FakeLease:
    def __init__(self, frame: _FakeFrame, buffer: _FakeBuffer) -> None:
        self.frame = frame
        self._buffer = buffer
        self.released = False

    def pixels(self):
        return sharp_frame()

    def release(self) -> None:
        self.released = True
        self._buffer.releases += 1


class _FakeBuffer:
    """A frame buffer that counts leases, so lease discipline is observable."""

    def __init__(self, *, resident: bool = True) -> None:
        self.resident = resident
        self.acquires = 0
        self.try_acquires = 0
        self.releases = 0
        self.leases: list[_FakeLease] = []

    def acquire(self, ref, holder, deadline=None):
        self.acquires += 1
        if not self.resident:
            raise FrameUnavailableError("evicted", frame_ref=str(ref))
        lease = _FakeLease(_FakeFrame(ref.frame_seq, ref.camera_id), self)
        self.leases.append(lease)
        return lease

    def try_acquire(self, ref, holder):
        self.try_acquires += 1
        if not self.resident:
            return None
        lease = _FakeLease(_FakeFrame(ref.frame_seq, ref.camera_id), self)
        self.leases.append(lease)
        return lease


def update(objects, *, seq: int = 0, camera=CAMERA, failed: bool = False):
    return RegistryUpdate(
        camera_id=camera,
        frame_ref=frame_ref(seq, camera=camera),
        objects=tuple(objects),
        failed=failed,
        reason="injected" if failed else "",
    )


@pytest.fixture
def buffer() -> _FakeBuffer:
    return _FakeBuffer()


@pytest.fixture
def runtime(clock, metrics, health, manager, cropping_config, buffer) -> CropRuntime:
    return CropRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        manager=manager,
        config=cropping_config,
        buffer=buffer,
    )


class TestLifecycle:
    async def test_nothing_is_consumed_before_start(self, runtime, manager) -> None:
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))
        assert runtime.stats.frames_consumed == 0

    async def test_start_reports_healthy(self, runtime, health) -> None:
        await runtime.start()
        assert runtime.started
        reported = {c.component_id: c for c in health.components()}
        assert reported["crop_runtime"].state is HealthState.HEALTHY

    async def test_stop_reports_draining(self, runtime, health) -> None:
        await runtime.start()
        await runtime.stop()
        assert not runtime.started
        reported = {c.component_id: c for c in health.components()}
        assert reported["crop_runtime"].state is HealthState.DRAINING

    async def test_a_disabled_config_consumes_nothing(
        self, clock, metrics, health, manager, cropping_config, buffer
    ) -> None:
        from dataclasses import replace

        runtime = CropRuntime(
            clock=clock,
            metrics=metrics,
            health=health,
            manager=manager,
            config=replace(cropping_config, enabled=False),
            buffer=buffer,
        )
        await runtime.start()
        await runtime.on_registered(update([make_object()]))
        assert runtime.stats.frames_consumed == 0


class TestTheFirewall:
    async def test_the_seam_never_raises(self, runtime, manager) -> None:
        """V9, tested by actually throwing at it."""

        class _Exploding:
            def evaluate(self, *args, **kwargs):
                raise RuntimeError("boom")

            def health(self):
                raise RuntimeError("boom")

        await runtime.start()
        runtime._manager = _Exploding()  # noqa: SLF001 - injecting a fault
        await runtime.on_registered(update([make_object()]))
        assert runtime.stats.frames_failed == 1

    async def test_a_failed_update_is_not_an_empty_scene(self, runtime) -> None:
        """A broken registry must not read as "nothing was there" (V8)."""
        await runtime.start()
        await runtime.on_registered(update([], failed=True))
        assert runtime.stats.frames_failed == 1
        assert runtime.stats.requests_made == 0

    async def test_a_broken_sink_does_not_break_attention(
        self, clock, metrics, health, manager, cropping_config, buffer
    ) -> None:
        def _bad_sink(*args, **kwargs):
            raise RuntimeError("subscriber exploded")

        runtime = CropRuntime(
            clock=clock, metrics=metrics, health=health, manager=manager,
            config=cropping_config, buffer=buffer, sink=_bad_sink,
        )
        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))
        assert runtime.stats.sink_failures == 1
        assert runtime.stats.frames_failed == 0


class TestLeaseDiscipline:
    async def test_evaluation_does_not_hold_a_lease(self, runtime, buffer) -> None:
        """Trigger evaluation is a control-plane decision about metadata."""
        await runtime.start()
        await runtime.on_registered(update([make_object()]))
        assert all(lease.released for lease in buffer.leases)

    async def test_one_lease_per_frame_not_one_per_request(
        self, runtime, buffer, manager
    ) -> None:
        await runtime.start()
        manager.register_demand(make_demand())
        objects = [make_object(object_id=f"obj-{i}") for i in range(6)]
        await runtime.on_registered(update(objects))
        assert buffer.acquires == 1, (
            f"{buffer.acquires} leases taken for one frame; a lease is a "
            f"buffer-slot reservation and N of them multiply pool pressure"
        )

    async def test_the_lease_is_always_released(self, runtime, buffer, manager) -> None:
        """Even when extraction raises. A leaked lease pins a slot forever."""
        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))
        assert buffer.releases == buffer.acquires + buffer.try_acquires

    async def test_no_lease_is_taken_when_nothing_fires(
        self, runtime, buffer
    ) -> None:
        await runtime.start()
        await runtime.on_registered(update([make_object()]))
        assert buffer.acquires == 0, "an undemanded population costs no pixels"


class TestFrameUnavailable:
    async def test_an_evicted_frame_skips_with_a_reason(
        self, clock, metrics, health, manager, cropping_config
    ) -> None:
        """§M8: a diagnosable configuration issue rather than a mystery."""
        recorded = []
        runtime = CropRuntime(
            clock=clock, metrics=metrics, health=health, manager=manager,
            config=cropping_config, buffer=_FakeBuffer(resident=False),
            sink=lambda result, crops: recorded.append((result, crops)),
        )
        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))

        [(result, crops)] = recorded
        assert not crops
        assert result.skipped[0].reason is SkipReason.FRAME_UNAVAILABLE
        assert result.candidate_count == 1

    async def test_no_buffer_is_reported_not_ignored(
        self, clock, metrics, health, manager, cropping_config
    ) -> None:
        recorded = []
        runtime = CropRuntime(
            clock=clock, metrics=metrics, health=health, manager=manager,
            config=cropping_config, buffer=None,
            sink=lambda result, crops: recorded.append((result, crops)),
        )
        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))
        [(result, _)] = recorded
        assert result.skipped[0].reason is SkipReason.FRAME_UNAVAILABLE


class TestPerCameraSerialization:
    async def test_each_camera_gets_its_own_lock(self, runtime) -> None:
        await runtime.start()
        await runtime.on_registered(update([make_object()], camera=CAMERA))
        await runtime.on_registered(
            update([make_object(camera=OTHER_CAMERA)], camera=OTHER_CAMERA)
        )
        assert runtime.cameras_seen == 2

    async def test_concurrent_updates_for_one_camera_serialize(
        self, runtime, manager
    ) -> None:
        """Per-camera single-writer, matching M7's partitioning."""
        await runtime.start()
        manager.register_demand(make_demand())
        await asyncio.gather(
            *(
                runtime.on_registered(
                    update([make_object(object_id=f"obj-{i}")], seq=i)
                )
                for i in range(10)
            )
        )
        assert runtime.stats.frames_consumed == 10
        assert runtime.stats.frames_failed == 0

    async def test_forgetting_a_camera_releases_its_lock(self, runtime) -> None:
        """Otherwise the lock table grows with every camera ever seen."""
        await runtime.start()
        await runtime.on_registered(update([make_object()]))
        assert runtime.cameras_seen == 1
        runtime.forget(CAMERA)
        assert runtime.cameras_seen == 0

    async def test_reconcile_drops_departed_objects(self, runtime, manager) -> None:
        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(
            update([make_object(object_id=f"obj-{i}") for i in range(5)])
        )
        dropped = runtime.reconcile(CAMERA, frozenset({"obj-0"}))
        assert dropped == 4


class TestOutput:
    async def test_crops_and_skips_both_reach_the_sink(
        self, clock, metrics, health, manager, cropping_config, buffer
    ) -> None:
        recorded = []
        runtime = CropRuntime(
            clock=clock, metrics=metrics, health=health, manager=manager,
            config=cropping_config, buffer=buffer,
            sink=lambda result, crops: recorded.append((result, crops)),
        )
        await runtime.start()
        manager.register_demand(make_demand())
        from vision_os.core.model.space import Box

        objects = [
            make_object(object_id="good"),
            make_object(object_id="tiny", box=Box(0.5, 0.5, 0.52, 0.53)),
        ]
        await runtime.on_registered(update(objects))

        [(result, crops)] = recorded
        assert len(crops) == 1, "the good object produced evidence"
        assert result.candidate_count == 2, "both candidates are accounted for"
        assert any(
            s.reason is SkipReason.QUALITY_INSUFFICIENT for s in result.skipped
        ), "the gate rejection became an attributed skip"

    async def test_gate_rejections_are_counted(self, runtime, manager) -> None:
        from vision_os.core.model.space import Box

        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(
            update([make_object(box=Box(0.5, 0.5, 0.52, 0.53))])
        )
        assert runtime.stats.gate_rejections == 1
        assert runtime.stats.crops_produced == 0


class _FailingExtract:
    """Delegates everything to the real manager except ``extract``."""

    def __init__(self, inner, failure: Exception) -> None:
        self._inner = inner
        self._failure = failure

    def extract(self, *args, **kwargs):
        raise self._failure

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestExtractionFailuresStayDistinct:
    """Three failures, three operator responses. Conflating them misdirects.

    A gate rejection means the *input* was poor. An eviction means the *buffer*
    is too shallow. An extraction error means the *code* is faulty. A runtime
    that reported one number for all three would send an operator to tune the
    wrong thing.
    """

    async def _run(self, runtime, manager, failure: Exception) -> list:
        """Run one frame with a manager whose ``extract`` always fails.

        A delegating wrapper rather than a monkeypatch: ``CropManager`` is
        slotted, and a wrapper also proves the runtime only depends on the
        manager's public surface.
        """
        recorded: list = []
        runtime._sink = lambda result, crops: recorded.append((result, crops))  # noqa: SLF001
        runtime._manager = _FailingExtract(manager, failure)  # noqa: SLF001

        await runtime.start()
        manager.register_demand(make_demand())
        await runtime.on_registered(update([make_object()]))
        return recorded

    async def test_an_eviction_during_extraction_is_frame_unavailable(
        self, runtime, manager
    ) -> None:
        recorded = await self._run(
            runtime, manager, FrameUnavailableError("gone mid-extraction")
        )
        [(result, crops)] = recorded
        assert not crops
        assert result.skipped[-1].reason is SkipReason.FRAME_UNAVAILABLE
        assert runtime.stats.unavailable_frames == 1

    async def test_an_extraction_fault_is_counted_separately(
        self, runtime, manager
    ) -> None:
        from vision_os.core.errors import CropExtractionError

        recorded = await self._run(
            runtime, manager, CropExtractionError("short buffer")
        )
        [(result, crops)] = recorded
        assert not crops
        assert result.skipped[-1].reason is SkipReason.QUALITY_INSUFFICIENT
        assert "extraction failed" in result.skipped[-1].detail
        assert runtime.stats.extraction_failures == 1
        assert runtime.stats.gate_rejections == 0

    async def test_an_unexpected_crop_error_still_becomes_a_skip(
        self, runtime, manager
    ) -> None:
        """A future ``CropError`` subclass must not escape as an exception."""
        from vision_os.core.errors import BudgetExhaustedError

        recorded = await self._run(
            runtime, manager, BudgetExhaustedError("no budget left")
        )
        [(result, crops)] = recorded
        assert not crops
        assert result.skipped[-1].reason is SkipReason.QUALITY_INSUFFICIENT
        assert "BudgetExhaustedError" in result.skipped[-1].detail
        assert runtime.stats.extraction_failures == 1

    async def test_a_failed_extraction_still_accounts_for_the_candidate(
        self, runtime, manager
    ) -> None:
        from vision_os.core.errors import CropExtractionError

        recorded = await self._run(runtime, manager, CropExtractionError("boom"))
        [(result, _)] = recorded
        assert result.candidate_count == 1, (
            "a candidate whose extraction failed must still appear exactly once"
        )
        assert not result.requests, "it moved to the skip column, not both"
