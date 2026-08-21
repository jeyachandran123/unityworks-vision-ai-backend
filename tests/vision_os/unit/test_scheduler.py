"""M3 Frame Scheduler — cadence, budget, and attributed drops.

Invariant V8 is the theme: every discard carries a reason, and sustained shedding
degrades *published* observability rather than quietly thinning perception.
"""

from __future__ import annotations

import pytest

from vision_os.acquisition import FrameScheduler
from vision_os.adapters.scheduling import (
    AdmitAllPolicy,
    CadenceAdmissionPolicy,
    ResolutionLadderPolicy,
    SampledDigestChangeDetector,
)
from vision_os.core.model.camera import PipelineProfile, SourceSemantics
from vision_os.core.model.frame import FrameDimensions
from vision_os.core.model.health import HealthState, ObservabilityReason
from vision_os.core.model.ids import CameraId, ProfileId
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.scheduling import (
    AdmissionContext,
    AdmissionVerdict,
    DropReason,
)
from vision_os.kernel.clock import VirtualClock
from vision_os.kernel.config.schema import SchedulerSection
from vision_os.kernel.events import EventBus
from vision_os.kernel.health import HealthMonitor
from vision_os.kernel.metrics import MetricName, MetricsEngine

from ..conftest import CAMERA, FRAME_BYTES

PROFILE = PipelineProfile(profile_id=ProfileId("standard"), target_fps=5.0, max_in_flight=4)


def _context(**overrides) -> AdmissionContext:
    defaults = {
        "camera_id": CAMERA,
        "profile": PROFILE,
        "semantics": SourceSemantics.REALTIME,
        "monotonic_now": Instant(10_000_000_000),
        "last_admitted_monotonic": None,
        "in_flight": 0,
        "budget_pressure": 0.0,
        "queue_full": False,
    }
    defaults.update(overrides)
    return AdmissionContext(**defaults)


class TestVerdictInvariant:
    def test_a_drop_without_a_reason_is_structurally_impossible(self) -> None:
        """Invariant V8, enforced by the type rather than by discipline."""
        with pytest.raises(ValueError, match="every drop must carry an attributed reason"):
            AdmissionVerdict(admit=False)

    def test_an_admitted_frame_cannot_carry_a_drop_reason(self) -> None:
        with pytest.raises(ValueError, match="cannot carry a drop reason"):
            AdmissionVerdict(admit=True, reason=DropReason.CADENCE)


class TestCadencePolicy:
    def test_admits_when_idle(self) -> None:
        verdict = CadenceAdmissionPolicy().evaluate(_context())
        assert verdict.admit
        assert verdict.fidelity is not None

    def test_enforces_the_target_interval(self) -> None:
        policy = CadenceAdmissionPolicy()
        now = Instant(10_000_000_000)
        too_soon = policy.evaluate(
            _context(monotonic_now=now, last_admitted_monotonic=Instant(now.ns - 50_000_000))
        )
        assert not too_soon.admit
        assert too_soon.reason is DropReason.CADENCE

        due = policy.evaluate(
            _context(monotonic_now=now, last_admitted_monotonic=Instant(now.ns - 250_000_000))
        )
        assert due.admit

    def test_queue_full_takes_precedence_over_cadence(self) -> None:
        """Never pile work onto a stage that is already behind."""
        verdict = CadenceAdmissionPolicy().evaluate(_context(queue_full=True))
        assert verdict.reason is DropReason.QUEUE_FULL

    def test_in_flight_limit_is_respected(self) -> None:
        verdict = CadenceAdmissionPolicy().evaluate(_context(in_flight=4))
        assert verdict.reason is DropReason.QUEUE_FULL

    def test_budget_is_checked_after_cadence(self) -> None:
        """A healthy low-rate camera is never charged for a neighbour's saturation."""
        policy = CadenceAdmissionPolicy()
        now = Instant(10_000_000_000)
        verdict = policy.evaluate(
            _context(
                monotonic_now=now,
                last_admitted_monotonic=Instant(now.ns - 10_000_000),
                budget_pressure=5.0,
            )
        )
        assert verdict.reason is DropReason.CADENCE

    def test_budget_exhaustion_is_attributed(self) -> None:
        verdict = CadenceAdmissionPolicy().evaluate(_context(budget_pressure=1.5))
        assert verdict.reason is DropReason.BUDGET_EXHAUSTED

    def test_is_deterministic(self) -> None:
        policy = CadenceAdmissionPolicy()
        context = _context()
        assert policy.evaluate(context).admit == policy.evaluate(context).admit


class TestResolutionLadder:
    def test_degrades_fidelity_under_pressure_and_says_so(self) -> None:
        """Lowering resolution changes what the platform can see — it is recorded."""
        policy = ResolutionLadderPolicy(degraded_scale=0.5, pressure_threshold=0.8)
        normal = policy.evaluate(_context(budget_pressure=0.1))
        assert normal.fidelity.tier == "primary"

        degraded = policy.evaluate(_context(budget_pressure=0.9))
        assert degraded.admit
        assert degraded.fidelity.tier == "degraded"
        assert degraded.fidelity.inference_width < normal.fidelity.inference_width


class TestSchedulerAdmission:
    def test_offer_admits_and_counts(
        self, scheduler: FrameScheduler, metrics: MetricsEngine
    ) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        verdict = scheduler.offer(CAMERA)
        assert verdict.admit
        assert metrics.snapshot().counter_value(
            MetricName.FRAMES_ADMITTED, camera_id=str(CAMERA)
        ) == 1

    def test_every_drop_is_counted_with_its_reason(
        self, scheduler: FrameScheduler, metrics: MetricsEngine, clock: VirtualClock
    ) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        scheduler.offer(CAMERA)
        scheduler.complete(CAMERA)
        scheduler.offer(CAMERA)  # too soon — cadence drop

        assert metrics.snapshot().counter_value(
            MetricName.FRAMES_DROPPED, camera_id=str(CAMERA), reason="cadence"
        ) == 1

    def test_unregistered_camera_is_dropped_not_crashed(
        self, scheduler: FrameScheduler
    ) -> None:
        verdict = scheduler.offer(CameraId("never-registered"))
        assert not verdict.admit
        assert verdict.reason is not None

    def test_complete_frees_an_in_flight_slot(
        self, scheduler: FrameScheduler, clock: VirtualClock
    ) -> None:
        profile = PipelineProfile(
            profile_id=ProfileId("p"), target_fps=1000.0, max_in_flight=1
        )
        scheduler.register_camera(CAMERA, profile, SourceSemantics.REALTIME)

        assert scheduler.offer(CAMERA).admit
        clock.advance(Duration.from_millis(10))
        assert scheduler.offer(CAMERA).reason is DropReason.QUEUE_FULL

        scheduler.complete(CAMERA)
        clock.advance(Duration.from_millis(10))
        assert scheduler.offer(CAMERA).admit

    def test_forget_camera_removes_state(self, scheduler: FrameScheduler) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        assert scheduler.camera_count == 1
        scheduler.forget_camera(CAMERA)
        assert scheduler.camera_count == 0


class TestDuplicateSuppression:
    def test_identical_frames_are_suppressed_when_enabled(
        self,
        clock: VirtualClock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        dimensions: FrameDimensions,
    ) -> None:
        """The cheapest saving available to the platform (invariant V7)."""
        scheduler = FrameScheduler(
            clock=clock,
            bus=bus,
            metrics=metrics,
            health=health,
            policy=AdmitAllPolicy(),
            config=SchedulerSection(global_budget_fps=1000.0, duplicate_suppression=True),
            change_detector=SampledDigestChangeDetector(),
        )
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        payload = memoryview(bytes([7]) * FRAME_BYTES)

        first = scheduler.offer(CAMERA, view=payload, dimensions=dimensions)
        second = scheduler.offer(CAMERA, view=payload, dimensions=dimensions)

        assert first.admit
        assert not second.admit
        assert second.reason is DropReason.DUPLICATE

    def test_suppression_is_off_by_default(
        self, scheduler: FrameScheduler, dimensions: FrameDimensions
    ) -> None:
        scheduler.register_camera(
            CAMERA,
            PipelineProfile(profile_id=ProfileId("p"), target_fps=1000.0),
            SourceSemantics.REALTIME,
        )
        payload = memoryview(bytes([7]) * FRAME_BYTES)
        assert scheduler.offer(CAMERA, view=payload, dimensions=dimensions).admit


class TestSustainedShedding:
    def test_pressure_shedding_degrades_published_observability(
        self,
        clock: VirtualClock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
    ) -> None:
        """The platform never quietly does less work than it appears to (V8)."""
        scheduler = FrameScheduler(
            clock=clock,
            bus=bus,
            metrics=metrics,
            health=health,
            policy=CadenceAdmissionPolicy(),
            config=SchedulerSection(
                global_budget_fps=0.0001,
                sustained_drop_threshold=0.9,
                drop_alarm_window_ms=100,
            ),
        )
        profile = PipelineProfile(profile_id=ProfileId("p"), target_fps=100.0)
        scheduler.register_camera(CAMERA, profile, SourceSemantics.REALTIME)
        health.register_camera(CAMERA)
        health.set_observability(CAMERA, HealthState.HEALTHY)

        alarms = bus.subscribe(["scheduler.sustained_drop"])
        for _ in range(20):
            scheduler.offer(CAMERA)
            scheduler.complete(CAMERA)
            clock.advance(Duration.from_millis(20))
        scheduler.offer(CAMERA)

        assert alarms.drain(), "sustained pressure shedding must raise an alarm"
        state = health.observability(CAMERA)
        assert state.status is HealthState.DEGRADED
        assert state.reason is ObservabilityReason.SCHEDULER_SHEDDING

    def test_cadence_drops_alone_do_not_degrade_observability(
        self, scheduler: FrameScheduler, clock: VirtualClock, health: HealthMonitor
    ) -> None:
        """Cadence drops are by design and must not read as lost observability."""
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        health.register_camera(CAMERA)
        health.set_observability(CAMERA, HealthState.HEALTHY)

        for _ in range(50):
            scheduler.offer(CAMERA)
            scheduler.complete(CAMERA)
            clock.advance(Duration.from_millis(5))
        scheduler.offer(CAMERA)

        assert health.observability(CAMERA).status is HealthState.HEALTHY


class TestCadenceOverride:
    def test_override_changes_the_effective_rate(
        self, scheduler: FrameScheduler, clock: VirtualClock
    ) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        scheduler.offer(CAMERA)
        scheduler.complete(CAMERA)

        scheduler.override_cadence(CAMERA, 1000.0, Duration.from_millis(10_000))
        clock.advance(Duration.from_millis(5))
        assert scheduler.offer(CAMERA).admit

    def test_override_expires(self, scheduler: FrameScheduler, clock: VirtualClock) -> None:
        """Operational overrides are always time-boxed.

        The window is chosen deliberately: 150ms is past the 100ms override TTL
        but inside the base profile's 200ms cadence interval, so the reinstated
        base rate is what produces the drop.
        """
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        scheduler.override_cadence(CAMERA, 1000.0, Duration.from_millis(100))
        scheduler.offer(CAMERA)
        scheduler.complete(CAMERA)

        clock.advance(Duration.from_millis(150))
        verdict = scheduler.offer(CAMERA)
        assert verdict.reason is DropReason.CADENCE

    def test_override_rejects_non_positive_rate(self, scheduler: FrameScheduler) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        with pytest.raises(ValueError, match="positive"):
            scheduler.override_cadence(CAMERA, 0.0, Duration.from_millis(100))


class TestPressureReport:
    def test_reports_budget_and_camera_count(self, scheduler: FrameScheduler) -> None:
        scheduler.register_camera(CAMERA, PROFILE, SourceSemantics.REALTIME)
        report = scheduler.current_pressure()
        assert report.budget_fps == 1000.0
        assert report.cameras == 1
        assert not report.saturated


class TestDropReasonSemantics:
    def test_cadence_is_not_a_pressure_signal(self) -> None:
        assert not DropReason.CADENCE.indicates_pressure
        assert not DropReason.DUPLICATE.indicates_pressure

    def test_saturation_reasons_indicate_pressure(self) -> None:
        for reason in (
            DropReason.BUDGET_EXHAUSTED,
            DropReason.TENANT_QUOTA,
            DropReason.QUEUE_FULL,
            DropReason.DEADLINE_EXPIRED,
        ):
            assert reason.indicates_pressure

    def test_there_is_no_unknown_drop_reason(self) -> None:
        """An unattributed discard would be a V8 violation."""
        assert "UNKNOWN" not in {member.name for member in DropReason}
