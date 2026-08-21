"""M3 Frame Scheduler — allocate scarce perception capacity; process nothing yourself.

The platform's economic regulator and the primary implementation of invariant V7.
At one camera it is nearly trivial; at 100 cameras it is the difference between a
system that works and one that collapses under its own input rate.

Two properties are non-negotiable:

**Every drop is counted and attributed.** ``DropReason`` has no ``UNKNOWN``
member. A discard the platform cannot explain is a V8 violation, and sustained
shedding raises an alarm and degrades published observability — the platform
never quietly does less work than it appears to.

**The decision is sub-microsecond.** It runs on every decoded frame from every
camera. Integer phase accumulators on monotonic time, atomic counters, no
allocation, and no per-drop logging (logging 3000 drops a second is itself a
failure mode).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ...core.model.camera import PipelineProfile, SourceSemantics
from ...core.model.frame import FrameDimensions
from ...core.model.health import HealthState, ObservabilityReason
from ...core.model.ids import CameraId
from ...core.model.timebase import Duration, Instant
from ...core.ports.clock import Clock
from ...core.ports.scheduling import (
    AdmissionContext,
    AdmissionPolicyPort,
    AdmissionVerdict,
    ChangeDetectorPort,
    DropReason,
    Fidelity,
)
from ...kernel.config.schema import SchedulerSection
from ...kernel.events import BudgetExceeded, EventBus, SustainedDropAlarm
from ...kernel.health import HealthMonitor
from ...kernel.metrics import MetricName, MetricsEngine

_NANOS_PER_SECOND = 1_000_000_000


@dataclass(slots=True)
class _CameraState:
    """Per-camera scheduling state. Single-writer by that camera's actor."""

    profile: PipelineProfile
    semantics: SourceSemantics
    last_admitted_monotonic: Instant | None = None
    in_flight: int = 0
    offered_window: int = 0
    admitted_window: int = 0
    dropped_window: dict[DropReason, int] = field(default_factory=dict)
    cadence_override_fps: float | None = None
    cadence_override_expires: Instant | None = None
    alarm_active: bool = False


@dataclass(frozen=True, slots=True)
class PressureReport:
    budget_fps: float
    admitted_last_window: int
    pressure: float
    cameras: int

    @property
    def saturated(self) -> bool:
        return self.pressure >= 1.0


@dataclass(frozen=True, slots=True)
class CameraRate:
    camera_id: CameraId
    offered: int
    admitted: int
    effective_rate: float
    drops: dict[DropReason, int]


class FrameScheduler:
    """Decide whether a frame is processed, at what fidelity, under what budget."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        health: HealthMonitor,
        policy: AdmissionPolicyPort,
        config: SchedulerSection,
        change_detector: ChangeDetectorPort | None = None,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._health = health
        self._policy = policy
        self._config = config
        self._change_detector = change_detector
        self._lock = threading.RLock()
        self._cameras: dict[CameraId, _CameraState] = {}
        self._window_started = clock.monotonic()
        self._admitted_in_window = 0
        self._budget_alarm_active = False

    # --- registration ------------------------------------------------------ #

    def register_camera(
        self, camera_id: CameraId, profile: PipelineProfile, semantics: SourceSemantics
    ) -> None:
        with self._lock:
            self._cameras[camera_id] = _CameraState(profile=profile, semantics=semantics)

    def forget_camera(self, camera_id: CameraId) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)
        if self._change_detector is not None:
            self._change_detector.forget(camera_id)

    # --- admission ---------------------------------------------------------- #

    def offer(
        self,
        camera_id: CameraId,
        *,
        view: memoryview | None = None,
        dimensions: FrameDimensions | None = None,
    ) -> AdmissionVerdict:
        """Offer a decoded frame for processing.

        Args:
            view: Optional pixel view for duplicate suppression. Supplying it
                enables the cheapest saving available to the platform — in most
                deployments the majority of frames contain nothing new.
        """
        now = self._clock.monotonic()
        self._maybe_roll_window(now)

        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return AdmissionVerdict(admit=False, reason=DropReason.QUEUE_FULL)

            state.offered_window += 1
            self._expire_override(state, now)
            profile = self._effective_profile(state)
            pressure = self._pressure_locked()
            context = AdmissionContext(
                camera_id=camera_id,
                profile=profile,
                semantics=state.semantics,
                monotonic_now=now,
                last_admitted_monotonic=state.last_admitted_monotonic,
                in_flight=state.in_flight,
                budget_pressure=pressure,
                queue_full=state.in_flight >= profile.max_in_flight,
            )

        verdict = self._policy.evaluate(context)

        if verdict.admit and self._should_suppress(camera_id, view, dimensions):
            verdict = AdmissionVerdict(admit=False, reason=DropReason.DUPLICATE)

        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return AdmissionVerdict(admit=False, reason=DropReason.QUEUE_FULL)
            if verdict.admit:
                state.last_admitted_monotonic = now
                state.admitted_window += 1
                state.in_flight += 1
                self._admitted_in_window += 1
            else:
                reason = verdict.reason
                assert reason is not None  # guaranteed by AdmissionVerdict
                state.dropped_window[reason] = state.dropped_window.get(reason, 0) + 1

        if verdict.admit:
            self._metrics.counter(
                MetricName.FRAMES_ADMITTED, camera_id=str(camera_id)
            ).increment()
            if verdict.fidelity is None:
                verdict = AdmissionVerdict(
                    admit=True,
                    fidelity=Fidelity(
                        inference_width=context.profile.inference_width,
                        inference_height=context.profile.inference_height,
                    ),
                )
        else:
            assert verdict.reason is not None
            self._metrics.counter(
                MetricName.FRAMES_DROPPED,
                camera_id=str(camera_id),
                reason=verdict.reason.value,
            ).increment()
        return verdict

    def complete(self, camera_id: CameraId) -> None:
        """Mark one in-flight frame as finished. Balances ``offer``."""
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is not None and state.in_flight > 0:
                state.in_flight -= 1

    def _should_suppress(
        self,
        camera_id: CameraId,
        view: memoryview | None,
        dimensions: FrameDimensions | None,
    ) -> bool:
        if not self._config.duplicate_suppression:
            return False
        if self._change_detector is None or view is None or dimensions is None:
            return False
        return not self._change_detector.observe(camera_id, view, dimensions).changed

    # --- budget and pressure ------------------------------------------------ #

    def _pressure_locked(self) -> float:
        elapsed_ns = max(1, self._clock.monotonic().ns - self._window_started.ns)
        elapsed_seconds = elapsed_ns / _NANOS_PER_SECOND
        if elapsed_seconds <= 0:
            return 0.0
        rate = self._admitted_in_window / elapsed_seconds
        return min(4.0, rate / self._config.global_budget_fps)

    def current_pressure(self) -> PressureReport:
        with self._lock:
            return PressureReport(
                budget_fps=self._config.global_budget_fps,
                admitted_last_window=self._admitted_in_window,
                pressure=self._pressure_locked(),
                cameras=len(self._cameras),
            )

    def _maybe_roll_window(self, now: Instant) -> None:
        """Reconcile the accounting window and evaluate sustained shedding.

        Budget accounting deliberately trades precision for contention-freedom:
        being 2% off on a budget is irrelevant, being a lock contention point at
        3000 calls a second is fatal.
        """
        window_ns = self._config.drop_alarm_window_ms * 1_000_000
        with self._lock:
            if now.ns - self._window_started.ns < window_ns:
                return
            pressure = self._pressure_locked()
            rates = [
                CameraRate(
                    camera_id=camera_id,
                    offered=state.offered_window,
                    admitted=state.admitted_window,
                    effective_rate=self._effective_rate(state),
                    drops=dict(state.dropped_window),
                )
                for camera_id, state in self._cameras.items()
            ]
            alarm_states = {
                camera_id: state.alarm_active for camera_id, state in self._cameras.items()
            }
            for state in self._cameras.values():
                state.offered_window = 0
                state.admitted_window = 0
                state.dropped_window.clear()
            self._window_started = now
            self._admitted_in_window = 0
            budget_alarm_was_active = self._budget_alarm_active
            self._budget_alarm_active = pressure >= 1.0

        if pressure >= 1.0 and not budget_alarm_was_active:
            self._metrics.gauge(MetricName.BUDGET_PRESSURE).set(pressure)
            self._bus.publish(
                BudgetExceeded(occurred_at=self._clock.now(), pressure=pressure)
            )
        self._metrics.gauge(MetricName.BUDGET_PRESSURE).set(pressure)

        for rate in rates:
            self._evaluate_camera_rate(rate, alarm_states.get(rate.camera_id, False))

    def _evaluate_camera_rate(self, rate: CameraRate, alarm_was_active: bool) -> None:
        """Publish observability when shedding thins perception (V8)."""
        if rate.offered == 0:
            return
        pressure_drops = sum(
            count for reason, count in rate.drops.items() if reason.indicates_pressure
        )
        shedding = pressure_drops > 0
        below_threshold = rate.effective_rate < self._config.sustained_drop_threshold

        self._metrics.gauge(
            MetricName.EFFECTIVE_RATE, camera_id=str(rate.camera_id)
        ).set(rate.effective_rate)

        if shedding and below_threshold:
            with self._lock:
                state = self._cameras.get(rate.camera_id)
                if state is not None:
                    state.alarm_active = True
            if not alarm_was_active:
                self._bus.publish(
                    SustainedDropAlarm(
                        occurred_at=self._clock.now(),
                        partition_key=str(rate.camera_id),
                        camera_id=rate.camera_id,
                        reason="pressure",
                        effective_rate=rate.effective_rate,
                    )
                )
            self._health.set_observability(
                rate.camera_id,
                HealthState.DEGRADED,
                ObservabilityReason.SCHEDULER_SHEDDING,
                effective_rate=rate.effective_rate,
                detail=f"{pressure_drops} pressure drops in window",
            )
        elif alarm_was_active:
            with self._lock:
                state = self._cameras.get(rate.camera_id)
                if state is not None:
                    state.alarm_active = False
            self._health.set_observability(
                rate.camera_id,
                HealthState.HEALTHY,
                ObservabilityReason.NORMAL,
                effective_rate=rate.effective_rate,
            )

    def _effective_rate(self, state: _CameraState) -> float:
        """Admitted / expected-by-cadence, clamped to [0,1].

        Cadence drops are *by design* and must not read as lost observability;
        only shortfall against the camera's own target counts.
        """
        window_seconds = self._config.drop_alarm_window_ms / 1000.0
        profile = self._effective_profile(state)
        expected = max(1.0, profile.target_fps * window_seconds)
        return min(1.0, state.admitted_window / expected)

    # --- operational overrides ---------------------------------------------- #

    def override_cadence(self, camera_id: CameraId, fps: float, ttl: Duration) -> None:
        """Time-boxed operational cadence override. Always expires."""
        if fps <= 0:
            raise ValueError(f"cadence override must be positive, got {fps}")
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return
            state.cadence_override_fps = fps
            state.cadence_override_expires = self._clock.monotonic().plus(ttl)

    def _expire_override(self, state: _CameraState, now: Instant) -> None:
        if state.cadence_override_expires is not None and (
            state.cadence_override_expires.ns <= now.ns
        ):
            state.cadence_override_fps = None
            state.cadence_override_expires = None

    def _effective_profile(self, state: _CameraState) -> PipelineProfile:
        if state.cadence_override_fps is None:
            return state.profile
        base = state.profile
        return PipelineProfile(
            profile_id=base.profile_id,
            target_fps=state.cadence_override_fps,
            max_in_flight=base.max_in_flight,
            priority_class=base.priority_class,
            inference_width=base.inference_width,
            inference_height=base.inference_height,
        )

    # --- introspection ------------------------------------------------------- #

    def camera_rates(self) -> tuple[CameraRate, ...]:
        with self._lock:
            return tuple(
                CameraRate(
                    camera_id=camera_id,
                    offered=state.offered_window,
                    admitted=state.admitted_window,
                    effective_rate=self._effective_rate(state),
                    drops=dict(state.dropped_window),
                )
                for camera_id, state in self._cameras.items()
            )

    @property
    def camera_count(self) -> int:
        with self._lock:
            return len(self._cameras)
