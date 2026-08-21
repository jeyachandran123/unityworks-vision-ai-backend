"""M20 Health Monitor — know what is working. Fix nothing, decide no consequence.

Two responsibilities set this module apart from ordinary health checking:

**Silence is never health.** A component that stops reporting is treated as
unhealthy after a timeout. This default is the single most important line in the
module.

**Coverage, not just health.** The monitor translates component health into the
*observability* signal consumers depend on (invariant V8): whether a camera can
currently see, and if not, why. A camera that is streaming and decoding perfectly
while pointed at the back of a parked truck is healthy by every naive metric and
useless in fact.

**Flow 1 boundary.** This module produces observability *state and events*. The
conversion of that state into ``coverage``-type Observations belongs to the
Observation Builder (Flow 6) and is deliberately absent.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from ...core.model.health import (
    ComponentHealth,
    CoverageGap,
    HealthState,
    ObservabilityReason,
    ObservabilityState,
)
from ...core.model.ids import CameraId, ModuleId
from ...core.model.timebase import Duration, Instant
from ...core.ports.clock import Clock
from ..config.schema import HealthSection
from ..events import CoverageChanged, EventBus, HealthChanged, SilentFailureSuspected
from ..metrics import MetricName, MetricsEngine


@dataclass(slots=True)
class _ComponentRecord:
    health: ComponentHealth
    consecutive_matching: int = 0
    pending_state: HealthState | None = None


@dataclass(slots=True)
class _CameraRecord:
    state: ObservabilityState
    open_gap: CoverageGap | None = None
    frozen_frame_streak: int = 0
    last_frame_digest: int | None = None
    closed_gaps: deque[CoverageGap] = field(default_factory=lambda: deque(maxlen=256))


@dataclass(frozen=True, slots=True)
class SiteHealth:
    total_cameras: int
    observing: int
    degraded: int
    blind: int
    failed: int

    @property
    def observable_fraction(self) -> float:
        if self.total_cameras == 0:
            return 1.0
        return self.observing / self.total_cameras


class HealthMonitor:
    """Aggregate component reports into camera and site observability."""

    def __init__(
        self,
        *,
        clock: Clock,
        bus: EventBus,
        metrics: MetricsEngine,
        config: HealthSection,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._metrics = metrics
        self._config = config
        self._lock = threading.RLock()
        self._components: dict[ModuleId, _ComponentRecord] = {}
        self._cameras: dict[CameraId, _CameraRecord] = {}

    # --- component reporting ---------------------------------------------- #

    def report(self, health: ComponentHealth) -> None:
        """Components push; the kernel decides.

        A module that computes its own health verdict cannot be composed.
        """
        changed = False
        with self._lock:
            record = self._components.get(health.component_id)
            if record is None:
                self._components[health.component_id] = _ComponentRecord(health=health)
                changed = True
            else:
                previous_state = record.health.state
                record.health = health
                if health.state is previous_state:
                    record.pending_state = None
                    record.consecutive_matching = 0
                else:
                    # Hysteresis: a state change requires persistence, so that a
                    # flapping component does not produce an alarm storm.
                    if record.pending_state is health.state:
                        record.consecutive_matching += 1
                    else:
                        record.pending_state = health.state
                        record.consecutive_matching = 1
                    if record.consecutive_matching < self._config.hysteresis_samples:
                        record.health = ComponentHealth(
                            component_id=health.component_id,
                            state=previous_state,
                            reported_at=health.reported_at,
                            detail=health.detail,
                            metrics=health.metrics,
                        )
                    else:
                        record.pending_state = None
                        record.consecutive_matching = 0
                        changed = True

        if changed:
            self._bus.publish(
                HealthChanged(
                    occurred_at=self._clock.now(),
                    partition_key=str(health.component_id),
                    component_id=health.component_id,
                    state=health.state.value,
                )
            )

    def component_health(self, component_id: ModuleId) -> ComponentHealth:
        """Silence is never health: a stale report is reported as ``FAILED``."""
        with self._lock:
            record = self._components.get(component_id)
            if record is None:
                return ComponentHealth(
                    component_id=component_id,
                    state=HealthState.FAILED,
                    reported_at=self._clock.now(),
                    detail="never reported",
                )
            age = self._clock.now().since(record.health.reported_at)
            if age.millis > self._config.report_timeout_ms:
                return ComponentHealth(
                    component_id=component_id,
                    state=HealthState.FAILED,
                    reported_at=record.health.reported_at,
                    detail=f"no report for {age.millis:.0f}ms (timeout "
                    f"{self._config.report_timeout_ms}ms)",
                )
            return record.health

    def components(self) -> tuple[ComponentHealth, ...]:
        with self._lock:
            ids = list(self._components)
        return tuple(self.component_health(cid) for cid in ids)

    # --- camera observability ---------------------------------------------- #

    def register_camera(self, camera_id: CameraId) -> None:
        with self._lock:
            if camera_id in self._cameras:
                return
            self._cameras[camera_id] = _CameraRecord(
                state=ObservabilityState(
                    camera_id=camera_id,
                    status=HealthState.STARTING,
                    since=self._clock.now(),
                    reason=ObservabilityReason.STARTING_UP,
                    effective_rate=0.0,
                )
            )

    def forget_camera(self, camera_id: CameraId) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)

    def set_observability(
        self,
        camera_id: CameraId,
        status: HealthState,
        reason: ObservabilityReason = ObservabilityReason.NORMAL,
        *,
        effective_rate: float = 1.0,
        detail: str = "",
    ) -> None:
        """Record an observability transition and open or close a coverage gap.

        Every step down a degradation ladder lands here, and every one emits an
        event. No degradation is silent, ever (10_RELIABILITY §4.5).
        """
        now = self._clock.now()
        publish = False
        with self._lock:
            record = self._cameras.get(camera_id)
            if record is None:
                self.register_camera(camera_id)
                record = self._cameras[camera_id]

            previous = record.state
            if (
                previous.status is status
                and previous.reason is reason
                and abs(previous.effective_rate - effective_rate) < 1e-6
            ):
                return

            record.state = ObservabilityState(
                camera_id=camera_id,
                status=status,
                since=now,
                reason=reason,
                effective_rate=effective_rate,
                detail=detail,
            )
            publish = True

            if not status.observing and record.open_gap is None:
                record.open_gap = CoverageGap(
                    camera_id=camera_id, start=now, end=None, reason=reason, detail=detail
                )
                self._metrics.counter(
                    MetricName.BLIND_TRANSITIONS, camera_id=str(camera_id)
                ).increment()
            elif status.observing and record.open_gap is not None:
                closed = CoverageGap(
                    camera_id=camera_id,
                    start=record.open_gap.start,
                    end=now,
                    reason=record.open_gap.reason,
                    detail=record.open_gap.detail,
                )
                record.closed_gaps.append(closed)
                record.open_gap = None

        self._metrics.gauge(MetricName.EFFECTIVE_RATE, camera_id=str(camera_id)).set(
            effective_rate
        )
        self._metrics.gauge(MetricName.OBSERVABLE, camera_id=str(camera_id)).set(
            1.0 if status.observing else 0.0
        )
        if publish:
            self._bus.publish(
                CoverageChanged(
                    occurred_at=now,
                    partition_key=str(camera_id),
                    camera_id=camera_id,
                    status=status.value,
                    reason=reason.value,
                    effective_rate=effective_rate,
                )
            )

    def observability(self, camera_id: CameraId) -> ObservabilityState:
        with self._lock:
            record = self._cameras.get(camera_id)
            if record is None:
                return ObservabilityState(
                    camera_id=camera_id,
                    status=HealthState.FAILED,
                    since=self._clock.now(),
                    reason=ObservabilityReason.STREAM_DISCONNECTED,
                    effective_rate=0.0,
                    detail="camera not registered",
                )
            return record.state

    def coverage_gaps(self, camera_id: CameraId) -> tuple[CoverageGap, ...]:
        """Closed gaps plus the currently open one, if any."""
        with self._lock:
            record = self._cameras.get(camera_id)
            if record is None:
                return ()
            gaps = list(record.closed_gaps)
            if record.open_gap is not None:
                gaps.append(record.open_gap)
            return tuple(gaps)

    def site_health(self) -> SiteHealth:
        with self._lock:
            states = [r.state.status for r in self._cameras.values()]
        return SiteHealth(
            total_cameras=len(states),
            observing=sum(1 for s in states if s.observing),
            degraded=sum(1 for s in states if s is HealthState.DEGRADED),
            blind=sum(1 for s in states if s is HealthState.BLIND),
            failed=sum(1 for s in states if s is HealthState.FAILED),
        )

    # --- silent failure detection ------------------------------------------ #

    def observe_frame_digest(self, camera_id: CameraId, digest: int) -> None:
        """Frame-content liveness check (10_RELIABILITY §5.1).

        A frozen camera delivers frames at full rate and decodes perfectly; only
        content comparison reveals it. Raises a **suspicion**, never a verdict:
        it degrades coverage confidence and alerts, but never blinds a camera
        automatically, because a false positive that blinds a working camera is
        itself an outage.
        """
        suspect = False
        streak = 0
        with self._lock:
            record = self._cameras.get(camera_id)
            if record is None:
                return
            if record.last_frame_digest == digest:
                record.frozen_frame_streak += 1
            else:
                record.frozen_frame_streak = 0
                record.last_frame_digest = digest
            streak = record.frozen_frame_streak
            if streak == self._config.frozen_frame_threshold:
                suspect = True

        if suspect:
            self._metrics.counter(
                MetricName.SILENT_FAILURE_SUSPECTED,
                camera_id=str(camera_id),
                detector="frozen_frame",
            ).increment()
            self._bus.publish(
                SilentFailureSuspected(
                    occurred_at=self._clock.now(),
                    partition_key=str(camera_id),
                    camera_id=camera_id,
                    detector="frozen_frame",
                    evidence=f"{streak} identical consecutive frames",
                )
            )

    # --- probes ------------------------------------------------------------ #

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        for health in self.components():
            if health.state in (HealthState.STARTING, HealthState.FAILED):
                reasons.append(f"{health.component_id}:{health.state.value}")
        return (not reasons, tuple(reasons))

    def liveness(self) -> tuple[bool, tuple[str, ...]]:
        """Impaired components are reported, but the platform stays alive.

        Health is observational, never load-bearing.
        """
        impaired = [
            str(h.component_id) for h in self.components() if h.state is HealthState.FAILED
        ]
        return (True, tuple(impaired))

    def stale_after(self) -> Duration:
        return Duration.from_millis(self._config.report_timeout_ms)

    def now(self) -> Instant:
        return self._clock.now()
