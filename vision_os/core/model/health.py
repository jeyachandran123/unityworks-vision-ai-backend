"""Health and observability state (05_MODULES_PLATFORM_KERNEL M20, 07_STATE §7).

``BLIND`` is the state that justifies the Health Monitor's existence. A camera
that is streaming, decoding, and detecting nothing because a delivery truck
parked in front of it is *healthy* by every naive metric and *useless* in fact.
Distinguishing these is what stops a consumer concluding "the area was clear".

**Flow 1 boundary.** This module produces observability *state and events*. The
conversion of that state into ``coverage``-type Observations is the Observation
Builder's responsibility (Flow 6) and is deliberately absent here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .ids import CameraId, ModuleId, RegionId
from .timebase import Instant


class HealthState(enum.Enum):
    """Component and camera health (05_KERNEL M20)."""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    """Working with reduced capability. Observations are valid but thinner."""

    BLIND = "blind"
    """Connected but not perceiving. Absence of observations means nothing."""

    FAILED = "failed"
    DRAINING = "draining"

    @property
    def observing(self) -> bool:
        return self in (HealthState.HEALTHY, HealthState.DEGRADED)


class ObservabilityReason(enum.Enum):
    """Why observability is reduced (07_STATE §7.2)."""

    NORMAL = "normal"
    STREAM_DISCONNECTED = "stream_disconnected"
    DECODE_FAILING = "decode_failing"
    PRIVACY_MASK_FAILED = "privacy_mask_failed"
    SCHEDULER_SHEDDING = "scheduler_shedding"
    SCENE_OBSCURED = "scene_obscured"
    CALIBRATION_SUSPECT = "calibration_suspect"
    STARTING_UP = "starting_up"
    DRAINING = "draining"
    # Reasons owned by later flows (DETECTOR_UNAVAILABLE, UNDERSTANDING_BUDGET_
    # EXHAUSTED, MODEL_CAPABILITY_GAP, PARTITION_UNAVAILABLE) are intentionally
    # absent until those flows exist.


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """A component's self-report. Components report; the kernel decides."""

    component_id: ModuleId
    state: HealthState
    reported_at: Instant
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservabilityState:
    """Whether a camera can currently see, and why not (07_STATE §7.2)."""

    camera_id: CameraId
    status: HealthState
    since: Instant
    reason: ObservabilityReason = ObservabilityReason.NORMAL
    effective_rate: float = 1.0
    """Frames actually processed / frames expected. 1.0 = full observability."""

    regions_affected: tuple[RegionId, ...] = ()
    detail: str = ""

    @property
    def observing(self) -> bool:
        return self.status.observing


@dataclass(frozen=True, slots=True)
class CoverageGap:
    """A recorded interval of reduced or absent observability."""

    camera_id: CameraId
    start: Instant
    end: Instant | None
    reason: ObservabilityReason
    detail: str = ""

    @property
    def closed(self) -> bool:
        return self.end is not None
