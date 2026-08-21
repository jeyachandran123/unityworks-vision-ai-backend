"""Scheduling ports — P5 ``AdmissionPolicyPort``, P6 ``ChangeDetectorPort``.

Owner: M3 Frame Scheduler.

The scheduler is the platform's economic regulator and the primary implementation
of invariant V7. At one camera it is nearly trivial. At 100 cameras it is the
difference between a system that works and one that collapses under its own input
rate.

**Every drop is attributed.** ``DropReason`` is not optional and has no
``UNKNOWN`` member: a discard the platform cannot explain is a V8 violation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model.camera import PipelineProfile, SourceSemantics
from ..model.frame import FrameDimensions
from ..model.ids import CameraId
from ..model.timebase import Instant


class DropReason(enum.Enum):
    """Why a frame was not processed (03_MODULES M3).

    ``CADENCE`` is by design and dominates in a healthy system. The others are
    signals: sustained ``BUDGET_EXHAUSTED`` or ``QUEUE_FULL`` means the platform
    is thinning perception and must say so.
    """

    CADENCE = "cadence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TENANT_QUOTA = "tenant_quota"
    QUEUE_FULL = "queue_full"
    DUPLICATE = "duplicate"
    QUALITY_REJECT = "quality_reject"
    DEADLINE_EXPIRED = "deadline_expired"

    @property
    def indicates_pressure(self) -> bool:
        """Whether this reason means the platform is losing observability."""
        return self in (
            DropReason.BUDGET_EXHAUSTED,
            DropReason.TENANT_QUOTA,
            DropReason.QUEUE_FULL,
            DropReason.DEADLINE_EXPIRED,
        )


@dataclass(frozen=True, slots=True)
class Fidelity:
    """Processing fidelity selected for an admitted frame."""

    inference_width: int
    inference_height: int
    tier: str = "primary"


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    """Everything a policy may consider. Deliberately closed.

    Note what is absent: no business priority, no notion of which camera
    "matters". ``priority_class`` on the profile is an opaque label the policy
    may order by and may not interpret (invariant V1/V2).
    """

    camera_id: CameraId
    profile: PipelineProfile
    semantics: SourceSemantics
    monotonic_now: Instant
    last_admitted_monotonic: Instant | None
    in_flight: int
    budget_pressure: float
    """0.0 = idle, 1.0 = saturated."""

    queue_full: bool = False


@dataclass(frozen=True, slots=True)
class AdmissionVerdict:
    admit: bool
    reason: DropReason | None = None
    fidelity: Fidelity | None = None

    def __post_init__(self) -> None:
        if self.admit and self.reason is not None:
            raise ValueError("an admitted frame cannot carry a drop reason")
        if not self.admit and self.reason is None:
            raise ValueError("every drop must carry an attributed reason (V8)")


@runtime_checkable
class AdmissionPolicyPort(Protocol):
    """P5 — decide whether a frame is processed, and at what fidelity.

    The decision runs on every decoded frame from every camera (~3000/s at 100
    cameras), so implementations must be allocation-free and lock-free on the
    common path (03_MODULES M3 performance).

    Implementations: fixed cadence, weighted fair share, priority classes,
    deadline-aware, activity-adaptive.
    """

    def evaluate(self, context: AdmissionContext) -> AdmissionVerdict: ...


@dataclass(frozen=True, slots=True)
class ChangeVerdict:
    changed: bool
    score: float = 1.0
    """Normalized magnitude of visual change since the last observed frame."""


@runtime_checkable
class ChangeDetectorPort(Protocol):
    """P6 — suppress frames that carry no new information.

    The highest-value extension in the scheduler: in most real deployments the
    majority of frames contain nothing new, and suppressing them here is the
    cheapest possible saving (03_MODULES M3 extension points).

    Implementations: frame differencing, codec motion vectors (nearly free — the
    encoder already computed them), background subtraction, learned change
    detection.
    """

    def observe(
        self, camera_id: CameraId, view: memoryview, dimensions: FrameDimensions
    ) -> ChangeVerdict: ...

    def forget(self, camera_id: CameraId) -> None:
        """Drop per-camera state, e.g. on epoch advance or detach."""
        ...
