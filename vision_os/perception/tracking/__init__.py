"""M6 Tracking Engine — temporal continuity within one camera.

> **Single responsibility:** *Associate detections across time within one camera.
> Never assert durable identity — that is M7.*

This package holds the platform side of tracking and the shared tracking
toolkit that adapters build on. The split matters:

* ``engine``, ``manager``, ``runtime`` are the **platform**. They hold
  ``TrackerPort`` and never name a tracker.
* ``lifecycle``, ``association``, ``table`` are the **toolkit**. Adapters import
  them so that lifecycle correctness, deterministic assignment and bounded
  memory are guaranteed once rather than re-implemented — and re-implemented
  subtly wrong — in every adapter.

The dependency direction is one-way: adapters may import the toolkit; nothing
here may import an adapter.
"""

from .association import (
    AssociationPolicy,
    CostMatrixBuilder,
    GreedyAssociator,
    OptimalAssociator,
    iou,
)
from .engine import TRACKING_ENGINE_ID, TrackingEngine, TrackingOutcome
from .lifecycle import (
    LifecycleMachine,
    LifecyclePolicy,
    Transition,
    TransitionReason,
    check_transition,
    is_legal,
)
from .manager import TrackerBinding, TrackingManager
from .runtime import TRACKING_RUNTIME_ID, TrackingRuntime, TrackingRuntimeStats
from .table import TableStats, TrackRecord, TrackTable

__all__ = [
    "TRACKING_ENGINE_ID",
    "TRACKING_RUNTIME_ID",
    "AssociationPolicy",
    "CostMatrixBuilder",
    "GreedyAssociator",
    "LifecycleMachine",
    "LifecyclePolicy",
    "OptimalAssociator",
    "TableStats",
    "TrackRecord",
    "TrackTable",
    "TrackerBinding",
    "TrackingEngine",
    "TrackingManager",
    "TrackingOutcome",
    "TrackingRuntime",
    "TrackingRuntimeStats",
    "Transition",
    "TransitionReason",
    "check_transition",
    "iou",
    "is_legal",
]
