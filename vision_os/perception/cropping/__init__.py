"""M8 Crop Manager — the platform's attention mechanism.

> **Single responsibility:** *Choose what to look at closely, and produce a crop
> worth looking at.*

This package holds the platform side of attention. The split mirrors Flow 4's:

* ``engine`` and ``runtime`` are the **module** — they own crop lifecycle and
  implement the documented M8 public API.
* ``demands``, ``budget``, ``gate``, ``state`` are the machinery, each with one
  responsibility and testable alone.

Nothing here imports an adapter. M8 holds ``TriggerPolicyPort``,
``QualityEstimatorPort``, ``CropStrategyPort`` and ``CropExtractorPort``; which
implementation satisfies each is a composition fact decided in
``cropping_bootstrap``.
"""

from .budget import (
    BudgetStatus,
    CacheStats,
    CropDeduplicationCache,
    PriorityQueue,
    UnderstandingBudget,
)
from .demands import (
    CapabilityView,
    DemandRegistry,
    RegistryStats,
    check_transition,
    is_legal,
)
from .engine import CROP_MANAGER_ID, CropManager, FrameContext
from .gate import GateThresholds, QualityGate
from .runtime import CROP_RUNTIME_ID, CropRuntime, CropRuntimeStats
from .state import (
    CameraTriggerPartition,
    GateRejectionWindow,
    ObjectTriggerState,
    TriggerStateStore,
)

__all__ = [
    "CROP_MANAGER_ID",
    "CROP_RUNTIME_ID",
    "BudgetStatus",
    "CacheStats",
    "CameraTriggerPartition",
    "CapabilityView",
    "CropDeduplicationCache",
    "CropManager",
    "CropRuntime",
    "CropRuntimeStats",
    "DemandRegistry",
    "FrameContext",
    "GateRejectionWindow",
    "GateThresholds",
    "ObjectTriggerState",
    "PriorityQueue",
    "QualityGate",
    "RegistryStats",
    "TriggerStateStore",
    "UnderstandingBudget",
    "check_transition",
    "is_legal",
]
