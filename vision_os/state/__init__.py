"""M12 Vision State Manager — the single writer of visual truth.

> **Single responsibility:** *Be the single writer of visual truth, and never
> interpret it.*

Two modules, and the split is the architecture:

* ``projection`` is **pure** — ``(partition, observation) -> partition``, with no
  I/O, no clock and no events. That is what makes 07_STATE §9.1's *"fix, rebuild
  into a shadow projection, atomic swap"* lose no data.
* ``manager`` owns durability, partitioning, snapshots and recovery.

Nothing here imports a perception module. §M12: *"**No dependency on any
perception module** — it consumes observations and knows nothing of how they were
made, which is what allows the entire perception stack to be replaced beneath
it."*
"""

from .manager import (
    LOG_START,
    VISION_STATE_ID,
    RebuildHandle,
    SweepReport,
    VisionStateManager,
)
from .projection import ProjectionBounds, ProjectionOutcome, project

__all__ = [
    "LOG_START",
    "VISION_STATE_ID",
    "ProjectionBounds",
    "ProjectionOutcome",
    "RebuildHandle",
    "SweepReport",
    "VisionStateManager",
    "project",
]
