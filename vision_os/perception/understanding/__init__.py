"""M9 Vision Understanding Engine — pixels to schema-conformant claims.

> **Single responsibility:** *Ask a model what is true of these pixels, and return
> only what fits the declared schema.*

This package holds the platform side of understanding. The split mirrors Flow 5's:

* ``engine`` and ``runtime`` are the **module** — they own the understanding
  lifecycle and implement the documented M9 public API.
* ``routing``, ``validation``, ``cache`` are the machinery, each with one
  responsibility and testable alone.

Nothing here imports an adapter. M9 holds ``UnderstanderPort``,
``OutputCoercionPort`` and the ``PromptProvider`` seam; which implementation
satisfies each is a composition fact decided in ``understanding_bootstrap``.
"""

from .cache import (
    BatchGroup,
    CacheStats,
    ModelSemaphore,
    ResponseCache,
    cache_key,
    group_for_batching,
)
from .engine import (
    UNDERSTANDING_ENGINE_ID,
    UnderstandingEngine,
    blob_ref,
)
from .routing import (
    BoundUnderstander,
    CapabilityRouter,
    CircuitBreaker,
    RoutingDecision,
    RoutingPolicy,
)
from .runtime import (
    UNDERSTANDING_RUNTIME_ID,
    UnderstandingBatchReport,
    UnderstandingRuntime,
    UnderstandingRuntimeStats,
)
from .validation import AttributeValidator, ValidationOutcome, unsatisfied

__all__ = [
    "UNDERSTANDING_ENGINE_ID",
    "UNDERSTANDING_RUNTIME_ID",
    "AttributeValidator",
    "BatchGroup",
    "BoundUnderstander",
    "CacheStats",
    "CapabilityRouter",
    "CircuitBreaker",
    "ModelSemaphore",
    "ResponseCache",
    "RoutingDecision",
    "RoutingPolicy",
    "UnderstandingBatchReport",
    "UnderstandingEngine",
    "UnderstandingRuntime",
    "UnderstandingRuntimeStats",
    "ValidationOutcome",
    "blob_ref",
    "cache_key",
    "group_for_batching",
    "unsatisfied",
]
