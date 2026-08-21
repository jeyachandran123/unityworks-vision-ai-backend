"""M7 Object Registry — durable object identity.

> **Single responsibility:** *Decide what is the same thing over time, and be the
> only module allowed to decide it.*

This package holds the platform side of the registry. The split mirrors Flow 3's:

* ``engine`` and ``runtime`` are the **module** — they own objects and implement
  the documented M7 public API.
* ``lifecycle``, ``binding``, ``regions``, ``attributes``, ``partition`` are the
  machinery, each with one responsibility and testable alone.

Nothing here imports an adapter. The registry holds ``IdentityResolverPort`` and
``ObjectStorePort``; which implementation satisfies them is a composition fact.
"""

from .attributes import (
    AttributeRegistry,
    AttributeSchema,
    AttributeValueType,
    Cardinality,
    EvidenceRequirement,
    SchemaStatus,
    check_neutrality,
)
from .binding import BindingDecision, BindingPolicy, Candidate, TrackBinder
from .engine import REGISTRY_ENGINE_ID, ObjectRegistry, RegistryUpdate
from .lifecycle import (
    LifecyclePolicy,
    LifecycleTransition,
    LifecycleTrigger,
    ObjectLifecycleMachine,
    check_transition,
    is_legal,
)
from .partition import (
    ClassDistribution,
    ObjectRecord,
    PartitionStats,
    RegistryPartition,
)
from .regions import (
    RegionIndex,
    RegionMembership,
    RegionOccupancy,
    RegionTracker,
    RegionTransition,
)
from .runtime import REGISTRY_RUNTIME_ID, RegistryRuntime, RegistryRuntimeStats

__all__ = [
    "REGISTRY_ENGINE_ID",
    "REGISTRY_RUNTIME_ID",
    "AttributeRegistry",
    "AttributeSchema",
    "AttributeValueType",
    "BindingDecision",
    "BindingPolicy",
    "Candidate",
    "Cardinality",
    "ClassDistribution",
    "EvidenceRequirement",
    "LifecyclePolicy",
    "LifecycleTransition",
    "LifecycleTrigger",
    "ObjectLifecycleMachine",
    "ObjectRecord",
    "ObjectRegistry",
    "PartitionStats",
    "RegionIndex",
    "RegionMembership",
    "RegionOccupancy",
    "RegionTracker",
    "RegionTransition",
    "RegistryPartition",
    "RegistryRuntime",
    "RegistryRuntimeStats",
    "RegistryUpdate",
    "SchemaStatus",
    "TrackBinder",
    "check_neutrality",
    "check_transition",
    "is_legal",
]
