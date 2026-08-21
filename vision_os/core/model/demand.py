"""Demand contracts (09_API_CONTRACTS section 4).

> *"I need headwear_present on person in region Z3, refreshed every 30s"* ✅
> *"Alert me when a customer waits too long"* ❌

A demand says **what** is needed, never **why**. That single restriction is what
lets one platform serve a hospital and a warehouse from one codebase: the
platform schedules work to satisfy demands within budget, and never learns what
any demand is for (`00_CHARTER` §2).

Two mechanisms here do real work:

**Registration is a gate.** A demand referencing an unregistered attribute is
rejected at registration with a pointer to the registration process — *"the
fourth and outermost ring of Semantic Ceiling enforcement"* (02_VOM §9).

**``effective_freshness`` is where the platform tells the truth about its
limits.** A consumer asking for 1-second freshness on 200 objects across 40
cameras is asking for something the budget will not buy. Rather than accepting
and silently under-delivering, the platform answers with what it can actually
sustain, and the consumer decides whether to narrow scope, raise budget, or
accept.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .ids import (
    AttributeKey,
    CameraId,
    ClassId,
    DemandId,
    RegionId,
    SiteId,
    SubscriberId,
)
from .timebase import Duration, Instant


class TriggerHint(enum.Enum):
    """What a consumer suggests should prompt a re-look.

    A **hint**, not a command: the platform decides, because it alone knows the
    budget. A demand that could dictate triggering could exhaust the node.
    """

    ON_FIRST_SIGHT = "on_first_sight"
    ON_CHANGE = "on_change"
    ON_REGION_ENTRY = "on_region_entry"
    PERIODIC = "periodic"


class DemandStatus(enum.Enum):
    """The lifecycle of 09_API_CONTRACTS section 4.4. Closed set.

    > *"Every transition notifies the subscriber. A demand that quietly stops
    > being satisfied is the failure mode this lifecycle exists to prevent."*
    """

    VALIDATED = "validated"
    """Passed schema, neutrality and budget checks; not yet accepted."""

    ACTIVE = "active"
    THROTTLED = "throttled"
    """Budget pressure — reduced freshness, consumer notified. Still served, at a
    rate the platform states rather than one it silently degrades to."""

    UNSATISFIABLE = "unsatisfiable"
    """Capability lost: a model was evicted, or a camera went blind. Recoverable."""

    EXPIRED = "expired"
    REVOKED = "revoked"
    REJECTED = "rejected"

    @property
    def is_serving(self) -> bool:
        """Whether the platform is currently doing work for this demand."""
        return self in (DemandStatus.ACTIVE, DemandStatus.THROTTLED)

    @property
    def is_terminal(self) -> bool:
        return self in (
            DemandStatus.EXPIRED,
            DemandStatus.REVOKED,
            DemandStatus.REJECTED,
        )


class AcknowledgementStatus(enum.Enum):
    ACCEPTED = "accepted"
    PARTIALLY_ACCEPTED = "partially_accepted"
    REJECTED = "rejected"


class UnsatisfiableReason(enum.Enum):
    """Why an attribute cannot be delivered. Reported at registration.

    *"The consumer learns this at registration, in seconds, instead of
    discovering it as a permanent absence of data weeks later. This is V8 applied
    at integration time, and it eliminates the single most common integration
    failure in vision platforms."*
    """

    ATTRIBUTE_NOT_REGISTERED = "attribute_not_registered"
    NO_CAPABLE_MODEL = "no_capable_model"
    CLASS_NOT_PRODUCIBLE = "class_not_producible"
    """No detector at this site produces the subject class."""

    SCOPE_NOT_OBSERVABLE = "scope_not_observable"
    """No camera covers the requested scope."""

    BUDGET_INSUFFICIENT = "budget_insufficient"


@dataclass(frozen=True, slots=True)
class DemandScope:
    """Where a demand applies. Ids only — never what any of them mean (V2)."""

    site_ids: tuple[SiteId, ...] = ()
    camera_ids: tuple[CameraId, ...] = ()
    region_ids: tuple[RegionId, ...] = ()

    def covers_camera(self, camera_id: CameraId) -> bool:
        """An empty camera list means "every camera in scope", not "none".

        The degenerate case is the common one: most demands are site-wide.
        """
        return not self.camera_ids or camera_id in self.camera_ids

    def covers_region(self, region_ids: Sequence[RegionId]) -> bool:
        if not self.region_ids:
            return True
        return any(r in self.region_ids for r in region_ids)


@dataclass(frozen=True, slots=True)
class SubjectFilter:
    """Which objects a demand is about."""

    class_ids: tuple[ClassId, ...] = ()
    lifecycle: tuple[str, ...] = ()
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.min_confidence is not None and not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0,1]")

    def matches_class(self, class_id: ClassId) -> bool:
        """Hierarchical: a demand for ``person`` covers ``person.child``."""
        if not self.class_ids:
            return True
        return any(
            class_id == allowed or class_id.startswith(f"{allowed}.")
            for allowed in self.class_ids
        )


@dataclass(frozen=True, slots=True)
class DemandBudget:
    """A consumer's own ceiling, distinct from the platform's.

    A demand may cap its cost; it may not raise the platform's. The node budget
    is a property of the hardware and is never negotiable by a consumer.
    """

    max_calls_per_hour: int | None = None
    max_cost_per_hour: float | None = None

    def __post_init__(self) -> None:
        if self.max_calls_per_hour is not None and self.max_calls_per_hour < 0:
            raise ValueError("max_calls_per_hour must be non-negative")
        if self.max_cost_per_hour is not None and self.max_cost_per_hour < 0:
            raise ValueError("max_cost_per_hour must be non-negative")


@dataclass(frozen=True, slots=True)
class Demand:
    """A consumer's declared need (09_API_CONTRACTS section 4.1).

    Note what cannot be expressed here: no rule, no threshold, no conclusion, no
    justification. ``priority_class`` is an opaque label the platform orders by
    and never interprets — the pressure to make it meaningful is exactly the
    pressure V1 exists to resist.
    """

    demand_id: DemandId
    subscriber: SubscriberId
    scope: DemandScope
    subject_filter: SubjectFilter
    required_attributes: tuple[AttributeKey, ...]
    freshness: Duration
    """Maximum acceptable staleness. The platform answers with what it can
    actually sustain, which may be longer."""

    trigger_hints: tuple[TriggerHint, ...] = ()
    priority_class: str = ""
    """**Opaque.** The platform orders by it; the reason a class exists lives
    with the consumer (V1/V2)."""

    budget: DemandBudget = DemandBudget()
    expires_at: Instant | None = None
    registered_at: Instant = Instant(0)
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.required_attributes:
            raise ValueError(
                "a demand must name at least one attribute; a demand for nothing "
                "schedules nothing and would sit in the registry forever"
            )
        if self.freshness.ns <= 0:
            raise ValueError("freshness must be positive")

    def is_expired(self, now: Instant) -> bool:
        return self.expires_at is not None and now.ns >= self.expires_at.ns


@dataclass(frozen=True, slots=True)
class DemandAcknowledgement:
    """What the platform promises back (09_API_CONTRACTS section 4.3).

    ``effective_freshness`` is the honest half. Accepting a demand the budget
    cannot serve and quietly under-delivering is the failure this type exists to
    prevent.
    """

    demand_id: DemandId
    status: AcknowledgementStatus
    satisfiable: tuple[AttributeKey, ...] = ()
    unsatisfiable: tuple[tuple[AttributeKey, UnsatisfiableReason], ...] = ()
    effective_freshness: Duration = Duration(0)
    """What the platform can **actually** deliver, which may be longer than
    requested."""

    estimated_calls_per_hour: float = 0.0
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.status is not AcknowledgementStatus.REJECTED

    @property
    def fully_satisfiable(self) -> bool:
        return self.status is AcknowledgementStatus.ACCEPTED and not self.unsatisfiable


@dataclass(frozen=True, slots=True)
class DemandState:
    """A registered demand and its current standing."""

    demand: Demand
    status: DemandStatus
    acknowledgement: DemandAcknowledgement
    activated_at: Instant
    last_served: Instant | None = None
    calls_served: int = 0
    effective_freshness: Duration = Duration(0)
    """Recomputed under pressure. When it drifts from the demand's request, the
    subscriber is notified rather than left to infer it."""

    @property
    def is_serving(self) -> bool:
        return self.status.is_serving

    def satisfies(self, attribute: AttributeKey) -> bool:
        return attribute in self.acknowledgement.satisfiable
