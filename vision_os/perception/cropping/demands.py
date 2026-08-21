"""The demand registry — what consumers asked for, and what we promised back.

> **Single responsibility:** *Hold registered demands and their standing. Trigger
> nothing, crop nothing.*

Two things happen here that happen nowhere else.

**Registration is the outermost ring of Semantic Ceiling enforcement.** A demand
naming an unregistered attribute is rejected *at registration*, with a pointer to
the registration process — not accepted and silently never served. That is
09_API_CONTRACTS §4.2, and it is the fourth gate after the attribute registry,
the prompt validator and the observation builder.

**Refusal is honest.** ``effective_freshness`` reports what the platform can
*actually* sustain, which may be longer than requested. Accepting a demand the
budget will not buy and quietly under-delivering is the failure this module
exists to prevent — and it is the single most common integration failure in
vision platforms.

The lifecycle is a closed machine, transcribed from the 09_API_CONTRACTS §4.4
diagram: *"A demand that quietly stops being satisfied is the failure mode this
lifecycle exists to prevent."*
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ...core.errors import DemandNotFoundError, DemandRejectedError
from ...core.model.demand import (
    AcknowledgementStatus,
    Demand,
    DemandAcknowledgement,
    DemandState,
    DemandStatus,
    UnsatisfiableReason,
)
from ...core.model.ids import AttributeKey, CameraId, ClassId, DemandId, new_ulid
from ...core.model.timebase import Duration, Instant

#: The closed transition table, from the 09_API_CONTRACTS section 4.4 diagram.
#:
#: Read the absences: ``expired``, ``revoked`` and ``rejected`` are terminal, and
#: nothing returns from them. A consumer whose demand expired registers a new one
#: — resurrecting it would leave the acknowledgement it was given describing a
#: budget that no longer exists.
_ALLOWED: frozenset[tuple[DemandStatus, DemandStatus]] = frozenset(
    {
        (DemandStatus.VALIDATED, DemandStatus.ACTIVE),
        (DemandStatus.VALIDATED, DemandStatus.REJECTED),
        (DemandStatus.ACTIVE, DemandStatus.ACTIVE),
        (DemandStatus.ACTIVE, DemandStatus.THROTTLED),
        (DemandStatus.ACTIVE, DemandStatus.UNSATISFIABLE),
        (DemandStatus.ACTIVE, DemandStatus.EXPIRED),
        (DemandStatus.ACTIVE, DemandStatus.REVOKED),
        (DemandStatus.THROTTLED, DemandStatus.ACTIVE),
        (DemandStatus.THROTTLED, DemandStatus.THROTTLED),
        (DemandStatus.THROTTLED, DemandStatus.EXPIRED),
        (DemandStatus.THROTTLED, DemandStatus.REVOKED),
        (DemandStatus.THROTTLED, DemandStatus.UNSATISFIABLE),
        (DemandStatus.UNSATISFIABLE, DemandStatus.ACTIVE),
        (DemandStatus.UNSATISFIABLE, DemandStatus.UNSATISFIABLE),
        (DemandStatus.UNSATISFIABLE, DemandStatus.EXPIRED),
        (DemandStatus.UNSATISFIABLE, DemandStatus.REVOKED),
    }
)


#: The tightest freshness the platform will promise, whatever a consumer asks.
#:
#: A demand asking for 10 ms freshness is asking for something no camera at 5 fps
#: can deliver. Promising it anyway would make the acknowledgement a lie, which
#: is precisely what ``effective_freshness`` exists to prevent.
MIN_EFFECTIVE_FRESHNESS = Duration.from_millis(1_000)


def is_legal(previous: DemandStatus, current: DemandStatus) -> bool:
    return (previous, current) in _ALLOWED


def check_transition(previous: DemandStatus, current: DemandStatus) -> None:
    """Raise unless the edge is legal.

    Raises:
        DemandRejectedError: for an edge outside the table. The lifecycle is
            closed, so an illegal edge means a caller is constructing state
            incorrectly rather than encountering bad input.
    """
    if not is_legal(previous, current):
        raise DemandRejectedError(
            f"illegal demand transition {previous.value} -> {current.value}; the "
            f"lifecycle is a closed machine (09_API_CONTRACTS section 4.4)",
            previous=previous.value,
            current=current.value,
        )


@dataclass(frozen=True, slots=True)
class CapabilityView:
    """What the platform can currently produce, for honest refusal.

    Injected rather than discovered, so the registry stays a pure holder of
    demands and the capability question has one owner.
    """

    registered_attributes: frozenset[AttributeKey] = frozenset()
    producible_attributes: frozenset[AttributeKey] = frozenset()
    """Attributes some loaded model can actually produce. Empty until Flow 6
    binds an understander — which is why every demand today reports
    ``NO_CAPABLE_MODEL`` for its attributes and says so at registration rather
    than leaving the consumer waiting."""

    producible_classes: frozenset[ClassId] = frozenset()
    observed_cameras: frozenset[CameraId] = frozenset()

    def classifies(self, attribute: AttributeKey) -> UnsatisfiableReason | None:
        """Why this attribute cannot be delivered, or ``None`` if it can."""
        if attribute not in self.registered_attributes:
            return UnsatisfiableReason.ATTRIBUTE_NOT_REGISTERED
        if attribute not in self.producible_attributes:
            return UnsatisfiableReason.NO_CAPABLE_MODEL
        return None


@dataclass(frozen=True, slots=True)
class RegistryStats:
    total: int
    by_status: dict[DemandStatus, int]

    @property
    def serving(self) -> int:
        return sum(
            count for status, count in self.by_status.items() if status.is_serving
        )


class DemandRegistry:
    """Registered demands and their standing. Single-writer.

    Owned by M8 (§M8 Public API exposes ``register_demand``/``revoke_demand``);
    the Observation API will forward to it in Flow 8. That is the only reading
    under which §M8's API, §M8's dependency list and 09_API_CONTRACTS §4.1 all
    hold at once.
    """

    __slots__ = ("_capabilities", "_demands", "_id_factory", "_min_freshness")

    def __init__(
        self,
        *,
        capabilities: CapabilityView | None = None,
        min_effective_freshness: Duration = MIN_EFFECTIVE_FRESHNESS,
        id_factory: Callable[[Instant], DemandId] | None = None,
    ) -> None:
        self._demands: dict[DemandId, DemandState] = {}
        self._capabilities = capabilities or CapabilityView()
        self._min_freshness = min_effective_freshness
        self._id_factory = id_factory or (
            lambda now: DemandId(new_ulid(now_ms=now.ns // 1_000_000))
        )

    # --- registration ---------------------------------------------------------- #

    def register(
        self, demand: Demand, *, now: Instant, sustainable_freshness: Duration | None = None
    ) -> DemandAcknowledgement:
        """Validate, classify and admit a demand.

        Args:
            sustainable_freshness: What the budget can actually buy. When it is
                longer than the demand's request, the acknowledgement says so —
                the platform telling the truth about its limits rather than
                accepting and under-delivering.

        Raises:
            DemandRejectedError: **no** requested attribute is registered. A
                demand that can never be served is refused outright rather than
                admitted into a registry it will sit in forever.
        """
        satisfiable: list[AttributeKey] = []
        unsatisfiable: list[tuple[AttributeKey, UnsatisfiableReason]] = []

        for attribute in demand.required_attributes:
            reason = self._capabilities.classifies(attribute)
            if reason is None:
                satisfiable.append(attribute)
            else:
                unsatisfiable.append((attribute, reason))

        unregistered = [
            key
            for key, reason in unsatisfiable
            if reason is UnsatisfiableReason.ATTRIBUTE_NOT_REGISTERED
        ]
        if unregistered and len(unregistered) == len(demand.required_attributes):
            raise DemandRejectedError(
                f"demand names only unregistered attributes "
                f"({', '.join(str(k) for k in unregistered)}). Attributes must pass "
                f"the neutrality gate before they can be demanded (02_VOM section "
                f"9.1); register them first.",
                demand_id=str(demand.demand_id),
                attributes=tuple(str(k) for k in unregistered),
            )

        effective = self._effective_freshness(demand.freshness, sustainable_freshness)
        status = (
            AcknowledgementStatus.ACCEPTED
            if not unsatisfiable
            else AcknowledgementStatus.PARTIALLY_ACCEPTED
        )

        demand_id = demand.demand_id or self._id_factory(now)
        stored = replace(demand, demand_id=demand_id, registered_at=now)

        acknowledgement = DemandAcknowledgement(
            demand_id=demand_id,
            status=status,
            satisfiable=tuple(satisfiable),
            unsatisfiable=tuple(unsatisfiable),
            effective_freshness=effective,
            detail=(
                ""
                if effective.ns <= demand.freshness.ns
                else (
                    f"requested {demand.freshness.millis:.0f}ms freshness; the "
                    f"budget sustains {effective.millis:.0f}ms"
                )
            ),
        )

        # The diagram admits only ``validated -> active``; ``unsatisfiable`` is
        # reached *from* active. So a demand whose attributes are all currently
        # unproducible is activated and then immediately marked — two legal
        # edges rather than one invented one. It is still admitted, because
        # capability can return (a model reloads, a camera recovers) and the
        # lifecycle has a state for exactly that.
        check_transition(DemandStatus.VALIDATED, DemandStatus.ACTIVE)
        self._demands[demand_id] = DemandState(
            demand=stored,
            status=DemandStatus.ACTIVE,
            acknowledgement=acknowledgement,
            activated_at=now,
            effective_freshness=effective,
        )
        if not satisfiable:
            self.mark_unsatisfiable(
                demand_id,
                detail="no loaded model can currently produce any requested attribute",
            )
        return acknowledgement

    def revoke(self, demand_id: DemandId) -> None:
        """Withdraw a demand.

        Raises:
            DemandNotFoundError: unknown id.
        """
        state = self._require(demand_id)
        if state.status.is_terminal:
            return
        check_transition(state.status, DemandStatus.REVOKED)
        self._demands[demand_id] = replace(state, status=DemandStatus.REVOKED)

    # --- lifecycle -------------------------------------------------------------- #

    def throttle(self, demand_id: DemandId, effective: Duration) -> DemandState:
        """Reduce a demand's freshness under budget pressure.

        The consumer is *told* rather than left to notice. A demand that quietly
        stops being satisfied is the failure the lifecycle exists to prevent.
        """
        state = self._require(demand_id)
        check_transition(state.status, DemandStatus.THROTTLED)
        updated = replace(
            state, status=DemandStatus.THROTTLED, effective_freshness=effective
        )
        self._demands[demand_id] = updated
        return updated

    def restore(self, demand_id: DemandId) -> DemandState:
        """Return a throttled or unsatisfiable demand to active service."""
        state = self._require(demand_id)
        check_transition(state.status, DemandStatus.ACTIVE)
        updated = replace(
            state,
            status=DemandStatus.ACTIVE,
            effective_freshness=state.acknowledgement.effective_freshness,
        )
        self._demands[demand_id] = updated
        return updated

    def mark_unsatisfiable(self, demand_id: DemandId, detail: str = "") -> DemandState:
        """Capability was lost — a model evicted, a camera blind."""
        state = self._require(demand_id)
        check_transition(state.status, DemandStatus.UNSATISFIABLE)
        updated = replace(
            state,
            status=DemandStatus.UNSATISFIABLE,
            acknowledgement=replace(state.acknowledgement, detail=detail),
        )
        self._demands[demand_id] = updated
        return updated

    def expire_due(self, now: Instant) -> tuple[DemandId, ...]:
        """Expire demands past ``expires_at``. Returns what changed."""
        expired: list[DemandId] = []
        for demand_id in sorted(self._demands):
            state = self._demands[demand_id]
            if state.status.is_terminal or not state.demand.is_expired(now):
                continue
            check_transition(state.status, DemandStatus.EXPIRED)
            self._demands[demand_id] = replace(state, status=DemandStatus.EXPIRED)
            expired.append(demand_id)
        return tuple(expired)

    def record_served(self, demand_id: DemandId, now: Instant) -> None:
        """Note that work was done for this demand, for cost attribution."""
        state = self._demands.get(demand_id)
        if state is None:
            return
        self._demands[demand_id] = replace(
            state, last_served=now, calls_served=state.calls_served + 1
        )

    # --- reads ------------------------------------------------------------------ #

    def get(self, demand_id: DemandId) -> DemandState | None:
        return self._demands.get(demand_id)

    def active(self) -> tuple[DemandState, ...]:
        """Demands currently being served, in stable id order.

        Ordering is deterministic rather than dict-insertion-dependent, because
        trigger evaluation walks this list and a reordering would change which
        demand wins a tie (V13).
        """
        return tuple(
            self._demands[key]
            for key in sorted(self._demands)
            if self._demands[key].is_serving
        )

    def all(self) -> tuple[DemandState, ...]:
        return tuple(self._demands[key] for key in sorted(self._demands))

    def matching(
        self, *, camera_id: CameraId, class_id: ClassId, region_ids: Sequence = ()
    ) -> tuple[DemandState, ...]:
        """Serving demands whose scope and filter cover this object."""
        return tuple(
            state
            for state in self.active()
            if state.demand.scope.covers_camera(camera_id)
            and state.demand.scope.covers_region(region_ids)
            and state.demand.subject_filter.matches_class(class_id)
        )

    def required_attributes(
        self, *, camera_id: CameraId, class_id: ClassId, region_ids: Sequence = ()
    ) -> dict[AttributeKey, tuple[Duration, str, tuple[DemandId, ...]]]:
        """Attributes wanted for this object, with the **strictest** freshness.

        Two demands can want the same attribute at different rates. Taking the
        minimum is the only correct answer: satisfying the looser one would leave
        the stricter consumer silently under-served, which is exactly what
        ``effective_freshness`` exists to prevent elsewhere.
        """
        wanted: dict[AttributeKey, tuple[Duration, str, tuple[DemandId, ...]]] = {}
        for state in self.matching(
            camera_id=camera_id, class_id=class_id, region_ids=region_ids
        ):
            for attribute in state.acknowledgement.satisfiable:
                freshness = state.effective_freshness or state.demand.freshness
                existing = wanted.get(attribute)
                if existing is None:
                    wanted[attribute] = (
                        freshness,
                        state.demand.priority_class,
                        (state.demand.demand_id,),
                    )
                    continue
                current_freshness, priority, ids = existing
                wanted[attribute] = (
                    Duration(min(current_freshness.ns, freshness.ns)),
                    priority or state.demand.priority_class,
                    (*ids, state.demand.demand_id),
                )
        return wanted

    def stats(self) -> RegistryStats:
        by_status: dict[DemandStatus, int] = {}
        for state in self._demands.values():
            by_status[state.status] = by_status.get(state.status, 0) + 1
        return RegistryStats(total=len(self._demands), by_status=by_status)

    def __len__(self) -> int:
        return len(self._demands)

    def __contains__(self, demand_id: object) -> bool:
        return demand_id in self._demands

    # --- internals --------------------------------------------------------------- #

    def _require(self, demand_id: DemandId) -> DemandState:
        state = self._demands.get(demand_id)
        if state is None:
            raise DemandNotFoundError(
                f"demand '{demand_id}' is not registered", demand_id=str(demand_id)
            )
        return state

    def _effective_freshness(
        self, requested: Duration, sustainable: Duration | None
    ) -> Duration:
        floor = max(self._min_freshness.ns, requested.ns)
        if sustainable is None:
            return Duration(floor)
        return Duration(max(floor, sustainable.ns))
