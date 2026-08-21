"""Capability routing — which understander answers which question.

> **Single responsibility:** *Choose a capable model for a requested attribute
> set. Invoke nothing.*

This module is why 11_PERFORMANCE §7's migration is a configuration change:

> *Use the VLM to discover and validate an attribute, use its evidence to train a
> specialized head, then move that attribute to the head in production, with zero
> consumer impact.*

Routing is by **declared capability**, never by name. An adapter publishes
`producible_attributes` and a cost class; the router picks the cheapest capable
one. Swapping a 7B VLM for a 2 MB classifier on one attribute is then a change to
which adapters are bound — the engine's code does not mention either.

**Nothing here decides whether the call is worth making.** That is M8's, and
04_MODULES §M9 lists it explicitly as a non-responsibility. The router answers
*"who can"* and `estimate_cost` answers *"how much"*; the decision belongs
upstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.model.ids import AttributeKey, ModelId
from ...core.ports.understanding import UnderstanderCapabilities, UnderstanderPort


@dataclass(frozen=True, slots=True)
class BoundUnderstander:
    """One understander, with everything routing needs to reason about it.

    Capabilities are captured at binding rather than queried per request: a
    capability that changed under a running route would make two identical
    requests take different paths, which V13 forbids and which no consumer could
    explain.
    """

    adapter: UnderstanderPort
    capabilities: UnderstanderCapabilities
    is_fallback: bool = False
    """Whether this understander is reachable only through a fallback chain.

    A fallback is a *different accuracy profile* (10_RELIABILITY §4.3 step 4), so
    it must never win a primary route by being cheaper — otherwise the platform
    quietly runs on its worst model forever, which §7.2 calls out as one of the
    silent failures."""

    @property
    def adapter_id(self) -> str:
        return self.adapter.adapter_id

    @property
    def model_id(self) -> ModelId:
        return self.capabilities.model_id

    @property
    def cost_class(self) -> float:
        return self.capabilities.cost_class


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Who will answer, for what, and what is left uncovered.

    ``uncovered`` is the field invariant V8 rests on. A route that quietly
    dropped the attributes nobody can produce would leave the consumer waiting
    for data that will never arrive; naming them lets M8 publish a capability
    gap.
    """

    selected: BoundUnderstander | None = None
    covered: tuple[AttributeKey, ...] = ()
    uncovered: tuple[AttributeKey, ...] = ()
    fallbacks: tuple[BoundUnderstander, ...] = ()
    """The chain to try, in order, if the selection fails. 10_RELIABILITY §7.2:
    *"The last link is always explicit unavailability, never a guess."* — so this
    tuple terminating is itself the last link."""

    considered: int = 0
    reason: str = ""

    @property
    def has_route(self) -> bool:
        return self.selected is not None

    @property
    def fully_covered(self) -> bool:
        return self.has_route and not self.uncovered


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """How to choose among capable understanders.

    Cost-first by default, which is the policy 11_PERFORMANCE §7 is written
    around: when a specialized head and a VLM can both produce an attribute, the
    head should win, every time, without anyone editing code.
    """

    prefer_local: bool = True
    """`12_SECURITY`: a site with a data-residency policy must not silently ship
    imagery to a remote endpoint because it was marginally cheaper."""

    prefer_deterministic: bool = False
    """Breaks ties toward reproducibility. Off by default because it would
    otherwise override cost, and cost is the lever that matters at scale."""

    prefer_coverage: bool = True
    """Prefer an understander covering more of the request over a cheaper one
    covering less. Two calls are almost always worse than one: 04_MODULES §M9
    puts attribute batching in a single prompt at a *"3-5x saving"*."""

    max_fallback_depth: int = 2

    def __post_init__(self) -> None:
        if self.max_fallback_depth < 0:
            raise ValueError("max_fallback_depth must be non-negative")


class CapabilityRouter:
    """Routes an attribute set to a capable understander.

    Holds bound adapters and nothing else — no model names, no vendor knowledge,
    no per-model special cases. Adding a model is binding an adapter; the router
    is unchanged, which is the whole point of P15.
    """

    __slots__ = ("_bound", "_policy")

    def __init__(
        self,
        understanders: Sequence[BoundUnderstander] = (),
        policy: RoutingPolicy | None = None,
    ) -> None:
        self._policy = policy or RoutingPolicy()
        self._bound: list[BoundUnderstander] = list(understanders)
        self._check_duplicates()

    def bind(self, understander: BoundUnderstander) -> None:
        self._bound.append(understander)
        self._check_duplicates()

    def _check_duplicates(self) -> None:
        seen: set[str] = set()
        for bound in self._bound:
            if bound.adapter_id in seen:
                raise ValueError(
                    f"understander '{bound.adapter_id}' is bound twice; a "
                    f"duplicate makes routing order depend on binding order, "
                    f"which is not reproducible"
                )
            seen.add(bound.adapter_id)

    # --- the public question ------------------------------------------------- #

    def route(self, attributes: Sequence[AttributeKey]) -> RoutingDecision:
        """Pick an understander for this attribute set.

        Deterministic: identical bindings and an identical request always produce
        an identical decision, including the fallback ordering. Ties break on
        ``adapter_id`` so the answer never depends on binding order (V13).
        """
        if not attributes:
            return RoutingDecision(reason="no attributes requested")

        requested = tuple(attributes)
        candidates = [
            bound
            for bound in self._bound
            if not bound.is_fallback and bound.capabilities.covers(requested)
        ]
        if not candidates:
            producible = self.producible_attributes()
            return RoutingDecision(
                uncovered=tuple(k for k in requested if k not in producible),
                covered=tuple(k for k in requested if k in producible),
                considered=len(self._bound),
                reason=(
                    "no bound understander declares any requested attribute"
                    if not producible
                    else "no non-fallback understander covers this request"
                ),
            )

        ranked = sorted(candidates, key=lambda b: self._rank(b, requested))
        selected = ranked[0]
        covered = selected.capabilities.covers(requested)
        uncovered = tuple(k for k in requested if k not in set(covered))

        return RoutingDecision(
            selected=selected,
            covered=covered,
            uncovered=uncovered,
            fallbacks=self._chain(selected, covered),
            considered=len(candidates),
            reason=f"selected on cost class {selected.cost_class:g}",
        )

    def _rank(
        self, bound: BoundUnderstander, requested: Sequence[AttributeKey]
    ) -> tuple:
        """The ordering key. Lower sorts first.

        Every component is a *declared* property of the adapter. None of them is
        a model name, and none is a hard-coded preference for a particular
        vendor — which is what keeps V3 true as models churn.
        """
        covered = len(bound.capabilities.covers(requested))
        missing = len(requested) - covered
        return (
            missing if self._policy.prefer_coverage else 0,
            1 if (self._policy.prefer_local and bound.capabilities.is_remote) else 0,
            0 if (self._policy.prefer_deterministic and bound.capabilities.deterministic) else 1,
            bound.cost_class,
            bound.adapter_id,
        )

    def _chain(
        self, selected: BoundUnderstander, covered: Sequence[AttributeKey]
    ) -> tuple[BoundUnderstander, ...]:
        """The fallback chain for a selection.

        Only understanders that cover **at least what the selection covers**
        qualify. A fallback producing less would silently narrow the answer while
        looking like a successful degradation — and 10_RELIABILITY §7.2 rule 1
        insists a fallback is never silent.
        """
        if not covered or self._policy.max_fallback_depth == 0:
            return ()
        alternatives = [
            bound
            for bound in self._bound
            if bound.adapter_id != selected.adapter_id
            and bound.capabilities.covers(covered) == tuple(covered)
        ]
        ranked = sorted(alternatives, key=lambda b: (b.is_fallback, b.cost_class, b.adapter_id))
        return tuple(ranked[: self._policy.max_fallback_depth])

    # --- capability publication ------------------------------------------------ #

    def producible_attributes(self) -> frozenset[AttributeKey]:
        """Everything any bound understander can produce.

        Published *"so capability gaps are visible"* (04_MODULES §M9). This is the
        set M8's demand registry consults to answer a consumer honestly at
        registration rather than leaving it waiting.
        """
        producible: set[AttributeKey] = set()
        for bound in self._bound:
            producible.update(bound.capabilities.producible_attributes)
        return frozenset(producible)

    def capabilities(self) -> tuple[UnderstanderCapabilities, ...]:
        return tuple(bound.capabilities for bound in self._bound)

    def find(self, adapter_id: str) -> BoundUnderstander | None:
        for bound in self._bound:
            if bound.adapter_id == adapter_id:
                return bound
        return None

    @property
    def bound(self) -> tuple[BoundUnderstander, ...]:
        return tuple(self._bound)

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    def __len__(self) -> int:
        return len(self._bound)


@dataclass(slots=True)
class CircuitBreaker:
    """Per-model failure state (04_MODULES §M9 State Ownership).

    Opens after consecutive failures and closes after a cooldown. The point is
    not to save the failing model — it is to stop spending the understanding
    budget on calls that will not succeed, and to make the failure *visible*
    rather than expressed as latency.

    10_RELIABILITY classifies an adapter crash as **Systemic**: retry makes it
    worse. A breaker is the documented response.
    """

    model_id: ModelId
    threshold: int = 3
    cooldown_ns: int = 30_000_000_000
    consecutive_failures: int = 0
    opened_at_ns: int | None = None
    """When the breaker opened, or ``None`` for closed.

    ``None`` rather than ``0``: a virtual clock starts at zero, so a breaker that
    tripped at monotonic time zero would be indistinguishable from one that never
    tripped — and every deterministic test would silently exercise the
    circuit-closed path.
    """

    trips: int = 0

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError("circuit breaker threshold must be >= 1")
        if self.cooldown_ns <= 0:
            raise ValueError("circuit breaker cooldown must be positive")

    def record_success(self) -> None:
        """A success closes the breaker immediately.

        Immediately rather than gradually: a model that answered is working, and
        holding the circuit open after that would be the platform disbelieving
        evidence it just received.
        """
        self.consecutive_failures = 0
        self.opened_at_ns = None

    def record_failure(self, now_ns: int) -> bool:
        """Record a failure. Returns whether the breaker is now open."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold and self.opened_at_ns is None:
            self.opened_at_ns = now_ns
            self.trips += 1
        return self.opened_at_ns is not None

    def is_open(self, now_ns: int) -> bool:
        """Whether calls are currently refused.

        A half-open state is implicit: once the cooldown elapses the breaker
        reports closed, and the next call is the probe. An explicit half-open
        counter would add state without changing behaviour.
        """
        if self.opened_at_ns is None:
            return False
        if now_ns - self.opened_at_ns >= self.cooldown_ns:
            self.opened_at_ns = None
            self.consecutive_failures = 0
            return False
        return True
