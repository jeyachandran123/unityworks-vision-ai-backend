"""Capability routing, fallback chains, and the circuit breaker.

The property this module exists to guarantee, from 11_PERFORMANCE §7:

> *Use the VLM to discover and validate an attribute, use its evidence to train a
> specialized head, then move that attribute to the head in production, with zero
> consumer impact.*

``test_a_specialized_head_beats_a_vlm_on_cost`` is that migration, executable. It
passes because routing reads a **declared cost class**, not a name — which is the
only reason the migration is a binding change rather than a rewrite.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import ModelId
from vision_os.perception.understanding import (
    CapabilityRouter,
    CircuitBreaker,
    RoutingPolicy,
)

from ..conftest import (
    CARRYING,
    HEADWEAR,
    HEIGHT,
    POSTURE,
    UNREGISTERED,
    bind,
    scripted,
)


def head(attribute=HEADWEAR, *, cost: float = 0.01, adapter_id="attr.head"):
    from vision_os.adapters.understanding import StaticAttributeHead

    return StaticAttributeHead(
        attribute=attribute, value=True, cost_class=cost, adapter_id=adapter_id
    )


class TestCapabilityRouting:
    def test_an_understander_covering_the_request_is_selected(self) -> None:
        adapter = scripted(producible=(POSTURE, HEADWEAR))
        router = CapabilityRouter([bind(adapter)])
        decision = router.route((POSTURE,))
        assert decision.has_route
        assert decision.selected.adapter_id == adapter.adapter_id
        assert decision.covered == (POSTURE,)

    def test_an_unproducible_attribute_is_reported_uncovered(self) -> None:
        """The V8 field. A route that quietly dropped it would leave the consumer
        waiting for data that will never arrive."""
        router = CapabilityRouter([bind(scripted(producible=(POSTURE,)))])
        decision = router.route((UNREGISTERED,))
        assert not decision.has_route
        assert decision.uncovered == (UNREGISTERED,)

    def test_no_bound_understander_is_a_capability_gap_not_a_crash(self) -> None:
        router = CapabilityRouter([])
        decision = router.route((POSTURE,))
        assert not decision.has_route
        assert decision.uncovered == (POSTURE,)
        assert "no bound understander" in decision.reason

    def test_an_empty_request_routes_nowhere(self) -> None:
        router = CapabilityRouter([bind(scripted(producible=(POSTURE,)))])
        assert not router.route(()).has_route

    def test_a_specialized_head_beats_a_vlm_on_cost(self) -> None:
        """**11_PERFORMANCE §7's migration, executable.**

        A 2 MB classifier and a 7B VLM both produce ``headwear_present``. The
        head wins because it declares a cost class two orders of magnitude lower
        — and the router never learns which is which.
        """
        vlm = scripted(producible=(POSTURE, HEADWEAR), cost_class=1.0, adapter_id="vlm.big")
        specialist = head(HEADWEAR, cost=0.01)
        router = CapabilityRouter([bind(vlm), bind(specialist)])

        decision = router.route((HEADWEAR,))
        assert decision.selected.adapter_id == "attr.head"
        assert decision.selected.cost_class == pytest.approx(0.01)

    def test_coverage_beats_cost_when_preferred(self) -> None:
        """Two calls are almost always worse than one.

        §M9 puts attribute batching in a single prompt at a *"3-5x saving"*, so a
        cheap understander covering half the request loses to a dearer one
        covering all of it.
        """
        cheap = head(HEADWEAR, cost=0.01)
        broad = scripted(
            producible=(POSTURE, HEADWEAR), cost_class=1.0, adapter_id="vlm.broad"
        )
        router = CapabilityRouter(
            [bind(cheap), bind(broad)], policy=RoutingPolicy(prefer_coverage=True)
        )
        decision = router.route((POSTURE, HEADWEAR))
        assert decision.selected.adapter_id == "vlm.broad"
        assert decision.fully_covered

    def test_a_local_model_is_preferred_over_a_remote_one(self) -> None:
        """12_SECURITY: residency must not lose to a marginal cost saving."""
        remote = scripted(
            producible=(POSTURE,),
            cost_class=0.5,
            adapter_id="vlm.remote",
            data_residency="remote(us-east-1)",
        )
        local = scripted(producible=(POSTURE,), cost_class=1.0, adapter_id="vlm.local")
        router = CapabilityRouter(
            [bind(remote), bind(local)], policy=RoutingPolicy(prefer_local=True)
        )
        assert router.route((POSTURE,)).selected.adapter_id == "vlm.local"

    def test_routing_is_deterministic(self) -> None:
        """Identical bindings and request always produce an identical decision."""
        adapters = [
            scripted(producible=(POSTURE,), cost_class=1.0, adapter_id=f"vlm.{i}")
            for i in range(5)
        ]
        router = CapabilityRouter([bind(a) for a in adapters])
        picks = {router.route((POSTURE,)).selected.adapter_id for _ in range(10)}
        assert len(picks) == 1

    def test_ties_break_on_adapter_id_not_binding_order(self) -> None:
        """Otherwise the answer depends on which adapter was registered first."""
        forward = CapabilityRouter(
            [
                bind(scripted(producible=(POSTURE,), adapter_id="vlm.aaa")),
                bind(scripted(producible=(POSTURE,), adapter_id="vlm.zzz")),
            ]
        )
        backward = CapabilityRouter(
            [
                bind(scripted(producible=(POSTURE,), adapter_id="vlm.zzz")),
                bind(scripted(producible=(POSTURE,), adapter_id="vlm.aaa")),
            ]
        )
        assert (
            forward.route((POSTURE,)).selected.adapter_id
            == backward.route((POSTURE,)).selected.adapter_id
            == "vlm.aaa"
        )

    def test_binding_the_same_adapter_twice_is_refused(self) -> None:
        adapter = scripted(producible=(POSTURE,))
        with pytest.raises(ValueError, match="bound twice"):
            CapabilityRouter([bind(adapter), bind(adapter)])

    def test_producible_attributes_are_published(self) -> None:
        """*"So capability gaps are visible"* (§M9). M8's demand registry reads
        this to refuse a demand honestly at registration."""
        router = CapabilityRouter(
            [bind(scripted(producible=(POSTURE,))), bind(head(HEADWEAR))]
        )
        assert router.producible_attributes() == frozenset({POSTURE, HEADWEAR})


class TestFallbackChains:
    def test_a_fallback_never_wins_a_primary_route(self) -> None:
        """A fallback is a *different accuracy profile*, not a cheaper option.

        If cost alone decided, the platform would quietly run on its worst model
        forever — 10_RELIABILITY §7.2 calls that one of the silent failures.
        """
        fallback = scripted(
            producible=(POSTURE,), cost_class=0.01, adapter_id="vlm.fallback"
        )
        primary = scripted(
            producible=(POSTURE,), cost_class=10.0, adapter_id="vlm.primary"
        )
        router = CapabilityRouter([bind(primary), bind(fallback, is_fallback=True)])
        decision = router.route((POSTURE,))
        assert decision.selected.adapter_id == "vlm.primary"
        assert [f.adapter_id for f in decision.fallbacks] == ["vlm.fallback"]

    def test_a_fallback_must_cover_what_the_primary_covered(self) -> None:
        """A narrower fallback would silently shrink the answer while looking
        like a successful degradation."""
        primary = scripted(
            producible=(POSTURE, HEADWEAR), cost_class=1.0, adapter_id="vlm.primary"
        )
        narrow = scripted(
            producible=(POSTURE,), cost_class=0.1, adapter_id="vlm.narrow"
        )
        router = CapabilityRouter([bind(primary), bind(narrow, is_fallback=True)])
        decision = router.route((POSTURE, HEADWEAR))
        assert decision.fallbacks == (), "a narrower model is not a fallback"

    def test_the_chain_is_depth_bounded(self) -> None:
        primary = scripted(producible=(POSTURE,), adapter_id="vlm.primary")
        others = [
            scripted(producible=(POSTURE,), adapter_id=f"vlm.f{i}") for i in range(5)
        ]
        router = CapabilityRouter(
            [bind(primary), *[bind(o, is_fallback=True) for o in others]],
            policy=RoutingPolicy(max_fallback_depth=2),
        )
        assert len(router.route((POSTURE,)).fallbacks) == 2

    def test_a_zero_depth_policy_produces_no_chain(self) -> None:
        router = CapabilityRouter(
            [
                bind(scripted(producible=(POSTURE,), adapter_id="a")),
                bind(scripted(producible=(POSTURE,), adapter_id="b"), is_fallback=True),
            ],
            policy=RoutingPolicy(max_fallback_depth=0),
        )
        assert router.route((POSTURE,)).fallbacks == ()

    def test_the_chain_is_ordered_deterministically(self) -> None:
        primary = scripted(producible=(POSTURE,), adapter_id="vlm.primary")
        cheap = scripted(producible=(POSTURE,), cost_class=0.1, adapter_id="vlm.cheap")
        dear = scripted(producible=(POSTURE,), cost_class=9.0, adapter_id="vlm.dear")
        router = CapabilityRouter(
            [bind(primary), bind(dear, is_fallback=True), bind(cheap, is_fallback=True)]
        )
        assert [f.adapter_id for f in router.route((POSTURE,)).fallbacks] == [
            "vlm.cheap",
            "vlm.dear",
        ]

    @pytest.mark.parametrize("depth", [-1])
    def test_a_negative_depth_is_refused(self, depth) -> None:
        with pytest.raises(ValueError):
            RoutingPolicy(max_fallback_depth=depth)


class TestCircuitBreaker:
    def test_it_opens_after_the_threshold(self) -> None:
        breaker = CircuitBreaker(model_id=ModelId("m"), threshold=3)
        assert not breaker.record_failure(now_ns=0)
        assert not breaker.record_failure(now_ns=1)
        assert breaker.record_failure(now_ns=2)
        assert breaker.is_open(now_ns=3)

    def test_a_success_closes_it_immediately(self) -> None:
        """A model that answered is working; holding the circuit open after that
        would be the platform disbelieving evidence it just received."""
        breaker = CircuitBreaker(model_id=ModelId("m"), threshold=2)
        breaker.record_failure(now_ns=0)
        breaker.record_success()
        assert breaker.consecutive_failures == 0
        assert not breaker.record_failure(now_ns=1), "the count restarted"

    def test_it_closes_after_the_cooldown(self) -> None:
        breaker = CircuitBreaker(
            model_id=ModelId("m"), threshold=1, cooldown_ns=1_000
        )
        breaker.record_failure(now_ns=0)
        assert breaker.is_open(now_ns=500)
        assert not breaker.is_open(now_ns=1_500)

    def test_reopening_is_counted(self) -> None:
        """Trip count is the signal that a model is flapping rather than down."""
        breaker = CircuitBreaker(model_id=ModelId("m"), threshold=1, cooldown_ns=10)
        breaker.record_failure(now_ns=0)
        breaker.is_open(now_ns=100)
        breaker.record_failure(now_ns=100)
        assert breaker.trips == 2

    @pytest.mark.parametrize(
        "kwargs", [{"threshold": 0}, {"cooldown_ns": 0}, {"cooldown_ns": -1}]
    )
    def test_invalid_configuration_is_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(model_id=ModelId("m"), **kwargs)


class TestNoModelNamesInRouting:
    def test_routing_reasons_about_declarations_only(self) -> None:
        """The router must reach the same answer for two adapters that differ
        only in name — otherwise something is branching on identity."""
        first = scripted(producible=(POSTURE,), cost_class=1.0, adapter_id="vlm.qwen")
        second = scripted(producible=(POSTURE,), cost_class=0.5, adapter_id="vlm.gpt")
        router = CapabilityRouter([bind(first), bind(second)])
        assert router.route((POSTURE,)).selected.adapter_id == "vlm.gpt"

        flipped = CapabilityRouter(
            [
                bind(scripted(producible=(POSTURE,), cost_class=0.5, adapter_id="vlm.qwen")),
                bind(scripted(producible=(POSTURE,), cost_class=1.0, adapter_id="vlm.gpt")),
            ]
        )
        assert flipped.route((POSTURE,)).selected.adapter_id == "vlm.qwen", (
            "the cheaper adapter won both times; the router read cost, not name"
        )

    def test_a_router_holds_no_model_specific_branch(self) -> None:
        import inspect

        from vision_os.perception.understanding import routing

        source = inspect.getsource(routing)
        for vendor in ("qwen", "gpt", "claude", "gemini", "llava", "internvl"):
            assert vendor not in source.lower(), (
                f"the router names '{vendor}'; routing must reason about declared "
                f"capability alone (invariant V3)"
            )


class TestMixedCapabilities:
    def test_partial_coverage_is_served_and_the_gap_reported(self) -> None:
        """Half an answer beats none, provided the missing half is named.

        Refusing the whole request because one attribute is unproducible would
        cost the consumer the attributes it *could* have had. Serving what is
        possible and reporting ``uncovered`` gives it both the data and the
        capability gap (V8).
        """
        router = CapabilityRouter([bind(scripted(producible=(POSTURE,)))])
        decision = router.route((POSTURE, HEIGHT))
        assert decision.has_route
        assert decision.covered == (POSTURE,)
        assert decision.uncovered == (HEIGHT,)
        assert not decision.fully_covered

    def test_a_composite_of_heads_covers_a_mixed_request(self) -> None:
        """The ``understander.router`` composite, assembled from parts."""
        both = scripted(producible=(POSTURE, HEADWEAR, CARRYING), adapter_id="vlm.all")
        router = CapabilityRouter([bind(both), bind(head(HEADWEAR))])
        decision = router.route((POSTURE, HEADWEAR))
        assert decision.fully_covered
        assert decision.selected.adapter_id == "vlm.all"
