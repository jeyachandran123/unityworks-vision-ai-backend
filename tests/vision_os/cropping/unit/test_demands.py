"""The demand registry and its closed lifecycle.

09_API_CONTRACTS section 4.4: *"A demand that quietly stops being satisfied is
the failure mode this lifecycle exists to prevent."*

Two properties carry the weight here.

**Refusal is honest.** A demand naming an unregistered attribute is rejected *at
registration*, with a pointer to the registration process — not accepted and
silently never served. That is the fourth ring of Semantic Ceiling enforcement.

**The acknowledgement tells the truth about limits.** ``effective_freshness``
reports what the platform can actually sustain, which may be longer than
requested. Accepting a demand the budget will not buy and quietly
under-delivering is the single most common integration failure in vision
platforms.
"""

from __future__ import annotations

import pytest

from vision_os.core.errors import DemandNotFoundError, DemandRejectedError
from vision_os.core.model.demand import (
    AcknowledgementStatus,
    DemandStatus,
    UnsatisfiableReason,
)
from vision_os.core.model.ids import AttributeKey, DemandId
from vision_os.core.model.timebase import Duration, Instant
from vision_os.perception.cropping import (
    CapabilityView,
    DemandRegistry,
    check_transition,
    is_legal,
)

from ..conftest import (
    CAMERA,
    COLOUR,
    GARMENT,
    OTHER_CAMERA,
    PERSON,
    at,
    make_demand,
)

UNKNOWN = AttributeKey("appearance.mood")


class TestRegistration:
    def test_a_satisfiable_demand_is_accepted(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        assert ack.status is AcknowledgementStatus.ACCEPTED
        assert ack.satisfiable == (COLOUR,)

    def test_an_unregistered_attribute_is_refused_at_registration(
        self, demand_registry
    ) -> None:
        """Refused with a pointer, not accepted and silently never served."""
        with pytest.raises(DemandRejectedError, match="register them first"):
            demand_registry.register(make_demand(attributes=(UNKNOWN,)), now=at(0))

    def test_a_partially_satisfiable_demand_says_so(self, capabilities) -> None:
        registry = DemandRegistry(
            capabilities=CapabilityView(
                registered_attributes=frozenset({COLOUR, GARMENT}),
                producible_attributes=frozenset({COLOUR}),
            )
        )
        ack = registry.register(make_demand(attributes=(COLOUR, GARMENT)), now=at(0))
        assert ack.status is AcknowledgementStatus.PARTIALLY_ACCEPTED
        assert ack.satisfiable == (COLOUR,)
        assert ack.unsatisfiable == ((GARMENT, UnsatisfiableReason.NO_CAPABLE_MODEL),)

    def test_a_registered_but_unproducible_attribute_is_admitted_then_marked(
        self,
    ) -> None:
        """No model can produce it *today*; capability can return.

        The demand is admitted so the consumer holds a real contract, then marked
        unsatisfiable so it knows nothing is arriving. Refusing outright would
        make a consumer re-register on a schedule to discover a model had loaded.
        """
        registry = DemandRegistry(
            capabilities=CapabilityView(registered_attributes=frozenset({COLOUR}))
        )
        ack = registry.register(make_demand(), now=at(0))
        assert ack.status is AcknowledgementStatus.PARTIALLY_ACCEPTED
        assert registry.get(ack.demand_id).status is DemandStatus.UNSATISFIABLE

    def test_an_id_is_minted_when_absent(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        assert ack.demand_id

    def test_a_supplied_id_is_honoured(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(demand_id="mine"), now=at(0))
        assert ack.demand_id == DemandId("mine")


class TestHonestFreshness:
    def test_the_acknowledgement_reports_what_can_be_sustained(
        self, demand_registry
    ) -> None:
        """The platform tells the truth about its limits, unprompted."""
        ack = demand_registry.register(
            make_demand(freshness_ms=1_000),
            now=at(0),
            sustainable_freshness=Duration.from_millis(30_000),
        )
        assert ack.effective_freshness.millis == pytest.approx(30_000)
        assert "budget sustains" in ack.detail

    def test_a_sustainable_request_is_honoured_exactly(self, demand_registry) -> None:
        ack = demand_registry.register(
            make_demand(freshness_ms=60_000),
            now=at(0),
            sustainable_freshness=Duration.from_millis(10_000),
        )
        assert ack.effective_freshness.millis == pytest.approx(60_000)
        assert ack.detail == "", "no warning when the request is affordable"

    def test_an_impossible_freshness_is_floored(self, demand_registry) -> None:
        """No camera at 5 fps delivers 10 ms freshness; promising it would lie."""
        ack = demand_registry.register(make_demand(freshness_ms=10), now=at(0))
        assert ack.effective_freshness.millis >= 1_000


class TestLifecycle:
    def test_the_transition_table_is_closed(self) -> None:
        assert is_legal(DemandStatus.VALIDATED, DemandStatus.ACTIVE)
        assert not is_legal(DemandStatus.VALIDATED, DemandStatus.THROTTLED)

    def test_terminal_states_are_terminal(self) -> None:
        """A consumer whose demand expired registers a new one.

        Resurrecting it would leave the acknowledgement it was given describing a
        budget that no longer exists.
        """
        for terminal in (
            DemandStatus.EXPIRED,
            DemandStatus.REVOKED,
            DemandStatus.REJECTED,
        ):
            assert terminal.is_terminal
            for target in DemandStatus:
                assert not is_legal(terminal, target)

    def test_an_illegal_transition_raises(self) -> None:
        with pytest.raises(DemandRejectedError, match="closed machine"):
            check_transition(DemandStatus.EXPIRED, DemandStatus.ACTIVE)

    def test_throttling_records_the_reduced_freshness(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        state = demand_registry.throttle(ack.demand_id, Duration.from_millis(90_000))
        assert state.status is DemandStatus.THROTTLED
        assert state.effective_freshness.millis == pytest.approx(90_000)

    def test_a_throttled_demand_is_still_served(self, demand_registry) -> None:
        """Throttled means slower, not silent."""
        ack = demand_registry.register(make_demand(), now=at(0))
        demand_registry.throttle(ack.demand_id, Duration.from_millis(90_000))
        assert DemandStatus.THROTTLED.is_serving
        assert demand_registry.get(ack.demand_id) in demand_registry.active()

    def test_restore_returns_the_acknowledged_freshness(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(freshness_ms=30_000), now=at(0))
        demand_registry.throttle(ack.demand_id, Duration.from_millis(90_000))
        state = demand_registry.restore(ack.demand_id)
        assert state.status is DemandStatus.ACTIVE
        assert state.effective_freshness == ack.effective_freshness

    def test_an_unsatisfiable_demand_is_not_served(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        demand_registry.mark_unsatisfiable(ack.demand_id, detail="model evicted")
        assert not demand_registry.active()

    def test_capability_can_return(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        demand_registry.mark_unsatisfiable(ack.demand_id)
        demand_registry.restore(ack.demand_id)
        assert demand_registry.get(ack.demand_id).status is DemandStatus.ACTIVE

    def test_revocation_is_idempotent(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        demand_registry.revoke(ack.demand_id)
        demand_registry.revoke(ack.demand_id)
        assert demand_registry.get(ack.demand_id).status is DemandStatus.REVOKED

    def test_revoking_an_unknown_demand_is_typed(self, demand_registry) -> None:
        with pytest.raises(DemandNotFoundError):
            demand_registry.revoke(DemandId("ghost"))

    def test_expiry_is_driven_by_a_sweep(self, demand_registry) -> None:
        """A demand with a TTL expires even at a camera that has gone quiet."""
        ack = demand_registry.register(make_demand(expires_ms=5_000), now=at(0))
        assert demand_registry.expire_due(Instant(1_000_000_000)) == ()
        assert demand_registry.expire_due(Instant(9_000_000_000)) == (ack.demand_id,)
        assert demand_registry.get(ack.demand_id).status is DemandStatus.EXPIRED


class TestScopeAndMatching:
    def test_a_camera_scope_excludes_others(self, demand_registry) -> None:
        demand_registry.register(make_demand(cameras=(CAMERA,)), now=at(0))
        assert demand_registry.matching(
            camera_id=CAMERA, class_id=PERSON, region_ids=()
        )
        assert not demand_registry.matching(
            camera_id=OTHER_CAMERA, class_id=PERSON, region_ids=()
        )

    def test_an_empty_scope_covers_everything(self, demand_registry) -> None:
        demand_registry.register(make_demand(cameras=()), now=at(0))
        assert demand_registry.matching(
            camera_id=OTHER_CAMERA, class_id=PERSON, region_ids=()
        )

    def test_a_class_filter_excludes_other_classes(self, demand_registry) -> None:
        from vision_os.core.model.ids import ClassId

        demand_registry.register(make_demand(classes=(PERSON,)), now=at(0))
        assert not demand_registry.matching(
            camera_id=CAMERA, class_id=ClassId("vehicle"), region_ids=()
        )

    def test_active_is_ordered_deterministically(self, demand_registry) -> None:
        """Trigger evaluation walks this list; a reordering changes tie outcomes."""
        for name in ("zzz", "aaa", "mmm"):
            demand_registry.register(make_demand(demand_id=name), now=at(0))
        assert [str(s.demand.demand_id) for s in demand_registry.active()] == [
            "aaa",
            "mmm",
            "zzz",
        ]


class TestCostAttribution:
    def test_served_calls_are_counted_per_demand(self, demand_registry) -> None:
        ack = demand_registry.register(make_demand(), now=at(0))
        demand_registry.record_served(ack.demand_id, at(1))
        demand_registry.record_served(ack.demand_id, at(2))
        state = demand_registry.get(ack.demand_id)
        assert state.calls_served == 2
        assert state.last_served == at(2)

    def test_recording_against_an_unknown_demand_is_silent(
        self, demand_registry
    ) -> None:
        """A revoked demand's in-flight crop must not crash the accounting."""
        demand_registry.record_served(DemandId("ghost"), at(0))

    def test_stats_report_the_serving_subset(self, demand_registry) -> None:
        first = demand_registry.register(make_demand(demand_id="a"), now=at(0))
        demand_registry.register(make_demand(demand_id="b"), now=at(0))
        demand_registry.revoke(first.demand_id)
        stats = demand_registry.stats()
        assert stats.total == 2
        assert stats.serving == 1
