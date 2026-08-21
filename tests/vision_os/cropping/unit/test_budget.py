"""The understanding budget, deduplication cache, and priority queue.

§M8 responsibility 5: *"Manage the understanding budget — a hard ceiling on
expensive inference per unit time."* A hard ceiling, not a target: a budget that
can be exceeded under load is not a budget, it is a suggestion that fails at
exactly the moment it was needed.

The budget is **shared across cameras**, deliberately. Understanding cost is a
property of the node's GPU, not of any camera, and a per-camera cap cannot stop
100 cameras each staying under their own limit while collectively exhausting the
device. Concurrency tests for that live in ``test_concurrency.py``.
"""

from __future__ import annotations

import pytest

from vision_os.core.model.ids import CropId
from vision_os.core.model.timebase import Duration, Instant
from vision_os.perception.cropping import (
    CropDeduplicationCache,
    PriorityQueue,
    UnderstandingBudget,
)

from ..conftest import OTHER_TENANT, TENANT


def budget_of(per_hour: float, *, window_ms: int = 3_600_000) -> UnderstandingBudget:
    return UnderstandingBudget(
        ceiling_per_hour=per_hour,
        window=Duration.from_millis(window_ms),
        now=Instant(0),
    )


class TestTheCeilingIsHard:
    def test_spending_within_the_ceiling_succeeds(self) -> None:
        budget = budget_of(10)
        assert all(budget.try_spend(Instant(0)) for _ in range(10))

    def test_the_ceiling_is_not_exceeded(self) -> None:
        """The whole point. Eleven calls against a ten-call budget yields ten."""
        budget = budget_of(10)
        granted = sum(1 for _ in range(50) if budget.try_spend(Instant(0)))
        assert granted == 10

    def test_exhaustion_never_blocks(self) -> None:
        """Waiting for budget converts a cost problem into a latency problem.

        The frame will be gone by the time the wait ends, so a blocked caller
        pays the delay and still gets nothing.
        """
        budget = budget_of(1)
        budget.try_spend(Instant(0))
        assert budget.try_spend(Instant(0)) is False

    def test_a_zero_ceiling_grants_nothing(self) -> None:
        assert budget_of(0).try_spend(Instant(0)) is False

    def test_shedding_is_counted(self) -> None:
        budget = budget_of(2)
        for _ in range(5):
            budget.try_spend(Instant(0))
        assert budget.status(Instant(0)).shed_in_window == 3

    def test_a_negative_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError):
            budget_of(-1)

    def test_a_non_positive_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            UnderstandingBudget(ceiling_per_hour=10, window=Duration(0))


class TestWindowReconciliation:
    def test_the_window_refills(self) -> None:
        budget = budget_of(3_600, window_ms=1_000)
        for _ in range(5):
            budget.try_spend(Instant(0))
        assert budget.try_spend(Instant(0)) is False
        assert budget.try_spend(Instant(2_000_000_000)) is True, (
            "elapsed time must earn fresh allowance"
        )

    def test_a_sub_one_allowance_still_eventually_spends(self) -> None:
        """The defect a token bucket exists to prevent.

        30 calls/hour over a 60-second window earns 0.5 calls per window. A
        budget that discarded the remainder each window would never reach an
        affordable integer call, and the platform would go permanently blind on
        a configuration that reads as perfectly reasonable.
        """
        budget = budget_of(30, window_ms=60_000)
        assert budget.try_spend(Instant(0)) is True, "the bucket starts full"
        assert budget.try_spend(Instant(0)) is False

        one_window = 60_000_000_000
        assert budget.try_spend(Instant(one_window)) is False, "0.5 is not enough"
        assert budget.try_spend(Instant(2 * one_window)) is True, (
            "two windows accrue a whole call"
        )

    def test_credit_does_not_accumulate_without_bound(self) -> None:
        """A quiet night must not buy an unbounded morning burst.

        A ceiling that can be saved up is not a ceiling on instantaneous load,
        which is the limit the GPU actually has.
        """
        budget = budget_of(3_600, window_ms=1_000)
        idle_for_an_hour = 3_600 * 1_000_000_000
        budget.status(Instant(idle_for_an_hour))
        granted = sum(
            1 for _ in range(100) if budget.try_spend(Instant(idle_for_an_hour))
        )
        assert granted <= 2, f"{granted} calls burst through a one-per-second ceiling"

    def test_status_reports_pressure(self) -> None:
        budget = budget_of(10)
        for _ in range(5):
            budget.try_spend(Instant(0))
        assert budget.status(Instant(0)).pressure == pytest.approx(0.5)

    def test_pressure_above_one_means_over_ceiling(self) -> None:
        budget = budget_of(0)
        budget._spent_in_window = 3  # noqa: SLF001 - forcing the over-spend state
        assert budget.status(Instant(0)).pressure > 1.0

    def test_remaining_never_goes_negative(self) -> None:
        budget = budget_of(10)
        for _ in range(20):
            budget.try_spend(Instant(0))
        assert budget.status(Instant(0)).remaining_in_window >= 0.0


class TestRefunds:
    def test_a_refund_restores_capacity(self) -> None:
        """A rejected crop bought nothing, so it must not cost anything.

        Without this, a run of gate rejections exhausts the budget having bought
        nothing at all, and the platform stops looking at the objects it *could*
        have answered for.
        """
        budget = budget_of(1)
        assert budget.try_spend(Instant(0))
        assert budget.try_spend(Instant(0)) is False
        budget.refund()
        assert budget.try_spend(Instant(0))

    def test_refunds_do_not_go_below_zero(self) -> None:
        budget = budget_of(10)
        budget.refund(cost=5)
        assert budget.status(Instant(0)).spent_in_window == 0

    def test_per_demand_spend_is_attributed(self) -> None:
        """Cost attribution: which consumer is spending the node's GPU."""
        budget = budget_of(100)
        budget.try_spend(Instant(0), demand_ids=("d1",))
        budget.try_spend(Instant(0), demand_ids=("d1", "d2"))
        assert budget.demand_calls("d1") == 2
        assert budget.demand_calls("d2") == 1

    def test_a_refund_reverses_attribution(self) -> None:
        budget = budget_of(100)
        budget.try_spend(Instant(0), demand_ids=("d1",))
        budget.refund(demand_ids=("d1",))
        assert budget.demand_calls("d1") == 0


class TestDeduplicationCache:
    def test_a_hit_returns_the_stored_crop(self) -> None:
        cache = CropDeduplicationCache(capacity=4)
        cache.put(TENANT, "hash-1", CropId("hash-1"))
        assert cache.get(TENANT, "hash-1") == CropId("hash-1")

    def test_tenants_never_share_a_cache_entry(self) -> None:
        """12_SECURITY section 4: every cache key includes ``tenant_id``.

        A cache keyed on content alone would let one tenant's crop satisfy
        another tenant's request — a cross-tenant data path wearing the disguise
        of an optimization.
        """
        cache = CropDeduplicationCache(capacity=4)
        cache.put(TENANT, "identical-pixels", CropId("crop-a"))
        assert cache.get(OTHER_TENANT, "identical-pixels") is None

    def test_the_cache_is_bounded(self) -> None:
        """§M8 names cache growth as a failure mode.

        An unbounded dedup cache is a memory leak that looks like a hit-rate
        improvement, which is the hardest kind to get anyone to fix.
        """
        cache = CropDeduplicationCache(capacity=8)
        for index in range(100):
            cache.put(TENANT, f"hash-{index}", CropId(f"crop-{index}"))
        assert len(cache) == 8
        assert cache.stats().evictions == 92

    def test_eviction_is_least_recently_used(self) -> None:
        cache = CropDeduplicationCache(capacity=2)
        cache.put(TENANT, "a", CropId("a"))
        cache.put(TENANT, "b", CropId("b"))
        cache.get(TENANT, "a")
        cache.put(TENANT, "c", CropId("c"))
        assert cache.get(TENANT, "a") is not None
        assert cache.get(TENANT, "b") is None

    def test_forgetting_a_tenant_reaches_the_cache(self) -> None:
        """Erasure that could not reach the cache would leave references alive."""
        cache = CropDeduplicationCache(capacity=8)
        cache.put(TENANT, "a", CropId("a"))
        cache.put(OTHER_TENANT, "b", CropId("b"))
        assert cache.forget_tenant(TENANT) == 1
        assert cache.get(TENANT, "a") is None
        assert cache.get(OTHER_TENANT, "b") is not None

    def test_hit_rate_is_reported(self) -> None:
        cache = CropDeduplicationCache(capacity=4)
        cache.put(TENANT, "a", CropId("a"))
        cache.get(TENANT, "a")
        cache.get(TENANT, "missing")
        assert cache.stats().hit_rate == pytest.approx(0.5)

    def test_a_zero_capacity_cache_is_refused(self) -> None:
        with pytest.raises(ValueError):
            CropDeduplicationCache(capacity=0)


class _Request:
    def __init__(self, name: str, priority_class: str) -> None:
        self.name = name
        self.priority_class = priority_class


class TestPriorityQueue:
    def test_ordering_follows_the_configured_sequence(self) -> None:
        queue = PriorityQueue(("urgent", "standard", "background"))
        ordered = queue.order(
            [
                _Request("c", "background"),
                _Request("a", "urgent"),
                _Request("b", "standard"),
            ]
        )
        assert [r.name for r in ordered] == ["a", "b", "c"]

    def test_an_unknown_class_sorts_last_rather_than_raising(self) -> None:
        """A consumer typo must degrade one demand, not stop the pipeline."""
        queue = PriorityQueue(("urgent", "standard"))
        ordered = queue.order([_Request("typo", "urgnet"), _Request("ok", "urgent")])
        assert [r.name for r in ordered] == ["ok", "typo"]

    def test_ties_break_deterministically_by_arrival(self) -> None:
        """Without this, equal-priority requests shed in heap order (V13)."""
        queue = PriorityQueue(("standard",))
        requests = [_Request(f"r{i}", "standard") for i in range(10)]
        assert [r.name for r in queue.order(list(requests))] == [
            r.name for r in requests
        ]

    def test_priority_is_never_interpreted(self) -> None:
        """The queue maps a class to a rank and asks nothing about its meaning."""
        queue = PriorityQueue(("☃", "🔥"))
        ordered = queue.order([_Request("fire", "🔥"), _Request("snow", "☃")])
        assert [r.name for r in ordered] == ["snow", "fire"]

    def test_an_empty_ordering_puts_everything_at_the_same_rank(self) -> None:
        queue = PriorityQueue(())
        requests = [_Request(f"r{i}", f"class-{i}") for i in range(5)]
        assert [r.name for r in queue.order(list(requests))] == [
            r.name for r in requests
        ]

    def test_known_classes_are_reported_in_order(self) -> None:
        queue = PriorityQueue(("a", "b", "c"))
        assert queue.known_classes == ("a", "b", "c")
