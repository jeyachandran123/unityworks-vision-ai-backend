"""The response cache, batching, and bounded concurrency.

04_MODULES §M9 on why the cache key is the whole design:

> *Because `CropId` is a content hash and prompt/model versions are explicit, the
> cache is **correct by construction** — a cache hit is guaranteed to be the
> answer the current configuration would produce. Caches keyed on object id or
> timestamp instead are the usual source of stale-attribute bugs.*

``test_a_prompt_change_produces_a_different_key`` and its model counterpart are
the executable form of that sentence. Nothing here invalidates: a stale entry
cannot be *reached*, so it cannot be served.
"""

from __future__ import annotations

import threading

import pytest

from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.understanding import UnderstandingOutcome
from vision_os.perception.understanding import (
    ModelSemaphore,
    ResponseCache,
    cache_key,
    group_for_batching,
)

from ..conftest import (
    CARRYING,
    HEADWEAR,
    OTHER_TENANT,
    POSTURE,
    TENANT,
    at,
)
from .test_engine import build_result


def key(**overrides):
    payload = {
        "tenant_id": TENANT,
        "crop_id": "crop-1",
        "prompt_version": "person.appearance@1.0.0",
        "model_version": "model-a",
        "attributes": (POSTURE,),
    }
    payload.update(overrides)
    return cache_key(**payload)


class TestTheCacheKey:
    def test_the_same_question_produces_the_same_key(self) -> None:
        assert key() == key()

    def test_a_prompt_change_produces_a_different_key(self) -> None:
        """**Correct by construction.** There is nothing to invalidate.

        A prompt edit changes the key, so the old answer becomes unreachable
        rather than stale — which is why this cache cannot serve an answer the
        current configuration would not produce.
        """
        assert key() != key(prompt_version="person.appearance@1.1.0")

    def test_a_model_change_produces_a_different_key(self) -> None:
        assert key() != key(model_version="model-b")

    def test_different_pixels_produce_a_different_key(self) -> None:
        """``CropId`` is a content hash, so this follows from the crop's design."""
        assert key() != key(crop_id="crop-2")

    def test_a_different_attribute_set_produces_a_different_key(self) -> None:
        assert key() != key(attributes=(POSTURE, HEADWEAR))

    def test_attribute_order_does_not_matter(self) -> None:
        """Asking for ``(a, b)`` and ``(b, a)`` is the same question.

        A key that distinguished them would halve the hit rate and make the
        cache's behaviour depend on the order a demand happened to list its
        attributes.
        """
        assert key(attributes=(POSTURE, CARRYING)) == key(
            attributes=(CARRYING, POSTURE)
        )

    def test_tenants_never_share_a_key(self) -> None:
        """12_SECURITY §4. Content addressing means two tenants photographing the
        same scene could hash identically; an untenanted key would let one
        tenant's answer serve the other's request."""
        assert key() != key(tenant_id=OTHER_TENANT)
        assert str(TENANT) in key()[0]


class TestTheCache:
    def test_a_hit_returns_the_stored_result(self) -> None:
        cache = ResponseCache(capacity=8)
        result = build_result()
        cache.put(key(), result, at(0))
        assert cache.get(key(), at(0)) is not None

    def test_a_miss_returns_none(self) -> None:
        assert ResponseCache(capacity=8).get(key(), at(0)) is None

    def test_failures_are_not_cached(self) -> None:
        """A timeout is a fact about this moment, not about this crop.

        Caching it would extend a transient outage for the life of the entry —
        turning a one-second blip into an hour of missing attributes.
        """
        cache = ResponseCache(capacity=8)
        cache.put(key(), build_result(outcome=UnderstandingOutcome.TIMED_OUT), at(0))
        assert cache.get(key(), at(0)) is None

    def test_an_honest_empty_answer_is_cached(self) -> None:
        """*"Nothing fit the schema"* **is** a property of these pixels and this
        prompt, so it is worth caching. It is not a failure."""
        cache = ResponseCache(capacity=8)
        cache.put(key(), build_result(outcome=UnderstandingOutcome.NO_ATTRIBUTES), at(0))
        assert cache.get(key(), at(0)) is not None

    def test_raw_output_is_stripped_before_storage(self) -> None:
        """The cache holds control-plane data. Keeping megabytes of verbatim
        model output per entry would make the cache the platform's largest
        allocation (V12)."""
        cache = ResponseCache(capacity=8)
        cache.put(key(), build_result(raw_output=b"x" * 4096), at(0))
        assert cache.get(key(), at(0)).raw_output is None

    def test_the_cache_is_bounded(self) -> None:
        cache = ResponseCache(capacity=4)
        for index in range(50):
            cache.put(key(crop_id=f"crop-{index}"), build_result(), at(0))
        assert len(cache) == 4
        assert cache.stats().evictions == 46

    def test_eviction_is_least_recently_used(self) -> None:
        cache = ResponseCache(capacity=2)
        cache.put(key(crop_id="a"), build_result(), at(0))
        cache.put(key(crop_id="b"), build_result(), at(0))
        cache.get(key(crop_id="a"), at(0))
        cache.put(key(crop_id="c"), build_result(), at(0))
        assert cache.get(key(crop_id="a"), at(0)) is not None
        assert cache.get(key(crop_id="b"), at(0)) is None

    def test_entries_expire(self) -> None:
        """The TTL exists for *age*, not correctness — the key already handles
        correctness. A consumer reading a six-hour-old answer should be told."""
        cache = ResponseCache(capacity=8, ttl=Duration.from_millis(1_000))
        cache.put(key(), build_result(), Instant(0))
        assert cache.get(key(), Instant(500_000_000)) is not None
        assert cache.get(key(), Instant(2_000_000_000)) is None
        assert cache.stats().expirations == 1

    def test_no_ttl_means_no_expiry(self) -> None:
        cache = ResponseCache(capacity=8, ttl=None)
        cache.put(key(), build_result(), Instant(0))
        assert cache.get(key(), Instant(10**15)) is not None

    def test_erasure_reaches_the_cache(self) -> None:
        cache = ResponseCache(capacity=8)
        cache.put(key(), build_result(), at(0))
        cache.put(key(tenant_id=OTHER_TENANT), build_result(), at(0))
        assert cache.forget_tenant(TENANT) == 1
        assert cache.get(key(), at(0)) is None
        assert cache.get(key(tenant_id=OTHER_TENANT), at(0)) is not None

    def test_hit_rate_is_reported(self) -> None:
        cache = ResponseCache(capacity=8)
        cache.put(key(), build_result(), at(0))
        cache.get(key(), at(0))
        cache.get(key(crop_id="missing"), at(0))
        assert cache.stats().hit_rate == pytest.approx(0.5)

    def test_a_zero_capacity_cache_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ResponseCache(capacity=0)


class TestBatchGrouping:
    def test_compatible_requests_group_together(self) -> None:
        groups = group_for_batching(
            [("vlm.a", "p@1"), ("vlm.a", "p@1"), ("vlm.a", "p@1")], max_batch_size=8
        )
        assert len(groups) == 1
        assert groups[0].size == 3

    def test_different_prompts_never_batch(self) -> None:
        """08_RUNTIME §1: only ``(model, prompt_version)``-compatible requests
        batch. Two prompts are two questions, and answering one while attributing
        it to both is fabrication with extra steps."""
        groups = group_for_batching(
            [("vlm.a", "p@1"), ("vlm.a", "p@2")], max_batch_size=8
        )
        assert len(groups) == 2

    def test_different_models_never_batch(self) -> None:
        groups = group_for_batching(
            [("vlm.a", "p@1"), ("vlm.b", "p@1")], max_batch_size=8
        )
        assert len(groups) == 2

    def test_batches_are_size_bounded(self) -> None:
        groups = group_for_batching([("vlm.a", "p@1")] * 10, max_batch_size=3)
        assert [g.size for g in groups] == [3, 3, 3, 1]

    def test_composition_is_a_pure_function_of_input(self) -> None:
        """08_RUNTIME §4.3 requires composition not to depend on arrival timing.

        Two identical inputs must produce identical batches, or a replay batches
        differently and the results are not comparable.
        """
        keys = [("vlm.a", "p@1"), ("vlm.b", "p@1"), ("vlm.a", "p@2"), ("vlm.a", "p@1")]
        first = group_for_batching(keys, max_batch_size=4)
        second = group_for_batching(keys, max_batch_size=4)
        assert [(g.adapter_id, g.indices) for g in first] == [
            (g.adapter_id, g.indices) for g in second
        ]

    def test_indices_are_ascending_within_a_group(self) -> None:
        groups = group_for_batching(
            [("vlm.a", "p@1"), ("vlm.b", "p@1"), ("vlm.a", "p@1")], max_batch_size=8
        )
        for group in groups:
            assert list(group.indices) == sorted(group.indices)

    def test_every_request_lands_in_exactly_one_group(self) -> None:
        keys = [("vlm.a", "p@1"), ("vlm.b", "p@2"), ("vlm.a", "p@1"), ("vlm.c", "p@3")]
        groups = group_for_batching(keys, max_batch_size=2)
        placed = [index for group in groups for index in group.indices]
        assert sorted(placed) == list(range(len(keys)))

    def test_an_empty_input_produces_no_groups(self) -> None:
        assert group_for_batching([], max_batch_size=4) == ()

    def test_a_zero_batch_size_is_refused(self) -> None:
        with pytest.raises(ValueError):
            group_for_batching([("a", "b")], max_batch_size=0)


class TestBoundedConcurrency:
    def test_the_limit_is_enforced(self) -> None:
        semaphore = ModelSemaphore(limit=2)
        assert semaphore.try_acquire()
        assert semaphore.try_acquire()
        assert not semaphore.try_acquire()

    def test_release_frees_a_slot(self) -> None:
        semaphore = ModelSemaphore(limit=1)
        semaphore.try_acquire()
        semaphore.release()
        assert semaphore.try_acquire()

    def test_try_acquire_never_blocks(self) -> None:
        """08_RUNTIME §5.2: understanding is best-effort, so a full semaphore
        sheds rather than queues. Blocking would convert a capacity problem into
        a latency problem in the layer beneath."""
        semaphore = ModelSemaphore(limit=1)
        semaphore.try_acquire()
        assert semaphore.try_acquire() is False

    def test_rejections_are_counted(self) -> None:
        semaphore = ModelSemaphore(limit=1)
        semaphore.try_acquire()
        for _ in range(5):
            semaphore.try_acquire()
        assert semaphore.rejections == 5

    def test_peak_is_tracked(self) -> None:
        semaphore = ModelSemaphore(limit=4)
        for _ in range(3):
            semaphore.try_acquire()
        semaphore.release()
        assert semaphore.peak == 3
        assert semaphore.in_flight == 2

    def test_the_limit_holds_under_contention(self) -> None:
        """The cap 08_RUNTIME §4.4 says protects the detector's latency budget:
        *"Long VLM calls are not preempted; instead concurrency is capped."*"""
        semaphore = ModelSemaphore(limit=3)
        observed: list[int] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            for _ in range(200):
                if semaphore.try_acquire():
                    with lock:
                        observed.append(semaphore.in_flight)
                    semaphore.release()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert observed, "some acquisitions must have succeeded"
        assert max(observed) <= 3, (
            f"{max(observed)} calls were in flight against a limit of 3; an "
            f"uncapped VLM starves the detector that gates the whole pipeline"
        )

    def test_release_below_zero_is_ignored(self) -> None:
        semaphore = ModelSemaphore(limit=2)
        semaphore.release()
        semaphore.release()
        assert semaphore.in_flight == 0

    def test_a_zero_limit_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelSemaphore(limit=0)
