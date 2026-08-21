"""Determinism, replay, and behaviour under stress.

Invariant V13 says a replay reproduces the run. For M9 that means something
specific and checkable: **the same crop, prompt and model produce the same
attributes, the same rejections, and the same decision path** — which is exactly
the property the response cache's key already assumes, and which port obligation
U5 makes an adapter's duty.

The stress half is not a benchmark. It checks that the bounded things stay
bounded and the guarded things stay guarded when a hundred requests arrive at
once, because 11_PERFORMANCE puts understanding at 20–2000 calls/second on a
saturated node and the failure mode there is a memory leak, not a slow response.
"""

from __future__ import annotations

import threading

import pytest

from vision_os.core.model.understanding import (
    UnderstandingOutcome,
    UnderstandingStep,
)
from vision_os.perception.understanding import ModelSemaphore, ResponseCache

from .conftest import (
    HEADWEAR,
    POSTURE,
    UNREGISTERED,
    LeakyUnderstander,
    answer_posture,
    build_engine,
    make_crop,
    make_request,
    scripted,
)


class TestReplayReproduces:
    def test_the_same_request_produces_the_same_attributes(self, engine) -> None:
        first = engine.understand(make_request(crop_id="a"), crops=[make_crop(crop_id="a")])
        second = engine.understand(make_request(crop_id="b"), crops=[make_crop(crop_id="b")])
        assert [(a.key, a.value) for a in first.attributes] == [
            (a.key, a.value) for a in second.attributes
        ]

    def test_the_decision_path_is_reproducible(self, engine) -> None:
        """*"Six months later, that is the difference between explaining a result
        and guessing at it"* — and an unreproducible path explains a run that
        never happened."""
        first = engine.understand(make_request(crop_id="a"), crops=[make_crop(crop_id="a")])
        second = engine.understand(make_request(crop_id="b"), crops=[make_crop(crop_id="b")])
        assert first.evidence.steps() == second.evidence.steps()

    def test_the_input_hash_is_stable(self, engine) -> None:
        """Two results with the same input hash and different answers prove the
        model is non-deterministic — a claim worth being able to make with
        evidence."""
        first = engine.understand(make_request(), crops=[make_crop()])
        engine.cache.clear()
        second = engine.understand(make_request(), crops=[make_crop()])
        assert first.evidence.input_hash == second.evidence.input_hash

    def test_different_crops_hash_differently(self, engine) -> None:
        first = engine.understand(make_request(crop_id="a"), crops=[make_crop(crop_id="a")])
        second = engine.understand(make_request(crop_id="b"), crops=[make_crop(crop_id="b")])
        assert first.evidence.input_hash != second.evidence.input_hash

    def test_rejections_are_reproducible(self, engine) -> None:
        adapter_fields = {str(POSTURE): "levitating", str(UNREGISTERED): True}
        outcomes = []
        for index in range(3):
            e = build_engine(
                engine._clock, engine._metrics, engine._events, engine._config,
                understanders=[LeakyUnderstander(fields=adapter_fields)],
            )
            result = e.understand(
                make_request(crop_id=f"c-{index}"), crops=[make_crop(crop_id=f"c-{index}")]
            )
            outcomes.append(
                tuple((f.field_name, f.reason) for f in result.rejected_fields)
            )
        assert outcomes[0] == outcomes[1] == outcomes[2]

    def test_attribute_order_is_stable(self, engine) -> None:
        """Dict iteration order is insertion order; a response whose keys arrived
        differently must still produce the same attribute list."""
        forward = LeakyUnderstander(
            fields={str(POSTURE): "standing", str(HEADWEAR): True},
            producible=(POSTURE, HEADWEAR),
        )
        backward = LeakyUnderstander(
            fields={str(HEADWEAR): True, str(POSTURE): "standing"},
            producible=(POSTURE, HEADWEAR),
            adapter_id="vlm.leaky2",
        )
        keys = []
        for adapter in (forward, backward):
            e = build_engine(
                engine._clock, engine._metrics, engine._events, engine._config,
                understanders=[adapter],
            )
            result = e.understand(
                make_request(attributes=(POSTURE, HEADWEAR)), crops=[make_crop()]
            )
            keys.append(result.attribute_keys)
        assert keys[0] == keys[1]

    def test_the_prompt_version_is_pinned_on_every_result(self, engine) -> None:
        result = engine.understand(make_request(), crops=[make_crop()])
        assert result.prompt_used.version == "1.0.0"
        assert result.prompt_used.content_hash.startswith("sha256:")


class TestTimeIndependence:
    def test_wall_time_does_not_change_the_answer(self, engine, clock) -> None:
        """An attribute stamped with inference time would make the evidence a
        measurement of the platform rather than of the world (V11)."""
        from vision_os.core.model.timebase import Duration

        first = engine.understand(make_request(crop_id="a"), crops=[make_crop(crop_id="a")])
        clock.advance(Duration.from_millis(60_000))
        second = engine.understand(make_request(crop_id="b"), crops=[make_crop(crop_id="b")])
        assert first.attribute(POSTURE).observed_at == second.attribute(POSTURE).observed_at

    def test_the_attribute_time_is_the_crops_capture_time(self, engine) -> None:
        from .conftest import at

        result = engine.understand(make_request(seq=11), crops=[make_crop(seq=11)])
        assert result.attribute(POSTURE).observed_at == at(11)


class TestUnderStress:
    def test_a_hundred_requests_all_produce_results(self, engine) -> None:
        """Every request id comes back. A dropped id is a lost answer in disguise.

        The adapter is given its own long script rather than the shared fixture's:
        running past a script yields an empty answer, which would make this pass
        for the wrong reason.
        """
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[scripted(*[answer_posture() for _ in range(200)])],
        )
        requests = [
            make_request(request_id=f"r-{i}", crop_id=f"c-{i}") for i in range(100)
        ]
        crops = {r.request_id: [make_crop(crop_id=str(r.crop_ids[0]))] for r in requests}
        results = e.understand_batch(requests, crops=crops)
        assert set(results) == {r.request_id for r in requests}
        assert all(
            r.outcome is UnderstandingOutcome.SUCCEEDED for r in results.values()
        )

    def test_the_cache_stays_bounded_under_load(self, engine) -> None:
        for index in range(5_000):
            engine.understand(
                make_request(request_id=f"r-{index}", crop_id=f"c-{index}"),
                crops=[make_crop(crop_id=f"c-{index}")],
            )
        assert len(engine.cache) <= engine.cache.stats().capacity

    def test_a_flood_of_failures_does_not_leak_breakers(self, engine) -> None:
        """One breaker per model, not per request."""
        from vision_os.adapters.understanding import ScriptedAnswer

        adapter = scripted(
            *[ScriptedAnswer(raise_unavailable=True) for _ in range(500)],
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        for index in range(100):
            e.understand(
                make_request(request_id=f"r-{index}", crop_id=f"c-{index}"),
                crops=[make_crop(crop_id=f"c-{index}")],
            )
        assert len(e._breakers) == 1  # noqa: SLF001

    def test_the_drift_window_stays_bounded(self, engine) -> None:
        """A rolling window, not every result ever seen."""
        for index in range(500):
            engine.understand(
                make_request(request_id=f"r-{index}", crop_id=f"c-{index}"),
                crops=[make_crop(crop_id=f"c-{index}")],
            )
        assert len(engine._rejection_window) <= engine._config.schema_drift_window  # noqa: SLF001

    def test_a_failure_flood_never_raises(self, engine) -> None:
        """The pipeline keeps running whatever the model does."""
        from vision_os.adapters.understanding import ScriptedAnswer

        adapter = scripted(
            *[
                ScriptedAnswer(raise_timeout=True) if i % 2 else ScriptedAnswer(refused=True)
                for i in range(400)
            ],
            producible=(POSTURE,),
        )
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[adapter],
        )
        for index in range(50):
            result = e.understand(
                make_request(request_id=f"r-{index}", crop_id=f"c-{index}"),
                crops=[make_crop(crop_id=f"c-{index}")],
            )
            assert result.attributes == (), "no failure may produce an attribute"


class TestConcurrency:
    def test_the_cache_is_thread_safe(self) -> None:
        from .unit.test_engine import build_result

        cache = ResponseCache(capacity=64)
        errors: list[Exception] = []

        def churn(offset: int) -> None:
            try:
                for index in range(300):
                    key = (f"t{offset}", f"c{index}", "p@1", "m@1", "posture")
                    cache.put(key, build_result(), _instant())
                    cache.get(key, _instant())
            except Exception as exc:  # noqa: BLE001 - recorded and re-raised
                errors.append(exc)

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        assert len(cache) <= 64

    def test_the_semaphore_cap_holds_under_contention(self) -> None:
        semaphore = ModelSemaphore(limit=2)
        observed: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(300):
                if semaphore.try_acquire():
                    with lock:
                        observed.append(semaphore.in_flight)
                    semaphore.release()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert max(observed) <= 2, (
            f"{max(observed)} calls in flight against a limit of 2; an uncapped "
            f"VLM starves the detector that gates the whole pipeline"
        )


def _instant():
    from vision_os.core.model.timebase import Instant

    return Instant(0)


class TestEvidenceIsAlwaysComplete:
    def test_every_result_carries_an_input_hash(self, engine) -> None:
        result = engine.understand(make_request(), crops=[make_crop()])
        assert result.evidence.input_hash

    def test_even_a_failure_carries_a_decision_path(self, engine) -> None:
        """A failure with no path is a failure nobody can diagnose."""
        e = build_engine(
            engine._clock, engine._metrics, engine._events, engine._config,
            understanders=[],
        )
        result = e.understand(make_request(), crops=[make_crop()])
        assert result.evidence.decision_path
        assert result.evidence.took(UnderstandingStep.NO_CAPABLE_MODEL)

    def test_a_result_without_a_crop_reference_is_refused(self) -> None:
        """A claim that cannot name the pixels behind it is not evidence."""
        from dataclasses import replace

        from .unit.test_engine import build_result

        base = build_result()
        with pytest.raises(ValueError, match="not evidence"):
            replace(
                base,
                object_id="obj-1",
                evidence=replace(base.evidence, crop_ref=None),
            )
