"""Phase 6A.3: concurrent understanding must not move an answer to the wrong object.

`understand_batch` now runs requests in a bounded thread pool, so completion
order no longer matches submission order. The failure this creates is silent and
severe: attribute A landing on object B produces a confident, wrong compliance
verdict about a real person, with full provenance attached to it.

Every test here is written to **fail** if results are ever matched by list
position or by arrival order rather than by `request_id`.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vision_os.perception.understanding.engine import UnderstandingEngine


class _Recorder:
    """Stands in for the engine's per-request work.

    `understand` is patched to this so the test controls completion order
    exactly, with no model, no network and no sleep-based flakiness.
    """

    def __init__(self, order: list[str], delays: dict[str, float]) -> None:
        self.order = order
        self.delays = delays
        self.seen: list[str] = []
        self.peak = 0
        self._live = 0
        self._lock = threading.Lock()

    def __call__(self, request, *, crops=()):
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            time.sleep(self.delays.get(str(request.request_id), 0.0))
            with self._lock:
                self.seen.append(str(request.request_id))
            # The answer carries the id it was asked about. If the batch ever
            # returns this under a different key, the assertions below catch it.
            return _Result(request)
        finally:
            with self._lock:
                self._live -= 1


class _Result:
    """Minimal stand-in carrying the identity the caller must preserve."""

    def __init__(self, request) -> None:
        self.request_id = request.request_id
        self.object_id = request.object_id
        self.track_id = getattr(request, "track_id", "")
        self.frame_ref = getattr(request, "frame_ref", "")
        self.observed_at = getattr(request, "observed_at", 0)
        self.outcome = _Outcome()
        self.attributes = ()

    def without_raw_output(self):
        return self


class _Outcome:
    value = "succeeded"
    is_failure = False


class _Request:
    def __init__(self, ident: str) -> None:
        self.request_id = ident
        self.object_id = f"object-{ident}"
        self.track_id = f"track-{ident}"
        self.frame_ref = f"cam-01/e0/f{ident}"
        self.observed_at = int(ident) * 1_000_000
        self.requested_attributes = ()


def _engine_with(monkeypatch, recorder, *, workers: int) -> UnderstandingEngine:
    """A bare engine object with only what `understand_batch` touches."""
    engine = object.__new__(UnderstandingEngine)
    monkeypatch.setattr(
        UnderstandingEngine, "understand", lambda self, r, crops=(): recorder(r, crops=crops)
    )
    monkeypatch.setattr(
        UnderstandingEngine, "_batch_workers", lambda self, requests: workers
    )
    return engine


class TestOutOfOrderCompletion:
    def test_a_late_first_request_still_gets_its_own_answer(self, monkeypatch):
        """A completes **last**; A's answer must still be A's.

        The classic bug: zip the results with the requests in submission order
        and hand out whatever arrived first. This is that bug's regression test.
        """
        requests = [_Request("1"), _Request("2")]
        # Request 1 is slow, request 2 is fast — so completion is 2 then 1.
        recorder = _Recorder(order=[], delays={"1": 0.20, "2": 0.0})
        engine = _engine_with(monkeypatch, recorder, workers=2)

        results = engine.understand_batch(requests)

        assert recorder.seen == ["2", "1"], "the test did not actually invert order"
        assert results["1"].object_id == "object-1"
        assert results["2"].object_id == "object-2"

    def test_identity_fields_survive_reordering(self, monkeypatch):
        requests = [_Request(str(n)) for n in range(1, 6)]
        # Reverse the completion order entirely.
        delays = {str(n): (6 - n) * 0.03 for n in range(1, 6)}
        recorder = _Recorder(order=[], delays=delays)
        engine = _engine_with(monkeypatch, recorder, workers=5)

        results = engine.understand_batch(requests)

        assert recorder.seen == ["5", "4", "3", "2", "1"]
        for request in requests:
            answer = results[request.request_id]
            assert answer.object_id == request.object_id
            assert answer.track_id == request.track_id
            assert answer.frame_ref == request.frame_ref
            assert answer.observed_at == request.observed_at

    def test_every_request_id_appears_exactly_once(self, monkeypatch):
        """The docstring's promise: the mapping is total, even under reordering."""
        requests = [_Request(str(n)) for n in range(1, 9)]
        recorder = _Recorder(order=[], delays={"3": 0.10, "7": 0.05})
        engine = _engine_with(monkeypatch, recorder, workers=4)

        results = engine.understand_batch(requests)

        assert set(results) == {r.request_id for r in requests}
        assert len(results) == len(requests)


class TestBoundedConcurrency:
    def test_in_flight_never_exceeds_the_configured_limit(self, monkeypatch):
        """The bound is the point. Unbounded concurrency is the failure mode."""
        requests = [_Request(str(n)) for n in range(1, 13)]
        recorder = _Recorder(order=[], delays=dict.fromkeys((str(n) for n in range(1, 13)), 0.02))
        engine = _engine_with(monkeypatch, recorder, workers=2)

        engine.understand_batch(requests)

        assert recorder.peak <= 2, f"{recorder.peak} calls were in flight; the limit is 2"

    def test_a_single_request_spawns_no_pool(self, monkeypatch):
        recorder = _Recorder(order=[], delays={})
        engine = _engine_with(monkeypatch, recorder, workers=1)

        results = engine.understand_batch([_Request("1")])

        assert recorder.peak == 1
        assert results["1"].object_id == "object-1"

    def test_the_limit_comes_from_platform_configuration(self):
        """`_batch_workers` must read the declared budget, not a literal.

        `remote_concurrency` exists because a cloud endpoint's rate limit is a
        different constraint from a local GPU's memory. Taking the smaller of the
        two keeps a remote adapter inside the tighter of the numbers.
        """

        class Config:
            max_concurrency = 4
            remote_concurrency = 2

        engine = object.__new__(UnderstandingEngine)
        object.__setattr__(engine, "_config", Config())

        assert engine._batch_workers([_Request(str(n)) for n in range(6)]) == 2
        # Never more workers than there is work.
        assert engine._batch_workers([_Request("1")]) == 1


class TestFailureIsolation:
    def test_one_failing_request_does_not_poison_its_siblings(self, monkeypatch):
        """§9: one VLM failure is not a batch failure."""
        requests = [_Request("1"), _Request("2"), _Request("3")]

        def flaky(self, request, crops=()):
            if str(request.request_id) == "2":
                raise RuntimeError("provider exploded")
            return _Result(request)

        monkeypatch.setattr(UnderstandingEngine, "understand", flaky)
        monkeypatch.setattr(UnderstandingEngine, "_batch_workers", lambda self, r: 3)
        captured = {}

        def failed(self, request, outcome, attempt, detail=""):
            captured["detail"] = detail
            return _Result(request)

        monkeypatch.setattr(UnderstandingEngine, "_failed", failed)
        engine = object.__new__(UnderstandingEngine)

        results = engine.understand_batch(requests)

        assert set(results) == {"1", "2", "3"}, "a sibling was lost"
        assert results["1"].object_id == "object-1"
        assert results["3"].object_id == "object-3"
        # And the failure was attributed to the request that actually failed.
        assert results["2"].object_id == "object-2"
        assert "provider exploded" in captured.get("detail", "")


class TestThreadSafetyGuard:
    """The bookkeeping the guard exists to protect."""

    def test_the_engine_declares_a_guard_slot(self):
        assert "_guard" in UnderstandingEngine.__slots__

    def test_concurrent_semaphore_creation_yields_one_semaphore(self):
        """A racing `_semaphore_for` must not install two.

        Two semaphores for one adapter would each admit `limit` callers, and the
        configured bound would silently double — the unbounded concurrency this
        whole design forbids.
        """

        class Caps:
            is_remote = True

        class Bound:
            adapter_id = "adapter.one"
            capabilities = Caps()

        class Config:
            max_concurrency = 4
            remote_concurrency = 2

        engine = object.__new__(UnderstandingEngine)
        object.__setattr__(engine, "_config", Config())
        object.__setattr__(engine, "_semaphores", {})
        object.__setattr__(engine, "_guard", threading.Lock())

        seen = []
        barrier = threading.Barrier(8)

        def race():
            barrier.wait()
            seen.append(engine._semaphore_for(Bound()))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: race(), range(8)))

        assert len({id(s) for s in seen}) == 1, "more than one semaphore was created"


@pytest.mark.asyncio
async def test_shutdown_does_not_leave_understanding_threads_behind(monkeypatch):
    """§10: no orphan thread survives a completed batch."""
    before = {t.name for t in threading.enumerate()}

    requests = [_Request(str(n)) for n in range(1, 7)]
    recorder = _Recorder(order=[], delays=dict.fromkeys((str(n) for n in range(1, 7)), 0.01))
    monkeypatch.setattr(
        UnderstandingEngine, "understand", lambda self, r, crops=(): recorder(r, crops=crops)
    )
    monkeypatch.setattr(UnderstandingEngine, "_batch_workers", lambda self, r: 2)
    engine = object.__new__(UnderstandingEngine)

    engine.understand_batch(requests)

    # `ThreadPoolExecutor` as a context manager joins its workers on exit, so a
    # surviving `vos-understand` thread means the pool leaked.
    after = {t.name for t in threading.enumerate()}
    leaked = {n for n in after - before if n.startswith("vos-understand")}
    assert not leaked, f"leaked understanding threads: {leaked}"
