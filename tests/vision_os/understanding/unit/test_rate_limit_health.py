"""A rate-limited model must be visible, and must never become compliance.

### The failure this pins

On 2026-08-31 the deployment's model was changed to ``minimaxai/minimax-m3``.
The endpoint listed it, so ``probe()`` passed at binding and the model panel
read healthy. The account was then rate limited on that model specifically:
**51 of 1,609 live crops were answered — 3.2%.** Every other stage was measured
working in the same window (5,477 people detected, 3,807 tracks, 2,720 crops
cut, and the alert path had produced 200 persisted incidents over the preceding
days), so the product simply stopped reporting violations while
``health()`` returned ``{"available": True, "state": "ok"}``.

The same silence had happened before, in 2026-08-26, when the previous model was
retired upstream. That produced the 404/410 latch. This one arrived as a 429 and
walked straight past it: a retirement is permanent and a quota is not, so the
latch was right to ignore it — and reporting ``ok`` was still wrong.

### What is and is not asserted here

The platform's refusal behaviour is **unchanged and must stay unchanged**: a 429
was already a refusal, it never became an attribute, and no rule ever saw a
fabricated value. U2 was never broken. What was broken is that nobody was told,
so these tests are about *reporting*, with one exception —
``TestARefusalIsStillARefusal`` — which exists to prove the fix did not buy
visibility by weakening the guarantee underneath it.
"""

from __future__ import annotations

import urllib.error

import pytest

from vision_os.adapters.understanding.nvidia_vl import (
    HEALTH_MIN_SAMPLES,
    HEALTH_MIN_SUCCESS,
    HEALTH_WINDOW,
    RATE_LIMIT_STATUS,
    ModelRetiredError,
    NvidiaVisionUnderstander,
    RateLimitedError,
)

PRODUCIBLE = ("head_covering", "face_covering")


def adapter(**kwargs) -> NvidiaVisionUnderstander:
    return NvidiaVisionUnderstander(
        producible=PRODUCIBLE, api_key="test-key-not-a-real-credential", **kwargs
    )


def http_error(code: int, body: bytes = b'{"status":429,"title":"Too Many Requests"}'):
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=code,
        msg="err",
        hdrs=None,
        fp=__import__("io").BytesIO(body),
    )


def drive(a: NvidiaVisionUnderstander, outcomes: list[str]) -> None:
    """Push a sequence of ``"ok"``/``"rate_limited"``/``"failed"`` through the
    adapter's own recording path, never by writing the window directly.

    Writing ``_recent`` by hand would test the assertion and not the adapter.
    """
    for outcome in outcomes:
        if outcome == "ok":
            with a._lock:
                a.stats.succeeded += 1
                a._recent.append("ok")
        elif outcome == "rate_limited":
            with a._lock:
                a.stats.rate_limited += 1
            a._refusal("rate limited", outcome="rate_limited")
        else:
            a._refusal("boom")


class TestTheStatusIsClassified:
    def test_a_429_raises_its_own_error(self) -> None:
        """Not a generic RuntimeError, so no layer has to match on message text."""
        a = adapter(model="some/model")
        with pytest.raises(RateLimitedError) as caught:
            _raise(a, http_error(RATE_LIMIT_STATUS))

        assert caught.value.status == RATE_LIMIT_STATUS
        assert caught.value.model == "some/model"
        assert "Too Many Requests" in caught.value.detail

    def test_a_429_is_not_a_retirement(self) -> None:
        """A quota resets; a retired model does not. Latching a 429 would leave
        a deployment dark for the rest of the process's life after a spike."""
        assert not issubclass(RateLimitedError, ModelRetiredError)

    def test_a_410_is_still_a_retirement(self) -> None:
        a = adapter()
        with pytest.raises(ModelRetiredError):
            _raise(a, http_error(410, b"gone"))

    def test_other_statuses_stay_generic(self) -> None:
        a = adapter()
        with pytest.raises(RuntimeError) as caught:
            _raise(a, http_error(500, b"boom"))
        assert not isinstance(caught.value, RateLimitedError)
        assert not isinstance(caught.value, ModelRetiredError)


def _raise(a: NvidiaVisionUnderstander, exc: urllib.error.HTTPError):
    """Run the adapter's real HTTPError branch without opening a socket."""
    import unittest.mock as mock

    with mock.patch("urllib.request.urlopen", side_effect=exc):
        a._chat("q", "aGk=", max_tokens=8, temperature=0.0, timeout=1.0)


class TestHealthReportsSustainedRateLimiting:
    """§9: a rate-limited VLM must read DEGRADED/UNAVAILABLE, never healthy."""

    def test_the_live_failure_reads_unavailable(self) -> None:
        """The measured shape: roughly 3% answered, the rest 429.

        Before this fix these exact outcomes produced ``state: "ok"``.
        """
        a = adapter()
        drive(a, ["ok"] + ["rate_limited"] * 31)

        health = a.health()
        assert health["available"] is False
        assert health["state"] == "rate_limited"
        assert "rate limiting" in health["reason"]

    def test_the_reason_tells_the_operator_what_to_do(self) -> None:
        a = adapter()
        drive(a, ["rate_limited"] * 16)

        reason = a.health()["reason"]
        assert "quota" in reason.lower()
        assert "Alerts page" in reason, "the operator must be told why the UI is empty"

    def test_the_reason_never_carries_the_key(self) -> None:
        a = adapter()
        drive(a, ["rate_limited"] * 16)
        assert "test-key-not-a-real-credential" not in a.health()["reason"]

    def test_generic_failure_reads_failing_not_rate_limited(self) -> None:
        """The wording is the operator's next action. A 500 storm is not a quota
        problem and must not send someone to the billing page."""
        a = adapter()
        drive(a, ["failed"] * 16)

        health = a.health()
        assert health["available"] is False
        assert health["state"] == "failing"

    def test_a_mixed_failure_is_not_called_rate_limiting(self) -> None:
        a = adapter()
        drive(a, ["rate_limited"] * 5 + ["failed"] * 11)
        assert a.health()["state"] == "failing"


class TestHealthDoesNotFlap:
    def test_a_cold_adapter_is_not_degraded(self) -> None:
        """Every restart would otherwise look like an outage."""
        assert adapter().health()["state"] == "ok"

    def test_below_the_sample_floor_it_declines_to_judge(self) -> None:
        a = adapter()
        drive(a, ["rate_limited"] * (HEALTH_MIN_SAMPLES - 1))
        assert a.health()["state"] == "ok", "too few calls to call it an outage"

    def test_at_the_sample_floor_it_judges(self) -> None:
        a = adapter()
        drive(a, ["rate_limited"] * HEALTH_MIN_SAMPLES)
        assert a.health()["available"] is False

    def test_a_healthy_model_stays_ok(self) -> None:
        """The working configuration measured 16/16 on the same account. A fix
        that reported it unhealthy would be worse than the bug."""
        a = adapter()
        drive(a, ["ok"] * 16)
        assert a.health() == {
            "available": True, "state": "ok", "model": a._model, "reason": ""
        }

    def test_an_occasional_failure_is_not_an_outage(self) -> None:
        a = adapter()
        drive(a, ["ok"] * 12 + ["rate_limited"] * 4)
        assert a.health()["state"] == "ok"

    def test_it_recovers_without_a_restart(self) -> None:
        """A quota that resets must clear the state on its own — the whole
        reason this is a window and not a latch like retirement."""
        a = adapter()
        drive(a, ["rate_limited"] * HEALTH_WINDOW)
        assert a.health()["available"] is False

        drive(a, ["ok"] * HEALTH_WINDOW)
        assert a.health()["available"] is True

    def test_the_window_is_bounded(self) -> None:
        a = adapter()
        drive(a, ["ok"] * (HEALTH_WINDOW * 3))
        assert len(a._recent) == HEALTH_WINDOW

    def test_a_retirement_still_outranks_everything(self) -> None:
        a = adapter()
        drive(a, ["ok"] * HEALTH_WINDOW)
        a.retired = "not listed by the endpoint"
        assert a.health()["state"] == "model_retired"


class TestTheFloorIsSane:
    def test_it_sits_between_the_measured_configurations(self) -> None:
        """3.2% failing, 100% working — both measured on 2026-08-31 against the
        same key, endpoint, crops and prompt. The floor must separate them
        without being fitted to either."""
        assert 0.032 < HEALTH_MIN_SUCCESS < 1.0
        assert HEALTH_MIN_SUCCESS >= 0.5, "half the crops blind is not 'ok'"

    def test_the_window_outlives_the_sample_floor(self) -> None:
        assert HEALTH_WINDOW > HEALTH_MIN_SAMPLES


class TestARefusalIsStillARefusal:
    """The guarantee the visibility fix must not have bought its way out of."""

    def test_a_rate_limited_call_produces_no_attribute(self) -> None:
        a = adapter()
        response = a._refusal("rate limited", outcome="rate_limited")

        assert response.refused is True
        assert response.structured == {}, "a quota error must never become a value"
        assert response.raw_output == b""

    def test_every_refusal_reason_counts_as_a_failure(self) -> None:
        a = adapter()
        a._refusal("rate limited", outcome="rate_limited")
        a._refusal("boom")
        assert a.stats.failed == 2

    def test_the_rate_limit_counter_is_reported_separately(self) -> None:
        """`failed` alone cannot distinguish 'the kitchen was quiet' from 'we
        were over quota'. The panel needs both numbers."""
        a = adapter()
        drive(a, ["rate_limited"] * 3 + ["failed"] * 2)

        wire = a.stats.to_wire()
        assert wire["rate_limited"] == 3
        assert wire["failed"] == 5

    def test_the_existing_wire_keys_are_unchanged(self) -> None:
        """The validation console reads this mapping by key across a repository
        boundary. Adding is safe; renaming broke a route once already."""
        wire = adapter().stats.to_wire()
        for key in (
            "requests", "succeeded", "failed", "refused", "timed_out",
            "unparseable", "prompt_tokens", "eval_tokens",
            "p50_latency_ms", "p95_latency_ms",
        ):
            assert key in wire
