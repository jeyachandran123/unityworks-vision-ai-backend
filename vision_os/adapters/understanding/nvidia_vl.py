"""P15 ``UnderstanderPort`` — a hosted vision-language model.

The sibling of ``ScriptedUnderstander`` and ``StaticAttributeHead``, and 06_PORTS
is explicit about why that matters: the platform is *"genuinely indifferent to
whether a 7-billion-parameter generalist or a 2-megabyte specialist answered."*
Nothing above this file knows this adapter exists; the composition root binds it.

### What this adapter does not decide

It does not decide **what to ask** — the ``RenderedPrompt`` arrives rendered and
version-pinned by P17. It does not decide **what may come back** — the
``OutputSchema`` arrives with the request. It does not decide **what becomes an
attribute** — ``AttributeValidator`` does, against the registry, after this
returns. There is no domain vocabulary anywhere in this file, and none may be
added: an adapter that knew what a hairnet was would have to be edited for every
new deployment, which is the coupling the port exists to prevent.

Its whole job is: pixels + a question in, structured fields + evidence out.

### The field that gates deployment

``data_residency="remote"``. Crops leave the machine and are answered by a
service on the internet. ``UnderstanderCapabilities.is_remote`` derives from it,
and a site with a residency policy refuses this binding at composition time — in
the open — rather than discovering the export in an audit months later. The local
adapters declare ``"local"`` and nothing about them changes.

### What travels

Only crop pixels and the rendered prompt. ``CropView`` deliberately carries no
object id, no track id and no tenant — 12_SECURITY's point that an adapter is
handed *"pixels and a question, never a subject"* — so there is nothing
identifying here to leak even by accident.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from ...core.model.ids import AttributeKey, ModelId
from ...core.model.understanding import CostEstimate, ModelMeta, Timing
from ...core.ports.understanding import (
    UnderstanderCapabilities,
    UnderstandingPortRequest,
    UnderstandingPortResponse,
)
from .payload import encode_png_base64, extract_json, split_by_schema

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: The **default layer only** — the base of `defaults -> file -> environment`.
#:
#: A hosted model identifier is deployment configuration, and this constant is
#: the last resort for a deployment that names none. Set ``VISION_NVIDIA_MODEL``
#: to choose; the value here is not authoritative and must never be the only
#: place a model can be named.
#:
#: It previously read ``nvidia/llama-3.1-nemotron-nano-vl-8b-v1``, which NVIDIA
#: retired on **2026-08-26T09:00:00Z**. Every call then returned ``410 Gone``,
#: every crop became a refusal, no attribute was produced, and the product
#: reported "no alerts" for eighteen hours — on a safety monitor, where "nothing
#: to report" and "the analysis is dead" look identical from the outside. That
#: is why `probe()` exists below and why a 410 is now reported as a retirement
#: rather than counted as one more failed call.
DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"

#: Statuses that mean *the model is gone*, not *this call failed*.
#:
#: 410 is the documented one. 404 is included because a delisted model on this
#: endpoint answers "Not found for account" once it stops being served, which is
#: the same operator problem wearing a different number: no amount of retrying
#: will fix it, and a deployment must be told to name a different model.
MODEL_RETIRED_STATUSES = frozenset({404, 410})

#: The account is over quota for this model. Retryable in principle, and
#: therefore NOT a retirement — but see `health()` for why a *sustained* one is
#: still an operator problem rather than weather.
RATE_LIMIT_STATUS = 429

#: How many recent calls `health()` judges the adapter on.
#:
#: 32 so that a recovered quota is reflected within about half a minute of live
#: traffic rather than after a restart, and so a single unlucky call cannot move
#: the verdict by more than ~3 percentage points.
HEALTH_WINDOW = 32

#: Below this many observed calls `health()` declines to judge at all.
#:
#: A cold adapter has answered nothing, and "0 successes out of 0" is not
#: evidence of ill health. Ten is the point at which a 50% floor stops being a
#: coin flip.
HEALTH_MIN_SAMPLES = 10

#: Recent success fraction below which the analysis is reported as unavailable.
#:
#: **Measured, not chosen.** On 2026-08-31 the same API key, endpoint, crops and
#: prompt produced:
#:
#:   minimaxai/minimax-m3                 51/1609 live (3.2%), 1/16 controlled
#:   meta/llama-3.2-11b-vision-instruct   16/16 controlled (100%)
#:
#: A floor of 0.5 sits an order of magnitude above the failing configuration and
#: far below the working one, so it separates them without being fitted to
#: either. It is also the point past which the product is misleading on its own
#: terms: when more than half of crops yield no evidence, an empty Alerts page
#: says more about the quota than about the kitchen.
HEALTH_MIN_SUCCESS = 0.5


class ModelRetiredError(RuntimeError):
    """The configured model no longer exists upstream.

    Separate from every other transport failure because the operator response is
    different in kind. A timeout, a 429 or a 503 will pass; **this will not.**
    Retrying it forever is what turned a model retirement into eighteen hours of
    silent "no alerts", so it is raised under its own name, latched on the
    adapter, and reported by `health()` as an unavailable analysis rather than
    counted as one more failed call.
    """

    def __init__(self, status: int, detail: str, model: str) -> None:
        super().__init__(f"model '{model}' is no longer available (HTTP {status}): {detail}")
        self.status = status
        self.detail = detail
        self.model = model


class RateLimitedError(RuntimeError):
    """The endpoint refused this call for quota, not for content.

    Its own type rather than a string in a generic `RuntimeError` because two
    layers need to tell it apart from every other failure and neither should be
    matching on message text: `understand()` records it as a distinct outcome,
    and `health()` reports a sustained run of it as `rate_limited` — an operator
    problem with an operator fix (raise the quota, or name a model this account
    can actually serve), not a flaky network.

    It is deliberately **not** a `ModelRetiredError`. The model exists and the
    endpoint lists it; nothing about the configuration is misspelled. Latching
    it permanently would leave a deployment dark after its quota reset.
    """

    def __init__(self, detail: str, model: str) -> None:
        super().__init__(f"model '{model}' is rate limited (HTTP {RATE_LIMIT_STATUS}): {detail}")
        self.status = RATE_LIMIT_STATUS
        self.detail = detail
        self.model = model


#: Self-reported and uncalibrated. Surfaced through ``field_confidence`` so M9
#: can label it ``SELF_REPORTED`` (U4). This endpoint returns no per-field
#: probability, and inventing a spread per field would be fabrication dressed as
#: precision — one conservative value for everything the model answered is the
#: honest shape.
SELF_REPORTED_CONFIDENCE = 0.80


class _Stats:
    """Adapter-local counters, for the model panel. Not a metrics system.

    The platform's own understanding metrics are emitted by M9 around every
    invocation; these exist so an operator can see per-adapter totals without
    joining across a metrics backend, and they are deliberately not authoritative.
    """

    __slots__ = ("failed", "latencies", "prompt_tokens", "rate_limited", "refused",
                 "requests", "eval_tokens", "succeeded", "timed_out", "unparseable")

    def __init__(self) -> None:
        self.requests = 0
        self.succeeded = 0
        self.failed = 0
        self.refused = 0
        self.timed_out = 0
        self.unparseable = 0
        #: Counted apart from `failed` (which still includes it) because "the
        #: kitchen was quiet" and "we were over quota" are the two readings an
        #: empty Alerts page has, and only this number tells them apart.
        self.rate_limited = 0
        self.prompt_tokens = 0
        self.eval_tokens = 0
        self.latencies: list[float] = []

    def observe(self, latency_ms: float, *, prompt_tokens: int = 0, eval_tokens: int = 0) -> None:
        self.latencies.append(latency_ms)
        if len(self.latencies) > 512:
            del self.latencies[0]
        self.prompt_tokens += prompt_tokens
        self.eval_tokens += eval_tokens

    def percentile(self, fraction: float) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def to_wire(self) -> dict[str, Any]:
        """The shape the model panel reads.

        Named for the wire rather than for Python because that is what the
        established stats object in this codebase was called, and the route
        reading it does `getattr(understander, "stats", None).to_wire()`.
        Renaming it broke that route with a 500 while every unit test passed —
        the seam had no test because it crosses two repositories.
        """
        return {
            "requests": self.requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "refused": self.refused,
            "timed_out": self.timed_out,
            "unparseable": self.unparseable,
            # Added, never renamed — see the note above. The console reads this
            # mapping by key, so a new one is invisible to an old client and
            # available to a new one.
            "rate_limited": self.rate_limited,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "p50_latency_ms": self.percentile(0.5),
            "p95_latency_ms": self.percentile(0.95),
        }


class NvidiaVisionUnderstander:
    """A hosted VLM behind P15, over an OpenAI-compatible endpoint.

    Transport is ``urllib`` rather than a client library: an adapter that
    required a new wheel to satisfy a core path would make the platform harder
    to deploy than the model it wraps.
    """

    __slots__ = ("_base", "_id", "_key", "_lock", "_max_side", "_model",
                 "_producible", "_recent", "_timeout", "binding_calls", "retired", "stats")

    def __init__(
        self,
        *,
        producible: Sequence[AttributeKey],
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 60.0,
        max_side: int = 448,
        adapter_id: str = "understander.nvidia_vl",
    ) -> None:
        if not producible:
            raise ValueError(
                "an understander must declare at least one producible attribute; "
                "one that can produce nothing can never be routed to"
            )
        if not api_key:
            raise ValueError(
                "the NVIDIA understander requires an API key. It is read from "
                "configuration by the composition root and never from this module"
            )
        self._id = adapter_id
        self._base = base_url.rstrip("/")
        self._model = model
        self._key = api_key
        self._timeout = timeout_s
        self._max_side = max_side
        self._producible = tuple(producible)
        self._lock = threading.Lock()
        self.stats = _Stats()
        #: Recent per-call outcomes, newest last: "ok" | "rate_limited" | "failed".
        #:
        #: A bounded window rather than the lifetime totals in `stats`, because
        #: health is a question about *now*. An adapter that answered ten
        #: thousand crops yesterday and is refusing every one today is not
        #: healthy, and any ratio taken over all time would say it was.
        self._recent: deque[str] = deque(maxlen=HEALTH_WINDOW)
        self.binding_calls = 0
        #: Latched detail once the model is known to be gone, else empty.
        #:
        #: Latched rather than recomputed because a retirement is permanent and
        #: the operator question — "is my analysis dead?" — must be answerable
        #: between calls, not only during one.
        self.retired = ""

    # --- port surface ----------------------------------------------------------- #

    @property
    def adapter_id(self) -> str:
        return self._id

    def capabilities(self) -> UnderstanderCapabilities:
        return UnderstanderCapabilities(
            producible_attributes=self._producible,
            model_id=ModelId(self._model),
            max_crops_per_request=1,
            min_resolution=(16, 16),
            max_resolution=(4096, 4096),
            colour_space="bgr24",
            # Declared False deliberately. The endpoint is OpenAI-compatible but
            # this model does not honour a JSON response format, so conformance
            # is *recovered* from the text rather than guaranteed by decoding.
            # Declaring True would tell M9 to trust a shape nobody enforced.
            supports_structured_output=False,
            supports_temporal=False,
            supports_batching=False,
            max_batch_size=1,
            max_output_tokens=512,
            cost_class=1.0,
            latency_p50_ms=self.stats.percentile(0.5),
            latency_p95_ms=self.stats.percentile(0.95),
            # A hosted model behind a load balancer is not reproducible even at
            # temperature zero. V13 asserts replay determinism over the
            # observation log instead, and claiming otherwise here would make
            # that guarantee one the platform could not keep.
            deterministic=False,
            data_residency="remote",
        )

    def understand(self, request: UnderstandingPortRequest) -> UnderstandingPortResponse:
        """Answer one request. **Never fabricates (U2).**

        Every failure path — encoding, transport, HTTP status, timeout, empty
        body, unparseable text — returns an explicit result with ``structured``
        empty. None of them returns a plausible value.
        """
        with self._lock:
            self.stats.requests += 1

        crop = request.crops[0]
        try:
            image_b64 = encode_png_base64(
                crop.pixels,
                crop.width,
                crop.height,
                colour_space=crop.colour_space,
                max_side=self._max_side,
            )
        except Exception as exc:  # noqa: BLE001
            return self._refusal(f"crop encoding failed: {type(exc).__name__}: {exc}")

        timeout = self._timeout
        if request.timeout is not None:
            # The caller's budget wins when it is more generous; M8 already
            # decided this crop was worth the wait.
            timeout = max(timeout, request.timeout.millis / 1000.0)

        started = time.perf_counter()
        try:
            body = self._chat(
                request.prompt.text,
                image_b64,
                max_tokens=min(int(request.max_tokens), request.prompt.max_output_tokens),
                temperature=float(request.temperature),
                timeout=timeout,
            )
        except TimeoutError as exc:
            with self._lock:
                self.stats.timed_out += 1
            return self._refusal(str(exc), started=started)
        except RateLimitedError as exc:
            # Downstream this is a refusal like any other — U2 holds, and a
            # quota error must never become an attribute. What changes is only
            # what an operator is told: `health()` can now say the analysis is
            # rate limited instead of reporting "ok" while every crop dies.
            with self._lock:
                self.stats.rate_limited += 1
            return self._refusal(str(exc), started=started, outcome="rate_limited")
        except ModelRetiredError as exc:
            # Latched, and phrased so the reason reads as a configuration fault
            # rather than a flaky call. Downstream this is still a refusal — it
            # must never become an answer — but `health()` can now tell an
            # operator that the analysis is unavailable rather than quiet.
            with self._lock:
                self.retired = exc.detail or str(exc)
            return self._refusal(f"model retired: {exc}", started=started)
        except Exception as exc:  # noqa: BLE001 - a refusing service is an outcome
            return self._refusal(f"{type(exc).__name__}: {exc}", started=started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = _content_of(body)
        raw_bytes = raw_text.encode("utf-8", errors="replace")
        usage = body.get("usage") or {}

        with self._lock:
            self.stats.observe(
                latency_ms,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                eval_tokens=int(usage.get("completion_tokens") or 0),
            )

        if not raw_text.strip():
            with self._lock:
                self.stats.unparseable += 1
                # Not "ok": the transport worked but no field survived, so this
                # produced exactly as much evidence as a refusal did.
                self._recent.append("failed")
            return UnderstandingPortResponse(
                structured={},
                unparsed="",
                raw_output=raw_bytes,
                model_meta=self._meta(),
                timing=self._timing(latency_ms),
            )

        decoded = extract_json(raw_text)
        if decoded is None:
            # U3 — the bytes are kept as evidence and M9 decides. An answer that
            # did not parse is not an answer that said nothing.
            with self._lock:
                self.stats.unparseable += 1
                # Not "ok": the transport worked but no field survived, so this
                # produced exactly as much evidence as a refusal did.
                self._recent.append("failed")
            return UnderstandingPortResponse(
                structured={},
                unparsed=raw_text,
                raw_output=raw_bytes,
                model_meta=self._meta(),
                timing=self._timing(latency_ms),
            )

        structured, leftover = split_by_schema(decoded, request.output_schema)

        with self._lock:
            self.stats.succeeded += 1
            self._recent.append("ok")

        return UnderstandingPortResponse(
            structured=structured,
            unparsed=leftover,
            field_confidence=(
                {key: SELF_REPORTED_CONFIDENCE for key in structured} or None
            ),
            raw_output=raw_bytes,
            model_meta=self._meta(),
            timing=self._timing(latency_ms),
        )

    def understand_batch(
        self, requests: Sequence[UnderstandingPortRequest]
    ) -> Mapping[Any, UnderstandingPortResponse]:
        """Answer many. **Every request id appears in the result.**

        Sequential because the endpoint takes one image per call. A dropped id
        is an answer nobody can distinguish from a lost one, so the mapping is
        total even when every entry is a refusal.
        """
        return {request.request_id: self.understand(request) for request in requests}

    def estimate_cost(self, request: UnderstandingPortRequest) -> CostEstimate:
        """What this would cost, before spending it (U7)."""
        return CostEstimate(
            cost_units=1.0 * len(request.crops),
            model_id=ModelId(self._model),
            attributes_covered=tuple(request.output_schema.fields),
            estimated_latency=None,
        )

    # --- health, for the composition root ----------------------------------------- #

    def health(self) -> dict[str, Any]:
        """Is this adapter able to answer at all? For diagnostics and health.

        Exists so the product can distinguish **"zero violations"** from **"the
        analysis is not running"**. On a safety monitor those look identical
        from the outside — an empty Alerts page — and for eighteen hours they
        were, after the configured model was retired upstream.

        ### Why a retirement was not enough

        That first fix caught only 404 and 410. On 2026-08-31 the same silence
        returned wearing a 429: `minimaxai/minimax-m3` was listed by `/models`,
        so `probe()` passed, and the account was then rate limited to **51 of
        1,609 crops (3.2%)**. Detection, tracking, cropping, the registry, the
        rules and the alert path were all measured working — 5,477 people
        detected, 2,720 crops cut — and the product still reported nothing,
        because 96.8% of the evidence died at inference while this method
        answered `"ok"`.

        So health is no longer a single latched flag. A **sustained** run of
        unanswered crops is reported as unavailable whatever caused it, and a
        rate limit is named because its fix is an operator action rather than a
        wait. Transient failure is deliberately not reported: see
        `HEALTH_WINDOW` and `HEALTH_MIN_SAMPLES`.

        This changes what is *reported*, never what is *produced*. A refusal was
        already a refusal and never became an attribute (U2); the bug was that
        nobody was told.

        `reason` is safe to display: it carries the upstream explanation and the
        model name, and never the key.
        """
        if self.retired:
            return {
                "available": False,
                "state": "model_retired",
                "model": self._model,
                "reason": f"the model '{self._model}' is no longer available upstream: "
                          f"{self.retired}",
            }

        with self._lock:
            recent = list(self._recent)

        # Too early to judge. Reporting "degraded" on a cold adapter would make
        # every restart look like an outage.
        if len(recent) < HEALTH_MIN_SAMPLES:
            return {"available": True, "state": "ok", "model": self._model, "reason": ""}

        successes = sum(1 for outcome in recent if outcome == "ok")
        rate = successes / len(recent)
        if rate >= HEALTH_MIN_SUCCESS:
            return {"available": True, "state": "ok", "model": self._model, "reason": ""}

        # Below the floor. Which kind of failure decides the wording, because
        # the operator's next action differs: a quota needs raising or the model
        # needs changing, whereas a mixed failure needs looking at.
        limited = sum(1 for outcome in recent if outcome == "rate_limited")
        percent = f"{rate * 100:.0f}%"
        if limited > (len(recent) - successes) / 2:
            return {
                "available": False,
                "state": "rate_limited",
                "model": self._model,
                "reason": (
                    f"the model service is rate limiting this account: only "
                    f"{successes} of the last {len(recent)} crops were answered "
                    f"({percent}). No PPE attribute is being produced for the rest, "
                    f"so an empty Alerts page reflects the quota and not the scene. "
                    f"Raise the quota for '{self._model}' or name a model this "
                    f"account can serve."
                ),
            }
        return {
            "available": False,
            "state": "failing",
            "model": self._model,
            "reason": (
                f"only {successes} of the last {len(recent)} crops were answered "
                f"({percent}); no PPE attribute is being produced for the rest, so "
                f"an empty Alerts page reflects the analysis and not the scene."
            ),
        }

    def probe(self) -> dict[str, Any]:
        """Reachable and authorised? Reported, never assumed.

        Called at binding so an expired key is found there rather than on the
        first crop of a live session.
        """
        probe = urllib.request.Request(
            f"{self._base}/models",
            headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(probe, timeout=10) as response:
                payload = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "endpoint": self._base,
                "model": self._model,
                "error": f"{type(exc).__name__}: {exc}",
            }
        available = [str(entry.get("id", "")) for entry in (payload.get("data") or [])]
        listed = self._model in available
        if not listed:
            # Found at binding, where someone is watching, rather than on the
            # first crop of a live shift. The endpoint lists what it serves, so
            # a model missing from that list is retired or misspelled — both are
            # configuration faults and neither improves by waiting.
            with self._lock:
                self.retired = f"not listed by {self._base}/models"
        return {
            "available": True,
            "endpoint": self._base,
            "model": self._model,
            "model_listed": listed,
            "error": None,
        }

    # --- transport ------------------------------------------------------------------ #

    def _chat(
        self,
        prompt: str,
        image_b64: str,
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        call = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(call, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            # A retirement is not a failed call — it is a permanently wrong
            # configuration, and nothing downstream can recover from it.
            if exc.code in MODEL_RETIRED_STATUSES:
                raise ModelRetiredError(exc.code, detail, self._model) from exc
            # Quota, not content. Its own type so `health()` can report a
            # sustained run of it without parsing this message — see
            # RateLimitedError for why it is not treated as a retirement.
            if exc.code == RATE_LIMIT_STATUS:
                raise RateLimitedError(detail, self._model) from exc
            # The status is named rather than folded into a generic failure: a
            # 429 is retryable and a 401 is not, and an operator reading a log
            # needs to tell those apart without a packet capture.
            raise RuntimeError(f"HTTP {exc.code} from the model service: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise TimeoutError(f"inference exceeded {timeout:.0f}s") from exc
            raise
        except TimeoutError as exc:
            raise TimeoutError(f"inference exceeded {timeout:.0f}s") from exc

    # --- non-answers ------------------------------------------------------------------ #

    def _refusal(
        self,
        reason: str,
        *,
        started: float | None = None,
        outcome: str = "failed",
    ) -> UnderstandingPortResponse:
        """An explicit non-answer. **Never a plausible default (U2).**

        ``outcome`` records *why* in the health window without changing what is
        returned. Every refusal is equally a non-answer to M9; the distinction
        exists solely so an operator can be told whether to wait or to act.
        """
        with self._lock:
            self.stats.failed += 1
            self._recent.append(outcome)
        elapsed = (time.perf_counter() - started) * 1000.0 if started else 0.0
        return UnderstandingPortResponse(
            structured={},
            unparsed=None,
            raw_output=b"",
            model_meta=self._meta(),
            timing=self._timing(elapsed),
            refused=True,
            refusal_reason=reason,
        )

    def _timing(self, latency_ms: float) -> Timing:
        return Timing(inference_ms=latency_ms, total_ms=latency_ms, batch_size=1)

    def _meta(self) -> ModelMeta:
        return ModelMeta(
            model_id=ModelId(self._model),
            model_version=self._model.rsplit("-", 1)[-1] or "unknown",
            artifact_hash=f"nvidia:{self._model}",
            adapter_id=self._id,
            deterministic=False,
        )


# --- reading an OpenAI-compatible envelope -------------------------------------------- #
#
# Schema discipline, JSON recovery and crop encoding live in `payload.py`, shared
# with the local adapter. What stays here is the one thing that is specific to
# this vendor's wire format: where the answer sits in the response body.


def _content_of(body: Mapping[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "HEALTH_MIN_SAMPLES",
    "HEALTH_MIN_SUCCESS",
    "HEALTH_WINDOW",
    "ModelRetiredError",
    "NvidiaVisionUnderstander",
    "RATE_LIMIT_STATUS",
    "RateLimitedError",
    "SELF_REPORTED_CONFIDENCE",
]
