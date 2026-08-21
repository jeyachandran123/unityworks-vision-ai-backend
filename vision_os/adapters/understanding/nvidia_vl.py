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
DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

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

    __slots__ = ("failed", "latencies", "prompt_tokens", "refused", "requests",
                 "eval_tokens", "succeeded", "timed_out", "unparseable")

    def __init__(self) -> None:
        self.requests = 0
        self.succeeded = 0
        self.failed = 0
        self.refused = 0
        self.timed_out = 0
        self.unparseable = 0
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
                 "_producible", "_timeout", "binding_calls", "stats")

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
        self.binding_calls = 0

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
        return {
            "available": True,
            "endpoint": self._base,
            "model": self._model,
            "model_listed": self._model in available,
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

    def _refusal(self, reason: str, *, started: float | None = None) -> UnderstandingPortResponse:
        """An explicit non-answer. **Never a plausible default (U2).**"""
        with self._lock:
            self.stats.failed += 1
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
    "NvidiaVisionUnderstander",
    "SELF_REPORTED_CONFIDENCE",
]
