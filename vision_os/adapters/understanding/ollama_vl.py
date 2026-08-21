"""P15 ``UnderstanderPort`` — a locally-served vision-language model.

The local sibling of ``NvidiaVisionUnderstander``. Both satisfy the same port
and produce the same registered attributes; nothing downstream can tell which
answered except by reading the provenance that says so.

The difference that matters is one field: this adapter declares
``data_residency="local"``. Crops never leave the machine, so a site with a
residency policy can bind this where it must refuse the hosted one. That choice
is made at composition time, in the open, by ``_bind`` — not discovered later.

Two honest declarations worth reading before trusting a number from here:

**``supports_structured_output=True``**, unlike the hosted adapter. Ollama's
``format: "json"`` genuinely constrains decoding, so schema conformance is
guaranteed rather than recovered from prose. The claim is different because the
capability is different, not because the adapters were written by different
hands.

**``deterministic=False``** even at temperature zero with a fixed seed. A CPU
VLM is reproducible *in practice* on identical input but not bit-exact across
builds or thread counts, and declaring ``True`` would make V13 a promise the
platform could not keep. Replay determinism is asserted over the observation log
instead.

Cost, measured on a CPU-only host: a cold model load is 138-210 s and a warm
call 5-20 s. ``warm()`` exists so the composition root can pay that once, up
front, where it is visible — rather than having the first crop of a live session
appear to hang for three minutes.
"""

from __future__ import annotations

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

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"

#: Self-reported and uncalibrated, labelled ``SELF_REPORTED`` by M9 (U4).
SELF_REPORTED_CONFIDENCE = 0.80


class OllamaVisionUnderstander:
    """A local VLM behind P15, over Ollama's generate endpoint."""

    __slots__ = ("_endpoint", "_id", "_keep_alive", "_lock", "_max_side", "_model",
                 "_producible", "_timeout", "binding_calls", "cold_start_ms", "stats")

    def __init__(
        self,
        *,
        producible: Sequence[AttributeKey],
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 180.0,
        keep_alive: str = "30m",
        max_side: int = 224,
        adapter_id: str = "understander.ollama_vl",
    ) -> None:
        if not producible:
            raise ValueError(
                "an understander must declare at least one producible attribute; "
                "one that can produce nothing can never be routed to"
            )
        self._id = adapter_id
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._keep_alive = keep_alive
        self._max_side = max_side
        self._producible = tuple(producible)
        self._lock = threading.Lock()
        self.stats = _LocalStats()
        self.cold_start_ms: float = 0.0
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
            supports_structured_output=True,
            supports_temporal=False,
            supports_batching=False,
            max_batch_size=1,
            max_output_tokens=512,
            cost_class=1.0,
            latency_p50_ms=self.stats.percentile(0.5),
            latency_p95_ms=self.stats.percentile(0.95),
            deterministic=False,
            data_residency="local",
        )

    def understand(self, request: UnderstandingPortRequest) -> UnderstandingPortResponse:
        """Answer one request. **Never fabricates (U2).**"""
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
            timeout = max(timeout, request.timeout.millis / 1000.0)

        payload = {
            "model": self._model,
            "prompt": request.prompt.text,
            "images": [image_b64],
            "stream": False,
            # Constrained decoding. This is what `supports_structured_output`
            # declares upward, and the reason coercion is nearly free here.
            "format": "json",
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": float(request.temperature),
                "seed": 42,
                "num_predict": min(int(request.max_tokens), request.prompt.max_output_tokens),
            },
        }

        started = time.perf_counter()
        try:
            body = self._post("/api/generate", payload, timeout=timeout)
        except TimeoutError as exc:
            with self._lock:
                self.stats.timed_out += 1
            return self._refusal(str(exc), started=started)
        except Exception as exc:  # noqa: BLE001 - a dead model is a reported outcome
            return self._refusal(f"{type(exc).__name__}: {exc}", started=started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_text = str(body.get("response", ""))
        raw_bytes = raw_text.encode("utf-8", errors="replace")

        with self._lock:
            self.stats.observe(
                latency_ms,
                prompt_tokens=int(body.get("prompt_eval_count") or 0),
                eval_tokens=int(body.get("eval_count") or 0),
            )

        decoded = extract_json(raw_text)
        if decoded is None:
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

        Sequential because Ollama serves one model instance; a dropped id is an
        answer nobody can distinguish from a lost one.
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

    # --- health, for the composition root ------------------------------------------ #

    def warm(self) -> dict[str, Any]:
        """Load the model before the first real crop arrives.

        The warm-up latency is recorded separately and **excluded from the
        reported percentiles**: folding a 200-second model load into a p50 would
        misreport steady-state performance by two orders of magnitude.
        """
        pixels = bytes(bytearray([28, 28, 28]) * (32 * 32))
        image_b64 = encode_png_base64(memoryview(pixels), 32, 32, max_side=32)
        payload = {
            "model": self._model,
            "prompt": 'Reply with JSON: {"ok": true}',
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0.0, "seed": 42, "num_predict": 8},
        }
        started = time.perf_counter()
        try:
            self._post("/api/generate", payload, timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - a cold model is a reported state
            return {
                "warmed": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        elapsed = (time.perf_counter() - started) * 1000.0
        self.cold_start_ms = elapsed
        return {"warmed": True, "elapsed_ms": elapsed, "error": None}

    def probe(self) -> dict[str, Any]:
        """Is the model installed and reachable? Reported, never assumed."""
        try:
            with urllib.request.urlopen(f"{self._endpoint}/api/tags", timeout=10) as response:
                tags = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "endpoint": self._endpoint,
                "model": self._model,
                "error": f"{type(exc).__name__}: {exc}",
            }

        installed = [str(entry.get("name", "")) for entry in (tags.get("models") or [])]
        if self._model not in installed:
            return {
                "available": False,
                "endpoint": self._endpoint,
                "model": self._model,
                "installed_models": installed,
                "error": f"model '{self._model}' is not installed",
            }
        return {
            "available": True,
            "endpoint": self._endpoint,
            "model": self._model,
            "installed_models": installed,
            "error": None,
        }

    # --- transport -------------------------------------------------------------------- #

    def _post(self, path: str, payload: dict, *, timeout: float) -> dict:
        request = urllib.request.Request(
            f"{self._endpoint}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise TimeoutError(f"inference exceeded {timeout:.0f}s") from exc
            raise
        except TimeoutError as exc:
            raise TimeoutError(f"inference exceeded {timeout:.0f}s") from exc

    # --- non-answers ------------------------------------------------------------------- #

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
            model_version=self._model.split(":")[-1] if ":" in self._model else "unknown",
            artifact_hash=f"ollama:{self._model}",
            adapter_id=self._id,
            deterministic=False,
        )


class _LocalStats:
    """Adapter-local counters for the model panel. Not a metrics system."""

    __slots__ = ("eval_tokens", "failed", "latencies", "prompt_tokens", "refused",
                 "requests", "succeeded", "timed_out", "unparseable")

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


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "SELF_REPORTED_CONFIDENCE",
    "OllamaVisionUnderstander",
]
