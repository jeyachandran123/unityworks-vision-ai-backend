"""Logging, metrics and the request id that ties them together.

### Two metric systems, kept distinguishable

Vision OS has its own `MetricsEngine` with its own stable names, exported through
`MetricsExportPort`. This module does **not** replace it and must never
re-instrument the platform. Application metrics are prefixed ``uwv_`` and
describe HTTP; platform metrics keep their own names and describe perception.

An operator reading a dashboard needs to know which layer a number came from,
and merging the namespaces would destroy that.

### Three logging rules, each of which was learned the hard way

1. **Never log a credential, a full RTSP URL, or an exception message that might
   quote one.** The redaction machinery in the CCTV source exists because a
   password in a log line is on disk and in a screen recording forever.
2. **Never swallow an exception without recording its type.** In Phase 6.7 a bare
   ``except: continue`` in diagnostic code reported 391 successful write-backs
   when every one had raised ``AttributeRejectedError``. The measurement was
   confidently wrong for four sub-phases.
3. **Log the decision, not just the outcome.** "skipped: fresh_enough, age 42s,
   validity 120s" is diagnosable; "skipped" is not.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.configuration.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request, Response

# ── Application metrics. Prefixed so they never collide with the platform's. ──

REQUESTS = Counter(
    "uwv_http_requests_total",
    "HTTP requests handled by the application.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "uwv_http_request_duration_seconds",
    "Wall time per request.",
    ["method", "path"],
)
AUTH_FAILURES = Counter(
    "uwv_auth_failures_total",
    "Rejected authentication attempts, by reason.",
    ["reason"],
)
AUTHZ_DENIALS = Counter(
    "uwv_authorization_denied_total",
    "Authorization denials, by the permission that was missing. "
    "Security-relevant: a spike here is either a misconfigured role or probing.",
    ["permission"],
)
EVIDENCE_ACCESS = Counter(
    "uwv_evidence_access_total",
    "Evidence retrievals. Every increment is an access to CCTV imagery of an "
    "identifiable person and is expected to be matched by an audit record.",
    ["outcome"],
)
VISION_READY = Gauge(
    "uwv_vision_os_ready",
    "1 when Vision OS is assembled in this process, 0 otherwise. "
    "Distinct from 'observed nothing' — see invariant V8.",
)


def configure_logging(settings: Settings) -> None:
    """Install the process logger. Idempotent."""
    import sys

    logger.remove()
    if settings.log_format == "json":
        logger.add(sys.stderr, level=settings.log_level, serialize=True, backtrace=False, diagnose=False)
    else:
        logger.add(sys.stderr, level=settings.log_level, backtrace=False, diagnose=False)

    # `diagnose=False` is not a style choice. With it enabled loguru renders local
    # variables into the traceback, which puts password hashes, tokens and secret
    # values into the log the moment anything raises near them.


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


async def request_context_middleware(request: Request, call_next):
    """Attach a request id, time the request, and count the outcome.

    The id goes into every error envelope and every log line for the request, so
    a user reporting "I got an error, the id was abc123" hands an engineer an
    exact index into the logs — without the error body having to carry any
    internal detail.
    """
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    request.state.request_id = request_id

    # The route template, never the concrete path: `/users/{id}` rather than
    # `/users/9f2…`. Concrete paths make the label cardinality unbounded, and a
    # metrics store with a million label values stops being queryable.
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path

    started = time.perf_counter()
    with logger.contextualize(request_id=request_id):
        response: Response = await call_next(request)

    elapsed = time.perf_counter() - started
    REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    response.headers["X-Request-Id"] = request_id
    return response


__all__ = [
    "AUTHZ_DENIALS",
    "AUTH_FAILURES",
    "EVIDENCE_ACCESS",
    "REQUESTS",
    "REQUEST_LATENCY",
    "VISION_READY",
    "configure_logging",
    "metrics_payload",
    "request_context_middleware",
]
