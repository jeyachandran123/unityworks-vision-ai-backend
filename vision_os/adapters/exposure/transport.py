"""P32 adapters — in-process transport.

> §M14: *"The contract is transport-independent by design — this is why
> `09_API_CONTRACTS.md` specifies semantics rather than a wire protocol, and why
> adopting a new transport in 2031 will not be a platform change."*

`InProcessTransport` carries a call across a function boundary. That sounds like a
no-op, and architecturally it is the point: if the API can be served by something
this thin, then nothing HTTP-shaped has leaked into it. The day an HTTP adapter is
written, it will implement the same three members and the API will not change.

**Obligation T2 — never interpret.** A transport translates shapes, not meanings.
Neither adapter here filters a result, aggregates one, or decides what a consumer
should see. Everything they do is dispatch and error rendering.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ...core.errors import ApiError, VisionOSError
from ...core.model.api import ApiErrorView
from ...core.ports.exposure import TransportRequest, TransportResponse


@dataclass(frozen=True, slots=True)
class Route:
    """One operation the transport can dispatch.

    ``streaming`` is declared per route so a transport that cannot stream refuses
    ``subscribe`` at setup rather than accepting it and delivering nothing —
    obligation T5, and V8 applied to integration.
    """

    operation: str
    handler: Callable[..., object]
    streaming: bool = False


class InProcessTransport:
    """``transport.in_process`` — dispatch without a wire.

    For an embedded deployment where the consumer is in the same process, and for
    every test that needs the API's real error rendering without a socket.
    """

    __slots__ = ("_calls", "_lock", "_routes", "_streaming")

    def __init__(
        self, routes: Sequence[Route] = (), *, supports_streaming: bool = True
    ) -> None:
        self._routes: dict[str, Route] = {r.operation: r for r in routes}
        self._streaming = supports_streaming
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def transport_id(self) -> str:
        return "transport.in_process"

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    def register(self, route: Route) -> None:
        self._routes[route.operation] = route

    def serve(self, request: TransportRequest) -> TransportResponse:
        """Dispatch one request. **Never raises** — every failure is rendered.

        A transport that let an exception escape would make the consumer's error
        handling depend on the platform's internal exception hierarchy, and 09_API
        §8's stable ``code`` exists precisely so it does not.
        """
        with self._lock:
            self._calls += 1

        route = self._routes.get(request.operation)
        if route is None:
            return TransportResponse(
                error=ApiErrorView(
                    code="NOT_FOUND",
                    message=f"unknown operation '{request.operation}'",
                    retryable=False,
                    request_id=request.request_id,
                ),
                version=request.accepted_major,
            )

        if route.streaming and not self._streaming:
            # T5. Refusing at setup beats appearing to work: a consumer that
            # believes it is subscribed and receives nothing is exactly the
            # silence V8 exists to prevent.
            return TransportResponse(
                error=ApiErrorView(
                    code="UNSUPPORTED_VERSION",
                    message=(
                        f"operation '{request.operation}' requires streaming and "
                        f"this transport does not support it"
                    ),
                    retryable=False,
                    request_id=request.request_id,
                ),
                version=request.accepted_major,
            )

        try:
            result = route.handler(request)
        except VisionOSError as exc:
            return TransportResponse(
                error=error_view(exc, request_id=request.request_id),
                version=request.accepted_major,
            )
        except Exception as exc:  # noqa: BLE001 - the transport is the last boundary
            return TransportResponse(
                error=ApiErrorView(
                    code="INTERNAL",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=False,
                    request_id=request.request_id,
                ),
                version=request.accepted_major,
            )

        return TransportResponse(result=result, version=request.accepted_major)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(sorted(self._routes))


class RecordingTransport(InProcessTransport):
    """``transport.recording`` — keeps every request and response.

    For integration tests that need to assert what a consumer would have
    received, and for a deployment debugging an integration. Bounded, because an
    unbounded recorder in a long-running process grows fastest exactly when the
    platform is busiest.
    """

    __slots__ = ("_capacity", "_exchanges")

    def __init__(
        self,
        routes: Sequence[Route] = (),
        *,
        supports_streaming: bool = True,
        capacity: int = 500,
    ) -> None:
        super().__init__(routes, supports_streaming=supports_streaming)
        self._exchanges: list[tuple[TransportRequest, TransportResponse]] = []
        self._capacity = capacity

    @property
    def transport_id(self) -> str:
        return "transport.recording"

    def serve(self, request: TransportRequest) -> TransportResponse:
        response = super().serve(request)
        self._exchanges.append((request, response))
        if len(self._exchanges) > self._capacity:
            del self._exchanges[: -self._capacity]
        return response

    @property
    def exchanges(self) -> tuple[tuple[TransportRequest, TransportResponse], ...]:
        return tuple(self._exchanges)

    @property
    def failures(self) -> tuple[TransportResponse, ...]:
        return tuple(response for _, response in self._exchanges if response.failed)


def error_view(error: VisionOSError, *, request_id: str = "") -> ApiErrorView:
    """Render a platform error as the consumer's envelope (09_API §8).

    ``retryable`` comes from the error's ``failure_class`` and is written into the
    payload as a field. §8: *"Consumers must never infer retryability from a
    status code or a message string. Inferring it is how retry storms begin."*
    """
    code = getattr(error, "code", None) or _fallback_code(error)
    return ApiErrorView(
        code=code,
        message=error.message,
        retryable=error.retryable,
        retry_after=None,
        details=dict(error.context),
        request_id=request_id,
    )


def _fallback_code(error: VisionOSError) -> str:
    """A stable code for an error that predates the API's taxonomy.

    Derived from the class name rather than from the message, because §8 requires
    codes be *"stable, machine-readable, never reworded"* while messages *"may
    change"*.
    """
    name = type(error).__name__
    trimmed = name[:-5] if name.endswith("Error") else name
    out: list[str] = []
    for index, character in enumerate(trimmed):
        if character.isupper() and index:
            out.append("_")
        out.append(character.upper())
    return "".join(out)


@dataclass(frozen=True, slots=True)
class TransportReport:
    """What an operator needs to know about a transport."""

    transport_id: str
    calls: int = 0
    failures: int = 0
    operations: tuple[str, ...] = field(default_factory=tuple)
    streaming: bool = True

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0


#: Selectable by name. Closed, like every factory table in the platform.
TRANSPORT_FACTORIES: Mapping[str, object] = {
    "transport.in_process": InProcessTransport,
    "transport.recording": RecordingTransport,
}


def routes_for(api, *, include_streaming: bool = True) -> tuple[Route, ...]:
    """The standard route table over an `ObservationApi`.

    Built here rather than inside the API so that the API never learns it is
    being served over anything. Adding a transport means adding a table like this
    one; it never means touching M14.
    """
    table = [
        Route("query_state", lambda r: api.query_state(r.principal, **r.payload)),
        Route(
            "query_observations",
            lambda r: api.query_observations(r.principal, **r.payload),
        ),
        Route("get_object", lambda r: api.get_object(r.principal, **r.payload)),
        Route("coverage", lambda r: api.coverage(r.principal, **r.payload)),
        Route("capabilities", lambda r: api.capabilities(r.principal, **r.payload)),
        Route(
            "get_evidence",
            lambda r: api.get_evidence(r.principal, purpose=r.purpose, **r.payload),
        ),
        Route(
            "register_demand",
            lambda r: api.demands.register(r.principal, **r.payload),
        ),
        Route("revoke_demand", lambda r: api.demands.revoke(r.principal, **r.payload)),
        Route("list_demands", lambda r: api.demands.list_for(r.principal)),
    ]
    if include_streaming:
        table.append(
            Route(
                "subscribe",
                lambda r: api.subscribe(r.principal, **r.payload),
                streaming=True,
            )
        )
    return tuple(table)


def apply_error(exc: ApiError) -> ApiErrorView:
    """Convenience for a caller holding an already-typed API error."""
    return error_view(exc)
