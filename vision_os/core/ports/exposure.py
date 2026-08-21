"""M14's ports — P31 `AuthorizationPort`, P32 `ApiTransportPort` (06_PORTS §2).

Two boundaries, both named by §M14's Extension Points, and both replaceable for
reasons the architecture states outright.

**P32 is transport-independent by design.** §M14:

> *"The contract is transport-independent by design — this is why
> `09_API_CONTRACTS.md` specifies semantics rather than a wire protocol, and why
> adopting a new transport in 2031 will not be a platform change."*

Nothing in this module or in `exposure/` mentions HTTP, gRPC, a socket or a
serialization format. The API answers questions; a transport adapter decides how
the question arrived and how the answer leaves.

**P31 exists because authorization models differ by deployment.** §M14 lists
*"role-based, attribute-based, per-camera and per-region scoping"*, and
12_SECURITY §5.2's permission tuple `(action, resource_scope, conditions)` admits
all of them. The platform enforces *that* a decision was made, never *how*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.api import Action, AuditRecord, Principal, Scope
from ..model.ids import CameraId


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Whether an action is permitted, and **on what narrowed scope**.

    The narrowed scope is the architecturally important half. 12_SECURITY §4.2
    forbids post-filtering:

    > *"A query that fetches broadly and filters afterwards leaks whenever the
    > filter has a bug, whenever an error path returns unfiltered data, whenever
    > pagination interacts badly, and whenever a new code path forgets to apply
    > it. Constructing the query already scoped means **there is no moment at
    > which cross-tenant data exists in memory to leak**."*

    So the port does not return a boolean for the caller to act on — it returns
    the scope the caller is permitted to query, and the caller queries *that*.
    A decision that granted access without narrowing would push the narrowing
    back to the consumer of the decision, which is where it gets forgotten.
    """

    granted: bool
    scope: Scope
    reason: str = ""
    conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.granted and not self.reason:
            raise ValueError(
                "a denial must carry a reason; an unexplained denial is "
                "indistinguishable from a bug to whoever is debugging it"
            )

    @property
    def denied(self) -> bool:
        return not self.granted


@runtime_checkable
class AuthorizationPort(Protocol):
    """P31 — decide what a principal may do, and on which scope.

    Implementations: RBAC, ABAC, per-camera scoping.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **Z1** | **Narrow, never post-filter.** A grant returns the scope the principal may see. 12_SECURITY §4.2: post-filtering *"is how leaks happen"*. |
    | **Z2** | **Deny across tenants, always.** A principal of tenant A asking about tenant B is denied regardless of any role, and the attempt is auditable. This is not configurable. |
    | **Z3** | **Separate `read_observations` from `read_evidence`** (12_SECURITY §5.3). Granting the first must never imply the second. |
    | **Z4** | **Deterministic.** The same principal, action and scope yield the same decision. A decision that varied between two calls would make an audit trail unfalsifiable. |
    | **Z5** | **Fail closed.** An adapter that cannot reach its policy source denies; it never defaults to permit. |
    | **Z6** | Every decision is explainable — a denial carries a reason. |
    """

    @property
    def authorizer_id(self) -> str:
        ...

    def authorize(
        self, principal: Principal, action: Action, scope: Scope
    ) -> AuthorizationDecision:
        """Decide, and narrow the scope to what is permitted (Z1).

        Never raises for a denial: a denial is an *answer*. Raising is reserved
        for an adapter that cannot decide at all, which under Z5 must be treated
        as a denial by the caller.
        """
        ...

    def visible_cameras(self, principal: Principal, scope: Scope) -> Sequence[CameraId]:
        """Which cameras this principal may read within the scope.

        Separate from ``authorize`` because §M14 requires per-camera and
        per-region scoping, and a query must be *constructed* against the
        permitted set rather than checked against it afterwards.
        """
        ...


@runtime_checkable
class AuditSinkPort(Protocol):
    """Where audit records go (§M14 responsibility 8, 12_SECURITY §8).

    Not in `06_PORTS`' catalogue of 32 as a named port. It is expressed as a
    Protocol rather than a concrete class for the same reason the catalogue's
    ports are: a regulated deployment writes audit to an append-only external
    system, and an embedded one writes to a file. Neither belongs in M14.

    Kept deliberately narrow — one method, no queries. An audit trail M14 could
    *read* would be an audit trail M14 could be asked to filter, and the module
    that generates records should not also serve them.
    """

    def record(self, entry: AuditRecord) -> None:
        """Persist one audit entry. Must not raise into the request path.

        An audit sink that failed a consumer's query would make observability a
        source of outages. Failures are counted and reported through health.
        """
        ...


@dataclass(frozen=True, slots=True)
class TransportRequest:
    """One inbound request, already authenticated by the transport.

    Authentication belongs to the transport because that is where the credential
    material lives — a bearer token, a client certificate, a message-bus identity.
    §M14 owns *authorization*; the transport establishes *who*.
    """

    principal: Principal
    operation: str
    payload: dict = field(default_factory=dict)
    accepted_major: int = 1
    request_id: str = ""
    purpose: str = ""
    """Declared purpose, required for evidence access (12_SECURITY §5.4)."""


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """One outbound answer, still in platform types.

    Deliberately **not** serialized. The transport adapter renders; the API
    answers. Returning bytes here would put a wire format inside the module
    §M14 requires be transport-independent.
    """

    result: object = None
    error: object = None
    version: int = 1

    @property
    def failed(self) -> bool:
        return self.error is not None


@runtime_checkable
class ApiTransportPort(Protocol):
    """P32 — carry requests and responses between consumers and M14.

    Implementations: synchronous request/response, streaming push, message-bus
    publication, webhook delivery, future protocols.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **T1** | **Authenticate.** Establish the principal, or reject. The API authorizes; the transport authenticates. |
    | **T2** | **Never interpret.** A transport translates shapes, never meanings. It must not filter results, aggregate them, or decide what a consumer should see. |
    | **T3** | **Negotiate version explicitly.** Carry the consumer's accepted major; never guess one (§M14). |
    | **T4** | **Backpressure is the transport's problem.** A slow consumer must not be able to stall the API; the declared `OverflowPolicy` is applied and a `Gap` emitted. |
    | **T5** | **Declare streaming support honestly.** A transport that cannot stream must say so, so a subscription request fails at setup rather than appearing to work. |
    | **T6** | Failure is a typed `ApiErrorView`, carrying `retryable` explicitly (09_API §8). |
    """

    @property
    def transport_id(self) -> str:
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether this transport can carry a subscription (T5).

        Declared rather than discovered: a consumer subscribing over a transport
        that cannot stream must be refused at setup. Discovering it later means
        the consumer believes it is subscribed and is receiving nothing, which is
        exactly the silence V8 exists to prevent.
        """
        ...

    def serve(self, request: TransportRequest) -> TransportResponse:
        """Dispatch one request to the API and return its answer."""
        ...


@dataclass(frozen=True, slots=True)
class RateLimit:
    """A budget for one class of request (§M14 responsibility 6).

    Evidence gets its own, tighter budget: §6 notes evidence payloads are *"large
    and sensitive"*, and a consumer that can pull imagery as fast as it can pull
    facts has effectively been granted bulk imagery export.
    """

    requests_per_minute: int = 600
    burst: int = 60

    def __post_init__(self) -> None:
        if self.requests_per_minute < 1 or self.burst < 1:
            raise ValueError("a rate limit must permit at least one request")


@dataclass(frozen=True, slots=True)
class ApiLimits:
    """The bounds M14 enforces (§M14 failure table).

    Every one of these exists so a single consumer cannot degrade the platform
    for everyone else — §M14: *"Reject with a bound and a cursor rather than
    degrading the service for everyone."*
    """

    query: RateLimit = field(default_factory=RateLimit)
    evidence: RateLimit = field(default_factory=lambda: RateLimit(60, 10))
    subscribe: RateLimit = field(default_factory=lambda: RateLimit(60, 10))
    max_page_size: int = 1_000
    max_window_ms: int = 86_400_000
    """24 hours. §2.2: ``max_span`` is *"enforced by policy"*."""

    max_subscriptions_per_principal: int = 32

    def __post_init__(self) -> None:
        if self.max_page_size < 1:
            raise ValueError("max_page_size must be positive")
        if self.max_window_ms < 1:
            raise ValueError("max_window_ms must be positive")


__all__ = [
    "ApiLimits",
    "ApiTransportPort",
    "AuditSinkPort",
    "AuthorizationDecision",
    "AuthorizationPort",
    "RateLimit",
    "TransportRequest",
    "TransportResponse",
]
