"""Composition root for Flow 8 — storage interfaces and the Observation API.

Assembles M13's evidence store and M14's API against an already-built platform.
Which implementation satisfies P22, P31 and P32 is decided **only here**, and
every one is gated through its conformance kit before use (obligation A7).

**The API is constructed with a read-only view of M12 and nothing else.** It
receives the state manager, an authorizer, an audit trail and — optionally — an
evidence store and a demand registry. It does not receive the observation log,
the builder, the registry, or any perception module. §M14's Dependencies name the
Vision State Manager, and giving L7 anything lower would let a query bypass the
layer that owns partitioning and consistency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .adapters.exposure import DenyAll, StaticAuthorizer
from .adapters.exposure.authorization import AllowReadsWithinTenant
from .adapters.persistence import (
    FileEvidenceStore,
    InMemoryEvidenceStore,
    NullEvidenceStore,
)
from .bootstrap import VisionPlatform
from .conformance import (
    API_TRANSPORT_KIT,
    AUTHORIZATION_KIT,
    EVIDENCE_STORE_KIT,
)
from .core.errors import ApiError, PersistenceError
from .core.model.api import CapabilitySummary
from .core.ports.exposure import (
    ApiLimits,
    ApiTransportPort,
    AuditSinkPort,
    AuthorizationPort,
    RateLimit,
)
from .core.ports.persistence import EvidenceQuota, EvidenceStorePort, RetentionPolicy
from .exposure import (
    AuditTrail,
    CountingAuditSink,
    DemandIntake,
    ObservationApi,
    SubscriptionHub,
)
from .exposure.demands import DemandStore
from .kernel.plugins.manifest import PortCatalogue
from .state import VisionStateManager

_AUTHORIZERS = {
    "authz.static": StaticAuthorizer,
    "authz.deny_all": DenyAll,
    "authz.tenant_reads": AllowReadsWithinTenant,
}


@dataclass(frozen=True, slots=True)
class ExposureLayer:
    """Everything Flow 8 assembled."""

    api: ObservationApi
    hub: SubscriptionHub
    audit: AuditTrail
    evidence: EvidenceStorePort
    retention: RetentionPolicy
    demands: DemandIntake | None = None
    transport: ApiTransportPort | None = None

    @property
    def authorizer_id(self) -> str:
        return self.api._authz.authorizer_id  # noqa: SLF001


def build_evidence_store(platform: VisionPlatform) -> EvidenceStorePort:
    """Select an evidence store by name.

    Raises:
        PersistenceError: the configured store is unknown. Refusing beats
            defaulting: a typo that silently fell back to ``evidence.memory``
            would give a site claiming 72-hour retention a store that forgets
            everything on restart, and nothing would say so.
    """
    settings = platform.config.storage()
    quota = EvidenceQuota(
        max_bytes=settings.evidence_max_bytes,
        max_blobs=settings.evidence_max_blobs,
        max_blob_bytes=settings.evidence_max_blob_bytes,
    )

    if settings.evidence_store == "evidence.memory":
        return InMemoryEvidenceStore(quota=quota)
    if settings.evidence_store == "evidence.file":
        return FileEvidenceStore(Path(settings.evidence_path), quota=quota)
    if settings.evidence_store == "evidence.null":
        return NullEvidenceStore()
    raise PersistenceError(
        f"unknown evidence store '{settings.evidence_store}'; known stores are "
        f"evidence.memory, evidence.file, evidence.null",
        requested=settings.evidence_store,
    )


def build_authorizer(
    platform: VisionPlatform, *, grants: Sequence = ()
) -> AuthorizationPort:
    """Select an authorization model by name.

    Raises:
        ApiError: the configured model is unknown. An unknown authorizer must
            never fall back to a permissive one — a defaulted authorization model
            is how a deployment ends up serving more than anybody intended.
    """
    name = platform.config.api().authorizer
    factory = _AUTHORIZERS.get(name)
    if factory is None:
        raise ApiError(
            f"unknown authorizer '{name}'; known models are {sorted(_AUTHORIZERS)}",
            requested=name,
        )
    if factory is StaticAuthorizer:
        return StaticAuthorizer(grants)
    return factory()


def build_exposure_layer(
    platform: VisionPlatform,
    state: VisionStateManager,
    *,
    authorizer: AuthorizationPort | None = None,
    evidence: EvidenceStorePort | None = None,
    audit_sinks: Sequence[AuditSinkPort] = (),
    demand_registry=None,
    demand_store: DemandStore | None = None,
    capabilities: CapabilitySummary | None = None,
    grants: Sequence = (),
) -> ExposureLayer:
    """Assemble M13's evidence store and M14's API.

    Args:
        state: The Vision State Manager. **The API's only window onto the
            platform** — everything it serves arrives through this object, which
            is what makes the L6/L7 split real rather than nominal.
        demand_registry: The registry M8 reads. Passed in rather than created,
            because 01_LAYERED §3.2 breaks the demand cycle by having both
            modules touch one record store: the API writes, the Crop Manager
            reads, and neither calls the other.

    Raises:
        ApiError: the API is disabled, or an adapter failed its kit.
    """
    settings = platform.config.api()
    storage = platform.config.storage()

    if not settings.enabled:
        raise ApiError(
            "api.enabled is false; a site that does not want to serve consumers "
            "should not build the layer rather than build one that refuses "
            "every request"
        )

    selected_evidence = (
        evidence if evidence is not None else build_evidence_store(platform)
    )
    selected_authorizer = (
        authorizer if authorizer is not None else build_authorizer(platform, grants=grants)
    )

    _gate(platform, PortCatalogue.EVIDENCE_STORE, EVIDENCE_STORE_KIT, selected_evidence)
    _gate(platform, PortCatalogue.AUTHORIZATION, AUTHORIZATION_KIT, selected_authorizer)

    audit = AuditTrail(
        clock=platform.clock,
        metrics=platform.metrics,
        sinks=tuple(audit_sinks) or (CountingAuditSink(capacity=settings.audit_capacity),),
    )
    hub = SubscriptionHub(clock=platform.clock, metrics=platform.metrics)

    demands = None
    if demand_registry is not None:
        demands = DemandIntake(
            clock=platform.clock,
            metrics=platform.metrics,
            registry=demand_registry,
            authorizer=selected_authorizer,
            audit=audit,
            store=demand_store or DemandStore(),
        )

    api = ObservationApi(
        clock=platform.clock,
        metrics=platform.metrics,
        state=state,
        authorizer=selected_authorizer,
        audit=audit,
        hub=hub,
        demands=demands,
        evidence=selected_evidence,
        limits=_limits_of(settings),
        capabilities=capabilities,
    )

    return ExposureLayer(
        api=api,
        hub=hub,
        audit=audit,
        evidence=selected_evidence,
        retention=RetentionPolicy(
            evidence_ttl_ms=storage.evidence_ttl_ms,
            raw_output_ttl_ms=storage.raw_output_ttl_ms,
        ),
        demands=demands,
    )


def build_transport(
    api: ObservationApi, *, recording: bool = False, streaming: bool = True
) -> ApiTransportPort:
    """Bind a transport over an assembled API.

    Separate from ``build_exposure_layer`` because §M14 makes the API
    transport-independent: an API with no transport is a complete, usable module,
    and a deployment may bind several at once.
    """
    from .adapters.exposure.transport import (
        InProcessTransport,
        RecordingTransport,
        routes_for,
    )

    routes = routes_for(api, include_streaming=streaming)
    factory = RecordingTransport if recording else InProcessTransport
    return factory(routes, supports_streaming=streaming)


def _limits_of(settings) -> ApiLimits:
    return ApiLimits(
        query=RateLimit(requests_per_minute=settings.queries_per_minute),
        evidence=RateLimit(requests_per_minute=settings.evidence_per_minute),
        subscribe=RateLimit(requests_per_minute=settings.subscribes_per_minute),
        max_page_size=settings.max_page_size,
        max_window_ms=settings.max_window_ms,
        max_subscriptions_per_principal=settings.max_subscriptions_per_principal,
    )


def _gate(platform: VisionPlatform, port, kit, adapter) -> None:
    """Run an adapter's conformance kit before it is ever used.

    For P22 this catches a store that collapses ``Expired`` into ``NotFound``,
    which would make retention indistinguishable from data loss for the life of
    the deployment. For P31 it catches an authorizer that could be talked across
    a tenant boundary.
    """
    registered = platform.conformance.get(port)
    effective = registered or kit
    report = effective.run(adapter, fast_only=True)
    if not report.passed:
        raise ApiError(
            f"adapter for {port} failed conformance: {'; '.join(report.failures)}",
            port=str(port),
        )


def gate_transport(platform: VisionPlatform, transport: ApiTransportPort) -> None:
    """Gate a transport after it has been bound to an API.

    Separate because a transport's routes reference the API, so it cannot be
    constructed before the API exists — but it must still pass its kit before a
    consumer reaches it.
    """
    _gate(platform, PortCatalogue.API_TRANSPORT, API_TRANSPORT_KIT, transport)
