"""Conformance kits for P22, P31 and P32.

Each check guards something that fails **silently** otherwise:

``evidence/expired_is_not_not_found``
    §M13: *"Collapsing these two is how retention behaviour becomes
    indistinguishable from data loss."* An adapter that reported both as missing
    would make a working retention policy look like a storage bug forever.

``evidence/never_persist_stores_nothing``
    12_SECURITY §2.3's no-evidence mode. An adapter that stored it anyway voids a
    privacy guarantee, and nothing downstream would ever detect it.

``evidence/erasure_leaves_a_tombstone``
    07_STATE §8.2. Without the tombstone, an erasure and a silent data loss are
    indistinguishable in the audit trail.

``authz/denies_across_tenants``
    12_SECURITY §4.1's boundary is absolute. An adapter that could be configured
    to cross it would make isolation advisory.

``authz/narrows_rather_than_filters``
    §4.2: post-filtering *"is how leaks happen"*. An adapter granting an unnarrowed
    scope pushes the narrowing to its caller, which is where it gets forgotten.

**What these kits cannot check:** whether an authorization model encodes the
*right* policy, or whether a transport's wire format is correct. Both need a
deployment's judgment. The kits verify contracts — the structural properties whose
violation is silent.
"""

from __future__ import annotations

from ..core.model.api import Action, Principal, Scope
from ..core.model.crop import PrivacyClass, RetentionMode
from ..core.model.ids import BlobRef, CameraId, ObjectId, TenantId
from ..core.model.timebase import Instant
from ..core.ports.exposure import TransportRequest
from ..core.ports.persistence import EraseScope, EvidenceStatus, hash_payload
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_TENANT = TenantId("kit-tenant")
_OTHER_TENANT = TenantId("kit-other-tenant")
_CAMERA = CameraId("kit-cam")
_OBJECT = ObjectId("kit-object")
_PAYLOAD = b"kit-evidence-payload"
_HASH = hash_payload(_PAYLOAD)


# --- P22 EvidenceStorePort -------------------------------------------------- #


def _check_evidence_shape(adapter) -> None:
    assert hasattr(adapter, "store_id"), "an evidence store must expose store_id"
    assert isinstance(adapter.store_id, str) and adapter.store_id


def _check_evidence_round_trips(adapter) -> None:
    """Obligation E1. A store that cannot return what it stored is not a store."""
    reference = adapter.put(
        _HASH,
        _PAYLOAD,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(10_000_000_000),
        camera_id=_CAMERA,
        object_id=_OBJECT,
    )
    fetched = adapter.get(reference, tenant_id=_TENANT)
    if adapter.store_id == "evidence.null":
        assert fetched.status is EvidenceStatus.NOT_FOUND, (
            "a store declaring itself null must report NOT_FOUND, not fabricate a hit"
        )
        return
    assert fetched.available, f"stored evidence was not retrievable: {fetched.detail}"
    assert fetched.payload == _PAYLOAD, "the payload changed in storage"


def _check_evidence_deduplicates(adapter) -> None:
    """Obligation E1. The same bytes stored twice occupy one slot."""
    if adapter.store_id == "evidence.null":
        return
    payload = b"kit-dedup"
    digest = hash_payload(payload)
    first = adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(10_000_000_000),
    )
    second = adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(10_000_000_000),
    )
    assert first == second, "the same content produced two different references"
    assert adapter.exists(digest, tenant_id=_TENANT)


def _check_never_persist_stores_nothing(adapter) -> None:
    """Obligation E3. 12_SECURITY §2.3's no-evidence mode is a hard guarantee.

    A crop marked ``never_persist`` must not touch storage. An adapter that wrote
    it — even briefly, even encrypted — has broken a promise the deployment made
    to whoever authorized the cameras.
    """
    payload = b"kit-must-never-be-stored"
    digest = hash_payload(payload)
    adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.NEVER_PERSIST,
        privacy_class=PrivacyClass.C1_IMAGERY,
    )
    assert not adapter.exists(digest, tenant_id=_TENANT), (
        "a never_persist blob was stored; 12_SECURITY §2.3 requires it exist in "
        "memory for one inference and nowhere else"
    )
    fetched = adapter.get(BlobRef(f"sha256:{digest}"), tenant_id=_TENANT)
    assert not fetched.available, "a never_persist blob was retrievable"


def _check_missing_is_not_found(adapter) -> None:
    """Obligation E2, first half. Never stored means NOT_FOUND."""
    fetched = adapter.get(BlobRef("sha256:kit-never-written"), tenant_id=_TENANT)
    assert fetched.status is EvidenceStatus.NOT_FOUND, (
        f"an unstored blob reported {fetched.status.value} rather than NOT_FOUND; "
        f"a caller cannot tell a bug from normal retention"
    )
    assert fetched.payload is None


def _check_expired_is_not_not_found(adapter) -> None:
    """Obligation E2, second half. Retention removing a blob is **normal**.

    The check runs expiry and then asserts the store no longer holds the blob.
    An adapter conflating the two states makes a working platform look broken.
    """
    if adapter.store_id == "evidence.null":
        return
    payload = b"kit-expiring"
    digest = hash_payload(payload)
    adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(1_000),
    )
    report = adapter.expire(before=Instant(2_000))
    assert report.removed >= 1, (
        "expiry removed nothing despite an expired blob; a store whose retention "
        "silently does not run looks identical to one working correctly until the "
        "disk fills"
    )
    assert not adapter.exists(digest, tenant_id=_TENANT)


def _check_erasure_leaves_a_tombstone(adapter) -> None:
    """Obligation E5. 07_STATE §8.2: the content goes, the record stays."""
    if adapter.store_id == "evidence.null":
        return
    payload = b"kit-to-erase"
    digest = hash_payload(payload)
    adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(10_000_000_000),
        object_id=_OBJECT,
    )
    report = adapter.erase(
        EraseScope(tenant_id=_TENANT, object_ids=(_OBJECT,), authority="kit")
    )
    assert report.erased >= 1, "erasure removed nothing"
    assert report.tombstones, (
        "erasure left no tombstone; 07_STATE §8.2 requires the record that an "
        "observation existed and was erased survive, or the audit trail cannot "
        "distinguish erasure from data loss"
    )
    fetched = adapter.get(BlobRef(f"sha256:{digest}"), tenant_id=_TENANT)
    assert fetched.status is EvidenceStatus.ERASED, (
        f"an erased blob reported {fetched.status.value}; ERASED and EXPIRED "
        f"answer different audit questions"
    )


def _check_evidence_is_tenant_scoped(adapter) -> None:
    """Obligation E7. 12_SECURITY §4.1's boundary applies to storage too."""
    if adapter.store_id == "evidence.null":
        return
    payload = b"kit-tenant-scoped"
    digest = hash_payload(payload)
    adapter.put(
        digest,
        payload,
        tenant_id=_TENANT,
        retention=RetentionMode.EVIDENCE,
        privacy_class=PrivacyClass.C1_IMAGERY,
        expires_at=Instant(10_000_000_000),
    )
    assert not adapter.exists(digest, tenant_id=_OTHER_TENANT), (
        "one tenant's evidence was visible to another by content hash; imagery "
        "with identical bytes must not become a cross-tenant channel"
    )
    fetched = adapter.get(BlobRef(f"sha256:{digest}"), tenant_id=_OTHER_TENANT)
    assert not fetched.available, "another tenant retrieved stored evidence"


def _check_erasure_refuses_an_unscoped_request(adapter) -> None:
    """An 'erase everything' call is a deployment decision, not an API call."""
    try:
        EraseScope(tenant_id=_TENANT)
    except ValueError:
        return
    raise AssertionError(
        "an erasure scope naming no objects, cameras or window was accepted"
    )


EVIDENCE_STORE_KIT = ConformanceKit(
    port_id=PortCatalogue.EVIDENCE_STORE,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_evidence_shape),
        ConformanceCheck(
            "round_trips", KitSection.SEMANTICS, _check_evidence_round_trips, obligation="E1"
        ),
        ConformanceCheck(
            "deduplicates", KitSection.SEMANTICS, _check_evidence_deduplicates, obligation="E1"
        ),
        ConformanceCheck(
            "never_persist_stores_nothing",
            KitSection.SEMANTICS,
            _check_never_persist_stores_nothing,
            obligation="E3",
        ),
        ConformanceCheck(
            "missing_is_not_found",
            KitSection.FAILURE,
            _check_missing_is_not_found,
            obligation="E2",
        ),
        ConformanceCheck(
            "expired_is_not_not_found",
            KitSection.FAILURE,
            _check_expired_is_not_not_found,
            obligation="E2",
        ),
        ConformanceCheck(
            "erasure_leaves_a_tombstone",
            KitSection.SEMANTICS,
            _check_erasure_leaves_a_tombstone,
            obligation="E5",
        ),
        ConformanceCheck(
            "is_tenant_scoped",
            KitSection.SEMANTICS,
            _check_evidence_is_tenant_scoped,
            obligation="E7",
        ),
        ConformanceCheck(
            "erasure_refuses_unscoped",
            KitSection.FAILURE,
            _check_erasure_refuses_an_unscoped_request,
        ),
    ),
)


# --- P31 AuthorizationPort -------------------------------------------------- #


def _principal(subject: str = "kit-consumer", tenant: TenantId = _TENANT) -> Principal:
    return Principal(subject=subject, tenant_id=tenant)


def _check_authz_shape(adapter) -> None:
    assert hasattr(adapter, "authorizer_id")
    assert isinstance(adapter.authorizer_id, str) and adapter.authorizer_id


def _check_authz_denies_across_tenants(adapter) -> None:
    """Obligation Z2. Not configurable, not overridable, not a role."""
    decision = adapter.authorize(
        _principal(tenant=_TENANT),
        Action.READ_STATE,
        Scope(tenant_id=_OTHER_TENANT),
    )
    assert decision.denied, (
        "a principal of one tenant was permitted a scope in another; "
        "12_SECURITY §4.1 makes the isolation boundary absolute"
    )
    assert decision.reason, "a denial must explain itself"


def _check_authz_narrows_rather_than_filters(adapter) -> None:
    """Obligation Z1. A granted decision must never **widen** the request.

    The universal property, checkable whatever policy an adapter encodes: if a
    caller asked about one camera, the scope it is handed back must not name two.
    An adapter that widened would hand the API a query reaching data the consumer
    never asked for and may not hold — and §4.2 names *"whenever a new code path
    forgets to apply it"* as one of four ways that leaks.
    """
    requested = (_CAMERA,)
    scope = Scope(tenant_id=_TENANT, camera_ids=requested)
    decision = adapter.authorize(_principal(), Action.READ_STATE, scope)

    assert decision.scope.tenant_id == _TENANT, (
        "a decision returned a scope in a different tenant"
    )
    if not decision.granted:
        return

    assert set(decision.scope.camera_ids) <= set(requested), (
        f"a grant widened the requested scope from {requested} to "
        f"{decision.scope.camera_ids}; authorization may narrow, never expand"
    )


def _check_authz_grant_scope_is_queryable(adapter) -> None:
    """Obligation Z1, the other half: a grant must return a *usable* scope.

    An adapter that granted access and returned an empty scope would be denying
    by another name — the API would query nothing and the consumer would read the
    empty result as an empty scene rather than as a permission problem (V8).
    """
    scope = Scope(tenant_id=_TENANT)
    decision = adapter.authorize(_principal(), Action.READ_STATE, scope)
    if not decision.granted:
        return
    assert decision.scope.tenant_id, (
        "a granted decision returned a scope with no tenant, which no query can "
        "be constructed from"
    )


def _check_authz_denial_carries_a_reason(adapter) -> None:
    """Obligation Z6. An unexplained denial is indistinguishable from a bug."""
    decision = adapter.authorize(
        _principal(subject="kit-unknown-principal"),
        Action.READ_EVIDENCE,
        Scope(tenant_id=_TENANT),
    )
    if decision.denied:
        assert decision.reason, "a denial carried no reason"


def _check_authz_is_deterministic(adapter) -> None:
    """Obligation Z4. A decision that varied would make an audit unfalsifiable."""
    scope = Scope(tenant_id=_TENANT, camera_ids=(_CAMERA,))
    first = adapter.authorize(_principal(), Action.READ_STATE, scope)
    second = adapter.authorize(_principal(), Action.READ_STATE, scope)
    assert first.granted == second.granted, (
        "two identical authorization requests produced different answers; an "
        "audit trail over a non-deterministic authorizer proves nothing"
    )
    assert first.scope == second.scope


def _check_authz_fails_closed(adapter) -> None:
    """Obligation Z5. An unknown principal is denied, never defaulted."""
    decision = adapter.authorize(
        _principal(subject="kit-never-registered"),
        Action.READ_EVIDENCE,
        Scope(tenant_id=_TENANT),
    )
    assert decision.denied, (
        "an unknown principal was granted evidence access; an authorizer that "
        "defaults to permit turns a configuration gap into a breach"
    )


def _check_authz_visible_cameras_is_scoped(adapter) -> None:
    cameras = adapter.visible_cameras(
        _principal(tenant=_TENANT), Scope(tenant_id=_OTHER_TENANT)
    )
    assert not cameras, "cameras were listed for a scope in another tenant"


AUTHORIZATION_KIT = ConformanceKit(
    port_id=PortCatalogue.AUTHORIZATION,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_authz_shape),
        ConformanceCheck(
            "denies_across_tenants",
            KitSection.SEMANTICS,
            _check_authz_denies_across_tenants,
            obligation="Z2",
        ),
        ConformanceCheck(
            "narrows_rather_than_filters",
            KitSection.SEMANTICS,
            _check_authz_narrows_rather_than_filters,
            obligation="Z1",
        ),
        ConformanceCheck(
            "grant_scope_is_queryable",
            KitSection.SEMANTICS,
            _check_authz_grant_scope_is_queryable,
            obligation="Z1",
        ),
        ConformanceCheck(
            "is_deterministic",
            KitSection.SEMANTICS,
            _check_authz_is_deterministic,
            obligation="Z4",
        ),
        ConformanceCheck(
            "fails_closed", KitSection.FAILURE, _check_authz_fails_closed, obligation="Z5"
        ),
        ConformanceCheck(
            "denial_carries_a_reason",
            KitSection.FAILURE,
            _check_authz_denial_carries_a_reason,
            obligation="Z6",
        ),
        ConformanceCheck(
            "visible_cameras_is_scoped",
            KitSection.SEMANTICS,
            _check_authz_visible_cameras_is_scoped,
            obligation="Z2",
        ),
    ),
)


# --- P32 ApiTransportPort --------------------------------------------------- #


def _check_transport_shape(adapter) -> None:
    assert hasattr(adapter, "transport_id")
    assert isinstance(adapter.transport_id, str) and adapter.transport_id
    assert isinstance(adapter.supports_streaming, bool), (
        "a transport must declare streaming support; discovering it later means "
        "a consumer believes it is subscribed and is receiving nothing (T5)"
    )


def _check_transport_never_raises(adapter) -> None:
    """Obligation T6. Every failure is a rendered error, not an exception.

    A transport that let an exception escape would make consumer error handling
    depend on the platform's internal exception hierarchy — which is exactly what
    09_API §8's stable ``code`` exists to prevent.
    """
    response = adapter.serve(
        TransportRequest(principal=_principal(), operation="kit-unknown-operation")
    )
    assert response.failed, "an unknown operation succeeded"
    assert response.error.code, "a failure carried no stable code"


def _check_transport_reports_retryability(adapter) -> None:
    """09_API §8: ``retryable`` is a field, never inferred by the consumer."""
    response = adapter.serve(
        TransportRequest(principal=_principal(), operation="kit-unknown-operation")
    )
    assert isinstance(response.error.retryable, bool), (
        "an error omitted retryable; consumers inferring it from a code is how "
        "retry storms begin"
    )


def _check_transport_echoes_version(adapter) -> None:
    """Obligation T3. The negotiated major travels with the answer."""
    response = adapter.serve(
        TransportRequest(
            principal=_principal(), operation="kit-unknown-operation", accepted_major=1
        )
    )
    assert response.version == 1, "the transport did not echo the negotiated major"


API_TRANSPORT_KIT = ConformanceKit(
    port_id=PortCatalogue.API_TRANSPORT,
    version="1.0.0",
    checks=(
        ConformanceCheck("interface", KitSection.SHAPE, _check_transport_shape, obligation="T5"),
        ConformanceCheck(
            "never_raises", KitSection.FAILURE, _check_transport_never_raises, obligation="T6"
        ),
        ConformanceCheck(
            "reports_retryability",
            KitSection.FAILURE,
            _check_transport_reports_retryability,
            obligation="T6",
        ),
        ConformanceCheck(
            "echoes_version",
            KitSection.SEMANTICS,
            _check_transport_echoes_version,
            obligation="T3",
        ),
    ),
)


ALL_EXPOSURE_KITS: tuple[ConformanceKit, ...] = (
    EVIDENCE_STORE_KIT,
    AUTHORIZATION_KIT,
    API_TRANSPORT_KIT,
)
