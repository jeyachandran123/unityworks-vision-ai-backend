"""Fixtures for the Flow 8 storage and exposure suite.

Real modules throughout: a real state manager over a real log, a real authorizer,
a real evidence store. Nothing is mocked at a module boundary — the API's whole
contract is *what a consumer sees*, and a mocked M12 would test the mock.

The one shortcut is the observation *source*: observations are built by M11
directly rather than driven from a camera, because M14's contract is that it
serves what M12 holds and never learns how it got there.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.exposure import InProcessTransport, StaticAuthorizer
from vision_os.adapters.exposure.authorization import full_grant, read_only_grant
from vision_os.adapters.persistence import (
    InMemoryEvidenceStore,
    NullEvidenceStore,
)
from vision_os.adapters.synthesis import AlwaysPublish, InMemoryObservationLog
from vision_os.core.model.api import (
    Action,
    CapabilitySummary,
    DeliveryPolicy,
    Principal,
    Scope,
    StateFilter,
    SubscriptionFilter,
    TimeWindow,
)
from vision_os.core.model.crop import PrivacyClass, RetentionMode
from vision_os.core.model.ids import AttributeKey, CameraId, ClassId, SiteId, TenantId
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.exposure import ApiLimits
from vision_os.core.ports.persistence import RetentionPolicy, hash_payload
from vision_os.exposure import AuditTrail, CountingAuditSink, ObservationApi
from vision_os.exposure.subscriptions import SubscriptionHub
from vision_os.state import VisionStateManager

from ..synthesis.conftest import (  # noqa: TID252 - the Flow 7 harness, reused deliberately
    CAMERA,
    OTHER_CAMERA,
    OTHER_TENANT,
    PERSON,
    POSTURE,
    SITE,
    TENANT,
    at,
    attribute,
    context,
    make_builder,
    make_object,
    state_config,
    understanding,
)

CONSUMER = "consumer-a"
OTHER_CONSUMER = "consumer-b"
EVIDENCE_PAYLOAD = b"a-crop-worth-of-bytes"
EVIDENCE_HASH = hash_payload(EVIDENCE_PAYLOAD)


# --- principals and grants ------------------------------------------------------ #


def principal(
    subject: str = CONSUMER, tenant: TenantId = TENANT
) -> Principal:
    return Principal(subject=subject, tenant_id=tenant)


@pytest.fixture
def reader() -> Principal:
    """A consumer that may read facts but never imagery.

    12_SECURITY §5.3's common case: *"Most consumers need the first and must
    never have the second."*
    """
    return principal()


@pytest.fixture
def operator() -> Principal:
    return principal(subject="operator")


@pytest.fixture
def authorizer() -> StaticAuthorizer:
    return StaticAuthorizer(
        [
            read_only_grant(CONSUMER, TENANT),
            full_grant("operator", TENANT),
            read_only_grant(OTHER_CONSUMER, OTHER_TENANT),
        ]
    )


def scope(*cameras: CameraId, tenant: TenantId = TENANT) -> Scope:
    return Scope(tenant_id=tenant, camera_ids=tuple(cameras))


# --- the platform pieces --------------------------------------------------------- #


@pytest.fixture
def log() -> InMemoryObservationLog:
    return InMemoryObservationLog()


@pytest.fixture
def state(clock, metrics, bus, log) -> VisionStateManager:
    return VisionStateManager(
        clock=clock,
        metrics=metrics,
        events=bus,
        config=state_config(),
        log=log,
        site_id=SITE,
    )


@pytest.fixture
def evidence() -> InMemoryEvidenceStore:
    return InMemoryEvidenceStore()


@pytest.fixture
def audit_sink() -> CountingAuditSink:
    return CountingAuditSink()


@pytest.fixture
def audit(clock, metrics, audit_sink) -> AuditTrail:
    return AuditTrail(clock=clock, metrics=metrics, sinks=(audit_sink,))


@pytest.fixture
def hub(clock, metrics) -> SubscriptionHub:
    return SubscriptionHub(clock=clock, metrics=metrics)


@pytest.fixture
def api(clock, metrics, state, authorizer, audit, hub, evidence) -> ObservationApi:
    return ObservationApi(
        clock=clock,
        metrics=metrics,
        state=state,
        authorizer=authorizer,
        audit=audit,
        hub=hub,
        evidence=evidence,
        capabilities=CapabilitySummary(
            taxonomy_version="taxonomy-1",
            producible_classes=(PERSON,),
            producible_attributes=(POSTURE,),
        ),
    )


@pytest.fixture
def api_transport(api) -> InProcessTransport:
    """Named ``api_transport`` rather than ``transport``.

    The root conftest already binds ``transport`` to the event bus's transport,
    and shadowing it here made ``bus`` depend on ``api``, which depends on
    ``bus`` — a fixture cycle pytest correctly refused to resolve.
    """
    from vision_os.adapters.exposure.transport import routes_for

    return InProcessTransport(routes_for(api))


# --- populating state ------------------------------------------------------------ #


def publish(
    state: VisionStateManager,
    *,
    count: int = 3,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    start: int = 0,
):
    """Put observations into state the only way anything can — by appending.

    Uses the real builder, so what lands in state is exactly what the platform
    would have produced.
    """
    builder = make_builder(policy=AlwaysPublish())
    published = []
    for i in range(count):
        observation = builder.build_presence(
            make_object(
                object_id=f"obj-{start + i}", camera=camera, tenant=tenant, seq=start + i
            ),
            context(seq=start + i, camera=camera, tenant=tenant),
        )
        if observation is not None:
            published.append(observation)
    if published:
        state.append(published)
    return tuple(published)


def publish_attributes(
    state: VisionStateManager,
    *,
    object_id: str = "obj-0",
    value: str = "standing",
    camera: CameraId = CAMERA,
    seq: int = 5,
):
    builder = make_builder(policy=AlwaysPublish())
    observations = builder.build_attribute(
        make_object(object_id=object_id, camera=camera, seq=seq),
        understanding(
            object_id=object_id,
            camera=camera,
            seq=seq,
            attributes=(attribute(POSTURE, value, observed_at=at(seq)),),
        ),
        context(seq=seq, camera=camera),
    )
    if observations:
        state.append(observations)
    return tuple(observations)


def store_evidence(
    store, *, tenant: TenantId = TENANT, payload: bytes = EVIDENCE_PAYLOAD, **kwargs
):
    defaults = {
        "retention": RetentionMode.EVIDENCE,
        "privacy_class": PrivacyClass.C1_IMAGERY,
        "expires_at": Instant(10_000_000_000),
    }
    return store.put(
        hash_payload(payload), payload, tenant_id=tenant, **{**defaults, **kwargs}
    )


def window(start_seq: int = 0, end_seq: int = 100) -> TimeWindow:
    return TimeWindow(start=at(start_seq), end=at(end_seq))


__all__ = [
    "CAMERA",
    "CONSUMER",
    "EVIDENCE_HASH",
    "EVIDENCE_PAYLOAD",
    "OTHER_CAMERA",
    "OTHER_CONSUMER",
    "OTHER_TENANT",
    "PERSON",
    "POSTURE",
    "SITE",
    "TENANT",
    "Action",
    "ApiLimits",
    "AttributeKey",
    "ClassId",
    "DeliveryPolicy",
    "Duration",
    "NullEvidenceStore",
    "RetentionMode",
    "RetentionPolicy",
    "Scope",
    "SiteId",
    "StateFilter",
    "SubscriptionFilter",
    "at",
    "principal",
    "publish",
    "publish_attributes",
    "scope",
    "store_evidence",
    "window",
]
