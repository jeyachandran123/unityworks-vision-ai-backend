"""A deterministic Vision OS fixture session.

### What this is, precisely

A **real** `VisionStateManager` holding **real** `Observation` objects, served by
a **real** `ObservationApi` through a **real** `StaticAuthorizer`. Scoping,
tenant isolation, authorization, cursors, audit and the evidence privilege all
behave exactly as they do in production, because they *are* production code.

### The one shortcut, named

Observations are constructed here rather than driven from a camera. The
platform's own exposure suite takes the same shortcut and explains why:

    "The one shortcut is the observation *source*: observations are built by M11
     directly rather than driven from a camera, because M14's contract is that it
     serves what M12 holds and never learns how it got there."

Phase 1 binds no `SourcePort`, so there is no camera to drive from. Phase 3
replaces this with a file-replay source and the fixture becomes redundant.

### Why it exists at all

To keep a promise. The migration's stated danger is that the validation
console's capability erodes — that DevTools quietly stops showing real platform
data and nobody notices, because there is nothing to notice against. This
fixture is the something: a **known observation count** that the frontend smoke
test asserts is rendered.

It is not sample data for a demo. It never appears in the product surface, it is
reachable only with `ACCESS_DEVTOOLS`, and it is labelled as a fixture in every
response that carries it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The count the smoke test asserts on both sides of the wire. Changing it means
#: changing the frontend test in the same commit — deliberately, so the two
#: cannot drift apart silently.
FIXTURE_OBSERVATION_COUNT = 6

FIXTURE_SESSION_ID = "fixture-kitchen-01"
FIXTURE_CAMERA = "cam-fixture-01"
FIXTURE_SITE = "site-fixture"

#: Three subjects, and deliberately one of each outcome the product must render
#: differently. A fixture where everything is compliant would let a UI that
#: cannot draw NOT_VISIBLE pass its own smoke test.
_SUBJECTS = (
    # (object_id, head_covering, hand_covering)
    ("obj-fixture-1", "hairnet", "gloves"),        # fully observed, compliant
    ("obj-fixture-2", "none", "gloves"),           # observed absent  → violation
    ("obj-fixture-3", "hairnet", "not_visible"),   # refused          → UNKNOWN
)


@dataclass(frozen=True, slots=True)
class FixtureSession:
    """An assembled fixture, and the handles the routes need."""

    api: Any
    state: Any
    tenant_id: str
    camera_id: str
    observation_count: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "session_id": FIXTURE_SESSION_ID,
            "kind": "fixture",
            "camera_id": self.camera_id,
            "tenant_id": self.tenant_id,
            "observation_count": self.observation_count,
            "note": (
                "A deterministic fixture. Real state manager, real Observation "
                "API, real authorization; observations constructed rather than "
                "acquired, because no source adapter is bound before Phase 3."
            ),
        }


def build_fixture_session(tenant_id: str) -> FixtureSession:
    """Assemble the fixture for one tenant. Deterministic and side-effect free."""
    from vision_os.adapters.exposure.authorization import StaticAuthorizer, full_grant
    from vision_os.adapters.persistence import InMemoryEvidenceStore
    from vision_os.adapters.synthesis import InMemoryObservationLog
    from vision_os.core.model.api import CapabilitySummary
    from vision_os.core.model.ids import AttributeKey, CameraId, ClassId, TenantId
    from vision_os.core.model.ids import SiteId
    from vision_os.exposure import AuditTrail, CountingAuditSink, ObservationApi
    from vision_os.kernel.clock import VirtualClock
    from vision_os.kernel.events import EventBus
    from vision_os.kernel.metrics import MetricsEngine
    from vision_os.state import VisionStateManager

    clock = VirtualClock()
    metrics = MetricsEngine(clock)
    events = EventBus(clock)
    tenant = TenantId(tenant_id)
    camera = CameraId(FIXTURE_CAMERA)

    state = VisionStateManager(
        clock=clock,
        metrics=metrics,
        events=events,
        log=InMemoryObservationLog(),
        config=_state_config(),
        site_id=SiteId(FIXTURE_SITE),
    )

    observations = _observations(tenant=tenant, camera=camera)
    state.append(observations)

    authorizer = StaticAuthorizer(
        # A full grant scoped to this one camera. Not tenant-wide: the fixture
        # must not be the reason a scoping bug goes unnoticed.
        grants=[full_grant("devtools", tenant, cameras=(camera,))]
    )

    api = ObservationApi(
        clock=clock,
        metrics=metrics,
        state=state,
        authorizer=authorizer,
        # A real audit trail. Every read the fixture serves is recorded, exactly
        # as it would be in production — including evidence access, which is the
        # one that matters legally.
        audit=AuditTrail(clock=clock, metrics=metrics, sinks=(CountingAuditSink(),)),
        evidence=InMemoryEvidenceStore(),
        capabilities=CapabilitySummary(
            taxonomy_version="fixture-taxonomy-1",
            producible_classes=(ClassId("person"),),
            producible_attributes=(
                AttributeKey("head_covering"),
                AttributeKey("hand_covering"),
            ),
        ),
    )

    return FixtureSession(
        api=api,
        state=state,
        tenant_id=tenant_id,
        camera_id=FIXTURE_CAMERA,
        observation_count=len(observations),
    )


def _state_config():
    from vision_os.kernel.config.schema import StateSection

    return StateSection(enabled=True, max_objects_per_partition=64)


def _observations(*, tenant, camera):
    """Two observations per subject: one head, one hand. Six in total."""
    from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
    from vision_os.core.model.frame import FrameRef
    from vision_os.core.model.ids import (
        AttributeKey,
        ClassId,
        ConfigRevision,
        FrameSeq,
        ModuleId,
        ObjectId,
        ObservationId,
        SiteId,
        StreamEpoch,
    )
    from vision_os.core.model.observation import (
        Attribute,
        Observation,
        ObservationType,
    )
    from vision_os.core.model.provenance import Provenance
    from vision_os.core.model.timebase import ClockQuality, Duration, Instant
    from vision_os.core.model.understanding import Timing
    from vision_os.core.model.visual_object import LifecycleState

    site = SiteId(FIXTURE_SITE)
    provenance = Provenance(
        producer_module=ModuleId("observation_builder"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("fixture"),
        deterministic=True,
    )

    built = []
    sequence = 0
    for object_id, head, hand in _SUBJECTS:
        for key, value in (("head_covering", head), ("hand_covering", hand)):
            sequence += 1
            moment = Instant(ns=sequence * 1_000_000_000)
            built.append(
                Observation(
                    observation_id=ObservationId(f"obs-fixture-{sequence:02d}"),
                    observation_type=ObservationType.ATTRIBUTE,
                    tenant_id=tenant,
                    site_id=site,
                    camera_id=camera,
                    frame_ref=FrameRef(camera, StreamEpoch(1), FrameSeq(sequence)),
                    t_capture=moment,
                    t_capture_unc=Duration.from_millis(10),
                    clock_quality=ClockQuality.NTP_SYNCED,
                    t_published=moment,
                    provenance=provenance,
                    timing=Timing(),
                    object_id=ObjectId(object_id),
                    class_id=ClassId("person"),
                    taxonomy_version="fixture-taxonomy-1",
                    lifecycle_state=LifecycleState.ACTIVE,
                    attributes=(
                        Attribute(
                            key=AttributeKey(key),
                            schema_version="1.0.0",
                            value=value,
                            # SELF_REPORTED, not a calibrated probability. 02_VOM
                            # §7.2 is explicit that a model's opinion about
                            # itself "is not a probability", and mislabelling it
                            # here would let a consumer order two of them.
                            confidence=Confidence.uncalibrated(
                                0.9, ConfidenceSemantics.SELF_REPORTED
                            ),
                            observed_at=moment,
                            producer=provenance,
                        ),
                    ),
                )
            )
    return tuple(built)


__all__ = [
    "FIXTURE_CAMERA",
    "FIXTURE_OBSERVATION_COUNT",
    "FIXTURE_SESSION_ID",
    "FixtureSession",
    "build_fixture_session",
]
