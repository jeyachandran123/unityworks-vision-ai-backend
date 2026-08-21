"""Fixtures for the Flow 7 synthesis and state suite.

Built from real modules and reference adapters. The only thing scripted here is
the *understanding result* — M11's input — because M11's contract is that it
consumes a validated result and never asks a model anything, and a real model
would make every assertion non-deterministic while testing nothing about M11.

The attribute vocabulary is deliberately the same shape as Flow 6's: an enum
with a domain, a bool, and a scalar. The final gate re-checks what the earlier
gates checked, and a vocabulary of one shape cannot show that it does.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.synthesis import (
    AlwaysPublish,
    CollectingSink,
    ExactSuppression,
    InMemoryObservationLog,
)
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.crop import TriggerReason
from vision_os.core.model.detection import QualityGrades, QualityLevel
from vision_os.core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    ConfigRevision,
    CropId,
    EvidenceId,
    FrameRef,
    FrameSeq,
    ModelId,
    ModuleId,
    ObjectId,
    RequestId,
    SiteId,
    StreamEpoch,
    TenantId,
    TrackId,
)
from vision_os.core.model.observation import (
    Observation,
    ObservationType,
)
from vision_os.core.model.provenance import ModelMeta, Provenance
from vision_os.core.model.space import Box, FrameOfReference, SpatialInfo
from vision_os.core.model.timebase import ClockQuality, Duration, Instant
from vision_os.core.model.understanding import (
    Timing,
    UnderstandingEvidence,
    UnderstandingOutcome,
    UnderstandingResult,
)
from vision_os.core.model.visual_object import (
    Attribute,
    ClassObservation,
    LifecycleState,
    VisualObject,
)
from vision_os.core.ports.synthesis import SuppressionPolicyPort
from vision_os.kernel.config.schema import StateSection, SynthesisSection
from vision_os.perception.registry.attributes import (
    AttributeRegistry,
    AttributeSchema,
    AttributeValueType,
)
from vision_os.state import VisionStateManager
from vision_os.synthesis import (
    BuildContext,
    CeilingGate,
    ObservationBuilder,
    TaxonomyView,
)

CAMERA = CameraId("cam-01")
OTHER_CAMERA = CameraId("cam-02")
TENANT = TenantId("acme")
OTHER_TENANT = TenantId("globex")
SITE = SiteId("site-sg-01")
PERSON = ClassId("person")
VEHICLE = ClassId("vehicle")

POSTURE = AttributeKey("posture")
HEADWEAR = AttributeKey("headwear_present")
HEIGHT = AttributeKey("apparent_height_ratio")
UNREGISTERED = AttributeKey("is_authorized")

TAXONOMY_VERSION = "taxonomy-1"
FRAME_INTERVAL_MS = 200


def at(seq: int) -> Instant:
    return Instant(seq * FRAME_INTERVAL_MS * 1_000_000)


def frame_ref(seq: int = 0, *, camera: CameraId = CAMERA) -> FrameRef:
    return FrameRef(camera, StreamEpoch(1), FrameSeq(seq))


# --- the vocabulary the final gate polices ------------------------------------- #


def build_registry() -> AttributeRegistry:
    registry = AttributeRegistry()
    registry.register(
        AttributeSchema(
            key=POSTURE,
            value_type=AttributeValueType.ENUM,
            domain=("standing", "sitting", "lying", "crouching"),
            neutrality_justification="Body configuration is directly visible",
            applies_to=(PERSON,),
            validity=Duration.from_millis(60_000),
        )
    )
    registry.register(
        AttributeSchema(
            key=HEADWEAR,
            value_type=AttributeValueType.BOOL,
            neutrality_justification="Head region shows a covering",
            applies_to=(PERSON,),
        )
    )
    registry.register(
        AttributeSchema(
            key=HEIGHT,
            value_type=AttributeValueType.SCALAR,
            domain=("0:1",),
            neutrality_justification="Ratio of visible object height to frame height",
        )
    )
    return registry


@pytest.fixture
def attribute_registry() -> AttributeRegistry:
    return build_registry()


def build_taxonomy(version: str = TAXONOMY_VERSION) -> TaxonomyView:
    return TaxonomyView(version=version, classes=frozenset({PERSON, VEHICLE}))


@pytest.fixture
def taxonomy() -> TaxonomyView:
    return build_taxonomy()


# --- objects ------------------------------------------------------------------- #


def spatial(x: float = 0.4, y: float = 0.3, *, w: float = 0.15, h: float = 0.5) -> SpatialInfo:
    return SpatialInfo(
        frame_of_reference=FrameOfReference.NORMALIZED,
        bbox=Box(x, y, x + w, y + h),
    )


def make_object(
    *,
    object_id: str = "obj-1",
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    class_id: ClassId = PERSON,
    lifecycle: LifecycleState = LifecycleState.ACTIVE,
    seq: int = 3,
    confidence: float = 0.9,
    position: SpatialInfo | None = None,
    attributes: dict | None = None,
    observation_count: int = 1,
) -> VisualObject:
    moment = at(seq)
    return VisualObject(
        object_id=ObjectId(object_id),
        tenant_id=tenant,
        site_id=SITE,
        camera_id=camera,
        class_id=class_id,
        confidence=Confidence.uncalibrated(confidence, ConfidenceSemantics.IDENTITY),
        lifecycle=lifecycle,
        class_history=(
            ClassObservation(
                class_id=class_id,
                confidence=Confidence.uncalibrated(confidence, ConfidenceSemantics.CLASSIFICATION),
                observed_at=moment,
            ),
        ),
        track_bindings=(),
        current_spatial=position if position is not None else spatial(),
        spatial_history=(),
        attributes=attributes or {},
        first_seen=at(0),
        last_seen=moment,
        last_confirmed=moment,
        observation_count=observation_count,
        provenance=object_provenance(),
    )


def object_provenance() -> Provenance:
    return Provenance(
        producer_module=ModuleId("object_registry"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
    )


def builder_provenance() -> Provenance:
    return Provenance(
        producer_module=ModuleId("observation_builder"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
    )


def context(
    *,
    seq: int = 3,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    clock_quality: ClockQuality = ClockQuality.NTP_SYNCED,
    taxonomy_version: str = TAXONOMY_VERSION,
    uncertainty_ms: float = 0.0,
) -> BuildContext:
    return BuildContext(
        camera_id=camera,
        tenant_id=tenant,
        site_id=SITE,
        frame_ref=frame_ref(seq, camera=camera),
        t_capture=at(seq),
        t_capture_unc=Duration.from_millis(uncertainty_ms),
        clock_quality=clock_quality,
        taxonomy_version=taxonomy_version,
    )


# --- understanding results (M11's input) ---------------------------------------- #


def attribute(
    key: AttributeKey = POSTURE,
    value: object = "standing",
    *,
    confidence: float = 0.85,
    observed_at: Instant | None = None,
    valid_until: Instant | None = None,
    evidence_ref: str | None = "ev-1",
    schema_version: str = "1.0.0",
) -> Attribute:
    return Attribute(
        key=key,
        schema_version=schema_version,
        value=value,
        confidence=Confidence.uncalibrated(confidence, ConfidenceSemantics.ATTRIBUTE),
        observed_at=observed_at if observed_at is not None else at(3),
        producer=Provenance(
            producer_module=ModuleId("understanding_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        valid_until=valid_until,
        evidence_ref=evidence_ref,
    )


def evidence(
    *,
    evidence_id: str = "ev-1",
    crop: str = "crop-1",
    seq: int = 3,
    camera: CameraId = CAMERA,
    note: str | None = None,
    retention: str = "evidence",
) -> UnderstandingEvidence:
    """M9's evidence — everything except ``observation_id``.

    The omission is the point: M11 stamps that field, and a fixture that
    pre-filled it would hide whether M11 actually does.
    """
    return UnderstandingEvidence(
        evidence_id=EvidenceId(evidence_id),
        trigger_reason=TriggerReason.FIRST_SIGHT,
        input_hash="sha256:input",
        frame_ref=frame_ref(seq, camera=camera),
        crop_ref=CropId(crop),
        raw_output_ref="sha256:raw",
        unstructured_note=note,
        timing=Timing(queued_ms=1.0, inference_ms=8.0, total_ms=9.0),
        retention=retention,
    )


def model_meta() -> ModelMeta:
    return ModelMeta(
        model_id=ModelId("vlm-test"),
        model_version="1.0.0",
        artifact_hash="sha256:artifact",
    )


def understanding(
    *,
    request_id: str = "req-1",
    object_id: str = "obj-1",
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    class_id: ClassId = PERSON,
    outcome: UnderstandingOutcome = UnderstandingOutcome.SUCCEEDED,
    attributes: tuple[Attribute, ...] | None = None,
    seq: int = 3,
    demand_ids: tuple[str, ...] = (),
    evidence_id: str = "ev-1",
    requested: tuple[AttributeKey, ...] = (POSTURE,),
    note: str | None = None,
    retention: str = "evidence",
) -> UnderstandingResult:
    """A validated M9 result — M11's only semantic input."""
    return UnderstandingResult(
        request_id=RequestId(request_id),
        tenant_id=tenant,
        site_id=SITE,
        camera_id=camera,
        object_id=ObjectId(object_id),
        class_id=class_id,
        outcome=outcome,
        evidence=evidence(
            evidence_id=evidence_id,
            seq=seq,
            camera=camera,
            note=note,
            retention=retention,
        ),
        provenance=Provenance(
            producer_module=ModuleId("understanding_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        attributes=attributes if attributes is not None else (attribute(),),
        model_used=model_meta(),
        requested_attributes=requested,
        demand_ids=demand_ids,
    )


def quality(level: QualityLevel = QualityLevel.GOOD) -> QualityGrades:
    return QualityGrades(scale_pixels=264.0, overall=level)


# --- assembled modules ----------------------------------------------------------- #


def synthesis_config(**overrides) -> SynthesisSection:
    defaults = {
        "enabled": True,
        "heartbeat_ms": 30_000,
        "suppression_policy": "suppression.exact",
    }
    return SynthesisSection(**{**defaults, **overrides})


def state_config(**overrides) -> StateSection:
    defaults = {"enabled": True}
    return StateSection(**{**defaults, **overrides})


def make_builder(
    *,
    clock=None,
    metrics=None,
    events=None,
    registry: AttributeRegistry | None = None,
    view: TaxonomyView | None = None,
    policy: SuppressionPolicyPort | None = None,
    **config_overrides,
) -> ObservationBuilder:
    """Assemble a builder outside pytest's fixture graph.

    Needed because several tests build two builders with different policies in
    one test, and a fixture cannot be parameterised per-call. Defaults mirror the
    ``builder`` fixture so the two never drift.
    """
    from vision_os.kernel.events import EventBus
    from vision_os.kernel.metrics import MetricsEngine

    from ..conftest import VirtualClock  # noqa: TID252 - the shared platform clock

    the_clock = clock if clock is not None else VirtualClock()
    return ObservationBuilder(
        clock=the_clock,
        metrics=metrics if metrics is not None else MetricsEngine(clock=the_clock),
        events=events if events is not None else EventBus(clock=the_clock),
        config=synthesis_config(**config_overrides),
        gate=CeilingGate(
            registry if registry is not None else build_registry(),
            view if view is not None else build_taxonomy(),
        ),
        provenance=builder_provenance(),
        suppression_policy=policy if policy is not None else ExactSuppression(),
    )


@pytest.fixture
def builder(clock, metrics, bus, attribute_registry, taxonomy) -> ObservationBuilder:
    return make_builder(
        clock=clock,
        metrics=metrics,
        events=bus,
        registry=attribute_registry,
        view=taxonomy,
    )


@pytest.fixture
def loud_builder(clock, metrics, bus, attribute_registry, taxonomy) -> ObservationBuilder:
    """A builder that never suppresses.

    Suppression is correct behaviour, and it is also the thing that makes a test
    of *envelope* content need two distinct subjects to see two observations.
    Where the property under test is not suppression, this removes the noise.
    """
    return make_builder(
        clock=clock,
        metrics=metrics,
        events=bus,
        registry=attribute_registry,
        view=taxonomy,
        policy=AlwaysPublish(),
        suppression_policy="suppression.always",
    )


@pytest.fixture
def log() -> InMemoryObservationLog:
    return InMemoryObservationLog()


@pytest.fixture
def sink() -> CollectingSink:
    return CollectingSink()


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


# --- helpers -------------------------------------------------------------------- #


def presence_of(
    builder: ObservationBuilder,
    obj: VisualObject | None = None,
    *,
    seq: int = 3,
    **kwargs,
) -> Observation:
    """Build a presence observation and assert it was published.

    Tests that need *an observation* rather than *the suppression decision* use
    this so a stray ``None`` fails at its cause rather than as an attribute error
    three lines later.
    """
    subject = obj if obj is not None else make_object(seq=seq)
    result = builder.build_presence(subject, context(seq=seq), **kwargs)
    assert result is not None, "expected a first-sighting presence to publish"
    return result


def types_of(observations) -> list[ObservationType]:
    return [o.observation_type for o in observations]


def ids_of(observations) -> list[str]:
    return [str(o.observation_id) for o in observations]


def track(name: str = "trk-1") -> TrackId:
    return TrackId(name)


def timing(total_ms: float = 1.0) -> Timing:
    return Timing(total_ms=total_ms)
