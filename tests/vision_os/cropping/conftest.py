"""Fixtures for the Flow 5 crop suite.

Built from real modules and reference adapters — no mocks at a module boundary.
Objects come from the production registry driven by the production tracker, so
the Crop Manager is exercised against the output it will actually receive rather
than a hand-written approximation.

The synthetic frames are deterministic by construction: a fixed gradient plus a
fixed high-frequency patch, so a blur measurement is reproducible and a test that
asserts on sharpness is asserting on something real.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.cropping import (
    DefaultTriggerPolicy,
    HeuristicQualityEstimator,
    PaddedCropStrategy,
    ReferenceCropExtractor,
)
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.crop import (
    CropRequest,
    TriggerReason,
)
from vision_os.core.model.demand import (
    Demand,
    DemandBudget,
    DemandScope,
    SubjectFilter,
)
from vision_os.core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    ConfigRevision,
    DemandId,
    FrameRef,
    FrameSeq,
    ModuleId,
    ObjectId,
    SiteId,
    StreamEpoch,
    SubscriberId,
    TenantId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box, FrameOfReference, SpatialInfo
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.visual_object import (
    ClassObservation,
    LifecycleState,
    VisualObject,
)
from vision_os.core.ports.cropping import AttributeStatus, TriggerCandidate
from vision_os.kernel.config.schema import CroppingSection
from vision_os.perception.cropping import (
    CapabilityView,
    CropManager,
    CropRuntime,
    DemandRegistry,
    FrameContext,
    GateThresholds,
    QualityGate,
    UnderstandingBudget,
)

CAMERA = CameraId("cam-01")
OTHER_CAMERA = CameraId("cam-02")
TENANT = TenantId("acme")
OTHER_TENANT = TenantId("globex")
SITE = SiteId("site-sg-01")
PERSON = ClassId("person")
COLOUR = AttributeKey("appearance.dominant_colour")
GARMENT = AttributeKey("appearance.upper_garment")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_INTERVAL_MS = 200


def at(seq: int) -> Instant:
    return Instant(seq * FRAME_INTERVAL_MS * 1_000_000)


def frame_ref(seq: int = 0, *, camera: CameraId = CAMERA) -> FrameRef:
    return FrameRef(camera, StreamEpoch(1), FrameSeq(seq))


def frame_context(
    seq: int = 0,
    *,
    camera: CameraId = CAMERA,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> FrameContext:
    return FrameContext(
        frame_ref=frame_ref(seq, camera=camera),
        width=width,
        height=height,
        t_capture=at(seq),
    )


# --- synthetic pixels ------------------------------------------------------------ #


def sharp_frame(
    width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, channels: int = 3
) -> memoryview:
    """A deterministic high-frequency checkerboard.

    Alternating pixels maximise the Laplacian response, so this frame grades as
    sharp under any sane blur measure. Deterministic so a replay produces the
    same grade (V13).
    """
    buffer = bytearray(width * height * channels)
    for y in range(height):
        for x in range(width):
            value = 255 if (x + y) % 2 == 0 else 0
            base = (y * width + x) * channels
            for c in range(channels):
                buffer[base + c] = value
    return memoryview(bytes(buffer))


def _noise(width: int, height: int) -> list[int]:
    """A deterministic pseudo-random texture — the stand-in for a real scene.

    Deterministic so a replay grades identically (V13). Fine-grained everywhere,
    unlike a checkerboard, which has structure at exactly one frequency.
    """
    values = []
    state = 0x2545F491
    for _ in range(width * height):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        values.append((state >> 16) & 0xFF)
    return values


def noise_frame(
    width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, channels: int = 3
) -> memoryview:
    """In-focus texture: neighbouring pixels differ sharply."""
    values = _noise(width, height)
    buffer = bytearray(width * height * channels)
    for index, value in enumerate(values):
        for c in range(channels):
            buffer[index * channels + c] = value
    return memoryview(bytes(buffer))


def blurred_frame(
    width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, channels: int = 3
) -> memoryview:
    """The same texture, smeared.

    Still structured at coarse scales — what a box filter removes is the *local*
    gradient, which is exactly what a blur measure must be sensitive to and what
    a stride-sampled one is blind to.
    """
    values = _noise(width, height)
    # Uniform noise is far sharper than any real scene, so it takes several
    # passes to reach the softness of a genuinely out-of-focus crop.
    for _ in range(6):
        smoothed = list(values)
        for y in range(height):
            for x in range(width):
                total = count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sy, sx = y + dy, x + dx
                        if 0 <= sy < height and 0 <= sx < width:
                            total += values[sy * width + sx]
                            count += 1
                smoothed[y * width + x] = total // count
        values = smoothed
    buffer = bytearray(width * height * channels)
    for index, value in enumerate(values):
        for c in range(channels):
            buffer[index * channels + c] = value
    return memoryview(bytes(buffer))


def other_sharp_frame(
    width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, channels: int = 3
) -> memoryview:
    """A different high-frequency pattern that also grades as sharp.

    For proving that different pixels hash differently. A featureless frame
    cannot serve that purpose any more: the gate rejects it before a crop
    exists, which is the gate working correctly.
    """
    buffer = bytearray(width * height * channels)
    for y in range(height):
        for x in range(width):
            value = 255 if (x // 2 + y) % 2 == 0 else 0
            base = (y * width + x) * channels
            for c in range(channels):
                buffer[base + c] = value
    return memoryview(bytes(buffer))


def flat_frame(
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    channels: int = 3,
    value: int = 128,
) -> memoryview:
    """A featureless grey frame — zero Laplacian variance, maximum blur."""
    return memoryview(bytes([value]) * (width * height * channels))


def dark_frame(width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> memoryview:
    return flat_frame(width, height, value=5)


def bright_frame(width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> memoryview:
    return flat_frame(width, height, value=250)


# --- objects --------------------------------------------------------------------- #


def make_object(
    *,
    object_id: str = "obj-1",
    box: Box | None = None,
    lifecycle: LifecycleState = LifecycleState.ACTIVE,
    class_id: ClassId = PERSON,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    attributes: dict | None = None,
    first_seq: int = 0,
    seq: int = 10,
    observation_count: int = 10,
) -> VisualObject:
    """A ``VisualObject`` exactly as M7 emits one."""
    box = box or Box(0.4, 0.3, 0.55, 0.85)
    confidence = Confidence.uncalibrated(0.9, ConfidenceSemantics.IDENTITY)
    return VisualObject(
        object_id=ObjectId(object_id),
        tenant_id=tenant,
        site_id=SITE,
        camera_id=camera,
        class_id=class_id,
        confidence=confidence,
        lifecycle=lifecycle,
        class_history=(
            ClassObservation(
                class_id=class_id, confidence=confidence, observed_at=at(seq)
            ),
        ),
        track_bindings=(),
        current_spatial=SpatialInfo(
            frame_of_reference=FrameOfReference.NORMALIZED, bbox=box
        ),
        spatial_history=(),
        attributes=attributes or {},
        first_seen=at(first_seq),
        last_seen=at(seq),
        last_confirmed=at(seq),
        observation_count=observation_count,
        provenance=Provenance(
            producer_module=ModuleId("object_registry"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
    )


def make_candidate(
    *,
    object_id: str = "obj-1",
    box: Box | None = None,
    attributes: dict | None = None,
    last_analysed: Instant | None = None,
    appearance_delta: float | None = None,
    last_gate_rejection: bool = False,
    entered_region: bool = False,
    lifecycle_changed: bool = False,
    camera: CameraId = CAMERA,
) -> TriggerCandidate:
    return TriggerCandidate(
        object_id=ObjectId(object_id),
        camera_id=camera,
        class_id=PERSON,
        box=box or Box(0.4, 0.3, 0.55, 0.85),
        lifecycle="active",
        identity_confidence=0.9,
        first_seen=at(0),
        last_confirmed=at(10),
        observation_count=10,
        attributes=attributes if attributes is not None else {COLOUR: AttributeStatus(key=COLOUR)},
        appearance_delta=appearance_delta,
        last_analysed=last_analysed,
        last_gate_rejection=last_gate_rejection,
        entered_region_this_frame=entered_region,
        lifecycle_changed_this_frame=lifecycle_changed,
    )


def make_request(
    *,
    object_id: str = "obj-1",
    box: Box | None = None,
    seq: int = 0,
    camera: CameraId = CAMERA,
    tenant: TenantId = TENANT,
    reason: TriggerReason = TriggerReason.FIRST_SIGHT,
    attributes: tuple[str, ...] = (str(COLOUR),),
    priority: str = "",
    demand_ids: tuple[str, ...] = (),
) -> CropRequest:
    return CropRequest(
        object_id=ObjectId(object_id),
        camera_id=camera,
        frame_ref=frame_ref(seq, camera=camera),
        source_box=box or Box(0.4, 0.3, 0.55, 0.85),
        trigger_reason=reason,
        tenant_id=tenant,
        site_id=SITE,
        class_id=PERSON,
        required_attributes=attributes,
        priority_class=priority,
        demand_ids=demand_ids,
    )


def make_demand(
    *,
    demand_id: str = "",
    attributes: tuple[AttributeKey, ...] = (COLOUR,),
    freshness_ms: int = 30_000,
    cameras: tuple[CameraId, ...] = (),
    classes: tuple[ClassId, ...] = (PERSON,),
    priority: str = "standard",
    expires_ms: int | None = None,
    max_per_hour: float = 0.0,
) -> Demand:
    return Demand(
        demand_id=DemandId(demand_id),
        subscriber=SubscriberId("kit-subscriber"),
        required_attributes=attributes,
        freshness=Duration.from_millis(freshness_ms),
        scope=DemandScope(camera_ids=cameras),
        subject_filter=SubjectFilter(class_ids=classes),
        priority_class=priority,
        budget=DemandBudget(max_calls_per_hour=max_per_hour),
        expires_at=Instant(expires_ms * 1_000_000) if expires_ms is not None else None,
    )


# --- adapters ---------------------------------------------------------------------- #


@pytest.fixture
def trigger_policy() -> DefaultTriggerPolicy:
    return DefaultTriggerPolicy(
        appearance_threshold=0.25,
        low_confidence=0.5,
        refresh_interval=Duration.from_millis(300_000),
    )


@pytest.fixture
def estimator() -> HeuristicQualityEstimator:
    return HeuristicQualityEstimator(min_scale_pixels=48.0, good_scale_pixels=160.0)


@pytest.fixture
def strategy() -> PaddedCropStrategy:
    return PaddedCropStrategy(padding=0.15, output_size=(64, 64))


@pytest.fixture
def extractor() -> ReferenceCropExtractor:
    return ReferenceCropExtractor()


@pytest.fixture
def gate() -> QualityGate:
    return QualityGate(GateThresholds(min_scale_pixels=48.0))


# --- platform ------------------------------------------------------------------------ #


@pytest.fixture
def cropping_config() -> CroppingSection:
    return CroppingSection(
        enabled=True,
        understanding_calls_per_hour=360_000.0,
        budget_window_ms=60_000,
        priority_classes=("urgent", "standard", "background"),
        crop_width=64,
        crop_height=64,
        min_scale_pixels=48.0,
        good_scale_pixels=160.0,
        gate_rejection_sample_size=5,
        capability_gap_threshold=3,
    )


@pytest.fixture
def cropping_provenance() -> Provenance:
    return Provenance(
        producer_module=ModuleId("crop_manager"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
        deterministic=True,
    )


@pytest.fixture
def capabilities() -> CapabilityView:
    """Both test attributes registered and producible.

    Deliberately *not* the shipping default: with no understander bound the
    honest capability view is empty, and every demand would be marked
    unsatisfiable. These tests are about attention, so they grant the capability
    Flow 6 will provide.
    """
    return CapabilityView(
        registered_attributes=frozenset({COLOUR, GARMENT}),
        producible_attributes=frozenset({COLOUR, GARMENT}),
        producible_classes=frozenset({PERSON}),
        observed_cameras=frozenset({CAMERA, OTHER_CAMERA}),
    )


@pytest.fixture
def demand_registry(capabilities) -> DemandRegistry:
    return DemandRegistry(capabilities=capabilities)


@pytest.fixture
def budget(clock, cropping_config) -> UnderstandingBudget:
    return UnderstandingBudget(
        ceiling_per_hour=cropping_config.understanding_calls_per_hour,
        window=Duration.from_millis(cropping_config.budget_window_ms),
        now=clock.monotonic(),
    )


@pytest.fixture
def manager(
    clock,
    metrics,
    bus,
    cropping_config,
    trigger_policy,
    estimator,
    strategy,
    extractor,
    cropping_provenance,
    demand_registry,
    budget,
    gate,
) -> CropManager:
    return CropManager(
        clock=clock,
        metrics=metrics,
        events=bus,
        config=cropping_config,
        policy=trigger_policy,
        estimator=estimator,
        strategy=strategy,
        extractor=extractor,
        provenance=cropping_provenance,
        demands=demand_registry,
        budget=budget,
        gate=gate,
    )


@pytest.fixture
def crop_runtime(clock, metrics, health, manager, cropping_config) -> CropRuntime:
    return CropRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        manager=manager,
        config=cropping_config,
    )
