"""Fixtures for the Flow 4 registry suite.

Built from real modules and reference adapters — no mocks at a module boundary.
Tracks come from the production tracker, so the registry is exercised against the
output it will actually receive rather than a hand-written approximation.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vision_os.adapters.registry import InMemoryObjectStore
from vision_os.adapters.tracking import build_sort_tracker
from vision_os.conformance import platform_registry
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    ConfigRevision,
    FrameRef,
    FrameSeq,
    LocalTrackId,
    ModuleId,
    RegionId,
    SiteId,
    StreamEpoch,
    TenantId,
    TrackerEpoch,
    TrackId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.region import Region
from vision_os.core.model.space import (
    Box,
    FrameOfReference,
    Point,
    Polygon,
    SpatialInfo,
)
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.model.track import (
    Association,
    AssociationMethod,
    MeasurementBasis,
    MotionEstimate,
    MotionState,
    Track,
    TrackEvidence,
    TrackState,
    TrackUpdate,
)
from vision_os.kernel.config.schema import RegistrySection
from vision_os.perception.registry import (
    AttributeRegistry,
    BindingPolicy,
    LifecyclePolicy,
    ObjectRegistry,
    RegistryPartition,
    RegistryRuntime,
)

CAMERA = CameraId("cam-01")
OTHER_CAMERA = CameraId("cam-02")
TENANT = TenantId("acme")
SITE = SiteId("site-sg-01")
PERSON = ClassId("person")

#: 5 fps — the platform's documented processing rate (11_PERFORMANCE section 1.1).
FRAME_INTERVAL_MS = 200


def at(seq: int) -> Instant:
    """Capture instant for a frame index."""
    return Instant(seq * FRAME_INTERVAL_MS * 1_000_000)


def track_id(local: int = 0, *, camera: CameraId = CAMERA, epoch: int = 0) -> TrackId:
    return TrackId(camera, TrackerEpoch(epoch), LocalTrackId(local))


def spatial(box: Box) -> SpatialInfo:
    return SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED, bbox=box)


def make_track(
    *,
    local: int = 0,
    box: Box | None = None,
    seq: int = 0,
    camera: CameraId = CAMERA,
    epoch: int = 0,
    state: TrackState = TrackState.CONFIRMED,
    class_id: ClassId = PERSON,
    confidence: float = 0.9,
    measured: bool = True,
    first_seq: int = 0,
    hit_count: int | None = None,
    age_frames: int | None = None,
) -> Track:
    """A track exactly as Flow 3 emits one."""
    box = box or Box(0.3, 0.4, 0.5, 0.8)
    age = age_frames if age_frames is not None else max(1, seq - first_seq + 1)
    hits = hit_count if hit_count is not None else age
    return Track(
        track_id=track_id(local, camera=camera, epoch=epoch),
        camera_id=camera,
        tenant_id=TENANT,
        site_id=SITE,
        state=state,
        class_id=class_id,
        confidence=Confidence.uncalibrated(confidence, ConfidenceSemantics.ASSOCIATION),
        spatial=spatial(box),
        measurement_basis=(
            MeasurementBasis.MEASURED if measured else MeasurementBasis.PREDICTED
        ),
        motion=MotionEstimate(),
        motion_state=MotionState.UNKNOWN,
        first_seen=at(first_seq),
        last_seen=at(seq) if measured else at(max(first_seq, seq - 1)),
        last_updated=at(seq),
        age_frames=age,
        hit_count=min(hits, age),
        coast_frames=0 if measured else 1,
        detections=(FrameRef(camera, StreamEpoch(1), FrameSeq(seq)),),
        evidence=TrackEvidence(association_method=AssociationMethod.IOU),
        provenance=Provenance(
            producer_module=ModuleId("tracking_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
    )


def make_update(
    tracks: Sequence[Track],
    *,
    seq: int = 0,
    camera: CameraId = CAMERA,
    epoch: int = 0,
) -> TrackUpdate:
    """A ``TrackUpdate`` as M6 hands one to M7."""
    return TrackUpdate(
        camera_id=camera,
        frame_ref=FrameRef(camera, StreamEpoch(1), FrameSeq(seq)),
        tracker_epoch=epoch,
        active=tuple(tracks),
        associations=tuple(
            Association(
                track_id=t.track_id,
                detection_index=i,
                confidence=Confidence.uncalibrated(
                    0.9, ConfidenceSemantics.ASSOCIATION
                ),
                method=AssociationMethod.IOU,
            )
            for i, t in enumerate(tracks)
        ),
    )


def walking(seq: int, *, speed: float = 0.03, y: float = 0.4) -> Box:
    """An object crossing the frame, staying inside the unit square."""
    x = 0.15 + (seq % 20) * speed
    return Box(x, y, x + 0.1, y + 0.4)


def drive(
    registry: ObjectRegistry,
    frames: int,
    *,
    camera: CameraId = CAMERA,
    start: int = 0,
    local: int = 0,
    epoch: int = 0,
    box=walking,
):
    """Feed a steadily-tracked object through the registry."""
    results = []
    for step in range(frames):
        seq = start + step
        track = make_track(
            local=local, box=box(seq), seq=seq, camera=camera, epoch=epoch, first_seq=start
        )
        results.append(
            registry.ingest(camera, make_update([track], seq=seq, camera=camera, epoch=epoch))
        )
    return results


def coast(
    registry: ObjectRegistry,
    frames: int,
    *,
    camera: CameraId = CAMERA,
    start: int = 100,
):
    """Feed frames with no tracks.

    An empty frame carries **no capture time**, so it does not advance the
    camera's clock — it only tells the registry that nothing was measured, which
    moves an active object to ``occluded``. Aging past that is horizon-driven and
    belongs to ``expire_stale``; see ``age``.
    """
    return [
        registry.ingest(
            camera, make_update([], seq=start + step, camera=camera)
        )
        for step in range(frames)
    ]


def age(registry: ObjectRegistry, to_seq: int):
    """Advance horizons to a capture instant, as the scheduled sweep does.

    ``expire_stale(now)`` is M7's documented API for exactly this: a camera that
    goes quiet must still see its objects age, and ingestion cannot invent a
    capture time for a frame that contains none.
    """
    return registry.expire_stale(at(to_seq))


def make_region(
    region_id: str = "Z3",
    *,
    box: tuple[float, float, float, float] = (0.2, 0.2, 0.8, 0.9),
    version: str = "1.0.0",
) -> Region:
    x1, y1, x2, y2 = box
    return Region(
        region_id=RegionId(region_id),
        geometry=Polygon(
            (Point(x1, y1), Point(x2, y1), Point(x2, y2), Point(x1, y2))
        ),
        frame_of_reference=FrameOfReference.NORMALIZED,
        label=region_id,
        camera_id=CAMERA,
        version=version,
    )


# --- policies ------------------------------------------------------------------ #


@pytest.fixture
def lifecycle_policy() -> LifecyclePolicy:
    return LifecyclePolicy(
        min_observations_to_confirm=3,
        provisional_horizon=Duration.from_millis(2_000),
        occlusion_horizon=Duration.from_millis(2_000),
        dormant_horizon=Duration.from_millis(6_000),
        retention_horizon=Duration.from_millis(12_000),
        max_objects_per_camera=32,
    )


@pytest.fixture
def binding_policy() -> BindingPolicy:
    return BindingPolicy()


@pytest.fixture
def registry_config() -> RegistrySection:
    return RegistrySection(
        enabled=True,
        min_observations_to_confirm=3,
        provisional_horizon_ms=2_000,
        occlusion_horizon_ms=2_000,
        dormant_horizon_ms=6_000,
        retention_horizon_ms=12_000,
        max_objects_per_camera=32,
        persistence_enabled=False,
    )


@pytest.fixture
def registry_provenance() -> Provenance:
    return Provenance(
        producer_module=ModuleId("object_registry"),
        producer_version="1.0.0",
        config_revision=ConfigRevision("test"),
        deterministic=True,
    )


@pytest.fixture
def attribute_registry() -> AttributeRegistry:
    return AttributeRegistry()


@pytest.fixture
def registry(
    clock, bus, metrics, registry_config, lifecycle_policy, binding_policy,
    attribute_registry, registry_provenance,
) -> ObjectRegistry:
    return ObjectRegistry(
        clock=clock,
        bus=bus,
        metrics=metrics,
        config=registry_config,
        tenant_id=TENANT,
        site_id=SITE,
        provenance=registry_provenance,
        lifecycle=lifecycle_policy,
        binding=binding_policy,
        attributes=attribute_registry,
    )


@pytest.fixture
def partition(lifecycle_policy, registry_provenance) -> RegistryPartition:
    return RegistryPartition(
        CAMERA,
        tenant_id=TENANT,
        site_id=SITE,
        policy=lifecycle_policy,
        provenance=registry_provenance,
    )


@pytest.fixture
def object_store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


@pytest.fixture
def registry_runtime(clock, metrics, health, registry, registry_config, object_store):
    return RegistryRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        registry=registry,
        config=registry_config,
        store=object_store,
    )


@pytest.fixture
def tracker():
    """The production tracker, so the registry sees real Flow 3 output."""
    return build_sort_tracker()


@pytest.fixture
def conformance():
    return platform_registry()
