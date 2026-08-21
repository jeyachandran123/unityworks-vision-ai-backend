"""Fixtures for the Flow 3 tracking suite.

Everything is built from real modules and reference adapters — no mocks at a
module boundary. The tracker under test is the production tracker; the scenarios
are scripted detection sequences, which is the only part that needs to be
synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vision_os.adapters.tracking import (
    build_bytetrack_tracker,
    build_iou_tracker,
    build_sort_tracker,
)
from vision_os.conformance import platform_registry
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.detection import (
    Detection,
    DetectionEvidence,
    DetectionOutcome,
)
from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    ConfigRevision,
    DetectionId,
    FrameRef,
    FrameSeq,
    ModuleId,
    SiteId,
    StreamEpoch,
    TenantId,
)
from vision_os.core.model.provenance import InferenceTiming, Provenance
from vision_os.core.model.space import Box, FrameOfReference, SpatialInfo
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.tracking import TrackingRequest
from vision_os.kernel.config.schema import TrackingSection
from vision_os.perception.tracking import (
    AssociationPolicy,
    LifecyclePolicy,
    TrackingEngine,
    TrackingManager,
    TrackingRuntime,
)

CAMERA = CameraId("cam-01")
OTHER_CAMERA = CameraId("cam-02")
TENANT = TenantId("acme")
SITE = SiteId("site-sg-01")
PERSON = ClassId("person")

#: 5 fps — the platform's documented processing rate (11_PERFORMANCE section 1.1).
FRAME_INTERVAL_MS = 200


def make_detection(
    frame_ref: FrameRef,
    box: Box,
    *,
    score: float = 0.9,
    index: int = 0,
    class_id: ClassId = PERSON,
) -> Detection:
    """A standardized detection, exactly as Flow 2 emits one."""
    return Detection(
        detection_id=DetectionId(f"det-{frame_ref.frame_seq}-{index}"),
        frame_ref=frame_ref,
        tenant_id=TENANT,
        site_id=SITE,
        t_capture=Instant(frame_ref.frame_seq * FRAME_INTERVAL_MS * 1_000_000),
        t_capture_uncertainty=Duration.from_millis(5),
        class_id=class_id,
        taxonomy_version="1.0.0",
        confidence=Confidence.uncalibrated(score, ConfidenceSemantics.DETECTION_PRESENCE),
        spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED, bbox=box),
        provenance=Provenance(
            producer_module=ModuleId("detection_engine"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("test"),
        ),
        timing=InferenceTiming(inference_ms=3.0),
        evidence=DetectionEvidence(input_hash="test"),
    )


def make_request(
    seq: int,
    boxes: Sequence[Box],
    *,
    camera: CameraId = CAMERA,
    scores: Sequence[float] | None = None,
    elapsed_ms: int = FRAME_INTERVAL_MS,
    timestamp_ns: int | None = None,
    classes: Sequence[ClassId] | None = None,
) -> TrackingRequest:
    scores = list(scores) if scores is not None else [0.9] * len(boxes)
    classes = list(classes) if classes is not None else [PERSON] * len(boxes)
    frame_ref = FrameRef(camera, StreamEpoch(1), FrameSeq(seq))
    return TrackingRequest(
        camera_id=camera,
        frame_ref=frame_ref,
        timestamp=Instant(
            seq * FRAME_INTERVAL_MS * 1_000_000 if timestamp_ns is None else timestamp_ns
        ),
        elapsed=Duration.from_millis(elapsed_ms),
        detections=tuple(
            make_detection(frame_ref, box, score=scores[i], index=i, class_id=classes[i])
            for i, box in enumerate(boxes)
        ),
    )


def make_outcome(
    seq: int,
    boxes: Sequence[Box],
    *,
    camera: CameraId = CAMERA,
    scores: Sequence[float] | None = None,
    failed: bool = False,
    reason: str = "",
) -> DetectionOutcome:
    """What the Flow 2 seam hands to tracking."""
    scores = list(scores) if scores is not None else [0.9] * len(boxes)
    frame_ref = FrameRef(camera, StreamEpoch(1), FrameSeq(seq))
    return DetectionOutcome(
        frame_ref=frame_ref,
        detections=tuple(
            make_detection(frame_ref, box, score=scores[i], index=i)
            for i, box in enumerate(boxes)
        ),
        failed=failed,
        reason=reason,
    )


def walking_box(step: int, *, y: float = 0.4, speed: float = 0.04) -> Box:
    """An object crossing the frame left to right at a steady rate."""
    x = 0.1 + step * speed
    return Box(x, y, x + 0.1, y + 0.4)


def drive(tracker, steps: int, *, camera: CameraId = CAMERA, start: int = 0, **kwargs):
    """Walk an object through ``steps`` frames, returning every update."""
    return [
        tracker.update(make_request(start + s, [walking_box(start + s, **kwargs)], camera=camera))
        for s in range(steps)
    ]


def coast(tracker, frames: int, *, camera: CameraId = CAMERA, start: int = 100):
    """Feed empty frames — the normal case when the detector finds nothing."""
    return [
        tracker.update(make_request(start + s, [], camera=camera)) for s in range(frames)
    ]


# --- trackers ---------------------------------------------------------------- #


@pytest.fixture
def lifecycle_policy() -> LifecyclePolicy:
    return LifecyclePolicy(
        min_hits_to_confirm=3,
        max_coast_frames=5,
        max_lost_frames=10,
        max_age_frames=1_000,
        max_tracks_per_camera=32,
    )


@pytest.fixture
def association_policy() -> AssociationPolicy:
    return AssociationPolicy()


@pytest.fixture
def sort_tracker(lifecycle_policy, association_policy):
    return build_sort_tracker(
        lifecycle=lifecycle_policy,
        association=association_policy,
        config_revision="test",
    )


@pytest.fixture
def iou_tracker(lifecycle_policy, association_policy):
    return build_iou_tracker(
        lifecycle=lifecycle_policy,
        association=association_policy,
        config_revision="test",
    )


@pytest.fixture
def bytetrack_tracker(lifecycle_policy, association_policy):
    return build_bytetrack_tracker(
        lifecycle=lifecycle_policy,
        association=association_policy,
        config_revision="test",
    )


@pytest.fixture(params=["iou", "sort", "bytetrack"])
def any_tracker(request, lifecycle_policy, association_policy):
    """Every shipped tracker. Obligations hold for all of them or for none."""
    factories = {
        "iou": build_iou_tracker,
        "sort": build_sort_tracker,
        "bytetrack": build_bytetrack_tracker,
    }
    return factories[request.param](
        lifecycle=lifecycle_policy,
        association=association_policy,
        config_revision="test",
    )


# --- platform layer ----------------------------------------------------------- #


@pytest.fixture
def tracking_config() -> TrackingSection:
    return TrackingSection(
        enabled=True,
        tracker_id="tracker.sort",
        min_hits_to_confirm=3,
        max_coast_frames=5,
        max_lost_frames=10,
        max_age_frames=1_000,
        max_tracks_per_camera=32,
    )


@pytest.fixture
def tracking_manager(metrics, lifecycle_policy, association_policy):
    manager = TrackingManager(
        metrics=metrics,
        conformance=platform_registry(),
        fallback_factory=lambda: build_iou_tracker(
            config_revision="test"
        ),
    )
    manager.load(
        build_sort_tracker(
            lifecycle=lifecycle_policy,
            association=association_policy,
            config_revision="test",
        )
    )
    return manager


@pytest.fixture
def tracking_engine(clock, bus, metrics, tracking_manager, tracking_config):
    return TrackingEngine(
        clock=clock,
        bus=bus,
        metrics=metrics,
        manager=tracking_manager,
        config=tracking_config,
    )


@pytest.fixture
def tracking_runtime(clock, metrics, health, tracking_engine, tracking_config):
    return TrackingRuntime(
        clock=clock,
        metrics=metrics,
        health=health,
        engine=tracking_engine,
        config=tracking_config,
    )
