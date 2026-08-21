"""``kit.tracker`` — the executable contract for P9 (06_PORTS section P9).

An interface constrains the shape of a call, not the meaning of its result. Two
trackers can implement ``TrackerPort`` perfectly and still corrupt everything
downstream — one reuses ids after termination, one presents predictions as
measurements, one silently accepts out-of-order frames and integrates a negative
time step. None of that is visible from the signature.

**What this kit can and cannot establish, stated plainly:**

It *can* prove structural obligations — id uniqueness within an epoch, epoch
advance on reset, coasting marked predicted, termination carrying a reason,
bounded memory, statelessness across cameras, determinism, and that an empty
frame is handled rather than rejected. These are checkable because they are
properties of the output *shape and bookkeeping*, verifiable against any
tracker's own behaviour.

It *cannot* prove tracking **quality** — fragmentation rate, ID-switch rate,
occlusion recovery rate. Those require ground-truth annotated sequences
(14_TESTING section 7.2), which are deployment data rather than platform code.
A kit that claimed to measure them from synthetic input would be measuring its
own fixtures. That gap is closed by the golden section when a corpus exists, and
is recorded as a known limitation until then.

The fast subset runs at plugin load, so a mis-built tracker is rejected before a
single real frame reaches it.
"""

from __future__ import annotations

import gc

from ..core.errors import OutOfOrderFrameError
from ..core.model.confidence import Confidence, ConfidenceSemantics
from ..core.model.detection import Detection, DetectionEvidence
from ..core.model.ids import (
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
from ..core.model.provenance import InferenceTiming, Provenance
from ..core.model.space import Box, FrameOfReference, SpatialInfo
from ..core.model.timebase import Duration, Instant
from ..core.model.track import BreakReason, MeasurementBasis, TrackState
from ..core.ports.tracking import TrackingRequest
from ..kernel.plugins.manifest import PortCatalogue
from .kit import ConformanceCheck, ConformanceKit, KitSection

_CAMERA = CameraId("kit-cam-01")
_OTHER = CameraId("kit-cam-02")
_TENANT = TenantId("kit-tenant")
_SITE = SiteId("kit-site")
_CLASS = ClassId("person")


def _detection(
    frame_ref: FrameRef, box: Box, score: float = 0.9, index: int = 0
) -> Detection:
    return Detection(
        detection_id=DetectionId(f"kit-{frame_ref.frame_seq}-{index}"),
        frame_ref=frame_ref,
        tenant_id=_TENANT,
        site_id=_SITE,
        t_capture=Instant(frame_ref.frame_seq * 200_000_000),
        t_capture_uncertainty=Duration.from_millis(5),
        class_id=_CLASS,
        taxonomy_version="1.0.0",
        confidence=Confidence.uncalibrated(score, ConfidenceSemantics.DETECTION_PRESENCE),
        spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED, bbox=box),
        provenance=Provenance(
            producer_module=ModuleId("kit"),
            producer_version="1.0.0",
            config_revision=ConfigRevision("kit"),
        ),
        timing=InferenceTiming(),
        evidence=DetectionEvidence(input_hash="kit"),
    )


def _request(
    seq: int,
    boxes,
    *,
    camera: CameraId = _CAMERA,
    scores=None,
    elapsed_ms: int = 200,
    timestamp_ns: int | None = None,
) -> TrackingRequest:
    scores = scores or [0.9] * len(boxes)
    frame_ref = FrameRef(camera, StreamEpoch(1), FrameSeq(seq))
    return TrackingRequest(
        camera_id=camera,
        frame_ref=frame_ref,
        timestamp=Instant(
            seq * 200_000_000 if timestamp_ns is None else timestamp_ns
        ),
        elapsed=Duration.from_millis(elapsed_ms),
        detections=tuple(
            _detection(frame_ref, box, scores[i], i) for i, box in enumerate(boxes)
        ),
    )


def _walk(adapter, camera: CameraId = _CAMERA, frames: int = 6, *, start: int = 0):
    """Drive an object moving steadily left to right."""
    updates = []
    for step in range(frames):
        seq = start + step
        x = 0.1 + step * 0.05
        box = Box(x, 0.4, x + 0.1, 0.8)
        updates.append(adapter.update(_request(seq, [box], camera=camera)))
    return updates


# --- shape ------------------------------------------------------------------- #


def _declares_capabilities(adapter) -> None:
    capabilities = adapter.capabilities()
    assert capabilities.tracker_id, "a tracker must declare a tracker_id"
    assert capabilities.version, "a tracker must declare a version"
    assert capabilities.max_objects >= 1, "max_objects must be at least 1"
    assert capabilities.handles_occlusion in ("none", "short", "long")


def _capabilities_are_stable(adapter) -> None:
    """Callers cache capabilities; a tracker that changes them mid-run breaks
    every decision already made on the old values."""
    first = adapter.capabilities()
    _walk(adapter, CameraId("kit-cap"), frames=3)
    second = adapter.capabilities()
    assert first == second, "capabilities changed after processing frames"


def _empty_frame_is_handled(adapter) -> None:
    """A frame with no detections is **normal**, not an error.

    It is exactly when tracks coast and terminate. A tracker that rejects it
    freezes every track for the duration of a detector outage.
    """
    camera = CameraId("kit-empty")
    _walk(adapter, camera, frames=3)
    update = adapter.update(_request(10, [], camera=camera))
    assert update is not None, "an empty frame must return an update, not None"
    assert update.camera_id == camera


def _update_returns_the_right_camera(adapter) -> None:
    camera = CameraId("kit-echo")
    update = adapter.update(_request(0, [Box(0.1, 0.1, 0.2, 0.2)], camera=camera))
    assert update.camera_id == camera
    assert update.frame_ref.camera_id == camera


def _tracks_query_is_total(adapter) -> None:
    """Asking about an unknown camera returns empty, never raises."""
    result = adapter.tracks(CameraId("kit-never-seen"))
    assert result is not None
    assert len(tuple(result)) == 0


# --- semantics --------------------------------------------------------------- #


def _ids_are_unique_within_epoch(adapter) -> None:
    """T3 — ids unique within ``(camera, epoch)`` and never reused.

    Reuse inside an epoch lets a consumer join two unrelated objects into one
    continuous history: invisible downstream and unrecoverable afterwards.
    """
    camera = CameraId("kit-unique")
    seen: set = set()
    for update in _walk(adapter, camera, frames=8):
        frame_ids = [t.track_id for t in update.active]
        assert len(frame_ids) == len(set(frame_ids)), (
            f"duplicate track ids within one frame: {frame_ids}"
        )
        for track_id in update.new:
            assert track_id not in seen, (
                f"track id {track_id} was issued twice within one epoch "
                f"(port obligation T3)"
            )
            seen.add(track_id)


def _track_ids_carry_camera_and_epoch(adapter) -> None:
    """A bare integer id would compare equal across cameras and epochs."""
    camera = CameraId("kit-composite")
    updates = _walk(adapter, camera, frames=4)
    for track in updates[-1].active:
        assert track.track_id.camera_id == camera, (
            "a track id must name its camera; a camera-less id is an identity "
            "waiting to happen (invariant V10)"
        )
        assert track.track_id.tracker_epoch == updates[-1].tracker_epoch


def _out_of_order_is_rejected(adapter) -> None:
    """T1 — reject loudly rather than degrade silently.

    An out-of-order frame integrates a negative time step and runs positions
    backwards. A tracker that absorbs it produces degraded output that looks
    like poor tracker quality rather than the pipeline bug it is.
    """
    camera = CameraId("kit-order")
    _walk(adapter, camera, frames=4)
    try:
        adapter.update(_request(1, [Box(0.2, 0.4, 0.3, 0.8)], camera=camera))
    except OutOfOrderFrameError:
        return
    raise AssertionError(
        "an out-of-order frame was accepted; per-camera ordering violations must "
        "be rejected and alarmed, never absorbed (port obligation T1)"
    )


def _coasting_is_marked_predicted(adapter) -> None:
    """T5 — a predicted position is never presented as measured.

    The corruption V8 exists to prevent, and one no consumer can detect alone.
    """
    camera = CameraId("kit-coast")
    _walk(adapter, camera, frames=6)
    for step in range(3):
        update = adapter.update(_request(20 + step, [], camera=camera))
        for track in update.active:
            if track.state.is_predicted:
                assert track.measurement_basis is not MeasurementBasis.MEASURED, (
                    f"track {track.track_id} is {track.state.value} but reports a "
                    f"MEASURED position (port obligation T5)"
                )
                assert track.is_predicted


def _termination_carries_a_reason(adapter) -> None:
    """T6 — the diagnostic that makes tracker regressions attributable."""
    camera = CameraId("kit-terminate")
    _walk(adapter, camera, frames=6)
    saw_termination = False
    for step in range(40):
        update = adapter.update(_request(30 + step, [], camera=camera))
        for _track_id, reason in update.terminated:
            saw_termination = True
            assert reason is not BreakReason.NONE, (
                "a terminated track must carry a break_reason (port obligation T6)"
            )
        if not update.active:
            break
    assert saw_termination, (
        "no track terminated after 40 empty frames; tracking must not retain "
        "tracks indefinitely (port obligation T8)"
    )


def _association_confidence_semantics(adapter) -> None:
    """T4 — ``ASSOCIATION``, not the detector's presence score."""
    camera = CameraId("kit-confidence")
    updates = _walk(adapter, camera, frames=5)
    for update in updates:
        for track in update.active:
            assert track.confidence.semantics is ConfidenceSemantics.ASSOCIATION, (
                f"track confidence carries {track.confidence.semantics.value}; a "
                f"track's confidence measures association, not detection presence "
                f"(port obligation T4)"
            )
            assert 0.0 <= track.confidence.value <= 1.0
        for association in update.associations:
            assert association.confidence.semantics is ConfidenceSemantics.ASSOCIATION


def _reset_mints_a_new_epoch(adapter) -> None:
    """T7 — reset clears state and advances the epoch.

    The new epoch is what makes the discontinuity visible: without it a recycled
    local id lets a consumer infer that an object teleported.
    """
    camera = CameraId("kit-reset")
    _walk(adapter, camera, frames=5)
    before = adapter.tracks(camera)
    assert len(tuple(before)) > 0, "expected live tracks before reset"

    epoch = adapter.reset(camera, "kit")
    after = adapter.tracks(camera)
    assert len(tuple(after)) == 0, "reset must discard all tracks for the camera"

    update = adapter.update(_request(0, [Box(0.1, 0.4, 0.2, 0.8)], camera=camera))
    assert update.tracker_epoch == epoch, (
        f"tracks after reset carry epoch {update.tracker_epoch}, not the new "
        f"epoch {epoch} (port obligation T7)"
    )
    for track in update.active:
        assert track.track_id.tracker_epoch == epoch


def _no_cross_camera_state(adapter) -> None:
    """T7 — cameras are independent. Cross-camera identity is P11, not this port."""
    _walk(adapter, _CAMERA, frames=4)
    _walk(adapter, _OTHER, frames=4)

    first = adapter.tracks(_CAMERA)
    second = adapter.tracks(_OTHER)
    for track in first:
        assert track.camera_id == _CAMERA
    for track in second:
        assert track.camera_id == _OTHER

    adapter.reset(_CAMERA, "kit")
    survivors = adapter.tracks(_OTHER)
    assert len(tuple(survivors)) == len(tuple(second)), (
        "resetting one camera disturbed another; no cross-camera state may exist "
        "in this port (port obligation T7)"
    )


def _non_uniform_gaps_are_handled(adapter) -> None:
    """T2 — the single most common way an off-the-shelf tracker misbehaves here.

    The scheduler drops frames by design, so a tracker must integrate over
    elapsed *time*. This check feeds wildly uneven gaps covering the same
    distance and requires the track to survive: a tracker keying on frame count
    predicts positions far from the object and loses it.
    """
    camera = CameraId("kit-gaps")
    gaps_ms = [200, 1000, 100, 1500, 250, 800]
    x = 0.1
    timestamp = 0
    track_ids: set = set()

    for seq, gap in enumerate(gaps_ms):
        # Constant real-world speed: displacement proportional to elapsed time,
        # which is precisely what a frame-count-based model gets wrong.
        x += 0.00005 * gap
        timestamp += gap * 1_000_000
        update = adapter.update(
            _request(
                seq,
                [Box(x, 0.4, x + 0.1, 0.8)],
                camera=camera,
                elapsed_ms=gap,
                timestamp_ns=timestamp,
            )
        )
        track_ids.update(t.track_id for t in update.active)

    assert len(track_ids) <= 2, (
        f"a steadily moving object fragmented into {len(track_ids)} tracks under "
        f"non-uniform frame gaps; motion must integrate over elapsed time, never "
        f"over frame count (port obligation T2)"
    )


def _determinism(adapter_factory) -> None:
    """Identical input yields identical output, including tie-breaks.

    Non-determinism silently changes which object keeps which id when two
    candidates tie, producing ID switches no test can reproduce (invariant V13).
    """
    if not callable(adapter_factory):
        return

    def run():
        instance = adapter_factory()
        camera = CameraId("kit-determinism")
        signature = []
        for step in range(6):
            x = 0.1 + step * 0.05
            update = instance.update(
                _request(
                    step,
                    [Box(x, 0.4, x + 0.1, 0.8), Box(0.6 - x, 0.1, 0.7 - x, 0.5)],
                    camera=camera,
                )
            )
            signature.append(
                tuple(
                    (str(t.track_id), t.state.value, round(t.spatial.bbox.x1, 6))
                    for t in update.active
                )
            )
        return tuple(signature)

    first, second = run(), run()
    assert first == second, "two identical runs produced different tracks"


def _empty_result_is_not_an_error(adapter) -> None:
    """A tracker with no tracks and no detections returns a valid empty update."""
    camera = CameraId("kit-nothing")
    update = adapter.update(_request(0, [], camera=camera))
    assert update is not None
    assert len(update.active) == 0
    assert not update.failed, "an empty scene is not a tracking failure"


# --- failure ------------------------------------------------------------------ #


def _no_fabricated_tracks(adapter) -> None:
    """A tracker must never invent a track from nothing.

    The tracking analogue of the detector's no-fabrication obligation: with no
    detections ever supplied, there is nothing to be continuous *with*.
    """
    camera = CameraId("kit-fabricate")
    for step in range(5):
        update = adapter.update(_request(step, [], camera=camera))
        assert len(update.active) == 0, (
            f"tracker produced {len(update.active)} tracks from zero detections; "
            f"unknown is preferable to fabricated tracking"
        )
        assert len(update.new) == 0


def _degenerate_boxes_are_survived(adapter) -> None:
    """Extreme-but-legal geometry must not crash the frame."""
    camera = CameraId("kit-degenerate")
    boxes = [
        Box(0.0, 0.0, 1.0, 1.0),
        Box(0.0, 0.0, 0.001, 0.001),
        Box(0.999, 0.999, 1.0, 1.0),
        Box(0.0, 0.4, 1.0, 0.401),
    ]
    for step, box in enumerate(boxes):
        adapter.update(_request(step, [box], camera=camera))


def _reset_of_unknown_camera_is_safe(adapter) -> None:
    """Resetting a camera that was never tracked is a no-op, not a fault."""
    epoch = adapter.reset(CameraId("kit-unknown-reset"), "kit")
    assert epoch >= 0


def _many_objects_are_survived(adapter) -> None:
    """A crowd must degrade, never crash."""
    camera = CameraId("kit-crowd")
    boxes = []
    for i in range(60):
        x = (i % 10) * 0.09
        y = (i // 10) * 0.15
        boxes.append(Box(x, y, x + 0.05, y + 0.1))
    for step in range(3):
        update = adapter.update(_request(step, boxes, camera=camera))
        assert update is not None


# --- resource ------------------------------------------------------------------ #


def _memory_is_bounded(adapter) -> None:
    """T8 — memory bounded regardless of scene duration or object count.

    The 30-day soak failure caught in seconds: a tracker retaining per-frame
    history without a cap looks perfect on day one and exhausts the node on day
    twenty-six.
    """
    camera = CameraId("kit-memory")
    for step in range(40):
        x = 0.1 + (step % 10) * 0.05
        adapter.update(_request(step, [Box(x, 0.4, x + 0.1, 0.8)], camera=camera))

    gc.collect()
    before = len(gc.get_objects())
    for step in range(300):
        x = 0.1 + (step % 10) * 0.05
        adapter.update(_request(40 + step, [Box(x, 0.4, x + 0.1, 0.8)], camera=camera))
    gc.collect()
    after = len(gc.get_objects())

    growth = after - before
    assert growth < 20_000, (
        f"tracker retained {growth} objects across 300 frames; memory must be "
        f"bounded regardless of scene duration (port obligation T8)"
    )


def _track_count_stays_within_declared_maximum(adapter) -> None:
    """A declaration is a contract, not a hint."""
    capabilities = adapter.capabilities()
    camera = CameraId("kit-capacity")
    boxes = []
    for i in range(40):
        x = (i % 8) * 0.12
        y = (i // 8) * 0.19
        boxes.append(Box(x, y, x + 0.06, y + 0.12))

    for step in range(6):
        update = adapter.update(_request(step, boxes, camera=camera))
        assert len(update.active) <= capabilities.max_objects, (
            f"tracker holds {len(update.active)} tracks but declared a maximum of "
            f"{capabilities.max_objects}"
        )


def _terminated_tracks_are_released(adapter) -> None:
    """A terminated track must leave the live set, not linger."""
    camera = CameraId("kit-release")
    _walk(adapter, camera, frames=5)
    for step in range(60):
        update = adapter.update(_request(50 + step, [], camera=camera))
        for track in update.active:
            assert track.state is not TrackState.TERMINATED, (
                f"terminated track {track.track_id} was returned as active"
            )
        if not update.active:
            return
    raise AssertionError("tracks never cleared after 60 empty frames")


TRACKER_KIT = ConformanceKit(
    port_id=PortCatalogue.TRACKER,
    version="1.0.0",
    checks=(
        # shape
        ConformanceCheck(
            "declares_capabilities", KitSection.SHAPE, _declares_capabilities, "A1"
        ),
        ConformanceCheck(
            "capabilities_are_stable", KitSection.SHAPE, _capabilities_are_stable, "A1"
        ),
        ConformanceCheck("empty_frame_is_handled", KitSection.SHAPE, _empty_frame_is_handled),
        ConformanceCheck(
            "update_returns_the_right_camera",
            KitSection.SHAPE,
            _update_returns_the_right_camera,
            "T7",
        ),
        ConformanceCheck("tracks_query_is_total", KitSection.SHAPE, _tracks_query_is_total),
        # semantics
        ConformanceCheck(
            "ids_are_unique_within_epoch",
            KitSection.SEMANTICS,
            _ids_are_unique_within_epoch,
            "T3",
        ),
        ConformanceCheck(
            "track_ids_carry_camera_and_epoch",
            KitSection.SEMANTICS,
            _track_ids_carry_camera_and_epoch,
            "T3",
        ),
        ConformanceCheck(
            "out_of_order_is_rejected", KitSection.SEMANTICS, _out_of_order_is_rejected, "T1"
        ),
        ConformanceCheck(
            "non_uniform_gaps_are_handled",
            KitSection.SEMANTICS,
            _non_uniform_gaps_are_handled,
            "T2",
        ),
        ConformanceCheck(
            "association_confidence_semantics",
            KitSection.SEMANTICS,
            _association_confidence_semantics,
            "T4",
        ),
        ConformanceCheck(
            "coasting_is_marked_predicted",
            KitSection.SEMANTICS,
            _coasting_is_marked_predicted,
            "T5",
        ),
        ConformanceCheck(
            "termination_carries_a_reason",
            KitSection.SEMANTICS,
            _termination_carries_a_reason,
            "T6",
        ),
        ConformanceCheck(
            "reset_mints_a_new_epoch", KitSection.SEMANTICS, _reset_mints_a_new_epoch, "T7"
        ),
        ConformanceCheck(
            "no_cross_camera_state", KitSection.SEMANTICS, _no_cross_camera_state, "T7"
        ),
        ConformanceCheck(
            "empty_result_is_not_an_error",
            KitSection.SEMANTICS,
            _empty_result_is_not_an_error,
        ),
        # failure
        ConformanceCheck(
            "no_fabricated_tracks", KitSection.FAILURE, _no_fabricated_tracks, "A4"
        ),
        ConformanceCheck(
            "degenerate_boxes_are_survived", KitSection.FAILURE, _degenerate_boxes_are_survived
        ),
        ConformanceCheck(
            "reset_of_unknown_camera_is_safe",
            KitSection.FAILURE,
            _reset_of_unknown_camera_is_safe,
        ),
        ConformanceCheck(
            "many_objects_are_survived", KitSection.FAILURE, _many_objects_are_survived
        ),
        # resource
        ConformanceCheck("memory_is_bounded", KitSection.RESOURCE, _memory_is_bounded, "T8"),
        ConformanceCheck(
            "track_count_stays_within_declared_maximum",
            KitSection.RESOURCE,
            _track_count_stays_within_declared_maximum,
            "T8",
        ),
        ConformanceCheck(
            "terminated_tracks_are_released",
            KitSection.RESOURCE,
            _terminated_tracks_are_released,
            "T8",
        ),
    ),
)

#: Determinism needs a *factory*, not an instance — it must compare two
#: independent runs from a clean state. Run separately by the plugin loader when
#: a factory is available; see ``TRACKER_KIT`` for everything checkable from a
#: single instance.
DETERMINISM_CHECK = ConformanceCheck(
    "determinism", KitSection.SEMANTICS, _determinism, "V13"
)
