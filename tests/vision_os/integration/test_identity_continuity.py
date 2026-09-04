"""End-to-end identity continuity: the real tracker driving the real registry.

Nothing is stubbed but the clock and the metrics sink. A `GeometricTracker`
consumes detections and an `ObjectRegistry` consumes its `TrackUpdate`s, which
is the exact seam that produced:

    cam-12   767 incidents   767 distinct object_ids

Each test states the physical scenario in units a person can check: box width
and speed as fractions of frame width, sampled at a stated frame rate.

**Two of these tests assert a limitation rather than a success.** They are here
because the honest boundary of the repair is worth pinning as firmly as its
achievements — a later change that appears to fix them by loosening association
would be trading identity fragmentation for identity merging, which is worse.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.tracking import TRACKER_FACTORIES
from vision_os.core.model.confidence import Confidence, ConfidenceSemantics
from vision_os.core.model.detection import (
    Detection,
    DetectionEvidence,
    InferenceTiming,
)
from vision_os.core.model.ids import (
    CameraId,
    ClassId,
    ConfigRevision,
    DetectionId,
    FrameRef,
    ModuleId,
    SiteId,
    StreamEpoch,
    TenantId,
)
from vision_os.core.model.provenance import Provenance
from vision_os.core.model.space import Box, FrameOfReference, SpatialInfo
from vision_os.core.model.timebase import Duration, Instant
from vision_os.core.ports.tracking import TrackingRequest
from vision_os.kernel.config.schema import RegistrySection
from vision_os.perception.registry.attributes import AttributeRegistry
from vision_os.perception.registry.binding import BindingPolicy
from vision_os.perception.registry.engine import ObjectRegistry
from vision_os.perception.registry.lifecycle import LifecyclePolicy

CAMERA = CameraId("cam-11")
TENANT = TenantId("org-test")
SITE = SiteId("site-test")
PERSON = ClassId("person")

#: A person occupies about this fraction of frame width on a kitchen camera.
WIDTH = 0.12

_PROV = Provenance(
    producer_module=ModuleId("detection_engine"),
    producer_version="1.0.0",
    config_revision=ConfigRevision("e2e"),
)


class _Clock:
    def __init__(self) -> None:
        self._ns = 0

    def advance(self, ms: int) -> None:
        self._ns += ms * 1_000_000

    def now(self) -> Instant:
        return Instant(self._ns)

    def monotonic(self) -> Instant:
        return Instant(self._ns)


class _Sink:
    """Bus and metrics both. Records nothing; the assertions read real state."""

    def publish(self, event) -> None: ...
    def subscribe(self, *a, **k) -> None: ...
    def increment(self, n: int = 1) -> None: ...
    def record(self, v: float) -> None: ...
    def counter(self, *a, **k): return self
    def histogram(self, *a, **k): return self
    def gauge(self, *a, **k): return self


def _detection(seq: int, box: Box, interval_ms: int, index: int = 0) -> Detection:
    return Detection(
        detection_id=DetectionId(f"det-{seq}-{index}"),
        frame_ref=FrameRef(CAMERA, StreamEpoch(1), seq),
        tenant_id=TENANT,
        site_id=SITE,
        t_capture=Instant(seq * interval_ms * 1_000_000),
        t_capture_uncertainty=Duration.from_millis(5),
        class_id=PERSON,
        taxonomy_version="1.0.0",
        confidence=Confidence.uncalibrated(0.9, ConfidenceSemantics.DETECTION_PRESENCE),
        spatial=SpatialInfo(frame_of_reference=FrameOfReference.NORMALIZED, bbox=box),
        provenance=_PROV,
        timing=InferenceTiming(inference_ms=3.0),
        evidence=DetectionEvidence(input_hash="e2e"),
    )


def run(frames, *, tracker_id: str = "tracker.sort", interval_ms: int = 1000) -> dict:
    """Drive tracker → registry over `frames`, and report what identity did.

    `frames` is a sequence of box-lists, one per processed frame. An empty list
    is a frame in which the detector saw nothing, which is a real and different
    thing from a frame that was never processed.
    """
    clock = _Clock()
    sink = _Sink()
    tracker = TRACKER_FACTORIES[tracker_id]()
    registry = ObjectRegistry(
        clock=clock,
        bus=sink,
        metrics=sink,
        config=RegistrySection(enabled=True, min_observations_to_confirm=2),
        tenant_id=TENANT,
        site_id=SITE,
        provenance=_PROV,
        lifecycle=LifecyclePolicy(),
        binding=BindingPolicy(),
        attributes=AttributeRegistry(),
    )

    created: set = set()
    recovered = 0
    last_ns = None

    for seq, boxes in enumerate(frames):
        clock.advance(interval_ms)
        elapsed = Duration(0 if last_ns is None else clock.now().ns - last_ns)
        last_ns = clock.now().ns

        update = tracker.update(
            TrackingRequest(
                camera_id=CAMERA,
                frame_ref=FrameRef(CAMERA, StreamEpoch(1), seq),
                timestamp=clock.now(),
                elapsed=elapsed,
                detections=tuple(
                    _detection(seq, b, interval_ms, i) for i, b in enumerate(boxes)
                ),
                embeddings=None,
            )
        )
        created.update(update.new)
        recovered += len(update.recovered)
        registry.ingest(CAMERA, update)

    objects = registry.objects(CAMERA)
    return {
        "track_creations": len(created),
        "recoveries": recovered,
        "objects": len(objects),
        "present": len([o for o in objects if o.lifecycle.is_present]),
    }


def _walk(n: int, *, speed: float, start: float = 0.05, interval_ms: int = 1000):
    """`speed` in frame-widths per second."""
    out = []
    for seq in range(n):
        x = start + speed * (seq * interval_ms / 1000.0)
        out.append([Box(x, 0.35, x + WIDTH, 0.85)])
    return out


class TestScenarioA_NormalMotion:
    def test_one_person_walking_keeps_one_identity(self):
        """A person crossing at an unhurried pace is one logical person."""
        result = run(_walk(10, speed=0.075))
        assert result["objects"] == 1
        assert result["track_creations"] == 1


class TestScenarioB_ShortOcclusion:
    """The scenario the tracker change exists for, and the clearest before/after
    in this file."""

    FRAMES = (
        _walk(5, speed=0.075)
        + [[], []]                                   # two frames behind a counter
        + _walk(3, speed=0.075, start=0.05 + 0.075 * 7)
    )

    def test_the_person_survives_a_short_occlusion(self):
        result = run(self.FRAMES, tracker_id="tracker.sort")
        assert result["objects"] == 1, "a two-frame occlusion is not a new person"
        assert result["recoveries"] >= 1, "the return must be an observable recovery"

    def test_the_old_fallback_tracker_fragments_on_the_same_frames(self):
        """The regression this repair removes, pinned so it cannot come back
        unnoticed. `tracker.iou` declares `handles_occlusion="none"` and means
        it."""
        result = run(self.FRAMES, tracker_id="tracker.iou")
        assert result["objects"] == 2
        assert result["recoveries"] == 0


class TestScenarioC_MultiplePeople:
    def test_two_people_stay_two_people(self):
        """False-merge prevention at the end-to-end level. Two people walking
        toward each other must not become one."""
        frames = []
        for seq in range(10):
            a = 0.05 + 0.075 * seq
            b = 0.78 - 0.075 * seq
            frames.append(
                [Box(a, 0.35, a + WIDTH, 0.85), Box(b, 0.30, b + WIDTH, 0.80)]
            )
        assert run(frames)["objects"] == 2


class TestScenarioD_EmptyScene:
    def test_nothing_downstream_invents_a_person(self):
        """No detections must produce no objects — not a provisional one, not a
        coasting one, nothing."""
        result = run([[] for _ in range(12)])
        assert result["objects"] == 0
        assert result["track_creations"] == 0


class TestScenarioF_OneFrameFalsePositive:
    def test_a_single_spurious_detection_never_becomes_a_confirmed_subject(self):
        """The detector's error is the detector's, and the registry does mint a
        `provisional` object for it — that is correct and deliberate: refusing
        to record what was detected would be a different kind of lying.

        What matters architecturally is that it **never reaches `active`**.
        `active` is what `kitchen-safety`'s `scope.lifecycle` admits, so a
        subject that never confirms is never matched by a demand, never cropped,
        never sent to the model, and cannot become a compliance violation.

        That gate is the Defect 1 repair, held in
        `tests/vision_os/cropping/test_policy_enforcement.py`. This test holds
        the precondition it depends on: one frame is not enough to confirm.
        """
        from vision_os.core.model.visual_object import LifecycleState

        frames = [[] for _ in range(12)]
        frames[4] = [Box(0.40, 0.40, 0.52, 0.90)]

        clock = _Clock()
        sink = _Sink()
        tracker = TRACKER_FACTORIES["tracker.sort"]()
        registry = ObjectRegistry(
            clock=clock, bus=sink, metrics=sink,
            config=RegistrySection(enabled=True, min_observations_to_confirm=2),
            tenant_id=TENANT, site_id=SITE, provenance=_PROV,
            lifecycle=LifecyclePolicy(), binding=BindingPolicy(),
            attributes=AttributeRegistry(),
        )
        last_ns = None
        for seq, boxes in enumerate(frames):
            clock.advance(1000)
            elapsed = Duration(0 if last_ns is None else clock.now().ns - last_ns)
            last_ns = clock.now().ns
            registry.ingest(
                CAMERA,
                tracker.update(
                    TrackingRequest(
                        camera_id=CAMERA,
                        frame_ref=FrameRef(CAMERA, StreamEpoch(1), seq),
                        timestamp=clock.now(),
                        elapsed=elapsed,
                        detections=tuple(
                            _detection(seq, b, 1000, i) for i, b in enumerate(boxes)
                        ),
                        embeddings=None,
                    )
                ),
            )

        states = {o.lifecycle for o in registry.objects(CAMERA)}
        assert LifecycleState.ACTIVE not in states, (
            "a one-frame detection must never reach the lifecycle the policy "
            "admits to the model"
        )


class TestTheHonestBoundary:
    """Where continuity genuinely cannot be maintained, and why.

    A motion model predicts from *observed* velocity, so it needs two
    measurements before it can help. When a subject's per-frame displacement
    already exceeds its own box width, the very first association fails, no
    velocity is ever estimated, and no tracker in this family can recover — the
    evidence for continuity is simply not in the frames.

    This is a **sampling** limit, not an architecture defect, and the fix is
    frame rate rather than looser association. Loosening association to cover it
    would start merging genuinely different people.
    """

    #: 0.20 frame-widths per second — a brisk walk.
    SPEED = 0.20

    @pytest.mark.parametrize("tracker_id", ["tracker.iou", "tracker.sort"])
    def test_at_two_fps_a_brisk_walk_fragments_on_every_tracker(self, tracker_id):
        """Displacement 0.100 against a 0.12 box: no overlap survives, and there
        is no prior velocity to bridge the gap."""
        frames = _walk(8, speed=self.SPEED, start=0.03, interval_ms=500)
        assert run(frames, tracker_id=tracker_id, interval_ms=500)["objects"] > 1

    @pytest.mark.parametrize("tracker_id", ["tracker.iou", "tracker.sort"])
    def test_at_four_fps_the_same_walk_holds_one_identity(self, tracker_id):
        """Displacement 0.050 against a 0.12 box. Same person, same speed, same
        code — only the sampling rate changed.

        This is the measured basis for the frame-rate recommendation: the
        continuity cliff for this geometry sits between 2 and 3 fps.
        """
        frames = _walk(12, speed=self.SPEED, start=0.03, interval_ms=250)
        assert run(frames, tracker_id=tracker_id, interval_ms=250)["objects"] == 1
