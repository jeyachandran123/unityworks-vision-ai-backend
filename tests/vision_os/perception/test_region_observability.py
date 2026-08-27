"""P33 — the region-observability port and its first producer.

Driven by a **scripted session**, so the whole class is exercised without ONNX,
without weights and without a GPU. The measured accuracy of the real model is a
separate concern and lives in `tests/compliance/test_pose_gate_effect.py`, which
replays recorded verdicts; these tests are about the contract.

The distinction matters: a test that needs a 13 MB artefact to run is a test that
stops running.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.perception import PoseRegionObservability, PoseThresholds
from vision_os.conformance import REGION_OBSERVABILITY_KIT
from vision_os.core.model.ids import AttributeKey, CameraId
from vision_os.core.model.region_observability import RegionState, RegionVerdict
from vision_os.core.model.space import Box
from vision_os.core.ports.region_observability import (
    RegionObservabilityPort,
    RegionObservabilityRequest,
)

HEAD = AttributeKey("head_covering")
FACE = AttributeKey("face_covering")
HAND = AttributeKey("hand_covering")

WIDTH = HEIGHT = 64

#: A frozen value rather than an inline default: ``Box`` is immutable, but a
#: constructor call in a signature is evaluated once at import and reads as
#: though it were not.
SUBJECT = Box(0.2, 0.05, 0.7, 0.95)


class ScriptedPoseSession:
    """A pose model that returns exactly the skeletons it was handed.

    Emits the raw ``(1, 56, N)`` tensor shape the adapter decodes, so the
    decoding path — letterbox inverse, keypoint reshape, NMS, association — is
    under test rather than stubbed past.
    """

    def __init__(self, skeletons=(), input_size: int = 640) -> None:
        self._skeletons = skeletons
        self._input_size = input_size
        self.calls = 0

    def get_inputs(self):
        class _In:
            name = "images"

        return [_In()]

    def run(self, _outputs, feeds):
        import numpy as np

        self.calls += 1
        columns = []
        for box, score, points in self._skeletons:
            cx = (box[0] + box[2]) / 2 * self._input_size
            cy = (box[1] + box[3]) / 2 * self._input_size
            bw = (box[2] - box[0]) * self._input_size
            bh = (box[3] - box[1]) * self._input_size
            column = [cx, cy, bw, bh, score]
            for x, y, confidence in points:
                column += [x * self._input_size, y * self._input_size, confidence]
            columns.append(column)
        if not columns:
            return [np.zeros((1, 56, 0), dtype=np.float32)]
        return [np.array(columns, dtype=np.float32).T[None]]


def keypoints(head_confidence: float, *, body: float = 0.9):
    """17 COCO keypoints; the five head ones share ``head_confidence``."""
    points = []
    for index in range(17):
        confidence = head_confidence if index < 5 else body
        points.append((0.4, 0.2 if index < 5 else 0.6, confidence))
    return points


def skeleton(head_confidence: float, box=(0.2, 0.05, 0.7, 0.95), score: float = 0.9):
    return (box, score, keypoints(head_confidence))


def request(
    *,
    attributes=(HEAD,),
    box=SUBJECT,
    pixels: bool = True,
    frame_key: str = "f1",
) -> RegionObservabilityRequest:
    return RegionObservabilityRequest(
        camera_id=CameraId("cam-1"),
        box=box,
        attributes=tuple(attributes),
        source_width=WIDTH,
        source_height=HEIGHT,
        pixels=memoryview(bytes(WIDTH * HEIGHT * 3)) if pixels else None,
        frame_key=frame_key,
    )


def adapter(*skeletons, thresholds: PoseThresholds | None = None):
    return PoseRegionObservability(
        session=ScriptedPoseSession(skeletons), thresholds=thresholds
    )


class TestTheContract:
    def test_it_satisfies_the_port(self) -> None:
        assert isinstance(adapter(), RegionObservabilityPort)

    def test_it_passes_its_conformance_kit(self) -> None:
        report = REGION_OBSERVABILITY_KIT.run(adapter(skeleton(0.9)))
        assert report.passed, "\n".join(report.failures)

    def test_the_request_refuses_to_ask_nothing(self) -> None:
        """An empty attribute tuple would make 'one verdict per attribute' vacuous."""
        with pytest.raises(ValueError):
            request(attributes=())


class TestTheThreeStates:
    def test_a_confident_head_is_located(self) -> None:
        verdict = adapter(skeleton(0.9)).assess(request())[0]
        assert verdict.state is RegionState.LOCATED
        assert verdict.signals_seen == 5
        assert verdict.observable

    def test_a_weak_head_is_low_confidence_not_absent(self) -> None:
        """The case where the producer knows it is guessing.

        Folding this into LOCATED hands the model the crop most likely to be
        misread; folding it into NOT_LOCATED erases the distinction.
        """
        verdict = adapter(skeleton(0.35)).assess(request())[0]
        assert verdict.state is RegionState.LOW_CONFIDENCE
        assert not verdict.observable
        assert "below the 0.50 floor" in verdict.detail

    def test_no_head_signal_at_all_is_not_located(self) -> None:
        verdict = adapter(skeleton(0.02)).assess(request())[0]
        assert verdict.state is RegionState.NOT_LOCATED
        assert not verdict.observable

    def test_no_skeleton_overlapping_the_subject_is_not_located(self) -> None:
        """A head borrowed from the person behind is worse than no head at all."""
        elsewhere = skeleton(0.99, box=(0.80, 0.80, 0.95, 0.95))
        verdict = adapter(elsewhere).assess(request())[0]
        assert verdict.state is RegionState.NOT_LOCATED
        assert "overlapped" in verdict.detail

    def test_an_empty_frame_is_not_located(self) -> None:
        verdict = adapter().assess(request())[0]
        assert verdict.state is RegionState.NOT_LOCATED


class TestTheSemanticCeiling:
    def test_no_state_can_express_a_covering(self) -> None:
        """O3, checked on the type rather than on any one adapter.

        A producer able to say 'uncovered' would be a second attribute source
        sitting outside the registry's neutrality gate.
        """
        values = {state.value for state in RegionState}
        assert values == {"located", "low_confidence", "not_located", "unsupported"}

    def test_hand_covering_is_unsupported_not_refused(self) -> None:
        """Wrist keypoints locate a wrist.

        The policy's own wording is that *'a visible forearm, sleeve or cuff is
        NOT a visible hand'*, so claiming this attribute would answer a different
        question than the one asked.
        """
        verdict = adapter(skeleton(0.9)).assess(request(attributes=(HAND,)))[0]
        assert verdict.state is RegionState.UNSUPPORTED
        assert verdict.observable, "an unassessed attribute must behave as it did before"

    def test_face_covering_rides_the_same_verdict_as_head(self) -> None:
        """It shares head_covering's band character for character, so it shares
        the head's observability — one inference answers both."""
        verdicts = adapter(skeleton(0.9)).assess(request(attributes=(HEAD, FACE)))
        assert [v.state for v in verdicts] == [RegionState.LOCATED, RegionState.LOCATED]

    def test_a_refusal_must_name_its_reason(self) -> None:
        with pytest.raises(ValueError, match="name why"):
            RegionVerdict(attribute=HEAD, state=RegionState.NOT_LOCATED)


class TestObligations:
    def test_one_verdict_per_attribute_in_request_order(self) -> None:
        keys = (HAND, HEAD, FACE)
        verdicts = adapter(skeleton(0.9)).assess(request(attributes=keys))
        assert tuple(v.attribute for v in verdicts) == keys

    def test_missing_pixels_is_unsupported_never_a_refusal(self) -> None:
        """A plumbing gap must not become a perception result."""
        verdict = adapter(skeleton(0.9)).assess(request(pixels=False))[0]
        assert verdict.state is RegionState.UNSUPPORTED
        assert verdict.observable

    def test_the_geometry_type_refuses_a_zero_area_box_outright(self) -> None:
        """The adapter's own area guard is unreachable through ``Box``, and that
        is the stronger arrangement: degenerate geometry cannot be constructed,
        so no producer has to remember to check for it."""
        with pytest.raises(ValueError, match="degenerate box"):
            Box(0.0, 0.0, 0.0, 0.0)

    def test_a_sliver_of_a_box_refuses_rather_than_raising(self) -> None:
        """Legal but extreme (O5): a fraction of a pixel across."""
        verdict = adapter(skeleton(0.9)).assess(
            request(box=Box(0.5, 0.5, 0.5 + 1e-6, 0.5 + 1e-6))
        )[0]
        assert verdict.state is RegionState.NOT_LOCATED
        assert verdict.detail

    def test_inference_failure_degrades_rather_than_dying(self) -> None:
        """V9. A pose model that throws must not take the frame down with it."""

        class Exploding:
            def get_inputs(self):
                raise RuntimeError("session is gone")

            def run(self, *_args, **_kwargs):
                raise RuntimeError("session is gone")

        producer = PoseRegionObservability(session=Exploding())
        verdict = producer.assess(request())[0]
        assert verdict.state is RegionState.NOT_LOCATED
        assert "pose inference failed" in verdict.detail

    def test_one_inference_serves_every_subject_in_a_frame(self) -> None:
        session = ScriptedPoseSession((skeleton(0.9),))
        producer = PoseRegionObservability(session=session)
        for _ in range(4):
            producer.assess(request(frame_key="same-frame"))
        assert session.calls == 1, "N subjects in a frame must cost one inference"

    def test_a_new_frame_is_re_inferred(self) -> None:
        session = ScriptedPoseSession((skeleton(0.9),))
        producer = PoseRegionObservability(session=session)
        producer.assess(request(frame_key="frame-1"))
        producer.assess(request(frame_key="frame-2"))
        assert session.calls == 2


class TestTheOperatingPoint:
    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.90, RegionState.LOCATED),
            (0.50, RegionState.LOCATED),
            (0.49, RegionState.LOW_CONFIDENCE),
            (0.25, RegionState.LOW_CONFIDENCE),
            (0.24, RegionState.NOT_LOCATED),
        ],
    )
    def test_the_floor_and_the_half_floor(self, confidence, expected) -> None:
        """0.50 is the floor; half of it separates 'weak signal' from 'no signal'.

        Reported as the default rather than asserted as optimal: Phase 4.4
        measured the curve flat from 0.30 to 0.50 and steep above it, and with 11
        negative examples tuning further would be fitting noise.
        """
        assert adapter(skeleton(confidence)).assess(request())[0].state is expected

    def test_the_floor_is_configuration(self) -> None:
        strict = PoseThresholds(keypoint_confidence=0.6)
        assert (
            adapter(skeleton(0.55), thresholds=strict).assess(request())[0].state
            is RegionState.LOW_CONFIDENCE
        )

    def test_thresholds_must_be_probabilities(self) -> None:
        with pytest.raises(ValueError):
            PoseThresholds(keypoint_confidence=1.5)


class TestResampling:
    def test_downscaling_averages_rather_than_sampling(self) -> None:
        """Area-average, and it is load-bearing.

        At this camera's 1712->640 the scale is 0.374; nearest-neighbour keeps
        7 % of the pixels and takes whichever happens to land on the grid.
        Measured on kitchen-01 that cost the corpus's only true violation —
        subject f01500/s2's best head keypoint fell 0.59 -> 0.46, under the floor.
        """
        import numpy as np

        from vision_os.adapters.perception.pose import _resample

        # A 4x4 checkerboard halved: every output cell averages one 2x2 block, so
        # a correct downsample is uniformly 127/128 and a sampling one is 0 or 255.
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[::2, ::2] = 255
        frame[1::2, 1::2] = 255

        out = _resample(frame, 2, 2)
        assert out.shape == (2, 2, 3)
        assert out.min() >= 127 and out.max() <= 128, (
            f"expected the mean of each 2x2 block, got {out[:, :, 0].tolist()}"
        )

    def test_upscaling_does_not_average(self) -> None:
        import numpy as np

        from vision_os.adapters.perception.pose import _resample

        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0] = 255
        out = _resample(frame, 4, 4)
        assert out.shape == (4, 4, 3)
        assert set(np.unique(out).tolist()) <= {0, 255}, (
            "there is no information to average when upscaling; inventing "
            "intermediate values would invent detail"
        )
