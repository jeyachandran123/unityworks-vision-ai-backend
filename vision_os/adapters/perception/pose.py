"""Pose as a region-observability producer — P34's first adapter.

Promoted from ``tools/vision_eval/pose_observability.py``, which measured it
before anything was wired. The numbers below are that measurement, on
``datasets/kitchen-01`` (43 subjects, 15 frames, human head-location ground
truth), scored **only** against human annotation — hairnet labels, VLM output and
compliance verdicts were excluded, because using them would make the evaluation
circular:

| | |
|---|---|
| precision | **96.6 %** |
| recall | 87.5 % |
| unsafe acceptance (claimed a head where a human saw none) | **1 / 11** |
| false refusal (denied a head a human could read) | 4 / 32 |
| latency | 64 ms/frame, CPU |
| new dependencies | **none** — same onnxruntime, same input geometry |

### The head point, and why it is an average

``HEAD_KEYPOINTS`` is nose, both eyes, both ears. The head point is the
**confidence-weighted mean of whichever of those five cleared the floor** — one
definition, applied identically to every subject.

Averaging rather than preferring the nose is consequential, not stylistic: a
worker facing a counter has no visible nose and two visible ears, and that is
still a located head. A nose-anchored definition would refuse most of this
kitchen and discard evidence the camera plainly has.

### What this adapter must never do

It reports **where a head is**, never **what is on it** (port obligation O3).
There is no code path here that can express a covering, and the port's state enum
offers no value for one. That is what keeps a second attribute producer from
growing outside the registry's neutrality gate.

### Scope

``head_covering`` and ``face_covering`` only — both are questions about the head,
and pose locates heads. ``hand_covering`` is deliberately **not** claimed: wrist
keypoints locate a wrist, and *"a visible forearm, sleeve or cuff is NOT a visible
hand"* is the policy's own wording. Claiming it would answer a different question
than the one asked. Everything unclaimed returns ``UNSUPPORTED`` and behaves
exactly as it did before this adapter existed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.model.ids import AttributeKey
from ...core.model.region_observability import RegionState, RegionVerdict
from ...core.model.space import Box
from ...core.ports.region_observability import (
    RegionObservabilityCapabilities,
    RegionObservabilityRequest,
)

#: COCO-17 indices of the five keypoints that constitute a head.
HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left/right eye, left/right ear

#: Attributes this producer will speak to. Everything else is UNSUPPORTED.
HEAD_ATTRIBUTES = frozenset(
    {AttributeKey("head_covering"), AttributeKey("face_covering")}
)

#: Keypoint confidence at or above which a head keypoint counts as seen.
#:
#: **Reported as the default, not asserted as optimal.** Phase 4.4 measured both
#: sides: the curve is flat from 0.30 to 0.50 (1 unsafe acceptance, 4 false
#: refusals throughout) and then trades steeply — 0.60 eliminates the last unsafe
#: acceptance at the cost of two further valid observations. With 11 negative
#: examples, tuning past this would be fitting to noise, which is the same
#: evidential mistake the policy file already flags for its provisional hand
#: floors.
DEFAULT_KEYPOINT_CONFIDENCE = 0.5

#: Person-box score below which a pose candidate is discarded before matching.
DEFAULT_PERSON_CONFIDENCE = 0.25

#: IoU below which a pose skeleton is not considered the same subject as the
#: detector's box. Above this the two are the same person; below it, the pose
#: model simply did not find this subject and the honest answer is NOT_LOCATED.
DEFAULT_ASSOCIATION_IOU = 0.5

#: Greedy NMS threshold over the raw head's 8400 columns.
DEFAULT_NMS_IOU = 0.6

_POSE_INPUT_SIZE = 640


@dataclass(frozen=True, slots=True)
class PoseThresholds:
    """Every number this adapter decides with, in one reviewable place."""

    keypoint_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE
    person_confidence: float = DEFAULT_PERSON_CONFIDENCE
    association_iou: float = DEFAULT_ASSOCIATION_IOU
    nms_iou: float = DEFAULT_NMS_IOU

    def __post_init__(self) -> None:
        for name in (
            "keypoint_confidence",
            "person_confidence",
            "association_iou",
            "nms_iou",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")


class PoseRegionObservability:
    """Locate heads with a pose model, so an unseen head is refused, not guessed.

    The ONNX session is **injected**, not opened here: this adapter never touches
    a filesystem, so the composition root stays the only thing that decides which
    artefact is bound — and a test can drive the whole class with a scripted
    session and no model at all.
    """

    __slots__ = ("_frame_cache", "_producer_id", "_session", "_thresholds")

    def __init__(
        self,
        *,
        session,
        thresholds: PoseThresholds | None = None,
        producer_id: str = "observability.pose.yolov8n",
    ) -> None:
        self._session = session
        self._thresholds = thresholds or PoseThresholds()
        self._producer_id = producer_id
        # One inference serves every subject in a frame. Bounded to a single
        # entry: subjects of one frame arrive together, and holding more would
        # make this adapter a cache with a retention question attached.
        self._frame_cache: tuple[str, tuple] | None = None

    # --- port ------------------------------------------------------------------ #

    def capabilities(self) -> RegionObservabilityCapabilities:
        return RegionObservabilityCapabilities(
            producer_id=self._producer_id,
            assessable_attributes=HEAD_ATTRIBUTES,
            requires_pixels=True,
            deterministic=True,
        )

    def assess(
        self, request: RegionObservabilityRequest
    ) -> Sequence[RegionVerdict]:
        """One verdict per requested attribute, in request order (O1)."""
        wanted = [k for k in request.attributes if k in HEAD_ATTRIBUTES]
        if not wanted:
            return tuple(self._unsupported(k) for k in request.attributes)

        if request.pixels is None:
            return tuple(
                self._unsupported(k, "no frame pixels were supplied")
                if k in HEAD_ATTRIBUTES
                else self._unsupported(k)
                for k in request.attributes
            )

        state, confidence, seen, detail = self._locate_head(request)
        return tuple(
            RegionVerdict(
                attribute=key,
                state=state,
                confidence=confidence,
                signals_seen=seen,
                producer_id=self._producer_id,
                detail=detail,
            )
            if key in HEAD_ATTRIBUTES
            else self._unsupported(key)
            for key in request.attributes
        )

    # --- the measurement ------------------------------------------------------- #

    def _locate_head(
        self, request: RegionObservabilityRequest
    ) -> tuple[RegionState, float, int, str]:
        """Find this subject's head, or say honestly that there is none to find.

        Never raises (O5): a degenerate box, an empty frame or a model that
        returned nothing are all ``NOT_LOCATED`` with a reason.
        """
        if request.box.area <= 0.0:
            return (
                RegionState.NOT_LOCATED,
                0.0,
                0,
                "the subject box has no area",
            )

        try:
            candidates = self._skeletons(request)
        except Exception as error:  # noqa: BLE001 - degrade, never die (V9)
            return (
                RegionState.NOT_LOCATED,
                0.0,
                0,
                f"pose inference failed: {type(error).__name__}",
            )

        matched = self._match(request.box, candidates)
        if matched is None:
            return (
                RegionState.NOT_LOCATED,
                0.0,
                0,
                "no pose skeleton overlapped the subject box",
            )

        keypoints = matched
        floor = self._thresholds.keypoint_confidence
        seen = [keypoints[i] for i in HEAD_KEYPOINTS if keypoints[i][2] >= floor]

        if seen:
            weight = sum(k[2] for k in seen)
            best = max(k[2] for k in seen)
            _ = weight  # the head point itself is not needed to decide observability
            return (RegionState.LOCATED, min(1.0, best), len(seen), "")

        # Nothing cleared the floor. Distinguish "some signal, all weak" from "no
        # signal at all": the first is a near-miss worth recording, the second is
        # a confident absence.
        best = max(keypoints[i][2] for i in HEAD_KEYPOINTS)
        if best >= floor / 2.0:
            return (
                RegionState.LOW_CONFIDENCE,
                min(1.0, best),
                0,
                f"best head keypoint {best:.2f} is below the {floor:.2f} floor",
            )
        return (
            RegionState.NOT_LOCATED,
            min(1.0, max(0.0, best)),
            0,
            f"no head keypoint above {floor / 2.0:.2f}; best was {best:.2f}",
        )

    def _skeletons(self, request: RegionObservabilityRequest) -> tuple:
        """Every person skeleton in the frame, in normalized coordinates.

        Cached by ``frame_key`` so N subjects in one frame cost one inference.
        """
        key = request.frame_key
        if key and self._frame_cache is not None and self._frame_cache[0] == key:
            return self._frame_cache[1]

        result = tuple(
            self._infer(
                request.pixels,
                width=request.source_width,
                height=request.source_height,
            )
        )
        if key:
            self._frame_cache = (key, result)
        return result

    def _infer(self, pixels, *, width: int, height: int) -> list:
        """Run the model and return ``[(box, score, keypoints)]``.

        Preprocessing reuses the platform's own letterbox arithmetic, so the pose
        model sees exactly what the detector sees — the property that makes the
        two models' boxes comparable at all.
        """
        import numpy as np  # noqa: PLC0415 - optional, adapter-scoped

        from ..detection.letterbox import compute_transform  # noqa: PLC0415

        if width <= 0 or height <= 0:
            return []

        frame = np.frombuffer(pixels, dtype=np.uint8)
        if frame.size < width * height * 3:
            return []
        frame = frame[: width * height * 3].reshape(height, width, 3)

        transform = compute_transform(
            source_width=width,
            source_height=height,
            target_width=_POSE_INPUT_SIZE,
            target_height=_POSE_INPUT_SIZE,
        )
        scaled_w = max(1, round(width * transform.scale))
        scaled_h = max(1, round(height * transform.scale))
        scaled = _resample(frame, scaled_h, scaled_w)

        canvas = np.full((_POSE_INPUT_SIZE, _POSE_INPUT_SIZE, 3), 114, dtype=np.uint8)
        ox, oy = int(transform.pad_x), int(transform.pad_y)
        canvas[oy : oy + scaled_h, ox : ox + scaled_w] = scaled

        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        raw = self._session.run(None, {self._input_name(): tensor})[0][0]  # (56, N)

        out = []
        for column in raw.T:
            score = float(column[4])
            if score < self._thresholds.person_confidence:
                continue
            cx, cy, bw, bh = (float(v) for v in column[:4])
            box = (
                (cx - bw / 2 - transform.pad_x) / transform.scale / width,
                (cy - bh / 2 - transform.pad_y) / transform.scale / height,
                (cx + bw / 2 - transform.pad_x) / transform.scale / width,
                (cy + bh / 2 - transform.pad_y) / transform.scale / height,
            )
            kp = column[5:].reshape(17, 3).astype(float)
            points = tuple(
                (
                    (float(p[0]) - transform.pad_x) / transform.scale / width,
                    (float(p[1]) - transform.pad_y) / transform.scale / height,
                    float(p[2]),
                )
                for p in kp
            )
            out.append((box, score, points))
        return _suppress(out, self._thresholds.nms_iou)

    def _input_name(self) -> str:
        return self._session.get_inputs()[0].name

    def _match(self, box: Box, candidates: Sequence) -> tuple | None:
        """The skeleton that is this subject, or ``None``.

        Association is by IoU against the detector's box. Below the floor the
        honest statement is *"the pose model did not find this subject"*, never
        *"the nearest skeleton will do"* — a head borrowed from the person behind
        is worse than no head at all.
        """
        best: tuple | None = None
        best_iou = self._thresholds.association_iou
        for candidate_box, _score, points in candidates:
            overlap = _iou(
                (box.x1, box.y1, box.x2, box.y2), candidate_box
            )
            if overlap >= best_iou:
                best_iou = overlap
                best = points
        return best

    def _unsupported(self, key: AttributeKey, detail: str = "") -> RegionVerdict:
        return RegionVerdict(
            attribute=key,
            state=RegionState.UNSUPPORTED,
            producer_id=self._producer_id,
            detail=detail or "this producer does not assess this attribute",
        )


def _resample(frame, target_h: int, target_w: int):
    """Resize to the letterbox's inner rectangle, correctly for the direction.

    **Area-average when downscaling, and this is load-bearing.** This camera is
    1712x1032 into a 640 network, a scale of 0.374 — nearest-neighbour throws
    away 93 % of the pixels and takes whichever one happens to land on the grid.
    Measured on `datasets/kitchen-01`, that cost the one true violation in the
    corpus: subject `f01500/s2`'s best head keypoint fell from 0.59 to 0.46 and
    dropped below the confidence floor, turning a correct violation into a
    refusal. Aggregate accuracy was unchanged, which is exactly why the aggregate
    was not enough to catch it.

    Implemented with two ``reduceat`` passes rather than an integral image: a
    cumulative sum over 1.7M pixels overflows float32's mantissa, and the float64
    version costs 42 MB per frame to avoid a problem this does not have.

    Upscaling stays nearest-neighbour — there is no information to average, and a
    source smaller than the network is not this deployment's case anyway.
    """
    import numpy as np  # noqa: PLC0415 - optional, adapter-scoped

    height, width = frame.shape[0], frame.shape[1]
    if target_h >= height or target_w >= width:
        ys = (np.arange(target_h) * height // max(1, target_h)).clip(0, height - 1)
        xs = (np.arange(target_w) * width // max(1, target_w)).clip(0, width - 1)
        return frame[ys][:, xs]

    source = frame.astype(np.float32)

    row_starts = (np.arange(target_h) * height // target_h).astype(np.intp)
    rows = np.add.reduceat(source, row_starts, axis=0)
    rows /= np.diff(np.append(row_starts, height))[:, None, None]

    col_starts = (np.arange(target_w) * width // target_w).astype(np.intp)
    cells = np.add.reduceat(rows, col_starts, axis=1)
    cells /= np.diff(np.append(col_starts, width))[None, :, None]

    return cells.round().clip(0, 255).astype(np.uint8)


def _iou(a: tuple, b: tuple) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _suppress(candidates: list, iou_threshold: float) -> list:
    """Greedy suppression over person boxes. The raw head has 8400 columns."""
    kept: list = []
    for entry in sorted(candidates, key=lambda c: -c[1]):
        if all(_iou(entry[0], other[0]) < iou_threshold for other in kept):
            kept.append(entry)
    return kept
