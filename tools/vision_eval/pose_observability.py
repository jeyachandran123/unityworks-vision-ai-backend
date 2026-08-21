"""Can a pose model tell us whether a worker's head is *observable*?

    python -m tools.vision_eval.pose_observability datasets/kitchen-01

This is a **safety** experiment, not an accuracy one. The question is not "is the
head covered" — it is the prior question the system currently never asks:

    is there a head here to look at at all?

Today the pipeline crops the top of a person box, hands it to a model, and
accepts whatever comes back. When the worker is bent over a counter and the head
is nowhere in the frame, that crop shows a back, the model answers "no covering",
and the rule turns it into a violation against a compliant worker. A signal that
says *"no head here"* lets the system refuse instead of guess.

**Nothing here can recover a head the camera did not see.** Phase 4.3 confirmed
at least 5 of the 11 unobservable heads are simply not in the image. The value on
offer is honest refusal, and this file measures whether pose delivers it.

Lives under ``tools/`` and is imported by no runtime code. It reads the platform's
own letterbox helper so preprocessing is identical to the detector's, and it is
scored against **human head-location annotation only** — never against hairnet
labels, VLM output or compliance results, which would make the evaluation
circular (§12).
"""

from __future__ import annotations

import argparse
import enum
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: COCO-17 keypoint indices that lie on the head.
#:
#: Named explicitly rather than "the first five" so the definition survives a
#: model whose ordering differs, and so a reader can see exactly what "head"
#: means here (§14).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

#: Per-keypoint confidence at or above which a head keypoint counts as seen.
#:
#: A calibration constant, reported alongside every result and swept in the
#: output so the operating point is a choice the reader can inspect rather than
#: a number buried in code.
DEFAULT_KEYPOINT_CONFIDENCE = 0.5

#: Person-detection confidence floor for the pose model, matched to the
#: detector's own floor so the two see the same population.
POSE_PERSON_CONFIDENCE = 0.35

#: Overlap at which a pose person is taken to be the same subject as an
#: annotated one. Same value the detector benchmark uses (§13).
ASSOCIATION_IOU = 0.5


class HeadObservability(enum.Enum):
    """What pose can say about a head. Three states, never two.

    Collapsing ``LOW_CONFIDENCE`` into either neighbour throws away the only
    case where the system knows it is guessing — which is the case this whole
    experiment exists to surface.
    """

    LOCATED = "head_located"
    LOW_CONFIDENCE = "head_low_confidence"
    NOT_LOCATED = "head_not_located"

    @property
    def permits_a_covering_claim(self) -> bool:
        """Only a confidently located head may be asked about.

        ``LOW_CONFIDENCE`` deliberately returns False: a head the pose model is
        unsure about is exactly the head a VLM will confidently misread.
        """
        return self is HeadObservability.LOCATED


@dataclass(frozen=True, slots=True)
class HeadEstimate:
    """One subject's head, as pose sees it."""

    observability: HeadObservability
    point: tuple[float, float] | None
    """Head centre in normalized frame coordinates, or ``None``."""

    confidence: float
    keypoints_seen: int
    """How many of the five head keypoints cleared the confidence floor.
    Reported because one keypoint and five are different kinds of evidence."""

    association_iou: float = 0.0


def _load_dataset_module(dataset: Path, name: str):
    """Import a dataset-local Python module by path, without touching `sys.path`.

    Ground-truth datasets ship small Python files (`labels.py`, `head_locations.py`)
    that are data, not library code. Loading them by spec keeps them out of the
    global module namespace, so two datasets with the same filename cannot shadow
    one another inside a single process.
    """
    import importlib.util

    path = Path(dataset) / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required and was not found")

    spec = importlib.util.spec_from_file_location(f"_dataset_{dataset.name}_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head_estimate(
    keypoints: np.ndarray, threshold: float = DEFAULT_KEYPOINT_CONFIDENCE
) -> HeadEstimate:
    """Derive a head point from the facial keypoints (§14).

    The head centre is the **confidence-weighted mean of whichever of the five
    head keypoints were seen**. A single definition, applied identically to every
    subject: nose alone when only the nose is visible, the centroid of eyes and
    ears when the face is turned, and nothing at all when none clear the floor.

    Averaging rather than preferring the nose because a worker facing away has no
    nose but two visible ears, and that is still a located head. Preferring the
    nose would report those as unobservable and refuse evidence the camera
    plainly has.
    """
    seen = [(i, keypoints[i]) for i in HEAD_KEYPOINTS if keypoints[i][2] >= threshold]
    if not seen:
        # Nothing on the head cleared the floor. Distinguish "some signal, all
        # weak" from "no signal at all": the first is worth recording as a
        # near-miss, the second is a confident absence.
        best = max(float(keypoints[i][2]) for i in HEAD_KEYPOINTS)
        state = (
            HeadObservability.LOW_CONFIDENCE
            if best >= threshold / 2.0
            else HeadObservability.NOT_LOCATED
        )
        return HeadEstimate(state, None, best, 0)

    weight = sum(float(k[2]) for _, k in seen)
    x = sum(float(k[0]) * float(k[2]) for _, k in seen) / weight
    y = sum(float(k[1]) * float(k[2]) for _, k in seen) / weight
    return HeadEstimate(
        HeadObservability.LOCATED,
        (x, y),
        max(float(k[2]) for _, k in seen),
        len(seen),
    )


def run_pose(image, weights: Path, session=None):
    """One frame through the pose model, in normalized frame coordinates.

    Returns ``[(box, person_score, keypoints)]``. Preprocessing reuses the
    platform's own letterbox maths so the pose model sees exactly what the
    detector sees.
    """
    import onnxruntime as ort

    from vision_os.adapters.detection.letterbox import compute_transform

    if session is None:
        session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    _, _, target_h, target_w = session.get_inputs()[0].shape

    width, height = image.size
    transform = compute_transform(
        source_width=width, source_height=height,
        target_width=target_w, target_height=target_h,
    )
    scaled = image.resize(
        (max(1, round(width * transform.scale)), max(1, round(height * transform.scale)))
    )
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    ox, oy = int(transform.pad_x), int(transform.pad_y)
    canvas[oy:oy + scaled.height, ox:ox + scaled.width] = np.array(scaled)

    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    raw = session.run(None, {name: tensor})[0][0]          # (56, 8400)

    out = []
    for column in raw.T:
        score = float(column[4])
        if score < POSE_PERSON_CONFIDENCE:
            continue
        cx, cy, bw, bh = (float(v) for v in column[:4])
        box = (
            (cx - bw / 2 - transform.pad_x) / transform.scale / width,
            (cy - bh / 2 - transform.pad_y) / transform.scale / height,
            (cx + bw / 2 - transform.pad_x) / transform.scale / width,
            (cy + bh / 2 - transform.pad_y) / transform.scale / height,
        )
        kp = column[5:].reshape(17, 3).astype(np.float64).copy()
        kp[:, 0] = (kp[:, 0] - transform.pad_x) / transform.scale / width
        kp[:, 1] = (kp[:, 1] - transform.pad_y) / transform.scale / height
        out.append((box, score, kp))
    return out, session


def nms(candidates, iou_threshold: float = 0.6):
    """Greedy suppression over person boxes. The raw head has 8400 columns."""

    def overlap(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    kept = []
    for box, score, kp in sorted(candidates, key=lambda c: -c[1]):
        if all(overlap(box, k[0]) < iou_threshold for k in kept):
            kept.append((box, score, kp))
    return kept


def iou(a, b) -> float:
    x1, y1 = max(a.x1, b[0]), max(a.y1, b[1])
    x2, y2 = min(a.x2, b[2]), min(a.y2, b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> int:
    from PIL import Image

    from .schema import load_annotations

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n-pose.onnx"))
    parser.add_argument("--tag", default="pose_observability")
    args = parser.parse_args()

    # Loaded by file path rather than by putting the dataset directory on
    # `sys.path`. Two reasons, and the second is why it changed in Phase 1:
    #
    #   1. A dataset directory on `sys.path` shadows every module that happens
    #      to share a name with a file in it, for the rest of the process.
    #   2. This repository forbids `sys.path` mutation outright, because that is
    #      how the validation harness reached the platform across repositories.
    #      A rule with an exception in it is not a rule a test can enforce.
    #
    # `HEAD_BANDS` is human ground truth and never model output. That has not
    # changed; only how the file is read.
    HEAD_BANDS = _load_dataset_module(args.dataset, "head_locations").HEAD_BANDS

    frames = load_annotations(args.dataset / "annotations" / f"{args.dataset.name}.json")
    session = None
    rows = []
    latencies = []

    for frame in frames:
        image = Image.open(args.dataset / frame.image_path).convert("RGB")
        started = time.perf_counter()
        raw, session = run_pose(image, args.weights, session)
        latencies.append((time.perf_counter() - started) * 1000)
        people = nms(raw)

        for index, subject in enumerate(frame.subjects):
            band = HEAD_BANDS[(frame.frame_index, index)]
            best, best_iou = None, 0.0
            for candidate in people:
                score = iou(subject.box, candidate[0])
                if score > best_iou:
                    best, best_iou = candidate, score

            if best is None or best_iou < ASSOCIATION_IOU:
                estimate = HeadEstimate(HeadObservability.NOT_LOCATED, None, 0.0, 0, best_iou)
            else:
                estimate = head_estimate(best[2])
                estimate = HeadEstimate(
                    estimate.observability, estimate.point, estimate.confidence,
                    estimate.keypoints_seen, best_iou,
                )

            # Is the estimated head inside the person box, and inside the
            # existing top-45% evidence region? (§17)
            in_box = in_region = False
            if estimate.point is not None:
                b = subject.box
                in_box = b.x1 <= estimate.point[0] <= b.x2 and b.y1 <= estimate.point[1] <= b.y2
                region_bottom = b.y1 + (b.y2 - b.y1) * 0.45
                in_region = b.x1 <= estimate.point[0] <= b.x2 and b.y1 <= estimate.point[1] <= region_bottom

            rows.append({
                "frame_id": frame.frame_id,
                "subject_id": subject.subject_id,
                "human_head_located": band is not None,
                "human_band": band,
                "pose_state": estimate.observability.value,
                "pose_located": estimate.observability is HeadObservability.LOCATED,
                "confidence": round(estimate.confidence, 4),
                "keypoints_seen": estimate.keypoints_seen,
                "association_iou": round(estimate.association_iou, 3),
                "head_in_person_box": in_box,
                "head_in_evidence_region": in_region,
                "note": subject.note,
            })

    report(rows, latencies, args)
    out = args.dataset / "results" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "weights": str(args.weights),
            "keypoint_confidence": DEFAULT_KEYPOINT_CONFIDENCE,
            "association_iou": ASSOCIATION_IOU,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "rows": rows,
        }, indent=1),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


def report(rows, latencies, args) -> None:
    total = len(rows)
    human_yes = [r for r in rows if r["human_head_located"]]
    human_no = [r for r in rows if not r["human_head_located"]]

    tp = sum(1 for r in human_yes if r["pose_located"])
    fn = len(human_yes) - tp
    fp = sum(1 for r in human_no if r["pose_located"])
    tn = len(human_no) - fp

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and precision + recall else None)

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    print(f"\nmodel {args.weights.name}   subjects {total}")
    print(f"mean pose latency {sum(latencies) / len(latencies):.0f} ms/frame (CPU)")
    print("\n--- head observability (scored ONLY against human head locations) ---")
    print(f"  human head located     {len(human_yes)}")
    print(f"  human head NOT located {len(human_no)}")
    print(f"  pose head located      {tp + fp}")
    print(f"  pose head NOT located  {total - tp - fp}")
    print(f"\n  TP {tp}   FP {fp}   TN {tn}   FN {fn}")
    print(f"  precision {pct(precision)}   recall {pct(recall)}   F1 {pct(f1)}")

    print("\n--- THE SAFETY NUMBER ---")
    unsafe = fp / len(human_no) if human_no else None
    print(f"  unsafe acceptance: pose claimed a head where a human saw none")
    print(f"    {fp}/{len(human_no)} = {pct(unsafe)}   <- each one can still become a false violation")
    refusal = fn / len(human_yes) if human_yes else None
    print(f"  false refusal: pose denied a head a human could see")
    print(f"    {fn}/{len(human_yes)} = {pct(refusal)}   <- each one loses a valid observation")

    states = {}
    for r in rows:
        states[r["pose_state"]] = states.get(r["pose_state"], 0) + 1
    print(f"\n  pose states: {states}")

    print("\n--- evidence region (§17) ---")
    located = [r for r in rows if r["pose_located"]]
    print(f"  pose head inside person box      {sum(1 for r in located if r['head_in_person_box'])}/{len(located)}")
    print(f"  pose head inside top-45% region  {sum(1 for r in located if r['head_in_evidence_region'])}/{len(located)}")

    print("\n--- unsafe acceptances in detail ---")
    for r in human_no:
        if r["pose_located"]:
            print(f"    {r['frame_id'][-6:]} {r['subject_id']}  conf={r['confidence']:.2f} "
                  f"kpts={r['keypoints_seen']}  in_region={r['head_in_evidence_region']}  {r['note'][:44]}")


if __name__ == "__main__":
    raise SystemExit(main())
