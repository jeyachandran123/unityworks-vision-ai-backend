"""Compare two detector checkpoints on the same frames, under one configuration.

    python -m tools.vision_eval.detector_bench datasets/kitchen-01 \
        --weights models/yolov8n.onnx models/yolov8s.onnx

Everything except the weights file is held fixed: same frames, same adapter, same
preprocessing, same confidence floor, same NMS, same person-class filter. The
only variable is the checkpoint, so any difference is attributable to it.

**What this measures and what it does not.** The 43 annotated subjects are
detector proposals a human confirmed to be real people, so a person neither
detector ever proposed was never annotated. This file therefore reports *head
containment of an evaluated person box* and **never** overall person recall —
the dataset cannot support that claim (§12).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .schema import BoundingBox, load_annotations

#: Overlap at which a detection is taken to be the same subject as an annotation.
MATCH_IOU = 0.5


@dataclass(slots=True)
class DetectorRun:
    weights: str
    frames: int = 0
    detections: int = 0
    matched: int = 0
    unmatched_annotations: int = 0
    spurious: int = 0
    latency_ms: list[float] = field(default_factory=list)
    boxes: dict[str, list] = field(default_factory=dict)
    """frame_id -> [(subject_id|None, box, score)] so geometry can be compared."""

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_latency_ms if self.mean_latency_ms else 0.0


def run_detector(weights: Path, root: Path, annotated) -> DetectorRun:
    """One checkpoint over every annotated frame, through the real adapter."""
    import os

    import numpy as np
    from PIL import Image

    os.environ["VISION_DETECTOR_WEIGHTS"] = str(weights)

    from vision_os.adapters.configuration.detector_providers import build_detector
    from vision_os.core.model.frame import FrameDimensions
    from vision_os.core.model.ids import CameraId, FrameRef, FrameSeq, StreamEpoch
    from vision_os.core.ports.detection import DetectionRequest, FrameView
    from vision_os.kernel.clock import VirtualClock

    bound = build_detector(clock=VirtualClock())
    run = DetectorRun(weights=str(weights))

    for frame in annotated:
        image = Image.open(root / frame.image_path).convert("RGB")
        width, height = image.size
        bgr = np.array(image)[:, :, ::-1].copy()
        view = FrameView(
            frame_ref=FrameRef(CameraId("bench"), StreamEpoch(0), FrameSeq(frame.frame_index)),
            dimensions=FrameDimensions(width=width, height=height, colour_space="bgr24"),
            pixels=memoryview(bgr.tobytes()).toreadonly(),
        )
        started = time.perf_counter()
        result = bound.detector.detect([view], DetectionRequest(min_confidence=0.35))[0]
        run.latency_ms.append((time.perf_counter() - started) * 1000)

        people = [d for d in result.detections if str(d.class_id) == "person"]
        run.frames += 1
        run.detections += len(people)

        # Match each annotated subject to its best-overlapping detection.
        available = list(people)
        rows = []
        for subject in frame.subjects:
            best, best_iou = None, 0.0
            for candidate in available:
                box = BoundingBox(
                    candidate.box.x1, candidate.box.y1, candidate.box.x2, candidate.box.y2
                )
                score = subject.box.iou(box)
                if score > best_iou:
                    best, best_iou = candidate, score
            if best is not None and best_iou >= MATCH_IOU:
                available.remove(best)
                run.matched += 1
                rows.append((subject.subject_id, best, best_iou))
            else:
                run.unmatched_annotations += 1
                rows.append((subject.subject_id, None, best_iou))
        run.spurious += len(available)
        run.boxes[frame.frame_id] = rows

    return run


def geometry(run: DetectorRun) -> dict:
    """Box statistics over matched detections only."""
    import statistics as st

    widths, heights, scores, areas, ratios = [], [], [], [], []
    for rows in run.boxes.values():
        for _, det, _ in rows:
            if det is None:
                continue
            w = det.box.x2 - det.box.x1
            h = det.box.y2 - det.box.y1
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            ratios.append(h / w if w else 0.0)
            scores.append(float(det.score))
    if not widths:
        return {}
    return {
        "median_width": round(st.median(widths), 4),
        "median_height": round(st.median(heights), 4),
        "median_area": round(st.median(areas), 5),
        "median_aspect_h_over_w": round(st.median(ratios), 2),
        "median_confidence": round(st.median(scores), 3),
        "min_confidence": round(min(scores), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--tag", default="detector_bench")
    args = parser.parse_args()

    root: Path = args.dataset
    annotated = load_annotations(root / "annotations" / f"{root.name}.json")

    results = {}
    for weights in args.weights:
        if not weights.exists():
            print(f"MISSING checkpoint: {weights}")
            return 2
        run = run_detector(weights, root, annotated)
        results[weights.name] = run
        print(f"\n=== {weights.name} ===")
        print(f"  frames {run.frames}  person detections {run.detections}")
        print(f"  matched to annotation {run.matched}/{run.matched + run.unmatched_annotations}"
              f"  unmatched {run.unmatched_annotations}  spurious {run.spurious}")
        print(f"  latency mean {run.mean_latency_ms:.0f} ms  ({run.fps:.1f} fps, CPU)")
        for key, value in geometry(run).items():
            print(f"    {key:<26} {value}")

    out = root / "results" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                name: {
                    "weights": run.weights,
                    "frames": run.frames,
                    "person_detections": run.detections,
                    "matched": run.matched,
                    "unmatched_annotations": run.unmatched_annotations,
                    "spurious": run.spurious,
                    "mean_latency_ms": round(run.mean_latency_ms, 1),
                    "fps_cpu": round(run.fps, 2),
                    "geometry": geometry(run),
                    "matched_boxes": {
                        fid: [
                            {
                                "subject_id": sid,
                                "box": None if det is None else [
                                    round(det.box.x1, 5), round(det.box.y1, 5),
                                    round(det.box.x2, 5), round(det.box.y2, 5),
                                ],
                                "score": None if det is None else round(float(det.score), 4),
                                "iou_to_annotation": round(iou, 4),
                            }
                            for sid, det, iou in rows
                        ]
                        for fid, rows in run.boxes.items()
                    },
                }
                for name, run in results.items()
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
