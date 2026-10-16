"""The production detector is yolo11s, shipped with the checkout.

Measured on 30 PCC2 decision frames (2026-10-15) at the kitchen policy's 0.4
confidence floor, through the platform's own session and letterbox: yolov8n
found 82 people, 78 of them also found by Ultralytics' own pipeline; yolo11s
found 93, 89 of them. On the crowded camera-13 frame of the reported alert,
yolov8n boxed the five people on the left at 0.42-0.46 with overlapping boxes,
yolo11s at 0.52-0.94, one box each.

yolo11s keeps v8's output layout, `(1, 84, 8400)` with no objectness channel,
so the platform's decoder reads it unchanged. YOLO26 does not (its graph is
NMS-free, `(1, 300, 6)`), which is why it is not the choice here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from vision_os.adapters.configuration.detector_providers import (
    build_detector,
    default_weights_path,
)
from vision_os.kernel.clock import VirtualClock

REPO = Path(__file__).resolve().parents[2]


def test_the_default_weights_are_yolo11s() -> None:
    path = default_weights_path()
    assert path.name == "yolo11s.onnx"
    assert path.is_file()


def test_the_default_detector_binds_with_its_own_class_names() -> None:
    bound = build_detector(clock=VirtualClock(), env={})
    assert bound.model_id == "yolo11s"
    assert "person" in {str(c) for c in bound.classes}


def test_the_weights_are_not_ignored_by_git() -> None:
    """A checkout must reproduce the deployment; ignored weights would not."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "models/yolo11s.onnx"],
        cwd=REPO,
        check=False,
    )
    assert result.returncode == 1, "models/yolo11s.onnx is git-ignored"
