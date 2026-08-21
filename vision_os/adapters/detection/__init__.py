"""P8 detector adapters.

The only place in the codebase where a detector's framework, label space, or
coordinate convention appears. Adding RT-DETR, Grounding-DINO, or a segmentation
model means adding a sibling here — no platform module changes (invariant V3).
"""

from __future__ import annotations

from .letterbox import compute_transform, invert_to_normalized, truncation_of
from .reference import EmptyDetector, ReferenceDetector, ScriptedDetection
from .yolo import DEFAULT_INPUT_SIZE, YoloDetector

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "EmptyDetector",
    "ReferenceDetector",
    "ScriptedDetection",
    "YoloDetector",
    "compute_transform",
    "invert_to_normalized",
    "truncation_of",
]
