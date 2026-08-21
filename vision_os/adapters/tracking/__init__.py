"""Tracking adapters — concrete trackers and motion models behind P9.

**Nothing outside this package and the composition root may name a tracker.**
The platform holds ``TrackerPort``; which implementation satisfies it is a
configuration fact, exactly as Flow 2 keeps YOLO invisible to the platform.
"""

from .geometric import GeometricConfig, GeometricTracker
from .motion import LinearPredictor, StationaryPredictor, heading_of, speed_of
from .trackers import (
    FALLBACK_TRACKER_ID,
    TRACKER_FACTORIES,
    build_bytetrack_tracker,
    build_iou_tracker,
    build_sort_tracker,
)

__all__ = [
    "FALLBACK_TRACKER_ID",
    "TRACKER_FACTORIES",
    "GeometricConfig",
    "GeometricTracker",
    "LinearPredictor",
    "StationaryPredictor",
    "build_bytetrack_tracker",
    "build_iou_tracker",
    "build_sort_tracker",
    "heading_of",
    "speed_of",
]
