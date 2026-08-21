"""Regions — named geometry with an opaque label (02_VISION_OBJECT_MODEL §10.3).

The single most important constraint in this module: **the platform never
interprets ``label``.** UWV computes membership, entry, exit, and dwell as pure
geometry and publishes them against label ``Z3``. That ``Z3`` is "the pass" in a
kitchen, "the loading dock" in a warehouse, and "the crosswalk" in a city is
knowledge the platform does not hold and must not hold.

Every temptation to add ``zone_type: "kitchen_pass"`` here is a violation of
invariant V2 wearing a helpful disguise.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .ids import CameraId, RegionId
from .space import FrameOfReference, Point, Polygon


class MembershipState(enum.Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    BOUNDARY = "boundary"


class ContainmentMethod(enum.Enum):
    """How membership was computed.

    Recorded because containment from a bounding box's bottom edge and from a
    projected ground point disagree substantially at range — and a consumer
    comparing dwell across cameras deserves to know which was used.
    """

    GROUND_POINT = "ground_point"
    BBOX_BOTTOM_CENTRE = "bbox_bottom_centre"
    MASK_OVERLAP = "mask_overlap"


@dataclass(frozen=True, slots=True)
class Region:
    """Named geometry within a camera view or on a ground plane."""

    region_id: RegionId
    geometry: Polygon
    frame_of_reference: FrameOfReference
    label: str
    """Opaque. No platform logic may branch on this value."""

    camera_id: CameraId | None = None
    version: str = "1.0.0"

    def contains(self, point: Point) -> bool:
        return self.geometry.contains(point)
