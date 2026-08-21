"""The space model (02_VISION_OBJECT_MODEL §6).

Normalized image coordinates are mandatory and universal: every spatial value in
the platform carries them. They are resolution-independent, so changing a
camera's resolution or a pipeline's inference scale does not invalidate
historical data.

Pixel coordinates never escape an adapter. A detector adapter receives whatever
tensor layout it wants and returns normalized coordinates — which is what makes
detectors interchangeable when they disagree about letterboxing, stride, and
origin conventions, a notorious source of silent box misalignment.

Geometry here is pure Python. ``core`` may not import numpy or OpenCV.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from .ids import CalibrationId


class FrameOfReference(enum.Enum):
    """The coordinate stack (02_VOM §6.1)."""

    NORMALIZED = "normalized"
    """[0,1] on the rectified image. Always available; the universal frame."""

    CAMERA_GROUND = "camera_ground"
    """Metres on the camera-local ground plane. Requires a homography."""

    SITE = "site"
    """Metres in a site frame. Requires a site transform."""

    GEO = "geo"
    """WGS84. Smart city and aerial deployments."""


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point. Units depend on the declaring ``FrameOfReference``."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box, ``x1 < x2`` and ``y1 < y2``, origin top-left.

    In ``NORMALIZED`` space all four values lie in [0,1]. The convention is fixed
    platform-wide precisely because adapters disagree about it (06_PORTS D1).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: ({self.x1},{self.y1})-({self.x2},{self.y2})")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_centre(self) -> Point:
        """The usual proxy for an object's ground contact point."""
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def clamped_to_unit(self) -> Box:
        """Clamp into [0,1]. Adapters returning out-of-range boxes are counted."""
        return Box(
            max(0.0, min(1.0, self.x1)),
            max(0.0, min(1.0, self.y1)),
            max(0.0, min(1.0, self.x2)),
            max(0.0, min(1.0, self.y2)),
        )

    def is_within_unit(self) -> bool:
        return 0.0 <= self.x1 and 0.0 <= self.y1 and self.x2 <= 1.0 and self.y2 <= 1.0


@dataclass(frozen=True, slots=True)
class Polygon:
    """A closed polygon. Vertices in the declaring frame of reference."""

    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError(f"polygon needs >= 3 vertices, got {len(self.vertices)}")

    def contains(self, point: Point) -> bool:
        """Ray-casting point-in-polygon test.

        Pure geometry, no domain meaning. The platform computes membership; what
        the region *means* lives with the consumer (invariant V1).
        """
        inside = False
        count = len(self.vertices)
        j = count - 1
        for i in range(count):
            vi, vj = self.vertices[i], self.vertices[j]
            if (vi.y > point.y) != (vj.y > point.y):
                x_at_y = (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x
                if point.x < x_at_y:
                    inside = not inside
            j = i
        return inside

    @property
    def bounds(self) -> Box:
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return Box(min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True, slots=True)
class Ellipse:
    """An uncertainty ellipse, in the units of its frame of reference."""

    semi_major: float
    semi_minor: float
    rotation_degrees: float


@dataclass(frozen=True, slots=True)
class Homography:
    """A 3x3 projective transform from normalized image space to a ground plane.

    Stored row-major. Pure Python arithmetic keeps ``core`` dependency-free; the
    matrix is small and applied a handful of times per object per frame.
    """

    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def project(self, point: Point) -> Point:
        """Apply the transform. Raises on a degenerate (at-horizon) projection."""
        m = self.matrix
        w = m[2][0] * point.x + m[2][1] * point.y + m[2][2]
        if abs(w) < 1e-12:
            raise ValueError("degenerate projection: point lies on the horizon line")
        x = (m[0][0] * point.x + m[0][1] * point.y + m[0][2]) / w
        y = (m[1][0] * point.x + m[1][1] * point.y + m[1][2]) / w
        return Point(x, y)

    def is_degenerate(self) -> bool:
        """A homography with near-zero determinant is not invertible."""
        m = self.matrix
        det = (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
        return abs(det) < 1e-10


@dataclass(frozen=True, slots=True)
class Calibration:
    """A versioned calibration (02_VOM §10.1).

    Observations pin ``calibration_id``, so when a camera is re-calibrated old
    observations remain interpretable under the calibration in force when they
    were made. Re-projecting history is an explicit offline operation producing
    *new* records with lineage, never an in-place rewrite (invariant V5).
    """

    calibration_id: CalibrationId
    homography: Homography | None = None
    ground_uncertainty_at_unit_distance: float = 0.0
    site_transform: Homography | None = None
    suspect: bool = False
    """Set when viewpoint drift is suspected. Projection continues with inflated
    uncertainty rather than being invalidated — a false positive must not blind a
    working site (03_MODULES M1)."""

    @property
    def can_project_to_ground(self) -> bool:
        return self.homography is not None and not self.homography.is_degenerate()

    def project_to_ground(self, point: Point) -> tuple[Point, Ellipse]:
        """Project a normalized image point onto the ground plane.

        Returns the point **and its uncertainty ellipse**, always. Homography
        error grows sharply with distance and with deviation from the ground
        plane; a projected position without its error invites consumers to
        compute distances and speeds that are confidently wrong at the far end of
        the view (02_VOM §6.2).
        """
        if self.homography is None:
            raise ValueError(f"calibration {self.calibration_id} has no homography")
        ground = self.homography.project(point)
        # Uncertainty grows with distance from the camera's ground origin.
        distance = math.hypot(ground.x, ground.y)
        scale = self.ground_uncertainty_at_unit_distance * max(1.0, distance)
        if self.suspect:
            scale *= 3.0
        return ground, Ellipse(semi_major=scale, semi_minor=scale * 0.6, rotation_degrees=0.0)


@dataclass(frozen=True, slots=True)
class SpatialInfo:
    """Spatial payload carried by anything positioned in the world (02_VOM §6.2).

    ``frame_of_reference`` and ``calibration_id`` are declared together: a metre
    measurement without the calibration that produced it is unverifiable and
    unreproducible once the camera is re-calibrated or physically nudged.
    """

    frame_of_reference: FrameOfReference
    bbox: Box | None = None
    calibration_id: CalibrationId | None = None
    ground_point: Point | None = None
    ground_uncertainty: Ellipse | None = None

    def __post_init__(self) -> None:
        if self.frame_of_reference is not FrameOfReference.NORMALIZED:
            if self.calibration_id is None:
                raise ValueError(
                    f"{self.frame_of_reference.value} coordinates require a calibration_id"
                )
        if self.ground_point is not None and self.ground_uncertainty is None:
            raise ValueError("ground_point requires ground_uncertainty (02_VOM §6.2)")
