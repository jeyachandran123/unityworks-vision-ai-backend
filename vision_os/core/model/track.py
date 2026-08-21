"""The Track object (02_VISION_OBJECT_MODEL section 10.5).

An assertion that a sequence of detections is **one continuous thing, within one
camera**. That is the whole claim. A track does not know what the thing is beyond
its visual class, does not know whether it has been seen before on another
camera, and does not survive a tracker restart.

Three properties are load-bearing and each exists to prevent a specific, common
corruption:

``TrackId`` is composite
    ``(camera_id, tracker_epoch, local_id)``. A bare integer would compare equal
    across cameras and across resets — the mechanism by which a fragile handle
    quietly becomes an identity.

``detections`` holds ``FrameRef``, never ``Detection``
    References, not copies (02_VOM section 10.5). Copying detections into tracks
    makes tracking memory grow with track lifetime, which breaks port obligation
    T8 on exactly the long-lived tracks that matter most.

``measurement_basis`` is per-position, not per-track
    A coasting track's position is a *prediction*. Presenting it as a measurement
    is invariant V8 violated at object scale, and it is invisible downstream
    unless the field travels with the value.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from .confidence import Confidence, ConfidenceSemantics
from .ids import CameraId, ClassId, FrameRef, SiteId, TenantId, TrackId
from .provenance import Provenance
from .space import Point, SpatialInfo
from .timebase import Duration, Instant

TRACK_SCHEMA_VERSION = "1.0.0"


class TrackState(enum.Enum):
    """The five states of 03_MODULES M6 R2. Closed set.

    ``NEW`` and ``RECOVERED`` are deliberately **not** states here. Creation and
    recovery are transitions — they are observable as ``TrackCreated`` and
    ``TrackRecovered`` events — and modelling them as states would add two
    members that no architecture document defines and that no consumer could
    interpret consistently.
    """

    TENTATIVE = "tentative"
    """Seen, but not yet enough times to be believed. Never emitted as
    confirmed, so a one-frame detector false positive does not become a track."""

    CONFIRMED = "confirmed"
    """Associated across enough frames to be asserted."""

    COASTING = "coasting"
    """Alive but unmeasured this frame; position is predicted.

    First-class, not an implementation detail (02_VOM section 10.5): everything
    derived from a coasting track must be marked predicted."""

    LOST = "lost"
    """Coasted past the point of confidence, retained only for the recovery
    window. A lost track may still be recovered; a terminated one may not."""

    TERMINATED = "terminated"
    """Final. Never re-enters any other state."""

    @property
    def is_alive(self) -> bool:
        return self is not TrackState.TERMINATED

    @property
    def is_predicted(self) -> bool:
        """Whether a position in this state is inferred rather than measured."""
        return self in (TrackState.COASTING, TrackState.LOST)


class MotionState(enum.Enum):
    """Descriptive only. ``stationary`` is a fact about pixels, not about intent.

    Note what is absent: no ``loitering``, no ``queueing``, no ``abandoned``.
    Those are judgments, and they are rejected by the Semantic Ceiling (V1). A
    consumer that wants them derives them from ``stationary`` plus its own
    knowledge of what the region means.
    """

    STATIONARY = "stationary"
    MOVING = "moving"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"
    """Not yet determinable — too few observations, or motion below the noise
    floor. Distinct from ``stationary``, which is a positive claim."""


class BreakReason(enum.Enum):
    """Why a track stopped being measured (02_VOM section 10.5).

    The diagnostic that makes tracker regressions findable: *"we lost 40% more
    tracks this week, all with detector_miss"* points at the detector, not the
    tracker. Without it, both look identical from the outside.
    """

    NONE = "none"
    OCCLUSION = "occlusion"
    EXIT = "exit"
    """Left the frame — the only healthy way for a track to end."""

    DETECTOR_MISS = "detector_miss"
    ASSOCIATION_FAILURE = "association_failure"
    EPOCH_RESET = "epoch_reset"
    """The tracker was reset beneath it. Not a tracking failure; a discontinuity
    the consumer must see rather than infer teleportation from."""


class MeasurementBasis(enum.Enum):
    """Whether a value was observed or inferred (02_VOM section 11, V8)."""

    MEASURED = "measured"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"


class AssociationMethod(enum.Enum):
    """How a detection was tied to a track. Reported, never assumed."""

    IOU = "iou"
    MOTION_GATED_IOU = "motion_gated_iou"
    CENTROID_DISTANCE = "centroid_distance"
    APPEARANCE = "appearance"
    """Requires an embedding provider. Ships unbound: appearance embeddings are
    C2 biometric data, disabled by default (12_SECURITY section 4)."""

    REINITIALIZED = "reinitialized"
    """No association; a new track was started instead."""


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    """Velocity and acceleration in the declared frame of reference.

    Units are *per second*, never per frame. The platform drops frames by design
    (V7), so a per-frame velocity is a number whose meaning changes with load —
    the single most common way an off-the-shelf tracker misbehaves inside UWV
    (port obligation T2).
    """

    velocity: Point = Point(0.0, 0.0)
    """Units of the frame of reference per second."""

    acceleration: Point | None = None
    """``None`` until enough observations exist. Not zero — "not yet measurable"
    and "measured as zero" are different claims."""

    heading_degrees: float | None = None
    """Clockwise from +x. ``None`` when speed is below the noise floor, where
    heading is numerically meaningless rather than merely uncertain."""

    speed: float = 0.0
    uncertainty: float = 0.0
    """Standard deviation of the position prediction, in frame-of-reference
    units. Grows while coasting: a prediction five frames old is not as good as
    a prediction one frame old, and the consumer must be able to tell."""

    def __post_init__(self) -> None:
        if self.speed < 0.0:
            raise ValueError(f"speed must be non-negative, got {self.speed}")
        if self.uncertainty < 0.0:
            raise ValueError(f"uncertainty must be non-negative, got {self.uncertainty}")
        if self.heading_degrees is not None and not 0.0 <= self.heading_degrees < 360.0:
            raise ValueError(f"heading must be in [0,360), got {self.heading_degrees}")


@dataclass(frozen=True, slots=True)
class TrackEvidence:
    """What the continuity claim rests on.

    Not the ``Evidence`` object of 02_VOM section 10.9 — that is bound to an
    ``observation_id`` and assembled by the Observation Builder in Flow 6. This
    is the raw material.
    """

    association_method: AssociationMethod
    association_cost: float = 0.0
    """The winning cost from the assignment. Retained because a track associated
    at cost 0.05 and one associated at cost 0.94 are very different claims that
    an association *confidence* alone can blur."""

    runner_up_cost: float | None = None
    """Cost of the second-best candidate. The margin between these two is the
    honest measure of how ambiguous this association was — and a narrow margin
    is exactly the ID-switch risk M6 is required not to hide."""

    gated_candidates: int = 0
    """How many candidates survived gating. Zero with a live track means the
    prediction was wrong or the object left."""

    notes: str = ""

    @property
    def margin(self) -> float | None:
        """Cost gap to the runner-up. ``None`` when there was no contest."""
        if self.runner_up_cost is None:
            return None
        return self.runner_up_cost - self.association_cost


@dataclass(frozen=True, slots=True)
class Track:
    """One camera-local continuity assertion. Immutable (invariant V5).

    Carries no ``object_id``, no name, no person reference, and no cross-camera
    field. Those absences are enforced by architecture tests, because the
    pressure to add them arrives with the first consumer who wants to count
    unique visitors.
    """

    track_id: TrackId
    camera_id: CameraId
    tenant_id: TenantId
    site_id: SiteId

    state: TrackState
    class_id: ClassId
    """Best current class. A track holds one class; class flapping is resolved
    by the Object Registry (M7) using the retained distribution, not here."""

    confidence: Confidence
    """``ASSOCIATION`` semantics — P(this detection continues this track). Not
    the detector's presence score, which measures something else entirely."""

    spatial: SpatialInfo
    measurement_basis: MeasurementBasis
    motion: MotionEstimate
    motion_state: MotionState

    first_seen: Instant
    last_seen: Instant
    """Last **measured** sighting. Deliberately not updated while coasting: a
    consumer asking "how fresh is this?" must not be told a prediction is a
    sighting."""

    last_updated: Instant
    """Last time the track was touched at all, measured or predicted."""

    age_frames: int
    hit_count: int
    """Frames in which this track was actually measured. ``age_frames`` minus
    ``hit_count`` is the fragmentation signal."""

    coast_frames: int
    """Consecutive frames predicted without a detection."""

    detections: tuple[FrameRef, ...]
    """Bounded ring of contributing frames — references, never copies."""

    evidence: TrackEvidence
    provenance: Provenance
    break_reason: BreakReason = BreakReason.NONE

    schema_version: str = TRACK_SCHEMA_VERSION
    labels: Mapping[str, str] = field(default_factory=dict)
    """Opaque operational tags. No platform logic may branch on these."""

    def __post_init__(self) -> None:
        if self.confidence.semantics is not ConfidenceSemantics.ASSOCIATION:
            raise ValueError(
                f"a Track must carry ASSOCIATION confidence, got "
                f"{self.confidence.semantics.value} (port obligation T4)"
            )
        if self.track_id.camera_id != self.camera_id:
            raise ValueError(
                f"track_id names camera {self.track_id.camera_id} but the track "
                f"claims {self.camera_id}"
            )
        if self.spatial.bbox is None:
            raise ValueError("a Track must carry a bounding box")
        if not self.spatial.bbox.is_within_unit():
            raise ValueError(f"track box {self.spatial.bbox} escapes normalized [0,1] space")
        if self.age_frames < 0 or self.hit_count < 0 or self.coast_frames < 0:
            raise ValueError("track counters must be non-negative")
        if self.hit_count > self.age_frames:
            raise ValueError(
                f"hit_count {self.hit_count} exceeds age_frames {self.age_frames}; "
                f"a track cannot be measured more often than it existed"
            )
        if self.state.is_predicted and self.measurement_basis is MeasurementBasis.MEASURED:
            raise ValueError(
                f"a {self.state.value} track claims a MEASURED position; a predicted "
                f"position presented as measured is invariant V8 violated at object scale"
            )
        if self.state is TrackState.COASTING and self.coast_frames == 0:
            raise ValueError("a coasting track must have coasted at least one frame")
        if self.last_seen.ns > self.last_updated.ns:
            raise ValueError("last_seen cannot be later than last_updated")

    # --- derived properties, all pure ------------------------------------- #

    @property
    def tracker_epoch(self) -> int:
        return self.track_id.tracker_epoch

    @property
    def is_predicted(self) -> bool:
        """Whether this position is inferred rather than observed."""
        return self.measurement_basis is not MeasurementBasis.MEASURED

    @property
    def is_alive(self) -> bool:
        return self.state.is_alive

    @property
    def hit_ratio(self) -> float:
        """Measured frames over total frames. A fragmentation signal."""
        return self.hit_count / self.age_frames if self.age_frames else 0.0

    def lifetime(self) -> Duration:
        return Duration(self.last_updated.ns - self.first_seen.ns)

    def staleness(self, now: Instant) -> Duration:
        """Time since the last **measured** sighting.

        The object-level expression of V8: a position measured 40 seconds ago is
        a different claim from a fresh one, and the track says so without being
        asked.
        """
        return Duration(max(0, now.ns - self.last_seen.ns))

    def is_a(self, ancestor: ClassId) -> bool:
        """Hierarchical class match without consulting the registry."""
        return self.class_id == ancestor or self.class_id.startswith(f"{ancestor}.")


@dataclass(frozen=True, slots=True)
class Association:
    """One resolved detection-to-track binding, reported for explainability."""

    track_id: TrackId
    detection_index: int
    """Index into the frame's detection list. ``-1`` when the track was not
    associated this frame."""

    confidence: Confidence
    method: AssociationMethod
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.confidence.semantics is not ConfidenceSemantics.ASSOCIATION:
            raise ValueError("an Association must carry ASSOCIATION confidence")


@dataclass(frozen=True, slots=True)
class RefusedAssociation:
    """A binding the tracker declined because it was too ambiguous to assert.

    First-class rather than inferred. A refused track is usually terminated in
    the same frame, so it appears in neither ``active`` nor ``associations`` — and
    reporting only what *was* associated would hide precisely the cases
    03_MODULES M6 cares most about: the near-ties where a confident guess would
    have produced an ID switch.
    """

    track_id: TrackId
    best_cost: float
    runner_up_cost: float

    @property
    def margin(self) -> float:
        return self.runner_up_cost - self.best_cost


@dataclass(frozen=True, slots=True)
class TrackUpdate:
    """What one frame did to the track set — 03_MODULES M6 public API.

    Returned rather than only mutated so the caller can react to transitions
    without diffing state it does not own.
    """

    camera_id: CameraId
    frame_ref: FrameRef
    tracker_epoch: int

    active: tuple[Track, ...] = ()
    new: tuple[TrackId, ...] = ()
    terminated: tuple[tuple[TrackId, BreakReason], ...] = ()
    coasting: tuple[TrackId, ...] = ()
    recovered: tuple[TrackId, ...] = ()
    """Tracks that returned from coasting or lost to confirmed this frame. A
    transition, not a state (see ``TrackState``)."""

    associations: tuple[Association, ...] = ()
    refused: tuple[RefusedAssociation, ...] = ()
    """Associations declined for ambiguity. The tracker's admitted uncertainty."""

    unmatched_detections: tuple[int, ...] = ()
    """Detection indices that started no track — below the confirmation bar, or
    refused because the association was too ambiguous to assert."""

    failed: bool = False
    reason: str = ""
    """Set when tracking could not run at all. Distinct from an update that
    legitimately produced no tracks (invariant V8)."""

    @property
    def active_count(self) -> int:
        return len(self.active)

    @property
    def confirmed(self) -> tuple[Track, ...]:
        return tuple(t for t in self.active if t.state is TrackState.CONFIRMED)

    @property
    def measured(self) -> tuple[Track, ...]:
        """Only tracks whose position was observed this frame."""
        return tuple(t for t in self.active if not t.is_predicted)
