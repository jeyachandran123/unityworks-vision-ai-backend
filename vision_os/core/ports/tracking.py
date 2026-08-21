"""P9 ``TrackerPort``, P10 ``EmbeddingPort``, and the motion/association seams.

A port is **Interface + Semantic Contract + Conformance Kit** (06_PORTS section
5). The interface below is the shape; the numbered obligations are the meaning;
``conformance/tracker_kit.py`` is the executable proof. An adapter satisfying the
interface but not the obligations will type-check, run, and quietly corrupt every
track it touches — which is why the kit gates activation.

**The platform must never learn which tracker is bound.** No module outside
``adapters/`` and the composition root may name ByteTrack, SORT, or any other
implementation, exactly as Flow 2 hid YOLO.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.detection import Detection
from ..model.ids import CameraId, FrameRef, TrackerEpoch
from ..model.timebase import Duration, Instant
from ..model.track import Track, TrackUpdate

TRACKER_PORT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class TrackerCapabilities:
    """What an adapter claims it can do — 06_PORTS section P9.

    Declared **honestly** (adapter obligation A1). A capability gap that is
    declared is reportable; one that is discovered at runtime is an outage.
    """

    tracker_id: str
    version: str

    requires_embeddings: bool = False
    """When true, the tracker cannot run without an ``EmbeddingPort``. It fails
    to activate rather than degrading silently to geometry, because a silent
    degradation makes the gap invisible (invariant V8)."""

    handles_occlusion: str = "none"
    """``none`` | ``short`` | ``long``. Advisory: it tells the platform how long
    a coast is worth attempting before terminating."""

    max_objects: int = 256
    supports_ground_plane_tracking: bool = False
    deterministic: bool = True
    """V13 replay must know what to expect. A non-deterministic tracker may
    still be used, but deterministic mode will refuse to bind it."""

    state_per_camera_bytes: int = 0
    """Estimate, used for capacity planning at 100+ cameras."""

    def __post_init__(self) -> None:
        if not self.tracker_id:
            raise ValueError("a tracker must declare a tracker_id")
        if self.handles_occlusion not in ("none", "short", "long"):
            raise ValueError(
                f"handles_occlusion must be none|short|long, got {self.handles_occlusion!r}"
            )
        if self.max_objects < 1:
            raise ValueError("max_objects must be >= 1")


@dataclass(frozen=True, slots=True)
class TrackingRequest:
    """One frame's worth of work for a tracker.

    Carries ``elapsed`` explicitly rather than letting the adapter infer it.
    Port obligation T2: the platform drops frames by design, so an adapter that
    integrates over frame *count* produces velocities whose meaning changes with
    system load. Handing it the real elapsed time removes the temptation.
    """

    camera_id: CameraId
    frame_ref: FrameRef
    timestamp: Instant
    elapsed: Duration
    """Wall time since the previous processed frame for this camera. Zero on the
    first frame of an epoch."""

    detections: Sequence[Detection] = ()
    embeddings: Sequence[Sequence[float]] | None = None
    """Optional appearance vectors, index-aligned with ``detections``. ``None``
    unless an embedding provider is configured — and none ships, because
    appearance embeddings are C2 biometric data disabled by default
    (12_SECURITY section 4)."""

    def __post_init__(self) -> None:
        if self.embeddings is not None and len(self.embeddings) != len(self.detections):
            raise ValueError(
                f"embeddings length {len(self.embeddings)} does not match detections "
                f"length {len(self.detections)}; they must be index-aligned"
            )


@runtime_checkable
class TrackerPort(Protocol):
    """P9 — maintain temporal continuity within one camera.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **T1** | **Strictly sequential per camera.** Frames arrive in order; the adapter may assume it and must reject violations rather than degrade silently. |
    | **T2** | **Non-uniform time gaps are normal**, not exceptional. Motion models integrate over ``elapsed``, never over frame count. |
    | **T3** | Track ids are unique within ``(camera_id, tracker_epoch)`` and are **never reused** within an epoch. |
    | **T4** | Association confidence carries ``ASSOCIATION`` semantics and is honest — a low-confidence association is reported as such. |
    | **T5** | Coasting is **explicitly marked**; a predicted position is never presented as measured. |
    | **T6** | Termination carries a ``break_reason``. |
    | **T7** | State is per-camera and fully reset by ``reset()``. No cross-camera state exists in this port. |
    | **T8** | Memory is bounded regardless of scene duration or object count. |

    T2 deserves emphasis: it is *"the single most common way an off-the-shelf
    tracker misbehaves inside UWV"* (06_PORTS P9). A tracker validated only on
    continuous video is not validated for this platform.

    T7 is what keeps a track id from becoming an identity. Cross-camera matching
    is P11 ``IdentityResolverPort``, which is Phase 2 and unimplemented.
    """

    def update(self, request: TrackingRequest) -> TrackUpdate:
        """Advance tracking by one frame.

        An empty ``detections`` sequence is **normal**, not an error: it is
        exactly when tracks coast, age, and terminate. Returning an empty
        ``TrackUpdate`` for a frame with detections, however, is a contract
        violation.

        Raises:
            OutOfOrderFrameError: the frame precedes one already processed (T1).
            TrackerContractError: the adapter detected its own inconsistency.
        """
        ...

    def tracks(self, camera_id: CameraId) -> Sequence[Track]:
        """Current live tracks for a camera. Never includes terminated ones."""
        ...

    def reset(self, camera_id: CameraId, reason: str) -> TrackerEpoch:
        """Discard all state for a camera and mint a new epoch.

        The new epoch is what makes the discontinuity visible: without it, a
        recycled local id would let a consumer infer that an object teleported
        rather than that tracking restarted (03_MODULES M6 failure handling).
        """
        ...

    def capabilities(self) -> TrackerCapabilities:
        """Stable for the adapter's lifetime. Callers may cache it."""
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """P10 — appearance vectors for association and (later) re-identification.

    **Declared, unbound, and unimplemented in this flow, deliberately.**

    Appearance embeddings are classified **C2 · Biometric** (12_SECURITY section
    4.3): disabled by default, session-scoped when enabled, policy-gated,
    restricted access, separate retention. Threat #4 in the same document is
    *identity linkage* — "any persistent mapping that links sightings across time
    or cameras" — which is precisely what a retained embedding gallery is.

    The port exists so that ``requires_embeddings`` is a meaningful capability
    and a DeepSORT-class adapter is possible later under policy. No provider
    ships, nothing binds it by default, and the tracking layer stores no
    embedding it is given.
    """

    @property
    def embedding_id(self) -> str:
        """Model identity. Two embeddings from different models never compare."""
        ...

    @property
    def dimensions(self) -> int:
        ...

    def embed(self, frame_ref: FrameRef, boxes: Sequence[object]) -> Sequence[Sequence[float]]:
        """Produce one vector per box, index-aligned and L2-normalized."""
        ...


@dataclass(frozen=True, slots=True)
class Prediction:
    """Where a track is expected to be, and how sure the model is."""

    x1: float
    y1: float
    x2: float
    y2: float
    uncertainty: float = 0.0
    """Positional standard deviation in normalized units. **Grows with elapsed
    time**, which is what lets the association gate widen honestly during a long
    coast instead of pretending a five-second-old prediction is as good as a
    fresh one."""

    def __post_init__(self) -> None:
        if self.uncertainty < 0.0:
            raise ValueError("uncertainty must be non-negative")


@dataclass(slots=True)
class MotionObservation:
    """One measured position fed to a motion model."""

    x1: float
    y1: float
    x2: float
    y2: float
    elapsed: Duration


@runtime_checkable
class MotionPredictorPort(Protocol):
    """The motion-model seam — constant velocity, Kalman, or learned.

    **The platform does not depend on a Kalman filter.** Kalman is one adapter
    behind this port, not a built-in assumption. That matters because a Kalman
    filter carries tuning (process noise, measurement noise) that is correct for
    one camera geometry and wrong for another, and baking it into the platform
    would make those constants unchangeable.

    Implementations are **per-track**: each track owns its own predictor
    instance, so there is no shared state and no cross-track contamination.
    """

    @property
    def model_id(self) -> str:
        ...

    def observe(self, observation: MotionObservation) -> None:
        """Fold in a measured position."""
        ...

    def predict(self, elapsed: Duration) -> Prediction:
        """Where the object should be after ``elapsed`` more time.

        Called with a real elapsed duration, never a frame count (T2).
        """
        ...

    def velocity(self) -> tuple[float, float]:
        """Current estimate, in normalized units **per second**."""
        ...

    def acceleration(self) -> tuple[float, float] | None:
        """``None`` until enough observations exist — not zero. "Not yet
        measurable" and "measured as zero" are different claims."""
        ...


@dataclass(frozen=True, slots=True)
class AssociationCandidate:
    """One (track, detection) pair the gate considered plausible."""

    track_index: int
    detection_index: int
    cost: float
    """Lower is better. Normalized to [0,1] so costs from different signals —
    geometry, motion, appearance — combine meaningfully."""


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """The resolved matching for one frame."""

    matches: tuple[tuple[int, int], ...] = ()
    """``(track_index, detection_index)`` pairs."""

    unmatched_tracks: tuple[int, ...] = ()
    unmatched_detections: tuple[int, ...] = ()
    costs: Mapping[tuple[int, int], float] = field(default_factory=dict)
    """``(track_index, detection_index) -> cost`` for matched pairs."""

    runner_up: Mapping[int, float] = field(default_factory=dict)
    """``track_index -> second-best cost``. The margin between the winner and
    this is the honest measure of association ambiguity, and a narrow margin is
    exactly the ID-switch risk the tracker is required not to hide."""


@runtime_checkable
class AssociationPort(Protocol):
    """The assignment seam — greedy, Hungarian, or learned.

    Separated from the tracker so that the *policy* for resolving a cost matrix
    is replaceable independently of the *signals* that populate it. A tracker
    changing from greedy to optimal assignment should not require a new tracker.
    """

    @property
    def method_id(self) -> str:
        ...

    def assign(
        self,
        *,
        track_count: int,
        detection_count: int,
        candidates: Sequence[AssociationCandidate],
        max_cost: float,
    ) -> AssignmentResult:
        """Resolve a cost matrix into a one-to-one matching.

        **Must be deterministic**: identical input yields an identical result,
        including tie-breaking order. Non-determinism here silently changes
        which object keeps which id when two candidates tie, producing ID
        switches that no test can reproduce (invariant V13).
        """
        ...
