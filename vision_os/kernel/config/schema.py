"""The closed configuration schema (05_MODULES_PLATFORM_KERNEL M16).

**The schema is closed, and that is the point.** A vertical may supply exactly
four things: taxonomy mappings, region geometry with opaque labels, prompt pack
selection, and resource profiles. There is no schema slot for a threshold with
business meaning, a role definition, or a rule. Adding one requires a schema
change, which is a reviewed, visible act.

Closing the schema turns "don't put business logic in config" from a code-review
convention into a structural property (invariant V2).

Flow 1 declares the acquisition and kernel sections. Taxonomy and prompt-pack
sections arrive with Flows 2 and 5 respectively.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ...core.errors import ValidationError


class DeploymentProfile(enum.Enum):
    """13_DEPLOYMENT_ARCHITECTURE §1 — the topology family."""

    EMBEDDED = "embedded"
    EDGE = "edge"
    NODE = "node"
    CLUSTER = "cluster"


class ClockMode(enum.Enum):
    SYSTEM = "system"
    VIRTUAL = "virtual"
    SCALED = "scaled"


# --- typed slices --------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlatformSection:
    deployment_profile: DeploymentProfile = DeploymentProfile.EMBEDDED
    clock_mode: ClockMode = ClockMode.SYSTEM
    clock_scale_factor: float = 60.0
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class BufferSection:
    """Pool sizing is by *pipeline depth*, not camera count (03_MODULES M4)."""

    slots_per_camera: int = 4
    bytes_per_slot: int = 1920 * 1080 * 3
    lease_deadline_ms: int = 2_000
    history_window_ms: int = 1_500
    jitter_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.slots_per_camera < 1:
            raise ValidationError("buffer.slots_per_camera must be >= 1")
        if self.bytes_per_slot < 1:
            raise ValidationError("buffer.bytes_per_slot must be >= 1")
        if self.lease_deadline_ms < 1:
            raise ValidationError("buffer.lease_deadline_ms must be >= 1")


@dataclass(frozen=True, slots=True)
class SchedulerSection:
    global_budget_fps: float = 150.0
    """Aggregate admitted frames/second across every camera on this node."""

    sustained_drop_threshold: float = 0.5
    """Effective rate below which a sustained-drop alarm fires."""

    drop_alarm_window_ms: int = 5_000
    duplicate_suppression: bool = False

    def __post_init__(self) -> None:
        if self.global_budget_fps <= 0:
            raise ValidationError("scheduler.global_budget_fps must be positive")
        if not 0.0 <= self.sustained_drop_threshold <= 1.0:
            raise ValidationError("scheduler.sustained_drop_threshold must be in [0,1]")


@dataclass(frozen=True, slots=True)
class SourceSection:
    reconnect_backoff_initial_ms: int = 500
    reconnect_backoff_max_ms: int = 30_000
    reconnect_backoff_jitter: float = 0.2
    stall_watchdog_ms: int = 10_000
    """No frames while the socket is open — the most common real-world RTSP
    failure and the one naive implementations miss entirely."""

    max_consecutive_decode_errors: int = 30
    max_connect_attempts: int = 0
    """0 = unlimited. Bounded for persistent failures like bad credentials, so
    the platform does not hammer a camera and lock the account."""

    def __post_init__(self) -> None:
        if self.reconnect_backoff_initial_ms < 1:
            raise ValidationError("source.reconnect_backoff_initial_ms must be >= 1")
        if self.reconnect_backoff_max_ms < self.reconnect_backoff_initial_ms:
            raise ValidationError("source.reconnect_backoff_max_ms must be >= initial")
        if self.stall_watchdog_ms < 1:
            raise ValidationError("source.stall_watchdog_ms must be >= 1")


@dataclass(frozen=True, slots=True)
class HealthSection:
    report_timeout_ms: int = 15_000
    """Silence is never health. A component that stops reporting is unhealthy."""

    aggregation_interval_ms: int = 1_000
    frozen_frame_threshold: int = 30
    """Identical consecutive frames before silent-failure suspicion."""

    hysteresis_samples: int = 3
    """State changes require persistence, to avoid alarm storms."""

    def __post_init__(self) -> None:
        if self.report_timeout_ms < 1:
            raise ValidationError("health.report_timeout_ms must be >= 1")
        if self.hysteresis_samples < 1:
            raise ValidationError("health.hysteresis_samples must be >= 1")


@dataclass(frozen=True, slots=True)
class MetricsSection:
    max_label_cardinality: int = 512
    histogram_window: int = 2048
    export_interval_ms: int = 10_000


@dataclass(frozen=True, slots=True)
class DetectionSection:
    """Detection operating envelope (Flow 2).

    Resource- and capability-shaped only. There is no slot here for "detect
    people more carefully in the kitchen": ``priority_class`` on a camera profile
    is the only ordering input, and the platform never interprets it (V1/V2).
    """

    enabled: bool = False
    """Off unless a deployment declares detectors. Flow 1 behaviour is the
    default, so adding Flow 2 to an existing site is an explicit act."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    """Used only when the platform applies NMS because the adapter declared it
    does not (port obligation D4)."""

    max_detections_per_frame: int = 300
    max_batch_size: int = 8
    batch_max_wait_ms: int = 5
    """Dual trigger with ``max_batch_size``. **0 means flush immediately**, which
    is what deterministic mode requires: batch composition must not depend on
    arrival timing (08_RUNTIME section 4.3)."""

    inference_timeout_ms: int = 2_000
    queue_capacity: int = 64
    """Bounded, always. An unbounded inference queue is a memory leak with a
    delayed fuse."""

    half_precision: bool = False
    dynamic_resolution: bool = True
    """Honour the fidelity tier the Frame Scheduler selected under pressure."""

    warmup_enabled: bool = True
    slow_inference_warn_ms: int = 500
    apply_platform_nms: bool = True
    """Apply NMS when the adapter declares it did not. A platform cannot correct
    for what it does not know, but it can act on what was declared."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValidationError("detection.confidence_threshold must be in [0,1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValidationError("detection.iou_threshold must be in [0,1]")
        if self.max_detections_per_frame < 1:
            raise ValidationError("detection.max_detections_per_frame must be >= 1")
        if self.max_batch_size < 1:
            raise ValidationError("detection.max_batch_size must be >= 1")
        if self.batch_max_wait_ms < 0:
            raise ValidationError("detection.batch_max_wait_ms must be >= 0")
        if self.queue_capacity < 1:
            raise ValidationError("detection.queue_capacity must be >= 1")
        if self.inference_timeout_ms < 1:
            raise ValidationError("detection.inference_timeout_ms must be >= 1")


@dataclass(frozen=True, slots=True)
class TrackingSection:
    """Tracking operating envelope (M6, Flow 3).

    Resource- and capability-shaped only. There is no slot here for "track
    people more carefully near the till": tracking has no vocabulary for what a
    region means, and adding one would breach the Semantic Ceiling (V1/V2).

    Every bound is finite and none may be disabled. Together they are the whole
    of "track memory" — tracking owns short temporal continuity and must never
    become long-term memory (03_MODULES M6 state ownership).
    """

    enabled: bool = False
    """Off unless a deployment declares a tracker. Flow 2 behaviour is the
    default, so adding Flow 3 to a running site is an explicit act."""

    tracker_id: str = ""
    """Selects the adapter. Resolved through a factory table in the composition
    root; the platform never learns what it names.

    **No default on purpose.** Naming one here would make the config schema —
    a platform module — the place that decides which tracker is right, which is
    the coupling the port structure exists to prevent. A deployment that enables
    tracking must say which tracker it wants."""

    # --- association ------------------------------------------------------- #
    iou_threshold: float = 0.1
    """Hard geometric gate. Below this a pair is not a candidate at all."""

    max_association_cost: float = 0.7
    ambiguity_margin: float = 0.05
    """Minimum cost gap to the runner-up before a match is asserted. Below it
    the association is refused and the track coasts — 03_MODULES M6's *"prefer
    terminating a track over a wrong association"* as a number."""

    iou_weight: float = 0.6
    distance_weight: float = 0.25
    scale_weight: float = 0.15
    gate_multiplier: float = 3.0

    # --- lifecycle / track memory ------------------------------------------ #
    min_hits_to_confirm: int = 3
    max_coast_frames: int = 5
    max_lost_frames: int = 15
    """Recovery window. After this the track terminates and its id is retired
    for the epoch."""

    max_age_frames: int = 36_000
    max_tracks_per_camera: int = 256
    history_length: int = 32
    """Frames retained per track. Bounded, so memory is a function of track
    count rather than of how long any track has lived (port obligation T8)."""

    # --- runtime ----------------------------------------------------------- #
    frame_timeout_ms: int = 500
    """Ceiling on one frame's tracking work.

    Backpressure on the Detection-to-Tracking edge is **blocking** by design
    (08_RUNTIME section 5.2: *"ordering matters; dropping here corrupts
    tracks"*), so a hung tracker would otherwise stall its camera forever. This
    bounds the wait and lets the frame fail instead."""

    slow_frame_warn_ms: int = 50
    require_deterministic: bool = False
    """When true, a tracker declaring itself non-deterministic fails to bind
    (invariant V13)."""

    appearance_enabled: bool = False
    """Appearance embeddings are C2 biometric data, **disabled by default**
    (12_SECURITY section 4.3). No provider ships; enabling this without one is a
    configuration error rather than a silent downgrade to geometry."""

    def __post_init__(self) -> None:
        if self.enabled and not self.tracker_id:
            raise ValidationError(
                "tracking.enabled is true but no tracking.tracker_id is named; the "
                "platform has no opinion about which tracker is right for a site"
            )
        for name in ("iou_threshold", "max_association_cost"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"tracking.{name} must be in [0,1]")
        if self.ambiguity_margin < 0.0:
            raise ValidationError("tracking.ambiguity_margin must be >= 0")
        for name in ("iou_weight", "distance_weight", "scale_weight"):
            if getattr(self, name) < 0.0:
                raise ValidationError(f"tracking.{name} must be >= 0")
        if self.iou_weight + self.distance_weight + self.scale_weight <= 0.0:
            raise ValidationError("tracking association weights must sum above zero")
        if self.gate_multiplier < 0.0:
            raise ValidationError("tracking.gate_multiplier must be >= 0")
        if self.min_hits_to_confirm < 1:
            raise ValidationError("tracking.min_hits_to_confirm must be >= 1")
        if self.max_coast_frames < 0:
            raise ValidationError("tracking.max_coast_frames must be >= 0")
        if self.max_lost_frames < 0:
            raise ValidationError("tracking.max_lost_frames must be >= 0")
        if self.max_age_frames < 1:
            raise ValidationError("tracking.max_age_frames must be >= 1")
        if self.max_tracks_per_camera < 1:
            raise ValidationError("tracking.max_tracks_per_camera must be >= 1")
        if self.history_length < 1:
            raise ValidationError("tracking.history_length must be >= 1")
        if self.frame_timeout_ms < 1:
            raise ValidationError("tracking.frame_timeout_ms must be >= 1")


@dataclass(frozen=True, slots=True)
class RegistrySection:
    """Object Registry operating envelope (M7, Flow 4).

    Resource- and horizon-shaped only. There is no slot here for "treat objects
    near the till as customers": the registry has no vocabulary for what a region
    means, and adding one would breach the Semantic Ceiling (V1/V2).

    Every horizon is finite. Section M7 calls an unbounded registry *"a memory
    leak with a face"*, and the bounds below are what make that impossible rather
    than merely unlikely.
    """

    enabled: bool = False
    """Off unless a deployment opts in. Flow 3 behaviour is the default, so
    adding the registry to a running site is an explicit act."""

    # --- lifecycle horizons -------------------------------------------------- #
    min_observations_to_confirm: int = 3
    """Sightings before a provisional object is asserted as real. Below this it
    is probably tracker noise, and confirming it would put a phantom into the
    platform's first durable state."""

    provisional_horizon_ms: int = 3_000
    occlusion_horizon_ms: int = 10_000
    """Believed-present without measurement. Past this the claim weakens from
    ``occluded`` to ``dormant`` — still retained, no longer asserted present."""

    dormant_horizon_ms: int = 120_000
    retention_horizon_ms: int = 600_000
    """How long a departed object is kept before expiry. The bound on the
    registry's memory."""

    max_objects_per_camera: int = 512

    # --- binding ------------------------------------------------------------- #
    max_reentry_distance: float = 0.25
    max_reentry_gap_ms: int = 30_000
    ambiguity_margin: float = 0.15
    """Minimum score gap between the best and second-best re-entry candidate.
    Below it the match is refused, a new object is minted, and the alternatives
    are published — section M7's *"never guess silently"* as a number."""

    min_binding_confidence: float = 0.3
    epoch_rebind_penalty: float = 0.5
    """Multiplier applied when re-binding across a tracker epoch. 07_STATE
    section 9.3 requires re-binding after a restart carry *explicitly reduced
    confidence*."""

    class_must_match: bool = True

    # --- bounded history ------------------------------------------------------ #
    spatial_history_length: int = 64
    class_history_length: int = 32
    """Both are rings. Unbounded history here is the most likely long-run memory
    leak in the platform, which is why bounding is structural rather than a
    tuning parameter (section M7 Performance)."""

    # --- geometry -------------------------------------------------------------- #
    edge_margin: float = 0.02
    """How close to a frame edge counts as having left the field of view. This is
    what separates ``active -> dormant`` from ``active -> occluded``: an object
    that walks out of frame is a different claim from one that stops being
    measurable in place."""

    # --- durability ------------------------------------------------------------ #
    persistence_enabled: bool = True
    persistence_interval_ms: int = 5_000
    """Durable writes are batched and asynchronous; the hot path updates memory
    and enqueues, never blocking on I/O (section M7 Performance)."""

    expiry_interval_ms: int = 1_000
    """How often horizons are advanced for cameras that have gone quiet. Without
    it, a camera that stops sending frames would freeze its objects forever."""

    slow_frame_warn_ms: int = 5

    def __post_init__(self) -> None:
        if self.min_observations_to_confirm < 1:
            raise ValidationError("registry.min_observations_to_confirm must be >= 1")
        for name in (
            "provisional_horizon_ms",
            "occlusion_horizon_ms",
            "dormant_horizon_ms",
            "retention_horizon_ms",
            "max_reentry_gap_ms",
            "persistence_interval_ms",
            "expiry_interval_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValidationError(f"registry.{name} must be positive")
        if self.max_objects_per_camera < 1:
            raise ValidationError("registry.max_objects_per_camera must be >= 1")
        for name in (
            "max_reentry_distance",
            "ambiguity_margin",
            "min_binding_confidence",
            "edge_margin",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"registry.{name} must be in [0,1]")
        if not 0.0 < self.epoch_rebind_penalty <= 1.0:
            raise ValidationError("registry.epoch_rebind_penalty must be in (0,1]")
        if self.spatial_history_length < 1 or self.class_history_length < 1:
            raise ValidationError("registry history lengths must be >= 1")
        if self.occlusion_horizon_ms >= self.dormant_horizon_ms:
            raise ValidationError(
                "registry.occlusion_horizon_ms must be shorter than "
                "dormant_horizon_ms; an object cannot become dormant before it "
                "has finished being occluded"
            )


@dataclass(frozen=True, slots=True)
class CroppingSection:
    """Crop Manager operating envelope (M8, Flow 5).

    Budget-, quality- and geometry-shaped only. There is no slot here for "always
    analyse people near the entrance": M8 has no vocabulary for what a region
    means, and a priority class below is an **opaque string** the platform orders
    by and never interprets (V1/V2).

    Note what the defaults encode. ``understanding_calls_per_hour`` is the single
    most important number in a deployment's cost model — §M8 calls this *"the
    single most important cost-control point in the platform"* — and it is a hard
    ceiling rather than a target, because a budget that can be exceeded under
    load is not a budget.
    """

    enabled: bool = False
    """Off unless a deployment opts in. Flow 4 behaviour is the default, so
    adding crop management to a running site is an explicit act."""

    # --- budget ---------------------------------------------------------------- #
    understanding_calls_per_hour: float = 36_000.0
    """The ceiling. 10/s sustained — §M8's worked cost model puts a realistic
    100-camera site at 10–15 calls/second, so this is one GPU's worth."""

    budget_window_ms: int = 60_000
    """Reconciliation window. Shorter means tighter enforcement and more
    contention; longer allows burstier spending within the same hourly rate."""

    priority_classes: tuple[str, ...] = ()
    """Ordering, highest first. **Opaque.** The platform sorts by position and
    never asks what a class means. An unlisted class sorts last, so a consumer
    typo degrades one demand rather than stopping the pipeline."""

    # --- triggering ------------------------------------------------------------ #
    trigger_policy: str = "trigger.default"
    appearance_change_threshold: float = 0.25
    low_confidence_threshold: float = 0.5
    periodic_refresh_ms: int = 300_000
    """Cadence floor — the longest a demanded object goes without a look."""

    max_candidates_per_frame: int = 128
    """Bounds the per-frame evaluation cost. Beyond this, the lowest-priority
    candidates are skipped with ``PRIORITY_PREEMPTED`` rather than silently
    truncated (V8)."""

    # --- quality --------------------------------------------------------------- #
    quality_estimator: str = "quality.heuristic"
    min_scale_pixels: float = 48.0
    """The gate's scale floor — §M8 names scale *"the strongest single
    predictor"* of whether a claim will be usable."""

    good_scale_pixels: float = 160.0
    max_truncation: float = 0.5
    max_occlusion: float = 0.7
    max_blur: float = 0.85
    max_crowding: float = 0.9
    reject_extreme_exposure: bool = False

    # --- crop geometry --------------------------------------------------------- #
    crop_strategy: str = "crop.padded"
    crop_padding: float = 0.15
    crop_width: int = 224
    crop_height: int = 224
    """The canonical crop size. **One format for every model** — §M8: *"No YOLO
    crop. No CLIP crop. No Florence crop."* A model wanting something else
    resizes down in its own adapter."""

    preserve_aspect: bool = True
    """Letterbox rather than squash. A squashed crop produces attributes about a
    distorted object and the distortion is invisible in the output."""

    interpolation: str = "nearest"
    colour_space: str = "bgr24"

    # --- retention ------------------------------------------------------------- #
    retention_mode: str = "ephemeral"
    """``ephemeral`` | ``evidence`` | ``never_persist``. 12_SECURITY section 2.3's
    no-evidence mode is ``never_persist``."""

    evidence_ttl_ms: int = 86_400_000
    """24 hours. 12_SECURITY section 3 bounds C1 imagery at 24–72 hours."""

    # --- caching --------------------------------------------------------------- #
    dedup_cache_size: int = 4_096
    """Bounded LRU. §M8's failure table names cache growth as a failure mode: an
    unbounded dedup cache is a memory leak that looks like a hit-rate win."""

    # --- alarms ---------------------------------------------------------------- #
    gate_rejection_spike_threshold: float = 0.5
    gate_rejection_sample_size: int = 20
    """Rejections observed before a spike may be declared. A small sample would
    alarm on ordinary variance and train operators to ignore the alarm."""

    capability_gap_threshold: int = 50
    """Consecutive unsatisfiable attempts before a capability gap is published.
    High, because telling a consumer to stop waiting is a claim about the future
    and should be slow to make."""

    slow_frame_warn_ms: int = 10

    def __post_init__(self) -> None:
        if self.understanding_calls_per_hour < 0:
            raise ValidationError(
                "cropping.understanding_calls_per_hour must be non-negative"
            )
        for name in (
            "budget_window_ms",
            "periodic_refresh_ms",
            "evidence_ttl_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValidationError(f"cropping.{name} must be positive")
        for name in (
            "appearance_change_threshold",
            "low_confidence_threshold",
            "max_truncation",
            "max_occlusion",
            "max_blur",
            "max_crowding",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"cropping.{name} must be in [0,1]")
        if self.min_scale_pixels < 0:
            raise ValidationError("cropping.min_scale_pixels must be non-negative")
        if self.good_scale_pixels < self.min_scale_pixels:
            raise ValidationError(
                "cropping.good_scale_pixels must be >= min_scale_pixels; a crop "
                "cannot be 'good' at a scale the gate already rejects"
            )
        if not 0.0 <= self.crop_padding <= 4.0:
            raise ValidationError("cropping.crop_padding must be in [0,4]")
        if self.crop_width < 1 or self.crop_height < 1:
            raise ValidationError("cropping crop dimensions must be >= 1")
        if self.max_candidates_per_frame < 1:
            raise ValidationError("cropping.max_candidates_per_frame must be >= 1")
        if self.dedup_cache_size < 1:
            raise ValidationError("cropping.dedup_cache_size must be >= 1")
        if self.retention_mode not in ("ephemeral", "evidence", "never_persist"):
            raise ValidationError(
                f"cropping.retention_mode must be ephemeral, evidence or "
                f"never_persist, got '{self.retention_mode}'"
            )
        if not 0.0 <= self.gate_rejection_spike_threshold <= 1.0:
            raise ValidationError(
                "cropping.gate_rejection_spike_threshold must be in [0,1]"
            )
        if self.gate_rejection_sample_size < 1:
            raise ValidationError("cropping.gate_rejection_sample_size must be >= 1")
        if self.capability_gap_threshold < 1:
            raise ValidationError("cropping.capability_gap_threshold must be >= 1")
        if len(set(self.priority_classes)) != len(self.priority_classes):
            raise ValidationError(
                "cropping.priority_classes must not repeat a class; a duplicate "
                "makes the ordering ambiguous and shedding non-deterministic"
            )


@dataclass(frozen=True, slots=True)
class UnderstandingSection:
    """Understanding Engine operating envelope (M9, Flow 6).

    Resource-, reliability- and routing-shaped only. There is no slot here for
    "ask whether the person looks suspicious": the *question* lives in a prompt
    asset owned by M10, and the *vocabulary* lives in the Attribute Schema
    Registry. Both are outside this file on purpose — a business question
    reachable from configuration would bypass two of the ceiling's three
    enforcement points (00_CHARTER section 4.3).

    Note what is also absent: any model name. Routing is by declared capability,
    so adding a model is binding an adapter, never editing a setting.
    """

    enabled: bool = False
    """Off unless a deployment opts in. Flow 5 behaviour is the default, so adding
    understanding to a running site is an explicit act."""

    # --- reliability ----------------------------------------------------------- #
    timeout_ms: int = 2_000
    """Per-call deadline. 11_PERFORMANCE section 1.1 puts a VLM call at ~200 ms,
    so this is an order of magnitude of headroom before the call is abandoned."""

    max_retries: int = 1
    """10_RELIABILITY section 4.3: *"Retry once with backoff; then fallback
    model; then fail the request."* One, not three — a transient blip resolves on
    the first retry and a real outage does not resolve on the third."""

    retry_backoff_ms: int = 100
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_ms: int = 30_000
    """Consecutive failures before a model is shed, and how long for. An adapter
    crash is classified **systemic**, and retrying a systemic failure makes it
    worse."""

    fallback_depth: int = 2
    """How many fallbacks to try. The chain terminates in explicit
    unavailability, never in a guess (10_RELIABILITY section 7.2)."""

    # --- concurrency ------------------------------------------------------------ #
    max_concurrency: int = 4
    """In-flight calls per local model. 08_RUNTIME section 4.4: *"Long VLM calls
    are not preempted; instead concurrency is capped so that the detector's
    latency budget is protected."*"""

    remote_concurrency: int = 2
    """Remote adapters get their own, tighter budget — a cloud endpoint's rate
    limit is a different constraint from a local GPU's memory, and one number
    cannot express both."""

    max_batch_size: int = 4
    batch_max_wait_ms: int = 10
    """Dual trigger with ``max_batch_size``. **0 flushes immediately**, which is
    what deterministic mode requires: batch composition must not depend on
    arrival timing (08_RUNTIME section 4.3)."""

    # --- routing ---------------------------------------------------------------- #
    prefer_local_models: bool = True
    """A site with a data-residency policy must not ship imagery to a remote
    endpoint because it was marginally cheaper (12_SECURITY)."""

    prefer_deterministic_models: bool = False
    prefer_coverage: bool = True
    """Prefer one understander covering the whole request over a cheaper one
    covering part. Section M9 puts attribute batching in one prompt at a
    *"3-5x saving"*."""

    allow_remote_understanders: bool = True
    """When false, an adapter declaring remote residency is refused at binding
    rather than discovered in an export audit."""

    # --- inference -------------------------------------------------------------- #
    temperature: float = 0.0
    """Zero by default: a deterministic answer is worth more to this platform
    than a fluent one, and V13 wants a replay to reproduce."""

    coercion_strategy: str = "coercion.json"

    # --- cache ------------------------------------------------------------------- #
    cache_capacity: int = 2_048
    cache_ttl_ms: int = 3_600_000
    """Bounded and TTL'd. The key is correct by construction, so the TTL exists
    for age rather than for correctness — a consumer reading a six-hour-old
    answer should be told it is six hours old."""

    # --- evidence ---------------------------------------------------------------- #
    evidence_retention: str = "evidence"
    max_unstructured_note_chars: int = 4_096
    """02_VOM section 9.3 requires the preserved note be bounded."""

    # --- alarms ------------------------------------------------------------------- #
    schema_drift_window: int = 20
    schema_drift_threshold: float = 0.5
    """Fraction of recent results carrying a ceiling violation before a drift
    alarm fires. Section M9: *"If the rate is sustained, alarm — this means a
    prompt has drifted beyond its declared schema."*"""

    slow_call_warn_ms: int = 1_000

    def __post_init__(self) -> None:
        for name in (
            "timeout_ms",
            "retry_backoff_ms",
            "circuit_breaker_cooldown_ms",
            "cache_ttl_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValidationError(f"understanding.{name} must be positive")
        if self.max_retries < 0:
            raise ValidationError("understanding.max_retries must be >= 0")
        if self.circuit_breaker_threshold < 1:
            raise ValidationError(
                "understanding.circuit_breaker_threshold must be >= 1"
            )
        if self.fallback_depth < 0:
            raise ValidationError("understanding.fallback_depth must be >= 0")
        if self.max_concurrency < 1 or self.remote_concurrency < 1:
            raise ValidationError("understanding concurrency limits must be >= 1")
        if self.max_batch_size < 1:
            raise ValidationError("understanding.max_batch_size must be >= 1")
        if self.batch_max_wait_ms < 0:
            raise ValidationError("understanding.batch_max_wait_ms must be >= 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("understanding.temperature must be in [0,2]")
        if self.cache_capacity < 1:
            raise ValidationError("understanding.cache_capacity must be >= 1")
        if self.schema_drift_window < 1:
            raise ValidationError("understanding.schema_drift_window must be >= 1")
        if not 0.0 <= self.schema_drift_threshold <= 1.0:
            raise ValidationError(
                "understanding.schema_drift_threshold must be in [0,1]"
            )
        if self.max_unstructured_note_chars < 1:
            raise ValidationError(
                "understanding.max_unstructured_note_chars must be >= 1"
            )
        if self.evidence_retention not in ("ephemeral", "evidence", "never_persist"):
            raise ValidationError(
                f"understanding.evidence_retention must be ephemeral, evidence or "
                f"never_persist, got '{self.evidence_retention}'"
            )


@dataclass(frozen=True, slots=True)
class SynthesisSection:
    """Observation Builder operating envelope (M11, Flow 7).

    Suppression-, cadence- and alarm-shaped only. There is no slot here for
    "publish an alert when dwell exceeds 5 minutes": the *threshold* is a
    business judgment (V1) and the *attribute vocabulary* lives in the Attribute
    Schema Registry. Both are outside this file on purpose — a business
    conclusion reachable from configuration would bypass the ceiling's final
    gate.
    """

    enabled: bool = False
    """Off unless a deployment opts in. Flow 6 behaviour is the default, so
    adding synthesis to a running site is an explicit act."""

    # --- suppression ----------------------------------------------------------- #
    suppression_policy: str = "suppression.exact"
    heartbeat_ms: int = 30_000
    """The V8 floor. §M11: *"a consumer must be able to distinguish 'unchanged'
    from 'stopped observing,' so unchanged objects still publish at a slow floor
    rate."* Suppression without a heartbeat makes a working camera and a dead one
    produce the same silence."""

    position_threshold: float = 0.01
    """Normalized displacement below which a spatial observation says nothing
    new. Used by ``suppression.threshold`` only."""

    suppression_capacity: int = 4_096
    """Tracked subjects per camera. Unbounded would grow with every object a
    camera has ever seen; an evicted subject simply republishes, and §M11 is
    explicit that *"brief duplication is harmless, missing data is not."*"""

    # --- alarms ---------------------------------------------------------------- #
    rejection_window: int = 20
    rejection_alarm_rate: float = 0.5
    """Fraction of recent observations rejected by the ceiling gate before an
    alarm fires. §M11: *"count, alarm on sustained rate."* One rejection is a
    producer being creative; a sustained rate means a producer has drifted.

    Named for *rejection* rather than *violation* deliberately. A config key
    reading ``violation_threshold`` is indistinguishable from a business rule —
    the tuning knob for "how many safety violations before we alert" — and the
    architecture guard that policed this namespace could not tell the two apart.
    These count schema rejections, which is the platform's own enforcement of
    V1 and the opposite of a business rule."""

    # --- evidence -------------------------------------------------------------- #
    evidence_retention: str = "evidence"
    require_evidence_for_attributes: bool = True
    """An attribute observation without evidence cannot be audited. Configurable
    only so a forensic-free deployment can state that choice explicitly rather
    than discovering it."""

    slow_build_warn_ms: int = 5

    def __post_init__(self) -> None:
        if self.heartbeat_ms <= 0:
            raise ValidationError("synthesis.heartbeat_ms must be positive")
        if not 0.0 <= self.position_threshold <= 1.0:
            raise ValidationError("synthesis.position_threshold must be in [0,1]")
        if self.suppression_capacity < 1:
            raise ValidationError("synthesis.suppression_capacity must be >= 1")
        if self.rejection_window < 1:
            raise ValidationError("synthesis.rejection_window must be >= 1")
        if not 0.0 <= self.rejection_alarm_rate <= 1.0:
            raise ValidationError("synthesis.rejection_alarm_rate must be in [0,1]")
        if self.evidence_retention not in ("ephemeral", "evidence", "never_persist"):
            raise ValidationError(
                f"synthesis.evidence_retention must be ephemeral, evidence or "
                f"never_persist, got '{self.evidence_retention}'"
            )


@dataclass(frozen=True, slots=True)
class StateSection:
    """Vision State operating envelope (M12, Flow 7).

    Every history bound here is finite, and 07_STATE section 6.3 explains why
    that is structural rather than a tuning choice: *"All in-memory history is
    bounded by both count and time, and the bound is a structural property of the
    ring buffers rather than a tunable that might be misconfigured to infinity."*
    A node's steady-state memory is calculable before deployment because of these
    numbers.

    There is no slot for a business aggregate, an alert or a retention rule with
    a business meaning. Section 10's test applies: *"would this field mean the
    same thing in a hospital, a warehouse, and a city street?"*
    """

    enabled: bool = False

    # --- projection ------------------------------------------------------------ #
    max_objects_per_partition: int = 512
    trajectory_points: int = 64
    attribute_history: int = 8
    class_history: int = 16
    """All four are ring bounds. §6.3: *"Because every dimension is bounded, a
    node's steady-state memory is calculable before deployment rather than
    discovered in production."*"""

    working_history_ms: int = 300_000
    """~5 minutes. §6.2's working horizon — for perception continuity, not for
    analytics. §6.1: *"History exists for perception, not for analytics."*"""

    # --- durability ------------------------------------------------------------- #
    log_buffer_capacity: int = 4_096
    """Bounded local buffer for a storage outage. 10_RELIABILITY section 4.4 step
    4: when it fills the partition **stops accepting observations** rather than
    dropping facts silently."""

    commit_batch_size: int = 64
    commit_interval_ms: int = 100
    """Append is *"sequential and batched — the cheapest possible durable write
    pattern"* (§M12 Performance)."""

    # --- retention -------------------------------------------------------------- #
    log_retention_ms: int = 604_800_000
    """7 days hot (07_STATE section 8.1). Archive tiers are a storage-adapter
    concern, not a projection one."""

    # --- coverage ---------------------------------------------------------------- #
    stale_object_ms: int = 60_000
    """How long without a measured sighting before an object is reported stale.
    Descriptive only — the platform reports staleness and never decides what it
    means."""

    def __post_init__(self) -> None:
        for name in (
            "max_objects_per_partition",
            "trajectory_points",
            "attribute_history",
            "class_history",
            "log_buffer_capacity",
            "commit_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValidationError(f"state.{name} must be >= 1")
        for name in (
            "working_history_ms",
            "commit_interval_ms",
            "log_retention_ms",
            "stale_object_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValidationError(f"state.{name} must be positive")


@dataclass(frozen=True, slots=True)
class ModelsSection:
    """Model artifact and device policy (M18, Flow 2)."""

    artifact_cache_dir: str = ".vision_os/models"
    device_preference: str = "auto"
    """``auto`` | ``cpu`` | a concrete device id such as ``cuda:0``."""

    allow_cpu_fallback: bool = True
    """When false, a site that loses its accelerators reports the capability
    unavailable rather than silently running 50x slower."""

    vram_headroom_fraction: float = 0.1
    warmup_enabled: bool = True
    deployment_context: str = "on_premise"
    """Checked against each model's licence at registration, never discovered in
    production."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.vram_headroom_fraction < 1.0:
            raise ValidationError("models.vram_headroom_fraction must be in [0,1)")


@dataclass(frozen=True, slots=True)
class RuntimeSection:
    drain_timeout_ms: int = 30_000
    pipeline_restart_backoff_ms: int = 1_000
    max_pipeline_restarts: int = 5
    """After this, the camera is marked failed and the platform keeps running."""

    attach_stagger_ms: int = 50
    """One hundred cameras connecting at once is a self-inflicted thundering
    herd that can cause boot itself to fail (08_RUNTIME §7.1)."""

    max_pipelines: int = 512


# --- declarations --------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegionDeclaration:
    region_id: str
    label: str
    """Opaque. Never interpreted by the platform."""

    vertices: tuple[tuple[float, float], ...]
    frame_of_reference: str = "normalized"
    camera_id: str | None = None
    version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class ProfileDeclaration:
    profile_id: str
    target_fps: float
    max_in_flight: int = 4
    priority_class: str = "default"
    inference_width: int = 640
    inference_height: int = 640


@dataclass(frozen=True, slots=True)
class CalibrationDeclaration:
    calibration_id: str
    homography: tuple[tuple[float, float, float], ...] | None = None
    ground_uncertainty_at_unit_distance: float = 0.05


@dataclass(frozen=True, slots=True)
class CameraDeclaration:
    camera_id: str
    tenant_id: str
    site_id: str
    uri: str
    transport: str
    source_semantics: str
    profile_id: str
    width: int = 1920
    height: int = 1080
    fps: float = 25.0
    codec: str = "raw"
    colour_space: str = "bgr24"
    credential_ref: str | None = None
    privacy_policy_id: str | None = None
    region_ids: tuple[str, ...] = ()
    calibration: CalibrationDeclaration | None = None
    labels: dict[str, str] = field(default_factory=dict)
    source_options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TaxonomyClassDeclaration:
    """One visual kind (Flow 2).

    A vertical enters the platform partly through this list. ``person``,
    ``vehicle.forklift``, ``container.tray`` are admissible; ``staff_member``,
    ``patient`` and ``customer`` are roles that no crop can evidence and are
    rejected at registration (invariant V1).
    """

    class_id: str
    geometry_kinds: tuple[str, ...] = ("box",)
    description: str = ""
    status: str = "active"
    superseded_by: str | None = None


@dataclass(frozen=True, slots=True)
class MappingEntryDeclaration:
    native_label: str
    class_id: str
    mapping_confidence: float = 1.0
    notes: str = ""


@dataclass(frozen=True, slots=True)
class DetectorDeclaration:
    """A detector adapter bound to a model artifact (Flow 2).

    This is where a model is *named*, not where one is chosen: the adapter, the
    weights and the label mapping are data, so replacing YOLO with RT-DETR is a
    configuration change plus an adapter, never a platform change (V3).
    """

    detector_id: str
    adapter_id: str
    model_id: str
    model_version: str
    artifact_uri: str
    artifact_hash: str
    role: str = "primary_detector"
    precision: str = "fp32"
    device_kind: str = "cpu"
    vram_bytes: int = 0
    licence: str = "unspecified"
    permitted_contexts: tuple[str, ...] = ()
    native_label_space: str = ""
    unmapped_policy: str = "drop"
    mappings: tuple[MappingEntryDeclaration, ...] = ()
    calibration_id: str | None = None
    runtime_options: tuple[tuple[str, str], ...] = ()
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class StorageSection:
    """M13's adapter selection and retention policy (Flow 8).

    §M13's purpose is that an edge box and a cloud cluster *"differ only in
    adapter selection"*, and this section is where that selection happens. No
    field here describes *how* a store works — only which one is bound and how
    long it keeps things.

    07_STATE §8.1 gives ranges rather than constants, because how long imagery
    may be kept is a regulator's answer, not the platform's.
    """

    evidence_store: str = "evidence.memory"
    """``evidence.memory`` | ``evidence.file`` | ``evidence.null``.

    A site operating 12_SECURITY §2.3's no-evidence mode binds ``evidence.null``
    explicitly, so the choice appears in configuration rather than as an absence
    somebody has to notice."""

    evidence_path: str = ""
    evidence_ttl_ms: int = 172_800_000
    """48 hours — the middle of 07_STATE §8.1's 24-72 hour range for crops.
    Evidence has the shortest tier deliberately: it is the only one containing
    imagery, so retention here is a privacy decision rather than an engineering
    one."""

    raw_output_ttl_ms: int = 604_800_000
    """7 days (§8.1). Longer than imagery because it is text."""

    evidence_max_bytes: int = 4 * 1024 * 1024 * 1024
    evidence_max_blobs: int = 100_000
    evidence_max_blob_bytes: int = 32 * 1024 * 1024
    """Bounded, always. An unbounded evidence store is a memory leak whose fuse
    burns for exactly as long as the disk has space."""

    expiry_interval_ms: int = 3_600_000
    """How often retention runs. A store whose expiry silently stops looks
    identical to one working correctly until capacity is reached."""

    def __post_init__(self) -> None:
        if self.evidence_store not in (
            "evidence.memory",
            "evidence.file",
            "evidence.null",
        ):
            raise ValidationError(
                f"unknown evidence store '{self.evidence_store}'; known stores are "
                f"evidence.memory, evidence.file, evidence.null"
            )
        if self.evidence_store == "evidence.file" and not self.evidence_path:
            raise ValidationError(
                "storage.evidence_path is required for evidence.file; a durable "
                "store with nowhere to write would fail at the first crop rather "
                "than at boot"
            )
        for name in (
            "evidence_ttl_ms",
            "raw_output_ttl_ms",
            "evidence_max_bytes",
            "evidence_max_blobs",
            "evidence_max_blob_bytes",
            "expiry_interval_ms",
        ):
            if getattr(self, name) < 1:
                raise ValidationError(f"storage.{name} must be positive")


@dataclass(frozen=True, slots=True)
class ApiSection:
    """M14's operating envelope (Flow 8).

    Every bound here exists so one consumer cannot degrade the platform for the
    rest — §M14: *"Reject with a bound and a cursor rather than degrading the
    service for everyone."*

    There is no field describing what to serve, only how much. What the API
    serves is fixed by 09_API's contract, and a configurable answer would make
    the contract a per-deployment negotiation rather than a promise.
    """

    enabled: bool = False

    authorizer: str = "authz.deny_all"
    """Defaults to denying everything. A platform that served data until somebody
    remembered to configure a policy has exactly one failure mode and it is a
    breach (obligation Z5)."""

    transport: str = "transport.in_process"

    # --- bounds ------------------------------------------------------------- #
    queries_per_minute: int = 600
    evidence_per_minute: int = 60
    """Tighter than queries. 09_API §6: evidence payloads are *"large and
    sensitive"*, and a consumer able to pull imagery as fast as facts has
    effectively been granted bulk imagery export."""

    subscribes_per_minute: int = 60
    max_page_size: int = 1_000
    max_window_ms: int = 86_400_000
    max_subscriptions_per_principal: int = 32

    # --- subscriptions ------------------------------------------------------- #
    subscription_queue_capacity: int = 1_024
    """Bounded, always. 09_API §3.4: *"Never: unbounded buffering."*"""

    heartbeat_ms: int = 10_000
    """§3.1's default. Without it a healthy subscription over a quiet camera and
    a dead connection produce identical silence."""

    # --- audit ---------------------------------------------------------------- #
    audit_capacity: int = 1_000
    require_evidence_purpose: bool = True
    """12_SECURITY §5.4. Configurable only so a deployment can state the choice
    explicitly rather than discovering it."""

    def __post_init__(self) -> None:
        for name in (
            "queries_per_minute",
            "evidence_per_minute",
            "subscribes_per_minute",
            "max_page_size",
            "max_window_ms",
            "max_subscriptions_per_principal",
            "subscription_queue_capacity",
            "heartbeat_ms",
            "audit_capacity",
        ):
            if getattr(self, name) < 1:
                raise ValidationError(f"api.{name} must be positive")


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    """The fully resolved, validated configuration tree."""

    platform: PlatformSection
    buffer: BufferSection
    scheduler: SchedulerSection
    source: SourceSection
    health: HealthSection
    metrics: MetricsSection
    runtime: RuntimeSection
    detection: DetectionSection = DetectionSection()
    models: ModelsSection = ModelsSection()
    tracking: TrackingSection = TrackingSection()
    registry: RegistrySection = RegistrySection()
    cropping: CroppingSection = CroppingSection()
    understanding: UnderstandingSection = UnderstandingSection()
    synthesis: SynthesisSection = SynthesisSection()
    state: StateSection = StateSection()
    storage: StorageSection = StorageSection()
    api: ApiSection = ApiSection()
    profiles: tuple[ProfileDeclaration, ...] = ()
    regions: tuple[RegionDeclaration, ...] = ()
    cameras: tuple[CameraDeclaration, ...] = ()
    taxonomy: tuple[TaxonomyClassDeclaration, ...] = ()
    detectors: tuple[DetectorDeclaration, ...] = ()


# --- the closed key set --------------------------------------------------- #

SECTION_TYPES: dict[str, type] = {
    "platform": PlatformSection,
    "buffer": BufferSection,
    "scheduler": SchedulerSection,
    "source": SourceSection,
    "health": HealthSection,
    "metrics": MetricsSection,
    "runtime": RuntimeSection,
    "detection": DetectionSection,
    "models": ModelsSection,
    "tracking": TrackingSection,
    "registry": RegistrySection,
    "cropping": CroppingSection,
    "understanding": UnderstandingSection,
    "synthesis": SynthesisSection,
    "state": StateSection,
    "storage": StorageSection,
    "api": ApiSection,
}

LIST_SECTIONS: frozenset[str] = frozenset(
    {"profiles", "regions", "cameras", "taxonomy", "detectors"}
)

ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(SECTION_TYPES) | LIST_SECTIONS

_ENUM_FIELDS: dict[tuple[str, str], type[enum.Enum]] = {
    ("platform", "deployment_profile"): DeploymentProfile,
    ("platform", "clock_mode"): ClockMode,
}


def allowed_keys(section: str) -> frozenset[str]:
    """The closed key set for a scalar section."""
    section_type = SECTION_TYPES.get(section)
    if section_type is None:
        return frozenset()
    return frozenset(section_type.__dataclass_fields__)


def validate(document: dict[str, Any]) -> tuple[str, ...]:
    """Validate a merged document against the closed schema.

    Returns a tuple of violation messages; empty means valid. Unknown keys are
    violations, not warnings — that is what makes the schema closed.
    """
    violations: list[str] = []

    for key in document:
        if key not in ALLOWED_TOP_LEVEL:
            violations.append(
                f"unknown configuration section '{key}'. The schema is closed "
                f"(V2); allowed sections: {sorted(ALLOWED_TOP_LEVEL)}"
            )

    for section in SECTION_TYPES:
        raw = document.get(section)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            violations.append(f"section '{section}' must be a mapping, got {type(raw).__name__}")
            continue
        permitted = allowed_keys(section)
        for key in raw:
            if key not in permitted:
                violations.append(
                    f"unknown key '{section}.{key}'. The schema is closed (V2); "
                    f"allowed keys: {sorted(permitted)}"
                )
        for key, value in raw.items():
            enum_type = _ENUM_FIELDS.get((section, key))
            if enum_type is not None and not _valid_enum(enum_type, value):
                allowed = sorted(m.value for m in enum_type)
                violations.append(f"'{section}.{key}' must be one of {allowed}, got {value!r}")

    for section in LIST_SECTIONS:
        raw = document.get(section)
        if raw is None:
            continue
        if not isinstance(raw, list):
            violations.append(f"section '{section}' must be a list, got {type(raw).__name__}")

    violations.extend(_validate_cameras(document))
    violations.extend(_validate_taxonomy(document))
    violations.extend(_validate_detectors(document))
    return tuple(violations)


#: Role and judgment vocabulary that may never name a taxonomy class.
#:
#: 02_VOM section 8.3 rule 4: ``person`` and ``vehicle.forklift`` are visual kinds
#: any observer would name; ``staff_member``, ``patient`` and ``customer`` are
#: *roles*, and no crop evidences a role. This is the taxonomy's neutrality gate
#: — the Flow 2 counterpart of the attribute registry's gate in Flow 5.
_FORBIDDEN_CLASS_TOKENS: frozenset[str] = frozenset(
    {
        "staff", "employee", "waiter", "chef", "cashier", "clerk", "manager",
        "patient", "nurse", "doctor", "customer", "shopper", "guest", "visitor",
        "intruder", "suspect", "thief", "trespasser",
        "violation", "compliant", "noncompliant", "authorized", "unauthorized",
        "anomaly", "alert", "hazard", "unsafe", "danger",
    }
)


def _validate_taxonomy(document: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    classes = document.get("taxonomy")
    if not isinstance(classes, list):
        return violations

    seen: set[str] = set()
    for index, declaration in enumerate(classes):
        if not isinstance(declaration, dict):
            violations.append(f"taxonomy[{index}] must be a mapping")
            continue
        class_id = declaration.get("class_id")
        if not class_id:
            violations.append(f"taxonomy[{index}].class_id is required")
            continue
        if class_id in seen:
            violations.append(f"duplicate taxonomy class '{class_id}'")
        seen.add(class_id)

        tokens = {token.lower() for token in str(class_id).replace(".", "_").split("_")}
        leaked = tokens & _FORBIDDEN_CLASS_TOKENS
        if leaked:
            violations.append(
                f"taxonomy class '{class_id}' uses {sorted(leaked)}, which names a role "
                f"or a judgment rather than a visual kind. No crop evidences a role "
                f"(invariant V1); register the appearance instead and let the consumer "
                f"assign meaning."
            )

        status = declaration.get("status", "active")
        if status not in ("active", "deprecated", "superseded"):
            violations.append(
                f"taxonomy['{class_id}'].status must be one of "
                f"['active', 'deprecated', 'superseded'], got {status!r}"
            )
        for kind in declaration.get("geometry_kinds", ("box",)) or ("box",):
            if kind not in ("box", "oriented_box", "mask", "keypoints"):
                violations.append(
                    f"taxonomy['{class_id}'] declares unknown geometry kind {kind!r}"
                )

    for declaration in classes:
        if not isinstance(declaration, dict):
            continue
        class_id = declaration.get("class_id")
        if not class_id or "." not in str(class_id):
            continue
        parent = str(class_id).rsplit(".", 1)[0]
        if parent not in seen:
            violations.append(
                f"taxonomy class '{class_id}' has no declared parent '{parent}'; "
                f"an orphan breaks every hierarchical query for '{parent}'"
            )

    return violations


def _validate_detectors(document: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    detectors = document.get("detectors")
    if not isinstance(detectors, list):
        return violations

    declared_classes = {
        c.get("class_id")
        for c in (document.get("taxonomy") or [])
        if isinstance(c, dict)
    }
    declared_classes.add("unknown")

    seen: set[str] = set()
    for position, declaration in enumerate(detectors):
        if not isinstance(declaration, dict):
            violations.append(f"detectors[{position}] must be a mapping")
            continue
        detector_id = declaration.get("detector_id")
        if not detector_id:
            violations.append(f"detectors[{position}].detector_id is required")
            continue
        if detector_id in seen:
            violations.append(f"duplicate detector_id '{detector_id}'")
        seen.add(detector_id)

        for required in (
            "adapter_id",
            "model_id",
            "model_version",
            "artifact_uri",
            "artifact_hash",
        ):
            if not declaration.get(required):
                violations.append(f"detectors['{detector_id}'].{required} is required")

        policy = declaration.get("unmapped_policy", "drop")
        if policy not in ("drop", "emit_as_unknown"):
            violations.append(
                f"detectors['{detector_id}'].unmapped_policy must be one of "
                f"['drop', 'emit_as_unknown'], got {policy!r}"
            )

        precision = declaration.get("precision", "fp32")
        if precision not in ("fp32", "fp16", "int8", "int4"):
            violations.append(
                f"detectors['{detector_id}'].precision must be one of "
                f"['fp32', 'fp16', 'int8', 'int4'], got {precision!r}"
            )

        for entry in declaration.get("mappings", ()) or ():
            if not isinstance(entry, dict):
                violations.append(f"detectors['{detector_id}'] has a malformed mapping entry")
                continue
            class_id = entry.get("class_id")
            if class_id and declared_classes and class_id not in declared_classes:
                violations.append(
                    f"detectors['{detector_id}'] maps '{entry.get('native_label')}' to "
                    f"undeclared taxonomy class '{class_id}'. A mapping is validated at "
                    f"load, not at first frame."
                )

    return violations


def _valid_enum(enum_type: type[enum.Enum], value: Any) -> bool:
    if isinstance(value, enum_type):
        return True
    return any(member.value == value for member in enum_type)


def _validate_cameras(document: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    cameras = document.get("cameras")
    if not isinstance(cameras, list):
        return violations

    profiles = document.get("profiles")
    profile_ids = {
        p.get("profile_id") for p in profiles if isinstance(p, dict)
    } if isinstance(profiles, list) else set()

    regions = document.get("regions")
    region_ids = {
        r.get("region_id") for r in regions if isinstance(r, dict)
    } if isinstance(regions, list) else set()

    seen: set[str] = set()
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            violations.append(f"cameras[{index}] must be a mapping")
            continue
        camera_id = camera.get("camera_id")
        if not camera_id:
            violations.append(f"cameras[{index}].camera_id is required")
            continue
        if camera_id in seen:
            violations.append(f"duplicate camera_id '{camera_id}'")
        seen.add(camera_id)

        for required in ("tenant_id", "site_id", "uri", "transport", "profile_id"):
            if not camera.get(required):
                violations.append(f"cameras['{camera_id}'].{required} is required")

        semantics = camera.get("source_semantics")
        if semantics not in ("realtime", "archival", "discrete"):
            violations.append(
                f"cameras['{camera_id}'].source_semantics must be one of "
                f"['archival', 'discrete', 'realtime'], got {semantics!r}"
            )

        profile_id = camera.get("profile_id")
        if profile_id and profile_ids and profile_id not in profile_ids:
            violations.append(
                f"cameras['{camera_id}'].profile_id '{profile_id}' is not declared. "
                f"Provisioning fails fast at startup, not at first frame."
            )

        for region_id in camera.get("region_ids", ()) or ():
            if region_ids and region_id not in region_ids:
                violations.append(
                    f"cameras['{camera_id}'] references undeclared region '{region_id}'"
                )

    return violations
