"""Typed platform events — Flow 1 set (05_MODULES_PLATFORM_KERNEL M19).

Events are how information travels *upward* without an upward dependency
(01_LAYERED §2). The bus itself is untyped infrastructure; the vocabulary lives
here so that event types are registered rather than stringly-invented at call
sites.

Payloads are small, bounded, and contain no pixels. Anything large becomes a
reference to storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from ...core.model.ids import CameraId, ConfigRevision, ModuleId, PluginId, StreamEpoch
from ...core.model.timebase import Instant


@dataclass(frozen=True, slots=True)
class Event:
    """Base event. ``partition_key`` preserves per-key ordering where declared."""

    event_type: ClassVar[str] = "event"

    occurred_at: Instant
    partition_key: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Flat, transport-ready representation.

        Several subclasses narrow ``detail`` to a human-readable ``str`` — an
        alarm's cause reads better as a sentence than as a bag of keys. Both
        shapes are handled here rather than at each call site, because a
        transport that raised on one event type would drop that event and only
        that event, which is the hardest kind of observability gap to notice.
        """
        base = {
            "event_type": type(self).event_type,
            "occurred_at_ns": self.occurred_at.ns,
            "partition_key": self.partition_key,
        }
        if isinstance(self.detail, dict):
            return {**base, **self.detail}
        return {**base, "detail": self.detail} if self.detail else base


# --- stream lifecycle (M2) ----------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StreamConnected(Event):
    event_type: ClassVar[str] = "stream.connected"
    camera_id: CameraId = CameraId("")
    stream_epoch: StreamEpoch = StreamEpoch(0)


@dataclass(frozen=True, slots=True)
class StreamLost(Event):
    event_type: ClassVar[str] = "stream.lost"
    camera_id: CameraId = CameraId("")
    stream_epoch: StreamEpoch = StreamEpoch(0)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class EpochAdvanced(Event):
    """A reconnect or reconfigure minted a new stream epoch (02_VOM §4.1)."""

    event_type: ClassVar[str] = "stream.epoch_advanced"
    camera_id: CameraId = CameraId("")
    stream_epoch: StreamEpoch = StreamEpoch(0)


@dataclass(frozen=True, slots=True)
class DecodeFailed(Event):
    event_type: ClassVar[str] = "stream.decode_failed"
    camera_id: CameraId = CameraId("")
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MaskFailure(Event):
    """Privacy masking failed; the frame was dropped. Fails closed."""

    event_type: ClassVar[str] = "privacy.mask_failed"
    camera_id: CameraId = CameraId("")
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ClockQualityChanged(Event):
    event_type: ClassVar[str] = "stream.clock_quality_changed"
    camera_id: CameraId = CameraId("")
    quality: str = ""


# --- scheduling (M3) ------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class SustainedDropAlarm(Event):
    """The scheduler is shedding beyond tolerance; perception is thinned (V8)."""

    event_type: ClassVar[str] = "scheduler.sustained_drop"
    camera_id: CameraId = CameraId("")
    reason: str = ""
    effective_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class BudgetExceeded(Event):
    event_type: ClassVar[str] = "scheduler.budget_exceeded"
    pressure: float = 0.0


# --- buffering (M4) ------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PoolPressure(Event):
    event_type: ClassVar[str] = "buffer.pool_pressure"
    location: str = "host"
    in_use: int = 0
    total: int = 0


@dataclass(frozen=True, slots=True)
class LeaseLeaked(Event):
    """A holder exceeded its lease deadline and was force-broken.

    One stuck stage must not exhaust the pool for every camera (V9).
    """

    event_type: ClassVar[str] = "buffer.lease_leaked"
    holder_id: str = ""
    frame_ref: str = ""


@dataclass(frozen=True, slots=True)
class FrameEvicted(Event):
    event_type: ClassVar[str] = "buffer.frame_evicted"
    camera_id: CameraId = CameraId("")
    frame_ref: str = ""


# --- camera registry (M1) ------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CameraChanged(Event):
    event_type: ClassVar[str] = "camera.changed"
    camera_id: CameraId = CameraId("")
    change: str = ""


@dataclass(frozen=True, slots=True)
class ViewpointDriftSuspected(Event):
    """The camera may have moved; calibration no longer describes the view.

    Publishes a suspicion, never an automatic invalidation — a false positive
    must not blind a working site (03_MODULES M1).
    """

    event_type: ClassVar[str] = "camera.viewpoint_drift_suspected"
    camera_id: CameraId = CameraId("")
    evidence: str = ""


# --- configuration (M16) -------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConfigChanged(Event):
    event_type: ClassVar[str] = "config.changed"
    revision: ConfigRevision = ConfigRevision("")
    requires_restart: tuple[str, ...] = ()


# --- plugins (M17) -------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PluginLoaded(Event):
    event_type: ClassVar[str] = "plugin.loaded"
    plugin_id: PluginId = PluginId("")
    port_id: str = ""


@dataclass(frozen=True, slots=True)
class PluginRejected(Event):
    event_type: ClassVar[str] = "plugin.rejected"
    plugin_id: PluginId = PluginId("")
    reason: str = ""


# --- health (M20) --------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HealthChanged(Event):
    event_type: ClassVar[str] = "health.changed"
    component_id: ModuleId = ModuleId("")
    state: str = ""


@dataclass(frozen=True, slots=True)
class CoverageChanged(Event):
    """Observability changed. Consumers must distinguish this from silence (V8)."""

    event_type: ClassVar[str] = "health.coverage_changed"
    camera_id: CameraId = CameraId("")
    status: str = ""
    reason: str = ""
    effective_rate: float = 1.0


@dataclass(frozen=True, slots=True)
class SilentFailureSuspected(Event):
    """A suspicion, not a verdict (10_RELIABILITY §5.2).

    Degrades coverage confidence and alerts an operator; it never blinds a camera
    automatically, because a false positive that blinds a working camera is
    itself an outage.
    """

    event_type: ClassVar[str] = "health.silent_failure_suspected"
    camera_id: CameraId = CameraId("")
    detector: str = ""
    evidence: str = ""


# --- runtime (M15) -------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PipelineAttached(Event):
    event_type: ClassVar[str] = "runtime.pipeline_attached"
    camera_id: CameraId = CameraId("")


@dataclass(frozen=True, slots=True)
class PipelineDetached(Event):
    event_type: ClassVar[str] = "runtime.pipeline_detached"
    camera_id: CameraId = CameraId("")
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Gap(Event):
    """An explicit delivery gap. Never silence (V8, 09_API §3.3)."""

    event_type: ClassVar[str] = "bus.gap"
    subscription_id: str = ""
    dropped: int = 0
    reason: str = ""


# --- models (M18, Flow 2) -------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ModelLoaded(Event):
    event_type: ClassVar[str] = "model.loaded"
    model_id: str = ""
    version: str = ""
    device_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelEvicted(Event):
    event_type: ClassVar[str] = "model.evicted"
    model_id: str = ""
    version: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ModelSwapped(Event):
    """A role's bound version changed.

    A fallback that is never noticed becomes permanent, and the platform quietly
    runs on its worst model forever. This event is how that gets noticed.
    """

    event_type: ClassVar[str] = "model.swapped"
    role: str = ""
    model_id: str = ""
    version: str = ""
    previous: str = ""


@dataclass(frozen=True, slots=True)
class DevicePressure(Event):
    event_type: ClassVar[str] = "model.device_pressure"
    device_id: str = ""
    utilization: float = 0.0


@dataclass(frozen=True, slots=True)
class CanaryPromoted(Event):
    event_type: ClassVar[str] = "model.canary_promoted"
    role: str = ""
    model_id: str = ""
    version: str = ""


# --- detection (M5, Flow 2) ------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class DetectionCompleted(Event):
    """A frame was detected on. Carries counts, never pixels (invariant V12)."""

    event_type: ClassVar[str] = "detection.completed"
    camera_id: CameraId = CameraId("")
    frame_ref: str = ""
    detection_count: int = 0
    inference_ms: float = 0.0
    batch_size: int = 1
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class DetectionFailed(Event):
    """Detection could not produce a result.

    Distinct from a completed detection with zero results: "nothing was there"
    and "we could not look" are different facts (port obligation D5).
    """

    event_type: ClassVar[str] = "detection.failed"
    camera_id: CameraId = CameraId("")
    frame_ref: str = ""
    reason: str = ""
    failure_class: str = ""


@dataclass(frozen=True, slots=True)
class DetectorLoaded(Event):
    event_type: ClassVar[str] = "detection.detector_loaded"
    adapter_id: str = ""
    model_id: str = ""
    producible_classes: int = 0


@dataclass(frozen=True, slots=True)
class DetectorUnloaded(Event):
    event_type: ClassVar[str] = "detection.detector_unloaded"
    adapter_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ThresholdExceeded(Event):
    """A configured operating bound was crossed.

    Names the bound and both values, so the reader never has to guess which
    threshold or by how much.
    """

    event_type: ClassVar[str] = "detection.threshold_exceeded"
    camera_id: CameraId = CameraId("")
    threshold_name: str = ""
    limit: float = 0.0
    observed: float = 0.0


@dataclass(frozen=True, slots=True)
class PerformanceWarning(Event):
    """Detection is slower than its declared profile.

    A warning, not a verdict: it degrades nothing by itself and exists so an
    operator sees drift before it becomes shedding.
    """

    event_type: ClassVar[str] = "detection.performance_warning"
    camera_id: CameraId = CameraId("")
    metric: str = ""
    observed: float = 0.0
    expected: float = 0.0


# --- tracking (M6, Flow 3) ------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrackCreated(Event):
    """A new track was started.

    ``track_id`` is the full composite ``camera/epoch/#local``. Publishing the
    bare local id would let a subscriber treat it as an identity, which it is
    not (invariant V10).
    """

    event_type: ClassVar[str] = "tracking.track_created"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    tracker_epoch: int = 0
    class_id: str = ""
    frame_ref: str = ""


@dataclass(frozen=True, slots=True)
class TrackUpdated(Event):
    """A track was measured again this frame."""

    event_type: ClassVar[str] = "tracking.track_updated"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    state: str = ""
    association_confidence: float = 0.0
    measurement_basis: str = "measured"


@dataclass(frozen=True, slots=True)
class TrackRecovered(Event):
    """A coasting or lost track was re-associated.

    A transition, not a state: recovery is expressible entirely as
    ``coasting|lost -> confirmed``, so it is published rather than modelled as a
    sixth lifecycle state the architecture does not define.
    """

    event_type: ClassVar[str] = "tracking.track_recovered"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    coasted_frames: int = 0
    previous_state: str = ""


@dataclass(frozen=True, slots=True)
class TrackLost(Event):
    """A track exhausted its coast budget and entered the recovery window."""

    event_type: ClassVar[str] = "tracking.track_lost"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    coasted_frames: int = 0
    break_reason: str = ""


@dataclass(frozen=True, slots=True)
class TrackTerminated(Event):
    """A track ended. Final — the id is retired for the epoch.

    ``break_reason`` is the field that makes tracker regressions findable:
    *"we lost 40% more tracks this week, all with detector_miss"* points at the
    detector rather than the tracker (02_VOM section 10.5).
    """

    event_type: ClassVar[str] = "tracking.track_terminated"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    break_reason: str = ""
    age_frames: int = 0
    hit_count: int = 0
    lifetime_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class AssociationFailure(Event):
    """An association was **refused** as too ambiguous to assert.

    Not a wrong association — the platform cannot know that without ground
    truth. This is the tracker declining to bind rather than risking an ID
    switch, which 03_MODULES M6 requires it prefer.
    """

    event_type: ClassVar[str] = "tracking.association_failure"
    camera_id: CameraId = CameraId("")
    track_id: str = ""
    best_cost: float = 0.0
    runner_up_cost: float = 0.0
    margin: float = 0.0


@dataclass(frozen=True, slots=True)
class TrackingWarning(Event):
    """Tracking is degraded but running."""

    event_type: ClassVar[str] = "tracking.warning"
    camera_id: CameraId = CameraId("")
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TrackerLoaded(Event):
    event_type: ClassVar[str] = "tracking.tracker_loaded"
    adapter_id: str = ""
    version: str = ""
    deterministic: bool = True
    is_fallback: bool = False
    """True when the platform dropped to ``tracker.iou`` because the configured
    tracker failed. Degradation is announced, never silent (invariant V9)."""


@dataclass(frozen=True, slots=True)
class TrackerUnloaded(Event):
    event_type: ClassVar[str] = "tracking.tracker_unloaded"
    adapter_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TrackerEpochAdvanced(Event):
    """A camera's tracker was reset and a new epoch minted.

    Consumers must see this: without it, every track vanishing at once and new
    ids appearing looks like the whole scene teleported (03_MODULES M6 failure
    handling).
    """

    event_type: ClassVar[str] = "tracking.epoch_advanced"
    camera_id: CameraId = CameraId("")
    previous_epoch: int = 0
    epoch: int = 0
    reason: str = ""
    discarded_tracks: int = 0


# --- object registry (M7, Flow 4) ------------------------------------------ #
#
# Exactly the five types named by M7's ``subscribe()`` (03_MODULES section M7),
# plus the population alarm its failure table requires. No semantic or business
# event exists here: the registry publishes what happened to an object, never
# what it means.


@dataclass(frozen=True, slots=True)
class ObjectCreated(Event):
    """A new canonical object was minted.

    ``object_id`` is a site-scoped ULID. The registry is the only module that may
    mint one (01_LAYERED section 8).
    """

    event_type: ClassVar[str] = "registry.object_created"
    camera_id: CameraId = CameraId("")
    object_id: str = ""
    track_id: str = ""
    class_id: str = ""


@dataclass(frozen=True, slots=True)
class ObjectLifecycleChanged(Event):
    """An object moved between lifecycle states (02_VOM section 10.6).

    Every transition is observable. ``merged_into`` and ``expired`` appear here
    like any other state — an object is never silently removed, because a
    consumer that saw it must be able to see it end.
    """

    event_type: ClassVar[str] = "registry.lifecycle_changed"
    camera_id: CameraId = CameraId("")
    object_id: str = ""
    previous: str = ""
    current: str = ""
    trigger: str = ""


@dataclass(frozen=True, slots=True)
class IdentityAsserted(Event):
    """A track was claimed to continue an object.

    A **claim**, not a truth (02_VOM section 4.2). ``ambiguous`` with a non-zero
    ``alternatives`` is the case section M7 cares most about: candidates existed
    but none was decisive, so a new object was minted and the alternatives are
    published rather than one being guessed.
    """

    event_type: ClassVar[str] = "registry.identity_asserted"
    camera_id: CameraId = CameraId("")
    object_id: str = ""
    track_id: str = ""
    binding_id: str = ""
    method: str = ""
    confidence: float = 0.0
    ambiguous: bool = False
    alternatives: int = 0


@dataclass(frozen=True, slots=True)
class IdentityRevised(Event):
    """An earlier identity claim was superseded — a merge or a split.

    History is never rewritten (V5). The correction is a new fact carrying
    ``supersedes``; the prior record survives and stays resolvable.
    """

    event_type: ClassVar[str] = "registry.identity_revised"
    camera_id: CameraId = CameraId("")
    object_id: str = ""
    successor_id: str = ""
    """Merge target, or the new object minted by a split."""

    reason: str = ""
    supersedes: str = ""
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class RegionTransition(Event):
    """An object entered or left a region. **Pure geometry.**

    ``dwell_ms`` on exit is computed from ``t_capture``, never processing time
    (V11): a dwell of 45 s means the object was present for 45 s in the world,
    regardless of whether the platform was keeping up.

    Deliberately absent: any judgment about what the dwell *means*. There is no
    `exceeded_threshold`, because the threshold is a business decision (V1).
    """

    event_type: ClassVar[str] = "registry.region_transition"
    camera_id: CameraId = CameraId("")
    object_id: str = ""
    region_id: str = ""
    geometry_version: str = ""
    entered: bool = True
    dwell_ms: float = 0.0
    method: str = ""


@dataclass(frozen=True, slots=True)
class ObjectPopulationCapped(Event):
    """The per-camera object population hit its bound.

    03_MODULES section M7 failure handling requires this be alarmed: *"Cap
    per-camera population; shed `provisional` objects first; **alarm** — a
    runaway registry is a memory leak with a face."*
    """

    event_type: ClassVar[str] = "registry.population_capped"
    camera_id: CameraId = CameraId("")
    population: int = 0
    capacity: int = 0
    shed: int = 0
    detail: str = ""


# --- crop manager (M8, Flow 5) --------------------------------------------- #
#
# ``subscribe()`` names exactly two (03_MODULES section M8). ``CapabilityGap`` is
# the third, required by the same section's failure table: *"publish a capability
# gap so the consumer stops waiting for data that will never arrive."*
#
# Note what is absent. There is no `CropProduced`: a crop is data-plane evidence
# and the Event Bus is a lossy control-plane notifier (01_LAYERED). Announcing
# every crop would put megabyte-scale traffic on a bus designed for kilobytes and
# would make a dropped notification look like a missing crop.


@dataclass(frozen=True, slots=True)
class BudgetExhausted(Event):
    """The understanding budget is spent; requests are being shed by priority.

    Distinct from ``BudgetExceeded``, which is M3's per-node frame budget. Both
    are real and they fail differently: exceeding the frame budget drops frames,
    exhausting this one thins **attributes**, and a consumer told the wrong one
    would tune the wrong knob.

    Always accompanied by a coverage change, so a consumer learns its attributes
    are thinned rather than inferring it from silence (V8).
    """

    event_type: ClassVar[str] = "cropping.budget_exhausted"
    ceiling_per_hour: float = 0.0
    spent_in_window: int = 0
    shed_in_window: int = 0
    pressure: float = 0.0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class GateRejectionSpike(Event):
    """Gate rejections rose sharply for one camera, with the dominant reason.

    Almost always a **physical** problem: a camera nudged, a lens fouled, a light
    failed. Carrying ``reason`` is what turns this from "understanding stopped
    working" into "objects at camera-7 are now 30 px tall", which is actionable
    on the first read.
    """

    event_type: ClassVar[str] = "cropping.gate_rejection_spike"
    camera_id: CameraId = CameraId("")
    reason: str = ""
    rejection_rate: float = 0.0
    sample_size: int = 0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CapabilityGap(Event):
    """A demand that cannot be satisfied at this camera, persistently.

    §M8: *"Detect the persistent pattern, publish a capability gap so the
    consumer stops waiting for data that will never arrive."* Silence would make
    an impossible demand indistinguishable from a slow one — and the consumer
    would wait forever for either.
    """

    event_type: ClassVar[str] = "cropping.capability_gap"
    camera_id: CameraId = CameraId("")
    demand_id: str = ""
    attribute_key: str = ""
    reason: str = ""
    consecutive_failures: int = 0
    detail: str = ""


# --- understanding (M9, Flow 6) --------------------------------------------- #
#
# Note what is absent. There is no `UnderstandingSucceeded` and no event carrying
# an attribute value: attributes reach consumers through observations built by
# M11, not through a lossy control-plane bus. Announcing every result would put
# the platform's highest-volume enrichment traffic on a notifier designed for
# alarms — and a dropped notification would look like a missing attribute.


@dataclass(frozen=True, slots=True)
class UnderstandingFailed(Event):
    """A request could not be answered after retries and fallbacks.

    Published rather than silently counted because *"understanding failure is
    never pipeline failure"* cuts both ways: the pipeline keeps running, so
    nothing else will make the loss visible. ``outcome`` distinguishes a timeout
    from an exhausted chain from an unsupported capability — three different
    operator responses.
    """

    event_type: ClassVar[str] = "understanding.failed"
    camera_id: CameraId = CameraId("")
    outcome: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ModelFallbackEngaged(Event):
    """The primary understander was unavailable and a fallback answered.

    `10_RELIABILITY` §7.2 rule 1: *"A fallback is never silent."* Without this
    event a fallback *"becomes permanent, and the platform quietly runs on its
    worst model forever"* — one of the silent failures §5.1 names.
    """

    event_type: ClassVar[str] = "understanding.fallback_engaged"
    camera_id: CameraId = CameraId("")
    primary_model: str = ""
    fallback_model: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SchemaDriftSuspected(Event):
    """Sustained unregistered-key rejections from a model.

    04_MODULES §M9: *"If the rate is sustained, alarm — this means a prompt has
    drifted beyond its declared schema."* A deploy-time problem surfacing at
    inference time, and the only signal that it is happening at all.

    Deliberately *suspected* rather than *detected*: the platform observes a rate,
    and whether the prompt or the model changed is a question for a human.
    """

    event_type: ClassVar[str] = "understanding.schema_drift_suspected"
    camera_id: CameraId = CameraId("")
    rejection_rate: float = 0.0
    sample_size: int = 0
    detail: str = ""


# --- synthesis and state (M11, M12, Flow 7) --------------------------------- #
#
# Note what is absent. There is no `ObservationPublished`: observations reach
# consumers through the observation log and the state projection, not through a
# lossy control-plane bus. At 500-2000 observations/second (07_STATE §M12
# Performance) announcing each one would put the platform's highest-volume
# traffic on a notifier designed for alarms — and a dropped notification would
# look like a missing fact.


@dataclass(frozen=True, slots=True)
class ObservationRejected(Event):
    """An observation was refused by the final semantic gate.

    04_MODULES §M11: *"an unexplainable observation is worse than no observation
    — it is a fact nobody can audit (V4)."* The refusal is published because a
    silently refused observation is a fact that vanished, and the producer would
    never learn its output is being discarded.
    """

    event_type: ClassVar[str] = "synthesis.observation_rejected"
    camera_id: CameraId = CameraId("")
    observation_type: str = ""
    kind: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SchemaViolationSpike(Event):
    """Sustained ceiling violations at the final gate.

    §M11: *"count, alarm on sustained rate."* One unregistered attribute is a
    producer being creative; a sustained rate means a producer has drifted beyond
    the registered vocabulary — a new prompt, a new model, or a partial
    deployment — which is a deploy-time problem surfacing at publication time.
    """

    event_type: ClassVar[str] = "synthesis.schema_violation_spike"
    camera_id: CameraId = CameraId("")
    violation_rate: float = 0.0
    sample_size: int = 0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PartitionDegraded(Event):
    """A state partition stopped accepting observations.

    10_RELIABILITY §4.4 step 4 halts a partition loudly when its durability
    buffer fills, because *"losing observations invisibly is a V8 violation of
    the worst kind."* This event is the loudness.
    """

    event_type: ClassVar[str] = "state.partition_degraded"
    camera_id: CameraId = CameraId("")
    reason: str = ""
    buffered: int = 0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PartitionRecovered(Event):
    """A degraded partition drained its buffer and resumed.

    Paired with `PartitionDegraded` so a consumer can bound the gap rather than
    inferring its end from the resumption of traffic. §4.4 step 5 also emits a
    coverage observation for the window, so the gap is in the record as data.
    """

    event_type: ClassVar[str] = "state.partition_recovered"
    camera_id: CameraId = CameraId("")
    drained: int = 0
    gap_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ObservationQuarantined(Event):
    """An observation was appended but could not be projected.

    §M12: *"Quarantine that observation, continue the projection, alarm. One bad
    record must not stop the world."* Distinct from a rejection: this observation
    **is** in the log and is therefore still part of the record — only the
    projection could not absorb it, which is a projection bug rather than a
    producer one.
    """

    event_type: ClassVar[str] = "state.observation_quarantined"
    camera_id: CameraId = CameraId("")
    observation_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StateRebuilt(Event):
    """A partition's projection was rebuilt from the log.

    07_STATE §9.2: rebuild runs into a shadow projection and swaps atomically, so
    *"consumers see a version change, never an outage."* Publishing it is what
    lets a consumer holding a snapshot know why its version jumped.
    """

    event_type: ClassVar[str] = "state.rebuilt"
    camera_id: CameraId = CameraId("")
    observations_replayed: int = 0
    from_position: int = 0
    detail: str = ""


ALL_EVENT_TYPES: tuple[type[Event], ...] = (
    StreamConnected,
    StreamLost,
    EpochAdvanced,
    DecodeFailed,
    MaskFailure,
    ClockQualityChanged,
    SustainedDropAlarm,
    BudgetExceeded,
    PoolPressure,
    LeaseLeaked,
    FrameEvicted,
    CameraChanged,
    ViewpointDriftSuspected,
    ConfigChanged,
    PluginLoaded,
    PluginRejected,
    HealthChanged,
    CoverageChanged,
    SilentFailureSuspected,
    PipelineAttached,
    PipelineDetached,
    Gap,
    ModelLoaded,
    ModelEvicted,
    ModelSwapped,
    DevicePressure,
    CanaryPromoted,
    DetectionCompleted,
    DetectionFailed,
    DetectorLoaded,
    DetectorUnloaded,
    ThresholdExceeded,
    PerformanceWarning,
    TrackCreated,
    TrackUpdated,
    TrackRecovered,
    TrackLost,
    TrackTerminated,
    AssociationFailure,
    TrackingWarning,
    TrackerLoaded,
    TrackerUnloaded,
    TrackerEpochAdvanced,
    ObjectCreated,
    ObjectLifecycleChanged,
    IdentityAsserted,
    IdentityRevised,
    RegionTransition,
    ObjectPopulationCapped,
    BudgetExhausted,
    GateRejectionSpike,
    CapabilityGap,
    UnderstandingFailed,
    ModelFallbackEngaged,
    SchemaDriftSuspected,
    ObservationRejected,
    SchemaViolationSpike,
    PartitionDegraded,
    PartitionRecovered,
    ObservationQuarantined,
    StateRebuilt,
)
