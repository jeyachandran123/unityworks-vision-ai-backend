"""M11 Observation Builder — the single choke point for every published fact.

> **Single responsibility:** *Turn internal signals into published facts, and
> refuse anything that is not one.*

`01_LAYERED` §8 explains why *single* is the operative word:

> *Schema & ceiling enforcement — L5 Observation Builder. One choke point through
> which every fact must pass. **Enforcement distributed across producers is
> enforcement that will be bypassed.***

The public API is 04_MODULES §M11's, implemented verbatim::

    build_presence(object, detection, frame)      -> Observation !ValidationFailed
    build_spatial(object, frame)                  -> Observation?    # None if unchanged
    build_attribute(object, understanding_result) -> Observation[] !ValidationFailed
    build_identity(assertion)                     -> Observation
    build_lifecycle(object, transition)           -> Observation
    build_coverage(scope, reason, window)         -> Observation
    validate(observation)                         -> Valid | Violations[]

Six builders rather than one, because no single upstream seam carries every
signal the envelope can hold: M7's update has objects and lifecycle, M9's result
has attributes and model provenance, and 02_VOM §11.2 assigns different content
to different types. A single builder would require every envelope to carry every
producer, which would make most observations unconstructible.

**What this module does not do**, and why each absence is load-bearing:

*It asks no model anything.* No inference, no prompt, no crop. M9 already
answered; M11 decides whether the answer is publishable.

*It owns no observations.* §M11: *"it builds them and hands them to M12. This
separation matters: the builder must be a pure, heavily-testable function of its
inputs, and giving it durable state would compromise that."*

*It interprets nothing.* There is no severity, no alert, no threshold. A
restaurant, a hospital and a factory read the same observation and reach entirely
different conclusions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import TaxonomyMismatchError, ValidationFailedError
from ...core.model.confidence import Confidence
from ...core.model.detection import QualityGrades
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import (
    CameraId,
    DemandId,
    EvidenceId,
    FrameRef,
    ObjectId,
    ObservationId,
    RegionId,
    SiteId,
    TenantId,
    TrackId,
    new_ulid,
)
from ...core.model.observation import (
    CoverageWindow,
    Evidence,
    EvidenceRef,
    IdentityAssertionRef,
    LifecycleTransition,
    MeasurementBasis,
    ObservabilityReason,
    ObservabilityStatus,
    Observation,
    ObservationBatch,
    ObservationType,
    ValidationResult,
    Violation,
)
from ...core.model.space import SpatialInfo
from ...core.model.timebase import ClockQuality, Duration, Instant
from ...core.model.understanding import (
    Timing,
    UnderstandingOutcome,
    UnderstandingResult,
)
from ...core.model.visual_object import Attribute, VisualObject
from ...core.ports.clock import Clock
from ...core.ports.synthesis import SuppressionPolicyPort
from ...kernel.config.schema import SynthesisSection
from ...kernel.events import (
    CoverageChanged,
    EventBus,
    ObservationRejected,
    SchemaViolationSpike,
)
from ...kernel.metrics import MetricName, MetricsEngine
from ..builder.suppression import (
    DEFAULT_HEARTBEAT,
    SuppressionStateStore,
    subject_key,
)
from ..builder.validation import CeilingGate, ceiling_violations

OBSERVATION_BUILDER_ID = "observation_builder"
OBSERVATION_BUILDER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Everything about the moment that every observation type needs.

    Passed rather than fetched: the builder is *"a pure, heavily-testable
    function of its inputs"*, and a builder that reached for a clock or a
    calibration would stop being one.
    """

    camera_id: CameraId
    tenant_id: TenantId
    site_id: SiteId
    frame_ref: FrameRef
    t_capture: Instant
    t_capture_unc: Duration = Duration(0)
    clock_quality: ClockQuality = ClockQuality.UNKNOWN
    taxonomy_version: str = ""
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if self.frame_ref.camera_id != self.camera_id:
            raise ValueError("a build context must name the frame's own camera")


class ObservationBuilder:
    """M11. Assembles, validates and suppresses; owns nothing durable."""

    __slots__ = (
        "_clock",
        "_config",
        "_events",
        "_gate",
        "_metrics",
        "_provenance",
        "_rejected",
        "_suppression",
        "_suppression_policy",
        "_violation_window",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        events: EventBus,
        config: SynthesisSection,
        gate: CeilingGate,
        provenance,
        suppression_policy: SuppressionPolicyPort,
        suppression: SuppressionStateStore | None = None,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._events = events
        self._config = config
        self._gate = gate
        self._provenance = provenance
        self._suppression_policy = suppression_policy
        self._suppression = (
            SuppressionStateStore(capacity_per_camera=config.suppression_capacity)
            if suppression is None
            else suppression
        )
        self._rejected = 0
        self._violation_window: list[bool] = []

    # --- the six builders ------------------------------------------------------ #

    def build_presence(
        self,
        obj: VisualObject,
        context: BuildContext,
        *,
        basis: MeasurementBasis = MeasurementBasis.MEASURED,
        quality: QualityGrades | None = None,
        track_id: TrackId | None = None,
    ) -> Observation | None:
        """An object was detected in a frame.

        Returns ``None`` when suppression says nothing changed — which is a
        *success*, not a failure. §M11 puts the typical reduction at 10-50x.

        Raises:
            ValidationFailedError: the envelope is incomplete or the taxonomy
                disagrees. §M11: *"an unexplainable observation is worse than no
                observation."*
        """
        candidate = self._envelope(
            ObservationType.PRESENCE,
            obj,
            context,
            confidence=obj.confidence,
            spatial=obj.current_spatial,
            basis=basis,
            quality=quality,
            track_id=track_id,
        )
        return self._finish(candidate, context)

    def build_spatial(
        self,
        obj: VisualObject,
        context: BuildContext,
        *,
        basis: MeasurementBasis = MeasurementBasis.MEASURED,
        track_id: TrackId | None = None,
    ) -> Observation | None:
        """Position, motion or region membership changed materially.

        The type §M11 documents as returning ``Observation?`` — null if unchanged
        — because a stationary object's position is the canonical case where
        publishing every frame carries no information.
        """
        candidate = self._envelope(
            ObservationType.SPATIAL,
            obj,
            context,
            confidence=obj.confidence,
            spatial=obj.current_spatial,
            basis=basis,
            track_id=track_id,
        )
        return self._finish(candidate, context)

    def build_attribute(
        self,
        obj: VisualObject,
        result: UnderstandingResult,
        context: BuildContext,
        *,
        track_id: TrackId | None = None,
        confidence: Confidence | None = None,
    ) -> list[Observation]:
        """Attributes were computed or revised.

        Returns a **list**, matching §M11's signature. Today one observation
        carries all of a result's attributes — they share a model, a prompt and a
        moment, so splitting them would multiply evidence records for no gain.
        The list shape is what lets a future policy split by privacy class or
        retention without a contract change.

        ``confidence`` is the *envelope's* confidence, and it is optional because
        an attribute observation's confidence properly lives on each attribute,
        where M9 measured it. A caller that does not know the subject's identity
        confidence — the understanding seam, which reconstructs its subject from
        the result — passes ``None`` rather than inventing a number.

        A failed understanding produces **no** observation: `NO_ATTRIBUTES` is an
        understanding outcome, not a published fact, and publishing an empty
        attribute observation would assert that the platform looked and found
        nothing when it may simply have failed.
        """
        if result.outcome is not UnderstandingOutcome.SUCCEEDED:
            self._metrics.counter(
                MetricName.OBSERVATIONS_SKIPPED,
                camera_id=str(context.camera_id),
                reason=result.outcome.value,
            ).increment()
            return []

        observation_id = ObservationId(self._new_id())
        evidence = self._complete_evidence(result, observation_id)

        candidate = self._envelope(
            ObservationType.ATTRIBUTE,
            obj,
            context,
            observation_id=observation_id,
            confidence=confidence,
            attributes=result.attributes,
            evidence_ref=EvidenceRef(
                evidence_id=evidence.evidence_id,
                status="stored",
                crop_ref=evidence.crop_ref,
                raw_output_ref=evidence.raw_output_ref,
                retention=evidence.retention,
            ),
            timing=result.evidence.timing,
            provenance=result.provenance,
            track_id=track_id,
            demand_ids=tuple(DemandId(d) for d in result.demand_ids),
        )
        published = self._finish(candidate, context)
        return [published] if published is not None else []

    def build_identity(
        self,
        obj: VisualObject,
        assertion: IdentityAssertionRef,
        context: BuildContext,
        *,
        confidence: Confidence | None = None,
    ) -> Observation | None:
        """An identity assertion was made or revised.

        A **claim**, not a truth (02_VOM §4.2). An ambiguous assertion publishes
        with its alternative count rather than resolving to a guess.
        """
        candidate = self._envelope(
            ObservationType.IDENTITY,
            obj,
            context,
            confidence=confidence or obj.confidence,
            identity=assertion,
        )
        return self._finish(candidate, context)

    def build_lifecycle(
        self,
        obj: VisualObject,
        transition: LifecycleTransition,
        context: BuildContext,
    ) -> Observation | None:
        """An object changed lifecycle state.

        Never suppressed by content: a transition is by definition a change, and
        the suppression policy sees a signature that includes both states.
        """
        candidate = self._envelope(
            ObservationType.LIFECYCLE,
            obj,
            context,
            confidence=obj.confidence,
            lifecycle_transition=transition,
        )
        return self._finish(candidate, context)

    def build_quality(
        self,
        obj: VisualObject,
        quality: QualityGrades,
        context: BuildContext,
    ) -> Observation | None:
        """Input quality changed materially."""
        candidate = self._envelope(
            ObservationType.QUALITY,
            obj,
            context,
            confidence=obj.confidence,
            quality=quality,
        )
        return self._finish(candidate, context)

    def build_coverage(
        self,
        context: BuildContext,
        *,
        status: ObservabilityStatus,
        reason: ObservabilityReason,
        since: Instant,
        effective_rate: float = 1.0,
        regions_affected: Sequence[RegionId] = (),
        capability_gaps: Sequence[tuple[str, str]] = (),
        until: Instant | None = None,
    ) -> Observation:
        """Observability changed. **Never suppressed, never optional.**

        02_VOM §11.2: *"This type is the difference between a platform that is
        honest about its limits and one that is dangerously silent — and it is
        not optional."* A suppressed coverage observation would be the platform
        deciding its own blindness was not worth mentioning.
        """
        window = CoverageWindow(
            status=status,
            reason=reason,
            since=since,
            effective_rate=effective_rate,
            regions_affected=tuple(str(r) for r in regions_affected),
            capability_gaps=tuple(capability_gaps),
            until=until,
        )
        candidate = Observation(
            observation_id=ObservationId(self._new_id()),
            observation_type=ObservationType.COVERAGE,
            tenant_id=context.tenant_id,
            site_id=context.site_id,
            camera_id=context.camera_id,
            frame_ref=context.frame_ref,
            t_capture=context.t_capture,
            t_capture_unc=self._uncertainty(context),
            clock_quality=context.clock_quality,
            t_published=self._clock.now(),
            provenance=self._provenance,
            timing=Timing(total_ms=0.01),
            coverage=window,
            taxonomy_version=context.taxonomy_version,
        )
        self._count(candidate)
        self._events.publish(
            CoverageChanged(
                occurred_at=self._clock.now(),
                partition_key=str(context.camera_id),
                camera_id=context.camera_id,
                status=status.value,
                reason=reason.value,
                effective_rate=effective_rate,
            )
        )
        return candidate

    # --- the gate -------------------------------------------------------------- #

    def validate(self, candidate: Observation) -> ValidationResult:
        """§M11's ``validate``. Pure — no state, no counters, no events.

        Exposed so a caller can check without publishing, which is what a
        conformance kit and an operator's dry run both need.
        """
        return self._gate.validate(candidate)

    # --- assembly -------------------------------------------------------------- #

    def _envelope(
        self,
        kind: ObservationType,
        obj: VisualObject,
        context: BuildContext,
        *,
        observation_id: ObservationId | None = None,
        confidence: Confidence | None = None,
        spatial: SpatialInfo | None = None,
        attributes: Sequence[Attribute] = (),
        basis: MeasurementBasis = MeasurementBasis.MEASURED,
        quality: QualityGrades | None = None,
        evidence_ref: EvidenceRef | None = None,
        timing: Timing | None = None,
        provenance=None,
        track_id: TrackId | None = None,
        identity: IdentityAssertionRef | None = None,
        lifecycle_transition: LifecycleTransition | None = None,
        demand_ids: Sequence[DemandId] = (),
    ) -> Observation:
        """Assemble one envelope from the signals available here.

        ``taxonomy_version`` comes from the *context*, not from the object: a
        version stamped by the producer would report what the producer believed,
        and the whole point of the mismatch check is to catch a producer that
        believes something different from the site.
        """
        return Observation(
            observation_id=observation_id or ObservationId(self._new_id()),
            observation_type=kind,
            tenant_id=obj.tenant_id,
            site_id=obj.site_id,
            camera_id=context.camera_id,
            frame_ref=context.frame_ref,
            t_capture=context.t_capture,
            t_capture_unc=self._uncertainty(context),
            clock_quality=context.clock_quality,
            t_published=self._clock.now(),
            provenance=provenance or self._provenance,
            timing=timing or Timing(total_ms=0.01),
            object_id=obj.object_id,
            track_id=track_id,
            class_id=obj.class_id,
            taxonomy_version=context.taxonomy_version,
            lifecycle_state=obj.lifecycle,
            confidence=confidence,
            spatial=spatial,
            attributes=tuple(attributes),
            measurement_basis=basis,
            quality=quality,
            evidence_ref=evidence_ref,
            identity=identity,
            lifecycle_transition=lifecycle_transition,
            demand_ids=tuple(demand_ids),
        )

    def _finish(
        self, candidate: Observation, context: BuildContext
    ) -> Observation | None:
        """Validate, then suppress. **In that order.**

        Validation first because a rejected observation must not update the
        suppression signature: if it did, the *next* valid observation of the
        same content would be suppressed against a fact that was never published,
        and the subject would go silent for a reason nobody could find.

        **Only an envelope failure raises.** §M11's failure table prescribes two
        opposite responses, and the distinction survives all the way to the
        caller: an incomplete envelope is a constitutional failure that must be
        loud, while an attribute the registry refuses is the ceiling working
        exactly as designed. An attribute observation whose every attribute was
        dropped has nothing left to say, so it returns ``None`` — the same signal
        as suppression, because in both cases the correct outcome is that no
        fact is published and nothing is wrong. Raising there would tell a
        caller the platform is broken when it has just successfully refused to
        publish something it was never allowed to publish.
        """
        result = self._gate.validate(candidate)
        self._record_violations(candidate, result)

        if result.envelope_violations:
            self._rejected += 1
            self._publish_rejection(candidate, result)
            raise ValidationFailedError(
                f"observation for camera '{candidate.camera_id}' was refused: "
                + "; ".join(v.detail for v in result.envelope_violations[:3]),
                camera_id=str(candidate.camera_id),
                observation_type=candidate.observation_type.value,
                violations=tuple(v.kind.value for v in result.violations),
            )

        published = result.observation
        if published is None:
            # Every attribute was dropped by the ceiling. Not an error — the gate
            # did its job, and there is simply no fact left to record.
            self._metrics.counter(
                MetricName.OBSERVATIONS_SKIPPED,
                camera_id=str(candidate.camera_id),
                reason="all_attributes_dropped",
            ).increment()
            return None

        if not self._admit(published, context):
            return None

        self._count(published)
        return published

    def _admit(self, observation: Observation, context: BuildContext) -> bool:
        """Ask the suppression policy whether this says anything new."""
        state = self._suppression.partition(context.camera_id)
        key = subject_key(observation)
        previous = state.last(key)

        elapsed = (
            Duration(max(0, observation.t_capture.ns - previous.published_at.ns))
            if previous
            else Duration(0)
        )
        heartbeat = Duration.from_millis(self._config.heartbeat_ms) or DEFAULT_HEARTBEAT

        decision = self._suppression_policy.should_publish(
            observation,
            previous.signature if previous else None,
            elapsed=elapsed,
            heartbeat=heartbeat,
        )
        if not decision.publish:
            state.suppressed += 1
            self._metrics.counter(
                MetricName.OBSERVATIONS_SUPPRESSED,
                camera_id=str(context.camera_id),
                observation_type=observation.observation_type.value,
            ).increment()
            return False

        if previous is not None and "heartbeat" in decision.reason:
            state.heartbeats += 1
            self._metrics.counter(
                MetricName.OBSERVATION_HEARTBEATS, camera_id=str(context.camera_id)
            ).increment()

        state.record(key, self._suppression_policy.signature(observation), observation)
        return True

    # --- evidence -------------------------------------------------------------- #

    def _complete_evidence(
        self, result: UnderstandingResult, observation_id: ObservationId
    ) -> Evidence:
        """Stamp ``observation_id`` onto M9's evidence.

        The promised half of the two-part construction Flow 6 documented: M9
        produced everything except this field, because it may not mint an
        identifier for an object it is forbidden to create. 02_VOM §10.9's
        ``Evidence`` is complete only here.
        """
        source = result.evidence
        return Evidence(
            evidence_id=EvidenceId(source.evidence_id),
            observation_id=observation_id,
            trigger_reason=source.trigger_reason,
            input_hash=source.input_hash,
            frame_ref=source.frame_ref,
            crop_ref=source.crop_ref,
            raw_output_ref=source.raw_output_ref,
            unstructured_note=source.unstructured_note,
            decision_path=source.decision_path,
            timing=source.timing,
            retention=source.retention,
        )

    def _uncertainty(self, context: BuildContext) -> Duration:
        """Capture uncertainty, floored by clock quality.

        §M11's failure table: *"Clock quality UNKNOWN — emit with maximal
        `t_capture_unc`; consumers decide whether it is usable (V11)."* A zero
        uncertainty on an unsynced clock would be a precision the platform does
        not have.
        """
        floor = Duration.from_millis(context.clock_quality.typical_uncertainty_ms)
        return context.t_capture_unc if context.t_capture_unc.ns > floor.ns else floor

    # --- observability --------------------------------------------------------- #

    def _count(self, observation: Observation) -> None:
        self._metrics.counter(
            MetricName.OBSERVATIONS_BUILT,
            camera_id=str(observation.camera_id),
            observation_type=observation.observation_type.value,
        ).increment()
        self._metrics.counter(
            MetricName.ATTRIBUTES_PUBLISHED, camera_id=str(observation.camera_id)
        ).increment(len(observation.attributes))

    def _record_violations(
        self, candidate: Observation, result: ValidationResult
    ) -> None:
        camera = str(candidate.camera_id)
        for violation in result.violations:
            self._metrics.counter(
                MetricName.OBSERVATION_VIOLATIONS,
                camera_id=camera,
                kind=violation.kind.value,
            ).increment()
        for key in result.dropped_attributes:
            self._metrics.counter(
                MetricName.ATTRIBUTES_DROPPED, camera_id=camera
            ).increment()
            assert key is not None

        self._note_violation_rate(candidate, result)

    def _note_violation_rate(
        self, candidate: Observation, result: ValidationResult
    ) -> None:
        """Alarm on a sustained ceiling-violation rate.

        §M11: *"count, alarm on sustained rate."* One unregistered attribute is a
        producer being creative; a sustained rate means a producer has drifted —
        a new prompt, a new model, a partial deployment — and that is a
        deploy-time problem surfacing at publication time.
        """
        window = self._config.rejection_window
        self._violation_window.append(bool(ceiling_violations(result.violations)))
        if len(self._violation_window) > window:
            del self._violation_window[:-window]
        if len(self._violation_window) < window:
            return
        rate = sum(self._violation_window) / len(self._violation_window)
        if rate < self._config.rejection_alarm_rate:
            return
        self._violation_window.clear()
        self._events.publish(
            SchemaViolationSpike(
                occurred_at=self._clock.now(),
                partition_key=str(candidate.camera_id),
                camera_id=candidate.camera_id,
                violation_rate=rate,
                sample_size=window,
                detail=(
                    "sustained unregistered-attribute rejections at the final "
                    "gate; a producer has drifted beyond the registered "
                    "vocabulary"
                ),
            )
        )
        self._metrics.counter(MetricName.SCHEMA_VIOLATION_ALARMS).increment()

    def _publish_rejection(
        self, candidate: Observation, result: ValidationResult
    ) -> None:
        first = result.violations[0] if result.violations else None
        self._events.publish(
            ObservationRejected(
                occurred_at=self._clock.now(),
                partition_key=str(candidate.camera_id),
                camera_id=candidate.camera_id,
                observation_type=candidate.observation_type.value,
                kind=first.kind.value if first else "unknown",
                detail=first.detail if first else "envelope refused",
            )
        )
        self._metrics.counter(
            MetricName.OBSERVATIONS_REJECTED,
            camera_id=str(candidate.camera_id),
            observation_type=candidate.observation_type.value,
        ).increment()

    def _new_id(self) -> str:
        """A ULID, time-sortable, minted from platform time.

        From the clock rather than the wall so a deterministic run reproduces the
        same ordering (V13); 02_VOM §11 requires the id itself be time-sortable
        so a lexicographic log scan is a chronological one.
        """
        return new_ulid(now_ms=self._clock.now().ns // 1_000_000)

    # --- access ---------------------------------------------------------------- #

    def health(self) -> ComponentHealth:
        state = HealthState.HEALTHY
        detail = "synthesis nominal"
        if self._rejected:
            state = HealthState.DEGRADED
            detail = (
                f"{self._rejected} observation(s) refused by the final gate; an "
                f"unexplainable observation is worse than no observation"
            )
        return ComponentHealth(
            component_id=OBSERVATION_BUILDER_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "rejected": float(self._rejected),
                "suppressed": float(self._suppression.suppressed),
                "tracked_subjects": float(self._suppression.tracked_subjects),
            },
        )

    @property
    def suppression(self) -> SuppressionStateStore:
        return self._suppression

    @property
    def gate(self) -> CeilingGate:
        return self._gate

    @property
    def rejected(self) -> int:
        return self._rejected

    def forget_camera(self, camera_id: CameraId) -> None:
        """Release a camera's suppression state after it detaches."""
        self._suppression.drop(camera_id)

    def forget_object(self, camera_id: CameraId, object_id: ObjectId) -> int:
        """Drop a departed object's signatures."""
        return self._suppression.partition(camera_id).forget(object_id)


def batch(
    observations: Sequence[Observation | None],
    *,
    violations: Sequence[Violation] = (),
    suppressed: int = 0,
) -> ObservationBatch:
    """Collect a build pass, dropping the suppressed ``None`` entries."""
    published = tuple(o for o in observations if o is not None)
    return ObservationBatch(
        observations=published,
        violations=tuple(violations),
        suppressed=suppressed + sum(1 for o in observations if o is None),
    )


def unknown_taxonomy(version: str, expected: str) -> TaxonomyMismatchError:
    return TaxonomyMismatchError(
        f"producer declares taxonomy {version} but this site runs {expected}; "
        f"this indicates a partial deployment and must be loud",
        declared=version,
        expected=expected,
    )
