"""M8 Crop Manager — the platform's attention mechanism.

> **Single responsibility:** *Choose what to look at closely, and produce a crop
> worth looking at.*

This module is why a 100-camera deployment is affordable. Without it,
understanding cost is `cameras x fps x objects`. With it, cost is
`demands x changes` — smaller by two to three orders of magnitude. §M8 is blunt
about what that means: *"That reduction is not an optimization; it is the
architecture."*

The public API is 03_MODULES section M8's, implemented verbatim::

    evaluate(object_ids, frame_ref)  -> CropRequest[] | Skipped[]
    extract(crop_request)            -> Crop !GateRejected !FrameUnavailable
    register_demand(demand)          -> DemandId
    revoke_demand(demand_id)         -> void
    budget_status()                  -> BudgetStatus
    subscribe()                      => BudgetExhausted | GateRejectionSpike

``evaluate`` **never raises**: an attention failure may not stop the registry,
which may not stop tracking, which may not stop acquisition (V9). ``extract``
*does* raise, because its two failure modes — a gate rejection and an evicted
frame — are answers the caller must handle explicitly, and swallowing them would
turn "we refused" into "there was nothing there".

**What this module does not do**, and why each absence is load-bearing:

*It runs no model.* Not a detector, not a classifier, not a VLM. M8 prepares
evidence and hands it upward; the L3/L4 boundary is what keeps attention cheap
and understanding replaceable (01_LAYERED section 1.2).

*It reads no meaning.* A trigger fires on a measured appearance delta, never on
what a region is for. Priority is an opaque string the platform orders by and
never interprets (V1/V2).

*It creates no objects and no attributes.* M7 is the only writer of Vision
Objects; M9 is the only producer of attributes. M8 reads the first and feeds the
second.

*It stores nothing durably.* Trigger state is ephemeral and node-local by
§M8's own State Ownership section — a restart costs one round of
``FIRST_SIGHT``, which is bounded and conservative.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...core.errors import (
    CropExtractionError,
    FrameUnavailableError,
    GateRejectedError,
)
from ...core.model.confidence import Confidence
from ...core.model.crop import (
    Crop,
    CropRequest,
    EvaluationResult,
    GateResult,
    PrivacyClass,
    RetentionMode,
    Skipped,
    SkipReason,
    TriggerReason,
)
from ...core.model.demand import Demand, DemandAcknowledgement
from ...core.model.detection import QualityGrades, QualityLevel
from ...core.model.health import ComponentHealth, HealthState
from ...core.model.ids import (
    AttributeKey,
    CameraId,
    ClassId,
    CropId,
    DemandId,
    FrameRef,
    ObjectId,
)
from ...core.model.provenance import Provenance
from ...core.model.space import Box
from ...core.model.timebase import Duration, Instant
from ...core.model.visual_object import VisualObject
from ...core.ports.clock import Clock
from ...core.ports.cropping import (
    AttributeStatus,
    LabelSpaceView,
    QualityRequest,
    TriggerCandidate,
    TriggerDecision,
)
from ...kernel.config.schema import CroppingSection
from ...kernel.events import (
    BudgetExhausted,
    CapabilityGap,
    EventBus,
    GateRejectionSpike,
)
from ...kernel.metrics import MetricName, MetricsEngine
from .budget import BudgetStatus, CropDeduplicationCache, PriorityQueue, UnderstandingBudget
from .demands import DemandRegistry
from .gate import QualityGate
from .state import GateRejectionWindow, TriggerStateStore

CROP_MANAGER_ID = "crop_manager"
CROP_MANAGER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Everything about a frame M8 needs, without holding the frame.

    Passed rather than fetched, so ``evaluate`` never touches the Frame Buffer:
    trigger evaluation is a control-plane decision about metadata, and only
    ``extract`` takes a lease. That separation is what lets a node evaluate
    thousands of candidates a second while leasing pixels for a handful.
    """

    frame_ref: FrameRef
    width: int
    height: int
    t_capture: Instant
    colour_space: str = "bgr24"


class CropManager:
    """M8. Decides what deserves analysis and prepares defensible evidence."""

    __slots__ = (
        "_budget",
        "_cache",
        "_clock",
        "_config",
        "_demands",
        "_estimator",
        "_events",
        "_extractor",
        "_failures",
        "_frames_evaluated",
        "_evidence_regions",
        "_gate",
        "_gate_windows",
        "_label_space",
        "_metrics",
        "_policy",
        "_priority",
        "_privacy_class",
        "_provenance",
        "_retention",
        "_retention_ttl",
        "_state",
        "_strategy",
    )

    def __init__(
        self,
        *,
        clock: Clock,
        metrics: MetricsEngine,
        events: EventBus,
        config: CroppingSection,
        policy,
        estimator,
        strategy,
        extractor,
        provenance: Provenance,
        demands: DemandRegistry | None = None,
        budget: UnderstandingBudget | None = None,
        gate: QualityGate | None = None,
        label_space: LabelSpaceView | None = None,
        evidence_regions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._clock = clock
        self._metrics = metrics
        self._events = events
        self._config = config
        self._policy = policy
        self._estimator = estimator
        self._strategy = strategy
        self._extractor = extractor
        self._provenance = provenance

        # ``is None`` rather than ``or``: an empty ``DemandRegistry`` defines
        # ``__len__`` and is therefore falsy, so ``demands or DemandRegistry()``
        # would silently discard an injected registry until its first demand
        # arrived — and the caller's capability view with it.
        self._demands = DemandRegistry() if demands is None else demands
        self._budget = (
            UnderstandingBudget(
                ceiling_per_hour=config.understanding_calls_per_hour,
                window=Duration.from_millis(config.budget_window_ms),
                now=clock.monotonic(),
            )
            if budget is None
            else budget
        )
        self._gate = QualityGate() if gate is None else gate

        # Where each attribute is visible on a subject, as declared by the
        # policy that asked for it. Used only to *group* attributes into
        # evidence requests; the geometry itself is the crop strategy's to
        # plan. Empty means one group, which is the previous behaviour.
        self._evidence_regions = dict(evidence_regions or {})

        # Undeclared by default. M8 never asks a detector what it can name — the
        # composition root, which chose the detector, states it. An empty view
        # reports "cannot tell" for every class, so a deployment that has not
        # wired it behaves exactly as before rather than treating every object as
        # outside the detector's capability.
        self._label_space = LabelSpaceView() if label_space is None else label_space

        self._cache = CropDeduplicationCache(capacity=config.dedup_cache_size)
        self._priority = PriorityQueue(config.priority_classes)
        self._state = TriggerStateStore()
        self._gate_windows: dict[CameraId, GateRejectionWindow] = {}

        self._retention = RetentionMode(config.retention_mode)
        self._retention_ttl = (
            Duration.from_millis(config.evidence_ttl_ms)
            if self._retention is RetentionMode.EVIDENCE
            else None
        )
        self._privacy_class = PrivacyClass.C1_IMAGERY
        """Every crop is C1 Imagery (12_SECURITY section 3). Fixed, not inferred:
        inferring a data classification is how imagery ends up in an
        unclassified path."""

        self._frames_evaluated = 0
        self._failures = 0

    # --- public API: evaluate ---------------------------------------------------- #

    def evaluate(
        self,
        objects: Sequence[VisualObject],
        frame: FrameContext,
        *,
        regions_of: Callable[[ObjectId], frozenset] | None = None,
        appearance_of: Callable[[VisualObject], float | None] | None = None,
    ) -> EvaluationResult:
        """Decide which objects deserve analysis on this frame. **Never raises.**

        **Every candidate appears exactly once** in the result, in ``requests`` or
        in ``skipped``. That is responsibility 7 and invariant V8 made structural:
        a consumer must be able to tell *"no attribute because nothing was
        there"* from *"no attribute because we could not afford to look"*.
        """
        camera_id = frame.frame_ref.camera_id
        started = self._clock.monotonic().ns
        self._frames_evaluated += 1

        try:
            result = self._evaluate(objects, frame, regions_of, appearance_of)
        except Exception as exc:  # noqa: BLE001 - attention must never stop perception
            self._failures += 1
            self._metrics.counter(
                MetricName.CROP_EXTRACTION_FAILURES,
                camera_id=str(camera_id),
                reason="evaluate_guard",
            ).increment()
            # Degrade honestly: every candidate is skipped with an attributed
            # reason rather than vanishing. A crashed evaluator that returned an
            # empty result would be indistinguishable from an idle scene.
            return EvaluationResult(
                camera_id=camera_id,
                frame_ref=frame.frame_ref,
                skipped=tuple(
                    Skipped(
                        object_id=obj.object_id,
                        camera_id=camera_id,
                        reason=SkipReason.QUALITY_INSUFFICIENT,
                        detail=f"evaluation failed: {type(exc).__name__}",
                    )
                    for obj in objects
                ),
            )

        elapsed_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._metrics.histogram(
            MetricName.CROP_TRIGGER_LATENCY_MS, camera_id=str(camera_id)
        ).record(elapsed_ms)
        self._metrics.counter(
            MetricName.CROP_CANDIDATES_EVALUATED, camera_id=str(camera_id)
        ).increment(result.candidate_count)
        return result

    def _evaluate(
        self,
        objects: Sequence[VisualObject],
        frame: FrameContext,
        regions_of: Callable[[ObjectId], frozenset] | None,
        appearance_of: Callable[[VisualObject], float | None] | None,
    ) -> EvaluationResult:
        camera_id = frame.frame_ref.camera_id
        now = self._clock.now()
        partition = self._state.partition(camera_id)
        partition.frames_evaluated += 1

        candidates: list[TriggerCandidate] = []
        by_id: dict[ObjectId, VisualObject] = {}
        for obj in objects:
            by_id[obj.object_id] = obj
            candidates.append(
                self._candidate(obj, frame, regions_of, appearance_of, now)
            )
        partition.candidates_evaluated += len(candidates)

        decisions = self._policy.evaluate(
            candidates,
            now=now,
            demands=[self._demands.required_attributes],
        )
        if len(decisions) != len(candidates):
            # Obligation G1 violated by the bound policy. Refusing here rather
            # than dropping the difference keeps V8 true even when an adapter
            # misbehaves.
            raise CropExtractionError(
                f"trigger policy '{getattr(self._policy, 'policy_id', '?')}' "
                f"returned {len(decisions)} decisions for {len(candidates)} "
                f"candidates; every candidate must produce exactly one decision"
            )

        firing: list[tuple[TriggerDecision, VisualObject]] = []
        skipped: list[Skipped] = []
        for decision in decisions:
            obj = by_id[decision.object_id]
            if decision.fires:
                firing.append((decision, obj))
            else:
                skipped.append(self._skip(decision, camera_id))
                self._state.partition(camera_id).state_for(
                    decision.object_id, now=now
                ).skips += 1

        requests, budget_skips = self._admit(firing, frame, now)
        skipped.extend(budget_skips)

        for skip in skipped:
            self._metrics.counter(
                MetricName.CROPS_SKIPPED,
                camera_id=str(camera_id),
                reason=skip.reason.value,
            ).increment()
        for request in requests:
            self._metrics.counter(
                MetricName.CROPS_REQUESTED,
                camera_id=str(camera_id),
                reason=request.trigger_reason.value,
            ).increment()

        self._count_verification(camera_id, decisions)

        return EvaluationResult(
            camera_id=camera_id,
            frame_ref=frame.frame_ref,
            requests=tuple(requests),
            skipped=tuple(skipped),
        )

    def _admit(
        self,
        firing: list[tuple[TriggerDecision, VisualObject]],
        frame: FrameContext,
        now: Instant,
    ) -> tuple[list[CropRequest], list[Skipped]]:
        """Order by priority, then spend budget until it runs out.

        Ordering happens **before** spending, which is what makes shedding
        meaningful: shedding in arrival order would drop a high-priority request
        because a low-priority one happened to arrive first.
        """
        camera_id = frame.frame_ref.camera_id
        candidates = [
            _AdmissionCandidate(decision=decision, obj=obj) for decision, obj in firing
        ]
        ordered = self._priority.order(candidates)

        requests: list[CropRequest] = []
        skipped: list[Skipped] = []
        exhausted = False

        for item in ordered:
            decision, obj = item.decision, item.obj

            # One request per **evidence group**, not per object.
            #
            # Attributes about different parts of a subject need different
            # pixels. Answering them from one crop means unioning their regions,
            # and a union of "the head" and "the hands" is most of a body — which
            # is the whole-person crop the split exists to escape. Grouping keeps
            # each question with evidence that actually contains its answer.
            #
            # Attributes sharing a region stay in one group, so this adds crops
            # only where the geometry genuinely differs. With no regions declared
            # every attribute lands in one group and the behaviour is unchanged.
            for group in self._evidence_groups(decision.attributes):
                if len(requests) >= self._config.max_candidates_per_frame:
                    skipped.append(
                        Skipped(
                            object_id=obj.object_id,
                            camera_id=camera_id,
                            reason=SkipReason.PRIORITY_PREEMPTED,
                            detail=(
                                f"per-frame candidate ceiling "
                                f"({self._config.max_candidates_per_frame}) reached"
                            ),
                            attribute_keys=tuple(str(k) for k in group),
                        )
                    )
                    continue

                # Each group is its own model call and is charged as one. A
                # subject whose head and hands are asked about separately costs
                # two units, and when the budget runs out mid-subject the
                # remaining groups are skipped with an attributed reason rather
                # than quietly answered from the wrong pixels.
                if not self._budget.try_spend(
                    self._clock.monotonic(), demand_ids=decision.demand_ids
                ):
                    exhausted = True
                    skipped.append(
                        Skipped(
                            object_id=obj.object_id,
                            camera_id=camera_id,
                            reason=SkipReason.BUDGET_EXHAUSTED,
                            detail="understanding budget spent for this window",
                            attribute_keys=tuple(str(k) for k in group),
                        )
                    )
                    continue

                requests.append(
                    CropRequest(
                        object_id=obj.object_id,
                        camera_id=camera_id,
                        frame_ref=frame.frame_ref,
                        source_box=self._box_of(obj),
                        trigger_reason=decision.reason,
                        tenant_id=obj.tenant_id,
                        site_id=obj.site_id,
                        class_id=obj.class_id,
                        required_attributes=tuple(str(k) for k in group),
                        priority_class=decision.priority_class,
                        demand_ids=decision.demand_ids,
                    )
                )
                for demand_id in decision.demand_ids:
                    self._demands.record_served(DemandId(demand_id), now)

        if exhausted:
            self._publish_budget_exhausted()
        return requests, skipped

    # --- public API: extract ------------------------------------------------------ #

    def extract(
        self,
        request: CropRequest,
        *,
        pixels: memoryview,
        frame: FrameContext,
        channels: int = 3,
        neighbour_boxes: Sequence[Box] = (),
    ) -> Crop:
        """Turn a request into canonical evidence.

        Raises:
            GateRejectedError: the input cannot support a defensible claim. Not a
                failure — the gate working. Budget is refunded, because a
                rejected crop bought nothing.
            FrameUnavailableError: the frame was evicted before the crop could be
                taken. Diagnoses pin TTL or buffer depth (§M8 failure table).
            CropExtractionError: a genuine fault in the extraction path.
        """
        camera_id = request.camera_id
        if request.frame_ref != frame.frame_ref:
            raise FrameUnavailableError(
                f"crop request names frame {request.frame_ref} but was handed "
                f"{frame.frame_ref}; evidence must be traceable to its own frame",
                frame_ref=str(request.frame_ref),
            )

        started = self._clock.monotonic().ns
        plan = self._strategy.plan(
            box=request.source_box,
            class_id=request.class_id,
            source_width=frame.width,
            source_height=frame.height,
            attributes=tuple(AttributeKey(k) for k in request.required_attributes),
        )

        # Cheap pre-check on geometry alone, before paying for pixels.
        pre_grades = self._estimator.estimate(
            QualityRequest(
                camera_id=camera_id,
                box=request.source_box,
                source_width=frame.width,
                source_height=frame.height,
                neighbour_boxes=neighbour_boxes,
            )
        )
        pre_result = self._gate.evaluate(pre_grades, request.required_attributes)
        if not pre_result.passed:
            self._reject(request, pre_grades, pre_result)

        try:
            crop_bytes, transform = self._extractor.extract(
                pixels,
                plan=plan,
                source_width=frame.width,
                source_height=frame.height,
                channels=channels,
                colour_space=frame.colour_space,
            )
        except CropExtractionError:
            self._metrics.counter(
                MetricName.CROP_EXTRACTION_FAILURES,
                camera_id=str(camera_id),
                reason="extractor",
            ).increment()
            self._budget.refund(demand_ids=request.demand_ids)
            raise

        view = memoryview(crop_bytes)
        grades = self._estimator.estimate(
            QualityRequest(
                camera_id=camera_id,
                box=request.source_box,
                source_width=frame.width,
                source_height=frame.height,
                neighbour_boxes=neighbour_boxes,
                pixels=view,
                crop_width=transform.output_width,
                crop_height=transform.output_height,
            )
        )
        gate_result = self._gate.evaluate(grades, request.required_attributes)
        self._record_gate(camera_id, gate_result)
        if not gate_result.passed:
            self._reject(request, grades, gate_result)

        crop_id = self.content_hash(crop_bytes)
        cached = self._cache.get(request.tenant_id, crop_id)
        if cached is not None:
            self._metrics.counter(
                MetricName.CROP_CACHE_HITS, camera_id=str(camera_id)
            ).increment()
        self._cache.put(request.tenant_id, crop_id, CropId(crop_id))

        crop = Crop(
            crop_id=CropId(crop_id),
            tenant_id=request.tenant_id,
            site_id=request.site_id,
            camera_id=camera_id,
            source_frame=request.frame_ref,
            object_id=request.object_id,
            source_box=request.source_box,
            padding_applied=plan.padding_applied,
            output_size=(transform.output_width, transform.output_height),
            transform=transform,
            quality=grades,
            gate_result=gate_result,
            retention=self._retention,
            privacy_class=self._privacy_class,
            t_capture=frame.t_capture,
            trigger_reason=request.trigger_reason,
            provenance=self._provenance,
            pixels=view,
            retention_ttl=self._retention_ttl,
        )

        state = self._state.partition(camera_id).state_for(
            request.object_id, now=self._clock.now()
        )
        state.note_analysis(
            self._clock.now(),
            tuple(AttributeKey(k) for k in request.required_attributes),
            CropId(crop_id),
        )

        elapsed_ms = (self._clock.monotonic().ns - started) / 1_000_000
        self._metrics.histogram(
            MetricName.CROP_EXTRACTION_MS, camera_id=str(camera_id)
        ).record(elapsed_ms)
        self._metrics.counter(
            MetricName.CROPS_PRODUCED,
            camera_id=str(camera_id),
            reason=request.trigger_reason.value,
        ).increment()
        if grades.scale_pixels is not None:
            self._metrics.histogram(
                MetricName.CROP_SCALE_PIXELS, camera_id=str(camera_id)
            ).record(grades.scale_pixels)
        if grades.overall is not None:
            self._metrics.counter(
                MetricName.CROP_QUALITY_GRADE,
                camera_id=str(camera_id),
                overall=grades.overall.value,
            ).increment()
        return crop

    @staticmethod
    def content_hash(payload: bytes) -> str:
        """The crop's identity: a hash of its normalized pixels.

        02_VOM section 4.1 requires content addressing, and the property that
        matters is *"the same pixels cropped twice must be one crop"* — which
        gives free deduplication, free cache keys, free integrity checking, and
        an evidence reference that survives storage migration. A sequential id
        would give none of those, and two runs over the same footage would
        produce different evidence.
        """
        return hashlib.sha256(payload).hexdigest()

    # --- public API: demands ------------------------------------------------------ #

    def register_demand(self, demand: Demand) -> DemandAcknowledgement:
        """Admit a demand, or refuse it honestly.

        The acknowledgement reports ``effective_freshness`` — what the platform
        can actually sustain, which may be longer than requested. Accepting a
        demand the budget will not buy and quietly under-delivering is the
        single most common integration failure in vision platforms, and
        09_API_CONTRACTS section 4.2 exists to prevent it.
        """
        acknowledgement = self._demands.register(
            demand,
            now=self._clock.now(),
            sustainable_freshness=self._sustainable_freshness(),
        )
        self._metrics.counter(
            MetricName.DEMANDS_REGISTERED, status=acknowledgement.status.value
        ).increment()
        return acknowledgement

    def revoke_demand(self, demand_id: DemandId) -> None:
        self._demands.revoke(demand_id)
        self._metrics.counter(MetricName.DEMANDS_REVOKED).increment()

    def budget_status(self) -> BudgetStatus:
        return self._budget.status(self._clock.monotonic())

    def _count_verification(
        self, camera_id: CameraId, decisions: Sequence[TriggerDecision]
    ) -> None:
        """Count corroboration decisions from their outcomes, not from the policy.

        M8 does not know whether a verification policy is bound, and asking one
        would couple the engine to an adapter it is supposed to be indifferent
        to. The two outcomes are observable in the decision itself, which is
        enough — and keeps the policy stateless (obligation G6).

        The pair is what makes the ratio readable: a deployment needs the
        withheld count as much as the required one, because a request rate with
        no denominator cannot distinguish restraint from rules that never fired.
        """
        for decision in decisions:
            if decision.reason is TriggerReason.IDENTITY_UNVERIFIED:
                self._metrics.counter(
                    MetricName.VERIFICATION_REQUIRED,
                    camera_id=str(camera_id),
                    reason=decision.reason.value,
                ).increment()
            elif decision.skip is SkipReason.EVIDENCE_SUFFICIENT:
                self._metrics.counter(
                    MetricName.VERIFICATION_WITHHELD,
                    camera_id=str(camera_id),
                    skip=decision.skip.value,
                ).increment()
            else:
                continue
            self._metrics.counter(
                MetricName.VERIFICATION_CANDIDATES, camera_id=str(camera_id)
            ).increment()

    def _evidence_groups(
        self, attributes: Sequence[AttributeKey]
    ) -> tuple[tuple[AttributeKey, ...], ...]:
        """Partition demanded attributes by the region that answers them.

        The grouping key is the declared ``(top, height)`` band, so two
        attributes visible in the same place travel together and cost one call
        between them. M8 never learns what a region *means* — it compares two
        pairs of floats supplied by configuration, exactly as it compares a
        priority class it does not interpret.

        Order is deterministic (V13): groups come back sorted by band, and each
        group's attributes sorted by key, so the same demand always produces the
        same requests in the same sequence and a replay reproduces them.

        Attributes with no declared region share the empty band, which means a
        deployment that declared none gets exactly one group — the previous
        behaviour, unchanged.
        """
        if not attributes:
            return ()

        buckets: dict[tuple[float, float] | None, list[AttributeKey]] = {}
        for key in attributes:
            buckets.setdefault(self._evidence_regions.get(str(key)), []).append(key)

        # `None` sorts first and is the undeclared bucket; declared bands sort by
        # position down the subject, so a head group precedes a hand group.
        return tuple(
            tuple(sorted(keys, key=str))
            for _, keys in sorted(
                buckets.items(), key=lambda item: (item[0] is not None, item[0] or (0.0, 0.0))
            )
        )

    # --- candidate construction ---------------------------------------------------- #

    def _candidate(
        self,
        obj: VisualObject,
        frame: FrameContext,
        regions_of: Callable[[ObjectId], frozenset] | None,
        appearance_of: Callable[[VisualObject], float | None] | None,
        now: Instant,
    ) -> TriggerCandidate:
        camera_id = frame.frame_ref.camera_id
        state = self._state.partition(camera_id).state_for(obj.object_id, now=now)

        regions = regions_of(obj.object_id) if regions_of else frozenset()
        entered = bool(regions - state.last_region_ids)
        lifecycle = obj.lifecycle.value
        lifecycle_changed = bool(state.last_lifecycle) and state.last_lifecycle != lifecycle
        state.last_region_ids = regions
        state.last_lifecycle = lifecycle

        signature = appearance_of(obj) if appearance_of else None
        delta = state.note_appearance(signature)

        box = self._box_of(obj)
        grades = self._estimator.estimate(
            QualityRequest(
                camera_id=camera_id,
                box=box,
                source_width=frame.width,
                source_height=frame.height,
            )
        )

        return TriggerCandidate(
            object_id=obj.object_id,
            camera_id=camera_id,
            class_id=obj.class_id,
            box=box,
            lifecycle=lifecycle,
            identity_confidence=obj.confidence.value,
            first_seen=obj.first_seen,
            last_confirmed=obj.last_confirmed,
            observation_count=obj.observation_count,
            region_ids=tuple(sorted(regions)),
            attributes=self._attribute_status(obj),
            appearance_delta=delta,
            last_analysed=state.last_analysed,
            last_gate_rejection=state.last_gate_rejection is not None,
            entered_region_this_frame=entered,
            lifecycle_changed_this_frame=lifecycle_changed,
            estimated_quality=grades,
            class_confidence=self._class_confidence(obj),
            class_alternatives=self._class_alternatives(obj),
            label_space_kind=self._label_space.kind,
            class_in_native_vocabulary=self._label_space.covers(obj.class_id),
        )

    @staticmethod
    def _class_confidence(obj: VisualObject) -> Confidence | None:
        """The detector's own score for the class the object currently carries.

        Read from ``class_history`` rather than from ``obj.confidence``: the
        latter is ``IDENTITY`` confidence — P(this track is this object) — and
        handing it to a policy as a classification score would present one
        quantity as another, which 02_VOM section 7.2 exists to prevent.

        The most recent entry *for the published class* is used, not simply the
        most recent entry. A flapping object's last sighting may have been the
        losing class, and reporting that score would describe a claim the
        platform is not making.
        """
        for sighting in reversed(obj.class_history):
            if sighting.class_id == obj.class_id:
                return sighting.confidence
        return None

    @staticmethod
    def _class_alternatives(obj: VisualObject) -> tuple[tuple[ClassId, float], ...]:
        """Every class this object has been called, by share of retained evidence.

        Derived from ``class_history`` rather than from ``Detection.class_scores``
        — the per-frame distribution is not propagated past the Detection Engine,
        and what a policy actually wants here is *temporal* consistency, which is
        what the history records.

        The published class is excluded: a policy asking "what else might this
        be?" is not helped by being told the answer it already has. Empty means
        the object has only ever been called one thing, which is the stable case.
        """
        weights: dict[ClassId, float] = {}
        for sighting in obj.class_history:
            if sighting.class_id == obj.class_id:
                continue
            weights[sighting.class_id] = (
                weights.get(sighting.class_id, 0.0) + sighting.confidence.value
            )
        total = sum(weights.values()) + sum(
            s.confidence.value for s in obj.class_history if s.class_id == obj.class_id
        )
        if total <= 0.0:
            return ()
        # Sorted by share, then by class id: an arbitrary tie-break would make
        # the policy's input depend on dict ordering and break replay (V13).
        return tuple(
            (class_id, weights[class_id] / total)
            for class_id in sorted(weights, key=lambda c: (-weights[c], c))
        )

    def _attribute_status(self, obj: VisualObject) -> dict[AttributeKey, AttributeStatus]:
        """What the platform already knows, as the policy needs to see it.

        Read from M7, never cached here. A second copy of attribute freshness
        would drift from the registry's, and the drift would be invisible until
        the platform started re-analysing things it already knew.
        """
        return {
            key: AttributeStatus(
                key=key,
                observed_at=attribute.observed_at,
                confidence=attribute.confidence.value,
                valid_until=attribute.valid_until,
            )
            for key, attribute in obj.attributes.items()
        }

    @staticmethod
    def _box_of(obj: VisualObject) -> Box:
        box = obj.current_spatial.bbox
        if box is None:
            # An object with no box cannot be cropped. Returning a degenerate
            # unit box lets the gate reject it with DEGENERATE_GEOMETRY — a
            # counted, explicable outcome rather than an exception from a
            # geometry helper that has no idea what to do about it.
            return Box(0.0, 0.0, 1e-4, 1e-4)
        return box

    # --- rejection and alarms -------------------------------------------------------- #

    def _reject(
        self, request: CropRequest, grades: QualityGrades, result: GateResult
    ) -> None:
        """Record the rejection, refund the budget, and raise.

        Refunding matters: a run of rejections would otherwise exhaust the budget
        having bought nothing, and the platform would stop looking at the things
        it *could* have answered.
        """
        camera_id = request.camera_id
        reason = result.reason
        self._budget.refund(demand_ids=request.demand_ids)
        self._metrics.counter(
            MetricName.CROPS_GATE_REJECTED,
            camera_id=str(camera_id),
            reason=reason.value if reason else "unknown",
        ).increment()
        state = self._state.partition(camera_id).state_for(
            request.object_id, now=self._clock.now()
        )
        if reason is not None:
            state.note_gate_rejection(reason)
        self._record_gate(camera_id, result)
        self._maybe_publish_capability_gap(request, state, result)
        raise GateRejectedError(
            f"crop for object '{request.object_id}' rejected: {result.detail}",
            object_id=str(request.object_id),
            camera_id=str(camera_id),
            reason=reason.value if reason else "unknown",
            scale_pixels=grades.scale_pixels,
            overall=grades.overall.value if grades.overall else None,
        )

    def _record_gate(self, camera_id: CameraId, result: GateResult) -> None:
        window = self._gate_windows.get(camera_id)
        if window is None:
            window = GateRejectionWindow(
                camera_id=camera_id,
                window=max(self._config.gate_rejection_sample_size * 5, 20),
            )
            self._gate_windows[camera_id] = window
        window.record(passed=result.passed, reason=result.reason)

        if window.sample_size < self._config.gate_rejection_sample_size:
            return
        spiking = window.rejection_rate >= self._config.gate_rejection_spike_threshold
        if spiking and not window.alarm_active:
            window.alarm_active = True
            dominant = window.dominant_reason()
            self._events.publish(
                GateRejectionSpike(
                    occurred_at=self._clock.now(),
                    partition_key=str(camera_id),
                    camera_id=camera_id,
                    reason=dominant.value if dominant else "unknown",
                    rejection_rate=window.rejection_rate,
                    sample_size=window.sample_size,
                    detail=(
                        "gate rejections are dominated by "
                        f"{dominant.value if dominant else 'an unknown cause'}; "
                        "this is usually physical — a camera nudged, a lens "
                        "fouled, a light failed"
                    ),
                )
            )
        elif not spiking:
            # Hysteresis by state rather than by threshold: the alarm clears when
            # the rate recovers, and cannot re-fire every frame while elevated.
            window.alarm_active = False

    def _maybe_publish_capability_gap(self, request, state, result: GateResult) -> None:
        """Tell a consumer to stop waiting for data that will never arrive.

        §M8's failure table: a demand that can never be satisfied at this camera
        deserves an explicit answer. The threshold is high because this is a
        claim about the *future*, and a claim about the future should be slow to
        make.
        """
        if state.consecutive_gate_rejections != self._config.capability_gap_threshold:
            return
        for demand_id in request.demand_ids:
            self._events.publish(
                CapabilityGap(
                    occurred_at=self._clock.now(),
                    partition_key=str(request.camera_id),
                    camera_id=request.camera_id,
                    demand_id=str(demand_id),
                    attribute_key=(
                        request.required_attributes[0]
                        if request.required_attributes
                        else ""
                    ),
                    reason=result.reason.value if result.reason else "unknown",
                    consecutive_failures=state.consecutive_gate_rejections,
                    detail=(
                        "this object is persistently ungradable at this camera; "
                        "the demand cannot be served here"
                    ),
                )
            )
            self._metrics.counter(
                MetricName.DEMANDS_UNSATISFIABLE,
                camera_id=str(request.camera_id),
                reason=result.reason.value if result.reason else "unknown",
            ).increment()

    def _publish_budget_exhausted(self) -> None:
        status = self._budget.status(self._clock.monotonic())
        self._metrics.counter(MetricName.UNDERSTANDING_BUDGET_SHED).increment()
        self._metrics.gauge(MetricName.UNDERSTANDING_BUDGET_PRESSURE).set(
            status.pressure
        )
        self._events.publish(
            BudgetExhausted(
                occurred_at=self._clock.now(),
                ceiling_per_hour=status.ceiling_per_hour,
                spent_in_window=status.spent_in_window,
                shed_in_window=status.shed_in_window,
                pressure=status.pressure,
                detail=(
                    "attributes are being thinned by priority; consumers learn "
                    "this from coverage rather than from silence (V8)"
                ),
            )
        )

    def _skip(self, decision: TriggerDecision, camera_id: CameraId) -> Skipped:
        return Skipped(
            object_id=decision.object_id,
            camera_id=camera_id,
            reason=decision.skip,
            detail=decision.detail,
            attribute_keys=tuple(str(k) for k in decision.attributes),
        )

    # --- helpers ---------------------------------------------------------------------- #

    def _sustainable_freshness(self) -> Duration | None:
        """What the budget can actually buy, as a freshness interval.

        The honest half of ``register_demand``. With a ceiling of C calls/hour
        and N objects currently tracked, a single attribute can be refreshed at
        most every ``N/C`` hours — reported to the consumer rather than
        discovered by them from thin results.
        """
        ceiling = self._budget.ceiling_per_hour
        if ceiling <= 0:
            return None
        tracked = max(1, self._state.tracked_objects)
        seconds_per_call = 3600.0 / ceiling
        return Duration.from_millis(int(tracked * seconds_per_call * 1000))

    # --- observability ------------------------------------------------------------------ #

    def health(self) -> ComponentHealth:
        status = self.budget_status()
        state = HealthState.HEALTHY
        detail = "attention nominal"
        if status.exhausted:
            state = HealthState.DEGRADED
            detail = (
                f"understanding budget exhausted: {status.spent_in_window} calls "
                f"spent this window, {status.shed_in_window} shed"
            )
        elif self._failures:
            state = HealthState.DEGRADED
            detail = f"{self._failures} evaluation failures"
        return ComponentHealth(
            component_id=CROP_MANAGER_ID,
            state=state,
            reported_at=self._clock.now(),
            detail=detail,
            metrics={
                "frames_evaluated": float(self._frames_evaluated),
                "budget_pressure": status.pressure,
                "tracked_objects": float(self._state.tracked_objects),
                "demands_active": float(len(self._demands.active())),
            },
        )

    @property
    def demands(self) -> DemandRegistry:
        return self._demands

    @property
    def budget(self) -> UnderstandingBudget:
        return self._budget

    @property
    def cache(self) -> CropDeduplicationCache:
        return self._cache

    @property
    def trigger_state(self) -> TriggerStateStore:
        return self._state

    @property
    def frames_evaluated(self) -> int:
        return self._frames_evaluated

    @property
    def failures(self) -> int:
        return self._failures

    def forget_camera(self, camera_id: CameraId) -> None:
        """Release a camera's state after it detaches."""
        self._state.drop(camera_id)
        self._gate_windows.pop(camera_id, None)

    def expire_demands(self) -> tuple[DemandId, ...]:
        expired = self._demands.expire_due(self._clock.now())
        self._metrics.gauge(MetricName.DEMANDS_ACTIVE).set(
            float(len(self._demands.active()))
        )
        return expired


@dataclass(slots=True)
class _AdmissionCandidate:
    """A firing decision awaiting budget, carrying its object.

    ``priority_class`` is what ``PriorityQueue`` orders by; it is read through
    ``getattr`` there, so this attribute is the contract.
    """

    decision: TriggerDecision
    obj: VisualObject

    @property
    def priority_class(self) -> str:
        return self.decision.priority_class


def usable(grades: QualityGrades) -> bool:
    """Whether grades describe an input worth spending a model on."""
    return grades.overall is not None and grades.overall.is_usable


def quality_level_of(grades: QualityGrades) -> QualityLevel | None:
    return grades.overall
