"""The default trigger policy — P12.

> §M8: *"the trigger set above is a default policy, fully replaceable.
> Novelty-driven, uncertainty-driven, and learned-salience policies all plug in
> here."*

The policy evaluates the nine documented ``TriggerReason``s in a fixed order and
emits exactly one decision per candidate. The order is not arbitrary: it runs
from *cheapest and most certain* to *most speculative*, so a candidate that
obviously needs analysis is decided without computing an appearance delta.

**The ceiling holds here.** This policy sees measurements and ids — a class id, a
region id, a confidence, an appearance scalar. It never sees a region *label* or
a class *meaning*, so it cannot say "re-look because this is the kitchen" even if
someone wanted it to.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core.model.crop import SkipReason, TriggerReason
from ...core.model.ids import AttributeKey
from ...core.model.timebase import Duration, Instant
from ...core.ports.cropping import TriggerCandidate, TriggerDecision

#: Appearance delta beyond which a re-look is warranted.
#:
#: A *measurement* threshold, not a judgment: it says the pixels changed, never
#: what the change means.
DEFAULT_APPEARANCE_THRESHOLD = 0.25

#: Confidence below which a prior claim is worth re-taking with a better crop.
DEFAULT_LOW_CONFIDENCE = 0.5

#: Cadence floor — the longest an object goes without a look while demanded.
DEFAULT_REFRESH_INTERVAL = Duration.from_millis(300_000)


class DefaultTriggerPolicy:
    """The nine documented triggers, evaluated in cost order.

    Stateless: it receives the candidate's history in the candidate and returns
    a decision, so two cameras cannot disagree about what "stale" means and the
    policy is exhaustively testable without building a Crop Manager.
    """

    __slots__ = (
        "_appearance_threshold",
        "_low_confidence",
        "_refresh_interval",
    )

    def __init__(
        self,
        *,
        appearance_threshold: float = DEFAULT_APPEARANCE_THRESHOLD,
        low_confidence: float = DEFAULT_LOW_CONFIDENCE,
        refresh_interval: Duration = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        if not 0.0 <= appearance_threshold <= 1.0:
            raise ValueError("appearance_threshold must be in [0,1]")
        if not 0.0 <= low_confidence <= 1.0:
            raise ValueError("low_confidence must be in [0,1]")
        if refresh_interval.ns <= 0:
            raise ValueError("refresh_interval must be positive")
        self._appearance_threshold = appearance_threshold
        self._low_confidence = low_confidence
        self._refresh_interval = refresh_interval

    @property
    def policy_id(self) -> str:
        return "trigger.default"

    def evaluate(
        self,
        candidates: Sequence[TriggerCandidate],
        *,
        now: Instant,
        demands: Sequence[object],
    ) -> Sequence[TriggerDecision]:
        """One decision per candidate, in input order (obligations G1, G3).

        ``demands`` is a sequence of ``(camera, class, regions) -> wanted``
        resolvers supplied by the engine; the policy does not query a registry
        itself, which keeps it pure.
        """
        resolver = demands[0] if demands else None
        return [self._decide(candidate, now, resolver) for candidate in candidates]

    def _decide(
        self, candidate: TriggerCandidate, now: Instant, resolver
    ) -> TriggerDecision:
        wanted = self._wanted_for(candidate, resolver)

        # Nothing asked for this object. The largest bucket in a healthy
        # deployment, and the reason cost is `demands x changes`.
        if not wanted:
            return TriggerDecision(
                object_id=candidate.object_id,
                skip=SkipReason.NO_DEMAND,
                detail="no active demand covers this object",
            )

        priority = self._priority_of(wanted)
        demand_ids = self._demand_ids(wanted)

        # 1. Never analysed at all. Cheapest possible determination.
        missing = tuple(
            key
            for key in sorted(wanted)
            if key not in candidate.attributes
            or candidate.attributes[key].observed_at is None
        )
        if missing:
            reason = (
                TriggerReason.FIRST_SIGHT
                if candidate.last_analysed is None
                else TriggerReason.ATTRIBUTE_MISSING
            )
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=reason,
                attributes=missing,
                priority_class=priority,
                demand_ids=demand_ids,
                detail=f"{len(missing)} attribute(s) never computed",
            )

        # 2. Previously gate-rejected and conditions have improved. Checked
        #    before staleness so a recovered object is re-tried promptly rather
        #    than waiting out its freshness window.
        if candidate.last_gate_rejection and self._quality_adequate(candidate):
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.QUALITY_IMPROVED,
                attributes=tuple(sorted(wanted)),
                priority_class=priority,
                demand_ids=demand_ids,
                detail="previously rejected; conditions now adequate",
            )

        # 3. Beyond a demand's freshness SLA.
        stale = tuple(
            key
            for key in sorted(wanted)
            if candidate.attributes[key].is_stale(now, wanted[key][0])
        )
        if stale:
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.ATTRIBUTE_STALE,
                attributes=stale,
                priority_class=priority,
                demand_ids=demand_ids,
                detail=f"{len(stale)} attribute(s) past their freshness SLA",
            )

        # 4. The object changed lifecycle or crossed a region boundary.
        if candidate.lifecycle_changed_this_frame or candidate.entered_region_this_frame:
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.LIFECYCLE_TRANSITION,
                attributes=tuple(sorted(wanted)),
                priority_class=priority,
                demand_ids=demand_ids,
                detail="lifecycle or region transition",
            )

        # 5. Appearance changed beyond the measurement threshold.
        if self._appearance_changed(candidate):
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.APPEARANCE_CHANGED,
                attributes=tuple(sorted(wanted)),
                priority_class=priority,
                demand_ids=demand_ids,
                detail="appearance delta exceeds threshold",
            )

        # 6. A prior claim was weak and a better crop may be available.
        weak = tuple(
            key
            for key in sorted(wanted)
            if (confidence := candidate.attributes[key].confidence) is not None
            and confidence < self._low_confidence
        )
        if weak:
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.LOW_CONFIDENCE,
                attributes=weak,
                priority_class=priority,
                demand_ids=demand_ids,
                detail=f"{len(weak)} prior claim(s) below the confidence floor",
            )

        # 7. Cadence floor.
        if self._refresh_due(candidate, now):
            return TriggerDecision(
                object_id=candidate.object_id,
                reason=TriggerReason.PERIODIC_REFRESH,
                attributes=tuple(sorted(wanted)),
                priority_class=priority,
                demand_ids=demand_ids,
                detail="cadence floor reached",
            )

        # Everything demanded is present, fresh, confident and unchanged. This
        # is the *success* case for V7: the platform correctly spends nothing.
        return TriggerDecision(
            object_id=candidate.object_id,
            skip=SkipReason.FRESH_ENOUGH,
            attributes=tuple(sorted(wanted)),
            priority_class=priority,
            demand_ids=demand_ids,
            detail="all demanded attributes are fresh",
        )

    # --- helpers ------------------------------------------------------------- #

    def _wanted_for(self, candidate: TriggerCandidate, resolver) -> dict:
        if resolver is None:
            return {}
        return resolver(
            camera_id=candidate.camera_id,
            class_id=candidate.class_id,
            region_ids=candidate.region_ids,
        )

    def _priority_of(self, wanted: dict) -> str:
        """The highest-priority class among the demands wanting this object.

        Ordering is the engine's job; the policy only reports which class
        applies. Ties resolve on the attribute key so the answer is stable.
        """
        for key in sorted(wanted):
            priority = wanted[key][1]
            if priority:
                return priority
        return ""

    def _demand_ids(self, wanted: dict) -> tuple[str, ...]:
        ids: list[str] = []
        for key in sorted(wanted):
            for demand_id in wanted[key][2]:
                if demand_id not in ids:
                    ids.append(str(demand_id))
        return tuple(ids)

    def _appearance_changed(self, candidate: TriggerCandidate) -> bool:
        delta = candidate.appearance_delta
        if delta is None:
            return False
        return delta >= self._appearance_threshold

    def _refresh_due(self, candidate: TriggerCandidate, now: Instant) -> bool:
        if candidate.last_analysed is None:
            return True
        return now.ns - candidate.last_analysed.ns >= self._refresh_interval.ns

    def _quality_adequate(self, candidate: TriggerCandidate) -> bool:
        """Whether conditions look good enough to retry a rejected crop.

        A cheap pre-check on the grades already available, so a hopeless object
        is not re-queued every frame. The real gate still runs after extraction.
        """
        grades = candidate.estimated_quality
        if grades.overall is not None:
            return grades.overall.is_usable
        if grades.scale_pixels is not None:
            return grades.scale_pixels > 0.0
        return True


class ExplicitRequestPolicy:
    """Fires ``EXPLICIT_REQUEST`` for a bounded set of objects.

    Wraps another policy rather than replacing it, so an on-demand API call
    (§M8: *"bounded, rate-limited"*) composes with normal triggering instead of
    bypassing it. Everything not explicitly requested falls through.
    """

    __slots__ = ("_inner", "_requested")

    def __init__(self, inner, requested: frozenset = frozenset()) -> None:
        self._inner = inner
        self._requested = requested

    @property
    def policy_id(self) -> str:
        return f"trigger.explicit({self._inner.policy_id})"

    def request(self, object_id) -> None:
        self._requested = self._requested | {object_id}

    def evaluate(
        self,
        candidates: Sequence[TriggerCandidate],
        *,
        now: Instant,
        demands: Sequence[object],
    ) -> Sequence[TriggerDecision]:
        inner = list(self._inner.evaluate(candidates, now=now, demands=demands))
        outgoing: list[TriggerDecision] = []
        served: set = set()

        for candidate, decision in zip(candidates, inner, strict=True):
            if candidate.object_id in self._requested:
                served.add(candidate.object_id)
                outgoing.append(
                    TriggerDecision(
                        object_id=candidate.object_id,
                        reason=TriggerReason.EXPLICIT_REQUEST,
                        attributes=decision.attributes,
                        priority_class=decision.priority_class,
                        demand_ids=decision.demand_ids,
                        detail="explicit on-demand request",
                    )
                )
            else:
                outgoing.append(decision)

        self._requested = self._requested - served
        return outgoing


def attribute_keys(decision: TriggerDecision) -> tuple[AttributeKey, ...]:
    return tuple(AttributeKey(k) for k in decision.attributes)
