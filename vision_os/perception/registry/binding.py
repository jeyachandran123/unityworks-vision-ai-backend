"""Track-to-object binding — M7 responsibility 2.

> **Single responsibility:** *Decide which existing object a track continues, or
> that none does. Mint nothing, mutate nothing.*

This is where the registry's central judgement lives, and section M7 constrains
it sharply:

> *Re-entry ambiguity (two candidates match) → Create a **new** object and emit a
> low-confidence identity assertion linking candidates. **Never guess silently**;
> let the consumer choose a confidence threshold (V1).*

So this module produces a *decision with its alternatives intact*, and the caller
records both. A binder that returned only its winner would make the ambiguity
unrecoverable one function call after it was known.

Four binding methods, in descending strength:

``TRACK_CONTINUITY``
    The same ``TrackId`` within one epoch. The strongest claim, because M6
    already asserted the continuity and M7 is only recording it.

``SPATIO_TEMPORAL``
    A new track matched an occluded or dormant object by position and elapsed
    time. The default re-binding strategy, and the one that survives occlusion.

``EPOCH_REBIND``
    Re-binding across a tracker epoch — a restart or reset. ``07_STATE`` section
    9.3 requires this carry **explicitly reduced confidence**: every track is new
    after a restart, so continuity is inferred rather than observed.

``RESOLVER``
    Asserted by an ``IdentityResolverPort``. No implementations ship in Phase 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.model.ids import ObjectId, TrackId
from ...core.model.space import SpatialInfo
from ...core.model.timebase import Duration, Instant
from ...core.model.visual_object import BindingMethod, LifecycleState
from .partition import ObjectRecord, spatial_distance


@dataclass(frozen=True, slots=True)
class BindingPolicy:
    """Thresholds governing re-binding. Strongly typed, validated on build."""

    max_reentry_distance: float = 0.25
    """Normalized centre separation within which a re-entering track may be
    matched to a dormant object. Beyond this the claim is not credible."""

    max_reentry_gap: Duration = Duration.from_millis(30_000)
    """How long an object may go unmeasured and still be re-bindable by position.
    Past this, position tells you almost nothing: an object can cross the whole
    frame and come back."""

    ambiguity_margin: float = 0.15
    """Minimum score gap between the best and second-best candidate. Below it the
    match is refused, a new object is minted, and the alternatives are published
    — section M7's *"never guess silently"* made executable."""

    min_binding_confidence: float = 0.3
    """Below this a candidate is not proposed at all."""

    epoch_rebind_penalty: float = 0.5
    """Multiplier applied when re-binding across a tracker epoch. 07_STATE
    section 9.3: re-binding after restart happens *"with explicitly reduced
    confidence"*."""

    class_must_match: bool = True
    """Whether a re-bind requires class agreement. A person does not become a
    forklift, and allowing it produces objects whose class history is nonsense."""

    def __post_init__(self) -> None:
        for name in ("max_reentry_distance", "ambiguity_margin", "min_binding_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if not 0.0 < self.epoch_rebind_penalty <= 1.0:
            raise ValueError("epoch_rebind_penalty must be in (0,1]")
        if self.max_reentry_gap.ns <= 0:
            raise ValueError("max_reentry_gap must be positive")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One object a track might continue, with the evidence behind the score."""

    object_id: ObjectId
    score: float
    method: BindingMethod
    distance: float
    gap: Duration
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class BindingDecision:
    """What the binder concluded, **with the alternatives retained**.

    ``matched is None`` with a non-empty ``candidates`` is the important case: it
    means candidates existed but none was decisive, so the caller must mint a new
    object *and* publish the alternatives as a low-confidence assertion.
    """

    matched: Candidate | None
    candidates: tuple[Candidate, ...] = ()
    ambiguous: bool = False
    """True when two candidates were too close to separate. The registry mints a
    new object and links them rather than guessing."""

    reason: str = ""

    @property
    def margin(self) -> float | None:
        if len(self.candidates) < 2:
            return None
        return self.candidates[0].score - self.candidates[1].score

    @property
    def alternatives(self) -> tuple[tuple[ObjectId, float], ...]:
        """Every candidate considered, for the identity assertion."""
        return tuple((c.object_id, c.score) for c in self.candidates)


class TrackBinder:
    """Scores candidate objects for an unbound track. Pure, holds no state.

    Deliberately stateless: it receives records and returns a decision, so it is
    exhaustively testable without constructing a partition, and it cannot become
    a second place where object state lives.
    """

    __slots__ = ("_policy",)

    def __init__(self, policy: BindingPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> BindingPolicy:
        return self._policy

    def bind_continuing(
        self, records: Sequence[ObjectRecord], track_id: TrackId
    ) -> BindingDecision:
        """The strongest case: this track already owns an object.

        No scoring is involved. M6 asserted the continuity within its epoch, and
        re-deciding it here would be M7 second-guessing a claim it did not make
        and cannot improve on.
        """
        for record in records:
            if record.bound_track == track_id and not record.lifecycle.is_terminal:
                return BindingDecision(
                    matched=Candidate(
                        object_id=record.object_id,
                        score=1.0,
                        method=BindingMethod.TRACK_CONTINUITY,
                        distance=0.0,
                        gap=Duration(0),
                        rationale="same track id within one epoch",
                    ),
                    reason="track_continuity",
                )
        return BindingDecision(matched=None, reason="no_open_binding")

    def bind_reentry(
        self,
        records: Sequence[ObjectRecord],
        *,
        spatial: SpatialInfo,
        class_id: str,
        now: Instant,
        crossing_epoch: bool = False,
    ) -> BindingDecision:
        """Score unbound objects as candidates for a new track.

        Only objects that are plausibly still around are considered: ``occluded``
        (believed present) and ``dormant`` (retained for re-entry). An ``active``
        object already has a track; a ``departed`` one has left, and re-binding it
        would assert a continuity nobody observed.
        """
        policy = self._policy
        candidates: list[Candidate] = []

        for record in records:
            if record.lifecycle not in (LifecycleState.OCCLUDED, LifecycleState.DORMANT):
                continue
            if record.bound_track is not None:
                continue
            if policy.class_must_match and not _class_compatible(record.class_id, class_id):
                continue

            gap = Duration(max(0, now.ns - record.last_confirmed.ns))
            if gap.ns > policy.max_reentry_gap.ns:
                continue

            distance = spatial_distance(record.spatial, spatial)
            if distance > policy.max_reentry_distance:
                continue

            score = _score(distance, gap, policy)
            if crossing_epoch:
                score *= policy.epoch_rebind_penalty
            if score < policy.min_binding_confidence:
                continue

            candidates.append(
                Candidate(
                    object_id=record.object_id,
                    score=score,
                    method=(
                        BindingMethod.EPOCH_REBIND
                        if crossing_epoch
                        else BindingMethod.SPATIO_TEMPORAL
                    ),
                    distance=distance,
                    gap=gap,
                    rationale=(
                        f"distance {distance:.3f}, gap {gap.millis:.0f}ms"
                        + (", across tracker epoch" if crossing_epoch else "")
                    ),
                )
            )

        # Deterministic ordering including ties: score descending, then object id.
        # An arbitrary tie-break would make identity depend on dict order (V13).
        candidates.sort(key=lambda c: (-c.score, c.object_id))
        ordered = tuple(candidates)

        if not ordered:
            return BindingDecision(matched=None, reason="no_candidates")

        if len(ordered) >= 2:
            margin = ordered[0].score - ordered[1].score
            if margin < policy.ambiguity_margin:
                # Section M7: create a new object and emit a low-confidence
                # assertion linking candidates. Never guess silently.
                return BindingDecision(
                    matched=None,
                    candidates=ordered,
                    ambiguous=True,
                    reason=(
                        f"ambiguous re-entry: {len(ordered)} candidates within "
                        f"{margin:.3f} of each other"
                    ),
                )

        return BindingDecision(
            matched=ordered[0],
            candidates=ordered,
            reason="spatio_temporal_match",
        )


def _score(distance: float, gap: Duration, policy: BindingPolicy) -> float:
    """Combine spatial proximity and recency into [0,1].

    Both decay linearly to zero at their respective limits, and the product is
    used rather than a weighted sum: a candidate that is close but very stale, or
    recent but far away, should score low on *both* counts rather than having one
    strength mask the other weakness.
    """
    spatial_term = max(0.0, 1.0 - distance / policy.max_reentry_distance)
    temporal_term = max(0.0, 1.0 - gap.ns / policy.max_reentry_gap.ns)
    return spatial_term * temporal_term


def _class_compatible(existing: str, incoming: str) -> bool:
    """Whether two taxonomy classes may describe one object.

    Hierarchical: ``person`` and ``person.child`` are compatible, because a
    detector refining its answer is normal. ``person`` and ``vehicle`` are not.
    """
    return (
        existing == incoming
        or existing.startswith(f"{incoming}.")
        or incoming.startswith(f"{existing}.")
    )
