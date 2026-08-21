"""The VisualObject — the central persistent entity (02_VOM section 10.6).

The platform's first durable, semantically meaningful record. A ``Track`` says
"these detections are one thing, on this camera, for now"; a ``VisualObject``
says "this is a thing that persists", and it survives occlusion, track breaks,
re-entry, and process restart.

**Fields are exactly those of 02_VOM section 10.6.** Nothing renamed, nothing
merged in from ``Track``, nothing invented. Two absences are deliberate and worth
stating, because both are tempting:

``regions`` is not a field
    M7 owns region membership (section M7 State Ownership) but ``VisualObject``
    has no such field. Membership is partition state keyed by object. The richer
    ``ObjectState`` of 07_STATE section 3.1 — which *does* carry ``regions``,
    ``trajectory`` and ``last_observation`` — is the **L6 Vision State
    projection**, a different type at a different layer, and building it here
    would be implementing Flow 7 early.

``merged_into`` is a lifecycle state, not a deletion
    Observations that referenced the old id must stay valid and resolvable
    (invariant V5). An object is never removed to fix an identity error; it is
    superseded, and the correction is a new fact with lineage.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from .confidence import Confidence, ConfidenceSemantics
from .ids import (
    AttributeKey,
    BindingId,
    CameraId,
    ClassId,
    ObjectId,
    SiteId,
    TenantId,
    TrackId,
)
from .provenance import Provenance
from .space import SpatialInfo
from .timebase import Duration, Instant

OBJECT_SCHEMA_VERSION = "1.0.0"


class LifecycleState(enum.Enum):
    """The seven states of 02_VOM section 10.6. Closed set.

    Read ``occluded`` and ``dormant`` as different claims about *why* the object
    is unmeasured: occluded means "believed present, not currently measurable";
    dormant means "not observed recently, retained for possible re-entry". A
    consumer treats them differently, so collapsing them would lose information
    the platform has.
    """

    PROVISIONAL = "provisional"
    """Seen, not yet confirmed as a real object."""

    ACTIVE = "active"
    """Currently observed."""

    OCCLUDED = "occluded"
    """Believed present, not currently measurable."""

    DORMANT = "dormant"
    """Not observed recently, retained for possible re-entry."""

    DEPARTED = "departed"
    """Believed to have left the observable area."""

    MERGED_INTO = "merged_into"
    """Identity resolution merged this into another object. Terminal, and
    **not** a deletion: the record survives so history stays resolvable (V5)."""

    EXPIRED = "expired"
    """Retention horizon reached. Terminal."""

    @property
    def is_terminal(self) -> bool:
        return self in (LifecycleState.MERGED_INTO, LifecycleState.EXPIRED)

    @property
    def is_present(self) -> bool:
        """Whether the platform believes the object is in the observable area.

        ``occluded`` counts: the object is believed present, merely unmeasured.
        ``dormant`` does not — it is retained for re-entry, not asserted present.
        """
        return self in (
            LifecycleState.PROVISIONAL,
            LifecycleState.ACTIVE,
            LifecycleState.OCCLUDED,
        )

    @property
    def is_measurable(self) -> bool:
        """Whether a current measurement is expected this frame."""
        return self in (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE)


class BindingMethod(enum.Enum):
    """How a track was tied to an object. Reported, never assumed (02_VOM 4.2)."""

    FIRST_SIGHT = "first_sight"
    """The track created the object; no prior claim was involved."""

    TRACK_CONTINUITY = "track_continuity"
    """The same ``TrackId`` continued within one epoch — the strongest claim,
    because M6 already asserted the continuity and M7 is only recording it."""

    SPATIO_TEMPORAL = "spatio_temporal"
    """A new track matched a dormant or occluded object by position and elapsed
    time. The default re-binding strategy."""

    EPOCH_REBIND = "epoch_rebind"
    """Re-binding after a tracker epoch change (restart or reset). Carries
    explicitly reduced confidence: 07_STATE section 9.3 requires it."""

    APPEARANCE = "appearance"
    """Requires an embedding provider. Ships unbound — appearance embeddings are
    C2 biometric data, disabled by default (12_SECURITY section 4.3)."""

    RESOLVER = "resolver"
    """Asserted by an ``IdentityResolverPort`` adapter. No implementations ship
    in Phase 1 (15_ROADMAP section 3)."""

    MANUAL = "manual"
    """An operator correction. Recorded like any other claim."""


class RevisionReason(enum.Enum):
    """Why an identity claim was superseded."""

    MERGE = "merge"
    SPLIT = "split"
    BETTER_EVIDENCE = "better_evidence"
    OPERATOR_CORRECTION = "operator_correction"


@dataclass(frozen=True, slots=True)
class TrackBinding:
    """One ``(TrackId, from, to, binding_confidence, binding_method)`` record.

    ``to`` is ``None`` while the binding is open. Closing it rather than deleting
    it is what lets a consumer reconstruct which track contributed which interval
    of an object's history.
    """

    binding_id: BindingId
    track_id: TrackId
    bound_from: Instant
    bound_to: Instant | None = None
    confidence: Confidence | None = None
    method: BindingMethod = BindingMethod.FIRST_SIGHT
    superseded_by: BindingId | None = None
    """Set when a later assertion revises this one. The original is retained
    (V5); revision is a new fact, not an edit."""

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            self.confidence.semantics is not ConfidenceSemantics.IDENTITY
        ):
            raise ValueError(
                f"a track binding must carry IDENTITY confidence, got "
                f"{self.confidence.semantics.value}; association confidence "
                f"measures whether a detection continues a track, which is a "
                f"different claim (02_VOM section 7)"
            )
        if self.bound_to is not None and self.bound_to.ns < self.bound_from.ns:
            raise ValueError("a binding cannot close before it opened")

    @property
    def is_open(self) -> bool:
        return self.bound_to is None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    def duration(self, now: Instant) -> Duration:
        end = self.bound_to.ns if self.bound_to is not None else now.ns
        return Duration(max(0, end - self.bound_from.ns))


@dataclass(frozen=True, slots=True)
class ClassObservation:
    """One entry of ``class_history`` — ``(ClassId, PlatformTime, Confidence)``."""

    class_id: ClassId
    observed_at: Instant
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class Attribute:
    """One attribute instance (02_VOM section 9.2).

    M7 **holds** attributes; it never produces them. Extraction is M9
    Understanding, at L4. Holding is storage; producing is inference, and the
    distinction is the whole of the L2/L4 boundary.
    """

    key: AttributeKey
    schema_version: str
    value: object
    confidence: Confidence
    observed_at: Instant
    producer: Provenance
    valid_until: Instant | None = None
    """Staleness horizon. ``None`` means the value does not expire on a schedule."""

    evidence_ref: str | None = None
    superseded_by: str | None = None
    """``ObservationId`` of a later observation that revised this."""

    def __post_init__(self) -> None:
        if self.confidence.semantics not in (
            ConfidenceSemantics.ATTRIBUTE,
            ConfidenceSemantics.SELF_REPORTED,
        ):
            raise ValueError(
                f"an Attribute must carry ATTRIBUTE or SELF_REPORTED confidence, "
                f"got {self.confidence.semantics.value} (02_VOM section 9.2)"
            )

    def is_stale(self, now: Instant) -> bool:
        """The object-level expression of V8.

        A consumer reading a value observed 40 minutes ago is reading something
        quite different from a fresh measurement, and the platform says so
        without being asked.
        """
        return self.valid_until is not None and now.ns > self.valid_until.ns

    def staleness(self, now: Instant) -> Duration:
        return Duration(max(0, now.ns - self.observed_at.ns))


@dataclass(frozen=True, slots=True)
class VisualObject:
    """The canonical representation of a persisting visual thing.

    Immutable (V5). The Object Registry is the only writer; every mutation
    produces a new instance, and consumers hold snapshots that cannot drift.
    """

    object_id: ObjectId
    tenant_id: TenantId
    site_id: SiteId
    camera_id: CameraId
    """The partition that owns this object. Site-scoped identity, camera-owned
    partition (section M7 State Ownership)."""

    class_id: ClassId
    """Current best class, resolved from the accumulated distribution."""

    confidence: Confidence
    """``IDENTITY`` semantics — P(this track is this object). Not the tracker's
    association confidence, which measures something else."""

    lifecycle: LifecycleState
    class_history: tuple[ClassObservation, ...]
    track_bindings: tuple[TrackBinding, ...]
    current_spatial: SpatialInfo
    spatial_history: tuple[tuple[Instant, SpatialInfo], ...]
    """Bounded ring. Unbounded history here is the most likely long-run memory
    leak in the platform, which is why the bound is structural rather than a
    tuning parameter (section M7 Performance)."""

    attributes: Mapping[AttributeKey, Attribute]
    """Current values only. History lives in the observation log, not here."""

    first_seen: Instant
    last_seen: Instant
    """Last time the object was updated at all, measured or believed."""

    last_confirmed: Instant
    """Last **measured** sighting.

    The distinction from ``last_seen`` is *measured* versus *believed*, and it is
    what allows a consumer to ask "is this still true, or are we just assuming?"
    — the object-level expression of V8."""

    observation_count: int
    provenance: Provenance
    merged_into: ObjectId | None = None
    """Set with ``lifecycle = MERGED_INTO``. The record is retained so prior
    observations stay resolvable (V5)."""

    lineage: tuple[ObjectId, ...] = ()
    """Objects this was derived from — the other side of a merge, or the parent
    of a split."""

    schema_version: str = OBJECT_SCHEMA_VERSION
    labels: Mapping[str, str] = field(default_factory=dict)
    """Opaque operational tags. No platform logic may branch on these."""

    def __post_init__(self) -> None:
        if self.confidence.semantics is not ConfidenceSemantics.IDENTITY:
            raise ValueError(
                f"a VisualObject must carry IDENTITY confidence, got "
                f"{self.confidence.semantics.value} (02_VOM section 10.6)"
            )
        if (self.lifecycle is LifecycleState.MERGED_INTO) != (self.merged_into is not None):
            raise ValueError(
                "merged_into and LifecycleState.MERGED_INTO must be set together; "
                "a merged object without a target is unresolvable history (V5)"
            )
        if self.merged_into == self.object_id:
            raise ValueError("an object cannot be merged into itself")
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")
        if self.last_seen.ns < self.first_seen.ns:
            raise ValueError("last_seen cannot precede first_seen")
        if self.last_confirmed.ns > self.last_seen.ns:
            raise ValueError(
                "last_confirmed cannot follow last_seen; a measurement cannot be "
                "newer than the most recent update"
            )

    # --- derived, all pure --------------------------------------------------- #

    @property
    def is_present(self) -> bool:
        return self.lifecycle.is_present

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle.is_terminal

    @property
    def open_binding(self) -> TrackBinding | None:
        """The currently bound track, if any."""
        for binding in reversed(self.track_bindings):
            if binding.is_open and not binding.is_superseded:
                return binding
        return None

    @property
    def bound_track(self) -> TrackId | None:
        binding = self.open_binding
        return binding.track_id if binding else None

    def staleness(self, now: Instant) -> Duration:
        """Time since the last **measured** sighting (V8)."""
        return Duration(max(0, now.ns - self.last_confirmed.ns))

    def age(self, now: Instant) -> Duration:
        return Duration(max(0, now.ns - self.first_seen.ns))

    def attribute(self, key: AttributeKey) -> Attribute | None:
        return self.attributes.get(key)

    def stale_attributes(self, now: Instant) -> tuple[AttributeKey, ...]:
        return tuple(k for k, a in self.attributes.items() if a.is_stale(now))

    def bindings_for(self, track_id: TrackId) -> tuple[TrackBinding, ...]:
        return tuple(b for b in self.track_bindings if b.track_id == track_id)

    def is_a(self, ancestor: ClassId) -> bool:
        """Hierarchical class match without consulting the registry."""
        return self.class_id == ancestor or self.class_id.startswith(f"{ancestor}.")


@dataclass(frozen=True, slots=True)
class IdentityAssertion:
    """A revisable claim that a track belongs to an object (02_VOM section 4.2).

    > *An identity link is an assertion with confidence and evidence, never an
    > assumed truth.*

    Emitted rather than silently applied so a consumer can choose its own
    threshold: one counting unique visitors requires high confidence, one drawing
    a live overlay accepts low. The platform does not choose, because the right
    threshold is a business decision (V1).
    """

    binding_id: BindingId
    object_id: ObjectId
    track_id: TrackId
    asserted_at: Instant
    confidence: Confidence
    method: BindingMethod
    evidence: str = ""
    supersedes: BindingId | None = None
    reason: RevisionReason | None = None
    alternatives: tuple[tuple[ObjectId, float], ...] = ()
    """Candidate objects that also matched, with their scores.

    Populated when re-entry was ambiguous. Section M7 requires the registry
    create a *new* object and emit a low-confidence assertion linking candidates
    rather than guessing — this field is what makes the alternatives visible
    instead of discarded.
    """

    def __post_init__(self) -> None:
        if self.confidence.semantics is not ConfidenceSemantics.IDENTITY:
            raise ValueError("an IdentityAssertion must carry IDENTITY confidence")

    @property
    def is_revision(self) -> bool:
        return self.supersedes is not None

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.alternatives)
