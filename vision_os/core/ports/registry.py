"""P11 ``IdentityResolverPort`` and the durable object-store seam.

**P11 ships with no implementations, deliberately.** ``15_ROADMAP`` section 3 is
explicit: *"`IdentityResolverPort` (P11) — already specified, **no
implementations in Phase 1**"*, and *"M7 already accepts identity assertions from
a resolver."*

The distinction that makes this coherent:

* M7's **native** track-to-object binding — responsibility 2, *"bind tracks to
  objects with method and confidence; re-bind after breaks"* — is mandatory
  behaviour that must work with nothing bound. It is spatio-temporal,
  within-camera, and not pluggable.
* P11 is the seam for **replacing or augmenting** that with appearance-based,
  cross-camera, or learned strategies. That is Phase 2, and cross-camera identity
  is classified C2 and policy-gated (``12_SECURITY`` section 2.3).

So the port is declared with its full contract, and the frontier keeps it
unbindable — the same posture Flow 3 took with ``EmbeddingPort``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model.ids import CameraId, ObjectId, SiteId, TrackId
from ..model.space import SpatialInfo
from ..model.timebase import Instant
from ..model.visual_object import BindingMethod, VisualObject

IDENTITY_RESOLVER_PORT_VERSION = "1.0.0"
OBJECT_STORE_PORT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ResolutionCandidate:
    """One object a resolver considered, with its score.

    Scores are reported for *every* candidate, not just the winner. Section M7
    requires that an ambiguous re-entry produce a new object plus a
    low-confidence assertion linking the candidates — which is only possible if
    the alternatives survive the decision.
    """

    object_id: ObjectId
    score: float
    method: BindingMethod
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"candidate score must be in [0,1], got {self.score}")


@dataclass(frozen=True, slots=True)
class ResolutionRequest:
    """An unbound track seeking an object.

    Carries **no pixels and no embedding** — a resolver that needs appearance
    obtains it through ``EmbeddingPort``, which is separately gated because
    embeddings are C2 biometric data.
    """

    camera_id: CameraId
    site_id: SiteId
    track_id: TrackId
    observed_at: Instant
    spatial: SpatialInfo
    class_id: str
    candidates: Sequence[VisualObject] = ()
    """Objects the registry considers plausible. A resolver may score any subset;
    it may not invent an object outside this set, because minting identity is the
    registry's sole authority (01_LAYERED section 8)."""


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """A resolver's answer. **Advisory** — the registry decides.

    A resolver proposes; it never mutates. Keeping the decision inside M7 is what
    makes "exactly one module may mint or retire an object identity" true even
    when a third-party adapter is bound.
    """

    ranked: tuple[ResolutionCandidate, ...] = ()
    abstained: bool = False
    """True when the resolver had no basis to answer. Distinct from ranking
    nothing, which asserts that no candidate matches (invariant V8)."""

    reason: str = ""

    @property
    def best(self) -> ResolutionCandidate | None:
        return self.ranked[0] if self.ranked else None

    @property
    def margin(self) -> float | None:
        """Score gap between the top two candidates.

        A narrow margin is the ambiguity that section M7 forbids resolving
        silently: *"Create a new object and emit a low-confidence identity
        assertion linking candidates. **Never guess silently**."*
        """
        if len(self.ranked) < 2:
            return None
        return self.ranked[0].score - self.ranked[1].score


@runtime_checkable
class IdentityResolverPort(Protocol):
    """P11 — propose which existing object an unbound track continues.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **I1** | **Advisory only.** A resolver proposes; it never mutates an object and never mints an id. |
    | **I2** | Candidates are scored, not selected. All considered candidates are returned with their scores so the registry can detect ambiguity. |
    | **I3** | Abstention is explicit. "No basis to answer" and "no candidate matches" are different results (V8). |
    | **I4** | A resolver may not propose an object outside the supplied candidate set. |
    | **I5** | Deterministic: identical input yields identical ranking, including tie order (V13). |
    | **I6** | Stateless across calls, or state is per-camera and reset with the partition. |
    | **I7** | Declares whether it requires embeddings; a resolver requiring them fails to activate when none are available rather than degrading silently. |

    **No implementations ship in Phase 1.**
    """

    @property
    def resolver_id(self) -> str:
        ...

    @property
    def requires_embeddings(self) -> bool:
        """Appearance-based resolution is C2 biometric and disabled by default."""
        ...

    def resolve(self, request: ResolutionRequest) -> ResolutionResult:
        """Score the candidates. Never mutates, never mints."""
        ...


@dataclass(frozen=True, slots=True)
class PartitionSnapshot:
    """One camera partition's durable objects, for persistence and reload.

    Carries a ``version`` so a reload can detect a stale write, and the
    ``tracker_epoch`` in force when it was taken — after a restart every track is
    new, so re-binding must know it is crossing an epoch and reduce its
    confidence accordingly (07_STATE section 9.3).
    """

    camera_id: CameraId
    site_id: SiteId
    version: int
    taken_at: Instant
    objects: tuple[VisualObject, ...] = ()
    next_local_sequence: int = 0

    @property
    def count(self) -> int:
        return len(self.objects)


@runtime_checkable
class ObjectStorePort(Protocol):
    """Durable storage for the object population.

    Narrow by design. Section M7 lists "Storage Interfaces (durable object
    state)" among its dependencies, but that module (M12) belongs to a later
    flow; this port is the minimum contract M7 needs to satisfy ``07_STATE``
    section 9.3's *"object identity survives, tracks do not"* without
    implementing a storage layer it does not own.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **S1** | Writes are **atomic per partition**. A partially written partition is worse than a lost one, because it reloads as plausible corruption. |
    | **S2** | ``load`` returns ``None`` for an unknown partition — absence is not an error. |
    | **S3** | Never repairs. A snapshot that fails to decode raises; it is never silently downgraded to an empty partition, which would present data loss as a fresh start. |
    | **S4** | Persistence never blocks the caller's hot path; the registry enqueues and the store drains. |
    """

    @property
    def store_id(self) -> str:
        ...

    def save(self, snapshot: PartitionSnapshot) -> None:
        """Persist one partition atomically.

        Raises:
            ObjectStoreError: the write failed. Durability degrades; ingestion
                does not stop.
        """
        ...

    def load(self, camera_id: CameraId) -> PartitionSnapshot | None:
        """Read a partition. ``None`` when none was ever written.

        Raises:
            ObjectStoreError: the stored data exists but could not be decoded.
                Never silently returns an empty partition — see S3.
        """
        ...

    def forget(self, camera_id: CameraId) -> None:
        """Drop a partition's durable state. Used by retention and erasure."""
        ...
