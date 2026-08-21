"""M13's durability contracts — P22 `EvidenceStorePort` (06_PORTS §2).

> §M13's single responsibility: *"Describe what must persist and with what
> guarantees; **implement none of it**."*

M13 owns no state. This module defines the one storage contract Flow 8 adds and
says nothing about how a byte reaches a disk. The other four contracts §M13
enumerates are already realized where their consumers live: `ObservationLogPort`
(P20) in `ports/synthesis.py`, `ObjectStorePort` (P21, narrow use) in
`ports/registry.py`, `ConfigSourcePort` (P23) in `ports/configuration.py`, and
`ArtifactStorePort` (P25) in `ports/models.py`.

**Why evidence gets its own contract rather than sharing the log's.** §M13's table
separates them along every axis that matters: the log is append-heavy and
sequential, evidence is write-once and rarely read; the log is the system of
record, evidence is *"medium, policy-driven"*; and — decisively — 07_STATE §8.1
makes evidence *"the shortest tier by design"* at 24–72 hours against the log's
7 days to years. Storing them together would force imagery to inherit the log's
retention, and retention for imagery is a privacy decision rather than an
engineering one.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..model.crop import PrivacyClass, RetentionMode
from ..model.ids import BlobRef, CameraId, ObjectId, ObservationId, TenantId
from ..model.timebase import Instant


class EvidenceStatus(enum.Enum):
    """Why a blob is not available — the distinction §M13 refuses to collapse.

    > §M13: *"``EvidenceStore.get`` distinguishes ``NotFound`` (never existed — a
    > bug) from ``Expired`` (retention did its job — normal). Collapsing these two
    > is how retention behaviour becomes indistinguishable from data loss."*

    An operator seeing `EXPIRED` learns the platform is working. An operator
    seeing `NOT_FOUND` learns something wrote a reference to a blob that was never
    stored. Reporting both as "missing" would hide the second inside the first
    forever, because the second is rarer.
    """

    STORED = "stored"
    NOT_FOUND = "not_found"
    """No blob was ever written under this hash. A bug in whatever minted the ref."""

    EXPIRED = "expired"
    """Retention removed it on schedule. Normal, expected, not an error."""

    ERASED = "erased"
    """Removed ahead of schedule under an erasure request (07_STATE §8.2).

    Distinct from ``EXPIRED`` because the audit answer differs: one says *"policy
    ran"*, the other says *"a subject exercised a right, on this date, under this
    authority."*
    """


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """What the store holds alongside the bytes.

    Content-addressed: ``content_hash`` is the key, so two identical crops are
    stored once. §M13 Performance: *"content-addressed with deduplication."*
    """

    content_hash: str
    tenant_id: TenantId
    privacy_class: PrivacyClass
    retention: RetentionMode
    stored_at: Instant
    expires_at: Instant | None
    size_bytes: int
    camera_id: CameraId | None = None
    object_id: ObjectId | None = None
    observation_id: ObservationId | None = None
    """Indexing fields for §8.2's erasure scopes: *"by object", "by time window",
    "by camera"*. Erasure is only tractable if the store can find what to erase."""

    def is_expired(self, now: Instant) -> bool:
        return self.expires_at is not None and now.ns > self.expires_at.ns

    @property
    def blob_ref(self) -> BlobRef:
        return BlobRef(f"sha256:{self.content_hash}")


@dataclass(frozen=True, slots=True)
class EvidenceFetch:
    """The result of a ``get``. Carries *why* when it carries no bytes."""

    status: EvidenceStatus
    payload: bytes | None = None
    record: EvidenceRecord | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.status is EvidenceStatus.STORED and self.payload is not None

    def __post_init__(self) -> None:
        if self.status is EvidenceStatus.STORED and self.payload is None:
            raise ValueError(
                "a STORED fetch must carry its payload; reporting success with "
                "no bytes is the silent partial success M13 forbids"
            )
        if self.status is not EvidenceStatus.STORED and self.payload is not None:
            raise ValueError(
                f"a {self.status.value} fetch must not carry a payload; a "
                f"consumer reading the bytes without checking the status would "
                f"see expired evidence as current"
            )


@dataclass(frozen=True, slots=True)
class ExpireReport:
    """What a retention pass removed.

    Reported rather than silent because an operator needs to know retention is
    running. A store that quietly expired nothing for a month looks identical to
    one working correctly, right up until the disk fills.
    """

    examined: int = 0
    removed: int = 0
    bytes_reclaimed: int = 0
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def clean(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class EraseReport:
    """What an erasure request removed, and what it left behind on purpose.

    > 07_STATE §8.2: *"erasure removes evidence blobs and redacts identifying
    > content, while retaining an immutable **tombstone** record that an
    > observation existed and was erased, by whom and under what authority. The
    > audit trail survives; the content does not."*

    ``tombstones`` is therefore not a leftover — it is the deliverable. A store
    that erased without tombstoning would satisfy the request and destroy the
    property that makes the record trustworthy.
    """

    erased: int = 0
    bytes_reclaimed: int = 0
    tombstones: tuple[str, ...] = ()
    authority: str = ""
    erased_at: Instant | None = None
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def clean(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class EraseScope:
    """Which evidence an erasure request covers (07_STATE §8.2's four scopes).

    Deliberately **not** "by subject". §8.2: *"by default UWV holds no persistent
    biometric identity, which is a deliberate privacy posture, not a limitation."*
    There is no field here that could name a person, because there is nothing in
    the platform to name them with.
    """

    tenant_id: TenantId
    object_ids: tuple[ObjectId, ...] = ()
    camera_ids: tuple[CameraId, ...] = ()
    before: Instant | None = None
    after: Instant | None = None
    authority: str = ""
    """Who authorized this, recorded on every tombstone. An erasure nobody can
    attribute is indistinguishable from data loss."""

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError(
                "an erasure scope must name a tenant; an unscoped erasure could "
                "cross the isolation boundary 12_SECURITY §4.1 makes absolute"
            )
        if not (self.object_ids or self.camera_ids or self.before or self.after):
            raise ValueError(
                "an erasure scope must name objects, cameras or a time window; "
                "'erase everything for this tenant' is a deployment decision, "
                "not an API call"
            )


@runtime_checkable
class EvidenceStorePort(Protocol):
    """P22 — content-addressed storage for crops and raw model output.

    Implementations: local disk, S3-compatible, encrypted volume.

    ### Semantic contract

    | # | Obligation |
    |---|---|
    | **E1** | **Content-addressed.** The key is the hash of the bytes. Two identical crops are stored once; ``put`` of a known hash is a no-op reported as a hit. |
    | **E2** | **`get` distinguishes `NotFound` from `Expired`** (§M13). Collapsing them makes retention indistinguishable from data loss. |
    | **E3** | **Honours `RetentionMode`.** ``NEVER_PERSIST`` must not touch disk — 12_SECURITY §2.3's no-evidence mode is a hard guarantee, not a hint. |
    | **E4** | **Carries `PrivacyClass`.** The class travels with the blob so no reader has to infer it, and inferring it is how imagery reaches an unclassified path. |
    | **E5** | **Erasure tombstones, never rewrites** (07_STATE §8.2). The blob goes; the record that it existed and was erased remains. |
    | **E6** | Failure is a typed result, never a silent partial success. |
    | **E7** | **Tenant-scoped.** A blob stored under one tenant is unreachable from another, enforced at lookup rather than by filtering after. |
    """

    @property
    def store_id(self) -> str:
        ...

    def put(
        self,
        content_hash: str,
        payload: bytes,
        *,
        tenant_id: TenantId,
        retention: RetentionMode,
        privacy_class: PrivacyClass,
        expires_at: Instant | None = None,
        camera_id: CameraId | None = None,
        object_id: ObjectId | None = None,
        observation_id: ObservationId | None = None,
    ) -> BlobRef:
        """Store bytes under their hash (E1).

        ``RetentionMode.NEVER_PERSIST`` must be honoured by **not storing** and
        returning a reference that resolves to ``NOT_FOUND`` (E3). Silently
        storing it anyway would break a privacy guarantee a deployment relies on.

        Raises:
            EvidenceStoreError: the write failed. Never a silent partial success.
        """
        ...

    def get(self, blob_ref: BlobRef, *, tenant_id: TenantId) -> EvidenceFetch:
        """Retrieve bytes, or say precisely why not (E2, E7).

        Never raises for an absent blob: absence is an *answer*, and the three
        kinds of absence mean different things to whoever asked.
        """
        ...

    def exists(self, content_hash: str, *, tenant_id: TenantId) -> bool:
        """Deduplication check (E1). Cheap; never reads the payload."""
        ...

    def expire(self, *, before: Instant, tenant_id: TenantId | None = None) -> ExpireReport:
        """Run retention. Returns what it removed."""
        ...

    def erase(self, scope: EraseScope) -> EraseReport:
        """Right-to-erasure (E5). Removes blobs; leaves tombstones."""
        ...


@dataclass(frozen=True, slots=True)
class EvidenceTombstone:
    """The immutable record that evidence existed and was erased.

    07_STATE §8.2's resolution of the tension between V5 (observations are
    immutable) and regulation (a subject may demand erasure). This record is what
    survives, and it is why erasure does not make the log untrustworthy.
    """

    content_hash: str
    tenant_id: TenantId
    erased_at: Instant
    authority: str
    original_size_bytes: int = 0
    object_id: ObjectId | None = None
    camera_id: CameraId | None = None

    def __post_init__(self) -> None:
        if not self.authority:
            raise ValueError(
                "a tombstone must name the authority for the erasure; 07_STATE "
                "§8.2 requires the record say 'by whom and under what authority', "
                "and an anonymous erasure is not an audit trail"
            )


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """How long each retention mode keeps evidence (07_STATE §8.1).

    Configurable because §8.1 gives ranges rather than constants — 24–72 hours for
    crops, 7 days for raw model output — and a deployment's answer depends on its
    regulator, not on the platform.
    """

    evidence_ttl_ms: int = 172_800_000
    """48 hours, the middle of §8.1's 24–72 hour range for crops."""

    raw_output_ttl_ms: int = 604_800_000
    """7 days (§8.1). Longer than imagery because it is text, not pixels."""

    ephemeral_ttl_ms: int = 0
    """Not retained at all past the inference that needed it."""

    def ttl_for(self, mode: RetentionMode) -> int:
        return {
            RetentionMode.EPHEMERAL: self.ephemeral_ttl_ms,
            RetentionMode.EVIDENCE: self.evidence_ttl_ms,
            RetentionMode.NEVER_PERSIST: 0,
        }[mode]

    def expires_at(self, mode: RetentionMode, stored_at: Instant) -> Instant | None:
        ttl = self.ttl_for(mode)
        return Instant(stored_at.ns + ttl * 1_000_000) if ttl > 0 else None


def hash_payload(payload: bytes) -> str:
    """Content address for a blob.

    SHA-256 because §M13 requires ``ArtifactStore.fetch`` verify content hash and
    fail closed; using one algorithm across the storage contracts means one
    verification path to review.
    """
    import hashlib

    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceQuota:
    """Bounds on a store, so a disk cannot fill silently.

    10_RELIABILITY's posture applied to storage: an unbounded evidence store is a
    memory leak with a delayed fuse, and the fuse burns for exactly as long as the
    disk has space.
    """

    max_bytes: int = 4 * 1024 * 1024 * 1024
    max_blobs: int = 100_000
    max_blob_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_blobs", "max_blob_bytes"):
            if getattr(self, name) < 1:
                raise ValueError(f"EvidenceQuota.{name} must be positive")


@dataclass(frozen=True, slots=True)
class StoreHealth:
    """What an operator needs to know about a storage adapter.

    ``oldest_blob`` is here because a store whose oldest blob predates its
    retention window is a store whose expiry is not running — a failure that is
    otherwise invisible until capacity is reached.
    """

    store_id: str
    blobs: int = 0
    bytes_used: int = 0
    oldest_blob: Instant | None = None
    tombstones: int = 0
    quota: EvidenceQuota = field(default_factory=EvidenceQuota)

    @property
    def utilization(self) -> float:
        return self.bytes_used / self.quota.max_bytes if self.quota.max_bytes else 0.0

    @property
    def near_capacity(self) -> bool:
        return self.utilization >= 0.9


def erase_scope_matches(record: EvidenceRecord, scope: EraseScope) -> bool:
    """Whether one record falls inside an erasure scope.

    Tenant is checked **first and always**: 12_SECURITY §4.1 makes tenant a
    property of identity rather than a filter, and an erasure that crossed the
    boundary would delete another customer's evidence.
    """
    if record.tenant_id != scope.tenant_id:
        return False
    if scope.object_ids and record.object_id not in scope.object_ids:
        return False
    if scope.camera_ids and record.camera_id not in scope.camera_ids:
        return False
    if scope.before is not None and record.stored_at.ns >= scope.before.ns:
        return False
    return not (scope.after is not None and record.stored_at.ns <= scope.after.ns)


__all__ = [
    "EraseReport",
    "EraseScope",
    "EvidenceFetch",
    "EvidenceQuota",
    "EvidenceRecord",
    "EvidenceStatus",
    "EvidenceStorePort",
    "EvidenceTombstone",
    "ExpireReport",
    "RetentionPolicy",
    "StoreHealth",
    "erase_scope_matches",
    "hash_payload",
]


def blob_hash_of(blob_ref: BlobRef) -> str:
    """Extract the content hash from a reference.

    References carry an ``sha256:`` prefix so that a future algorithm change is
    visible in the data rather than inferred from length.
    """
    text = str(blob_ref)
    return text.split(":", 1)[1] if ":" in text else text


def all_scopes(scope: EraseScope) -> Sequence[str]:
    """The scope's dimensions, for the audit record.

    An erasure's audit entry must say what was asked for, not only what was
    removed — a request that matched nothing is still a request somebody made.
    """
    parts: list[str] = [f"tenant={scope.tenant_id}"]
    if scope.object_ids:
        parts.append(f"objects={len(scope.object_ids)}")
    if scope.camera_ids:
        parts.append(f"cameras={len(scope.camera_ids)}")
    if scope.before is not None:
        parts.append(f"before={scope.before.ns}")
    if scope.after is not None:
        parts.append(f"after={scope.after.ns}")
    return tuple(parts)
