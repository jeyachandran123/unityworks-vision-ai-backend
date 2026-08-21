"""P22 adapters — content-addressed evidence storage.

Three behaviours here are not optimizations but contract:

**Deduplication (E1).** The key is the hash of the bytes, so the same crop stored
twice occupies one slot. §M13 Performance: *"content-addressed with
deduplication."*

**`NEVER_PERSIST` stores nothing (E3).** 12_SECURITY §2.3's no-evidence mode. A
store that wrote it anyway would silently void a deployment's privacy posture, and
nothing downstream would ever notice.

**Erasure tombstones (E5).** 07_STATE §8.2 resolves the tension between V5 and
right-to-erasure by removing content while keeping the record that content
existed and was erased. *"The audit trail survives; the content does not."*
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path

from ...core.errors import EvidenceQuotaExceededError, EvidenceStoreError
from ...core.model.crop import PrivacyClass, RetentionMode
from ...core.model.ids import BlobRef, CameraId, ObjectId, ObservationId, TenantId
from ...core.model.timebase import Instant
from ...core.ports.persistence import (
    EraseReport,
    EraseScope,
    EvidenceFetch,
    EvidenceQuota,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceTombstone,
    ExpireReport,
    StoreHealth,
    blob_hash_of,
    erase_scope_matches,
)


class InMemoryEvidenceStore:
    """``evidence.memory`` — volatile, bounded, and honest about both.

    The default for an embedded deployment that wants evidence available for the
    life of the process and no longer. Its id says ``memory``, so a site claiming
    72-hour retention that finds this bound has a configuration bug it can see.
    """

    __slots__ = ("_blobs", "_lock", "_quota", "_records", "_tombstones")

    def __init__(self, *, quota: EvidenceQuota | None = None) -> None:
        self._quota = quota or EvidenceQuota()
        self._blobs: dict[str, bytes] = {}
        self._records: dict[str, EvidenceRecord] = {}
        self._tombstones: dict[str, EvidenceTombstone] = {}
        self._lock = threading.Lock()

    @property
    def store_id(self) -> str:
        return "evidence.memory"

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
        reference = BlobRef(f"sha256:{content_hash}")

        if retention is RetentionMode.NEVER_PERSIST:
            # E3. Not "store and expire immediately" — never written at all.
            # 12_SECURITY §2.3: the crop exists in memory for one inference and
            # nowhere else.
            return reference

        if len(payload) > self._quota.max_blob_bytes:
            raise EvidenceQuotaExceededError(
                f"blob of {len(payload)} bytes exceeds the "
                f"{self._quota.max_blob_bytes} byte limit",
                size=len(payload),
            )

        with self._lock:
            if self._scoped_key(content_hash, tenant_id) in self._records:
                return reference  # E1 — already held, one copy is enough.

            if len(self._records) >= self._quota.max_blobs:
                raise EvidenceQuotaExceededError(
                    f"store holds {len(self._records)} blobs, at its "
                    f"{self._quota.max_blobs} limit",
                    blobs=len(self._records),
                )
            used = sum(r.size_bytes for r in self._records.values())
            if used + len(payload) > self._quota.max_bytes:
                raise EvidenceQuotaExceededError(
                    f"store at {used} bytes cannot accept {len(payload)} more",
                    bytes_used=used,
                )

            key = self._scoped_key(content_hash, tenant_id)
            self._blobs[key] = payload
            self._records[key] = EvidenceRecord(
                content_hash=content_hash,
                tenant_id=tenant_id,
                privacy_class=privacy_class,
                retention=retention,
                stored_at=expires_at or Instant(0),
                expires_at=expires_at,
                size_bytes=len(payload),
                camera_id=camera_id,
                object_id=object_id,
                observation_id=observation_id,
            )
        return reference

    def get(self, blob_ref: BlobRef, *, tenant_id: TenantId) -> EvidenceFetch:
        """Retrieve, or say precisely why not (E2, E7).

        The four outcomes are distinct all the way to the consumer. §M13:
        *"Collapsing these two is how retention behaviour becomes
        indistinguishable from data loss."*
        """
        content_hash = blob_hash_of(blob_ref)
        key = self._scoped_key(content_hash, tenant_id)
        with self._lock:
            if key in self._tombstones:
                return EvidenceFetch(
                    status=EvidenceStatus.ERASED,
                    detail="erased under an erasure request",
                )
            record = self._records.get(key)
            payload = self._blobs.get(key)

        if record is None or payload is None:
            return EvidenceFetch(
                status=EvidenceStatus.NOT_FOUND,
                detail="no blob was ever stored under this hash for this tenant",
            )
        return EvidenceFetch(
            status=EvidenceStatus.STORED, payload=payload, record=record
        )

    def exists(self, content_hash: str, *, tenant_id: TenantId) -> bool:
        with self._lock:
            return self._scoped_key(content_hash, tenant_id) in self._records

    def expire(
        self, *, before: Instant, tenant_id: TenantId | None = None
    ) -> ExpireReport:
        examined = removed = reclaimed = 0
        with self._lock:
            for key, record in list(self._records.items()):
                if tenant_id is not None and record.tenant_id != tenant_id:
                    continue
                examined += 1
                if record.expires_at is None or record.expires_at.ns > before.ns:
                    continue
                removed += 1
                reclaimed += record.size_bytes
                self._records.pop(key, None)
                self._blobs.pop(key, None)
        return ExpireReport(
            examined=examined, removed=removed, bytes_reclaimed=reclaimed
        )

    def erase(self, scope: EraseScope) -> EraseReport:
        """Remove content; keep the record that it existed (E5).

        The tombstone is the deliverable, not a leftover. Without it, an erasure
        and a silent data loss look identical in the audit trail six months
        later.
        """
        erased = reclaimed = 0
        tombstones: list[str] = []
        at = scope.before or Instant(0)

        with self._lock:
            for key, record in list(self._records.items()):
                if not erase_scope_matches(record, scope):
                    continue
                erased += 1
                reclaimed += record.size_bytes
                tombstones.append(record.content_hash)
                self._tombstones[key] = EvidenceTombstone(
                    content_hash=record.content_hash,
                    tenant_id=record.tenant_id,
                    erased_at=at,
                    authority=scope.authority or "unattributed",
                    original_size_bytes=record.size_bytes,
                    object_id=record.object_id,
                    camera_id=record.camera_id,
                )
                self._records.pop(key, None)
                self._blobs.pop(key, None)

        return EraseReport(
            erased=erased,
            bytes_reclaimed=reclaimed,
            tombstones=tuple(tombstones),
            authority=scope.authority,
            erased_at=at,
        )

    def health(self) -> StoreHealth:
        with self._lock:
            return StoreHealth(
                store_id=self.store_id,
                blobs=len(self._records),
                bytes_used=sum(r.size_bytes for r in self._records.values()),
                tombstones=len(self._tombstones),
                quota=self._quota,
            )

    def tombstones(self) -> tuple[EvidenceTombstone, ...]:
        with self._lock:
            return tuple(self._tombstones.values())

    @staticmethod
    def _scoped_key(content_hash: str, tenant_id: TenantId) -> str:
        """Every key includes the tenant (E7).

        12_SECURITY §4.1: *"Every cache key includes `tenant_id` — including crop
        and understanding caches."* Two tenants storing byte-identical imagery
        get two entries, which costs a duplicate and buys an isolation boundary
        that cannot be crossed by a lookup.
        """
        return f"{tenant_id}/{content_hash}"


class FileEvidenceStore(InMemoryEvidenceStore):
    """``evidence.file`` — content-addressed blobs on disk.

    Survives a restart, which is what makes 07_STATE §8.1's 24–72 hour retention
    tier meaningful. The index is rebuilt by scanning on first touch, deriving it
    from the records themselves rather than from a sidecar that could disagree.
    """

    __slots__ = ("_loaded", "_root")

    def __init__(self, root: Path | str, *, quota: EvidenceQuota | None = None) -> None:
        super().__init__(quota=quota)
        self._root = Path(root)
        self._loaded = False

    @property
    def store_id(self) -> str:
        return "evidence.file"

    def _blob_path(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return self._root / f"{safe}.blob"

    def _meta_path(self, key: str) -> Path:
        safe = key.replace("/", "__")
        return self._root / f"{safe}.json"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._root.exists():
            return
        for meta in self._root.glob("*.json"):
            try:
                record = json.loads(meta.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            key = f"{record['tenant_id']}/{record['content_hash']}"
            if record.get("tombstone"):
                self._tombstones[key] = EvidenceTombstone(
                    content_hash=record["content_hash"],
                    tenant_id=TenantId(record["tenant_id"]),
                    erased_at=Instant(record.get("erased_at_ns", 0)),
                    authority=record.get("authority") or "unattributed",
                    original_size_bytes=record.get("size_bytes", 0),
                )
                continue
            self._records[key] = EvidenceRecord(
                content_hash=record["content_hash"],
                tenant_id=TenantId(record["tenant_id"]),
                privacy_class=PrivacyClass(record["privacy_class"]),
                retention=RetentionMode(record["retention"]),
                stored_at=Instant(record.get("stored_at_ns", 0)),
                expires_at=(
                    Instant(record["expires_at_ns"])
                    if record.get("expires_at_ns")
                    else None
                ),
                size_bytes=record.get("size_bytes", 0),
                camera_id=CameraId(record["camera_id"]) if record.get("camera_id") else None,
                object_id=ObjectId(record["object_id"]) if record.get("object_id") else None,
                observation_id=(
                    ObservationId(record["observation_id"])
                    if record.get("observation_id")
                    else None
                ),
            )

    def put(self, content_hash: str, payload: bytes, **kwargs) -> BlobRef:
        self._ensure_loaded()
        reference = super().put(content_hash, payload, **kwargs)
        if kwargs.get("retention") is RetentionMode.NEVER_PERSIST:
            return reference

        key = self._scoped_key(content_hash, kwargs["tenant_id"])
        record = self._records.get(key)
        if record is None:
            return reference
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            self._blob_path(key).write_bytes(payload)
            self._meta_path(key).write_text(
                json.dumps(_encode_record(record)), encoding="utf-8"
            )
        except OSError as exc:
            raise EvidenceStoreError(
                f"cannot write evidence to {self._root}: {exc}",
                content_hash=content_hash,
            ) from exc
        return reference

    def get(self, blob_ref: BlobRef, *, tenant_id: TenantId) -> EvidenceFetch:
        self._ensure_loaded()
        key = self._scoped_key(blob_hash_of(blob_ref), tenant_id)
        if key not in self._blobs:
            path = self._blob_path(key)
            if path.exists():
                try:
                    self._blobs[key] = path.read_bytes()
                except OSError as exc:
                    raise EvidenceStoreError(
                        f"cannot read evidence at {path}: {exc}",
                        blob_ref=str(blob_ref),
                    ) from exc
        return super().get(blob_ref, tenant_id=tenant_id)

    def exists(self, content_hash: str, *, tenant_id: TenantId) -> bool:
        self._ensure_loaded()
        return super().exists(content_hash, tenant_id=tenant_id)

    def expire(self, *, before: Instant, tenant_id: TenantId | None = None) -> ExpireReport:
        self._ensure_loaded()
        removed = [
            key
            for key, record in self._records.items()
            if (tenant_id is None or record.tenant_id == tenant_id)
            and record.expires_at is not None
            and record.expires_at.ns <= before.ns
        ]
        report = super().expire(before=before, tenant_id=tenant_id)
        for key in removed:
            self._blob_path(key).unlink(missing_ok=True)
            self._meta_path(key).unlink(missing_ok=True)
        return report

    def erase(self, scope: EraseScope) -> EraseReport:
        self._ensure_loaded()
        doomed = [
            key
            for key, record in self._records.items()
            if erase_scope_matches(record, scope)
        ]
        report = super().erase(scope)
        for key in doomed:
            self._blob_path(key).unlink(missing_ok=True)
            tombstone = self._tombstones.get(key)
            if tombstone is not None:
                # The tombstone replaces the record on disk. The bytes go; the
                # fact that they existed and were erased stays (07_STATE §8.2).
                self._meta_path(key).write_text(
                    json.dumps(_encode_tombstone(tombstone)), encoding="utf-8"
                )
        return report


class NullEvidenceStore:
    """``evidence.null`` — accepts nothing, keeps nothing.

    The honest binding for a deployment operating in 12_SECURITY §2.3's
    no-evidence mode. Every ``get`` reports ``NOT_FOUND``, which is true: nothing
    was ever stored. Declaring this explicitly beats binding nothing, because
    *"no evidence store configured"* and *"a store that deliberately keeps
    nothing"* are different operational statements.
    """

    __slots__ = ()

    @property
    def store_id(self) -> str:
        return "evidence.null"

    def put(self, content_hash: str, payload: bytes, **kwargs) -> BlobRef:
        return BlobRef(f"sha256:{content_hash}")

    def get(self, blob_ref: BlobRef, *, tenant_id: TenantId) -> EvidenceFetch:
        return EvidenceFetch(
            status=EvidenceStatus.NOT_FOUND,
            detail="this deployment retains no evidence by policy",
        )

    def exists(self, content_hash: str, *, tenant_id: TenantId) -> bool:
        return False

    def expire(self, *, before: Instant, tenant_id: TenantId | None = None) -> ExpireReport:
        return ExpireReport()

    def erase(self, scope: EraseScope) -> EraseReport:
        """Nothing to erase, and that is a complete answer.

        Reported as a clean report with zero erasures rather than as a failure: a
        deployment that stores no imagery has already satisfied every erasure
        request it will ever receive.
        """
        return EraseReport(authority=scope.authority)

    def health(self) -> StoreHealth:
        return StoreHealth(store_id=self.store_id)


def _encode_record(record: EvidenceRecord) -> dict:
    return {
        "content_hash": record.content_hash,
        "tenant_id": str(record.tenant_id),
        "privacy_class": record.privacy_class.value,
        "retention": record.retention.value,
        "stored_at_ns": record.stored_at.ns,
        "expires_at_ns": record.expires_at.ns if record.expires_at else None,
        "size_bytes": record.size_bytes,
        "camera_id": str(record.camera_id) if record.camera_id else None,
        "object_id": str(record.object_id) if record.object_id else None,
        "observation_id": (
            str(record.observation_id) if record.observation_id else None
        ),
    }


def _encode_tombstone(tombstone: EvidenceTombstone) -> dict:
    return {
        "tombstone": True,
        "content_hash": tombstone.content_hash,
        "tenant_id": str(tombstone.tenant_id),
        "erased_at_ns": tombstone.erased_at.ns,
        "authority": tombstone.authority,
        "size_bytes": tombstone.original_size_bytes,
    }


#: Selectable by name. Closed: an unknown store is refused rather than defaulted,
#: because defaulting a store that holds imagery is how a deployment ends up
#: retaining more than it intended.
EVIDENCE_FACTORIES: Mapping[str, object] = {
    "evidence.memory": InMemoryEvidenceStore,
    "evidence.file": FileEvidenceStore,
    "evidence.null": NullEvidenceStore,
}
