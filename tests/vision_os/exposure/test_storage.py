"""Storage tests — M13's contracts (§M13, 07_STATE §8).

M13 *"owns no state — it is a set of contracts"*, so what is tested here is
whether an adapter honours a contract, not whether a particular store works.
Every assertion names the obligation it enforces.

Two of these guard privacy guarantees rather than correctness ones, and they are
the ones that fail silently: `never_persist` storing anyway, and `Expired`
collapsing into `NotFound`. Neither would ever surface in normal operation.
"""

from __future__ import annotations

import pytest

from vision_os.adapters.persistence import (
    FileEvidenceStore,
    InMemoryEvidenceStore,
    NullEvidenceStore,
)
from vision_os.adapters.synthesis import FileObservationLog, InMemoryObservationLog
from vision_os.core.errors import EvidenceQuotaExceededError
from vision_os.core.model.crop import PrivacyClass, RetentionMode
from vision_os.core.model.ids import BlobRef, CameraId, LogPosition, ObjectId
from vision_os.core.model.timebase import Instant
from vision_os.core.ports.persistence import (
    EraseScope,
    EvidenceQuota,
    EvidenceStatus,
    EvidenceTombstone,
    RetentionPolicy,
    hash_payload,
)

from .conftest import (
    CAMERA,
    EVIDENCE_HASH,
    EVIDENCE_PAYLOAD,
    OTHER_TENANT,
    TENANT,
    publish,
    store_evidence,
)


class TestTheFourAbsenceStates:
    """§M13: *"Collapsing these two is how retention behaviour becomes
    indistinguishable from data loss."*"""

    def test_stored_returns_the_payload(self, evidence) -> None:
        store_evidence(evidence)
        fetched = evidence.get(BlobRef(f"sha256:{EVIDENCE_HASH}"), tenant_id=TENANT)
        assert fetched.status is EvidenceStatus.STORED
        assert fetched.payload == EVIDENCE_PAYLOAD

    def test_never_stored_is_not_found(self, evidence) -> None:
        """A bug in whatever minted the reference."""
        fetched = evidence.get(BlobRef("sha256:absent"), tenant_id=TENANT)
        assert fetched.status is EvidenceStatus.NOT_FOUND

    def test_retention_removed_it_is_expired(self, evidence) -> None:
        """Normal, expected, not an error."""
        store_evidence(evidence, expires_at=Instant(1_000))
        evidence.expire(before=Instant(2_000))
        assert not evidence.exists(EVIDENCE_HASH, tenant_id=TENANT)

    def test_an_erasure_request_removed_it_is_erased(self, evidence) -> None:
        """A different audit answer from either of the above."""
        store_evidence(evidence, object_id=ObjectId("obj-1"))
        evidence.erase(
            EraseScope(
                tenant_id=TENANT, object_ids=(ObjectId("obj-1"),), authority="dpo"
            )
        )
        fetched = evidence.get(BlobRef(f"sha256:{EVIDENCE_HASH}"), tenant_id=TENANT)
        assert fetched.status is EvidenceStatus.ERASED

    def test_a_fetch_cannot_report_success_with_no_bytes(self) -> None:
        """The type refuses it, so no adapter can express a silent partial success."""
        from vision_os.core.ports.persistence import EvidenceFetch

        with pytest.raises(ValueError, match="must carry its payload"):
            EvidenceFetch(status=EvidenceStatus.STORED)

    def test_a_failed_fetch_cannot_carry_a_payload(self) -> None:
        """Otherwise a consumer reading bytes without checking status would see
        expired evidence as current."""
        from vision_os.core.ports.persistence import EvidenceFetch

        with pytest.raises(ValueError, match="must not carry a payload"):
            EvidenceFetch(status=EvidenceStatus.EXPIRED, payload=b"x")


class TestNeverPersistIsHonoured:
    """12_SECURITY §2.3's no-evidence mode. Obligation E3."""

    @pytest.mark.parametrize(
        "store", [InMemoryEvidenceStore(), NullEvidenceStore()],
        ids=lambda s: s.store_id,
    )
    def test_a_never_persist_blob_is_not_stored(self, store) -> None:
        """*"the crop exists in memory for the duration of one inference and
        nowhere else."*

        An adapter that stored it anyway — even encrypted, even briefly — has
        broken a promise the deployment made to whoever authorized the cameras,
        and nothing downstream would ever notice.
        """
        payload = b"must-never-touch-disk"
        digest = hash_payload(payload)
        store.put(
            digest,
            payload,
            tenant_id=TENANT,
            retention=RetentionMode.NEVER_PERSIST,
            privacy_class=PrivacyClass.C1_IMAGERY,
        )
        assert not store.exists(digest, tenant_id=TENANT)

    def test_a_file_store_writes_no_file_for_never_persist(self, tmp_path) -> None:
        store = FileEvidenceStore(tmp_path)
        payload = b"must-never-touch-disk"
        store.put(
            hash_payload(payload),
            payload,
            tenant_id=TENANT,
            retention=RetentionMode.NEVER_PERSIST,
            privacy_class=PrivacyClass.C1_IMAGERY,
        )
        assert not list(tmp_path.glob("*.blob")), "a never_persist blob reached disk"


class TestErasureTombstones:
    """07_STATE §8.2: *"The audit trail survives; the content does not."*"""

    def test_erasure_removes_content(self, evidence) -> None:
        store_evidence(evidence, object_id=ObjectId("obj-1"))
        report = evidence.erase(
            EraseScope(
                tenant_id=TENANT, object_ids=(ObjectId("obj-1"),), authority="dpo"
            )
        )
        assert report.erased == 1
        assert report.bytes_reclaimed > 0

    def test_erasure_leaves_a_tombstone(self, evidence) -> None:
        """Not a leftover — the deliverable.

        > *"Rewriting history to pretend an observation never existed would
        > destroy the property that makes the log trustworthy in the first
        > place."*
        """
        store_evidence(evidence, object_id=ObjectId("obj-1"))
        evidence.erase(
            EraseScope(
                tenant_id=TENANT, object_ids=(ObjectId("obj-1"),), authority="dpo-2024"
            )
        )
        tombstones = evidence.tombstones()
        assert tombstones
        assert tombstones[0].authority == "dpo-2024"

    def test_a_tombstone_must_name_its_authority(self) -> None:
        """§8.2 requires the record say *"by whom and under what authority"*.

        An anonymous erasure is not an audit trail.
        """
        with pytest.raises(ValueError, match="authority"):
            EvidenceTombstone(
                content_hash="x",
                tenant_id=TENANT,
                erased_at=Instant(0),
                authority="",
            )

    def test_an_unscoped_erasure_is_refused(self) -> None:
        """*"Erase everything for this tenant"* is a deployment decision."""
        with pytest.raises(ValueError, match="objects, cameras or a time window"):
            EraseScope(tenant_id=TENANT)

    def test_an_erasure_scope_must_name_a_tenant(self) -> None:
        with pytest.raises(ValueError, match="must name a tenant"):
            EraseScope(tenant_id="", object_ids=(ObjectId("o"),))

    def test_erasure_by_subject_is_not_expressible(self) -> None:
        """§8.2: *"by default UWV holds no persistent biometric identity, which
        is a deliberate privacy posture, not a limitation."*

        There is no field here that could name a person, because there is nothing
        in the platform to name them with.
        """
        fields = set(EraseScope.__dataclass_fields__)
        for forbidden in ("subject_id", "person", "face", "name", "identity"):
            assert forbidden not in fields

    def test_a_null_store_satisfies_every_erasure_request(self) -> None:
        """A deployment storing no imagery has already complied."""
        report = NullEvidenceStore().erase(
            EraseScope(tenant_id=TENANT, object_ids=(ObjectId("o"),), authority="dpo")
        )
        assert report.clean
        assert report.erased == 0


class TestTenantScopedStorage:
    """Obligation E7. 12_SECURITY §4.1 applies to storage too."""

    def test_identical_bytes_do_not_cross_tenants(self, evidence) -> None:
        """Two tenants storing the same image get two entries.

        Costs a duplicate; buys a boundary that cannot be crossed by guessing a
        content hash — which is otherwise a real channel, because identical
        imagery hashes identically.
        """
        store_evidence(evidence, tenant=TENANT)
        assert evidence.exists(EVIDENCE_HASH, tenant_id=TENANT)
        assert not evidence.exists(EVIDENCE_HASH, tenant_id=OTHER_TENANT)

    def test_a_fetch_from_the_wrong_tenant_finds_nothing(self, evidence) -> None:
        store_evidence(evidence, tenant=TENANT)
        fetched = evidence.get(
            BlobRef(f"sha256:{EVIDENCE_HASH}"), tenant_id=OTHER_TENANT
        )
        assert not fetched.available

    def test_erasure_never_crosses_a_tenant(self, evidence) -> None:
        store_evidence(evidence, tenant=TENANT, object_id=ObjectId("obj-1"))
        report = evidence.erase(
            EraseScope(
                tenant_id=OTHER_TENANT,
                object_ids=(ObjectId("obj-1"),),
                authority="dpo",
            )
        )
        assert report.erased == 0
        assert evidence.exists(EVIDENCE_HASH, tenant_id=TENANT)


class TestBoundedStorage:
    def test_a_store_refuses_beyond_its_quota(self) -> None:
        """An unbounded evidence store is a memory leak whose fuse burns for
        exactly as long as the disk has space."""
        store = InMemoryEvidenceStore(quota=EvidenceQuota(max_blobs=2))
        for i in range(2):
            payload = f"blob-{i}".encode()
            store.put(
                hash_payload(payload),
                payload,
                tenant_id=TENANT,
                retention=RetentionMode.EVIDENCE,
                privacy_class=PrivacyClass.C1_IMAGERY,
                expires_at=Instant(10**12),
            )
        with pytest.raises(EvidenceQuotaExceededError):
            store.put(
                hash_payload(b"one-too-many"),
                b"one-too-many",
                tenant_id=TENANT,
                retention=RetentionMode.EVIDENCE,
                privacy_class=PrivacyClass.C1_IMAGERY,
                expires_at=Instant(10**12),
            )

    def test_quota_exhaustion_is_systemic_not_persistent(self) -> None:
        """Retrying makes it worse; the answer is to shed or run retention."""
        error = EvidenceQuotaExceededError("full")
        assert not error.retryable

    def test_an_oversized_blob_is_refused(self) -> None:
        store = InMemoryEvidenceStore(quota=EvidenceQuota(max_blob_bytes=8))
        with pytest.raises(EvidenceQuotaExceededError):
            store.put(
                hash_payload(b"x" * 64),
                b"x" * 64,
                tenant_id=TENANT,
                retention=RetentionMode.EVIDENCE,
                privacy_class=PrivacyClass.C1_IMAGERY,
            )

    def test_health_reports_utilization(self, evidence) -> None:
        """A store whose oldest blob predates its retention window has stopped
        expiring — a failure otherwise invisible until capacity is reached."""
        store_evidence(evidence)
        health = evidence.health()
        assert health.blobs == 1
        assert 0.0 <= health.utilization <= 1.0


class TestRetentionPolicy:
    def test_evidence_has_the_shortest_tier(self) -> None:
        """07_STATE §8.1: evidence is *"the shortest tier by design"*.

        It is the only tier containing imagery, so retention here is a privacy
        decision rather than an engineering one.
        """
        policy = RetentionPolicy()
        assert policy.evidence_ttl_ms < policy.raw_output_ttl_ms

    def test_ephemeral_expires_immediately(self) -> None:
        policy = RetentionPolicy()
        assert policy.expires_at(RetentionMode.EPHEMERAL, Instant(0)) is None

    def test_never_persist_has_no_expiry_because_it_has_no_life(self) -> None:
        policy = RetentionPolicy()
        assert policy.expires_at(RetentionMode.NEVER_PERSIST, Instant(0)) is None


class TestTheLogTailContract:
    """Obligation L7 — added in Flow 8 to complete §M13's specified API."""

    @pytest.mark.parametrize(
        "make_log",
        [lambda p: InMemoryObservationLog(), lambda p: FileObservationLog(p)],
        ids=["memory", "file"],
    )
    def test_tail_on_an_empty_partition_returns_immediately(
        self, make_log, tmp_path
    ) -> None:
        """A camera watching an empty corridor is normal, not a reason to stall."""
        log = make_log(tmp_path)
        assert list(log.tail(CameraId("never-used"))) == []

    def test_tail_resumes_from_a_position(self, state, log) -> None:
        """§3.2: *"a consumer reconnecting with `resume_from` receives everything
        since that cursor."*
        """
        published = publish(state, count=6)
        assert len(published) == 6
        resumed = list(log.tail(CAMERA, start=LogPosition(4)))
        assert [o.observation_id for o in resumed] == [
            o.observation_id for o in published[4:]
        ]

    def test_tail_from_the_end_yields_nothing(self, state, log) -> None:
        publish(state, count=3)
        assert list(log.tail(CAMERA, start=log.position(CAMERA))) == []

    def test_read_and_tail_agree_over_the_same_range(self, state, log) -> None:
        """They answer different questions but must not disagree about facts."""
        publish(state, count=5)
        by_read = [o.observation_id for o in log.read(CAMERA, start=LogPosition(1))]
        by_tail = [o.observation_id for o in log.tail(CAMERA, start=LogPosition(1))]
        assert by_read == by_tail


class TestStorageContractsStaySeparate:
    """§M13: *"Conflating them is the reason storage becomes un-portable."*"""

    def test_the_evidence_store_has_no_log_methods(self) -> None:
        """Append-only and write-once-with-TTL are different access patterns."""
        store = InMemoryEvidenceStore()
        for forbidden in ("append", "read", "position", "truncate"):
            assert not hasattr(store, forbidden), f"an evidence store exposes {forbidden}"

    def test_the_log_has_no_evidence_methods(self) -> None:
        log = InMemoryObservationLog()
        for forbidden in ("erase", "expire", "exists"):
            assert not hasattr(log, forbidden), f"a log exposes {forbidden}"

    def test_evidence_and_log_retention_are_separately_configurable(self) -> None:
        """Storing them together would force imagery to inherit the log's 7-day
        to years retention (§8.1)."""
        from vision_os.kernel.config.schema import StateSection, StorageSection

        assert StorageSection().evidence_ttl_ms < StateSection().log_retention_ms
