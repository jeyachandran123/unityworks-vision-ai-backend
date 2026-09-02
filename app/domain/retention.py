"""Retention: four categories, four clocks.

One global "delete after N days" would be wrong for every category at once. So
each is swept on its own schedule and by its own rule:

* **Evidence** — imagery of identifiable people. Shortest life, and it *expires*
  (stops being served) before it is *erased*. Two steps, because "no longer
  visible" and "no longer recoverable" are different promises and an operator
  should be able to see the first before the second happens.
* **Incidents** — the compliance record. Outlives the imagery it cites; a
  resolved incident is still the evidence that the finding was handled. Only
  RESOLVED incidents are ever pruned: an incident still open is not old, it is
  neglected, and deleting it would hide that.
* **Audit** — who looked at what. Longest life of the three, because it is the
  record of access to the other two.
* **Observations** — the platform's durable log of what a camera read. Swept by
  ``truncate``, which the ``ObservationLogPort`` contract defines as existing
  *"for retention alone"* and which removes only a time-bounded prefix. Not an
  application table: the application stores no perception result, so this sweep
  reaches into Vision OS's own store through its own port rather than deleting
  a row of its own.

### Marking is not erasing

`sweep()` marks expiry always; it erases only when the deployment has enabled
`RETENTION_SWEEP_ENABLED`. A process that starts should not begin deleting data
because it started. Deletion begins because somebody decided it should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.models import (
    AuditEvent,
    Camera,
    EvidenceRecord,
    EvidenceState,
    Incident,
    IncidentStatus,
)


@dataclass(slots=True)
class RetentionReport:
    """What a sweep did. Reported, never merely logged and forgotten."""

    evidence_expired: int = 0
    evidence_erased: int = 0
    incidents_pruned: int = 0
    audit_pruned: int = 0
    observations_truncated: int = 0
    #: Camera partitions the sweep could not truncate. Reported for the same
    #: reason `erase_failures` is: a retention promise that quietly failed is
    #: worse than one that was never made.
    observation_failures: list[str] = field(default_factory=list)
    #: Evidence the sweeper could not erase. A non-empty list is a finding: the
    #: database says gone and the filesystem disagrees.
    erase_failures: list[str] = field(default_factory=list)
    dry_run: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_expired": self.evidence_expired,
            "evidence_erased": self.evidence_erased,
            "incidents_pruned": self.incidents_pruned,
            "audit_pruned": self.audit_pruned,
            "observations_truncated": self.observations_truncated,
            "observation_failures": self.observation_failures,
            "erase_failures": self.erase_failures,
            "dry_run": self.dry_run,
        }


class RetentionService:
    """Applies the four policies. Idempotent; safe to run on every start."""

    __slots__ = (
        "_audit_days",
        "_evidence_days",
        "_incident_days",
        "_observation_days",
        "_observation_log",
        "_root",
        "_session",
    )

    def __init__(
        self,
        session: AsyncSession,
        *,
        root: Path | str,
        evidence_days: int,
        incident_days: int,
        audit_days: int,
        observation_days: int = 0,
        observation_log: Any = None,
    ) -> None:
        self._session = session
        self._root = Path(root)
        self._evidence_days = evidence_days
        self._incident_days = incident_days
        self._audit_days = audit_days
        self._observation_days = observation_days
        #: An `ObservationLogPort`, or `None` when synthesis is not assembled in
        #: this process. `None` is not a failure: there is no log to sweep, and
        #: the sweep says so rather than reporting a zero that reads as "swept".
        self._observation_log = observation_log

    async def sweep(self, *, erase: bool) -> RetentionReport:
        """Expire, then optionally erase and prune.

        Args:
            erase: When false, only marks — nothing is destroyed. This is the
                default posture, and the only safe one for a process that has
                just started and does not yet know whether its configuration is
                what the deployment intended.
        """
        report = RetentionReport(dry_run=not erase)

        report.evidence_expired = await self._expire_evidence()
        if not erase:
            logger.info(
                "retention sweep (mark only): {} evidence records expired; " "erasure is disabled",
                report.evidence_expired,
            )
            return report

        report.evidence_erased, report.erase_failures = await self._erase_expired()
        report.incidents_pruned = await self._prune_incidents()
        report.audit_pruned = await self._prune_audit()
        (
            report.observations_truncated,
            report.observation_failures,
        ) = await self._truncate_observations()

        logger.info("retention sweep complete: {}", report.as_dict())
        if report.erase_failures:
            logger.error(
                "{} evidence files could not be erased; the database and the " "store disagree",
                len(report.erase_failures),
            )
        if report.observation_failures:
            logger.error(
                "{} camera partitions could not be truncated; observations past "
                "their retention date are still on disk",
                len(report.observation_failures),
            )
        return report

    # -- evidence -------------------------------------------------------------

    async def _expire_evidence(self) -> int:
        """RETAINED to EXPIRED. Stops serving; keeps the bytes for now."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(EvidenceRecord).where(
                EvidenceRecord.state == EvidenceState.RETAINED.value,
                EvidenceRecord.expires_at.is_not(None),
                EvidenceRecord.expires_at <= now,
            )
        )
        count = 0
        trail = AuditTrail(self._session)
        for record in result.scalars().all():
            record.state = EvidenceState.EXPIRED.value
            count += 1
            await trail.record(
                action=AuditAction.EVIDENCE_EXPIRED,
                organization_id=record.organization_id,
                actor="retention",
                resource_type="evidence",
                resource_id=record.evidence_ref,
                detail={"expires_at": record.expires_at.isoformat()},
            )
        return count

    async def _erase_expired(self) -> tuple[int, list[str]]:
        """EXPIRED to DELETED. The bytes go; the tombstone stays.

        The row is never removed. A retention policy that leaves no trace of
        having been applied cannot be shown to have been applied.
        """
        result = await self._session.execute(
            select(EvidenceRecord).where(EvidenceRecord.state == EvidenceState.EXPIRED.value)
        )
        erased = 0
        failures: list[str] = []
        trail = AuditTrail(self._session)

        for record in result.scalars().all():
            if record.storage_ref:
                path = self._root / record.storage_ref
                try:
                    if path.is_file():
                        path.unlink()
                except OSError as exc:
                    # Left EXPIRED, not marked DELETED. A row claiming the bytes
                    # are gone while they sit on disk is the one state worse than
                    # a failed sweep.
                    logger.error(
                        "retention could not erase {}: {}: {}",
                        record.evidence_ref,
                        type(exc).__name__,
                        exc,
                    )
                    failures.append(record.evidence_ref)
                    await trail.record(
                        action=AuditAction.EVIDENCE_DELETED,
                        organization_id=record.organization_id,
                        actor="retention",
                        resource_type="evidence",
                        resource_id=record.evidence_ref,
                        outcome=AuditOutcome.FAILED,
                        detail={"error": type(exc).__name__},
                    )
                    continue

            record.state = EvidenceState.DELETED.value
            record.deleted_at = datetime.now(UTC)
            record.deleted_by = "retention"
            record.deletion_reason = f"retention: {self._evidence_days} days"
            record.storage_ref = ""
            record.size_bytes = 0
            erased += 1
            await trail.record(
                action=AuditAction.EVIDENCE_DELETED,
                organization_id=record.organization_id,
                actor="retention",
                resource_type="evidence",
                resource_id=record.evidence_ref,
                detail={"reason": "retention"},
            )
        return erased, failures

    # -- incidents ------------------------------------------------------------

    async def _prune_incidents(self) -> int:
        """Remove aged **resolved** incidents only.

        An unresolved incident is never pruned no matter how old. Age is not
        closure, and a violation nobody dealt with must not disappear because a
        year passed.
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._incident_days)
        result = await self._session.execute(
            delete(Incident).where(
                Incident.status == IncidentStatus.RESOLVED.value,
                Incident.resolved_at.is_not(None),
                Incident.resolved_at <= cutoff,
            )
        )
        return int(result.rowcount or 0)

    # -- audit ----------------------------------------------------------------

    async def _prune_audit(self) -> int:
        """Delete whole aged rows. Never edit a surviving one.

        Pruning by age is a retention policy. Editing a row that remains would be
        tampering, and the module that writes the trail deliberately offers no
        way to do it.
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._audit_days)
        result = await self._session.execute(
            delete(AuditEvent).where(AuditEvent.occurred_at <= cutoff)
        )
        return int(result.rowcount or 0)


    # -- observations ---------------------------------------------------------

    async def _truncate_observations(self) -> tuple[int, list[str]]:
        """Remove observations older than the retention window, per camera.

        ### Why this reaches into Vision OS rather than deleting a row

        There is no application table holding a perception result — `models.py`
        explains at length why there must never be one — so the record that
        needs a retention clock lives in the platform's own log. `P20` provides
        exactly one way to shorten it: `truncate(partition, before)`, which its
        own contract describes as existing *"for retention alone"* and which
        removes only a time-bounded prefix. Nothing here edits or rewrites an
        observation; the log stays append-only, and this only moves its floor.

        ### Partitions come from the camera table, not from the store

        The log partitions by `CameraId`, and asking the store which partitions
        exist would sweep whatever happens to be on disk — including a directory
        left behind by a camera that was deleted, which nothing would then
        attribute to an organisation. Reading the roster instead means every
        truncation belongs to a tenant and can be audited to one.

        A camera removed from the table therefore stops being swept, and its
        observations outlive their retention. That is a real gap, stated in
        `docs/architecture/NOT_YET_CONNECTED.md` rather than papered over here:
        the fix belongs at the delete, not in a sweep that would have to guess
        which orphaned directories were once cameras.
        """
        if self._observation_log is None:
            logger.info(
                "observation retention skipped: synthesis is not assembled in "
                "this process, so there is no log to sweep"
            )
            return 0, []
        if self._observation_days <= 0:
            logger.warning(
                "observation retention skipped: observation_retention_days is "
                "{}, so the log has no expiry at all",
                self._observation_days,
            )
            return 0, []

        from vision_os.core.model.ids import CameraId
        from vision_os.core.model.timebase import Instant

        cutoff = datetime.now(UTC) - timedelta(days=self._observation_days)
        before = Instant(int(cutoff.timestamp() * 1_000_000_000))

        rows = (
            await self._session.execute(select(Camera.organization_id, Camera.camera_key))
        ).all()

        removed = 0
        failures: list[str] = []
        per_org: dict[str, int] = {}

        for organization_id, camera_key in rows:
            try:
                count = int(self._observation_log.truncate(CameraId(camera_key), before))
            except Exception as exc:  # noqa: BLE001 - one bad partition must not stop the rest
                logger.error(
                    "observation retention could not truncate {}: {}: {}",
                    camera_key,
                    type(exc).__name__,
                    exc,
                )
                failures.append(camera_key)
                continue
            removed += count
            if count:
                per_org[organization_id] = per_org.get(organization_id, 0) + count

        trail = AuditTrail(self._session)
        for organization_id, count in per_org.items():
            await trail.record(
                action=AuditAction.OBSERVATIONS_TRUNCATED,
                organization_id=organization_id,
                actor="retention",
                resource_type="observation",
                # A count and a cutoff. Never an observation and never an
                # attribute value: an audit row that copied what it deleted
                # would defeat the deletion.
                detail={"before": cutoff.isoformat(), "removed": count},
            )

        return removed, failures


__all__ = ["RetentionReport", "RetentionService"]
