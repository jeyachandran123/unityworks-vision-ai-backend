"""Retention: three categories, three clocks.

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

### Marking is not erasing

`sweep()` marks expiry always; it erases only when the deployment has enabled
`RETENTION_SWEEP_ENABLED`. A process that starts should not begin deleting data
because it started. Deletion begins because somebody decided it should.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.audit import AuditAction, AuditOutcome, AuditTrail
from app.domain.models import (
    AuditEvent,
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
            "erase_failures": self.erase_failures,
            "dry_run": self.dry_run,
        }


class RetentionService:
    """Applies the three policies. Idempotent; safe to run on every start."""

    __slots__ = ("_audit_days", "_evidence_days", "_incident_days", "_root", "_session")

    def __init__(
        self,
        session: AsyncSession,
        *,
        root: Path | str,
        evidence_days: int,
        incident_days: int,
        audit_days: int,
    ) -> None:
        self._session = session
        self._root = Path(root)
        self._evidence_days = evidence_days
        self._incident_days = incident_days
        self._audit_days = audit_days

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

        logger.info("retention sweep complete: {}", report.as_dict())
        if report.erase_failures:
            logger.error(
                "{} evidence files could not be erased; the database and the " "store disagree",
                len(report.erase_failures),
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


__all__ = ["RetentionReport", "RetentionService"]
