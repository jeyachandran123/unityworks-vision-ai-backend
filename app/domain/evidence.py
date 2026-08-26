"""Durable evidence: metadata in the database, bytes on disk.

Replaces the in-memory store, which lost everything on restart — accidentally
private and completely unusable.

### The split, and why

Metadata is queryable and small; bytes are large and must never be dragged
through a query. An incident cites `evidence_ref` handles, not images, so listing
a hundred incidents reads kilobytes rather than gigabytes.

### The lifecycle is a state, never an absence

    RETAINED ──expiry──► EXPIRED ──erase──► DELETED (tombstone)

A deleted record keeps its row. That is deliberate: an erasure request has to be
*provable* afterwards, and a row that simply vanished proves nothing. The
tombstone records who erased it, when, and why, and the bytes are gone.

`EXPIRED` is refused for serving even while the bytes are still on disk waiting
for the sweeper. Retention is a promise about what is served, not only about what
is eventually erased.

### What this module never does

It never logs bytes, never returns a path a caller could read directly, and never
serves anything without the caller having already passed `VIEW_EVIDENCE` —
which is a different privilege from reading the observation the evidence
supports.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import EvidenceRecord, EvidenceState
from app.errors import EvidenceForbiddenError, NotFoundError


class EvidenceStore:
    """Durable evidence, tenant-scoped at every entry point."""

    __slots__ = ("_root", "_session")

    def __init__(self, session: AsyncSession, *, root: Path | str) -> None:
        self._session = session
        self._root = Path(root)

    # ── writing ──────────────────────────────────────────────────────────────

    async def put(
        self,
        *,
        organization_id: str,
        evidence_ref: str,
        camera_key: str,
        payload: bytes,
        captured_at: datetime,
        purpose: str,
        retention_days: int,
        frame_ref: str = "",
        object_id: str = "",
        observation_id: str = "",
        media_type: str = "image/jpeg",
        geometry: str = "",
    ) -> EvidenceRecord:
        """Store bytes and record what they are.

        Content-addressed: the same image stored twice occupies one file. The
        hash is also the integrity check — a byte that changed on disk no longer
        matches what the record says it is.
        """
        digest = hashlib.blake2b(payload, digest_size=32).hexdigest()

        # Fan out by hash prefix. A single directory with a million files is a
        # directory no filesystem enjoys listing.
        target = self._root / organization_id / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)

        record = EvidenceRecord(
            organization_id=organization_id,
            evidence_ref=evidence_ref,
            camera_key=camera_key,
            frame_ref=frame_ref,
            object_id=object_id,
            observation_id=observation_id,
            captured_at=captured_at,
            purpose=purpose,
            state=EvidenceState.RETAINED.value,
            expires_at=datetime.now(UTC) + timedelta(days=retention_days),
            # Relative, so moving the evidence root does not invalidate rows —
            # and so a stored value can never be an absolute path a caller might
            # be tempted to hand to a file server.
            storage_ref=str(target.relative_to(self._root)).replace("\\", "/"),
            content_hash=f"blake2b:{digest}",
            size_bytes=len(payload),
            media_type=media_type,
            geometry=geometry,
        )
        self._session.add(record)
        return record

    # ── reading ──────────────────────────────────────────────────────────────

    async def metadata(self, *, organization_id: str, evidence_ref: str) -> EvidenceRecord:
        """The record, tenant-scoped. Raises `NotFoundError` across tenants.

        Not `Forbidden`: telling a caller that evidence exists in another tenant
        is itself a disclosure.
        """
        result = await self._session.execute(
            select(EvidenceRecord).where(
                EvidenceRecord.organization_id == organization_id,
                EvidenceRecord.evidence_ref == evidence_ref,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise NotFoundError("no such evidence")
        return record

    async def fetch(
        self, *, organization_id: str, evidence_ref: str
    ) -> tuple[EvidenceRecord, bytes]:
        """The bytes, if this record may still be served.

        Authorization is the caller's job and has already happened; this enforces
        *lifecycle*, which authorization cannot know about.
        """
        record = await self.metadata(organization_id=organization_id, evidence_ref=evidence_ref)

        if record.state == EvidenceState.DELETED.value:
            raise NotFoundError("this evidence was deleted")

        if not record.servable:
            # Expired, and possibly still on disk. Refused anyway.
            raise EvidenceForbiddenError(
                "this evidence has passed its retention period and is no longer served",
                details={"state": record.state, "expired_at": _iso(record.expires_at)},
            )

        path = self._root / record.storage_ref
        if not path.is_file():
            # The row says retained and the bytes are gone. Reported, never
            # papered over: it means the store and the database disagree.
            logger.error(
                "evidence {} is RETAINED but its bytes are missing at {}",
                record.evidence_ref,
                record.storage_ref,
            )
            raise NotFoundError("the stored image is no longer available")

        payload = path.read_bytes()

        actual = f"blake2b:{hashlib.blake2b(payload, digest_size=32).hexdigest()}"
        if record.content_hash and actual != record.content_hash:
            logger.error("evidence {} failed its integrity check", record.evidence_ref)
            raise NotFoundError("the stored image failed its integrity check")

        return record, payload

    async def list_for_camera(
        self, *, organization_id: str, camera_key: str, limit: int = 50
    ) -> list[EvidenceRecord]:
        result = await self._session.execute(
            select(EvidenceRecord)
            .where(
                EvidenceRecord.organization_id == organization_id,
                EvidenceRecord.camera_key == camera_key,
            )
            .order_by(EvidenceRecord.captured_at.desc())
            .limit(min(max(limit, 1), 200))
        )
        return list(result.scalars().all())

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def expire_due(self, *, organization_id: str | None = None) -> list[str]:
        """Mark everything past its retention date as EXPIRED.

        Marks; does not erase. Separating the two means an expiry sweep is safe
        to run often and an erase is a deliberate second act — and it gives an
        operator a window in which "expired" is visible before "gone".
        """
        now = datetime.now(UTC)
        statement = select(EvidenceRecord).where(
            EvidenceRecord.state == EvidenceState.RETAINED.value,
            EvidenceRecord.expires_at.is_not(None),
            EvidenceRecord.expires_at <= now,
        )
        if organization_id:
            statement = statement.where(EvidenceRecord.organization_id == organization_id)

        expired = []
        for record in (await self._session.execute(statement)).scalars().all():
            record.state = EvidenceState.EXPIRED.value
            expired.append(record.evidence_ref)
        return expired

    async def delete(
        self,
        *,
        organization_id: str,
        evidence_ref: str,
        actor: str,
        reason: str,
    ) -> EvidenceRecord:
        """Erase the bytes and leave a tombstone.

        Never silent (§7): who, when and why are all recorded, and the row
        survives so the deletion can be evidenced. Idempotent — deleting twice is
        not an error, because an erasure request retried is not a failure.
        """
        record = await self.metadata(organization_id=organization_id, evidence_ref=evidence_ref)

        if record.state == EvidenceState.DELETED.value:
            return record

        path = self._root / record.storage_ref
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                # Reported rather than swallowed: a row marked DELETED whose
                # bytes remain is the worst of both worlds, and somebody must
                # know.
                logger.error(
                    "failed to erase evidence bytes for {}: {}: {}",
                    record.evidence_ref,
                    type(exc).__name__,
                    exc,
                )
                raise

        record.state = EvidenceState.DELETED.value
        record.deleted_at = datetime.now(UTC)
        record.deleted_by = actor
        record.deletion_reason = reason[:255]
        record.storage_ref = ""
        record.size_bytes = 0
        return record


def to_wire(record: EvidenceRecord) -> dict[str, Any]:
    """Evidence metadata for the API.

    Carries no bytes, no filesystem path and no storage credential. `storage_ref`
    is deliberately omitted — it is an internal locator, and a caller who has it
    is a caller tempted to bypass the lifecycle checks above.

    `geometry` is parsed rather than passed through as a string, so a caller
    receives a document instead of a document-shaped string it has to trust and
    parse itself. A row whose geometry is unreadable yields `None`: the image is
    still evidence, it simply cannot say where in itself the subject was.
    """
    return {
        "evidence_ref": record.evidence_ref,
        "geometry": _geometry(record.geometry),
        "camera_key": record.camera_key,
        "frame_ref": record.frame_ref,
        "object_id": record.object_id,
        "observation_id": record.observation_id,
        "captured_at": _iso(record.captured_at),
        "created_at": _iso(record.created_at),
        "purpose": record.purpose,
        "state": record.state,
        "servable": record.servable,
        "expires_at": _iso(record.expires_at),
        "content_hash": record.content_hash,
        "size_bytes": record.size_bytes,
        "media_type": record.media_type,
        "deleted_at": _iso(record.deleted_at),
        "deleted_by": record.deleted_by,
        "deletion_reason": record.deletion_reason,
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _geometry(raw: str) -> dict[str, Any] | None:
    """The stored geometry document, or `None`.

    Never raises. Geometry is an aid to reading a picture; a malformed one must
    cost the caller the highlight, not the evidence.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.debug("evidence geometry is not readable JSON; serving without it")
        return None
    return parsed if isinstance(parsed, dict) else None


__all__ = ["EvidenceStore", "to_wire"]
