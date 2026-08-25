"""Incident lifecycle.

    ACTIVE ──acknowledge──► ACKNOWLEDGED ──resolve──► RESOLVED
       └────────────────────resolve────────────────────┘

### What may resolve an incident

Exactly two things (§11):

* **A later grounded observation** that clears the condition. `head_covering =
  none` raised it; `head_covering = hairnet`, observed afterwards, closes it.
* **An explicit authorised operator action**, recorded with who and why.

A UI refresh is neither. There is no code path here that closes an incident
because somebody looked at it, and `resolution_kind` records which of the two it
was — so "the system saw it fixed" and "a manager said it was fixed" never blur.

### Why the finding is frozen

Findings are recomputed from live state on every read, which is correct and needs
no invalidation. An incident is the opposite: it must stay explicable in six
months, after the subject left the frame, the attribute expired and the rules
changed. `finding_snapshot` and `ruleset_version` are that frozen record, and
nothing here ever re-derives a historical incident from today's policy.

### De-duplication

One open incident per (camera, subject, rule). A subject standing in frame for
four minutes produces a finding on every evaluation; without this it would
produce hundreds of incidents and the queue would be useless. The repeat findings
are not lost — they update `observed_at`, so the incident shows it is ongoing.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Incident, IncidentStatus
from app.errors import ConflictError, NotFoundError, ValidationError


class IncidentService:
    """Creates and advances incidents. Tenant-scoped at every entry point."""

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def open(
        self,
        *,
        organization_id: str,
        camera_key: str,
        rule_id: str,
        object_id: str,
        observed_at: datetime,
        severity: str = "medium",
        summary: str = "",
        ruleset_version: str = "",
        finding: dict[str, Any] | None = None,
        restaurant_id: str | None = None,
        zone_id: str | None = None,
        track_id: str = "",
        evidence_refs: tuple[str, ...] = (),
    ) -> tuple[Incident, bool]:
        """Open an incident, or update the one already open.

        Returns `(incident, created)`. `created is False` means a repeat finding
        updated an existing incident rather than raising a second one.
        """
        existing = await self._open_for(
            organization_id=organization_id,
            camera_key=camera_key,
            object_id=object_id,
            rule_id=rule_id,
        )

        if existing is not None:
            # Still happening. Move the clock, keep the original snapshot — the
            # incident is about when it *started*, and re-freezing it would erase
            # that.
            #
            # Both sides are normalised to aware UTC first. The column is
            # `DateTime(timezone=True)`, which Postgres round-trips with its
            # offset intact and SQLite returns naive — so comparing a stored
            # value against a freshly built one raised `can't compare
            # offset-naive and offset-aware datetimes` on the repeat-finding
            # path, and only on the repeat, which is the path a continuous
            # violation takes every single time after the first.
            existing.observed_at = max(_aware(existing.observed_at), _aware(observed_at))
            if evidence_refs:
                existing.evidence_refs = _merge_refs(existing.evidence_refs, evidence_refs)
            return existing, False

        incident = Incident(
            # Assigned here, not left to the column default. The column default
            # runs at INSERT, so a caller that needs the handle before the
            # transaction commits — to cite it in an audit row, or to return it
            # from the request that raised it — would read `None`.
            id=uuid.uuid4().hex,
            organization_id=organization_id,
            restaurant_id=restaurant_id,
            zone_id=zone_id,
            camera_key=camera_key,
            object_id=object_id,
            track_id=track_id,
            rule_id=rule_id,
            ruleset_version=ruleset_version,
            severity=severity,
            summary=summary,
            status=IncidentStatus.ACTIVE.value,
            observed_at=observed_at,
            finding_snapshot=json.dumps(finding or {}, default=str)[:8000],
            evidence_refs=",".join(evidence_refs),
        )
        self._session.add(incident)
        return incident, True

    async def acknowledge(self, *, organization_id: str, incident_id: str, actor: str) -> Incident:
        """ "Somebody has seen this." Does not close it."""
        incident = await self.get(organization_id=organization_id, incident_id=incident_id)

        if incident.status == IncidentStatus.RESOLVED.value:
            raise ConflictError("a resolved incident cannot be acknowledged")
        if incident.status == IncidentStatus.ACKNOWLEDGED.value:
            return incident  # idempotent

        incident.status = IncidentStatus.ACKNOWLEDGED.value
        incident.acknowledged_at = datetime.now(UTC)
        incident.acknowledged_by = actor
        return incident

    async def resolve(
        self,
        *,
        organization_id: str,
        incident_id: str,
        actor: str,
        kind: str,
        note: str = "",
    ) -> Incident:
        """Close it — by observation or by operator, and never otherwise.

        Raises:
            ValidationError: `kind` is neither `observation` nor `operator`.
                There is no third way to close an incident, and refusing an
                unknown one keeps a future caller from inventing "auto" and
                quietly closing violations nobody looked at.
        """
        if kind not in {"observation", "operator"}:
            raise ValidationError(
                "an incident is resolved by a later grounded observation or by an "
                "authorised operator; nothing else closes one",
                details={"kind": kind},
            )
        if kind == "operator" and not note.strip():
            # A human closing a violation must say why. The note is what makes
            # the decision reviewable later.
            raise ValidationError("an operator resolution must carry a reason")

        incident = await self.get(organization_id=organization_id, incident_id=incident_id)
        if incident.status == IncidentStatus.RESOLVED.value:
            return incident  # idempotent

        incident.status = IncidentStatus.RESOLVED.value
        incident.resolved_at = datetime.now(UTC)
        incident.resolved_by = actor
        incident.resolution_kind = kind
        incident.resolution_note = note[:2000] or None
        return incident

    async def resolve_by_observation(
        self,
        *,
        organization_id: str,
        camera_key: str,
        object_id: str,
        rule_id: str,
    ) -> Incident | None:
        """Close the open incident because the condition is no longer met.

        Called when a later observation shows the subject now satisfies the rule.
        Returns `None` when there was nothing open, which is the common case and
        not an error.
        """
        incident = await self._open_for(
            organization_id=organization_id,
            camera_key=camera_key,
            object_id=object_id,
            rule_id=rule_id,
        )
        if incident is None:
            return None

        return await self.resolve(
            organization_id=organization_id,
            incident_id=incident.id,
            actor="vision-os",
            kind="observation",
            note="a later observation showed the rule satisfied",
        )

    # ── reads ────────────────────────────────────────────────────────────────

    async def get(self, *, organization_id: str, incident_id: str) -> Incident:
        result = await self._session.execute(
            select(Incident).where(
                Incident.organization_id == organization_id,
                Incident.id == incident_id,
            )
        )
        incident = result.scalar_one_or_none()
        if incident is None:
            # Not `Forbidden`: revealing that an incident exists in another
            # tenant is itself a disclosure.
            raise NotFoundError("no such incident")
        return incident

    async def list(
        self,
        *,
        organization_id: str,
        status: str | None = None,
        camera_keys: tuple[str, ...] | None = None,
        restaurant_id: str | None = None,
        limit: int = 100,
    ) -> list[Incident]:
        """List incidents, scoped by tenant and by the caller's camera reach.

        `camera_keys is None` means a tenant-wide grant; an **empty tuple means
        none** and returns nothing. The same three-state discipline as everywhere
        else — an empty list must never read as a wildcard.
        """
        if camera_keys is not None and len(camera_keys) == 0:
            return []

        statement = select(Incident).where(Incident.organization_id == organization_id)
        if status:
            statement = statement.where(Incident.status == status)
        if camera_keys is not None:
            statement = statement.where(Incident.camera_key.in_(camera_keys))
        if restaurant_id:
            statement = statement.where(Incident.restaurant_id == restaurant_id)

        statement = statement.order_by(Incident.created_at.desc()).limit(min(max(limit, 1), 500))
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def _open_for(
        self,
        *,
        organization_id: str,
        camera_key: str,
        object_id: str,
        rule_id: str,
    ) -> Incident | None:
        result = await self._session.execute(
            select(Incident).where(
                Incident.organization_id == organization_id,
                Incident.camera_key == camera_key,
                Incident.object_id == object_id,
                Incident.rule_id == rule_id,
                Incident.status != IncidentStatus.RESOLVED.value,
            )
        )
        return result.scalars().first()


def _aware(value: datetime) -> datetime:
    """A timezone-aware view of a stored timestamp.

    Naive values are read as UTC, which is what every writer here stores.
    Assuming local time instead would silently shift an incident's clock by the
    server's offset.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _merge_refs(existing: str, incoming: tuple[str, ...]) -> str:
    """Union of evidence handles, order-stable, bounded.

    Bounded because a subject in frame for an hour would otherwise accumulate an
    unbounded list on one row.
    """
    seen = [r for r in existing.split(",") if r]
    for ref in incoming:
        if ref and ref not in seen:
            seen.append(ref)
    return ",".join(seen[:50])


def to_wire(incident: Incident) -> dict[str, Any]:
    """One incident for the API, answering the eight operator questions."""
    try:
        finding = json.loads(incident.finding_snapshot) if incident.finding_snapshot else {}
    except json.JSONDecodeError:  # pragma: no cover - written as JSON
        finding = {}

    return {
        "id": incident.id,
        "status": incident.status,
        "severity": incident.severity,
        # WHAT
        "summary": incident.summary,
        "rule_id": incident.rule_id,
        "ruleset_version": incident.ruleset_version,
        # WHERE
        "restaurant_id": incident.restaurant_id,
        "zone_id": incident.zone_id,
        "camera_key": incident.camera_key,
        # WHEN
        "observed_at": _iso(incident.observed_at),
        "created_at": _iso(incident.created_at),
        # WHO
        "object_id": incident.object_id,
        "track_id": incident.track_id,
        # WHY — the frozen finding, not a recomputation
        "finding": finding,
        # EVIDENCE — handles, never images
        "evidence_refs": [r for r in incident.evidence_refs.split(",") if r],
        # STATUS
        "acknowledged_at": _iso(incident.acknowledged_at),
        "acknowledged_by": incident.acknowledged_by,
        "resolved_at": _iso(incident.resolved_at),
        "resolved_by": incident.resolved_by,
        "resolution_kind": incident.resolution_kind,
        "resolution_note": incident.resolution_note,
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


__all__ = ["IncidentService", "to_wire"]
