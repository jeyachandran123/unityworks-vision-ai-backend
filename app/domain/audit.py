"""The durable audit trail.

Append-only. There is no update path in this module and no delete path, because
the value of an audit trail is entirely in the fact that it was not edited. A
correction is another event (§15).

### What never enters an audit row

Passwords, access tokens, refresh tokens, API keys, RTSP credentials, and
evidence bytes. `_scrub()` strips them structurally rather than relying on every
caller to remember, because the one caller who forgets writes a credential into
the most permanently-retained table in the system.

### Why evidence access is the row that matters

Everything else here is operational history. `evidence.read` records that a named
person looked at CCTV imagery of an identifiable individual — which is the fact a
privacy regulator asks for, and the one nobody can reconstruct later if it was
never written.
"""

from __future__ import annotations

import enum
import json
import re
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import AuditEvent


class AuditAction(enum.Enum):
    """A closed set. An action that is not here is not audited, deliberately."""

    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"

    CAMERA_CREATED = "camera.created"
    CAMERA_UPDATED = "camera.updated"
    CAMERA_ENABLED = "camera.enabled"
    CAMERA_DISABLED = "camera.disabled"
    #: Removing a camera destroys its observation partition, so this row
    #: always sits beside an `observation.truncated` one. A deletion that
    #: left only the first would be a record of the configuration change
    #: with no record of the data it destroyed.
    CAMERA_DELETED = "camera.deleted"

    INCIDENT_CREATED = "incident.created"
    INCIDENT_ACKNOWLEDGED = "incident.acknowledged"
    INCIDENT_RESOLVED = "incident.resolved"

    #: The legally significant one.
    EVIDENCE_READ = "evidence.read"
    EVIDENCE_CREATED = "evidence.created"
    EVIDENCE_EXPIRED = "evidence.expired"
    EVIDENCE_DELETED = "evidence.deleted"
    EVIDENCE_DENIED = "evidence.denied"

    #: Reading what the cameras observed about people at work. Not imagery — an
    #: observation names a tracked object, never a person — but it is still a
    #: record about staff, and §12_SECURITY keeps "a person was here" and "here
    #: is their picture" as separate authorisations. Separate rows, too.
    OBSERVATIONS_READ = "observation.read"
    #: A retention sweep removed a time-bounded prefix of the observation log.
    #: Written per organisation so the deletion is provable, the same way an
    #: evidence erasure is.
    OBSERVATIONS_TRUNCATED = "observation.truncated"

    #: Running a report reads incidents, observations and zone attribution in
    #: one act. It is audited for the same reason an evidence retrieval is:
    #: assembling a picture of what staff did is not made harmless by the
    #: figures being aggregates.
    REPORT_GENERATED = "report.generated"
    #: A copy left the system. Recorded separately from generation, because
    #: the file outlives every retention policy this application enforces and
    #: "who took a copy" is the question an investigation actually asks.
    REPORT_EXPORTED = "report.exported"
    #: A report the caller was refused. Recorded with the same weight as a
    #: success, exactly as a refused evidence retrieval is.
    REPORT_DENIED = "report.denied"

    POLICY_CHANGED = "policy.changed"
    ADMIN_CHANGED = "admin.changed"

    RESTAURANT_CREATED = "restaurant.created"
    RESTAURANT_UPDATED = "restaurant.updated"
    ZONE_CREATED = "zone.created"
    ZONE_UPDATED = "zone.updated"


class AuditOutcome(enum.Enum):
    SUCCESS = "success"
    DENIED = "denied"
    FAILED = "failed"


#: Keys whose values never survive into an audit row, at any nesting depth.
_FORBIDDEN_KEYS = frozenset(
    {
        "password",
        "password_hash",
        "access_token",
        "refresh_token",
        "token",
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "credential",
        "credentials",
        "authorization",
        "cookie",
        "bytes",
        "payload",
        "image",
        "content",
    }
)

#: Values that look like a credential even under an innocent key name.
_SECRET_SHAPES = (
    re.compile(r"nvapi-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\brtsp://[^\s:@/]+:[^\s@/]+@"),  # rtsp://user:pass@host
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
    re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"),  # bcrypt hash
)

REDACTED = "***"


def _scrub(value: Any, depth: int = 0) -> Any:
    """Remove anything credential-shaped, by key and by value.

    Structural rather than advisory. A caller that passes a whole request body
    by accident gets a scrubbed row, not a leak — and audit rows are the longest
    retained data in the system, so a leak here is the most permanent kind.
    """
    if depth > 6:
        return REDACTED

    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                cleaned[key] = REDACTED
            else:
                cleaned[key] = _scrub(item, depth + 1)
        return cleaned

    if isinstance(value, list | tuple):
        return [_scrub(item, depth + 1) for item in value][:50]

    if isinstance(value, bytes | bytearray | memoryview):
        # Evidence bytes must never reach an audit row (§24). Size is the only
        # part worth keeping.
        return f"<{len(bytes(value))} bytes omitted>"

    if isinstance(value, str):
        text = value
        for pattern in _SECRET_SHAPES:
            text = pattern.sub(REDACTED, text)
        return text[:2000]

    return value


class AuditTrail:
    """Writes audit events. **Append only.**

    Deliberately offers no `update` and no `delete`. Retention pruning is a
    separate, explicit operation in `app/domain/retention.py`, and it deletes
    whole aged rows rather than editing live ones.
    """

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        action: AuditAction,
        organization_id: str,
        actor: str = "",
        actor_roles: tuple[str, ...] = (),
        resource_type: str = "",
        resource_id: str = "",
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        request_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append one event. Never raises into the caller's business logic.

        An audit write that fails must not roll back the action it describes —
        but it must also not be silent, because a missing audit row is exactly
        what an investigation later needs. So a failure is logged loudly and the
        caller continues.
        """
        event = AuditEvent(
            organization_id=organization_id,
            actor=actor,
            actor_roles=",".join(sorted(actor_roles)),
            action=action.value,
            resource_type=resource_type,
            resource_id=str(resource_id),
            outcome=outcome.value,
            request_id=request_id,
            detail=json.dumps(_scrub(detail or {}), default=str)[:4000],
        )
        self._session.add(event)
        return event

    async def query(
        self,
        *,
        organization_id: str,
        actor: str | None = None,
        action: AuditAction | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Read the trail, always tenant-scoped.

        Scoped in the query rather than filtered afterwards: there is no moment
        at which another tenant's audit rows exist in memory to leak.
        """
        statement = select(AuditEvent).where(AuditEvent.organization_id == organization_id)
        if actor:
            statement = statement.where(AuditEvent.actor == actor)
        if action:
            statement = statement.where(AuditEvent.action == action.value)
        if resource_type:
            statement = statement.where(AuditEvent.resource_type == resource_type)
        if resource_id:
            statement = statement.where(AuditEvent.resource_id == str(resource_id))

        statement = statement.order_by(AuditEvent.occurred_at.desc()).limit(min(max(limit, 1), 500))
        result = await self._session.execute(statement)
        return list(result.scalars().all())


def to_wire(event: AuditEvent) -> dict[str, Any]:
    """One audit row for the API. Already scrubbed at write time."""
    try:
        detail = json.loads(event.detail) if event.detail else {}
    except json.JSONDecodeError:  # pragma: no cover - written as JSON
        detail = {}
    return {
        "id": event.id,
        "actor": event.actor,
        "actor_roles": [r for r in event.actor_roles.split(",") if r],
        "action": event.action,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "outcome": event.outcome,
        "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
        "request_id": event.request_id,
        "detail": detail,
    }


def log_audit_failure(action: AuditAction, error: Exception) -> None:
    """A missing audit row is a finding in its own right. Never silent."""
    logger.error(
        "AUDIT WRITE FAILED for {} — {}: {}. The action proceeded; the record did not.",
        action.value,
        type(error).__name__,
        error,
    )


__all__ = [
    "REDACTED",
    "AuditAction",
    "AuditOutcome",
    "AuditTrail",
    "log_audit_failure",
    "to_wire",
]
