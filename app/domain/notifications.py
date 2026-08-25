"""Telling somebody a violation happened.

### Deliberately small

This is a port and two adapters, not a notification platform. There is no
template engine, no user preference matrix, no retry queue and no digest
scheduler, because none of those are needed to answer the question this phase
asks: *when a real violation opens a real incident, does the deployment have a
supported way to find out?*

Adding the rest later is a new adapter behind the same port.

### What is deliberately absent, and why

**No notification for UNKNOWN.** The caller never offers one — an incident only
exists for a violation, and the four-state design exists precisely so that "we
could not see" never becomes an accusation. Paging somebody about it would undo
that at the last possible moment.

**No notification for `informational` severity.** Those rules accrue findings
while their accuracy is unmeasured; they raise no incident, so they reach
nothing here.

**One per incident, not one per frame.** Dispatch happens only when
`IncidentService.open` reports `created`, so a chef who stays uncovered for ten
minutes produces one message rather than one every five seconds.

### Delivery is not the record

A violation that happened is a fact. Whether anyone was successfully told is a
different fact, recorded separately. A delivery failure never rolls back the
incident — losing the record of something that really occurred because an
outbound socket timed out would be the worse of the two failures by far.

### Nothing identifiable goes out

The payload carries ids, a rule, a severity, a camera and the sentence the rule
document already wrote. **No imagery, no crop, no evidence bytes** — only the
evidence *reference*, which is worthless without the separate privilege needed
to resolve it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger


@dataclass(frozen=True, slots=True)
class IncidentNotice:
    """What a channel is given. Structured, and free of imagery."""

    incident_id: str
    organization_id: str
    restaurant_id: str | None
    camera_key: str
    rule_id: str
    severity: str
    summary: str
    observed_at: datetime
    object_id: str
    evidence_ref: str
    #: Which conditions failed, and what was seen. Enough for a recipient to
    #: judge urgency without opening the console.
    reasons: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "incident.opened",
            "incident_id": self.incident_id,
            "organization_id": self.organization_id,
            "restaurant_id": self.restaurant_id,
            "camera_key": self.camera_key,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "summary": self.summary,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(),
            "object_id": self.object_id,
            # The handle only. Resolving it needs `VIEW_EVIDENCE` and an
            # `ALLOW_EVIDENCE` deployment, neither of which a message carries.
            "evidence_ref": self.evidence_ref,
            "reasons": list(self.reasons),
        }


@runtime_checkable
class NotificationChannel(Protocol):
    """Where a notice goes. One method, so a new destination is one class."""

    channel_id: str

    async def send(self, notice: IncidentNotice) -> bool: ...


class LogChannel:
    """Writes the notice to the application log at WARNING.

    The default, and a real one rather than a placeholder: on a single-node
    deployment the log is already collected, already retained and already the
    thing an operator greps at 2am. It needs no secret, cannot fail in a way
    that blocks a request, and makes the notification path observable from day
    one.
    """

    channel_id = "log"

    async def send(self, notice: IncidentNotice) -> bool:
        logger.warning(
            "INCIDENT {} [{}] camera={} rule={} — {}",
            notice.incident_id,
            notice.severity,
            notice.camera_key,
            notice.rule_id,
            notice.summary,
        )
        return True


class FileChannel:
    """Appends one JSON object per line to a file.

    JSON Lines because it is the format every log shipper, `jq` pipeline and
    spreadsheet import already understands, and because appending a line is
    atomic enough to survive a crash mid-write without corrupting the lines
    before it.
    """

    channel_id = "file"

    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def send(self, notice: IncidentNotice) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(notice.to_wire(), separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True


class NullChannel:
    """Delivers nothing, and says so.

    For a deployment that has not chosen a channel. Distinct from having no
    notifier at all, because `notifications_suppressed` counting up is a
    visible "nobody is being told", where a missing notifier is silence.
    """

    channel_id = "null"

    async def send(self, notice: IncidentNotice) -> bool:
        return False


@dataclass(slots=True)
class NotificationAudit:
    sent: int = 0
    failed: int = 0
    suppressed: int = 0
    last_error: str = ""

    def to_wire(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "failed": self.failed,
            "suppressed": self.suppressed,
            "last_error": self.last_error,
        }


class Notifier:
    """Turns an incident into a notice and hands it to the channel."""

    __slots__ = ("_channel", "audit")

    def __init__(self, channel: NotificationChannel) -> None:
        self._channel = channel
        self.audit = NotificationAudit()

    @property
    def channel_id(self) -> str:
        return self._channel.channel_id

    async def incident_opened(self, incident: Any, finding: Any = None) -> bool:
        """Announce a newly opened incident. Returns whether it was delivered.

        Raises nothing the caller must handle: a failed delivery is recorded
        and reported as `False`, because the incident is already a fact and
        must not be undone by a channel that was unreachable.
        """
        notice = _notice_from(incident, finding)
        try:
            delivered = await self._channel.send(notice)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            self.audit.failed += 1
            self.audit.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "notification channel '{}' failed for incident {}: {}: {}",
                self._channel.channel_id, notice.incident_id,
                type(exc).__name__, exc,
            )
            return False

        if delivered:
            self.audit.sent += 1
        else:
            self.audit.suppressed += 1
        return bool(delivered)


def _notice_from(incident: Any, finding: Any) -> IncidentNotice:
    reasons: tuple[str, ...] = ()
    if finding is not None:
        reasons = tuple(
            f"{c.attribute_key} observed {c.observed!r}"
            for c in getattr(finding, "conditions", ())
            if getattr(c, "satisfied", None) is False
        )
    return IncidentNotice(
        incident_id=str(getattr(incident, "id", "")),
        organization_id=str(getattr(incident, "organization_id", "")),
        restaurant_id=getattr(incident, "restaurant_id", None),
        camera_key=str(getattr(incident, "camera_key", "")),
        rule_id=str(getattr(incident, "rule_id", "")),
        severity=str(getattr(incident, "severity", "")),
        summary=str(getattr(incident, "summary", "")),
        observed_at=getattr(incident, "observed_at", None) or datetime.now(UTC),
        object_id=str(getattr(incident, "object_id", "")),
        evidence_ref=str(getattr(incident, "evidence_refs", "") or ""),
        reasons=reasons,
    )


def build_notifier(settings: Any) -> Notifier | None:
    """The configured notifier, or `None` when notifications are off.

    `None` is a supported configuration and not an error: a deployment may
    legitimately want durable incidents and no outbound messages at all.
    """
    channel_id = (getattr(settings, "notification_channel", "") or "").strip().lower()
    if not channel_id or channel_id == "off":
        return None
    if channel_id == "log":
        return Notifier(LogChannel())
    if channel_id == "file":
        return Notifier(FileChannel(settings.notification_file_path))
    if channel_id == "null":
        return Notifier(NullChannel())
    # Named rather than silently defaulted. A typo in a channel name must not
    # look like a working deployment that happens to tell nobody.
    raise ValueError(
        f"unknown notification channel '{channel_id}'; known channels are "
        "'log', 'file', 'null', 'off'"
    )


__all__ = [
    "FileChannel",
    "IncidentNotice",
    "LogChannel",
    "NotificationAudit",
    "NotificationChannel",
    "Notifier",
    "NullChannel",
    "build_notifier",
]
