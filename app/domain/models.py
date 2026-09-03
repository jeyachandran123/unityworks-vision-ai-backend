"""The durable application domain.

Everything here survives a restart. Nothing here belongs to Vision OS.

### The dividing line, restated because it is the whole architecture

Vision OS produces **observations** — "the head covering is not visible" — and
holds them in its own stores, behind its own ports. The application owns what a
business does about them: an `Incident` somebody must acknowledge, an `Evidence`
record with a retention clock, an `AuditEvent` that proves who looked.

So no table here stores a perception result. There is no `is_wearing_hairnet`
column and there never will be — that would be a second source of truth for what
the camera saw, and the second one drifts.

What the application *does* store is the **frozen finding** at the moment an
incident was raised. That is not a duplicate of live state; it is a historical
record that must stay explicable after the object has left the frame, the
attribute has expired, and the rules have changed.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import true as sa_true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


# ── Sites and cameras ────────────────────────────────────────────────────────


class Restaurant(Base):
    """A physical location. Maps to Vision OS `SiteId`."""

    __tablename__ = "restaurants"
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_restaurant_slug"),
        Index("ix_restaurants_org", "organization_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    zones: Mapped[list[Zone]] = relationship(back_populates="restaurant")


class Zone(Base):
    """A named area — "prep line", "wash station".

    Worth modelling from the start even while v1 renders one zone per camera: it
    is what lets an operator say "the prep line had four violations this week"
    without naming a camera.
    """

    __tablename__ = "zones"
    __table_args__ = (Index("ix_zones_restaurant", "restaurant_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    restaurant: Mapped[Restaurant] = relationship(back_populates="zones")
    cameras: Mapped[list[Camera]] = relationship(back_populates="zone")


class Camera(Base):
    """A configured video source.

    Replaces `CCTV_CHANNELS`. The DVR has 16 channels; this table decides which
    of them the application processes, and a row that is not `enabled` creates no
    Vision OS session at all — no socket, no decode, no model call. Cost follows
    configuration, not hardware.

    **No password lives here.** `credential_ref` is a reference the
    `SecretProvider` resolves at connect time; the row holds a pointer, never a
    value, so a database dump is not a credential dump.
    """

    __tablename__ = "cameras"
    __table_args__ = (
        UniqueConstraint("organization_id", "camera_key", name="uq_camera_key"),
        Index("ix_cameras_org_enabled", "organization_id", "enabled"),
        Index("ix_cameras_restaurant", "restaurant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("zones.id", ondelete="SET NULL"), nullable=True
    )

    #: The stable identity the pipeline partitions on — `cam-01`. Never inferred
    #: from a frame, and unique per organization so two tenants cannot collide.
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Why this camera is watched. Recorded because retention and lawful basis
    #: both depend on purpose, not on the fact that a lens exists.
    purpose: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    #: Transport
    host: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    rtsp_port: Mapped[int] = mapped_column(Integer, nullable=False, default=554)
    channel: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    stream_type: Mapped[str] = mapped_column(String(16), nullable=False, default="sub")
    username: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    #: `env:CCTV_PASSWORD`, `file:/run/secrets/dvr`. A REFERENCE, never a secret.
    credential_ref: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    #: Independent of camera fps: a 25 fps stream must not become 25 fps of work.
    analysis_fps: Mapped[float] = mapped_column(nullable=False, default=4.0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Whether this camera is *analysed*, as distinct from whether it streams.
    #:
    #: `enabled` alone was doing two jobs: it started the RTSP session and it
    #: enrolled the camera in perception. A site with sixteen channels and four
    #: kitchens could not watch the corridors on the wall without also paying
    #: detection, tracking, cropping and the understander's call budget for them
    #: — and that budget is a single global allowance the kitchens already spend
    #: in full.
    #:
    #: Defaults to true, matching the migration's `server_default`, so a row that
    #: predates the column keeps exactly the behaviour it had. Narrowing is a
    #: deliberate, per-camera, durable act.
    analysis_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=sa_true()
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    zone: Mapped[Zone | None] = relationship(back_populates="cameras")


class CameraZoneAssignment(Base):
    """Which zone a camera belonged to, **and for how long**.

    ### The problem this exists to solve

    `cameras.zone_id` is current state: it says where a camera is *now*. An
    observation, an incident and a piece of evidence all carry a `camera_key`
    and a capture time, and answering "which zone did this happen in" by joining
    `cameras.zone_id` reads today's answer onto a past event. Move a camera from
    the prep line to the wash station and every reading it ever produced
    silently relocates — a whole quarter of prep-line history rewritten by one
    dropdown, with nothing in the record showing it happened.

    That is the same class of error `Incident.finding_snapshot` exists to
    prevent, and it deserves the same shape of answer: freeze the attribution
    where it was made rather than recompute it later.

    ### Why an interval table rather than a column on the observation

    The natural fix is a `zone_id` written onto each observation at the moment
    it is produced. The application cannot do that: observations are produced by
    Vision OS, whose `Observation` envelope carries `site_id` and no zone, and
    which this phase may not modify. Nor should it — a zone is an organisational
    idea the platform deliberately does not hold.

    So the attribution is recorded once per *assignment* instead of once per
    observation, which is strictly better here: it costs one row per camera move
    rather than one field per reading, and it fixes historical zone attribution
    for incidents, evidence and frames at the same time, none of which carry a
    zone today either.

    ### Append-only, and never back-dated

    `effective_from` is the moment the assignment was made, and a row is closed
    by setting `effective_to` — never by editing `zone_id`. A camera's history is
    the ordered set of its intervals, and an instant that predates the first
    interval resolves to **no zone**, which is the honest answer: nobody
    recorded where that camera was. It is emphatically not backfilled from
    today's mapping, because that would commit the exact error the table exists
    to prevent.
    """

    __tablename__ = "camera_zone_assignments"
    __table_args__ = (
        Index("ix_cza_camera_time", "organization_id", "camera_key", "effective_from"),
        Index("ix_cza_open", "organization_id", "camera_key", "effective_to"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: The camera as the pipeline names it, not the row id: an observation
    #: carries `camera_id`, and matching on anything else would need a join that
    #: could itself go stale.
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    #: `None` records "assigned to no zone", which is a real assignment and
    #: different from having no interval at all.
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: The zone's name **as it was when the assignment was made**. Frozen for the
    #: same reason the finding is: renaming a zone must not rewrite what a past
    #: reading was labelled.
    zone_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    restaurant_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    #: `None` while this is the assignment in force. Set once, when the camera
    #: moves; the row is never otherwise edited.
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Who made the assignment. An attribution nobody can trace is worth less
    #: than one that is wrong and known to be.
    assigned_by: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )


# ── Evidence ─────────────────────────────────────────────────────────────────


class EvidenceState(enum.Enum):
    """The lifecycle §7 requires. Deletion is a state, not an absence."""

    RETAINED = "retained"
    EXPIRED = "expired"
    """Past its retention date. **Must not be served**, and not yet erased."""
    DELETED = "deleted"
    """Bytes erased. The row survives as a tombstone so the deletion is provable."""


class EvidenceRecord(Base):
    """Metadata for one piece of retained imagery.

    **The bytes are not in this row.** `storage_ref` points at them, so an
    incident that cites five pieces of evidence does not carry five images, and a
    query over incidents does not drag megabytes through the database.

    The row outlives the bytes: deleting evidence sets `state=DELETED` and clears
    the storage, leaving a tombstone that proves *what* was deleted, *when*, by
    *whom* and *why*. An evidence record that simply vanished would make an
    erasure request impossible to evidence afterwards.
    """

    __tablename__ = "evidence_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "evidence_ref", name="uq_evidence_ref"),
        Index("ix_evidence_org_state", "organization_id", "state"),
        Index("ix_evidence_camera_time", "camera_key", "captured_at"),
        Index("ix_evidence_expires", "expires_at"),
        Index("ix_evidence_object", "object_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: The handle an observation carries. Stable across storage backends.
    evidence_ref: Mapped[str] = mapped_column(String(255), nullable=False)

    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Vision OS references. Plain strings, not foreign keys: the platform owns
    #: those identities and the application does not mirror its tables.
    frame_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    object_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    observation_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    #: Why this image was retained. Retention without a purpose is collection.
    purpose: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    #: **Where in this image the subject is.** JSON text: the alert subject's
    #: normalized box, the other objects cut from the same frame, and the
    #: handles of the crops taken from it.
    #:
    #: Durable rather than derived, and that is the whole point. The boxes live
    #: in a bounded in-memory ring that holds a couple of minutes of frames; an
    #: incident is read for days. Recomputing a box later would mean running a
    #: detector over the stored JPEG and highlighting whoever it found — which
    #: is *a* person in the picture, not necessarily the one the verdict was
    #: about. Frozen here at capture, the highlight is the subject or nothing.
    geometry: Mapped[str] = mapped_column(Text, nullable=False, default="")

    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EvidenceState.RETAINED.value
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Where the bytes live. Opaque to callers; never a credential-bearing URL.
    storage_ref: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    media_type: Mapped[str] = mapped_column(String(64), nullable=False, default="image/jpeg")

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deletion_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    @property
    def servable(self) -> bool:
        """Whether the bytes may be returned at all.

        Expired evidence is refused even though the bytes may still be on disk
        until the sweeper runs. Retention is a promise about what is *served*,
        not only about what is eventually erased.
        """
        return self.state == EvidenceState.RETAINED.value


# ── Incidents ────────────────────────────────────────────────────────────────


class IncidentStatus(enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Incident(Base):
    """A durable record that somebody must act on.

    Created from a compliance finding, then it lives its own life. Vision OS
    knows nothing about it — `compliance` produces findings on read, and this is
    what the application decided to do about one.

    ### Why the finding is frozen here

    Findings are recomputed from live state, which is correct: a finding is a
    pure function of (rule set, observation, now), so it never needs
    invalidation. But an incident must remain explicable in six months, after the
    object has left the frame, the attribute has expired and the rules have
    changed. `finding_snapshot` is that frozen record, and `ruleset_version` says
    which rules produced it.

    **Do not recompute a historical incident against today's policy.**
    """

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_org_status", "organization_id", "status"),
        Index("ix_incidents_restaurant_time", "restaurant_id", "created_at"),
        Index("ix_incidents_camera_time", "camera_key", "created_at"),
        Index("ix_incidents_status_time", "status", "created_at"),
        # Supports the de-duplication lookup: one OPEN incident per subject per
        # rule, so a burst of findings becomes one incident rather than four
        # hundred. Deliberately **not** a unique constraint — uniqueness must
        # hold only while the incident is open, and a partial unique index is
        # spelled differently in SQLite and PostgreSQL. `IncidentService.open`
        # enforces it in one place, and a test proves a repeat finding does not
        # create a second row.
        Index(
            "ix_incident_dedupe",
            "organization_id",
            "camera_key",
            "object_id",
            "rule_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    restaurant_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    #: The subject, as Vision OS identified it. Not a person: the platform
    #: performs no biometric identification, and this is a tracked object id.
    object_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    track_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    rule_id: Mapped[str] = mapped_column(String(128), nullable=False)
    ruleset_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    #: The end-user sentence, from the rule document. Stored so it regenerates
    #: identically six months later rather than being re-derived.
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=IncidentStatus.ACTIVE.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    #: `observation` when a later grounded observation cleared it, or `operator`.
    #: A UI refresh is neither, and cannot resolve anything.
    resolution_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: The finding, frozen at creation. JSON text rather than a typed column:
    #: its shape belongs to the compliance layer, and mirroring that shape into
    #: columns here would couple this table to a schema it does not own.
    finding_snapshot: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    #: Evidence handles, comma-separated. Handles, not images.
    evidence_refs: Mapped[str] = mapped_column(Text, nullable=False, default="")

    @property
    def open(self) -> bool:
        return self.status != IncidentStatus.RESOLVED.value


# ── Frames ───────────────────────────────────────────────────────────────────


class FrameRecord(Base):
    """Frame **metadata**, for traceability. Not a recording service.

    §12 is explicit: this exists so an investigation can ask "what did the system
    see at 14:32:07 on camera 3" and get an answer that links to observations and
    evidence. It stores no pixels — `evidence_records` holds the crops that were
    deliberately retained, and everything else is gone by design.

    Retaining every raw frame would be a video recorder, a storage problem and a
    much larger privacy surface, and none of those is what traceability needs.
    """

    __tablename__ = "frame_records"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "camera_key", "epoch", "sequence", name="uq_frame_identity"
        ),
        Index("ix_frames_camera_time", "camera_key", "captured_at"),
        Index("ix_frames_org_time", "organization_id", "captured_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    camera_key: Mapped[str] = mapped_column(String(64), nullable=False)

    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Increments on reconnect. Two frames with the same sequence and different
    #: epochs are different frames, and nothing may associate across the gap.
    epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frame_ref: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    #: Capture time, not arrival. Freshness ages against this everywhere else and
    #: an investigation must read the same clock.
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: How many observations this frame contributed to. A count, not the payload.
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: `live` or `replay`. A replay frame is never presented as live.
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="replay")


# ── Audit ────────────────────────────────────────────────────────────────────


class AuditEvent(Base):
    """Who did what, and who looked at what.

    **Append-only by convention and by API.** There is no update path in the
    service that writes these, and a correction is another event rather than an
    edit — §15. Rewriting an audit row destroys the only thing an audit trail is
    for.

    The evidence-access rows are the ones that matter legally: every retrieval of
    imagery of an identifiable person leaves one.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_org_time", "organization_id", "occurred_at"),
        Index("ix_audit_actor_time", "actor", "occurred_at"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_action_time", "action", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: The principal's subject. Never a token, never a password.
    actor: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    actor_roles: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="success")

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    #: Ties an audit row to the request that produced it, and to the log lines
    #: for that request. Not a secret.
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: Free-form context. Scrubbed by the writer — see `app/domain/audit.py`.
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")


__all__ = [
    "AuditEvent",
    "Camera",
    "CameraZoneAssignment",
    "EvidenceRecord",
    "EvidenceState",
    "FrameRecord",
    "Incident",
    "IncidentStatus",
    "Restaurant",
    "Zone",
]
