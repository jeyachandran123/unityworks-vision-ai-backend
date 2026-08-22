"""Identity tables — the authentication and authorization foundation.

Four tables, and no more. This is deliberately **not** the restaurant domain:
there is no Restaurant, Zone, Camera, Incident or Notification here, because
those belong to Phase 4 and a table created early is a schema decision made
without the feature that would have informed it.

What is here is the minimum needed to answer *"who is asking, for which tenant,
and what may they reach"* — which the auth and authorization foundations require
in Phase 1 and every later phase builds on.

### Why AccessGrant carries a breadth column

`ScopeBreadth` exists in `app.authorization.model` because an empty camera list
is ambiguous: to Vision OS an empty camera tuple means *every camera in the
tenant*. Storing breadth explicitly means the database can represent "no access"
without an empty list that a later reader might pass through as a wildcard.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Organization(Base):
    """The customer boundary, and the tenant Vision OS scopes every query to.

    ``Organization.id`` is what becomes ``Scope.tenant_id``. That mapping is the
    reason cross-tenant leakage is structurally impossible rather than a
    filtering discipline — a scope cannot be constructed without it.
    """

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    users: Mapped[list[User]] = relationship(back_populates="organization")


class User(Base):
    """An authenticated human.

    Vision OS never learns a user exists. ``User.email`` becomes
    ``Principal.subject`` at the API edge and travels no further down — 12_SECURITY
    §5.1: *"There is no ambient user context inside the pipeline, which means no
    pipeline component can accidentally make an authorization decision."*
    """

    __tablename__ = "users"
    __table_args__ = (
        # Email is unique per organization, not globally: the same person may
        # legitimately hold accounts at two customers, and a global constraint
        # would also let anyone probe for an address's existence across tenants.
        UniqueConstraint("organization_id", "email", name="uq_users_org_email"),
        Index("ix_users_email", "email"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped[Organization] = relationship(back_populates="users")
    role_assignments: Mapped[list[RoleAssignment]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    access_grants: Mapped[list[AccessGrant]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RoleAssignment(Base):
    """One role held by one user. A user may hold several.

    A row rather than a column on ``User`` so that a role can be added or revoked
    without rewriting the user, and so the grant is individually auditable — who
    made someone a hygiene officer, and when.
    """

    __tablename__ = "role_assignments"
    __table_args__ = (UniqueConstraint("user_id", "role", name="uq_role_assignment"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: The string value of `app.authorization.model.Role`. Stored as text rather
    #: than a database enum so that adding a role is a migration of data, not of
    #: type definitions across two systems.
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    granted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="role_assignments")


class AccessGrant(Base):
    """Which cameras and sites a user reaches, stated explicitly.

    One row per user. Its absence means **no access**, which is the safe default
    and the reason `AccessDecision.to_grant()` refuses to build a Vision OS grant
    from it rather than sending an empty camera tuple the platform would read as
    a wildcard.
    """

    __tablename__ = "access_grants"
    __table_args__ = (UniqueConstraint("user_id", name="uq_access_grant_user"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: `app.authorization.model.ScopeBreadth` — "none" | "listed" | "all_in_tenant".
    #: Explicit, so that "no cameras" and "every camera" can never be represented
    #: by the same empty list.
    camera_breadth: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    #: Comma-separated camera ids. Meaningful only when breadth is "listed".
    camera_ids: Mapped[str] = mapped_column(Text, nullable=False, default="")
    site_ids: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    user: Mapped[User] = relationship(back_populates="access_grants")


__all__ = ["AccessGrant", "Organization", "RoleAssignment", "User"]
