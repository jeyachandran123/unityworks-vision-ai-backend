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
    #: `app.authorization.model.OrganizationStatus` — "active" | "suspended" |
    #: "archived". Stored as text for the same reason `RoleAssignment.role` is:
    #: adding a state is a migration of data, not of type definitions shared
    #: across two systems. Defaults to "active" so every row that predates this
    #: column — including `org-unityworks` — reads as unchanged behavior with no
    #: manual backfill step; see the migration's `server_default`.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    #: When the status last changed, and why. Nullable and empty by default
    #: rather than backfilled: an organization that has never changed status
    #: has no status change to describe, and answering with `created_at` would
    #: be a fabricated entry in a field people will read as history.
    #:
    #: The reason exists because "why have our cameras stopped" is the first
    #: question a suspension produces, and an operator console that cannot
    #: answer it sends somebody to read the audit log instead.
    status_changed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
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
    permission_overrides: Mapped[list[PermissionOverride]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    memberships: Mapped[list[OrganizationMembership]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RoleAssignment(Base):
    """One role held by one user, **in one organization**. A user may hold several.

    A row rather than a column on ``User`` so that a role can be added or revoked
    without rewriting the user, and so the grant is individually auditable — who
    made someone a hygiene officer, and when.

    ``organization_id`` is what makes "org_admin at Acme" a different fact from
    "org_admin at Borden". Without it, granting somebody a second organization
    would carry every role they already hold across with them, and a membership
    row would become an entry ticket to authority nobody granted.
    """

    __tablename__ = "role_assignments"
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", "role", name="uq_role_assignment"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: The organization this role is held *in*. Not derived from the user's home
    #: organization: a user may hold different roles in each organization they
    #: belong to, and collapsing the two would make that difference
    #: inexpressible.
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
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
    __table_args__ = (UniqueConstraint("user_id", "organization_id", name="uq_access_grant_user"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: Which organization's cameras and sites this grant names. A camera id is
    #: unique only within an organization (`app.domain.runtime_identity`), so a
    #: grant without one would name cameras in whichever tenant happened to be
    #: active — exactly the cross-tenant read this column exists to prevent.
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
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


class PermissionOverride(Base):
    """One user's explicit exception to what their roles would give them.

    A row per (user, permission), and its *presence* is the whole signal: no
    row means INHERIT — the role's own answer stands, and INHERIT is
    deliberately never written as a row. A stored row is always GRANT (add a
    permission no held role carries) or REVOKE (remove one that a held role
    does carry) — see `app.authorization.model.OverrideState`.

    Tenant scope is **stated**, not inherited. It was inherited once, and the
    argument was good: `user_id` already resolved to exactly one organization
    through the `users` table, so a column here would have been redundant, and
    redundant tenant fields are exactly the kind of drift that lets a row
    quietly stop matching its owner. That argument is now void.
    `organization_memberships` means `user_id` resolves to *several*
    organizations, and an override carrying no organization of its own would
    follow its holder into every one of them.
    """

    __tablename__ = "permission_overrides"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "organization_id",
            "permission",
            name="uq_permission_override_user_permission",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: The organization the exception applies in.
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: The string value of `app.authorization.model.Permission`. Stored as text
    #: for the same reason `RoleAssignment.role` is: adding a permission is a
    #: migration of data, not of type definitions shared across two systems.
    permission: Mapped[str] = mapped_column(String(64), nullable=False)
    #: `app.authorization.model.OverrideState` — "grant" | "revoke". Never
    #: "inherit": that state is the absence of a row, not a value in one.
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
    #: The acting user's id. Nullable because a migration-seeded or
    #: system-issued override has no human grantor; a human-issued one always
    #: has one, enforced at the domain layer in `app.authorization.overrides`.
    granted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="permission_overrides")


class OrganizationMembership(Base):
    """This user may enter this organization. The whole of the authorization.

    ### Why a row, and why only a row

    A membership is the *entry ticket*, and it is deliberately the only thing
    that is. Knowing an organization id is not access. Holding `org_admin`
    somewhere else is not access. Having entered an organization yesterday is
    not access. A token naming an organization is not access either — it is
    checked against this table on every request, so revoking a membership takes
    effect on the next call rather than at the next token expiry.

    That is the same discipline `decision_for_claims` already applies to roles,
    extended to the tenant itself. Before this table the tenant was carried
    solely by `User.organization_id` and a token could not name anything else;
    now that a token *can*, the claim needs a server-side fact to be checked
    against, and this is it.

    ### What it deliberately does not carry

    No roles, no permissions, no camera scope. Entering an organization and
    being able to do anything in it are separate questions, answered by
    `role_assignments`, `permission_overrides` and `access_grants` — each of
    which now names its own organization. A membership with no role rows means
    somebody who may enter and can see nothing, which is a coherent state and
    the correct default for a grant that has just been made.

    ### Its relationship to `User.organization_id`

    None, structurally. `User.organization_id` remains the *home* organization:
    it owns email uniqueness, it is where a failed login is filed, and it is
    the tenant an operator's own account lives in. It confers no entry by
    itself — the migration that creates this table writes an explicit
    membership for every existing user in their home organization precisely so
    that "may enter" has exactly one answer everywhere, with no special case
    for the organization a user happens to have been created in.
    """

    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
        Index("ix_memberships_user", "user_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    #: Who granted it. Nullable because the memberships the migration writes for
    #: existing users were granted by nobody — they are a restatement of access
    #: that already existed, and naming an actor for them would be a fabricated
    #: row in a table people will read as history.
    granted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="memberships")
    organization: Mapped[Organization] = relationship()


class PlatformOperatorGrant(Base):
    """This account administers organizations, rather than working inside one.

    A table rather than a column on `User` or a value in `Role`, and the
    distinction is the point — see `app.authorization.platform`. A role lives
    inside an organization and is granted by that organization's own
    administrators; if platform authority were a role, every organization
    admin could mint one and the tenant boundary would be decorative.

    There is no `organization_id` here, and its absence is meaningful rather
    than an omission: an operator is not scoped to a tenant. `user_id` still
    resolves to one through `users`, which is where the operator's own login
    lives, but that organization confers nothing on them and restricts nothing.

    Nothing in the HTTP API writes to this table. `scripts/manage.py
    grant-operator` does, deliberately out of band.
    """

    __tablename__ = "platform_operator_grants"
    __table_args__ = (UniqueConstraint("user_id", name="uq_platform_operator_user"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    #: Who granted it. Nullable only because the first operator in a deployment
    #: is granted from the command line by a human with database access, and
    #: there is no earlier operator to name.
    granted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Why. Free text, required by the CLI rather than by the column, because a
    #: privilege that reaches every customer should not be granted silently.
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")


__all__ = [
    "AccessGrant",
    "Organization",
    "OrganizationMembership",
    "PermissionOverride",
    "PlatformOperatorGrant",
    "RoleAssignment",
    "User",
]
