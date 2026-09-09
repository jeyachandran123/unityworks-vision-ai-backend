"""organization memberships, and per-organization authorization rows

The change that makes "this person works for two customers" expressible.

### `organization_memberships`

Until now the tenant was `User.organization_id` and nothing else could be
named: a token carried `ten`, and `decision_for_claims` refused any request
whose `ten` did not equal that column. That is a good property and it is not
being given up — it is being generalised. The token still carries exactly one
organization and request input still contributes nothing to it; what changes is
that the set of organizations a token may legitimately name is now a table
rather than a single column.

A membership row is the entry ticket and the only entry ticket. See
`app.users.models.OrganizationMembership` for the full argument.

### `organization_id` on the three authorization tables

Roles, grants and overrides all described "what this user may do" with no
statement of *where*. That was unambiguous while a user had one organization.
With two it is a bug waiting to be filed: without this column, granting
somebody a membership in a second organization would carry every role they hold
into it, and the membership would confer authority nobody granted.

### Preserving existing access, exactly

Both halves of this migration are restatements of facts that already exist, not
new grants:

* every user gets a membership in the organization they are already in, so
  nobody gains an organization and nobody loses one;
* every role, grant and override is stamped with its owner's current
  organization, so nobody gains a permission and nobody loses one.

The columns are added nullable, backfilled, and only then made NOT NULL — so a
row the backfill somehow missed fails the migration loudly rather than being
written as an empty string that would match no organization and silently revoke
somebody's access.

Revision ID: f2a9c4e18b73
Revises: e1b7c4d92f30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f2a9c4e18b73"
down_revision = "e1b7c4d92f30"
branch_labels = None
depends_on = None


#: The three tables that describe reach, and the unique constraint each one
#: carried when reach was implicitly tenant-wide.
SCOPED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("role_assignments", "uq_role_assignment", ("user_id", "role")),
    ("access_grants", "uq_access_grant_user", ("user_id",)),
    (
        "permission_overrides",
        "uq_permission_override_user_permission",
        ("user_id", "permission"),
    ),
)


def upgrade() -> None:
    op.create_table(
        "organization_memberships",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("granted_by", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "organization_id", name="uq_membership_user_org"),
    )
    op.create_index("ix_memberships_user", "organization_memberships", ["user_id"])

    # Every existing user, in the organization they are already in. Not a new
    # grant — a restatement of the access the previous schema expressed with a
    # column, so that "may this user enter this organization" has one answer
    # everywhere instead of one answer for the home organization and another
    # for the rest.
    #
    # `granted_by` is left NULL deliberately: nobody granted these, and naming
    # an actor would be a fabricated row in a table people will read as history.
    op.execute(
        sa.text(
            """
            INSERT INTO organization_memberships
                (id, user_id, organization_id, granted_at, granted_by)
            SELECT
                'mem-' || users.id,
                users.id,
                users.organization_id,
                CURRENT_TIMESTAMP,
                NULL
            FROM users
            """
        )
    )

    for table, constraint, _ in SCOPED:
        op.add_column(table, sa.Column("organization_id", sa.String(length=64), nullable=True))
        op.execute(
            sa.text(
                f"""
                UPDATE {table}
                SET organization_id = (
                    SELECT users.organization_id
                    FROM users
                    WHERE users.id = {table}.user_id
                )
                """  # noqa: S608 - table names come from the tuple above, never input
            )
        )

    # Fail loudly rather than quietly. A NULL surviving the backfill means a row
    # whose owner does not resolve to an organization; the NOT NULL below would
    # refuse it anyway, but it would refuse it with a constraint error that says
    # nothing about which table or how many rows.
    connection = op.get_bind()
    for table, _, _ in SCOPED:
        stranded = connection.execute(
            sa.text(f"SELECT COUNT(*) FROM {table} WHERE organization_id IS NULL")  # noqa: S608
        ).scalar_one()
        if stranded:
            raise RuntimeError(
                f"{stranded} row(s) in {table} could not be attributed to an "
                f"organization. Writing them as an empty string would silently "
                f"revoke somebody's access, so the migration stops here."
            )

    for table, constraint, old_columns in SCOPED:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                "organization_id", existing_type=sa.String(length=64), nullable=False
            )
            batch.drop_constraint(constraint, type_="unique")
            batch.create_unique_constraint(
                constraint, [*old_columns[:1], "organization_id", *old_columns[1:]]
            )
            batch.create_foreign_key(
                f"fk_{table}_organization",
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="CASCADE",
            )


def downgrade() -> None:
    for table, constraint, old_columns in SCOPED:
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(f"fk_{table}_organization", type_="foreignkey")
            batch.drop_constraint(constraint, type_="unique")
            batch.create_unique_constraint(constraint, list(old_columns))
            batch.drop_column("organization_id")

    op.drop_index("ix_memberships_user", table_name="organization_memberships")
    op.drop_table("organization_memberships")
