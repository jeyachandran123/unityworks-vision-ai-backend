"""two admin tiers: no super_admin, a Platform Admin with no organization

The product has two admins. The Platform Admin belongs to no organization and
can reach all of them; the Organization Admin holds full authority in one or
more organizations. This migration makes the data say so.

### 1. Refuse to merge two people

Email becomes unique across the deployment, because one person is one account
with many memberships. If two organizations already hold the same address, the
migration stops and names it. Merging two people's accounts is a decision
somebody has to make, not something a migration may guess at.

### 2. `super_admin` becomes `org_admin`

The role no longer exists. A row that would duplicate an `org_admin` row the
same user already holds in the same organization is deleted first, so the
rename cannot collide with `uq_role_assignment`.

### 3. The Platform Admin leaves every organization

For every account with a `platform_operator_grants` row: its memberships, roles,
camera grants and permission overrides are deleted, and its home organization
is set to NULL. Those rows described a tenant account the Platform Admin used to
have; he reaches every organization through the audited entry door instead, and
keeping them would list him as a member of one customer.

### 4. The default name follows the role

An operator account still carrying the seeded display name "Super Admin" is
renamed "Platform Admin", which is what the console header shows. Only that
exact default is touched; a name somebody chose is left alone.

### Downgrade

Restores per-organization email uniqueness and NOT NULL on the home
organization, and refuses if any account still has no organization — the rows
step 3 deleted cannot be recreated, and a downgrade that pretended otherwise
would leave an account nobody can sign in with.

Revision ID: a7e3d2c19f40
Revises: f2a9c4e18b73
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a7e3d2c19f40"
down_revision = "f2a9c4e18b73"
branch_labels = None
depends_on = None

#: Everything that describes an account's reach inside an organization.
TENANT_ROWS: tuple[str, ...] = (
    "organization_memberships",
    "role_assignments",
    "access_grants",
    "permission_overrides",
)

OPERATORS = "SELECT user_id FROM platform_operator_grants"


def upgrade() -> None:
    connection = op.get_bind()

    duplicates = [
        row[0]
        for row in connection.execute(
            sa.text(
                "SELECT lower(email) FROM users GROUP BY lower(email) "
                "HAVING COUNT(*) > 1 ORDER BY lower(email)"
            )
        )
    ]
    if duplicates:
        raise RuntimeError(
            "these email addresses belong to more than one account: "
            + ", ".join(duplicates)
            + ". One person must be one account before email can be unique. "
            "Decide which account each person keeps, then run this again."
        )

    op.execute(
        sa.text(
            "DELETE FROM role_assignments WHERE role = 'super_admin' AND EXISTS ("
            "  SELECT 1 FROM role_assignments other"
            "  WHERE other.user_id = role_assignments.user_id"
            "    AND other.organization_id = role_assignments.organization_id"
            "    AND other.role = 'org_admin')"
        )
    )
    op.execute(sa.text("UPDATE role_assignments SET role = 'org_admin' WHERE role = 'super_admin'"))

    for table in TENANT_ROWS:
        op.execute(sa.text(f"DELETE FROM {table} WHERE user_id IN ({OPERATORS})"))  # noqa: S608

    with op.batch_alter_table("users") as batch:
        batch.alter_column("organization_id", existing_type=sa.String(length=64), nullable=True)
        batch.drop_constraint("uq_users_org_email", type_="unique")
        batch.create_unique_constraint("uq_users_email", ["email"])

    op.execute(sa.text(f"UPDATE users SET organization_id = NULL WHERE id IN ({OPERATORS})"))  # noqa: S608
    op.execute(
        sa.text(
            "UPDATE users SET display_name = 'Platform Admin' "  # noqa: S608
            f"WHERE id IN ({OPERATORS}) AND lower(trim(display_name)) = 'super admin'"
        )
    )


def downgrade() -> None:
    connection = op.get_bind()
    homeless = connection.execute(
        sa.text("SELECT COUNT(*) FROM users WHERE organization_id IS NULL")
    ).scalar_one()
    if homeless:
        raise RuntimeError(
            f"{homeless} account(s) belong to no organization (the Platform Admin). "
            "The previous schema cannot represent that, and the memberships and "
            "roles removed on upgrade cannot be recreated. Give each such account "
            "an organization first."
        )

    with op.batch_alter_table("users") as batch:
        batch.drop_constraint("uq_users_email", type_="unique")
        batch.create_unique_constraint("uq_users_org_email", ["organization_id", "email"])
        batch.alter_column("organization_id", existing_type=sa.String(length=64), nullable=False)
