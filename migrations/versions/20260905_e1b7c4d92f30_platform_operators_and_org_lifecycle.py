"""platform operators and organization lifecycle metadata

Two changes, both in service of making multi-organization real rather than
nominal.

### `platform_operator_grants`

The principal that administers organizations. A table rather than a role,
because a role lives inside an organization and is granted by that
organization's own administrators — so if platform authority were a role, every
organization admin could mint one and the tenant boundary would be decorative.
See `app.authorization.platform` for the full argument.

**This migration seeds nobody.** A privilege that reaches every customer's data
must not arrive switched on for whoever happened to be in the database when it
was deployed. The first operator is granted deliberately, from the command
line: `python scripts/manage.py grant-operator --email ... --reason ...`.

### Organization lifecycle metadata

`status` already exists. What it lacked was any record of *when* it last
changed and *why*, which is the first thing anyone asks when a customer
reports that their cameras have stopped. Both columns are nullable with no
backfill: an organization that has never changed status has no status change
to describe, and inventing "created_at" as the answer would be a fabricated
record in a table people will read as history.

Revision ID: e1b7c4d92f30
Revises: d38dfad216a0
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e1b7c4d92f30"
down_revision = "d38dfad216a0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_operator_grants",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("granted_by", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", name="uq_platform_operator_user"),
    )

    op.add_column(
        "organizations",
        sa.Column("status_changed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("status_reason", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("organizations", "status_reason")
    op.drop_column("organizations", "status_changed_at")
    op.drop_table("platform_operator_grants")
