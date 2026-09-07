"""permission overrides and organization status

Two additive changes, neither touching existing data:

**`permission_overrides`.** One row per (user, permission) exception to what a
user's roles would otherwise give them. Absence of a row is INHERIT — the
role's own answer stands — so this migration adds zero rows and every existing
user's effective permissions are unchanged the moment it runs.
`uq_permission_override_user_permission` mirrors `uq_role_assignment`: at most
one statement about a given (user, permission) pair, ever.

**`organizations.status`.** A three-state lifecycle (`active` | `suspended` |
`archived`) alongside the existing `is_active` boolean, not replacing it.
`server_default='active'` backfills every existing row — including
`org-unityworks` — to the value that reproduces today's behavior exactly, with
no manual data step. `nullable=False` from the start: a status column that can
be NULL would need a fourth "unknown" case threaded through every reader, and
nothing here needs one.

Additive: one new table, one new column. No existing table's data is altered.

Revision ID: d38dfad216a0
Revises: d4a1c8e37b52
Create Date: 2026-09-04 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'd38dfad216a0'
down_revision: str | None = 'd4a1c8e37b52'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        'permission_overrides',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('permission', sa.String(length=64), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('granted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('granted_by', sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'user_id', 'permission', name='uq_permission_override_user_permission'
        ),
    )

    # `server_default` as well as a Python-side default: the column is NOT
    # NULL, and every row that exists before this migration runs has to be
    # given a value by the database itself. 'active' is the only safe
    # backfill — it is what every organization meant before this column
    # existed, so no deployment's tenants silently become suspended or
    # archived by upgrading.
    op.add_column(
        'organizations',
        sa.Column(
            'status',
            sa.String(length=32),
            nullable=False,
            server_default='active',
        ),
    )


def downgrade() -> None:
    op.drop_column('organizations', 'status')
    op.drop_table('permission_overrides')
