"""camera zone assignment history

Where a camera *was*, as distinct from where it is.

`cameras.zone_id` is current state. Answering "which zone did this observation
happen in" by joining it reads today's answer onto a past event, so moving a
camera silently relocates every reading it ever produced — the same class of
error `incidents.finding_snapshot` exists to prevent.

This table records the mapping as closed intervals instead. A camera's zone
history is the ordered set of its intervals; an assignment is closed by setting
`effective_to` and never by editing `zone_id`.

**Deliberately no backfill.** Existing cameras get no interval from this
migration. Seeding one from today's `cameras.zone_id` would assert that every
camera has always been where it is now, which is precisely the false claim the
table exists to prevent. An instant with no covering interval resolves to *no
zone recorded*, which is the truth: nobody wrote it down.

Additive: one new table, no existing table altered.

Revision ID: b7c41e08d5aa
Revises: a3c7e1b40d92
Create Date: 2026-09-02 09:10:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c41e08d5aa'
down_revision: str | None = 'a3c7e1b40d92'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        'camera_zone_assignments',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=64), nullable=False),
        sa.Column('camera_key', sa.String(length=64), nullable=False),
        sa.Column('zone_id', sa.String(length=64), nullable=True),
        sa.Column('zone_name', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('restaurant_id', sa.String(length=64), nullable=True),
        sa.Column('effective_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('effective_to', sa.DateTime(timezone=True), nullable=True),
        sa.Column('assigned_by', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_cza_camera_time',
        'camera_zone_assignments',
        ['organization_id', 'camera_key', 'effective_from'],
    )
    op.create_index(
        'ix_cza_open',
        'camera_zone_assignments',
        ['organization_id', 'camera_key', 'effective_to'],
    )


def downgrade() -> None:
    op.drop_index('ix_cza_open', table_name='camera_zone_assignments')
    op.drop_index('ix_cza_camera_time', table_name='camera_zone_assignments')
    op.drop_table('camera_zone_assignments')
