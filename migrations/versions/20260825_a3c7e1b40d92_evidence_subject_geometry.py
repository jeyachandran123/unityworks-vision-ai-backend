"""evidence subject geometry

Where in a stored frame the alert's subject actually is.

The box cannot be recomputed later. It lives in a bounded in-memory ring that
holds a couple of minutes of analysed frames, and an incident is read for days;
by then the only way to produce a box would be to run a detector over the
stored JPEG and highlight whoever it found — which is *a* person in the
picture, not necessarily the one the verdict was about. So it is frozen at
capture, beside the bytes it describes.

Additive and nullable-by-default: every existing evidence row keeps its meaning
and simply has no geometry, which is exactly what was true of it.

Revision ID: a3c7e1b40d92
Revises: 91946b102297
Create Date: 2026-08-25 16:40:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'a3c7e1b40d92'
down_revision: str | None = '91946b102297'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        'evidence_records',
        sa.Column('geometry', sa.Text(), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('evidence_records', 'geometry')
