"""each camera has its own Live Wall frame rate

The wall played every camera at one rate fixed in code, so each request for a
smoother picture was a code change and a restart. The rate becomes a value on
the camera row, chosen when a camera is added and changed when it is edited.

Every existing camera gets 4, the rate the wall already played it at, so
nothing looks different after the upgrade until somebody changes it.

Revision ID: a7c3e1f09b24
Revises: e5a1c7d9f302
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a7c3e1f09b24"
down_revision = "e5a1c7d9f302"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("cameras") as batch:
        batch.add_column(sa.Column("wall_fps", sa.Float(), nullable=False, server_default="4"))


def downgrade() -> None:
    with op.batch_alter_table("cameras") as batch:
        batch.drop_column("wall_fps")
