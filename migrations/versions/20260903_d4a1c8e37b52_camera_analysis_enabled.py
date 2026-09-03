"""Separate "this camera streams" from "this camera is analysed".

One flag was doing two jobs. `enabled` started a camera's RTSP session *and*
enrolled it in the perception pipeline, so a site could not have a channel it
watches on the wall without also paying to detect, track, crop and describe
every person in it.

That mattered here for a concrete reason. This deployment has sixteen channels
and four kitchens. The other twelve are corridors, a store room and a car park:
worth seeing, worth nothing to a PPE rule. Analysing them spends detection CPU
and — far more scarce — the understander's call budget, which is a single global
allowance already fully consumed by the four kitchens alone.

`analysis_enabled` defaults to true, so every existing row keeps exactly the
behaviour it had. A site narrows it deliberately, per camera, and the choice is
durable and auditable in the same place `enabled` is.

Revision ID: d4a1c8e37b52
Revises: c9d5f21ab340
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4a1c8e37b52"
down_revision = "c9d5f21ab340"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `server_default` as well as a Python default: the column is NOT NULL and
    # existing rows have to be given a value by the database itself. True is the
    # only safe backfill — it is what every row meant before this column
    # existed, and defaulting to false would silently stop analysing every
    # camera in every deployment that runs this migration.
    op.add_column(
        "cameras",
        sa.Column(
            "analysis_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("cameras", "analysis_enabled")
