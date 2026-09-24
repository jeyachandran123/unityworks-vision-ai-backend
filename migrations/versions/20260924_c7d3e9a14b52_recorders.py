"""a recorder is its own record; a camera is one of its channels

Every camera row carried its own copy of how to reach it — address, port,
account and credential reference. Sixteen cameras on one DVR meant sixteen
copies, a changed address meant sixteen edits, and adding a second DVR meant
nothing at all, because camera creation forced every row onto the one
credential the deployment was configured with.

### What this does

1. Creates `recorders`.
2. Creates **one recorder per distinct connection** already in use —
   `(organization, host, port, username, credential_ref)` — rather than one per
   organization. A camera that was on a different box from its neighbours keeps
   its own box. Collapsing them would have repointed it at another DVR's address,
   and it would have kept streaming, from the wrong place, with nothing to say
   so.
3. Points every camera at the recorder matching what it dialled, and **asserts
   none is left without one** before anything is dropped.
4. Drops `host`, `rtsp_port`, `username` and `credential_ref` from `cameras`.

### What does not change

- **Credentials.** Each recorder takes the reference its cameras already used —
  on the deployment this was written for, `env:CCTV_PASSWORD` — and nothing is
  sealed. No existing deployment has to type a password in to keep its cameras
  working. Moving a recorder onto a sealed, database-held password is a
  deliberate later act through the application.
- **The URL.** Brand `dahua` for every recorder created here, because the Dahua
  path was the only one the application could build.
- **`enabled`**, and every other camera column that stays.

### Names

The largest group in an organization is "<Organization> recorder". Any other is
"<Organization> recorder (<host>)", so the page says which box is which without
anybody having to rename them first.

### Downgrade refuses

Once a recorder has been edited, which of its values a given camera originally
held is not recorded anywhere, and a guess would put a camera on the wrong DVR.
Restore from the backup taken before the upgrade.

Revision ID: c7d3e9a14b52
Revises: b4c8e1a37d90
"""

from __future__ import annotations

from collections import defaultdict

import sqlalchemy as sa
from alembic import op

revision = "c7d3e9a14b52"
down_revision = "b4c8e1a37d90"
branch_labels = None
depends_on = None

#: The connection columns that move from each camera to its recorder.
MOVED = ("host", "rtsp_port", "username", "credential_ref")


def upgrade() -> None:
    connection = op.get_bind()

    # ── 1. the table ─────────────────────────────────────────────────────────
    op.create_table(
        "recorders",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "organization_id",
            sa.String(length=64),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("rtsp_port", sa.Integer(), nullable=False, server_default="554"),
        sa.Column("username", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("credential_ref", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("secret_ciphertext", sa.LargeBinary(), nullable=True),
        sa.Column("secret_nonce", sa.LargeBinary(), nullable=True),
        sa.Column("secret_key_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("brand", sa.String(length=32), nullable=False, server_default="dahua"),
        sa.Column("path_template", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("stream_main", sa.Integer(), nullable=True),
        sa.Column("stream_sub", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("organization_id", "name", name="uq_recorder_name"),
    )
    op.create_index("ix_recorders_organization", "recorders", ["organization_id"])

    # ── 2. one recorder per distinct connection already in use ───────────────
    names = dict(connection.execute(sa.text("SELECT id, name FROM organizations")).all())
    groups = connection.execute(
        sa.text(
            "SELECT organization_id, host, rtsp_port, username, credential_ref, count(*) "
            "FROM cameras "
            "GROUP BY organization_id, host, rtsp_port, username, credential_ref"
        )
    ).all()

    by_organization: dict[str, list[tuple]] = defaultdict(list)
    for organization_id, host, port, username, credential_ref, count in groups:
        by_organization[organization_id].append((count, host, port, username, credential_ref))

    assignments: list[dict] = []
    for organization_id, found in sorted(by_organization.items()):
        # Largest first, then by address, so the result does not depend on the
        # order the database happened to return rows in.
        found.sort(key=lambda row: (-row[0], row[1], row[2], row[3], row[4]))
        base = f"{names.get(organization_id) or organization_id} recorder"
        taken: set[str] = set()

        for position, (_, host, port, username, credential_ref) in enumerate(found, start=1):
            name = base if position == 1 else f"{base} ({host or 'no address'})"
            candidate, suffix = name, 2
            while candidate in taken:
                # Two boxes at one address differing only by account. Rare, and
                # numbered rather than guessed at.
                candidate, suffix = f"{name} #{suffix}", suffix + 1
            taken.add(candidate)

            recorder_id = f"rec-{organization_id}"[:58] + f"-{position}"
            connection.execute(
                sa.text(
                    "INSERT INTO recorders (id, organization_id, name, host, rtsp_port, "
                    "username, credential_ref, secret_key_id, brand, path_template, "
                    "is_active, created_at, updated_at) "
                    "VALUES (:id, :org, :name, :host, :port, :username, :credential_ref, "
                    "'', 'dahua', '', :active, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "id": recorder_id,
                    "org": organization_id,
                    "name": candidate,
                    "host": host,
                    "port": port,
                    "username": username,
                    "credential_ref": credential_ref,
                    "active": True,
                },
            )
            assignments.append(
                {
                    "recorder_id": recorder_id,
                    "org": organization_id,
                    "host": host,
                    "port": port,
                    "username": username,
                    "credential_ref": credential_ref,
                }
            )

    # ── 3. every camera names the recorder matching what it dialled ──────────
    op.add_column("cameras", sa.Column("recorder_id", sa.String(length=64), nullable=True))
    for assignment in assignments:
        connection.execute(
            sa.text(
                "UPDATE cameras SET recorder_id = :recorder_id "
                "WHERE organization_id = :org AND host = :host AND rtsp_port = :port "
                "AND username = :username AND credential_ref = :credential_ref"
            ),
            assignment,
        )

    stranded = connection.execute(
        sa.text("SELECT count(*) FROM cameras WHERE recorder_id IS NULL")
    ).scalar_one()
    if stranded:
        # Nothing has been dropped yet, so stopping here loses nothing.
        raise RuntimeError(
            f"{stranded} camera(s) matched no recorder; refusing to drop their "
            "connection columns, because their address would be lost"
        )

    # ── 4. the shape the application now reads ───────────────────────────────
    with op.batch_alter_table("cameras") as batch:
        batch.alter_column("recorder_id", existing_type=sa.String(length=64), nullable=False)
        batch.create_foreign_key(
            "fk_cameras_recorder",
            "recorders",
            ["recorder_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_index("ix_cameras_recorder", ["recorder_id"])
        for column in MOVED:
            batch.drop_column(column)


def downgrade() -> None:
    raise RuntimeError(
        "the recorders migration cannot be reversed: once a recorder has been edited, "
        "which of its values each camera originally held is not recorded, and a guess "
        "would put a camera on the wrong DVR. Restore from the backup taken before "
        "the upgrade."
    )
