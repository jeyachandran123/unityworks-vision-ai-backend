"""a recorder keeps its IP address and its domain as two values

`recorders.host` held one string that was sometimes an IP address and sometimes
a domain name. A site with both — `192.168.54.243` on its own network,
`site.freemyip.com` from anywhere else — could record only one of them, and
whoever set it up had to decide which to throw away. Both are true, and which
one this server should dial changes when the server moves, so both are kept.

### What this does

1. Adds `ip_address`, `hostname` and `connect_via`.
2. Moves each row's `host` into the column its value belongs in: a dotted IPv4
   address into `ip_address`, anything else into `hostname`, and points
   `connect_via` at it. Every recorder dials **exactly** the address it dialled
   before; nothing is resolved, guessed or tried.
3. Drops `host`.

A row whose `host` was empty — a recorder saved without an address — keeps
both new columns empty and is still not dialled, exactly as before.

### Downgrade

Restores `host` from the address each recorder was dialling. The other
address, if one was added after the upgrade, has nowhere to go in the old shape
and is dropped; the downgrade says how many were.

Revision ID: e5a1c7d9f302
Revises: c7d3e9a14b52
"""

from __future__ import annotations

import ipaddress

import sqlalchemy as sa
from alembic import op

revision = "e5a1c7d9f302"
down_revision = "c7d3e9a14b52"
branch_labels = None
depends_on = None


def _is_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def upgrade() -> None:
    connection = op.get_bind()

    with op.batch_alter_table("recorders") as batch:
        batch.add_column(
            sa.Column("ip_address", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("hostname", sa.String(length=255), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column(
                "connect_via", sa.String(length=16), nullable=False, server_default="ip_address"
            )
        )

    rows = connection.execute(sa.text("SELECT id, host FROM recorders")).all()
    for recorder_id, host in rows:
        value = (host or "").strip()
        if not value:
            continue
        if _is_ipv4(value):
            fields = {"ip_address": value, "hostname": "", "connect_via": "ip_address"}
        else:
            fields = {"ip_address": "", "hostname": value, "connect_via": "hostname"}
        connection.execute(
            sa.text(
                "UPDATE recorders SET ip_address = :ip_address, hostname = :hostname, "
                "connect_via = :connect_via WHERE id = :id"
            ),
            {**fields, "id": recorder_id},
        )

    # Every recorder that had an address must still dial that same address.
    # Checked before `host` goes, so a mismatch loses nothing.
    moved = connection.execute(
        sa.text(
            "SELECT count(*) FROM recorders WHERE host <> '' AND host <> "
            "CASE WHEN connect_via = 'hostname' THEN hostname ELSE ip_address END"
        )
    ).scalar_one()
    if moved:
        raise RuntimeError(
            f"{moved} recorder(s) would dial a different address after this upgrade; "
            "refusing to drop their old address"
        )

    with op.batch_alter_table("recorders") as batch:
        batch.drop_column("host")


def downgrade() -> None:
    connection = op.get_bind()
    with op.batch_alter_table("recorders") as batch:
        batch.add_column(
            sa.Column("host", sa.String(length=255), nullable=False, server_default="")
        )
    connection.execute(
        sa.text(
            "UPDATE recorders SET host = "
            "CASE WHEN connect_via = 'hostname' THEN hostname ELSE ip_address END"
        )
    )
    dropped = connection.execute(
        sa.text("SELECT count(*) FROM recorders WHERE ip_address <> '' AND hostname <> ''")
    ).scalar_one()
    if dropped:
        print(  # noqa: T201 - alembic output, the only place anyone will read it
            f"downgrade: {dropped} recorder(s) kept both an IP address and a domain; "
            "only the one each was dialling survives in `host`"
        )
    with op.batch_alter_table("recorders") as batch:
        batch.drop_column("connect_via")
        batch.drop_column("hostname")
        batch.drop_column("ip_address")
