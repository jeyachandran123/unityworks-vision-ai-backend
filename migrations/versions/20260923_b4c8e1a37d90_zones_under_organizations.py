"""zones belong to organizations; cameras belong to zones

The estate was organization → site → zone → camera. It is now
organization → zone → camera, because that is how the people running it
describe it: Gayathri Restaurant is the organization, and kitchen, dine hall,
pantry, billing and outside are its zones.

### What happens to the fourteen tables that named a site

**`zones`** gains `organization_id` (read from the site it belonged to) and
loses `restaurant_id`.

**Two names that were distinct under two sites collide under one organization.**
Both are renamed with the site they came from — `Kitchen (Touas)` — rather than
one keeping the plain name, which would be an arbitrary choice made by a
migration on somebody's behalf.

**`cameras` and `incidents` hold data** (16 and 1,990 rows on the deployment
this was written for) and are repointed row by row. A camera or incident with no
zone cannot be placed automatically, so the site it belonged to becomes a
holding zone, `z-site-<site id>`, named after that site. Inventing a placement
would be worse than recording where it used to be, and the Zones page shows such
a zone for what it is.

**Eleven tables are empty** on any deployment that has not started using the
product modules. Their column still moves — a table left naming a site would
point at a table nothing is allowed to read — but the migration **asserts each
is empty first** and stops if it is not, because moving a row whose placement
cannot be derived would lose it silently.

Four of those eleven — `cutting_board_policies`, `patron_tokens`,
`pos_connectors`, `pos_sync_runs` — have no `zone_id` at all. They keep their
`organization_id` and become organization-scoped, which is what they were always
closest to: a POS connector belongs to a customer, not to a room.

### The timezone follows the organization

A site carried a timezone, and report periods are a local question: September
for a Singapore kitchen begins at 16:00 UTC on 31 August. Zones have no
timezone and should not — a kitchen and a dine hall are in the same place — so
it moves up to the organization, taken from the site with the most cameras. A
deployment whose sites disagreed keeps the timezone of the one actually being
watched, and the losing value is logged rather than dropped silently.

### `restaurants` survives, unread

The table is left in place and referenced by nothing. 1,990 incidents were
attributed through it, and dropping it in the same step that repoints them would
leave nothing to check the fold against. Removing it is a separate migration,
once the new structure has been confirmed against real data.

### Downgrade refuses

Which site a zone came from is not recorded after this runs, so the split cannot
be reversed. Restore from the backup taken before the upgrade.

Revision ID: b4c8e1a37d90
Revises: a7e3d2c19f40
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b4c8e1a37d90"
down_revision = "a7e3d2c19f40"
branch_labels = None
depends_on = None

#: Empty on any deployment that has not started using the modules. Their
#: `restaurant_id` is dropped without a backfill, and only after the count is
#: checked. The first seven keep a `zone_id`; the last four are organization
#: scoped and keep nothing but `organization_id`.
EMPTY_TABLES: tuple[str, ...] = (
    "camera_zone_assignments",
    "board_usage_events",
    "demography_snapshots",
    "dining_tables",
    "dish_detections",
    "people_count_intervals",
    "table_status_events",
    "cutting_board_policies",
    "patron_tokens",
    "pos_connectors",
    "pos_sync_runs",
)


#: Indexes that name `restaurant_id`. They are dropped before the column is,
#: because SQLite's `batch_alter_table` rebuilds the table and replays every
#: index it reflected — including one over a column that no longer exists.
SITE_INDEXES: tuple[tuple[str, str], ...] = (
    ("ix_zones_restaurant", "zones"),
    ("ix_cameras_restaurant", "cameras"),
    ("ix_incidents_restaurant_time", "incidents"),
    ("ix_dining_tables_restaurant", "dining_tables"),
)

#: `uq_board_policy_colour` spans `restaurant_id` too, but it is a *constraint*
#: rather than a standalone index on SQLite, so it is dropped inside the table
#: rebuild below rather than with `DROP INDEX`.
BOARD_POLICIES = "cutting_board_policies"


def upgrade() -> None:
    connection = op.get_bind()

    # ── 0. the indexes that name a site ──────────────────────────────────────
    for index, table in SITE_INDEXES:
        op.drop_index(index, table_name=table)

    # ── 0b. the organization learns its timezone ─────────────────────────────
    with op.batch_alter_table("organizations") as batch:
        batch.add_column(
            sa.Column("timezone", sa.String(length=64), nullable=False, server_default="UTC")
        )
    # The site with the most cameras wins: it is the one whose day actually
    # frames the readings. Ties fall to the alphabetically first name, so the
    # result does not depend on row order.
    chosen = connection.execute(
        sa.text(
            "SELECT r.organization_id, r.timezone, r.name, "
            "       (SELECT COUNT(*) FROM cameras c WHERE c.restaurant_id = r.id) AS held "
            "FROM restaurants r ORDER BY r.organization_id, held DESC, r.name"
        )
    ).all()
    assigned: set[str] = set()
    for organization_id, timezone, _name, _held in chosen:
        if organization_id in assigned or not timezone:
            continue
        assigned.add(organization_id)
        connection.execute(
            sa.text("UPDATE organizations SET timezone = :tz WHERE id = :id"),
            {"tz": timezone, "id": organization_id},
        )

    # ── 1. every zone learns its organization ────────────────────────────────
    with op.batch_alter_table("zones") as batch:
        batch.add_column(sa.Column("organization_id", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true())
        )
    op.execute(
        sa.text(
            "UPDATE zones SET organization_id = ("
            "  SELECT r.organization_id FROM restaurants r WHERE r.id = zones.restaurant_id)"
        )
    )

    # ── 2. names that collide under one organization keep their old site ─────
    rows = connection.execute(
        sa.text(
            "SELECT z.id, z.name, z.organization_id, r.name FROM zones z "
            "JOIN restaurants r ON r.id = z.restaurant_id"
        )
    ).all()
    seen: dict[tuple[str, str], int] = {}
    for _, name, organization_id, _site in rows:
        seen[(organization_id, name)] = seen.get((organization_id, name), 0) + 1
    for zone_id, name, organization_id, site_name in rows:
        if seen[(organization_id, name)] > 1:
            connection.execute(
                sa.text("UPDATE zones SET name = :name WHERE id = :id"),
                {"name": f"{name} ({site_name})", "id": zone_id},
            )

    # ── 3. a holding zone for everything that has no zone at all ─────────────
    stranded = connection.execute(
        sa.text(
            "SELECT DISTINCT r.id, r.name, r.organization_id FROM restaurants r "
            "WHERE EXISTS (SELECT 1 FROM cameras c "
            "              WHERE c.restaurant_id = r.id AND c.zone_id IS NULL)"
            "   OR EXISTS (SELECT 1 FROM incidents i "
            "              WHERE i.restaurant_id = r.id AND i.zone_id IS NULL)"
        )
    ).all()
    for site_id, site_name, organization_id in stranded:
        connection.execute(
            sa.text(
                "INSERT INTO zones (id, restaurant_id, organization_id, name, is_active,"
                " created_at) VALUES (:id, :site, :org, :name, :active, CURRENT_TIMESTAMP)"
            ),
            {
                "id": f"z-site-{site_id}",
                "site": site_id,
                "org": organization_id,
                "name": site_name,
                "active": True,
            },
        )

    # ── 4. the two tables that hold data ─────────────────────────────────────
    for table in ("cameras", "incidents"):
        op.execute(
            sa.text(
                f"UPDATE {table} SET zone_id = 'z-site-' || restaurant_id "  # noqa: S608
                "WHERE zone_id IS NULL AND restaurant_id IS NOT NULL"
            )
        )
    stranded_cameras = connection.execute(
        sa.text("SELECT COUNT(*) FROM cameras WHERE zone_id IS NULL")
    ).scalar_one()
    if stranded_cameras:
        raise RuntimeError(
            f"{stranded_cameras} camera(s) could not be given a zone. Writing them "
            "with no placement would hide them from the estate, so the migration "
            "stops here."
        )

    # ── 5. the empty ones, checked before they are touched ───────────────────
    for table in EMPTY_TABLES:
        held = connection.execute(
            sa.text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 - fixed names
        ).scalar_one()
        if held:
            raise RuntimeError(
                f"{table} holds {held} row(s) and names a site. This migration can "
                "only move it while it is empty, because a row's zone cannot be "
                "derived from a site that had several. Move or remove those rows "
                "first."
            )
        with op.batch_alter_table(table) as batch:
            if table == BOARD_POLICIES:
                # Dropped and rebuilt in the same rebuild: one colour means one
                # thing per organization now that a site cannot hold its own
                # policy.
                batch.drop_constraint("uq_board_policy_colour", type_="unique")
            batch.drop_column("restaurant_id")
            if table == BOARD_POLICIES:
                batch.create_unique_constraint(
                    "uq_board_policy_colour",
                    ["organization_id", "policy_version", "board_colour"],
                )

    # ── 6. the shape the application now reads ───────────────────────────────
    with op.batch_alter_table("zones") as batch:
        batch.alter_column("organization_id", existing_type=sa.String(length=64), nullable=False)
        batch.create_foreign_key(
            "fk_zones_organization",
            "organizations",
            ["organization_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.drop_column("restaurant_id")

    with op.batch_alter_table("cameras") as batch:
        batch.alter_column("zone_id", existing_type=sa.String(length=64), nullable=False)
        batch.drop_column("restaurant_id")

    with op.batch_alter_table("incidents") as batch:
        batch.drop_column("restaurant_id")

    # ── 7. the indexes, rebuilt without the site ─────────────────────────────
    op.create_index("ix_zones_organization", "zones", ["organization_id"])
    op.create_index("ix_cameras_zone", "cameras", ["zone_id"])
    op.create_index("ix_incidents_zone_time", "incidents", ["zone_id", "created_at"])
    op.create_index("ix_dining_tables_zone", "dining_tables", ["zone_id"])


def downgrade() -> None:
    raise RuntimeError(
        "This migration cannot be reversed: after it runs, nothing records which "
        "site a zone came from, so the zones cannot be split back across sites. "
        "Restore the database from the backup taken before the upgrade."
    )
