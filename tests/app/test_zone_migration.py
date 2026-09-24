"""Sites fold into zones, over data written through the previous schema.

The estate was organization → site → zone → camera. It is now organization →
zone → camera: a zone belongs to the organization, and a camera belongs to a
zone. This is the only test that runs the real migration, and it is the only
thing standing between a fold and a silently misplaced camera.

Fourteen tables carried `restaurant_id`. Two hold data on the live deployment —
`cameras` and `incidents` — and both are checked row by row here. The other
twelve are empty, and the migration refuses to move any of them that is not.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The revision immediately before zones moved. Named rather than computed, so a
#: later migration cannot silently move this test's starting point.
BEFORE = "a7e3d2c19f40"
#: The zone migration itself. Pinned rather than `head`, so later migrations
#: reshaping the same tables - recorders moved four columns off `cameras` -
#: do not change what this file is testing.
AFTER = "b4c8e1a37d90"

#: Two sites, four zones (two of them both called Kitchen), three cameras — one
#: placed, two not — and two incidents, one of each. Shaped after the live
#: deployment, where all sixteen cameras have no zone at all.
SEED = """
INSERT INTO organizations (id, name, slug, is_active, status, status_reason, created_at)
VALUES ('org-g', 'Gayathri', 'g', 1, 'active', '', CURRENT_TIMESTAMP);

INSERT INTO restaurants (id, organization_id, name, slug, timezone, is_active, created_at)
VALUES ('s-main', 'org-g', 'Gayatri Restaurant', 'gayatri', 'Asia/Kolkata', 1, CURRENT_TIMESTAMP),
       ('s-touas', 'org-g', 'Touas', 'touas', 'Asia/Kolkata', 1, CURRENT_TIMESTAMP);

INSERT INTO zones (id, restaurant_id, name, created_at)
VALUES ('z-k1', 's-main', 'Kitchen', CURRENT_TIMESTAMP),
       ('z-off', 's-main', 'Officie', CURRENT_TIMESTAMP),
       ('z-k2', 's-touas', 'Kitchen', CURRENT_TIMESTAMP),
       ('z-dine', 's-touas', 'Dine in hall', CURRENT_TIMESTAMP);

INSERT INTO cameras (id, organization_id, restaurant_id, zone_id, camera_key, name, purpose,
                     host, rtsp_port, channel, stream_type, username, credential_ref,
                     analysis_fps, enabled, analysis_enabled, created_at, updated_at)
VALUES ('c-1', 'org-g', 's-main', 'z-k1', 'cam-01', 'Prep', '', 'dvr.example', 554, 1, 'main',
        'admin', 'env:CCTV_PASSWORD', 4, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
       ('c-2', 'org-g', 's-main', NULL, 'cam-02', 'Wash', '', 'dvr.example', 554, 2, 'main',
        'admin', 'env:CCTV_PASSWORD', 4, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
       ('c-3', 'org-g', 's-touas', NULL, 'cam-03', 'Hall', '', 'dvr.example', 554, 3, 'main',
        'admin', 'env:CCTV_PASSWORD', 4, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

INSERT INTO incidents (id, organization_id, restaurant_id, zone_id, camera_key, object_id,
                       track_id, rule_id, ruleset_version, severity, summary, status,
                       created_at, observed_at, finding_snapshot, evidence_refs)
VALUES ('i-1', 'org-g', 's-main', 'z-k1', 'cam-01', 'obj-1', 'trk-1', 'hairnet-required',
        'v1', 'high', 'No head covering', 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
        '{}', '[]'),
       ('i-2', 'org-g', 's-touas', NULL, 'cam-03', 'obj-2', 'trk-2', 'gloves-required',
        'v1', 'medium', 'No gloves', 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
        '{}', '[]');
"""


def _alembic(database: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["DATABASE_URL_OVERRIDE"] = "sqlite+aiosqlite:///" + database.as_posix()
    environment["SECRET_KEY"] = "a-very-long-development-secret-key-000000"
    environment.pop("APP_ENV", None)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _seeded(directory: Path, seed: str) -> Path:
    database = directory / "before.db"
    result = _alembic(database, "upgrade", BEFORE)
    assert result.returncode == 0, result.stdout + result.stderr
    connection = sqlite3.connect(database)
    try:
        connection.executescript(seed)
        connection.commit()
    finally:
        connection.close()
    return database


def _rows(database: Path, query: str) -> list[tuple]:
    connection = sqlite3.connect(database)
    try:
        return sorted(connection.execute(query).fetchall())
    finally:
        connection.close()


def _columns(database: Path, table: str) -> set[str]:
    connection = sqlite3.connect(database)
    try:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    finally:
        connection.close()


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> Path:
    database = _seeded(tmp_path_factory.mktemp("zones"), SEED)
    result = _alembic(database, "upgrade", AFTER)
    assert result.returncode == 0, result.stdout + result.stderr
    return database


def test_every_zone_belongs_to_the_organization(migrated) -> None:
    assert _rows(migrated, "SELECT id, organization_id FROM zones") == [
        ("z-dine", "org-g"),
        ("z-k1", "org-g"),
        ("z-k2", "org-g"),
        ("z-off", "org-g"),
        ("z-site-s-main", "org-g"),
        ("z-site-s-touas", "org-g"),
    ]


def test_a_duplicate_zone_name_keeps_the_site_it_came_from(migrated) -> None:
    """Two zones called Kitchen under different sites become one list.

    Renaming both is the only honest answer: leaving two rows called `Kitchen`
    in one organization makes the camera-placement dropdown unusable, and
    picking one to keep the plain name would be arbitrary.
    """
    names = dict(_rows(migrated, "SELECT id, name FROM zones"))
    assert names["z-k1"] == "Kitchen (Gayatri Restaurant)"
    assert names["z-k2"] == "Kitchen (Touas)"
    assert names["z-off"] == "Officie"
    assert names["z-dine"] == "Dine in hall"


def test_an_unplaced_camera_lands_in_a_zone_named_after_its_site(migrated) -> None:
    """The sixteen live cameras have no zone. Inventing one would be worse than
    saying where they used to be, so the old site becomes a holding zone."""
    assert _rows(migrated, "SELECT camera_key, zone_id FROM cameras") == [
        ("cam-01", "z-k1"),
        ("cam-02", "z-site-s-main"),
        ("cam-03", "z-site-s-touas"),
    ]


def test_every_incident_keeps_a_placement(migrated) -> None:
    assert _rows(migrated, "SELECT id, zone_id FROM incidents") == [
        ("i-1", "z-k1"),
        ("i-2", "z-site-s-touas"),
    ]


def test_no_table_still_names_a_site(migrated) -> None:
    for table in (
        "zones",
        "cameras",
        "incidents",
        "camera_zone_assignments",
        "board_usage_events",
        "cutting_board_policies",
        "demography_snapshots",
        "dining_tables",
        "dish_detections",
        "patron_tokens",
        "people_count_intervals",
        "pos_connectors",
        "pos_sync_runs",
        "table_status_events",
    ):
        assert "restaurant_id" not in _columns(migrated, table), table


def test_the_sites_themselves_are_left_alone(migrated) -> None:
    """`restaurants` survives this migration unread and unreferenced.

    1,990 incidents were attributed through it. Dropping the table in the same
    step that repoints them would leave nothing to check the fold against if it
    turns out wrong.
    """
    assert _rows(migrated, "SELECT id FROM restaurants") == [("s-main",), ("s-touas",)]


def test_a_camera_now_requires_a_zone(migrated) -> None:
    connection = sqlite3.connect(migrated)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO cameras (id, organization_id, zone_id, camera_key, name, purpose,"
                " host, rtsp_port, channel, stream_type, username, credential_ref, analysis_fps,"
                " enabled, analysis_enabled, created_at, updated_at)"
                " VALUES ('c-x', 'org-g', NULL, 'cam-99', 'Nowhere', '', 'h', 554, 9, 'main',"
                " 'admin', '', 4, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
    finally:
        connection.close()


def test_module_data_stops_the_migration_rather_than_losing_its_placement(tmp_path) -> None:
    """The twelve empty tables are moved without a backfill *because* they are
    empty. A deployment that has started using one must not be folded silently.
    """
    database = _seeded(
        tmp_path,
        SEED
        + """
        INSERT INTO dining_tables (id, organization_id, restaurant_id, zone_id, table_code,
                                   seats, camera_key, region, is_active, created_at, updated_at)
        VALUES ('t-1', 'org-g', 's-main', NULL, 'T1', 4, 'cam-01', '', 1,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        """,
    )
    result = _alembic(database, "upgrade", AFTER)
    assert result.returncode != 0
    assert "dining_tables" in result.stdout + result.stderr
