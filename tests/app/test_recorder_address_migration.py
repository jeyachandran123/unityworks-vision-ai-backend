"""Splitting `recorders.host` keeps every recorder dialling what it dialled.

`host` held an IP address on one row and a domain on the next. The migration
files each value under the column it belongs in and points `connect_via` at it,
so the address a recorder dials — `ip_address` or `hostname`, whichever
`connect_via` names — is exactly the old `host`. These tests build the old
shape with `alembic` on a throwaway SQLite file, as every migration test here
does, and compare the dialled address before and after.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .test_recorder_migration import _alembic, _rows

BEFORE = "c7d3e9a14b52"
AFTER = "e5a1c7d9f302"

SEED = """
INSERT INTO organizations (id, name, slug, is_active, timezone, status, status_reason, created_at)
VALUES ('org-in', 'Canteen, India', 'canteen-in', 1, 'Asia/Kolkata', 'active', '', CURRENT_TIMESTAMP),
       ('org-sg', 'Main, Singapore', 'main-sg', 1, 'Asia/Singapore', 'active', '', CURRENT_TIMESTAMP);

INSERT INTO recorders (id, organization_id, name, host, rtsp_port, username, credential_ref,
                       secret_key_id, brand, path_template, is_active, created_at, updated_at)
VALUES ('r-ip', 'org-in', 'Canteen DVR', '192.168.54.243', 554, 'admin', 'env:CCTV_PASSWORD',
        '', 'hikvision', '', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
       ('r-dns', 'org-sg', 'Singapore DVR', 'site-sg.example.net', 8554, 'viewer', 'recorder:r-dns',
        'k1', 'dahua', '', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
       ('r-none', 'org-sg', 'Not wired yet', '', 554, 'admin', 'recorder:r-none',
        'k1', 'dahua', '', 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
"""


def _seeded(directory: Path) -> Path:
    database = directory / "addresses.db"
    result = _alembic(database, "upgrade", BEFORE)
    assert result.returncode == 0, result.stdout + result.stderr
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SEED)
        connection.commit()
    finally:
        connection.close()
    return database


def _dialled(database: Path) -> dict[str, str]:
    return dict(
        _rows(
            database,
            "SELECT id, CASE WHEN connect_via = 'hostname' THEN hostname ELSE ip_address END "
            "FROM recorders",
        )
    )


def test_every_recorder_dials_exactly_what_it_dialled(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    before = dict(_rows(database, "SELECT id, host FROM recorders"))

    result = _alembic(database, "upgrade", AFTER)
    assert result.returncode == 0, result.stdout + result.stderr

    assert _dialled(database) == before


def test_each_value_lands_in_the_column_it_belongs_in(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", AFTER)

    assert _rows(database, "SELECT id, ip_address, hostname, connect_via FROM recorders") == [
        ("r-dns", "", "site-sg.example.net", "hostname"),
        ("r-ip", "192.168.54.243", "", "ip_address"),
        # No address before, none now: still not dialled, exactly as before.
        ("r-none", "", "", "ip_address"),
    ]


def test_nothing_else_about_a_recorder_moves(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    query = (
        "SELECT id, organization_id, name, rtsp_port, username, credential_ref, brand, "
        "is_active FROM recorders"
    )
    before = _rows(database, query)
    _alembic(database, "upgrade", AFTER)
    assert _rows(database, query) == before


def test_the_old_column_is_gone(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", AFTER)
    columns = {row[1] for row in _rows(database, "PRAGMA table_info(recorders)")}
    assert "host" not in columns
    assert {"ip_address", "hostname", "connect_via"} <= columns


def test_downgrade_restores_the_dialled_address(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    before = dict(_rows(database, "SELECT id, host FROM recorders"))
    _alembic(database, "upgrade", AFTER)

    result = _alembic(database, "downgrade", BEFORE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert dict(_rows(database, "SELECT id, host FROM recorders")) == before
