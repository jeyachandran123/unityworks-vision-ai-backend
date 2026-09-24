"""The recorders migration keeps every camera dialling exactly what it dialled.

Columns move on purpose — `host`, `rtsp_port`, `username` and `credential_ref`
leave the camera for its recorder — so comparing columns would prove nothing.
What must not change is the *connection each camera makes*. These tests rebuild
that connection from the old schema and from the new one and compare them.

The runs happen against a throwaway SQLite file through `alembic` itself, the
same way `test_zone_migration.py` does, because the suite otherwise builds its
schema with `create_all` and never exercises a migration at all.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The revision immediately before recorders existed.
BEFORE = "b4c8e1a37d90"

#: Two organizations, three distinct recorders' worth of connection details.
#:
#: `org-g` has fifteen cameras on one DVR and one camera on a second box — the
#: case a naive "one recorder per organization" migration would get wrong, by
#: silently repointing that camera at the other DVR's address.
SEED = """
INSERT INTO organizations (id, name, slug, is_active, timezone, status, status_reason, created_at)
VALUES ('org-g', 'Gayathri', 'gayathri', 1, 'Asia/Singapore', 'active', '', CURRENT_TIMESTAMP),
       ('org-p', 'PCC1', 'pcc1', 1, 'Asia/Singapore', 'active', '', CURRENT_TIMESTAMP),
       ('org-empty', 'Empty', 'empty', 1, 'UTC', 'active', '', CURRENT_TIMESTAMP);

INSERT INTO zones (id, organization_id, name, is_active, created_at)
VALUES ('z-g', 'org-g', 'Kitchen', 1, CURRENT_TIMESTAMP),
       ('z-p', 'org-p', 'Kitchen', 1, CURRENT_TIMESTAMP);
"""


def _camera(org: str, zone: str, key: str, channel: int, host: str, **extra) -> str:
    values = {
        "id": f"{org}-{key}",
        "organization_id": org,
        "zone_id": zone,
        "camera_key": key,
        "name": f"Camera {key}",
        "purpose": "",
        "host": host,
        "rtsp_port": extra.get("rtsp_port", 554),
        "channel": channel,
        "stream_type": extra.get("stream_type", "sub"),
        "username": extra.get("username", "admin"),
        "credential_ref": extra.get("credential_ref", "env:CCTV_PASSWORD"),
        "analysis_fps": 4.0,
        "enabled": extra.get("enabled", 1),
        "analysis_enabled": 1,
    }
    columns = ", ".join(values) + ", created_at, updated_at"
    rendered = ", ".join(
        f"'{value}'" if isinstance(value, str) else str(value) for value in values.values()
    )
    return f"INSERT INTO cameras ({columns}) VALUES ({rendered}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);"


CAMERAS = "\n".join(
    [
        *(_camera("org-g", "z-g", f"cam-{n:02d}", n, "gayatri.freemyip.com") for n in range(1, 16)),
        # The one on a different box. Its own recorder, not the big one's.
        _camera("org-g", "z-g", "cam-16", 1, "10.0.4.9", username="viewer"),
        _camera("org-p", "z-p", "cam-01", 3, "pcc1.example", stream_type="main", enabled=0),
    ]
)


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


def _seeded(directory: Path) -> Path:
    database = directory / "recorders.db"
    result = _alembic(database, "upgrade", BEFORE)
    assert result.returncode == 0, result.stdout + result.stderr
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SEED + CAMERAS)
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


def _connections_before(database: Path) -> dict[tuple[str, str], tuple]:
    """What each camera dialled, read from the camera row itself."""
    return {
        (org, key): (host, port, username, credential_ref, channel, stream_type)
        for org, key, host, port, username, credential_ref, channel, stream_type in _rows(
            database,
            "SELECT organization_id, camera_key, host, rtsp_port, username, credential_ref, "
            "channel, stream_type FROM cameras",
        )
    }


def _connections_after(database: Path) -> dict[tuple[str, str], tuple]:
    """What each camera dials now, read through its recorder."""
    return {
        (org, key): (host, port, username, credential_ref, channel, stream_type)
        for org, key, host, port, username, credential_ref, channel, stream_type in _rows(
            database,
            "SELECT c.organization_id, c.camera_key, r.host, r.rtsp_port, r.username, "
            "r.credential_ref, c.channel, c.stream_type "
            "FROM cameras c JOIN recorders r ON r.id = c.recorder_id",
        )
    }


def test_every_camera_keeps_the_connection_it_had(tmp_path: Path) -> None:
    """The only property that matters, stated directly."""
    database = _seeded(tmp_path)
    before = _connections_before(database)

    result = _alembic(database, "upgrade", "head")
    assert result.returncode == 0, result.stdout + result.stderr

    assert _connections_after(database) == before


def test_a_camera_on_a_different_box_gets_its_own_recorder(tmp_path: Path) -> None:
    """One recorder per organization would have repointed cam-16 at the big DVR,
    and it would have kept streaming — from the wrong place, with nothing to say
    so."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    recorders = _rows(
        database, "SELECT organization_id, host, username FROM recorders ORDER BY organization_id"
    )
    assert recorders == [
        ("org-g", "10.0.4.9", "viewer"),
        ("org-g", "gayatri.freemyip.com", "admin"),
        ("org-p", "pcc1.example", "admin"),
    ]


def test_an_organization_without_cameras_gets_no_recorder(tmp_path: Path) -> None:
    """A recorder invented for nothing would be a box on the Recorders page that
    does not exist."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    assert _rows(
        database, "SELECT count(*) FROM recorders WHERE organization_id = 'org-empty'"
    ) == [(0,)]


def test_existing_deployments_keep_their_environment_password(tmp_path: Path) -> None:
    """Nobody has to type a password in for the cameras they already have to
    keep working. The reference moves to the recorder unchanged, and nothing is
    sealed that was not sealed before."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    assert _rows(
        database,
        "SELECT DISTINCT credential_ref, secret_ciphertext IS NULL, secret_nonce IS NULL "
        "FROM recorders",
    ) == [("env:CCTV_PASSWORD", 1, 1)]


def test_the_largest_group_is_named_for_its_organization(tmp_path: Path) -> None:
    """Names are what people read on the Recorders page. The main DVR is simply
    "Gayathri recorder"; the second box says which address it is."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    names = dict(
        _rows(database, "SELECT host, name FROM recorders WHERE organization_id = 'org-g'")
    )
    assert names["gayatri.freemyip.com"] == "Gayathri recorder"
    assert names["10.0.4.9"] == "Gayathri recorder (10.0.4.9)"


def test_recorders_start_active_with_the_brand_that_was_hardcoded(tmp_path: Path) -> None:
    """Every camera was dialled with the Dahua path before this migration,
    because that was the only path there was."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    assert _rows(database, "SELECT DISTINCT brand, is_active FROM recorders") == [("dahua", 1)]


def test_no_camera_is_left_without_a_recorder(tmp_path: Path) -> None:
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    assert _rows(database, "SELECT count(*) FROM cameras WHERE recorder_id IS NULL") == [(0,)]


def test_the_columns_that_moved_are_gone_from_cameras(tmp_path: Path) -> None:
    """Left behind they would be read by something later and silently disagree
    with the recorder they were copied from."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    columns = {row[0] for row in _rows(database, "SELECT name FROM pragma_table_info('cameras')")}
    assert columns.isdisjoint({"host", "rtsp_port", "username", "credential_ref"})
    assert "recorder_id" in columns


def test_enabled_flags_survive(tmp_path: Path) -> None:
    """A disabled camera must not come back enabled because its row was rebuilt."""
    database = _seeded(tmp_path)
    before = _rows(database, "SELECT organization_id, camera_key, enabled FROM cameras")
    _alembic(database, "upgrade", "head")

    assert _rows(database, "SELECT organization_id, camera_key, enabled FROM cameras") == before


def test_downgrade_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """Once a recorder has been edited, which of its values a camera originally
    held is not recorded anywhere. Restore from the backup instead."""
    database = _seeded(tmp_path)
    _alembic(database, "upgrade", "head")

    result = _alembic(database, "downgrade", BEFORE)
    assert result.returncode != 0
    assert "backup" in (result.stdout + result.stderr).lower()
