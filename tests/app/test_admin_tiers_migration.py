"""The two-admin-tier migration, run for real over data from the previous schema.

`create_all` builds the schema from the models and never executes a migration,
so this is the only place the data half of the change is exercised:

    upgrade to f2a9c4e18b73   the revision before two admin tiers
    write a live-looking database
    upgrade to head
    assert: no super_admin, the Platform Admin left every organization, nobody
    else changed

SQLite, as in `test_membership_migration.py`, because `batch_alter_table`
rebuilds the table there and a rebuild is where rows would actually be lost.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BEFORE = "f2a9c4e18b73"

SEED = """
INSERT INTO organizations (id, name, slug, is_active, status, status_reason, created_at)
VALUES ('org-gayathri', 'Gayathri Restaurant', 'gayathri', 1, 'active', '', CURRENT_TIMESTAMP),
       ('org-borden', 'Borden Foods', 'borden', 1, 'active', '', CURRENT_TIMESTAMP);

INSERT INTO users (id, organization_id, email, display_name, password_hash, is_active, created_at)
VALUES ('op', 'org-gayathri', 'platform@example.com', 'Super Admin', 'x', 1, CURRENT_TIMESTAMP),
       ('sa', 'org-gayathri', 'super@example.com', 'Super Admin', 'x', 1, CURRENT_TIMESTAMP),
       ('both', 'org-borden', 'both@example.com', 'Both', 'x', 1, CURRENT_TIMESTAMP),
       ('mgr', 'org-borden', 'manager@example.com', 'Manager', 'x', 1, CURRENT_TIMESTAMP);

INSERT INTO platform_operator_grants (id, user_id, granted_at, granted_by, reason)
VALUES ('g-op', 'op', CURRENT_TIMESTAMP, NULL, 'first operator');

INSERT INTO organization_memberships (id, user_id, organization_id, granted_at)
VALUES ('m-op', 'op', 'org-gayathri', CURRENT_TIMESTAMP),
       ('m-sa', 'sa', 'org-gayathri', CURRENT_TIMESTAMP),
       ('m-both', 'both', 'org-borden', CURRENT_TIMESTAMP),
       ('m-mgr', 'mgr', 'org-borden', CURRENT_TIMESTAMP);

INSERT INTO role_assignments (id, user_id, organization_id, role, granted_at)
VALUES ('r-op', 'op', 'org-gayathri', 'super_admin', CURRENT_TIMESTAMP),
       ('r-sa', 'sa', 'org-gayathri', 'super_admin', CURRENT_TIMESTAMP),
       ('r-both-1', 'both', 'org-borden', 'super_admin', CURRENT_TIMESTAMP),
       ('r-both-2', 'both', 'org-borden', 'org_admin', CURRENT_TIMESTAMP),
       ('r-mgr', 'mgr', 'org-borden', 'restaurant_manager', CURRENT_TIMESTAMP);

INSERT INTO access_grants (id, user_id, organization_id, camera_breadth, camera_ids, site_ids, updated_at)
VALUES ('a-op', 'op', 'org-gayathri', 'all_in_tenant', '', '', CURRENT_TIMESTAMP),
       ('a-mgr', 'mgr', 'org-borden', 'listed', 'cam-01', 'site-01', CURRENT_TIMESTAMP);

-- `manage_sites` is deliberate: the permission was removed from the
-- application on 2026-09-23, and a migration must still carry the row
-- across unchanged. `parse_overrides` drops it at read time.
INSERT INTO permission_overrides (id, user_id, organization_id, permission, state, granted_at)
VALUES ('o-op', 'op', 'org-gayathri', 'view_live', 'revoke', CURRENT_TIMESTAMP),
       ('o-mgr', 'mgr', 'org-borden', 'manage_sites', 'grant', CURRENT_TIMESTAMP);
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


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> Path:
    database = _seeded(tmp_path_factory.mktemp("two-admin-tiers"), SEED)
    result = _alembic(database, "upgrade", "head")
    assert result.returncode == 0, result.stdout + result.stderr
    return database


def test_no_super_admin_remains(migrated) -> None:
    assert _rows(migrated, "SELECT id FROM role_assignments WHERE role = 'super_admin'") == []


def test_super_admin_becomes_org_admin_without_duplicates(migrated) -> None:
    assert _rows(migrated, "SELECT id, user_id, organization_id, role FROM role_assignments") == [
        ("r-both-2", "both", "org-borden", "org_admin"),
        ("r-mgr", "mgr", "org-borden", "restaurant_manager"),
        ("r-sa", "sa", "org-gayathri", "org_admin"),
    ]


def test_the_platform_admin_belongs_to_no_organization(migrated) -> None:
    assert _rows(migrated, "SELECT organization_id FROM users WHERE id = 'op'") == [(None,)]
    for table in (
        "organization_memberships",
        "role_assignments",
        "access_grants",
        "permission_overrides",
    ):
        left = _rows(migrated, f"SELECT id FROM {table} WHERE user_id = 'op'")  # noqa: S608
        assert left == [], f"{table} still holds {left} for the Platform Admin"
    assert _rows(migrated, "SELECT user_id FROM platform_operator_grants") == [("op",)]


def test_the_seeded_super_admin_name_becomes_platform_admin(migrated) -> None:
    """Only the operator's default name; an ordinary account keeps its own."""
    assert _rows(migrated, "SELECT id, display_name FROM users WHERE id IN ('op', 'sa')") == [
        ("op", "Platform Admin"),
        ("sa", "Super Admin"),
    ]


def test_nobody_else_loses_anything(migrated) -> None:
    assert _rows(migrated, "SELECT id FROM organization_memberships") == [
        ("m-both",),
        ("m-mgr",),
        ("m-sa",),
    ]
    assert _rows(migrated, "SELECT id, camera_breadth, camera_ids FROM access_grants") == [
        ("a-mgr", "listed", "cam-01")
    ]
    assert _rows(migrated, "SELECT id, permission, state FROM permission_overrides") == [
        ("o-mgr", "manage_sites", "grant")
    ]
    assert _rows(migrated, "SELECT id, organization_id FROM users WHERE id != 'op'") == [
        ("both", "org-borden"),
        ("mgr", "org-borden"),
        ("sa", "org-gayathri"),
    ]


def test_email_is_unique_across_the_deployment(migrated) -> None:
    connection = sqlite3.connect(migrated)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO users (id, organization_id, email, display_name, password_hash, "
                "is_active, created_at) VALUES ('dup', 'org-borden', 'super@example.com', "
                "'Dup', 'x', 1, CURRENT_TIMESTAMP)"
            )
    finally:
        connection.close()


def test_two_accounts_with_one_email_stop_the_migration(tmp_path) -> None:
    """The migration refuses to decide which of two people keeps an address."""
    database = _seeded(
        tmp_path,
        """
        INSERT INTO organizations (id, name, slug, is_active, status, status_reason, created_at)
        VALUES ('org-a', 'A', 'a', 1, 'active', '', CURRENT_TIMESTAMP),
               ('org-b', 'B', 'b', 1, 'active', '', CURRENT_TIMESTAMP);
        INSERT INTO users (id, organization_id, email, display_name, password_hash, is_active,
                           created_at)
        VALUES ('u1', 'org-a', 'priya@example.com', 'Priya', 'x', 1, CURRENT_TIMESTAMP),
               ('u2', 'org-b', 'priya@example.com', 'Priya', 'x', 1, CURRENT_TIMESTAMP);
        """,
    )
    result = _alembic(database, "upgrade", "head")
    assert result.returncode != 0
    assert "priya@example.com" in result.stdout + result.stderr
    assert _rows(database, "SELECT id FROM users") == [("u1",), ("u2",)]
