"""The migration preserves the access that already exists.

Every other test in this directory runs against `create_all`, which builds the
schema from the models and never executes a migration. That is the right
trade-off for speed, and it means the one thing this change could get
catastrophically wrong — silently revoking or widening the access of every
account in a live database — is the one thing nothing else covers.

So this file runs the real migration, against a real database, over data
written through the *previous* schema:

    upgrade to e1b7c4d92f30   the revision before memberships existed
    write a live-looking database
    upgrade to head
    assert nothing about anyone's access changed

It is slower than its neighbours because it shells out to alembic four times.
It is worth it: "every existing account retains precisely the access it has" is
not a property you want to discover was false in production.

SQLite is used deliberately rather than as a convenience. It is the dialect
where this migration does the most work — PostgreSQL alters the columns in
place, while SQLite's `batch_alter_table` rebuilds each table and copies every
row, which is where data would actually be lost.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The revision immediately before `organization_memberships`. Named rather than
#: computed so that a later migration does not silently move the starting point
#: of this test to somewhere the backfill has already happened.
BEFORE = "e1b7c4d92f30"

#: A live-looking database, written through the schema as it stood before this
#: change: two organizations (one of them suspended), three users, several
#: roles, all three camera breadths, and both override states.
SEED = """
INSERT INTO organizations (id, name, slug, is_active, status, status_reason, created_at)
VALUES ('org-acme', 'Acme', 'acme', 1, 'active', '', CURRENT_TIMESTAMP),
       ('org-borden', 'Borden', 'borden', 1, 'suspended', 'unpaid', CURRENT_TIMESTAMP);

INSERT INTO users (id, organization_id, email, display_name, password_hash, is_active, created_at)
VALUES ('u1', 'org-acme', 'a@example.com', 'A', 'x', 1, CURRENT_TIMESTAMP),
       ('u2', 'org-acme', 'b@example.com', 'B', 'x', 1, CURRENT_TIMESTAMP),
       ('u3', 'org-borden', 'c@example.com', 'C', 'x', 1, CURRENT_TIMESTAMP);

INSERT INTO role_assignments (id, user_id, role, granted_at)
VALUES ('r1', 'u1', 'org_admin', CURRENT_TIMESTAMP),
       ('r2', 'u1', 'auditor', CURRENT_TIMESTAMP),
       ('r3', 'u2', 'restaurant_manager', CURRENT_TIMESTAMP),
       ('r4', 'u3', 'kitchen_supervisor', CURRENT_TIMESTAMP);

INSERT INTO access_grants (id, user_id, camera_breadth, camera_ids, site_ids, updated_at)
VALUES ('g1', 'u1', 'all_in_tenant', '', '', CURRENT_TIMESTAMP),
       ('g2', 'u2', 'listed', 'cam-01,cam-02', 'site-01', CURRENT_TIMESTAMP),
       ('g3', 'u3', 'none', '', '', CURRENT_TIMESTAMP);

INSERT INTO permission_overrides (id, user_id, permission, state, granted_at)
VALUES ('o1', 'u2', 'manage_sites', 'grant', CURRENT_TIMESTAMP),
       ('o2', 'u3', 'view_live', 'revoke', CURRENT_TIMESTAMP);
"""

ACCESS_QUERIES = {
    "roles": "SELECT id, user_id, role FROM role_assignments",
    "grants": ("SELECT id, user_id, camera_breadth, camera_ids, site_ids FROM access_grants"),
    "overrides": "SELECT id, user_id, permission, state FROM permission_overrides",
}


def _alembic(database: Path, *args: str) -> None:
    environment = dict(os.environ)
    environment["DATABASE_URL_OVERRIDE"] = "sqlite+aiosqlite:///" + database.as_posix()
    # Alembic loads application settings, which refuse a default secret.
    environment["SECRET_KEY"] = "a-very-long-development-secret-key-000000"
    environment.pop("APP_ENV", None)

    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"alembic {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")


def _access(database: Path) -> dict[str, list[tuple]]:
    connection = sqlite3.connect(database)
    try:
        return {
            name: sorted(connection.execute(query).fetchall())
            for name, query in ACCESS_QUERIES.items()
        }
    finally:
        connection.close()


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> Path:
    """A database seeded before the change and upgraded across it.

    Module-scoped: four alembic subprocesses is the expensive part, and every
    assertion below reads the same end state.
    """
    if shutil.which(sys.executable) is None:  # pragma: no cover - defensive
        pytest.skip("no interpreter to run alembic with")

    database = tmp_path_factory.mktemp("membership-migration") / "before.db"

    _alembic(database, "upgrade", BEFORE)

    connection = sqlite3.connect(database)
    try:
        connection.executescript(SEED)
        connection.commit()
    finally:
        connection.close()

    return database


@pytest.fixture(scope="module")
def before(migrated: Path) -> dict[str, list[tuple]]:
    return _access(migrated)


@pytest.fixture(scope="module")
def after(migrated: Path, before: dict[str, list[tuple]]) -> dict[str, list[tuple]]:
    _alembic(migrated, "upgrade", "head")
    return _access(migrated)


def test_no_role_assignment_is_lost_or_changed(before, after) -> None:
    """The rows themselves, id for id.

    Not a count: a migration that dropped one role and added another would keep
    the count and change who can do what.
    """
    assert after["roles"] == before["roles"]


def test_no_camera_grant_is_lost_or_changed(before, after) -> None:
    """All three breadths survive, including the listed camera ids.

    `none` is the one to watch. It is the row whose meaning is "this account
    reaches no camera", and a table rebuild that dropped it would turn that
    into "no grant at all" — which the resolver reads the same way, and which
    would be right by accident rather than by preservation.
    """
    assert after["grants"] == before["grants"]


def test_no_permission_override_is_lost_or_changed(before, after) -> None:
    """Both directions. A lost GRANT narrows somebody silently; a lost REVOKE
    *widens* somebody silently, which is worse."""
    assert after["overrides"] == before["overrides"]


def test_every_authorization_row_is_stamped_with_its_owners_organization(migrated, after) -> None:
    """The backfill's whole job, checked by join rather than by trusting it.

    A row stamped with the wrong organization would be invisible to `decide()`,
    which filters on exactly this column — so it would revoke access without
    deleting anything, and the row would still be there to reassure whoever
    went looking.
    """
    connection = sqlite3.connect(migrated)
    try:
        for table in ("role_assignments", "access_grants", "permission_overrides"):
            wrong = connection.execute(
                f"SELECT COUNT(*) FROM {table} t "  # noqa: S608 - fixed table names
                f"JOIN users u ON u.id = t.user_id "
                f"WHERE t.organization_id IS NOT u.organization_id"
            ).fetchone()[0]
            assert wrong == 0, f"{table} has {wrong} misattributed row(s)"
    finally:
        connection.close()


def test_every_existing_user_gains_exactly_one_membership_in_their_own_organization(
    migrated, after
) -> None:
    """One each, and no extras.

    An extra membership is not a cosmetic error: it is an account admitted to a
    customer nobody admitted them to, which is the exact failure the whole
    entry-ticket rule exists to prevent.
    """
    connection = sqlite3.connect(migrated)
    try:
        memberships = sorted(
            connection.execute(
                "SELECT user_id, organization_id FROM organization_memberships"
            ).fetchall()
        )
    finally:
        connection.close()

    assert memberships == [
        ("u1", "org-acme"),
        ("u2", "org-acme"),
        ("u3", "org-borden"),
    ]


def test_a_backfilled_membership_names_no_grantor(migrated, after) -> None:
    """Nobody granted these. They restate access that already existed, and
    naming an actor would be a fabricated row in a table people read as
    history."""
    connection = sqlite3.connect(migrated)
    try:
        named = connection.execute(
            "SELECT COUNT(*) FROM organization_memberships WHERE granted_by IS NOT NULL"
        ).fetchone()[0]
    finally:
        connection.close()

    assert named == 0


def test_the_downgrade_reverts_without_losing_access(migrated, before, after) -> None:
    """Runs last in the module, because it moves the database backwards.

    A downgrade nobody has run is a downgrade that does not work, and this one
    rebuilds three tables to remove a column. If it is ever needed it will be
    needed at the worst possible moment.
    """
    _alembic(migrated, "downgrade", BEFORE)

    assert _access(migrated) == before

    connection = sqlite3.connect(migrated)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        connection.close()

    assert "organization_memberships" not in tables
