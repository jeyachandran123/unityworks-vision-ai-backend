"""Moving a migrated recorder's password out of the environment.

Recorders created by the recorders migration kept the reference their cameras
used — `env:CCTV_PASSWORD`, one environment variable for a whole
organization's DVR. `seal_legacy_credentials` (run by
`scripts/seal_recorder_passwords.py`) seals each such password onto its own
row, so nothing about a recorder is left in the deployment's configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from app.domain.models import Recorder
from app.domain.recorder_secrets import SealedSecret, master_key, open_sealed
from app.domain.recorders import legacy_credential_recorders, seal_legacy_credentials
from app.vision.secrets import EnvironmentSecretProvider

from .conftest import make_recorder

LEGACY_PASSWORD = "from-the-old-env"


@pytest.fixture
async def recorders(seeded):
    """The seeded `rec-org-test` and `rec-org-other` are on `env:CCTV_PASSWORD`;
    one more names a variable that does not exist."""
    async with seeded.state.database.session_scope() as session:
        session.add(
            make_recorder(
                "org-test", recorder_id="rec-missing", credential_ref="env:NOT_SET_ANYWHERE"
            )
        )
    return seeded


def _resolver():
    return EnvironmentSecretProvider({"CCTV_PASSWORD": LEGACY_PASSWORD}).resolve


async def _rows(app) -> dict[str, Recorder]:
    async with app.state.database.session_scope() as session:
        return {r.id: r for r in (await session.execute(select(Recorder))).scalars().all()}


async def test_legacy_recorders_are_named(recorders) -> None:
    async with recorders.state.database.session_scope() as session:
        legacy = await legacy_credential_recorders(session)
    assert ("org-test", "org-test recorder") in legacy
    assert ("org-other", "org-other recorder") in legacy
    assert ("org-test", "rec-missing") in legacy


async def test_a_dry_run_changes_nothing(recorders, settings) -> None:
    async with recorders.state.database.session_scope() as session:
        results = await seal_legacy_credentials(
            session, settings=settings, resolve=_resolver(), apply=False
        )
    outcomes = {recorder.id: outcome for recorder, outcome in results}
    assert outcomes["rec-org-test"] == "would_seal"
    assert outcomes["rec-missing"] == "unresolved"
    rows = await _rows(recorders)
    assert rows["rec-org-test"].credential_ref == "env:CCTV_PASSWORD"
    assert rows["rec-org-test"].secret_ciphertext is None


async def test_apply_seals_each_password_onto_its_own_row(recorders, settings) -> None:
    async with recorders.state.database.session_scope() as session:
        results = await seal_legacy_credentials(
            session, settings=settings, resolve=_resolver(), apply=True
        )
    outcomes = {recorder.id: outcome for recorder, outcome in results}
    assert outcomes == {
        "rec-org-test": "sealed",
        "rec-org-other": "sealed",
        "rec-missing": "unresolved",
    }

    rows = await _rows(recorders)
    for recorder_id in ("rec-org-test", "rec-org-other"):
        row = rows[recorder_id]
        assert row.credential_ref == f"recorder:{recorder_id}"
        assert LEGACY_PASSWORD.encode() not in bytes(row.secret_ciphertext)
        sealed = SealedSecret(
            bytes(row.secret_ciphertext), bytes(row.secret_nonce), row.secret_key_id
        )
        assert open_sealed(sealed, key=master_key(settings)) == LEGACY_PASSWORD
    # Nothing to move is left alone, to be set on its page.
    assert rows["rec-missing"].credential_ref == "env:NOT_SET_ANYWHERE"

    # And a second run finds only what it could not move.
    async with recorders.state.database.session_scope() as session:
        again = await seal_legacy_credentials(
            session, settings=settings, resolve=_resolver(), apply=True
        )
    assert [(r.id, outcome) for r, outcome in again] == [("rec-missing", "unresolved")]


def test_the_command_itself_seals_a_migrated_recorder(tmp_path: Path) -> None:
    """The command as an operator runs it: a fresh process, a migrated
    database, the password in `.env`. Nothing else of the application is
    imported first — which is exactly what a unit test cannot show."""
    import os
    import sqlite3
    import subprocess
    import sys

    from .test_recorder_migration import REPO, _alembic

    database = tmp_path / "seal.db"
    result = _alembic(database, "upgrade", "head")
    assert result.returncode == 0, result.stdout + result.stderr
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            INSERT INTO organizations (id, name, slug, is_active, timezone, status,
                                       status_reason, created_at)
            VALUES ('org-a', 'A', 'a', 1, 'UTC', 'active', '', CURRENT_TIMESTAMP);
            INSERT INTO recorders (id, organization_id, name, ip_address, hostname, connect_via,
                                   rtsp_port, username, credential_ref, secret_key_id, brand,
                                   path_template, is_active, created_at, updated_at)
            VALUES ('r-legacy', 'org-a', 'Legacy DVR', '10.1.2.3', '', 'ip_address', 554,
                    'admin', 'env:CCTV_PASSWORD', '', 'dahua', '', 1,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        """
        )
        connection.commit()
    finally:
        connection.close()

    env_file = tmp_path / ".env"
    env_file.write_text(f"CCTV_PASSWORD={LEGACY_PASSWORD}\n", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CCTV_PASSWORD", "APP_ENV", "DATABASE_URL_OVERRIDE"}
    }
    environment.update(
        DATABASE_URL_OVERRIDE="sqlite+aiosqlite:///" + database.as_posix(),
        SECRET_KEY="a-very-long-development-secret-key-000000",
        RECORDER_SECRET_KEY="AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.seal_recorder_passwords",
            "--apply",
            "--env-file",
            str(env_file),
        ],
        cwd=str(REPO),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "sealed" in run.stdout
    assert LEGACY_PASSWORD not in run.stdout + run.stderr

    connection = sqlite3.connect(database)
    try:
        ref, ciphertext = connection.execute(
            "SELECT credential_ref, secret_ciphertext FROM recorders WHERE id = 'r-legacy'"
        ).fetchone()
        actions = connection.execute(
            "SELECT action FROM audit_events WHERE resource_id = 'r-legacy'"
        ).fetchall()
    finally:
        connection.close()
    assert ref == "recorder:r-legacy"
    assert LEGACY_PASSWORD.encode() not in bytes(ciphertext)
    assert actions == [("recorder.credential_set",)]


def test_the_script_reads_the_env_file_beneath_the_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.seal_recorder_passwords import _environment

    env_file = tmp_path / ".env"
    env_file.write_text("CCTV_PASSWORD=from-file\nOTHER=x\n", encoding="utf-8")

    monkeypatch.delenv("CCTV_PASSWORD", raising=False)
    from_file = _environment(env_file).get("CCTV_PASSWORD") == "from-file"
    assert from_file, "the .env value was not layered in"

    monkeypatch.setenv("CCTV_PASSWORD", "from-process")
    process_wins = _environment(env_file).get("CCTV_PASSWORD") == "from-process"
    assert process_wins, "a real environment variable must win over the file"
