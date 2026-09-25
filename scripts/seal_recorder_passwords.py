"""Move recorder passwords out of the server's environment and into the database.

### Why this exists

Recorders created before passwords could be typed into the application took
the credential their cameras already used — on the first deployment,
`env:CCTV_PASSWORD`, one environment variable standing in for a whole
organization's DVR. The application no longer reads a CCTV password from its
configuration, so each such recorder needs its password sealed onto its own
row, under `RECORDER_SECRET_KEY`, like every recorder created since.

The alternative is typing each password into its recorder's page. This does
the same thing for every legacy recorder at once, without anybody having to
know or retype the password.

### Safety

- Dry run by default. `--apply` writes.
- Prints recorder names and outcomes only — never a password, never a
  reference's value, never ciphertext.
- A recorder whose reference names nothing is reported and left alone.
- Each sealed recorder gets a `recorder.credential_set` audit row.

Run from the repository root, with the server's own settings:

    python -m scripts.seal_recorder_passwords
    python -m scripts.seal_recorder_passwords --apply

After `--apply`, restart the server, then remove `CCTV_PASSWORD` from its
environment and `.env`: nothing reads it any more.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from app.configuration.settings import Settings
from app.domain.audit import AuditAction, AuditTrail
from app.domain.recorders import credential_scheme, seal_legacy_credentials
from app.infrastructure.database import Database

# Imported for the side effect of registering the identity tables on the shared
# metadata. Without it, `recorders.organization_id` cannot resolve its foreign
# key to `organizations`, and the first flush fails — outside the application,
# nothing else imports them.
from app.users import models as _identity_models  # noqa: F401
from app.vision.secrets import EnvironmentSecretProvider

ACTOR = "script:seal_recorder_passwords"


def _environment(env_file: Path | None) -> dict[str, str]:
    """The process environment, with the server's `.env` beneath it.

    The legacy references were resolved against exactly this layering while
    the server read them, so a password that worked yesterday resolves here.
    A real environment variable still wins over the file.
    """
    environ = dict(os.environ)
    if env_file is not None and env_file.is_file():
        from dotenv import dotenv_values

        for name, value in dotenv_values(env_file).items():
            if value is not None and not environ.get(name):
                environ[name] = value
    return environ


async def run(*, apply: bool, env_file: Path | None) -> int:
    settings = Settings()
    resolve = EnvironmentSecretProvider(_environment(env_file)).resolve
    database = Database(settings)
    database.connect()
    try:
        async with database.session_scope() as session:
            results = await seal_legacy_credentials(
                session, settings=settings, resolve=resolve, apply=apply
            )
            for recorder, outcome in results:
                if outcome == "sealed":
                    await AuditTrail(session).record(
                        action=AuditAction.RECORDER_CREDENTIAL_SET,
                        organization_id=recorder.organization_id,
                        actor=ACTOR,
                        resource_type="recorder",
                        resource_id=recorder.id,
                        detail={"credential_scheme": "recorder", "moved_from": "environment"},
                    )
            rows = [
                (
                    outcome,
                    recorder.organization_id,
                    recorder.name,
                    credential_scheme(recorder.credential_ref),
                )
                for recorder, outcome in results
            ]
    finally:
        await database.disconnect()

    if not rows:
        print("every recorder already keeps its password in the database; nothing to do")
        return 0
    for outcome, organization, name, scheme in rows:
        print(f"{outcome:<11} {organization} / {name}  (now: {scheme})")
    unresolved = sum(1 for row in rows if row[0] == "unresolved")
    if unresolved:
        print(
            f"\n{unresolved} recorder(s) name a password this server cannot find. Set each one "
            "on the recorder's page instead."
        )
    if not apply and any(row[0] == "would_seal" for row in rows):
        print("\ndry run: nothing was changed. Re-run with --apply to seal them.")
    return 1 if unresolved else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write; the default is a dry run")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="the server's .env, read beneath the process environment (default: .env)",
    )
    arguments = parser.parse_args()
    return asyncio.run(run(apply=arguments.apply, env_file=arguments.env_file))


if __name__ == "__main__":
    raise SystemExit(main())
