"""Rename observation partitions from bare camera keys to runtime ids.

### Why this exists

`FileObservationLog` names its partition file after the `CameraId` it is
handed, and until now that was the bare `camera_key`. A camera key is unique
only inside its organization, so the first two customers to both own a
`cam-01` would have appended their observations — records about identifiable
people at work — to one shared file, with no error and no way to separate them
afterwards.

`app.domain.runtime_identity` closes that by qualifying the identity with the
tenant. Everything written from now on lands in `<org>_<key>.jsonl`. This
script moves what was written *before* now, so the existing history stays
readable rather than being silently abandoned under its old name.

### What it does and does not touch

Only files whose stem matches a `camera_key` in the database, and only when a
single organization owns that key — which is the only situation this migration
can be certain about. A key owned by two organizations is exactly the collision
the change exists to prevent, so a file under that name holds two tenants'
records mixed together and **cannot** be split by a rename. Those are reported
and left alone; separating them needs a human decision about the data, not a
script.

Kit partitions (`kit-*`) are conformance fixtures, not customer data, and are
skipped by name.

### Safety

Nothing is deleted and nothing is merged. A rename whose destination already
exists is refused, because that destination is a real partition and appending
one file onto another would corrupt the position bookkeeping
`FileObservationLog._load` rebuilds by counting lines.

Run `--dry-run` first. It is the default.

    python -m scripts.migrate_observation_partitions --dry-run
    python -m scripts.migrate_observation_partitions --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.configuration.settings import Settings  # noqa: E402
from app.domain.models import Camera  # noqa: E402
from app.domain.runtime_identity import runtime_camera_id  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402


def _partition_filename(runtime_id: str) -> str:
    """Exactly `FileObservationLog._path`'s own sanitisation.

    Duplicated deliberately rather than imported: this script must produce the
    name that adapter will look for, and copying the two-line rule is safer
    than reaching into a private method that may be refactored.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in runtime_id)
    return f"{safe}.jsonl"


async def _owners_by_key(settings: Settings) -> dict[str, list[str]]:
    database = Database(settings)
    database.connect()
    try:
        async with database.session_scope() as session:
            rows = (
                await session.execute(select(Camera.camera_key, Camera.organization_id))
            ).all()
    finally:
        await database.disconnect()

    owners: dict[str, list[str]] = {}
    for camera_key, organization_id in rows:
        owners.setdefault(camera_key, []).append(organization_id)
    return owners


async def run(*, apply: bool) -> int:
    settings = Settings()
    root = Path(settings.observation_log_path)
    if not root.exists():
        print(f"no observation log at {root}; nothing to migrate")
        return 0

    owners = await _owners_by_key(settings)

    moved = skipped = ambiguous = 0
    for path in sorted(root.glob("*.jsonl")):
        stem = path.stem
        if stem.startswith("kit-"):
            continue

        organizations = owners.get(stem)
        if not organizations:
            # Either already migrated (the stem is a runtime id, not a key) or
            # orphaned by a camera that no longer exists. Neither is this
            # script's business: a rename it cannot justify is a rename it
            # should not make.
            skipped += 1
            continue

        if len(set(organizations)) > 1:
            print(
                f"AMBIGUOUS {path.name}: '{stem}' is owned by "
                f"{sorted(set(organizations))}. This file holds more than one "
                f"organization's observations and cannot be split by renaming. "
                f"Left untouched."
            )
            ambiguous += 1
            continue

        destination = path.with_name(
            _partition_filename(runtime_camera_id(organizations[0], stem))
        )
        if destination == path:
            skipped += 1
            continue
        if destination.exists():
            print(f"REFUSED {path.name} -> {destination.name}: destination exists")
            ambiguous += 1
            continue

        print(f"{'MOVE  ' if apply else 'WOULD '} {path.name} -> {destination.name}")
        if apply:
            path.rename(destination)
        moved += 1

    verb = "moved" if apply else "would move"
    print(f"\n{verb} {moved}, skipped {skipped}, needs attention {ambiguous}")
    if not apply and moved:
        print("re-run with --apply to perform the migration")
    return 1 if ambiguous else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(apply=bool(args.apply)))


if __name__ == "__main__":
    raise SystemExit(main())
