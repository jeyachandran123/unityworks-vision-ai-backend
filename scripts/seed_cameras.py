"""Seed the DVR's channels as camera rows.

Section 5: the mapping is explicit and comes from a verified scan, not from an
assumption that camera N is channel N. It happens to be true on this device and
that is a measured fact, recorded here.

Idempotent: re-running updates rather than duplicating, so it is safe to run
after a DVR change.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configuration.settings import get_settings
from app.domain.cameras import CameraService

# Imported for the side effect of registering every table on the shared
# metadata. Without identity, the cameras table cannot resolve its own
# organization foreign key.
from app.domain import models as _domain_models  # noqa: F401
from app.infrastructure.database import Database
from app.users import models as _identity_models  # noqa: F401


async def _ensure_tenant(session, organization_id: str, restaurant_id: str) -> None:
    """Create the organization and restaurant if this is a fresh database."""
    from sqlalchemy import select

    from app.domain.models import Restaurant
    from app.users.models import Organization

    existing = await session.execute(
        select(Organization).where(Organization.id == organization_id)
    )
    if existing.scalar_one_or_none() is None:
        session.add(Organization(id=organization_id, name="Gayatri Restaurant",
                                 slug=f"{organization_id}-org"))
        await session.flush()
        print(f"  created organization '{organization_id}'")

    existing = await session.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )
    if existing.scalar_one_or_none() is None:
        session.add(Restaurant(id=restaurant_id, organization_id=organization_id,
                               name="Gayatri Restaurant", slug=restaurant_id,
                               timezone="Asia/Singapore"))
        await session.flush()
        print(f"  created restaurant '{restaurant_id}'")


async def seed(scan_path: Path, restaurant_id: str, enable: bool) -> int:
    settings = get_settings()
    rows = json.loads(scan_path.read_text(encoding="utf-8")) if scan_path.is_file() else []
    if not rows:
        print(f"no scan at {scan_path}; nothing to seed")
        return 1

    database = Database(settings)
    database.connect()
    created = updated = 0
    async with database.session_scope() as session:
        # The tenant and site a camera hangs off must exist first — a camera row
        # is meaningless without an organization to scope it to, and the foreign
        # key says so.
        await _ensure_tenant(session, settings.default_tenant_id, restaurant_id)
        service = CameraService(session)
        for row in rows:
            channel = int(row["channel"])
            key = f"cam-{channel:02d}"
            # Only channels that actually decoded a frame are seeded as enabled;
            # the rest are recorded and left dark rather than omitted.
            decoded = row.get("status") == "PASS"
            fields = {
                "name": f"Channel {channel:02d}",
                "purpose": "live monitoring",
                "host": settings.cctv_host,
                "rtsp_port": settings.cctv_rtsp_port,
                "channel": channel,
                "stream_type": "main",
                "username": settings.cctv_username,
                "credential_ref": settings.cctv_credential_ref,
                "analysis_fps": settings.cctv_analysis_fps,
                "enabled": bool(enable and decoded),
            }
            try:
                await service.get(organization_id=settings.default_tenant_id, camera_key=key)
                await service.update(
                    organization_id=settings.default_tenant_id, camera_key=key, **fields
                )
                updated += 1
            except Exception:
                await service.create(
                    organization_id=settings.default_tenant_id,
                    restaurant_id=restaurant_id,
                    camera_key=key,
                    **fields,
                )
                await session.flush()
                created += 1
            print(f"  {key} -> channel {channel:2d}  "
                  f"{'enabled' if fields['enabled'] else 'disabled'}  "
                  f"({row.get('width','?')}x{row.get('height','?')})")
    await database.disconnect()
    print(f"\n{created} created, {updated} updated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed DVR channels as cameras")
    parser.add_argument("--scan", default="../vision_os_validation_console/channel_decode.json")
    parser.add_argument("--restaurant-id", default="gayatri-main")
    parser.add_argument("--enable", action="store_true", help="enable decoded channels")
    args = parser.parse_args()
    return asyncio.run(seed(Path(args.scan), args.restaurant_id, args.enable))


if __name__ == "__main__":
    raise SystemExit(main())
