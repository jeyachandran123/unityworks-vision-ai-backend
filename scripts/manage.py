"""Account administration from the command line.

There is no user-management API yet — Phase 1 said accounts were "seeded
directly" and never supplied the thing that seeds them. This is that thing.

    python scripts/manage.py list-users
    python scripts/manage.py create-user --email you@example.com --role developer
    python scripts/manage.py reset-password --email you@example.com
    python scripts/manage.py check-password --email you@example.com
    python scripts/manage.py grant --email you@example.com --cameras all

Passwords are prompted for, never taken as an argument: an argument lands in
shell history, in `ps` output, and in any terminal recording. `--password` exists
only for non-interactive setup and says so.

Every write goes through the same `hash_password` the login path verifies
against, so a password set here always works there.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import secrets
import sys
import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth.passwords import hash_password, verify_password
from app.authorization.model import Role, ScopeBreadth
from app.configuration.settings import Settings
from app.infrastructure.database import Database
from app.users.models import AccessGrant, Organization, RoleAssignment, User

DEFAULT_ORG_ID = "org-unityworks"
DEFAULT_ORG_NAME = "UnityWorks"


async def _session(settings: Settings):
    database = Database(settings)
    database.connect()
    return database


def _read_password(explicit: str | None, *, confirm: bool = True) -> str:
    if explicit:
        print("warning: a password given as an argument is in your shell history", file=sys.stderr)
        return explicit

    first = getpass.getpass("password: ")
    if confirm:
        second = getpass.getpass("confirm : ")
        if first != second:
            raise SystemExit("passwords did not match")
    return first


async def list_users(settings: Settings) -> int:
    database = await _session(settings)
    async with database.session_scope() as session:
        result = await session.execute(
            select(User).options(
                selectinload(User.role_assignments),
                selectinload(User.access_grants),
                selectinload(User.organization),
            )
        )
        users = result.scalars().all()

        if not users:
            print("no users. create one:")
            print("  python scripts/manage.py create-user --email you@example.com --role developer")
            await database.disconnect()
            return 0

        for user in users:
            roles = ",".join(sorted(a.role for a in user.role_assignments)) or "(none)"
            grant = user.access_grants[0] if user.access_grants else None
            breadth = grant.camera_breadth if grant else "(no grant)"
            print(f"{user.email}")
            print(f"    active   : {user.is_active}   org: {user.organization_id} "
                  f"(active: {user.organization.is_active})")
            print(f"    roles    : {roles}")
            print(f"    cameras  : {breadth}")
    await database.disconnect()
    return 0


async def create_user(settings: Settings, args) -> int:
    database = await _session(settings)
    email = args.email.strip().lower()
    password = _read_password(args.password)

    async with database.session_scope() as session:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none() is not None:
            print(f"{email} already exists — use reset-password", file=sys.stderr)
            await database.disconnect()
            return 1

        org = (
            await session.execute(select(Organization).where(Organization.id == args.org))
        ).scalar_one_or_none()
        if org is None:
            org = Organization(
                id=args.org,
                name=DEFAULT_ORG_NAME if args.org == DEFAULT_ORG_ID else args.org,
                slug=args.org,
            )
            session.add(org)
            print(f"created organization {args.org}")

        user = User(
            id=uuid.uuid4().hex,
            organization_id=args.org,
            email=email,
            display_name=args.name or email.split("@")[0],
            password_hash=hash_password(password, min_length=settings.password_min_length),
        )
        session.add(user)
        session.add(RoleAssignment(id=uuid.uuid4().hex, user_id=user.id, role=args.role))
        session.add(_grant_for(user.id, args.cameras))

    await database.disconnect()
    print(f"created {email} with role {args.role}")
    return 0


async def reset_password(settings: Settings, args) -> int:
    database = await _session(settings)
    email = args.email.strip().lower()
    generated = None

    if args.generate:
        generated = secrets.token_urlsafe(18)
        password = generated
    else:
        password = _read_password(args.password)

    async with database.session_scope() as session:
        user = (
            await session.execute(select(User).where(User.email == email))
        ).scalar_one_or_none()
        if user is None:
            print(f"no such user: {email}", file=sys.stderr)
            await database.disconnect()
            return 1
        user.password_hash = hash_password(password, min_length=settings.password_min_length)
        user.is_active = True

    await database.disconnect()
    print(f"password reset for {email}")
    if generated:
        # Printed once, deliberately, because a generated password nobody sees is
        # a locked account. It is not stored anywhere and cannot be recovered.
        print(f"generated password: {generated}")
    return 0


async def check_password(settings: Settings, args) -> int:
    """Verify a password against the stored hash without logging in.

    For exactly the situation this script was written for: a 401 that could be a
    wrong password, an inactive account, or an inactive organization — and the
    login endpoint deliberately will not say which.
    """
    database = await _session(settings)
    email = args.email.strip().lower()
    password = _read_password(args.password, confirm=False)

    async with database.session_scope() as session:
        user = (
            await session.execute(
                select(User)
                .where(User.email == email)
                .options(selectinload(User.organization))
            )
        ).scalar_one_or_none()

        if user is None:
            print(f"FAIL  no user with email {email}")
            await database.disconnect()
            return 1

        matches = verify_password(password, user.password_hash)
        print(f"password matches : {matches}")
        print(f"user active      : {user.is_active}")
        print(f"org active       : {user.organization.is_active}")

        would_login = matches and user.is_active and user.organization.is_active
        print(f"=> login would {'SUCCEED' if would_login else 'FAIL'}")

    await database.disconnect()
    return 0 if would_login else 1


async def grant(settings: Settings, args) -> int:
    database = await _session(settings)
    email = args.email.strip().lower()

    async with database.session_scope() as session:
        user = (
            await session.execute(
                select(User).where(User.email == email).options(selectinload(User.access_grants))
            )
        ).scalar_one_or_none()
        if user is None:
            print(f"no such user: {email}", file=sys.stderr)
            await database.disconnect()
            return 1

        for existing in user.access_grants:
            await session.delete(existing)
        session.add(_grant_for(user.id, args.cameras))

    await database.disconnect()
    print(f"camera access for {email}: {args.cameras}")
    return 0


def _grant_for(user_id: str, cameras: str) -> AccessGrant:
    """Build an access grant from a CLI argument.

    `all` is spelled out rather than implied by an empty list, because to Vision
    OS an empty camera tuple means *every camera in the tenant*. Making the
    wildcard explicit here is the same three-state discipline the authorization
    model uses, for the same reason.
    """
    value = (cameras or "none").strip().lower()

    if value in {"all", "all_in_tenant"}:
        breadth, ids = ScopeBreadth.ALL_IN_TENANT, ""
    elif value in {"none", ""}:
        breadth, ids = ScopeBreadth.NONE, ""
    else:
        breadth, ids = ScopeBreadth.LISTED, ",".join(
            part.strip() for part in cameras.split(",") if part.strip()
        )

    return AccessGrant(
        id=uuid.uuid4().hex,
        user_id=user_id,
        # Stored lower-case, matching what the resolver parses. The seeded row
        # this script was written to fix held "ALL_IN_TENANT"; the resolver
        # lower-cases before comparing, so both work — but writing the canonical
        # form keeps the table readable.
        camera_breadth=breadth.value,
        camera_ids=ids,
        site_ids="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-users", help="show every account and what it can reach")

    create = sub.add_parser("create-user", help="create an account")
    create.add_argument("--email", required=True)
    create.add_argument("--role", default=Role.DEVELOPER.value, choices=[r.value for r in Role])
    create.add_argument("--org", default=DEFAULT_ORG_ID)
    create.add_argument("--name", default="")
    create.add_argument("--cameras", default="all", help="all | none | cam-01,cam-02")
    create.add_argument("--password", default=None, help="non-interactive only; lands in shell history")

    reset = sub.add_parser("reset-password", help="set a new password")
    reset.add_argument("--email", required=True)
    reset.add_argument("--password", default=None)
    reset.add_argument("--generate", action="store_true", help="generate one and print it once")

    check = sub.add_parser("check-password", help="explain why a login would fail")
    check.add_argument("--email", required=True)
    check.add_argument("--password", default=None)

    access = sub.add_parser("grant", help="set camera access")
    access.add_argument("--email", required=True)
    access.add_argument("--cameras", required=True, help="all | none | cam-01,cam-02")

    args = parser.parse_args()
    settings = Settings()

    handlers = {
        "list-users": lambda: list_users(settings),
        "create-user": lambda: create_user(settings, args),
        "reset-password": lambda: reset_password(settings, args),
        "check-password": lambda: check_password(settings, args),
        "grant": lambda: grant(settings, args),
    }
    return asyncio.run(handlers[args.command]())


if __name__ == "__main__":
    raise SystemExit(main())
