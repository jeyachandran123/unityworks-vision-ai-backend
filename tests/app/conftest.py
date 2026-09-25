"""Fixtures for the application-foundation suite.

SQLite in memory, schema built directly. `create_all_for_tests` is named so that
its appearance outside a test is obviously wrong — production schema goes through
Alembic, where it can be reviewed and rolled back.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.auth.passwords import hash_password
from app.configuration.settings import Settings
from app.domain.models import Recorder
from app.infrastructure.database import create_all_for_tests
from app.main import create_app
from app.users.models import (
    AccessGrant,
    Organization,
    OrganizationMembership,
    RoleAssignment,
    User,
)


@pytest.fixture(autouse=True)
def _isolate_from_local_env(monkeypatch):
    """Tests read code defaults, never the developer's `.env`.

    Without this, creating a local `.env` — which every developer does — silently
    changes what the suite asserts. `test_devtools_and_live_cctv_are_off_by_default`
    started failing the moment a machine enabled DevTools locally, and
    `test_production_refuses_a_default_secret` started passing for the wrong
    reason because a real SECRET_KEY was present.

    A test that depends on an untracked file is a test that means something
    different on every machine, and the failure it produces points at the wrong
    thing entirely.
    """
    # `model_config` is a SettingsConfigDict — a dict, so setitem rather than
    # setattr. monkeypatch restores it after each test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # Environment variables leak the same way. Clear the ones the suite asserts
    # defaults for; anything else a developer exports is their own business.
    for name in (
        "SECRET_KEY",
        "DB_PASSWORD",
        "APP_DEBUG",
        "APP_ENV",
        "FEATURE_DEVTOOLS",
        "FEATURE_LIVE_CCTV",
        "SERVE_FRAMES",
        "ALLOW_EVIDENCE",
        "REDIS_ENABLED",
        "VISION_AUTOSTART",
        "CORS_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        secret_key="test-only-secret-value-not-for-any-deployment",
        database_url_override="sqlite+aiosqlite:///:memory:",
        redis_enabled=False,
        vision_autostart=False,
        feature_devtools=True,
        # No DVR is configured here, because a deployment configures none: every
        # recorder is an organization's own row, made by `make_recorder` or
        # through the API.
        #
        # A real key, so the sealing path is exercised rather than skipped.
        # Fixed rather than random: a test that fails only on some runs
        # because its key happened to be rejected is worse than no test.
        recorder_secret_key="AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
        metrics_enabled=False,
        cors_origins="http://localhost:5273",
    )


@pytest_asyncio.fixture
async def app(settings: Settings):
    application = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as _:
        # Entering the client runs lifespan, which connects the engine.
        pass
    application.state.database.connect()
    await create_all_for_tests(application.state.database)
    return application


@pytest_asyncio.fixture
async def client(app):
    # The engine and schema already exist on `app.state`; a second lifespan run
    # would dispose and rebuild them, dropping an in-memory SQLite database.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http


def make_user(
    *,
    org_id: str = "org-test",
    email: str = "manager@example.com",
    password: str = "correct-horse-battery",
    roles: tuple[str, ...] = ("restaurant_manager",),
    camera_breadth: str = "listed",
    camera_ids: str = "cam-01,cam-02",
    site_ids: str = "site-01",
) -> tuple[Organization, User]:
    org = Organization(id=org_id, name="Test Org", slug=f"{org_id}-slug")
    user = User(
        id=f"user-{email}",
        organization_id=org_id,
        email=email,
        display_name="Test User",
        password_hash=hash_password(password),
    )
    # The membership is what lets this account sign in at all: `login` binds a
    # session to an organization the user is a *member* of, and a fixture with
    # role rows and no membership would be an account whose roles describe an
    # organization it cannot enter.
    user.memberships = [
        OrganizationMembership(id=f"mem-{email}-{org_id}", user_id=user.id, organization_id=org_id)
    ]
    user.role_assignments = [
        RoleAssignment(id=f"ra-{email}-{r}", user_id=user.id, organization_id=org_id, role=r)
        for r in roles
    ]
    user.access_grants = [
        AccessGrant(
            id=f"ag-{email}",
            user_id=user.id,
            organization_id=org_id,
            camera_breadth=camera_breadth,
            camera_ids=camera_ids,
            site_ids=site_ids,
        )
    ]
    return org, user


def admit(
    user: User,
    organization_id: str,
    *,
    roles: tuple[str, ...] = (),
    camera_breadth: str = "all_in_tenant",
    camera_ids: str = "",
    site_ids: str = "",
) -> User:
    """Admit an existing user to a second organization, with reach there.

    The fixture-level equivalent of `manage.py grant-membership`. Roles and the
    camera grant are per-organization, so this writes rows naming
    `organization_id` — passing none produces somebody who may enter and can
    see nothing, which is the correct default and a state worth being able to
    construct in a test.
    """
    user.memberships.append(
        OrganizationMembership(
            id=f"mem-{user.email}-{organization_id}",
            user_id=user.id,
            organization_id=organization_id,
        )
    )
    for role in roles:
        user.role_assignments.append(
            RoleAssignment(
                id=f"ra-{user.email}-{organization_id}-{role}",
                user_id=user.id,
                organization_id=organization_id,
                role=role,
            )
        )
    if roles:
        user.access_grants.append(
            AccessGrant(
                id=f"ag-{user.email}-{organization_id}",
                user_id=user.id,
                organization_id=organization_id,
                camera_breadth=camera_breadth,
                camera_ids=camera_ids,
                site_ids=site_ids,
            )
        )
    return user


@pytest_asyncio.fixture
async def seeded(app):
    """One organization with four users covering the interesting role shapes."""
    database = app.state.database
    async with database.session_scope() as session:
        org, manager = make_user()
        session.add(org)
        session.add(manager)

        _, supervisor = make_user(
            email="supervisor@example.com",
            roles=("kitchen_supervisor",),
        )
        session.add(supervisor)

        _, developer = make_user(
            email="developer@example.com",
            roles=("developer",),
            camera_breadth="all_in_tenant",
            camera_ids="",
        )
        session.add(developer)

        _, stranded = make_user(
            email="nocameras@example.com",
            roles=("restaurant_manager",),
            camera_breadth="none",
            camera_ids="",
            site_ids="",
        )
        session.add(stranded)

        other_org = Organization(id="org-other", name="Other", slug="other-slug")
        session.add(other_org)
        _, outsider = make_user(
            org_id="org-other",
            email="outsider@example.com",
            roles=("org_admin",),
        )
        session.add(outsider)

        # One recorder per organization, shaped like a migrated one: its password
        # comes from the environment, so tests that do not care about sealed
        # credentials need no master key and type no password. Cameras built by
        # later fixtures name these ids.
        session.add(make_recorder("org-test"))
        session.add(make_recorder("org-other"))

    return app


def make_recorder(
    org_id: str = "org-test",
    *,
    recorder_id: str | None = None,
    host: str | None = None,
    ip_address: str = "",
    hostname: str = "dvr.example",
    connect_via: str | None = None,
    rtsp_port: int = 554,
    brand: str = "dahua",
    active: bool = True,
    credential_ref: str = "env:CCTV_PASSWORD",
) -> Recorder:
    """A recorder row for a fixture. Its id defaults to `rec-<org>`.

    The default credential is a migrated row's `env:` reference: most suites
    here are about cameras, not passwords, and never dial. Recorders created
    through the API — the only way the application creates one — are sealed.

    `host` is a shorthand that files the value where it belongs: a dotted IPv4
    address as `ip_address`, anything else as `hostname`. An empty `host` is a
    recorder with no address at all.
    """
    if host is not None:
        ip_address, hostname = ("", host) if not _is_ipv4(host) else (host, "")
    via = connect_via or ("ip_address" if ip_address else "hostname")
    return Recorder(
        id=recorder_id or f"rec-{org_id}",
        organization_id=org_id,
        name=f"{org_id} recorder" if recorder_id is None else recorder_id,
        ip_address=ip_address,
        hostname=hostname,
        connect_via=via,
        rtsp_port=rtsp_port,
        username="admin",
        credential_ref=credential_ref,
        brand=brand,
        is_active=active,
    )


def _is_ipv4(value: str) -> bool:
    import ipaddress

    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


async def login(client: AsyncClient, email: str, password: str = "correct-horse-battery"):
    return await client.post("/api/v1/auth/login", json={"email": email, "password": password})


async def bearer(client: AsyncClient, email: str) -> dict[str, str]:
    response = await login(client, email)
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
