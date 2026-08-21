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
from app.infrastructure.database import create_all_for_tests
from app.main import create_app
from app.users.models import AccessGrant, Organization, RoleAssignment, User


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        secret_key="test-only-secret-value-not-for-any-deployment",
        database_url_override="sqlite+aiosqlite:///:memory:",
        redis_enabled=False,
        vision_autostart=False,
        feature_devtools=True,
        metrics_enabled=False,
        cors_origins="http://localhost:5273",
    )


@pytest_asyncio.fixture
async def app(settings: Settings):
    application = create_app(settings)
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as _:
        # Entering the client runs lifespan, which connects the engine.
        pass
    application.state.database.connect()
    await create_all_for_tests(application.state.database)
    return application


@pytest_asyncio.fixture
async def client(app):
    # The engine and schema already exist on `app.state`; a second lifespan run
    # would dispose and rebuild them, dropping an in-memory SQLite database.
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
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
    user.role_assignments = [
        RoleAssignment(id=f"ra-{email}-{r}", user_id=user.id, role=r) for r in roles
    ]
    user.access_grants = [
        AccessGrant(
            id=f"ag-{email}",
            user_id=user.id,
            camera_breadth=camera_breadth,
            camera_ids=camera_ids,
            site_ids=site_ids,
        )
    ]
    return org, user


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

    return app


async def login(client: AsyncClient, email: str, password: str = "correct-horse-battery"):
    return await client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )


async def bearer(client: AsyncClient, email: str) -> dict[str, str]:
    response = await login(client, email)
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
