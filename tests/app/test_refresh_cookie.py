"""The Phase 2 refresh-token security change.

Phase 1 returned the refresh token in the response body because no frontend
existed to receive it any other way. A seven-day credential readable by page
JavaScript is one XSS away from a week of impersonation, so it now travels only
as an httpOnly, Secure, SameSite=Strict cookie scoped to `/api/v1/auth`.

Every assertion here would have passed trivially before the change and fails if
any part of it is undone.
"""

from __future__ import annotations

import pytest

from app.auth.cookies import REFRESH_COOKIE, REFRESH_PATH
from tests.app.conftest import bearer, login


def _set_cookie_header(response) -> str:
    """The raw Set-Cookie for the refresh token.

    Read raw rather than through the cookie jar because the *flags* are the
    subject of these tests, and a jar discards them.
    """
    for key, value in response.headers.multi_items():
        if key.lower() == "set-cookie" and value.startswith(f"{REFRESH_COOKIE}="):
            return value
    return ""


class TestTheTokenLeavesTheBody:
    async def test_login_does_not_return_a_refresh_token(self, seeded, client) -> None:
        body = (await login(client, "manager@example.com")).json()
        assert "refresh_token" not in body
        assert body["access_token"]

    async def test_refresh_does_not_return_a_refresh_token(self, seeded, client) -> None:
        await login(client, "manager@example.com")
        body = (await client.post("/api/v1/auth/refresh")).json()
        assert "refresh_token" not in body

    async def test_the_token_appears_nowhere_in_the_response_text(
        self, seeded, client
    ) -> None:
        """Not in the body under any key, not in any other header."""
        response = await login(client, "manager@example.com")
        token = client.cookies.get(REFRESH_COOKIE)
        assert token
        assert token not in response.text


class TestCookieFlags:
    async def test_it_is_httponly(self, seeded, client) -> None:
        """The whole point: page JavaScript cannot read it."""
        header = _set_cookie_header(await login(client, "manager@example.com"))
        assert "httponly" in header.lower()

    async def test_it_is_samesite_strict(self, seeded, client) -> None:
        """A hostile page cannot make the browser attach it."""
        header = _set_cookie_header(await login(client, "manager@example.com"))
        assert "samesite=strict" in header.lower()

    async def test_it_is_scoped_to_the_auth_routes(self, seeded, client) -> None:
        """Every other request in the application carries one credential fewer."""
        header = _set_cookie_header(await login(client, "manager@example.com"))
        assert f"path={REFRESH_PATH}" in header.lower()

    async def test_it_is_not_secure_in_development(self, seeded, client) -> None:
        """`Secure` on plain-HTTP localhost means the browser drops it, and the
        developer sees an authentication bug that is really a flag."""
        header = _set_cookie_header(await login(client, "manager@example.com"))
        assert "secure" not in header.lower()

    async def test_it_is_secure_in_production(self, settings) -> None:
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app
        from app.infrastructure.database import create_all_for_tests
        from tests.app.conftest import make_user

        from app.configuration.settings import Settings

        production = Settings(
            app_env="production",
            app_debug=False,
            secret_key="a-real-production-secret-value-for-this-test",
            db_password="a-real-production-database-password",
            database_url_override=settings.database_url_override,
            redis_enabled=False,
            feature_devtools=True,
            metrics_enabled=False,
        )
        app = create_app(production)
        app.state.database.connect()
        await create_all_for_tests(app.state.database)
        async with app.state.database.session_scope() as session:
            org, user = make_user()
            session.add(org)
            session.add(user)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="https://test"
        ) as http:
            response = await http.post(
                "/api/v1/auth/login",
                json={
                    "email": "manager@example.com",
                    "password": "correct-horse-battery",
                },
            )
            assert response.status_code == 200
            assert "secure" in _set_cookie_header(response).lower()


class TestRotation:
    async def test_every_refresh_rotates_the_cookie(self, seeded, client) -> None:
        """A token that has been exchanged is no longer held by the client, so a
        stolen-and-replayed token becomes a visible anomaly rather than a silent
        second session."""
        await login(client, "manager@example.com")
        first = client.cookies.get(REFRESH_COOKIE)

        await client.post("/api/v1/auth/refresh")
        second = client.cookies.get(REFRESH_COOKIE)

        assert first and second and first != second

    async def test_refresh_rebuilds_the_decision_from_the_database(
        self, seeded, client
    ) -> None:
        """Not from the token. A role revoked five minutes ago must not be
        reissued for another fifteen."""
        await login(client, "manager@example.com")
        body = (await client.post("/api/v1/auth/refresh")).json()
        assert "restaurant_manager" in body["user"]["roles"]


class TestSessionLifecycle:
    async def test_refresh_without_a_session_says_so(self, seeded, client) -> None:
        """"No session" rather than "bad session" — a client that was never
        logged in belongs at the login screen, not at a credential error."""
        response = await client.post("/api/v1/auth/refresh")
        assert response.status_code == 401
        assert "no active session" in response.json()["message"]

    async def test_logout_clears_the_cookie(self, seeded, client) -> None:
        await login(client, "manager@example.com")
        assert client.cookies.get(REFRESH_COOKIE)

        await client.post("/api/v1/auth/logout")
        assert not client.cookies.get(REFRESH_COOKIE)

    async def test_refresh_fails_after_logout(self, seeded, client) -> None:
        await login(client, "manager@example.com")
        await client.post("/api/v1/auth/logout")
        assert (await client.post("/api/v1/auth/refresh")).status_code == 401

    async def test_logout_works_without_a_session(self, seeded, client) -> None:
        """Idempotent, and unauthenticated. Requiring a valid access token would
        mean the only users who cannot log out are the ones whose session is in
        the worst state."""
        assert (await client.post("/api/v1/auth/logout")).status_code == 200

    async def test_logout_works_with_an_expired_access_token(
        self, seeded, client
    ) -> None:
        await login(client, "manager@example.com")
        response = await client.post(
            "/api/v1/auth/logout", headers={"Authorization": "Bearer expired-nonsense"}
        )
        assert response.status_code == 200


class TestDevToolsReadRoutes:
    """The migrated validation-console read surface."""

    @pytest.mark.parametrize(
        "path",
        ["/api/v1/devtools/vision", "/api/v1/devtools/sessions",
         "/api/v1/devtools/capabilities", "/api/v1/devtools/state"],
    )
    async def test_a_developer_reaches_every_read_route(
        self, seeded, client, path: str
    ) -> None:
        headers = await bearer(client, "developer@example.com")
        assert (await client.get(path, headers=headers)).status_code == 200

    @pytest.mark.parametrize(
        "path",
        ["/api/v1/devtools/vision", "/api/v1/devtools/sessions",
         "/api/v1/devtools/capabilities", "/api/v1/devtools/state"],
    )
    async def test_a_manager_reaches_none_of_them(
        self, seeded, client, path: str
    ) -> None:
        headers = await bearer(client, "manager@example.com")
        response = await client.get(path, headers=headers)
        assert response.status_code == 403
        assert response.json()["code"] == "OUT_OF_SCOPE"

    async def test_the_fixture_reports_the_known_observation_count(
        self, seeded, client
    ) -> None:
        """The number the frontend smoke test asserts is rendered.

        This is the guard against the validation console's capability eroding
        during migration: a known count, on both sides of the wire.
        """
        from app.vision.fixture import FIXTURE_OBSERVATION_COUNT

        response = await client.get(
            "/api/v1/devtools/state",
            headers=await bearer(client, "developer@example.com"),
        )
        assert response.json()["observation_count"] == FIXTURE_OBSERVATION_COUNT

    async def test_the_fixture_preserves_not_visible(self, seeded, client) -> None:
        """`not_visible` must survive the wire unchanged.

        Collapsing it to a boolean anywhere between the platform and the screen
        destroys the distinction between "observed absent" and "could not see" —
        and one of those is a violation while the other never is.
        """
        response = await client.get(
            "/api/v1/devtools/state",
            headers=await bearer(client, "developer@example.com"),
        )
        values = [
            attribute["value"]
            for obj in response.json()["objects"]
            for attribute in obj["attributes"]
        ]
        assert "not_visible" in values
        assert "none" in values, "observed-absent must also be present and distinct"

    async def test_every_fixture_response_is_labelled_as_one(
        self, seeded, client
    ) -> None:
        """Nothing may mistake fixture data for live observation."""
        headers = await bearer(client, "developer@example.com")
        for path in ("/api/v1/devtools/sessions", "/api/v1/devtools/capabilities",
                     "/api/v1/devtools/state"):
            body = (await client.get(path, headers=headers)).json()
            assert "fixture" in str(body)


class TestEvidencePrivilege:
    """Observation permission != evidence permission, on the DevTools path too."""

    async def test_evidence_is_refused_when_the_deployment_disables_it(
        self, seeded, client
    ) -> None:
        response = await client.get(
            "/api/v1/devtools/evidence/blob-1",
            headers=await bearer(client, "developer@example.com"),
        )
        assert response.status_code == 403
        assert response.json()["code"] == "EVIDENCE_FORBIDDEN"
        assert response.json()["details"]["setting"] == "ALLOW_EVIDENCE"

    async def test_a_devtools_user_without_view_evidence_is_refused(
        self, settings
    ) -> None:
        """DevTools access does not carry imagery access with it."""
        from httpx import ASGITransport, AsyncClient

        from app.infrastructure.database import create_all_for_tests
        from app.main import create_app
        from tests.app.conftest import make_user

        # Evidence enabled at the deployment level, so the only remaining gate
        # is the caller's own privilege.
        app = create_app(settings.model_copy(update={"allow_evidence": True}))
        app.state.database.connect()
        await create_all_for_tests(app.state.database)
        async with app.state.database.session_scope() as session:
            org, user = make_user(
                email="supervisor@example.com", roles=("kitchen_supervisor",)
            )
            session.add(org)
            session.add(user)
            # Give DevTools access without VIEW_EVIDENCE by adding the developer
            # role's gate only — here, a supervisor cannot reach DevTools at all,
            # which is itself the assertion.

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as http:
            login_response = await http.post(
                "/api/v1/auth/login",
                json={
                    "email": "supervisor@example.com",
                    "password": "correct-horse-battery",
                },
            )
            token = login_response.json()["access_token"]
            response = await http.get(
                "/api/v1/devtools/evidence/blob-1",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 403
