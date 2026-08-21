"""Authentication, authorization and the API contract.

The security-critical file in this suite is the scope section. Everything else
here is ordinary foundation testing; `TestTheEmptyScopeHazard` guards the single
most dangerous line in the migration.
"""

from __future__ import annotations

import pytest

from app.authorization.model import (
    AccessDecision,
    CameraScope,
    Permission,
    Role,
    ScopeBreadth,
    permissions_for,
)
from app.errors import ScopeError
from tests.app.conftest import bearer, login


# ── configuration ────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_development_tolerates_defaults(self) -> None:
        from app.configuration.settings import Settings

        Settings(app_env="development").assert_production_safe()

    def test_production_refuses_a_default_secret(self) -> None:
        from app.configuration.settings import ConfigurationError, Settings

        with pytest.raises(ConfigurationError) as caught:
            Settings(app_env="production").assert_production_safe()
        assert "SECRET_KEY" in str(caught.value)

    def test_the_refusal_names_the_variable_and_not_its_value(self) -> None:
        """An actionable error that is not itself a disclosure."""
        from app.configuration.settings import ConfigurationError, Settings

        with pytest.raises(ConfigurationError) as caught:
            Settings(app_env="production", db_password="postgres").assert_production_safe()
        message = str(caught.value)
        assert "DB_PASSWORD" in message
        assert "postgres" not in message

    def test_production_refuses_debug(self) -> None:
        from app.configuration.settings import ConfigurationError, Settings

        with pytest.raises(ConfigurationError) as caught:
            Settings(
                app_env="production",
                app_debug=True,
                secret_key="a-real-secret",
                db_password="a-real-password",
            ).assert_production_safe()
        assert "APP_DEBUG" in str(caught.value)

    def test_a_wildcard_cors_origin_is_refused(self) -> None:
        from app.configuration.settings import Settings

        with pytest.raises(ValueError, match="explicit origins"):
            Settings(cors_origins="*")

    def test_imagery_is_off_by_default(self) -> None:
        """The validation console's launcher turned these on for a laptop.

        This backend must not inherit that. They are deployment decisions, and
        the safe default is that CCTV imagery does not leave the process.
        """
        from app.configuration.settings import Settings

        settings = Settings()
        assert settings.serve_frames is False
        assert settings.allow_evidence is False

    def test_devtools_and_live_cctv_are_off_by_default(self) -> None:
        from app.configuration.settings import Settings

        settings = Settings()
        assert settings.feature_devtools is False
        assert settings.feature_live_cctv is False


# ── passwords and tokens ─────────────────────────────────────────────────────


class TestPasswords:
    def test_a_password_verifies_against_its_own_hash(self) -> None:
        from app.auth.passwords import hash_password, verify_password

        assert verify_password("correct-horse-battery", hash_password("correct-horse-battery"))

    def test_a_wrong_password_does_not(self) -> None:
        from app.auth.passwords import hash_password, verify_password

        assert not verify_password("wrong", hash_password("correct-horse-battery"))

    def test_a_corrupt_stored_hash_fails_rather_than_raises(self) -> None:
        """A data problem must not become an availability problem."""
        from app.auth.passwords import verify_password

        assert verify_password("anything", "not-a-bcrypt-hash") is False

    def test_a_too_short_password_is_refused(self) -> None:
        from app.auth.passwords import hash_password
        from app.errors import ValidationError

        with pytest.raises(ValidationError):
            hash_password("short")

    def test_an_over_long_password_is_refused_not_truncated(self) -> None:
        """bcrypt truncates at 72 bytes; silent truncation makes two different
        passphrases interchangeable."""
        from app.auth.passwords import hash_password
        from app.errors import ValidationError

        with pytest.raises(ValidationError):
            hash_password("x" * 200)

    def test_only_the_api_key_hash_is_stored(self) -> None:
        from app.auth.passwords import api_keys_match, generate_api_key

        raw, stored = generate_api_key()
        assert raw not in stored
        assert api_keys_match(raw, stored)
        assert not api_keys_match(raw + "x", stored)


class TestTokens:
    def test_an_access_token_round_trips(self, settings) -> None:
        from app.auth.tokens import TokenService, TokenType

        service = TokenService(settings)
        token, _ = service.issue_access(subject="a@b.c", tenant_id="org-1", roles=("auditor",))
        claims = service.verify(token, expect=TokenType.ACCESS)
        assert claims.subject == "a@b.c"
        assert claims.tenant_id == "org-1"

    def test_a_refresh_token_is_not_an_access_token(self, settings) -> None:
        """Without the type check, a 7-day token is a valid 15-minute token."""
        from app.auth.tokens import TokenService, TokenType
        from app.errors import AuthenticationError

        service = TokenService(settings)
        refresh, _ = service.issue_refresh(subject="a@b.c", tenant_id="org-1")
        with pytest.raises(AuthenticationError):
            service.verify(refresh, expect=TokenType.ACCESS)

    def test_a_refresh_token_carries_no_roles(self, settings) -> None:
        """Roles in a refresh token would delay a revocation by up to a week."""
        from app.auth.tokens import TokenService, TokenType

        service = TokenService(settings)
        refresh, _ = service.issue_refresh(subject="a@b.c", tenant_id="org-1")
        assert service.verify(refresh, expect=TokenType.REFRESH).roles == ()

    def test_a_tampered_token_is_rejected(self, settings) -> None:
        from app.auth.tokens import TokenService, TokenType
        from app.errors import AuthenticationError

        service = TokenService(settings)
        token, _ = service.issue_access(subject="a@b.c", tenant_id="org-1")
        with pytest.raises(AuthenticationError):
            service.verify(token[:-4] + "AAAA", expect=TokenType.ACCESS)

    def test_a_token_from_another_secret_is_rejected(self, settings) -> None:
        from app.auth.tokens import TokenService, TokenType
        from app.errors import AuthenticationError

        from app.configuration.settings import Settings

        mine = TokenService(settings)
        # Constructed, not `model_copy`d: `model_copy(update=...)` skips
        # validation, so the SecretStr field would be left holding a raw str.
        theirs = TokenService(
            Settings(app_env="test", secret_key="an-entirely-different-secret-value")
        )
        token, _ = theirs.issue_access(subject="a@b.c", tenant_id="org-1")
        with pytest.raises(AuthenticationError):
            mine.verify(token, expect=TokenType.ACCESS)


# ── the empty-scope hazard ───────────────────────────────────────────────────


class TestTheEmptyScopeHazard:
    """Phase 0 named this the single most dangerous line in the migration.

    Vision OS reads an empty camera tuple as *every camera in the tenant*. The
    natural application-side value for "this user has no camera access" is an
    empty list. Those two facts meeting quietly would grant site-wide CCTV access
    to an account intended to have none.
    """

    def test_no_access_cannot_become_a_grant(self) -> None:
        decision = AccessDecision(
            subject="nobody@example.com",
            tenant_id="org-1",
            roles=frozenset({Role.RESTAURANT_MANAGER}),
            cameras=CameraScope.none(),
        )
        with pytest.raises(ScopeError) as caught:
            decision.to_grant()
        assert "empty camera tuple means ALL" in str(caught.value)

    def test_no_access_cannot_become_a_scope(self) -> None:
        decision = AccessDecision(
            subject="nobody@example.com",
            tenant_id="org-1",
            roles=frozenset({Role.AUDITOR}),
            cameras=CameraScope.none(),
        )
        with pytest.raises(ScopeError):
            decision.to_scope()

    def test_tenant_wide_access_must_be_stated(self) -> None:
        """The wildcard is reachable — deliberately, and only on request."""
        decision = AccessDecision(
            subject="admin@example.com",
            tenant_id="org-1",
            roles=frozenset({Role.ORG_ADMIN}),
            cameras=CameraScope.all_in_tenant(),
        )
        assert decision.to_grant().cameras == ()

    def test_listed_access_passes_exactly_those_cameras(self) -> None:
        decision = AccessDecision(
            subject="manager@example.com",
            tenant_id="org-1",
            roles=frozenset({Role.RESTAURANT_MANAGER}),
            cameras=CameraScope.listed(("cam-01", "cam-02")),
        )
        assert [str(c) for c in decision.to_grant().cameras] == ["cam-01", "cam-02"]

    def test_an_empty_listed_scope_cannot_be_constructed(self) -> None:
        """`LISTED` with nothing listed is the ambiguity itself."""
        with pytest.raises(ValueError, match="ambiguous"):
            CameraScope(breadth=ScopeBreadth.LISTED, camera_ids=())

    def test_a_wildcard_cannot_also_carry_camera_ids(self) -> None:
        with pytest.raises(ValueError, match="must be empty"):
            CameraScope(breadth=ScopeBreadth.ALL_IN_TENANT, camera_ids=("cam-01",))

    def test_an_unreadable_stored_breadth_denies(self) -> None:
        """A row written by a newer version must narrow access, never widen it."""
        from app.authorization.resolver import parse_camera_scope
        from app.users.models import AccessGrant

        grant = AccessGrant(camera_breadth="everything-please", camera_ids="cam-01")
        assert parse_camera_scope(grant).breadth is ScopeBreadth.NONE

    def test_a_missing_grant_denies(self) -> None:
        from app.authorization.resolver import parse_camera_scope

        assert parse_camera_scope(None).breadth is ScopeBreadth.NONE

    def test_an_inconsistent_listed_grant_denies(self) -> None:
        from app.authorization.resolver import parse_camera_scope
        from app.users.models import AccessGrant

        grant = AccessGrant(camera_breadth="listed", camera_ids="")
        assert parse_camera_scope(grant).breadth is ScopeBreadth.NONE


# ── scope and tenancy ────────────────────────────────────────────────────────


class TestTenantScope:
    def test_a_scope_always_names_a_tenant(self) -> None:
        decision = AccessDecision(
            subject="a@b.c",
            tenant_id="org-1",
            roles=frozenset({Role.AUDITOR}),
            cameras=CameraScope.listed(("cam-01",)),
        )
        assert str(decision.to_scope().tenant_id) == "org-1"

    def test_a_decision_without_a_tenant_cannot_exist(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            AccessDecision(
                subject="a@b.c",
                tenant_id="",
                roles=frozenset(),
                cameras=CameraScope.none(),
            )

    def test_the_platform_refuses_an_unscoped_query(self) -> None:
        """The platform's own guarantee, verified after migration."""
        from vision_os.core.model.api import Scope
        from vision_os.core.model.ids import TenantId

        with pytest.raises(ValueError, match="tenant"):
            Scope(tenant_id=TenantId(""))


# ── role → permission → action ───────────────────────────────────────────────


class TestRolesAndPermissions:
    def test_a_kitchen_supervisor_cannot_view_evidence(self) -> None:
        """The role most likely to be a shared screen on a kitchen wall."""
        granted = permissions_for(frozenset({Role.KITCHEN_SUPERVISOR}))
        assert Permission.VIEW_OBSERVATIONS in granted
        assert Permission.VIEW_EVIDENCE not in granted

    def test_an_auditor_cannot_view_live(self) -> None:
        granted = permissions_for(frozenset({Role.AUDITOR}))
        assert Permission.VIEW_EVIDENCE in granted
        assert Permission.VIEW_LIVE not in granted

    def test_only_platform_roles_reach_devtools(self) -> None:
        for role in Role:
            has = Permission.ACCESS_DEVTOOLS in permissions_for(frozenset({role}))
            assert has == role.is_platform_role, role

    def test_registering_a_demand_is_not_a_read(self) -> None:
        """It spends money and causes computation, so most roles cannot."""
        assert Permission.REGISTER_DEMAND not in permissions_for(
            frozenset({Role.RESTAURANT_MANAGER})
        )
        assert Permission.REGISTER_DEMAND in permissions_for(frozenset({Role.ORG_ADMIN}))

    def test_read_evidence_is_never_implied_by_read_observations(self) -> None:
        """The two are different acts, and the action set must reflect that."""
        from vision_os.core.model.api import Action

        supervisor = AccessDecision(
            subject="s@example.com",
            tenant_id="org-1",
            roles=frozenset({Role.KITCHEN_SUPERVISOR}),
            cameras=CameraScope.listed(("cam-01",)),
        )
        actions = supervisor.to_grant().actions
        assert Action.READ_OBSERVATIONS in actions
        assert Action.READ_EVIDENCE not in actions

    def test_an_unknown_stored_role_is_dropped(self) -> None:
        from app.authorization.resolver import parse_roles

        assert parse_roles(["auditor", "wizard"]) == frozenset({Role.AUDITOR})

    def test_an_inactive_user_reaches_nothing(self) -> None:
        from app.authorization.resolver import decide
        from tests.app.conftest import make_user

        _, user = make_user()
        user.is_active = False
        decision = decide(user)
        assert decision.roles == frozenset()
        assert not decision.cameras.grants_anything


# ── HTTP surface ─────────────────────────────────────────────────────────────


class TestHealth:
    async def test_liveness_discloses_nothing(self, client) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_readiness_reports_booleans(self, client) -> None:
        response = await client.get("/health/ready")
        body = response.json()
        assert set(body) == {"ready", "database", "cache", "vision_os"}
        assert all(isinstance(v, bool) for v in body.values())

    async def test_every_response_carries_a_request_id(self, client) -> None:
        response = await client.get("/health")
        assert response.headers.get("X-Request-Id")


class TestLogin:
    async def test_a_valid_login_returns_a_session(self, seeded, client) -> None:
        response = await login(client, "manager@example.com")
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["tenant_id"] == "org-test"
        assert "restaurant_manager" in body["user"]["roles"]

    async def test_a_wrong_password_is_refused(self, seeded, client) -> None:
        response = await login(client, "manager@example.com", "wrong-password")
        assert response.status_code == 401
        assert response.json()["code"] == "INVALID_CREDENTIALS"

    async def test_an_unknown_email_is_indistinguishable(self, seeded, client) -> None:
        """Distinguishing them turns the login form into an enumeration oracle."""
        unknown = await login(client, "nobody@example.com")
        wrong = await login(client, "manager@example.com", "wrong-password")
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json()["code"] == wrong.json()["code"]
        assert unknown.json()["message"] == wrong.json()["message"]

    async def test_no_password_or_hash_appears_in_any_response(self, seeded, client) -> None:
        response = await login(client, "manager@example.com")
        text = response.text
        assert "correct-horse-battery" not in text
        assert "password_hash" not in text
        assert "$2b$" not in text


class TestAuthenticatedAccess:
    async def test_me_reports_identity_and_reach(self, seeded, client) -> None:
        response = await client.get("/api/v1/auth/me", headers=await bearer(client, "manager@example.com"))
        assert response.status_code == 200
        body = response.json()
        assert body["camera_scope"]["breadth"] == "listed"
        assert body["camera_scope"]["camera_ids"] == ["cam-01", "cam-02"]

    async def test_no_token_is_refused(self, seeded, client) -> None:
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHENTICATED"

    async def test_a_garbage_token_is_refused(self, seeded, client) -> None:
        response = await client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-token"}
        )
        assert response.status_code == 401

    async def test_a_refresh_token_is_not_accepted_as_a_bearer(self, seeded, client) -> None:
        """Read from the cookie jar, because it is no longer in the body."""
        from app.auth.cookies import REFRESH_COOKIE

        await login(client, "manager@example.com")
        refresh_token = client.cookies.get(REFRESH_COOKIE)
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {refresh_token}"},
        )
        assert response.status_code == 401

    async def test_refresh_issues_a_new_access_token(self, seeded, client) -> None:
        """No body is sent — the cookie carries the credential."""
        await login(client, "manager@example.com")
        response = await client.post("/api/v1/auth/refresh")
        assert response.status_code == 200
        assert response.json()["access_token"]

    async def test_status_is_tenant_scoped(self, seeded, client) -> None:
        response = await client.get(
            "/api/v1/status", headers=await bearer(client, "manager@example.com")
        )
        assert response.json()["tenant_id"] == "org-test"

    async def test_status_does_not_invent_a_clean_result(self, seeded, client) -> None:
        """Reporting zero cameras and zero incidents before those features exist
        would be the exact failure this product must never commit: an empty
        answer that reads as a clean one."""
        response = await client.get(
            "/api/v1/status", headers=await bearer(client, "manager@example.com")
        )
        body = response.json()
        assert "cameras" in body["not_yet_reported"]
        assert "incidents" in body["not_yet_reported"]


class TestDevToolsAuthorization:
    async def test_a_developer_reaches_devtools(self, seeded, client) -> None:
        response = await client.get(
            "/api/v1/devtools/vision", headers=await bearer(client, "developer@example.com")
        )
        assert response.status_code == 200

    async def test_a_manager_does_not(self, seeded, client) -> None:
        """Hiding the link is the frontend's courtesy. This is the control."""
        response = await client.get(
            "/api/v1/devtools/vision", headers=await bearer(client, "manager@example.com")
        )
        assert response.status_code == 403
        assert response.json()["code"] == "OUT_OF_SCOPE"
        assert response.json()["details"]["required"] == "access_devtools"

    async def test_an_anonymous_caller_does_not(self, seeded, client) -> None:
        assert (await client.get("/api/v1/devtools/vision")).status_code == 401

    async def test_devtools_are_absent_when_the_flag_is_off(self, settings) -> None:
        """Two independent gates: the deployment's, then the user's."""
        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        off = create_app(settings.model_copy(update={"feature_devtools": False}))
        async with AsyncClient(transport=ASGITransport(app=off), base_url="http://t") as http:
            assert (await http.get("/api/v1/devtools/vision")).status_code == 404

    async def test_devtools_reports_imagery_flags_as_off(self, seeded, client) -> None:
        response = await client.get(
            "/api/v1/devtools/vision", headers=await bearer(client, "developer@example.com")
        )
        imagery = response.json()["imagery"]
        assert imagery == {"serve_frames": False, "allow_evidence": False}

    async def test_failure_injection_does_not_exist(self, seeded, client) -> None:
        """Not permission-gated — absent. It deliberately breaks the pipeline."""
        for path in (
            "/api/v1/devtools/faults",
            "/api/v1/devtools/failure",
            "/api/v1/sessions/x/faults",
        ):
            response = await client.get(
                path, headers=await bearer(client, "developer@example.com")
            )
            assert response.status_code == 404, path


class TestErrorEnvelope:
    async def test_every_error_has_the_same_shape(self, seeded, client) -> None:
        response = await client.get("/api/v1/auth/me")
        body = response.json()
        assert set(body) == {"code", "message", "retryable", "details", "request_id"}

    async def test_an_error_leaks_no_internal_detail(self, seeded, client) -> None:
        response = await login(client, "manager@example.com", "wrong-password")
        text = response.text.lower()
        for leak in ("traceback", "sqlalchemy", "site-packages", ".py\"", "select "):
            assert leak not in text

    async def test_a_vision_route_reports_unavailable_not_empty(self, seeded, client) -> None:
        """The platform is not assembled in Phase 1. That must not look like
        'the platform observed nothing' — invariant V8."""
        response = await client.get(
            "/api/v1/devtools/vision", headers=await bearer(client, "developer@example.com")
        )
        assert response.json()["assembled"] is False
        assert response.json()["reason"]
