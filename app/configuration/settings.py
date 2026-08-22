"""Application configuration.

### Precedence

Resolution runs lowest-to-highest, and the last layer to speak wins:

    1. defaults in code          safe, minimal, never security-relevant
    2. .env file                 local development
    3. environment variables     deployment specifics
    4. SecretProviderPort        secrets, resolved by reference   (Phase 2)
    5. database                  per-organization settings        (Phase 4)

Layer 4 is deliberately not an override of layer 3. A secret enters the earlier
layers as a **reference** (``CCTV_CREDENTIAL_REF``) and is resolved at layer 4,
so the value itself never appears in a config file, an environment dump, a log
line or a ``repr()``. Vision OS declares ``SecretProviderPort`` for exactly this
and it is bound in Phase 2; until then the plain variables carry a
development-only marking in ``.env.example``.

Layers 4 and 5 are not implemented in Phase 1. Their position is fixed here so
that adding them later is an addition rather than a re-ordering.

### Why every Vision OS setting is *not* here

The platform's adapters read their own environment at their composition root —
``VISION_DETECTOR_PROVIDER``, ``VISION_SEMANTIC_POLICY`` and the rest. That is
the platform's design, documented in its provider modules, and duplicating those
reads into this file would create two sources of truth for the same value.

What this file holds is the subset the *application* needs to make decisions
about: which policy documents to hand the platform, and whether imagery may
leave the process at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Values that must never survive into production. Each is a usable local
#: default and a security incident on a public host.
_UNSAFE_DEFAULTS = {
    "secret_key": "dev-only-not-for-production",
    "db_password": "postgres",
}


class ConfigurationError(RuntimeError):
    """Configuration is invalid or unsafe. Raised at startup, never at request time.

    At startup because a misconfiguration discovered on the first request is a
    misconfiguration that has been silently in force since deployment.
    """


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "production", "test"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_debug: bool = False
    app_name: str = "UnityWorks Vision AI"
    secret_key: SecretStr = SecretStr(_UNSAFE_DEFAULTS["secret_key"])

    # ── Authentication ───────────────────────────────────────────────────────
    jwt_algorithm: Literal["HS256", "RS256"] = "HS256"
    jwt_private_key_path: str = ""
    jwt_public_key_path: str = ""
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    jwt_issuer: str = "unityworks-vision-ai"
    password_min_length: int = 12

    # ── Database ─────────────────────────────────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "unityworks_vision"
    db_user: str = "postgres"
    db_password: SecretStr = SecretStr(_UNSAFE_DEFAULTS["db_password"])
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False
    #: Overrides the composed URL entirely. Tests use it for SQLite; a
    #: deployment behind a managed database may use it for a full DSN.
    database_url_override: str = ""

    # ── Redis ────────────────────────────────────────────────────────────────
    #
    # Degraded, never fatal. The reference backend learned this the hard way:
    # refusing to boot without Redis took down every route, including the ones
    # that never touch it.
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: SecretStr = SecretStr("")
    redis_db: int = 0
    redis_max_connections: int = 50

    # ── Vision OS ────────────────────────────────────────────────────────────
    #
    # Paths handed to the platform's own configuration layer. The platform reads
    # its remaining settings itself; see the module docstring.
    vision_semantic_policy: str = ""
    vision_verification_rules: str = ""
    compliance_rules: str = ""
    #: Boot the platform at application startup. Off in Phase 1: no source
    #: adapter is bound yet, so a running platform would have nothing to read.
    vision_autostart: bool = False

    # ── CCTV / live runtime ──────────────────────────────────────────────────
    #
    # Nothing here starts a camera. `feature_live_cctv` gates the runtime,
    # `cctv_channels` names which channels exist, and the application lifespan
    # is the only caller that starts a session. Three deliberate acts.
    cctv_host: str = ""
    cctv_rtsp_port: int = 554
    #: EXPLICIT allowlist, e.g. "1,2,5,7". **Empty selects nothing** — a
    #: 16-channel DVR must not become 16 pipelines because nobody said otherwise.
    cctv_channels: str = ""
    cctv_stream_type: Literal["main", "sub"] = "sub"
    cctv_username: str = ""
    #: A REFERENCE, resolved through the secret provider — never a password.
    #: `env:CCTV_PASSWORD`, `file:/run/secrets/dvr`, `literal:…` (development).
    cctv_credential_ref: str = ""
    #: Independent of the camera's own frame rate: a 25 fps stream must not
    #: become 25 fps of detection and VLM work.
    cctv_analysis_fps: float = 4.0
    #: Frames held between decoder and pipeline, per camera. Bounded, always;
    #: there is no value meaning "unlimited". See app/vision/frames.py.
    cctv_queue_capacity: int = 8
    cctv_reconnect_initial_ms: float = 1_000.0
    cctv_reconnect_max_ms: float = 60_000.0
    #: 0 retries indefinitely, with the delay still capped.
    cctv_reconnect_max_attempts: int = 0
    #: Tenant that owns lifespan-started camera sessions. Phase 4 moves camera
    #: ownership into the database and this becomes a per-camera column.
    default_tenant_id: str = "default"

    # ── Evidence & imagery ───────────────────────────────────────────────────
    #
    # Both default OFF, and both are deployment decisions rather than user
    # settings. 12_SECURITY §5.3 keeps reading "a person was here" separate from
    # viewing their image; the validation console's launcher turned these on for
    # a laptop, and that is precisely the behaviour this backend must not carry.
    serve_frames: bool = False
    allow_evidence: bool = False
    evidence_store: Literal["memory", "local"] = "memory"
    evidence_path: str = "./data/evidence"
    evidence_retention_days: int = 30

    # ── Retention ────────────────────────────────────────────────────────────
    #
    # Three categories, three policies. One global rule would either delete an
    # audit trail on an evidence schedule or keep CCTV imagery on an audit
    # schedule, and both are wrong for different reasons.
    #
    # Defaults are deliberately conservative in different directions: imagery is
    # the most sensitive and expires soonest; the audit trail is the record of
    # who looked at that imagery and outlives it by far.
    evidence_retention_days: int = 30
    incident_retention_days: int = 365
    audit_retention_days: int = 730
    #: Sweeps mark and erase. Off by default: deletion should begin because a
    #: deployment decided so, not because a process started.
    retention_sweep_enabled: bool = False

    # ── Observability ────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True

    # ── CORS ─────────────────────────────────────────────────────────────────
    #
    # An explicit list. There is no value of this setting that means "any
    # origin", because a credentialed API that echoes arbitrary origins has no
    # same-origin protection left.
    cors_origins: str = "http://localhost:5273"

    # ── Feature flags ────────────────────────────────────────────────────────
    feature_devtools: bool = False
    feature_live_cctv: bool = False

    # ── Derived ──────────────────────────────────────────────────────────────

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        pwd = self.db_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.db_user}:{pwd}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def redis_url(self) -> str:
        pwd = self.redis_password.get_secret_value()
        auth = f":{pwd}@" if pwd else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard_origin(cls, value: str) -> str:
        if "*" in value:
            raise ValueError(
                "CORS_ORIGINS must name explicit origins; '*' with credentialed "
                "requests removes same-origin protection entirely"
            )
        return value

    # ── Production hardening ─────────────────────────────────────────────────

    def assert_production_safe(self) -> None:
        """Refuse to serve production traffic with development defaults.

        Names the variable at fault and **never its value**, so the error is
        actionable in a log aggregator without becoming a disclosure.
        """
        if not self.is_production:
            return

        unsafe: list[str] = []
        for name, default in _UNSAFE_DEFAULTS.items():
            current = getattr(self, name)
            value = current.get_secret_value() if isinstance(current, SecretStr) else current
            if not value or value == default:
                unsafe.append(name.upper())

        if self.app_debug:
            unsafe.append("APP_DEBUG (must be false in production)")

        if self.jwt_algorithm == "RS256" and not (
            self.jwt_private_key_path and self.jwt_public_key_path
        ):
            unsafe.append("JWT_PRIVATE_KEY_PATH / JWT_PUBLIC_KEY_PATH")

        if unsafe:
            raise ConfigurationError(
                "refusing to start in production with unset or default values for: "
                + ", ".join(sorted(unsafe))
            )


@lru_cache
def get_settings() -> Settings:
    """The process's settings. Cached — configuration is read once, at startup."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cache. For tests that construct a differently-configured app."""
    get_settings.cache_clear()


__all__ = [
    "ConfigurationError",
    "REPO_ROOT",
    "Settings",
    "get_settings",
    "reset_settings_cache",
]
