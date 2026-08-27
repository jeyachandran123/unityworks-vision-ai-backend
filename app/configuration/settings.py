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

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
    #: How often the compliance pass reads Vision State and moves the incident
    #: queue. A timer rather than a subscription keeps a slow database off the
    #: platform's publish path; the cost is latency bounded by this interval.
    compliance_interval_s: float = 5.0
    #: Boot the platform at application startup. Off in Phase 1: no source
    #: adapter is bound yet, so a running platform would have nothing to read.
    #: Open a live viewing stream per enabled camera at start-up.
    #:
    #: On by default, unlike the Vision OS flags: a monitoring wall is what an
    #: operator expects a CCTV product to do, and it costs a decode per camera
    #: rather than a model call per person.
    feature_camera_wall: bool = True
    vision_autostart: bool = False
    #: Bind detection and tracking at start-up.
    #:
    #: Off by default because binding loads detector weights and warms a device,
    #: which a deployment that only wants the observation API should not pay for.
    #: With it off the platform still registers, crops, understands and serves —
    #: what stops is finding things in pixels, and `/devtools/architecture`
    #: reports the layer as unbound rather than letting it look absent.
    vision_bind_perception: bool = False
    #: How many cameras this deployment will actually feed into the platform.
    #:
    #: The platform sizes its frame pool as
    #: ``slots_per_camera × len(cameras) × jitter_factor``, and this
    #: application deliberately declares **no** cameras in the platform
    #: configuration document — they are a domain entity, and listing them
    #: there would start a second acquisition path for cameras the app is
    #: already feeding. The cost of that honesty is that `len(cameras)` is 0,
    #: `max(1, 0)` is 1, and the pool comes out sized for a single camera no
    #: matter how many are running.
    #:
    #: Four live cameras against a one-camera pool exhausted it continuously:
    #: every frame failed to publish with `PoolExhaustedError`, so nothing
    #: reached detection at all while the sessions reported themselves healthy.
    #: This states the fan-in explicitly rather than inferring it from a list
    #: that is empty on purpose.
    vision_analysis_cameras: int = 1
    #: Run the platform on a virtual clock.
    #:
    #: **Off.** A virtual clock does not advance unless a test advances it, so
    #: every duration the platform measures against it — inference timeouts,
    #: lease deadlines, attribute validity windows, staleness — stops meaning
    #: elapsed time. Determinism is worth having in a replay-verification run and
    #: is wrong everywhere else, so it is named rather than inherited.
    vision_deterministic_clock: bool = False
    #: How old an attribute may be before the platform pays to re-ask.
    #:
    #: **The cost lever.** It is what makes `FRESH_ENOUGH` fire, and
    #: `FRESH_ENOUGH` is the largest saving in a working deployment. Too short
    #: spends money re-asking about things that have not changed; too long
    #: reports stale claims as current. 60 s is a starting point for kitchen
    #: PPE, where a hairnet does not come off between one minute and the next.
    vision_demand_freshness_ms: int = 60_000
    #: Which understander the platform should bind.
    #:
    #: Empty lets the platform read its own `VISION_UNDERSTANDER_PROVIDER` from
    #: the process environment. That is the trap this field exists to close:
    #: pydantic-settings loads `.env` into *this object*, not into `os.environ`,
    #: so a deployment whose `.env` named a real provider silently bound the
    #: static fallback instead — and the only symptom was every model result
    #: arriving as a failed outcome.
    vision_understander_provider: str = ""
    #: The understanding provider's credential.
    #:
    #: A `SecretStr`, so it cannot be printed by accident, and it is handed to
    #: the platform's provider registry as a composition default rather than
    #: exported to `os.environ` — a process-wide variable is readable by every
    #: library in the process, including the ones that log their configuration.
    #: Empty means the platform reads its own environment, which is the correct
    #: behaviour for a deployment that manages the key elsewhere.
    #: Read from `VISION_NVIDIA_API_KEY` as well, because that is the name the
    #: platform's own provider registry uses and the name deployments already
    #: have in their `.env`. Aliasing beats asking every deployment to rename a
    #: working variable.
    vision_understander_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("VISION_UNDERSTANDER_API_KEY", "VISION_NVIDIA_API_KEY"),
    )
    #: Which hosted model answers, and where it lives.
    #:
    #: **A model identifier is deployment configuration, not a constant.** It
    #: was neither: the name lived only in `nvidia_vl.DEFAULT_MODEL`, and
    #: `.env` could not override it because the provider factory reads
    #: `os.environ`, which pydantic-settings never writes to. When NVIDIA
    #: retired that model on 2026-08-26 the deployment had no way to name a
    #: replacement without a code change — and the symptom was 18 hours of
    #: "no alerts" on a safety product.
    #:
    #: Empty means "the adapter's default applies", so a deployment that sets
    #: neither is unaffected.
    vision_nvidia_model: str = ""
    vision_nvidia_base_url: str = ""
    #: Where a locally-served understander listens, and which model answers.
    #: Same trap, same fix — see `understander_options`.
    vision_ollama_base_url: str = ""
    vision_ollama_model: str = ""
    #: Longest crop edge sent to the model, in pixels.
    #:
    #: Vision tokens scale with **area**, so this is the main latency lever on a
    #: CPU-served model. It is also the resolution policy asks for on the head
    #: band, so lowering it trades measured head accuracy for speed. Left empty,
    #: the adapter's own default applies.
    vision_understander_max_side: str = ""
    #: Seconds one understanding call may take before it is a reported timeout.
    vision_understander_timeout_s: str = ""

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
    #: The DVR password itself, when the deployment keeps it in configuration.
    #:
    #: `cctv_credential_ref` names *where* the secret is; this is the value for
    #: the common case where "where" is this application's own `.env`.
    #: pydantic-settings loads `.env` into this object and **not** into
    #: `os.environ`, so a provider reading `os.environ` cannot see it — which is
    #: exactly how sixteen cameras sat at CONNECTING with a correct password on
    #: disk. `secret_environment()` closes that gap.
    cctv_password: SecretStr = SecretStr("")
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
    #: Capture a frame as durable evidence when a violation opens an incident.
    #:
    #: Separate from `allow_evidence`, which governs whether stored imagery may
    #: be **served**. Writing and reading are different authorisations: a
    #: deployment may want a durable record with retrieval still closed, and
    #: turning one on must never silently turn on the other. Off by default —
    #: storing images of identifiable people is a deployment decision, never
    #: an inherited one.
    evidence_capture: bool = False
    #: Where a new incident is announced: `log`, `file`, `null`, or `off`.
    #: Deliberately small — a new destination is a new adapter behind the same
    #: port, not a new setting shape.
    notification_channel: str = "log"
    notification_file_path: str = "./var/notifications.jsonl"
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

    def secret_environment(self) -> dict[str, str]:
        """Process environment, with deployment-configured secrets layered on.

        The secret provider resolves `env:NAME` against a mapping. Passing this
        instead of `os.environ` means a value in `.env` resolves exactly as one
        exported by an orchestrator would, and neither is copied into the
        process environment where every library in the process could read it.

        A real environment variable still wins: an operator supplying a rotated
        credential for one run must not be overridden by a stale file.
        """
        import os

        overlay = dict(os.environ)
        password = self.cctv_password.get_secret_value()
        if password and not overlay.get("CCTV_PASSWORD"):
            overlay["CCTV_PASSWORD"] = password
        key = self.vision_understander_api_key.get_secret_value()
        if key and not overlay.get("VISION_NVIDIA_API_KEY"):
            overlay["VISION_NVIDIA_API_KEY"] = key
        return overlay

    def understander_options(self) -> dict[str, str]:
        """Non-secret understander configuration, under the platform's own names.

        The **file** layer of `defaults -> file -> environment -> secret`. The
        provider factory resolves each name against the mapping it is handed,
        and that mapping is `os.environ` — which `.env` never reaches, because
        pydantic-settings loads a file into *this object* and not into the
        process environment. Without this bridge every one of these settings is
        parsed, stored, and silently ignored.

        Handed over as **defaults**, so a real environment variable still wins:
        an operator naming a replacement model for one run must not be overruled
        by a stale file. Secrets do not travel this way — the API key is a
        `SecretStr` and goes through its own path, so nothing here is printable.

        Only non-empty values are emitted. An empty setting must mean "the
        adapter's default applies", not "the empty string", or every unset field
        would override a working default with nothing — which for `base_url`
        would silently point the adapter at nowhere.
        """
        options = {
            "VISION_NVIDIA_MODEL": self.vision_nvidia_model,
            "VISION_NVIDIA_BASE_URL": self.vision_nvidia_base_url,
            "VISION_OLLAMA_BASE_URL": self.vision_ollama_base_url,
            "VISION_OLLAMA_MODEL": self.vision_ollama_model,
            "VISION_UNDERSTANDER_MAX_SIDE": self.vision_understander_max_side,
            "VISION_UNDERSTANDER_TIMEOUT_S": self.vision_understander_timeout_s,
        }
        return {name: value.strip() for name, value in options.items() if value.strip()}

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
