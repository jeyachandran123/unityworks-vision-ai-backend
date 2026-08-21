"""The FastAPI application.

### Startup ordering, and what may fail

    settings → production safety check → logging → database → cache → Vision OS

The production safety check runs **before** anything opens a socket, so a
deployment with a default secret fails at boot rather than serving traffic it
should not.

After that the rule is: **degrade, do not refuse**. Redis down warns; Vision OS
failing to assemble warns; both are reported through `/health/ready` and
`/api/v1/status`. Only configuration is fatal, because a misconfigured process
cannot be trusted to be wrong in a bounded way.

The database is the one dependency this application does not verify at boot. It
is verified at readiness and on first use, because a database that is briefly
unreachable while an orchestrator brings the stack up is not a reason to
crash-loop.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from app.api.routes import build_router, devtools_router
from app.api.websocket import router as websocket_router
from app.auth.service import AuthService
from app.auth.tokens import TokenService
from app.configuration.settings import Settings, get_settings
from app.errors import AppError
from app.infrastructure.cache import Cache
from app.infrastructure.database import Database
from app.infrastructure.observability import (
    VISION_READY,
    configure_logging,
    metrics_payload,
    request_context_middleware,
)
from app.vision.manager import LiveRuntime
from app.vision.runtime import VisionRuntime


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Takes settings so tests can construct a differently-configured app without
    mutating process environment.
    """
    cfg = settings or get_settings()
    cfg.assert_production_safe()
    configure_logging(cfg)

    app = FastAPI(
        title=cfg.app_name,
        version="1.0.0",
        description=(
            "Production backend for UnityWorks Vision AI. Owns the Vision OS "
            "perception platform and exposes it through an authorized API."
        ),
        # Schema documentation is a map of the attack surface. Debug builds only.
        docs_url="/docs" if cfg.app_debug else None,
        redoc_url="/redoc" if cfg.app_debug else None,
        openapi_url="/openapi.json" if cfg.app_debug else None,
        lifespan=_lifespan,
    )

    app.state.settings = cfg
    app.state.database = Database(cfg)
    app.state.cache = Cache(cfg)
    app.state.auth = AuthService(TokenService(cfg))
    app.state.vision = VisionRuntime(cfg)
    app.state.live = LiveRuntime(cfg)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        # Explicit, not "*". A credentialed API that reflects arbitrary methods
        # and headers has given up the preflight as a control.
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    )
    app.middleware("http")(request_context_middleware)

    _install_error_handlers(app)

    app.include_router(build_router())
    app.include_router(websocket_router)
    if cfg.feature_devtools:
        # The flag decides whether the routes exist at all; ACCESS_DEVTOOLS on
        # each route decides who may reach them. Two gates, independently set.
        app.include_router(devtools_router)
        logger.warning("DevTools routes are mounted (FEATURE_DEVTOOLS=true)")

    if cfg.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            body, content_type = metrics_payload()
            return Response(content=body, media_type=content_type)

    return app


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: Settings = app.state.settings
    database: Database = app.state.database
    cache: Cache = app.state.cache
    vision: VisionRuntime = app.state.vision

    logger.info("Starting {} [{}]", cfg.app_name, cfg.app_env)

    database.connect()
    logger.info("Database engine ready")

    if not await cache.connect():
        # Degraded, not fatal. Refusing to boot without Redis takes down every
        # route, including the ones that never touch it.
        logger.warning(
            "Redis unavailable — features that need it will fail at request "
            "time with the dependency named; other routes are unaffected"
        )

    VISION_READY.set(1 if await vision.start() else 0)

    # The ONLY place a camera session starts. Not on import, not on a DevTools
    # request, not as a side effect of anything else — and it starts nothing
    # unless FEATURE_LIVE_CCTV is on and a channel is named.
    live: LiveRuntime = app.state.live
    started = await live.start_configured()
    if started:
        logger.warning("live CCTV runtime started {} camera session(s)", started)

    if cfg.serve_frames or cfg.allow_evidence:
        # Loud on purpose. These are the two settings that decide whether CCTV
        # imagery of identifiable people can leave the process.
        logger.warning(
            "Imagery egress is enabled — serve_frames={}, allow_evidence={}",
            cfg.serve_frames,
            cfg.allow_evidence,
        )

    yield

    logger.info("Shutting down")
    await live.stop_all()
    await vision.stop()
    await cache.disconnect()
    await database.disconnect()
    logger.info("Shutdown complete")


def _install_error_handlers(app: FastAPI) -> None:
    def _request_id(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info("{}: {}", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status, content=exc.to_envelope(_request_id(request))
        )

    # Vision OS renders its own typed errors with their stable `code`. Registered
    # on the base class so every current and future platform error keeps it,
    # without this file enumerating them — and so a documented policy bound like
    # `WindowTooLargeError` reaches the client as guidance rather than as a 500.
    try:
        from vision_os.core.errors import VisionOSError

        @app.exception_handler(VisionOSError)
        async def _vision_error(request: Request, exc: VisionOSError) -> JSONResponse:
            code = getattr(exc, "code", "VISION_ERROR")
            logger.warning("Vision OS error {}: {}", code, exc)
            return JSONResponse(
                status_code=400,
                content={
                    "code": str(code),
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                    "request_id": _request_id(request),
                },
            )
    except Exception:  # noqa: BLE001 - platform absent; the generic handler covers it
        logger.warning("Vision OS error types unavailable; generic handling applies")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The detail goes to the log, indexed by request id. The client gets the
        # id and nothing else: a traceback in a response body is a file-system
        # map, a dependency inventory and often a fragment of a query.
        logger.exception("Unhandled {}", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL",
                "message": "an unexpected error occurred",
                "retryable": False,
                "details": {},
                "request_id": _request_id(request),
            },
        )


def run() -> None:
    """Console-script entry point."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=cfg.app_debug,
        log_config=None,
    )


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Build ``app`` on first access, so importing this module builds nothing.

    ``uvicorn app.main:app`` still resolves, because that is an attribute lookup.
    A module-level ``app = create_app()`` would instead construct a
    default-configured application every time anything imported this file —
    including a test that only wanted ``create_app`` with its own settings, and
    including tooling that imports for introspection.
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app", "run"]
