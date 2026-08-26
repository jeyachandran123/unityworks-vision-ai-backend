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

from app.api.product import router as product_router
from app.api.routes import build_router, devtools_router
from app.api.wall import router as wall_router
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
from app.vision.analysis_loop import ANALYSIS
from app.vision.demands import register_policy_demands
from app.vision.ingest import FrameIngest
from app.vision.manager import LiveRuntime
from app.vision.runtime import VisionRuntime
from app.vision.taps import TapBus
from app.vision.wall import CameraWall


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
    # Live viewing. Independent of Vision OS by design: the wall owns its own
    # sessions and never calls the perception path.
    app.state.wall = CameraWall(cfg)
    # Bounded rings over the platform's own bus. Attached at start-up when a
    # platform exists; harmless and empty when one does not.
    app.state.taps = TapBus()
    app.state.ingest = None
    app.state.demands = None
    app.state.notifier = None

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
    app.include_router(product_router)
    app.include_router(wall_router)
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

    # Started before any session can submit to it. CPU-bound perception runs
    # there instead of on this loop; see `app/vision/analysis_loop.py`.
    ANALYSIS.start()

    assembled = await vision.start()
    VISION_READY.set(1 if assembled else 0)

    # The seam. Frames reach the platform only because this line runs, and it
    # runs before any camera starts so every session goes through it.
    if assembled and vision.composition is not None:
        live_runtime: LiveRuntime = app.state.live
        ingest = FrameIngest(vision.composition, ledger=live_runtime.ledger)
        live_runtime.bind_ingest(ingest)
        app.state.ingest = ingest
        app.state.taps.attach(getattr(vision.composition.platform, "bus", None))

        # Without this the platform correctly analyses nothing: M8 skips every
        # candidate with `no_demand`, and zero model calls is the right answer
        # to a question nobody asked. This is the application stating what it is
        # willing to pay to look at.
        app.state.demands = register_policy_demands(
            vision.composition,
            freshness_ms=cfg.vision_demand_freshness_ms,
        )
        logger.info("Vision OS ingest bound — frames will reach the perception path")

        # The application's own decision layer. Vision OS reports what it sees
        # and knows nothing about rules, incidents or users; this reads the
        # Observation API like any other consumer and writes to the Incident
        # domain. Nothing in `vision_os` imports it or is aware it runs.
        app.state.compliance = _build_compliance_driver(app, cfg, vision)
    else:
        # Said plainly. A deployment whose cameras run but whose frames reach no
        # model is a deployment that looks healthy and observes nothing, and
        # that must never be discovered from a dashboard reading zero.
        app.state.ingest = None
        app.state.demands = None
        app.state.compliance = None
        logger.warning(
            "Vision OS is not assembled — camera frames will be acquired and "
            "released without reaching detection"
        )

    # Retention first, before anything is served. A record past its retention
    # date must not be servable for the window between boot and the first sweep.
    # Marking always happens; erasure only when the deployment enabled it.
    await _sweep_retention(app)

    # The ONLY place a camera session starts. Not on import, not on a DevTools
    # request, not as a side effect of anything else.
    #
    # The camera set now comes from the database, so restarting the process
    # restores exactly the cameras that were running before it stopped — which
    # is what "recovery" means here. A row that is not `enabled` opens no socket.
    # The camera wall. Started from the same durable camera rows, and started
    # whether or not Vision OS assembled — viewing must not depend on analysis.
    await _start_camera_wall(app)

    live: LiveRuntime = app.state.live
    started = await _start_cameras_from_database(app)
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
    await app.state.wall.stop_all()
    if vision.composition is not None:
        app.state.taps.detach(getattr(vision.composition.platform, "bus", None))
    await live.stop_all()
    await vision.stop()
    # After the sessions that submit to it, before the process exits.
    ANALYSIS.stop()
    await cache.disconnect()
    await database.disconnect()
    logger.info("Shutdown complete")


async def _sweep_retention(app: FastAPI) -> None:
    """Apply retention at boot. Never fatal.

    A database that is briefly unreachable while the stack comes up must not
    crash-loop the process, so a failed sweep is reported and start-up
    continues — the next sweep will catch what this one missed. What it must
    never do is fail *silently*: unswept data is a retention promise not kept.
    """
    from app.domain.retention import RetentionService

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    try:
        async with database.session_scope() as session:
            service = RetentionService(
                session,
                root=cfg.evidence_path,
                evidence_days=cfg.evidence_retention_days,
                incident_days=cfg.incident_retention_days,
                audit_days=cfg.audit_retention_days,
            )
            report = await service.sweep(erase=cfg.retention_sweep_enabled)
        logger.info("retention at boot: {}", report.as_dict())
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "retention sweep failed at boot: {}: {}. Data past its retention "
            "date may still be present.",
            type(exc).__name__,
            exc,
        )


async def _start_cameras_from_database(app: FastAPI) -> int:
    """Read the enabled cameras and start them. **This is the recovery path.**

    Returns 0 rather than raising if the database is unreachable: no camera is
    better than no application, and `/health/ready` reports the database anyway.
    """
    from app.domain.cameras import CameraService, to_rtsp_config

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    live: LiveRuntime = app.state.live

    try:
        async with database.session_scope() as session:
            rows = await CameraService(session).enabled_for_runtime(
                organization_id=cfg.default_tenant_id
            )
            configs = [to_rtsp_config(row) for row in rows if row.host]
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "could not read camera configuration: {}: {}. No camera session "
            "started; fix the database and restart.",
            type(exc).__name__,
            exc,
        )
        return 0

    if not configs:
        # Not an error. A deployment with no enabled camera is a valid
        # deployment, and it says so rather than failing to boot.
        logger.info(
            "no enabled camera rows; nothing to start. Cameras are managed at "
            "/api/v1/cameras, not by CCTV_CHANNELS."
        )
        return 0

    return await live.start_from_records(configs)


def _build_compliance_driver(app: FastAPI, cfg, vision):
    """Load the deployment's rules and start the evaluation pass.

    A rule document that will not load is reported and the driver is left off,
    rather than starting one that silently decides nothing: a compliance system
    that is quietly not running is worse than one that is visibly absent.
    """
    from app.vision.compliance_driver import ComplianceDriver
    from compliance import RuleDocumentError, load_rules

    if not cfg.compliance_rules:
        logger.warning(
            "COMPLIANCE_RULES is not set — observations will be produced and "
            "no compliance decision will be made"
        )
        return None

    try:
        rules = load_rules(cfg.compliance_rules)
    except RuleDocumentError as exc:
        logger.error("compliance rules not loaded: {}. No verdicts will be produced.", exc)
        return None
    if rules is None:
        return None

    producible = frozenset(vision.status().to_wire().get("attributes", ()))
    unproducible = frozenset(rules.required_attributes) - producible
    if unproducible:
        # Named at boot rather than left to look like caution. A rule waiting on
        # an attribute nothing observes sits at UNKNOWN forever, which reads as
        # a careful system and is actually a silent misconfiguration.
        logger.warning(
            "compliance rules depend on attribute(s) nothing observes: {}. "
            "Those conditions can only ever be UNKNOWN.",
            sorted(unproducible),
        )

    from app.domain.notifications import build_notifier

    try:
        notifier = build_notifier(cfg)
    except ValueError as exc:
        # A typo in a channel name must not look like a working deployment that
        # happens to tell nobody.
        logger.error("notifications not configured: {}. No one will be told.", exc)
        notifier = None
    app.state.notifier = notifier

    driver = ComplianceDriver(
        settings=cfg,
        vision=vision,
        database=app.state.database,
        rules=rules,
        interval_s=cfg.compliance_interval_s,
        # Evidence comes from the wall's already-encoded latest frame, so
        # capturing costs nothing on the analysis path.
        wall=app.state.wall,
        notifier=notifier,
    )
    driver.start()
    logger.info(
        "compliance engine started — ruleset {} with {} rule(s), every {}s; "
        "evidence capture {}, notifications {}",
        rules.version,
        len(rules.rules),
        cfg.compliance_interval_s,
        "on" if cfg.evidence_capture else "off",
        notifier.channel_id if notifier else "off",
    )
    return driver


async def _start_camera_wall(app: FastAPI) -> int:
    """Open a viewing stream for every enabled camera row.

    Separate from `_start_cameras_from_database`, which starts *analysis*
    sessions. A deployment may reasonably watch sixteen cameras and analyse
    none, and this is the line that keeps those two decisions apart.
    """
    from app.domain.cameras import CameraService

    cfg: Settings = app.state.settings
    database: Database = app.state.database
    wall: CameraWall = app.state.wall
    cameras: list = []
    elsewhere: dict[str, int] = {}

    if not cfg.feature_camera_wall:
        logger.info("camera wall disabled (FEATURE_CAMERA_WALL=false)")
        return 0

    try:
        async with database.session_scope() as session:
            service = CameraService(session)
            cameras = await service.list(organization_id=cfg.default_tenant_id)
            if not cameras:
                elsewhere = await _cameras_in_other_tenants(
                    session, cfg.default_tenant_id
                )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        logger.error(
            "camera wall could not read its camera list: {}: {}",
            type(exc).__name__,
            exc,
        )
        return 0

    if not cameras:
        if elsewhere:
            # The failure that produced "0 channels, 16 connecting": cameras
            # exist, but in a tenant this deployment does not serve. Silence
            # here reads as "no cameras configured", which sends somebody to
            # look at the DVR instead of at DEFAULT_TENANT_ID.
            logger.error(
                "camera wall found no cameras for tenant '{}', but {} camera "
                "row(s) exist in tenant(s) {}. DEFAULT_TENANT_ID does not match "
                "the tenant that owns the cameras.",
                cfg.default_tenant_id,
                sum(elsewhere.values()),
                sorted(elsewhere),
            )
        else:
            logger.info("camera wall: no camera rows configured")
        return 0
    return await wall.start_cameras(cameras)


async def _cameras_in_other_tenants(session, tenant_id: str) -> dict[str, int]:
    """Camera counts per tenant, excluding this one. For diagnosing a mismatch."""
    from sqlalchemy import func, select

    from app.domain.models import Camera

    result = await session.execute(
        select(Camera.organization_id, func.count())
        .where(Camera.organization_id != tenant_id)
        .group_by(Camera.organization_id)
    )
    return {row[0]: int(row[1]) for row in result.all()}


def _install_error_handlers(app: FastAPI) -> None:
    def _request_id(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    # Codes that are part of a normal flow rather than a problem. They still
    # return their status; they just do not deserve a line in the log every time
    # somebody opens the page.
    _routine = {"NO_SESSION"}

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.code in _routine:
            logger.debug("{}: {}", exc.code, exc.message)
        else:
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
