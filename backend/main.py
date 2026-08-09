"""
Bike Parking Buddy — Backend Entry Point
FastAPI app factory with SQLAlchemy data layer and XGBoost model.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import analytics, health, predict, rides, slm, weather, zones

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hooks."""
    logger.info("Starting Bike Parking Buddy backend", version="0.1.0")

    # ── Startup ─────────────────────────────────────────────────────────
    from backend.config import get_settings
    from backend.data import DuckDBStore
    from backend.services.hnsw_search import HNSWSearchService
    from backend.services.prediction import PredictionService
    from backend.services.slm_service import SLMService

    settings = get_settings()

    # If tests pre-injected app.state.db/prediction_service, skip full init
    # so we don't overwrite the in-memory fixture with a Postgres connection.
    if not hasattr(app.state, "db") or app.state.db is None:
        app.state.db = DuckDBStore(db_url=settings.DATABASE_URL)

    if not hasattr(app.state, "prediction_service") or app.state.prediction_service is None:
        app.state.prediction_service = PredictionService(app.state.db)

    if not hasattr(app.state, "hnsw_service") or app.state.hnsw_service is None:
        hnsw = HNSWSearchService()
        if hnsw._zone_index is not None and hnsw._zone_index.get_current_count() == 0:
            zone_list = app.state.db.get_zones()
            hnsw.build_from_zones(zone_list)
        else:
            zone_count = hnsw._zone_index.get_current_count() if hnsw._zone_index else 0
            logger.info(f"HNSW zone index loaded from disk ({zone_count} zones)")
        app.state.hnsw_service = hnsw

    hnsw = app.state.hnsw_service

    app.state.slm_service = SLMService(
        prediction_service=app.state.prediction_service,
        db_store=app.state.db,
        hnsw_service=app.state.hnsw_service,
    )

    zone_count = hnsw._zone_index.get_current_count() if (hnsw and hnsw._zone_index) else 0
    logger.info(
        "Services initialised — DB ready, XGBoost ready, "
        f"HNSW index has {zone_count} zones, SLM ready"
    )
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down Bike Parking Buddy backend")
    if hasattr(app.state, "db") and app.state.db is not None:
        app.state.db.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Bike Parking Buddy API",
        description=(
            "On-device SLM-powered predictive parking intelligence for "
            "dockless bike-sharing — Glasgow synthetic prototype."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(predict.router, prefix="/predict", tags=["prediction"])
    app.include_router(zones.router, prefix="/zones", tags=["zones"])
    app.include_router(rides.router, prefix="/rides", tags=["rides"])
    app.include_router(weather.router, prefix="/weather", tags=["weather"])
    app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
    app.include_router(slm.router, prefix="/slm", tags=["slm"])

    return app


app = create_app()
