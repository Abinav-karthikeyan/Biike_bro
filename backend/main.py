"""
Bike Parking Buddy — Backend Entry Point
FastAPI app factory with DuckDB data layer and XGBoost model.
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
    from backend.data.duckdb_store import DuckDBStore
    from backend.services.prediction import PredictionService

    logger.info("Starting Bike Parking Buddy backend", version="0.1.0")

    # ── Startup ─────────────────────────────────────────────────────────
    from backend.services.slm_service import SLMService

    app.state.db = DuckDBStore()
    app.state.prediction_service = PredictionService(app.state.db)
    app.state.hnsw_service = None  # TODO: wire up with real embeddings
    app.state.slm_service = SLMService(
        prediction_service=app.state.prediction_service,
        db_store=app.state.db,
    )

    logger.info("Services initialised — DuckDB loaded, XGBoost trained, SLM ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down Bike Parking Buddy backend")


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
