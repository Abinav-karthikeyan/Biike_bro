"""Analytics router — KPI dashboard data, heatmaps, and model metrics."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/summary", summary="Analytics KPI summary")
async def analytics_summary(request: Request):
    """Return analytics KPI summary for dashboard."""
    db = request.app.state.db
    return db.get_analytics_summary()


@router.get("/zone-heatmap", summary="Zone occupancy heatmap")
async def zone_heatmap(request: Request):
    """Return avg occupancy by zone and hour for heatmap visualisation."""
    db = request.app.state.db
    return db.get_zone_heatmap()


@router.get("/model-metrics", summary="ML model performance")
async def model_metrics(request: Request):
    """Return XGBoost model training metrics."""
    prediction_service = request.app.state.prediction_service
    return prediction_service.get_model_metrics()


@router.get("/events", summary="Local events")
async def local_events(request: Request):
    """Return all local events."""
    db = request.app.state.db
    return db.get_events()
