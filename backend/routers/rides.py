"""Rides router — ride history and redirect analytics."""

from fastapi import APIRouter, Query, Request

from backend.models.schemas import RideStats

router = APIRouter()


@router.get("/", summary="List recent rides")
async def list_rides(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    """Return recent rides from DuckDB."""
    db = request.app.state.db
    return db.get_rides(limit=limit)


@router.get("/stats", response_model=RideStats, summary="Ride statistics")
async def ride_stats(request: Request) -> RideStats:
    """Aggregate ride statistics including redirect rate."""
    db = request.app.state.db
    stats = db.get_ride_stats()
    return RideStats(**stats)
