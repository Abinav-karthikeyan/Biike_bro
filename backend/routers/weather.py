"""Weather router — hourly weather records from DuckDB."""

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/", summary="Recent weather")
async def list_weather(request: Request, hours: int = Query(default=48, ge=1, le=168)):
    """Return recent hourly weather records."""
    db = request.app.state.db
    return db.get_weather_recent(hours=hours)
