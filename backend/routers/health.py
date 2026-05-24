"""Health / readiness router."""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


@router.get("/", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Returns 200 OK when the service is running."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
    )
