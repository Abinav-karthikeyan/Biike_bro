"""Health / readiness router."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class IngestHealthResponse(BaseModel):
    status: str
    last_success_at: Optional[datetime]
    last_error: Optional[str]
    error_count: int
    snapshots_total: int
    feed_count: int


@router.get("/", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Returns 200 OK when the service is running."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
    )


@router.get("/ingest", response_model=IngestHealthResponse, summary="GBFS ingest status")
async def ingest_health(request: Request) -> IngestHealthResponse:
    """
    Reports the GBFS ingest job's last poll timestamp and error count.
    Returns status='disabled' when GBFS_INGEST_ENABLED=false.
    """
    job = getattr(request.app.state, "ingest_job", None)
    if job is None:
        return IngestHealthResponse(
            status="disabled",
            last_success_at=None,
            last_error=None,
            error_count=0,
            snapshots_total=0,
            feed_count=0,
        )
    d: Dict[str, Any] = job.status_dict()
    return IngestHealthResponse(**d)
