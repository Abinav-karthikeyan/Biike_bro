"""Zones router — zone metadata and real-time occupancy from DuckDB."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from backend.models.schemas import ZoneDetail, ZoneSummary

router = APIRouter()


@router.get("/", response_model=List[ZoneSummary], summary="List all zones")
async def list_zones(
    request: Request,
    venue_type: Optional[str] = Query(default=None),
    min_transit_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> List[ZoneSummary]:
    """Return all parking zones from DuckDB, optionally filtered."""
    db = request.app.state.db
    zones = db.get_zones()

    results = []
    for z in zones:
        if venue_type and z.get("venue_type") != venue_type:
            continue
        if min_transit_score and (z.get("transit_score") or 0) < min_transit_score:
            continue
        results.append(ZoneSummary(
            zone_id=z["zone_id"],
            name=z["name"],
            lat=z["lat"],
            lon=z["lon"],
            venue_type=z.get("venue_type"),
            transit_score=z.get("transit_score"),
        ))
    return results


@router.get("/{zone_id}", response_model=ZoneDetail, summary="Get zone detail")
async def get_zone(request: Request, zone_id: str) -> ZoneDetail:
    """Return detailed zone info with latest occupancy from DuckDB."""
    db = request.app.state.db
    zone = db.get_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")

    return ZoneDetail(
        zone_id=zone["zone_id"],
        name=zone["name"],
        lat=zone["lat"],
        lon=zone["lon"],
        venue_type=zone.get("venue_type"),
        transit_score=zone.get("transit_score"),
        neighborhood=zone.get("neighborhood"),
        avg_daily_turnover=None,
        current_occupancy_pct=zone.get("occupancy_pct"),
        last_snapshot_at=zone.get("last_snapshot_at"),
    )


@router.get("/{zone_id}/snapshots", summary="Get zone snapshots")
async def get_zone_snapshots(
    request: Request, zone_id: str, hours: int = Query(default=24, ge=1, le=168)
):
    """Return recent snapshots for a zone."""
    db = request.app.state.db
    snapshots = db.get_zone_snapshots(zone_id, hours=hours)
    return snapshots


@router.get("/{zone_id}/occupancy-history", summary="Occupancy time-series")
async def get_occupancy_history(request: Request, zone_id: str):
    """Return 7-day occupancy time series for charting."""
    db = request.app.state.db
    snapshots = db.get_zone_snapshots(zone_id, hours=168)  # 7 days
    # Downsample to hourly for chart
    hourly = {}
    for s in snapshots:
        ts = str(s.get("timestamp", ""))[:13]  # truncate to hour
        if ts not in hourly:
            hourly[ts] = []
        hourly[ts].append(s.get("occupancy_pct", 0))
    return [
        {"timestamp": k + ":00:00Z", "avg_occupancy": round(sum(v) / len(v), 1)}
        for k, v in sorted(hourly.items())
    ]
