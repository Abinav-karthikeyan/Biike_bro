"""
Pydantic schemas — request/response models mirroring the API contracts in PRD §8.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Shared
# ─────────────────────────────────────────────────────────────────────────────


class ZoneSummary(BaseModel):
    zone_id: str
    name: str
    lat: float
    lon: float
    venue_type: Optional[str] = None
    transit_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """POST /predict — request body (PRD §8)."""

    zone_id: str = Field(..., description="Target parking zone identifier")
    current_timestamp: datetime = Field(default_factory=datetime.utcnow)
    intent_embedding: Optional[List[float]] = Field(
        default=None,
        description="128-D SLM intent embedding; omit for unauthenticated calls",
    )
    lookahead_mins: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Prediction horizon in minutes (30 or 60 recommended)",
    )


class AlternativeZone(BaseModel):
    zone_id: str
    fill_probability: float = Field(..., ge=0.0, le=1.0)
    distance_m: Optional[float] = None


class PredictResponse(BaseModel):
    """POST /predict — response body (PRD §8)."""

    zone_id: str
    fill_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternative_zones: List[AlternativeZone] = []
    reason: str
    lookahead_mins: int
    model_version: str


# ─────────────────────────────────────────────────────────────────────────────
# SLM Tool-call Schemas (PRD §8 — SLM Tool Calls)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneSemanticSearchRequest(BaseModel):
    """zone_semantic_search() tool call payload."""

    current_zone_id: str
    k: int = Field(default=3, ge=1, le=20)
    max_distance_m: Optional[float] = None
    venue_type: Optional[str] = None


class ZoneOccupancyProfile(BaseModel):
    hour: int
    avg_occupancy_pct: float


class ZoneSemanticSearchResult(BaseModel):
    zone_id: str
    similarity: float
    occupancy_profile: List[ZoneOccupancyProfile] = []


class ZoneForecastRequest(BaseModel):
    """get_zone_forecast() tool call payload."""

    zone_id: str
    time_horizon_mins: int = Field(default=30, ge=10, le=120)


class ZoneForecastResponse(BaseModel):
    zone_id: str
    arrival_time: datetime
    expected_fill: float
    confidence: float
    recommended_action: str  # e.g. "park here", "reroute to zone X"


class LogOutcomeRequest(BaseModel):
    """log_outcome() telemetry payload."""

    zone_id: str
    timestamp: datetime
    actual_availability: float  # 0.0 = full, 1.0 = empty
    rider_satisfaction: Optional[int] = Field(default=None, ge=1, le=5)


# ─────────────────────────────────────────────────────────────────────────────
# Zone Metadata
# ─────────────────────────────────────────────────────────────────────────────


class ZoneDetail(ZoneSummary):
    neighborhood: Optional[str] = None
    avg_daily_turnover: Optional[float] = None
    current_occupancy_pct: Optional[float] = None
    last_snapshot_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Rides
# ─────────────────────────────────────────────────────────────────────────────


class RideSummary(BaseModel):
    ride_id: str
    bike_id: str
    start_zone_id: str
    end_zone_id: str
    start_time: datetime
    end_time: datetime
    duration_secs: int
    distance_m: int
    cost_pence: int
    was_redirected: bool
    redirect_reason: Optional[str] = None
    minutes_wasted: float


class RideStats(BaseModel):
    total_rides: int
    redirected_count: int
    redirect_rate: float
    avg_wasted_minutes: Optional[float] = None
    avg_cost_pence: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Weather
# ─────────────────────────────────────────────────────────────────────────────


class WeatherRecord(BaseModel):
    city_id: str
    timestamp: datetime
    temp_celsius: float
    precip_mm: float
    wind_kmh: float
    wmo_code: int
    is_daylight: bool


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────


class AnalyticsSummary(BaseModel):
    total_zones: int
    total_rides: int
    redirect_rate: float
    avg_occupancy: float
    busiest_hour: int
    avg_wasted_minutes: Optional[float] = None
