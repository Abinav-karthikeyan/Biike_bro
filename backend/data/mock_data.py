"""
Mock / seed data for local development.
Used when no real GBFS feeds are configured.
Provides sample zones and snapshots that exercise the full API surface.
"""

from datetime import datetime, timezone

# ─── Sample zone metadata ────────────────────────────────────────────────────

MOCK_ZONES = [
    {
        "zone_id": "zone-001",
        "name": "City Centre Hub",
        "lat": 55.8617,
        "lon": -4.2583,
        "venue_type_id": 1,
        "transit_score": 0.92,
        "neighborhood": "City Centre",
        "avg_daily_turnover": 120.5,
    },
    {
        "zone_id": "zone-002",
        "name": "Riverside Park",
        "lat": 55.8620,
        "lon": -4.2550,
        "venue_type_id": 2,
        "transit_score": 0.65,
        "neighborhood": "Riverside",
        "avg_daily_turnover": 75.0,
    },
    {
        "zone_id": "zone-003",
        "name": "Tech Campus North",
        "lat": 55.8700,
        "lon": -4.2800,
        "venue_type_id": 3,
        "transit_score": 0.78,
        "neighborhood": "West End",
        "avg_daily_turnover": 95.0,
    },
    {
        "zone_id": "zone-004",
        "name": "Market Street Station",
        "lat": 55.8580,
        "lon": -4.2500,
        "venue_type_id": 1,
        "transit_score": 0.88,
        "neighborhood": "Merchant City",
        "avg_daily_turnover": 140.0,
    },
    {
        "zone_id": "zone-005",
        "name": "Eastern Residential",
        "lat": 55.8500,
        "lon": -4.2200,
        "venue_type_id": 4,
        "transit_score": 0.45,
        "neighborhood": "East End",
        "avg_daily_turnover": 40.0,
    },
]

# ─── Sample snapshots (current moment) ──────────────────────────────────────

_NOW = datetime.now(timezone.utc)

MOCK_SNAPSHOTS = [
    {
        "zone_id": "zone-001",
        "timestamp": _NOW,
        "available_bikes": 3,
        "docks_used": 27,
        "occupancy_pct": 0.90,
        "weather_code": 800,
        "local_events_mask": 0,
    },
    {
        "zone_id": "zone-002",
        "timestamp": _NOW,
        "available_bikes": 12,
        "docks_used": 8,
        "occupancy_pct": 0.40,
        "weather_code": 800,
        "local_events_mask": 0,
    },
    {
        "zone_id": "zone-003",
        "timestamp": _NOW,
        "available_bikes": 7,
        "docks_used": 13,
        "occupancy_pct": 0.65,
        "weather_code": 801,
        "local_events_mask": 0,
    },
    {
        "zone_id": "zone-004",
        "timestamp": _NOW,
        "available_bikes": 1,
        "docks_used": 29,
        "occupancy_pct": 0.97,
        "weather_code": 800,
        "local_events_mask": 2,  # event nearby
    },
    {
        "zone_id": "zone-005",
        "timestamp": _NOW,
        "available_bikes": 18,
        "docks_used": 2,
        "occupancy_pct": 0.10,
        "weather_code": 801,
        "local_events_mask": 0,
    },
]

VENUE_TYPE_MAP = {
    1: "transit_hub",
    2: "park",
    3: "office",
    4: "residential",
}
