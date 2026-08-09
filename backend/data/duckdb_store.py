"""
DuckDB Data Store — loads synthetic seed CSVs into DuckDB in-memory for prototyping.

Provides fast analytical queries over 725k+ zone snapshots, 8k rides,
60 zones, weather, and events — all in-process with zero external dependencies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from backend.data.loaders import initialize_outcome_table, load_csv_seeds

logger = logging.getLogger(__name__)

# Path to synthetic seed data
SEED_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_seed"


class DuckDBStore:
    """
    In-memory DuckDB store loaded from CSV seed files.
    Thread-safe — DuckDB handles concurrent reads internally.
    """

    def __init__(self, seed_dir: Optional[str] = None):
        self._seed = Path(seed_dir) if seed_dir else SEED_DIR
        self.con = duckdb.connect(":memory:")
        self._initialize()

    # ── Initialization ───────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Initialize DuckDB store with seed CSVs and schema."""
        load_csv_seeds(self.con, self._seed)
        initialize_outcome_table(self.con)
        logger.info("DuckDB store initialised with all seed data")

    # ── Zone Queries ──────────────────────────────────────────────────────

    def get_zones(self) -> List[Dict[str, Any]]:
        """Return all 60 zones."""
        return self.con.execute(
            "SELECT zone_id, name, lat, lon, radius_m, zone_type, venue_type, "
            "capacity, transit_score, neighborhood FROM zones ORDER BY zone_id"
        ).df().to_dict("records")

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Single zone with latest snapshot occupancy."""
        row = self.con.execute("""
            SELECT z.zone_id, z.name, z.lat, z.lon, z.venue_type, z.transit_score,
                   z.capacity, z.neighborhood, z.radius_m, z.zone_type,
                   s.occupancy_pct, s.bikes_available, s.timestamp as last_snapshot_at
            FROM zones z
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY zone_id ORDER BY timestamp DESC) as rn
                FROM zone_snapshots
            ) s ON z.zone_id = s.zone_id AND s.rn = 1
            WHERE z.zone_id = ?
        """, [zone_id]).df()
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_latest_occupancy(self) -> List[Dict[str, Any]]:
        """All zones with their most recent occupancy snapshot."""
        return self.con.execute("""
            SELECT z.zone_id, z.name, z.lat, z.lon, z.venue_type, z.capacity,
                   z.transit_score, z.neighborhood,
                   s.occupancy_pct, s.bikes_available, s.weather_code, s.is_event_nearby,
                   s.timestamp
            FROM zones z
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY zone_id ORDER BY timestamp DESC) as rn
                FROM zone_snapshots
            ) s ON z.zone_id = s.zone_id AND s.rn = 1
            ORDER BY s.occupancy_pct DESC NULLS LAST
        """).df().to_dict("records")

    def get_zone_snapshots(
        self, zone_id: str, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Recent snapshots for a specific zone."""
        return self.con.execute("""
            SELECT snapshot_id, zone_id, timestamp, bikes_available, capacity,
                   occupancy_pct, weather_code, temp_celsius, is_event_nearby,
                   day_of_week, hour_of_day
            FROM zone_snapshots
            WHERE zone_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [zone_id, hours * 12]).df().to_dict("records")  # 12 snapshots/hour

    # ── Ride Queries ──────────────────────────────────────────────────────

    def get_rides(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recent rides ordered by start_time DESC."""
        return self.con.execute("""
            SELECT ride_id, bike_id, start_zone_id, end_zone_id,
                   start_time, end_time, duration_secs, distance_m,
                   cost_pence, was_redirected, redirect_reason, minutes_wasted
            FROM rides
            ORDER BY start_time DESC
            LIMIT ?
        """, [limit]).df().to_dict("records")

    def get_ride_stats(self) -> Dict[str, Any]:
        """Aggregate ride statistics."""
        row = self.con.execute("""
            SELECT
                COUNT(*)                                        AS total_rides,
                SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END) AS redirected_count,
                ROUND(AVG(CASE WHEN was_redirected THEN minutes_wasted END), 2) AS avg_wasted_minutes,
                ROUND(AVG(cost_pence), 1)                       AS avg_cost_pence,
                ROUND(100.0 * SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END) / COUNT(*), 1) AS redirect_rate
            FROM rides
        """).df().iloc[0].to_dict()
        return row

    # ── Weather Queries ───────────────────────────────────────────────────

    def get_weather_recent(self, hours: int = 48) -> List[Dict[str, Any]]:
        """Recent hourly weather records."""
        return self.con.execute("""
            SELECT city_id, timestamp, temp_celsius, precip_mm,
                   wind_kmh, wmo_code, is_daylight
            FROM weather
            ORDER BY timestamp DESC
            LIMIT ?
        """, [hours]).df().to_dict("records")

    # ── Event Queries ─────────────────────────────────────────────────────

    def get_events(self) -> List[Dict[str, Any]]:
        """All local events."""
        return self.con.execute("""
            SELECT event_id, city_id, event_date, start_time, end_time,
                   venue_lat, venue_lon, expected_attendance, category
            FROM local_events
            ORDER BY event_date
        """).df().to_dict("records")

    # ── Analytics Queries ─────────────────────────────────────────────────

    def get_analytics_summary(self) -> Dict[str, Any]:
        """KPI summary for dashboard."""
        summary = self.con.execute("""
            SELECT
                (SELECT COUNT(*) FROM zones) AS total_zones,
                (SELECT COUNT(*) FROM rides) AS total_rides,
                (SELECT ROUND(100.0 * SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END) / COUNT(*), 1) FROM rides) AS redirect_rate,
                (SELECT ROUND(AVG(occupancy_pct), 1) FROM zone_snapshots) AS avg_occupancy,
                (SELECT ROUND(AVG(CASE WHEN was_redirected THEN minutes_wasted END), 2) FROM rides) AS avg_wasted_minutes
        """).df().iloc[0].to_dict()

        # Busiest hour
        busiest = self.con.execute("""
            SELECT hour_of_day, ROUND(AVG(100.0 - occupancy_pct), 1) AS avg_demand
            FROM zone_snapshots
            GROUP BY hour_of_day
            ORDER BY avg_demand ASC
            LIMIT 1
        """).df().iloc[0].to_dict()
        summary["busiest_hour"] = int(busiest["hour_of_day"])

        # Occupancy by venue type
        by_venue = self.con.execute("""
            SELECT z.venue_type, ROUND(AVG(s.occupancy_pct), 1) AS avg_occ
            FROM zone_snapshots s
            JOIN zones z ON s.zone_id = z.zone_id
            GROUP BY z.venue_type
            ORDER BY avg_occ DESC
        """).df().to_dict("records")
        summary["occupancy_by_venue_type"] = by_venue

        return summary

    def get_zone_heatmap(self) -> List[Dict[str, Any]]:
        """Average occupancy by zone and hour for heatmap visualisation."""
        return self.con.execute("""
            SELECT zone_id, hour_of_day,
                   ROUND(AVG(occupancy_pct), 1) AS avg_occupancy
            FROM zone_snapshots
            GROUP BY zone_id, hour_of_day
            ORDER BY zone_id, hour_of_day
        """).df().to_dict("records")

    # ── ML Training Data ─────────────────────────────────────────────────

    def get_snapshots_for_training(self) -> pd.DataFrame:
        """
        Zone snapshots joined with zone metadata for ML training.
        Returns pandas DataFrame ready for feature engineering.
        """
        return self.con.execute("""
            SELECT s.snapshot_id, s.zone_id, s.timestamp,
                   s.bikes_available, s.capacity, s.occupancy_pct,
                   s.weather_code, s.temp_celsius, s.is_event_nearby,
                   s.day_of_week, s.hour_of_day,
                   z.name, z.venue_type, z.transit_score, z.neighborhood
            FROM zone_snapshots s
            JOIN zones z ON s.zone_id = z.zone_id
        """).df()

    def get_zone_occupancy_at_time(
        self, zone_id: str, timestamp: str, window_minutes: int = 30
    ) -> Optional[float]:
        """Historical occupancy near a specific timestamp."""
        row = self.con.execute("""
            SELECT AVG(occupancy_pct) AS avg_occ
            FROM zone_snapshots
            WHERE zone_id = ?
              AND timestamp BETWEEN
                  CAST(? AS TIMESTAMP) - INTERVAL (? || ' minutes')
                  AND CAST(? AS TIMESTAMP) + INTERVAL (? || ' minutes')
        """, [zone_id, timestamp, str(window_minutes), timestamp, str(window_minutes)]).df()
        if row.empty or row.iloc[0]["avg_occ"] is None:
            return None
        return float(row.iloc[0]["avg_occ"])

    # ── Outcome Telemetry ─────────────────────────────────────────────────

    def log_rider_outcome(
        self,
        zone_id: str,
        timestamp: str,
        actual_availability: float,
        rider_satisfaction: Optional[int] = None,
    ) -> None:
        """
        Persist a rider outcome to the rider_outcomes table.

        This is the feedback loop: riders report whether a zone was actually
        available when they arrived.  Over time, these rows let us compare
        predicted_fill vs actual_fill and retrain on real discrepancies.

        The table is created lazily on first write so DuckDB stays schema-free
        for seed-only runs.
        """
        from backend.data.loaders import initialize_outcome_table

        # Ensure outcome table is initialized (idempotent)
        initialize_outcome_table(self.con)

        # actual_availability: 0.0=full, 1.0=empty — invert to fill semantics
        actual_fill = 1.0 - float(actual_availability)
        self.con.execute(
            """
            INSERT INTO rider_outcomes (zone_id, arrival_time, actual_fill, satisfaction)
            VALUES (?, CAST(? AS TIMESTAMP), ?, ?)
            """,
            [zone_id, timestamp, actual_fill, rider_satisfaction],
        )
        logger.info(
            f"Rider outcome logged: zone={zone_id}, fill={actual_fill:.2f}, "
            f"satisfaction={rider_satisfaction}"
        )

    def get_rider_outcomes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent rider outcomes for analytics / model validation."""
        try:
            return self.con.execute("""
                SELECT id, zone_id, arrival_time, actual_fill, satisfaction, created_at
                FROM rider_outcomes
                ORDER BY created_at DESC
                LIMIT ?
            """, [limit]).df().to_dict("records")
        except Exception:
            return []  # table doesn't exist yet (no outcomes logged this session)
