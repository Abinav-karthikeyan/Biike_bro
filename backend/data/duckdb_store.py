"""
DataStore — SQLAlchemy-backed data layer for Bike Parking Buddy.

Reads use portable SQL via SQLAlchemy text() so the same queries run on
both DuckDB (in-memory, tests/dev) and PostgreSQL (production).
Writes use the ORM (RiderOutcome) so SQLAlchemy handles type coercion
and commit semantics.

Default URL is duckdb:///:memory: — tests can instantiate DuckDBStore()
with no arguments and get a fully seeded in-memory database.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import make_engine, make_session_factory
from backend.data.loaders import load_csv_seeds

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_seed"

_DEFAULT_DB_URL = "duckdb:///:memory:"


class DuckDBStore:
    """
    Unified data-access layer backed by SQLAlchemy.

    Pass db_url="postgresql+psycopg2://..." to target PostgreSQL.
    Omit db_url (or pass None) for an in-memory DuckDB instance seeded
    from the synthetic CSVs — the default for tests and local dev.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        seed_dir: Optional[str] = None,
        skip_seed: bool = False,
    ):
        self._seed = Path(seed_dir) if seed_dir else SEED_DIR
        url = db_url or _DEFAULT_DB_URL
        self.engine = make_engine(url)
        self._session_factory = make_session_factory(self.engine)
        if not skip_seed:
            self._initialize()

    # ── Initialization ───────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Seed data then ensure ORM-only tables exist."""
        is_duckdb = "duckdb" in self.engine.dialect.name

        # 1. Seed CSV data
        load_csv_seeds(self.engine, self._seed)

        # 2. Ensure the three ORM-only (non-CSV) tables exist.
        #    DuckDB 1.x does not support SERIAL / BIGSERIAL, so we use
        #    explicit SEQUENCE + DEFAULT nextval for autoincrement columns.
        #    For Postgres, `alembic upgrade head` creates these before the
        #    app starts, so we just check they exist and skip.
        if is_duckdb:
            self._create_orm_tables_duckdb()
        else:
            # On Postgres the tables must already exist (run alembic first).
            # Calling create_all here would still work as a safety net since
            # Postgres dialect supports SERIAL natively.
            from backend.models.db_models import (
                ModelArtifact,
                RiderOutcome,
                ZoneEmbeddingHNSW,
            )
            for model in (RiderOutcome, ModelArtifact, ZoneEmbeddingHNSW):
                model.__table__.create(self.engine, checkfirst=True)

        logger.info("DataStore initialised — engine=%s", self.engine.url.drivername)

    def _create_orm_tables_duckdb(self) -> None:
        """Create ORM-only tables using DuckDB-native DDL (sequences + nextval)."""
        ddl_statements = [
            "CREATE SEQUENCE IF NOT EXISTS rider_outcomes_seq",
            """CREATE TABLE IF NOT EXISTS rider_outcomes (
                id               INTEGER PRIMARY KEY DEFAULT nextval('rider_outcomes_seq'),
                rider_id         VARCHAR,
                session_id       VARCHAR,
                zone_id          VARCHAR NOT NULL,
                predicted_fill   DOUBLE,
                actual_fill      DOUBLE,
                arrival_time     TIMESTAMPTZ,
                satisfaction_score INTEGER,
                intent_embedding JSON,
                created_at       TIMESTAMPTZ DEFAULT current_timestamp
            )""",
            """CREATE TABLE IF NOT EXISTS model_artifacts (
                model_id    VARCHAR PRIMARY KEY,
                version     VARCHAR NOT NULL,
                timestamp   TIMESTAMPTZ DEFAULT current_timestamp,
                xgb_params  JSON,
                holdout_mae DOUBLE,
                s3_path     TEXT,
                city        VARCHAR NOT NULL
            )""",
            "CREATE SEQUENCE IF NOT EXISTS zone_embeddings_hnsw_seq",
            """CREATE TABLE IF NOT EXISTS zone_embeddings_hnsw (
                id            INTEGER PRIMARY KEY DEFAULT nextval('zone_embeddings_hnsw_seq'),
                zone_id       VARCHAR NOT NULL,
                embedding_vec JSON NOT NULL,
                timestamp     TIMESTAMPTZ DEFAULT current_timestamp,
                model_version VARCHAR NOT NULL
            )""",
        ]
        with self.engine.connect() as conn:
            for stmt in ddl_statements:
                conn.execute(text(stmt))
            conn.commit()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _query_df(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SELECT and return results as a DataFrame."""
        with self.engine.connect() as conn:
            stmt = text(sql)
            if params:
                stmt = stmt.bindparams(**params)
            result = conn.execute(stmt)
            rows = result.fetchall()
            return pd.DataFrame(rows, columns=list(result.keys()))

    def _query_one(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        df = self._query_df(sql, params)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    # ── Zone Queries ──────────────────────────────────────────────────────

    def get_zones(self, source: str = "synthetic") -> List[Dict[str, Any]]:
        """Return zones filtered by source ('synthetic', 'real', or None for all)."""
        if source:
            return self._query_df(
                "SELECT zone_id, name, lat, lon, radius_m, zone_type, venue_type, "
                "capacity, transit_score, neighborhood, zone_source FROM zones "
                "WHERE zone_source = :source ORDER BY zone_id",
                {"source": source},
            ).to_dict("records")
        return self._query_df(
            "SELECT zone_id, name, lat, lon, radius_m, zone_type, venue_type, "
            "capacity, transit_score, neighborhood, zone_source FROM zones ORDER BY zone_id"
        ).to_dict("records")

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Single zone with its latest snapshot occupancy."""
        return self._query_one(
            """
            SELECT z.zone_id, z.name, z.lat, z.lon, z.venue_type, z.transit_score,
                   z.capacity, z.neighborhood, z.radius_m, z.zone_type,
                   s.occupancy_pct, s.bikes_available, s.timestamp AS last_snapshot_at
            FROM zones z
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY zone_id ORDER BY timestamp DESC) AS rn
                FROM zone_snapshots
            ) s ON z.zone_id = s.zone_id AND s.rn = 1
            WHERE z.zone_id = :zone_id
            """,
            {"zone_id": zone_id},
        )

    def get_latest_occupancy(self, source: str = "synthetic") -> List[Dict[str, Any]]:
        """All zones with their most recent occupancy snapshot, filtered by source."""
        source_clause = "WHERE z.zone_source = :source" if source else ""
        params = {"source": source} if source else None
        return self._query_df(
            f"""
            SELECT z.zone_id, z.name, z.lat, z.lon, z.venue_type, z.capacity,
                   z.transit_score, z.neighborhood, z.zone_source,
                   s.occupancy_pct, s.bikes_available, s.weather_code,
                   s.is_event_nearby, s.timestamp
            FROM zones z
            LEFT JOIN (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY zone_id ORDER BY timestamp DESC) AS rn
                FROM zone_snapshots
            ) s ON z.zone_id = s.zone_id AND s.rn = 1
            {source_clause}
            ORDER BY s.occupancy_pct DESC NULLS LAST
            """,
            params,
        ).to_dict("records")

    def get_zone_snapshots(self, zone_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Recent snapshots for a specific zone."""
        limit = hours * 12  # 12 snapshots / hour at 5-min cadence
        return self._query_df(
            """
            SELECT snapshot_id, zone_id, timestamp, bikes_available, capacity,
                   occupancy_pct, weather_code, temp_celsius, is_event_nearby,
                   day_of_week, hour_of_day
            FROM zone_snapshots
            WHERE zone_id = :zone_id
            ORDER BY timestamp DESC
            LIMIT :limit
            """,
            {"zone_id": zone_id, "limit": limit},
        ).to_dict("records")

    # ── Ride Queries ──────────────────────────────────────────────────────

    def get_rides(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recent rides ordered by start_time DESC."""
        return self._query_df(
            """
            SELECT ride_id, bike_id, start_zone_id, end_zone_id,
                   start_time, end_time, duration_secs, distance_m,
                   cost_pence, was_redirected, redirect_reason, minutes_wasted
            FROM rides
            ORDER BY start_time DESC
            LIMIT :limit
            """,
            {"limit": limit},
        ).to_dict("records")

    def get_ride_stats(self) -> Dict[str, Any]:
        """Aggregate ride statistics."""
        row = self._query_one(
            """
            SELECT
                COUNT(*)                                                         AS total_rides,
                SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END)                 AS redirected_count,
                ROUND(CAST(AVG(CASE WHEN was_redirected THEN minutes_wasted END) AS NUMERIC), 2)
                                                                                 AS avg_wasted_minutes,
                ROUND(CAST(AVG(cost_pence) AS NUMERIC), 1)                      AS avg_cost_pence,
                ROUND(CAST(100.0 * SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END)
                      / COUNT(*) AS NUMERIC), 1)                                 AS redirect_rate
            FROM rides
            """
        )
        return row or {}

    # ── Weather Queries ───────────────────────────────────────────────────

    def get_weather_recent(self, hours: int = 48) -> List[Dict[str, Any]]:
        """Recent hourly weather records."""
        return self._query_df(
            """
            SELECT city_id, timestamp, temp_celsius, precip_mm,
                   wind_kmh, wmo_code, is_daylight
            FROM weather
            ORDER BY timestamp DESC
            LIMIT :limit
            """,
            {"limit": hours},
        ).to_dict("records")

    # ── Event Queries ─────────────────────────────────────────────────────

    def get_events(self) -> List[Dict[str, Any]]:
        """All local events."""
        return self._query_df(
            """
            SELECT event_id, city_id, event_date, start_time, end_time,
                   venue_lat, venue_lon, expected_attendance, category
            FROM local_events
            ORDER BY event_date
            """
        ).to_dict("records")

    # ── Analytics Queries ─────────────────────────────────────────────────

    def get_analytics_summary(self) -> Dict[str, Any]:
        """KPI summary for dashboard."""
        summary = self._query_one(
            """
            SELECT
                (SELECT COUNT(*) FROM zones WHERE zone_source = 'synthetic')  AS total_zones,
                (SELECT COUNT(*) FROM rides)       AS total_rides,
                (SELECT ROUND(CAST(100.0 * SUM(CASE WHEN was_redirected THEN 1 ELSE 0 END)
                              / COUNT(*) AS NUMERIC), 1) FROM rides) AS redirect_rate,
                (SELECT ROUND(CAST(AVG(occupancy_pct) AS NUMERIC), 1)
                 FROM zone_snapshots)              AS avg_occupancy,
                (SELECT ROUND(CAST(AVG(CASE WHEN was_redirected THEN minutes_wasted END)
                              AS NUMERIC), 2) FROM rides) AS avg_wasted_minutes
            """
        ) or {}

        busiest = self._query_one(
            """
            SELECT hour_of_day,
                   ROUND(CAST(AVG(100.0 - occupancy_pct) AS NUMERIC), 1) AS avg_demand
            FROM zone_snapshots
            GROUP BY hour_of_day
            ORDER BY avg_demand ASC
            LIMIT 1
            """
        )
        summary["busiest_hour"] = int(busiest["hour_of_day"]) if busiest else 0

        by_venue = self._query_df(
            """
            SELECT z.venue_type,
                   ROUND(CAST(AVG(s.occupancy_pct) AS NUMERIC), 1) AS avg_occ
            FROM zone_snapshots s
            JOIN zones z ON s.zone_id = z.zone_id
            GROUP BY z.venue_type
            ORDER BY avg_occ DESC
            """
        ).to_dict("records")
        summary["occupancy_by_venue_type"] = by_venue

        return summary

    def get_zone_heatmap(self) -> List[Dict[str, Any]]:
        """Average occupancy by zone and hour for heatmap visualisation."""
        return self._query_df(
            """
            SELECT zone_id, hour_of_day,
                   ROUND(CAST(AVG(occupancy_pct) AS NUMERIC), 1) AS avg_occupancy
            FROM zone_snapshots
            GROUP BY zone_id, hour_of_day
            ORDER BY zone_id, hour_of_day
            """
        ).to_dict("records")

    # ── ML Training Data ─────────────────────────────────────────────────

    def get_snapshots_for_training(self) -> pd.DataFrame:
        """Zone snapshots joined with zone metadata for ML training."""
        return self._query_df(
            """
            SELECT s.snapshot_id, s.zone_id, s.timestamp,
                   s.bikes_available, s.capacity, s.occupancy_pct,
                   s.weather_code, s.temp_celsius, s.is_event_nearby,
                   s.day_of_week, s.hour_of_day,
                   z.name, z.venue_type, z.transit_score, z.neighborhood
            FROM zone_snapshots s
            JOIN zones z ON s.zone_id = z.zone_id
            """
        )

    def get_zone_occupancy_at_time(
        self, zone_id: str, timestamp: str, window_minutes: int = 30
    ) -> Optional[float]:
        """Historical occupancy near a specific timestamp."""
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        lower = (dt - timedelta(minutes=window_minutes)).isoformat()
        upper = (dt + timedelta(minutes=window_minutes)).isoformat()

        row = self._query_one(
            """
            SELECT AVG(occupancy_pct) AS avg_occ
            FROM zone_snapshots
            WHERE zone_id = :zone_id
              AND timestamp BETWEEN :lower AND :upper
            """,
            {"zone_id": zone_id, "lower": lower, "upper": upper},
        )
        if not row or row.get("avg_occ") is None:
            return None
        return float(row["avg_occ"])

    # ── GBFS Real-Zone Upsert ─────────────────────────────────────────────

    def upsert_real_zone(
        self,
        zone_id: str,
        name: str,
        lat: float,
        lon: float,
        capacity: Optional[int] = None,
        gbfs_region_id: Optional[str] = None,
    ) -> None:
        """Insert or update a real GBFS zone (zone_source='real').

        DuckDB's CSV-loaded zones table has no PK index, so ON CONFLICT isn't
        available there.  We use a delete-then-insert pattern on DuckDB and
        the standard ON CONFLICT on Postgres (where zone_id is a true PK).
        """
        is_duckdb = "duckdb" in self.engine.dialect.name
        params = {
            "zone_id": zone_id, "name": name, "lat": lat, "lon": lon,
            "capacity": capacity, "region": gbfs_region_id,
        }
        with self.engine.connect() as conn:
            if is_duckdb:
                conn.execute(
                    text("DELETE FROM zones WHERE zone_id = :zone_id AND zone_source = 'real'"),
                    {"zone_id": zone_id},
                )
                conn.execute(text("""
                    INSERT INTO zones (zone_id, name, lat, lon, capacity, gbfs_region_id, zone_source)
                    VALUES (:zone_id, :name, :lat, :lon, :capacity, :region, 'real')
                """), params)
            else:
                conn.execute(text("""
                    INSERT INTO zones (zone_id, name, lat, lon, capacity, gbfs_region_id, zone_source)
                    VALUES (:zone_id, :name, :lat, :lon, :capacity, :region, 'real')
                    ON CONFLICT (zone_id) DO UPDATE SET
                        name = EXCLUDED.name, lat = EXCLUDED.lat, lon = EXCLUDED.lon,
                        capacity = EXCLUDED.capacity, gbfs_region_id = EXCLUDED.gbfs_region_id
                """), params)
            conn.commit()

    def insert_zone_snapshots(self, snapshots: List[Dict[str, Any]]) -> int:
        """
        Bulk-insert zone snapshots.  Returns the number of rows written.

        Caller is responsible for deduplication (e.g. checking last_updated
        hasn't advanced before calling).  Rows with unknown zone_id are skipped
        silently to avoid FK violations when the zones table isn't seeded yet.
        """
        if not snapshots:
            return 0

        # Resolve which zone_ids actually exist (FK guard)
        known_ids = {
            r["zone_id"]
            for r in self._query_df("SELECT zone_id FROM zones").to_dict("records")
        }
        valid = [s for s in snapshots if s["zone_id"] in known_ids]
        if not valid:
            return 0

        now = datetime.now(timezone.utc)
        with self.engine.connect() as conn:
            for snap in valid:
                ts = snap.get("timestamp") or now
                conn.execute(text("""
                    INSERT INTO zone_snapshots
                        (zone_id, timestamp, bikes_available, capacity, occupancy_pct,
                         weather_code, is_event_nearby)
                    VALUES
                        (:zone_id, :ts, :bikes, :cap, :occ, :wx, :ev)
                """), {
                    "zone_id": snap["zone_id"],
                    "ts": ts,
                    "bikes": snap.get("bikes_available", 0),
                    "cap": snap.get("capacity"),
                    "occ": snap.get("occupancy_pct", 0.0),
                    "wx": snap.get("weather_code"),
                    "ev": snap.get("is_event_nearby"),
                })
            conn.commit()
        logger.info("Inserted %d zone snapshots", len(valid))
        return len(valid)

    def get_real_zone_latest_occupancy(self) -> List[Dict[str, Any]]:
        """Latest snapshots for all real GBFS zones."""
        return self.get_latest_occupancy(source="real")

    # ── Outcome Telemetry ─────────────────────────────────────────────────

    def log_rider_outcome(
        self,
        zone_id: str,
        timestamp: str,
        actual_availability: float,
        rider_satisfaction: Optional[int] = None,
        intent_embedding: Optional[list] = None,
    ) -> None:
        """
        Persist a rider outcome via SQLAlchemy ORM.

        actual_availability: 0.0=full, 1.0=empty — stored as actual_fill
        intent_embedding: optional 128-D float list from the query context
        (inverted fill semantics: 0.0 available → fill = 1.0).
        """
        from backend.models.db_models import RiderOutcome

        actual_fill = 1.0 - float(actual_availability)
        arrival = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        session: Session = self._session_factory()
        try:
            outcome = RiderOutcome(
                zone_id=zone_id,
                arrival_time=arrival,
                actual_fill=actual_fill,
                satisfaction_score=rider_satisfaction,
                intent_embedding=intent_embedding,
            )
            session.add(outcome)
            session.commit()
            logger.info(
                "Rider outcome persisted: zone=%s fill=%.2f satisfaction=%s",
                zone_id,
                actual_fill,
                rider_satisfaction,
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_rider_outcomes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recent rider outcomes for analytics / model validation."""
        try:
            return self._query_df(
                """
                SELECT id, zone_id, arrival_time, actual_fill,
                       satisfaction_score, created_at
                FROM rider_outcomes
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"limit": limit},
            ).to_dict("records")
        except Exception:
            return []
