"""
Seed Postgres (or any DATABASE_URL backend) with synthetic CSV data.

Usage:
    # With docker-compose Postgres:
    docker compose up -d postgres
    python scripts/seed_postgres.py

    # Against a custom URL:
    DATABASE_URL=postgresql+psycopg2://... python scripts/seed_postgres.py

Prerequisites:
    alembic upgrade head   ← must run first to create the schema

What it does:
    1. Reads each synthetic CSV from synthetic_seed/
    2. Drops extra columns that don't exist in the target table
    3. Appends rows into the already-created tables
    4. Reports row counts per table

Notes:
    - zone_snapshots has 725k rows; expect ~30-60s on a local Postgres.
    - Running twice is safe: the script uses IF NOT EXISTS / dedup logic
      only for rider_outcomes (the mutable table).  For seed tables, it
      truncates first to avoid duplicates.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import make_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent / "synthetic_seed"

# (table_name, csv_file, keep_columns or None to keep all)
SEED_PLAN = [
    (
        "zones",
        "zones.csv",
        [
            "zone_id", "name", "lat", "lon", "radius_m", "zone_type",
            "venue_type", "capacity", "transit_score", "neighborhood",
            "gbfs_region_id", "created_at",
        ],
    ),
    (
        "zone_snapshots",
        "zone_snapshots.csv",
        [
            "snapshot_id", "zone_id", "timestamp", "bikes_available",
            "capacity", "occupancy_pct", "weather_code", "temp_celsius",
            "is_event_nearby", "day_of_week", "hour_of_day",
        ],
    ),
    (
        "rides",
        "rides.csv",
        [
            "ride_id", "bike_id", "rider_hash", "start_zone_id", "end_zone_id",
            "start_lat", "start_lon", "end_lat", "end_lon",
            "start_time", "end_time", "duration_secs", "distance_m",
            "cost_pence", "was_redirected", "redirect_reason", "minutes_wasted",
        ],
    ),
    (
        "weather",
        "weather.csv",
        ["city_id", "timestamp", "temp_celsius", "precip_mm", "wind_kmh", "wmo_code", "is_daylight"],
    ),
    (
        "local_events",
        "local_events.csv",
        [
            "event_id", "city_id", "event_date", "start_time", "end_time",
            "venue_lat", "venue_lon", "expected_attendance", "category",
        ],
    ),
]


def seed(database_url: str) -> None:
    engine = make_engine(database_url)
    logger.info("Connected to: %s", engine.url.render_as_string(hide_password=True))

    with engine.connect() as conn:
        for table, csv_file, keep_cols in SEED_PLAN:
            path = SEED_DIR / csv_file
            if not path.exists():
                logger.warning("Missing seed file: %s — skipping", path)
                continue

            df = pd.read_csv(path)
            if keep_cols:
                df = df[[c for c in keep_cols if c in df.columns]]
            df = df.where(pd.notna(df), None)

            # Truncate before re-seeding to avoid duplicate key errors
            conn.execute(__import__("sqlalchemy").text(f"TRUNCATE TABLE {table} CASCADE"))
            conn.commit()

            df.to_sql(
                table,
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
            logger.info("Seeded %s: %s rows", table, f"{len(df):,}")

    logger.info("Seeding complete.")


if __name__ == "__main__":
    url = (
        os.environ.get("DATABASE_URL", "")
        .replace("+asyncpg", "+psycopg2")
        or "postgresql+psycopg2://buddy:buddy@localhost:5432/cycle_buddy"
    )
    seed(url)
