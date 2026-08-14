"""
Data Loaders — seed CSV data into any SQLAlchemy-compatible database.

Two strategies:
  DuckDB  → read_csv_auto() via raw SQL (sub-second for 725k rows)
  Postgres → pandas to_sql() with chunked inserts

The public API is a single load_csv_seeds(engine, seed_dir) that dispatches
automatically based on the engine's dialect.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

# Ordered so FK-parent tables (zones) come before FK-child tables (zone_snapshots)
_SEED_TABLES = [
    ("zones", "zones.csv"),
    ("bikes", "bikes.csv"),
    ("rides", "rides.csv"),
    ("zone_snapshots", "zone_snapshots.csv"),
    ("weather", "weather.csv"),
    ("local_events", "local_events.csv"),
]


def load_csv_seeds(engine: Engine, seed_dir: Path) -> None:
    """
    Load all seed CSV files into the database.

    For DuckDB, uses read_csv_auto for speed (≪1 s for 725k rows).
    For Postgres, uses pandas to_sql with chunked multi-row inserts.
    """
    is_duckdb = "duckdb" in engine.dialect.name

    if is_duckdb:
        _load_duckdb(engine, seed_dir)
    else:
        _load_pandas(engine, seed_dir)


def _load_duckdb(engine: Engine, seed_dir: Path) -> None:
    """Fast DuckDB CSV load using read_csv_auto — creates/replaces tables."""
    with engine.connect() as conn:
        for table, filename in _SEED_TABLES:
            path = seed_dir / filename
            if not path.exists():
                logger.warning("Seed file not found: %s", path)
                continue
            csv_path = str(path).replace("\\", "/")
            conn.execute(
                text(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{csv_path}')")
            )
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            logger.info("DuckDB: loaded %s — %s rows", table, f"{count:,}")
        # Stamp zone_source so synthetic rows are distinguishable from real GBFS rows
        conn.execute(text(
            "ALTER TABLE zones ADD COLUMN IF NOT EXISTS zone_source VARCHAR DEFAULT 'synthetic'"
        ))
        conn.execute(text(
            "UPDATE zones SET zone_source = 'synthetic' WHERE zone_source IS NULL"
        ))
        conn.commit()


def _load_pandas(engine: Engine, seed_dir: Path) -> None:
    """Portable seed load via pandas — works for PostgreSQL and any DBAPI backend."""
    for table, filename in _SEED_TABLES:
        path = seed_dir / filename
        if not path.exists():
            logger.warning("Seed file not found: %s", path)
            continue
        df = pd.read_csv(path)
        # Replace nulls that pandas reads as NaN with None so SQL NULLs are correct
        df = df.where(pd.notna(df), None)
        if table == "zones":
            df["zone_source"] = "synthetic"
        df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=500)
        logger.info("Pandas: loaded %s — %s rows", table, f"{len(df):,}")


# ─────────────────────────────────────────────────────────────────────────────
# GBFS feed helpers (unchanged — used by gbfs_ingest.py)
# ─────────────────────────────────────────────────────────────────────────────


def parse_gbfs_feed_urls(gbfs_manifest: Dict[str, Any]) -> Dict[str, str]:
    """Extract feed URLs from a GBFS auto-discovery manifest (v2.3 format)."""
    feed_urls: Dict[str, str] = {}
    for lang_data in gbfs_manifest.get("data", {}).values():
        for feed in lang_data.get("feeds", []):
            feed_urls[feed["name"]] = feed["url"]
    return feed_urls


def normalize_zone_snapshot(station: Dict[str, Any], last_updated) -> Dict[str, Any]:
    """Normalise a GBFS station_status record to internal zone snapshot format."""
    num_bikes = station.get("num_bikes_available", 0)
    capacity = station.get("capacity") or (
        station.get("num_docks_available", 0) + num_bikes
    )
    occupancy = (capacity - num_bikes) / capacity if capacity > 0 else 0.0
    return {
        "zone_id": str(station["station_id"]),
        "timestamp": last_updated,
        "bikes_available": num_bikes,
        "capacity": capacity,
        "occupancy_pct": round(occupancy, 4),
        "weather_code": None,
        "is_event_nearby": None,
    }
