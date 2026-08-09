"""
Data Loaders — modular functions for loading and initializing data sources.

This module provides reusable loaders for:
- CSV seed data into DuckDB
- Mock/sample data for development
- GBFS feed data from external operators
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Seed Data Loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_csv_seeds(con: duckdb.DuckDBPyConnection, seed_dir: Path) -> None:
    """
    Load all seed CSV files into DuckDB tables.

    Tables created:
    - zones: Zone metadata (60 zones)
    - bikes: Bike inventory
    - rides: Historical ride records (8k)
    - zone_snapshots: Occupancy snapshots (725k+)
    - weather: Hourly weather data
    - local_events: Events near zones

    Args:
        con: Active DuckDB connection
        seed_dir: Path to directory containing CSV files
    """
    tables = {
        "zones": "zones.csv",
        "bikes": "bikes.csv",
        "rides": "rides.csv",
        "zone_snapshots": "zone_snapshots.csv",
        "weather": "weather.csv",
        "local_events": "local_events.csv",
    }

    for table, filename in tables.items():
        path = seed_dir / filename
        if not path.exists():
            logger.warning(f"Seed file not found: {path}")
            continue

        csv_path = str(path).replace("\\", "/")
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto('{csv_path}')"
        )
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        logger.info(f"Loaded {table}: {count:,} rows")


def load_mock_data(con: duckdb.DuckDBPyConnection, mock_zones: List[Dict[str, Any]],
                   mock_snapshots: List[Dict[str, Any]]) -> None:
    """
    Create mock zone and snapshot tables for development/testing.

    Args:
        con: Active DuckDB connection
        mock_zones: List of zone dictionaries
        mock_snapshots: List of snapshot dictionaries
    """
    if not mock_zones or not mock_snapshots:
        logger.debug("No mock data provided, skipping mock load")
        return

    # Create zones table from mock data
    zones_df = pd.DataFrame(mock_zones)
    con.register("zones", zones_df)
    logger.info(f"Registered mock zones: {len(mock_zones)} zones")

    # Create snapshots table from mock data
    snapshots_df = pd.DataFrame(mock_snapshots)
    con.register("zone_snapshots", snapshots_df)
    logger.info(f"Registered mock snapshots: {len(mock_snapshots)} snapshots")


def initialize_outcome_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create rider_outcomes table for outcome telemetry.

    Table is created lazily on first write to keep schema-free for seed-only runs.

    Args:
        con: Active DuckDB connection
    """
    con.execute("CREATE SEQUENCE IF NOT EXISTS rider_outcomes_id_seq")
    con.execute("""
        CREATE TABLE IF NOT EXISTS rider_outcomes (
            id              BIGINT PRIMARY KEY DEFAULT nextval('rider_outcomes_id_seq'),
            zone_id         VARCHAR NOT NULL,
            arrival_time    TIMESTAMP,
            actual_fill     DOUBLE,
            satisfaction    INTEGER,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)
    logger.debug("Rider outcomes table initialized")


# ─────────────────────────────────────────────────────────────────────────────
# GBFS Feed Loaders
# ─────────────────────────────────────────────────────────────────────────────


def parse_gbfs_feed_urls(gbfs_manifest: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract feed URLs from GBFS auto-discovery manifest.

    GBFS v2.3 format:
        {
          "data": {
            "en": {
              "feeds": [
                {"name": "station_status", "url": "https://..."},
                ...
              ]
            }
          }
        }

    Args:
        gbfs_manifest: Parsed GBFS manifest JSON

    Returns:
        Dictionary mapping feed name to URL
    """
    feed_urls = {}

    for lang_data in gbfs_manifest.get("data", {}).values():
        for feed in lang_data.get("feeds", []):
            feed_urls[feed["name"]] = feed["url"]

    return feed_urls


def normalize_zone_snapshot(station: Dict[str, Any], last_updated) -> Dict[str, Any]:
    """
    Normalize a GBFS station record to internal zone snapshot format.

    Args:
        station: Raw GBFS station_status record
        last_updated: Timestamp from GBFS manifest

    Returns:
        Normalized snapshot dictionary
    """
    num_bikes = station.get("num_bikes_available", 0)
    capacity = station.get("capacity") or (
        station.get("num_docks_available", 0) + num_bikes
    )
    occupancy = (capacity - num_bikes) / capacity if capacity > 0 else 0.0

    return {
        "zone_id": str(station["station_id"]),
        "timestamp": last_updated,
        "available_bikes": num_bikes,
        "docks_used": capacity - num_bikes,
        "occupancy_pct": round(occupancy, 4),
        "weather_code": None,  # TODO: enrich
        "local_events_mask": None,  # TODO: enrich
    }
