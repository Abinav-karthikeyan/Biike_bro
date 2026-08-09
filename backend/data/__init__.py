"""
Backend Data Layer — DuckDB store, GBFS ingestion, seed data, and validation.

Public API:
- DuckDBStore: In-memory DuckDB wrapper for seed data and queries
- GBFSClient, GBFSIngestJob: GBFS feed polling and ingestion
- load_csv_seeds, load_mock_data: Modular data loaders
- validate_zone, validate_snapshot, validate_ride: Data validation utilities
"""

from backend.data.duckdb_store import DuckDBStore
from backend.data.gbfs_ingest import GBFSClient, GBFSIngestJob
from backend.data.loaders import (
    load_csv_seeds,
    normalize_zone_snapshot,
    parse_gbfs_feed_urls,
)
from backend.data.validation import (
    validate_ride,
    validate_rider_outcome,
    validate_snapshot,
    validate_snapshots,
    validate_zone,
    validate_zones,
)

__all__ = [
    # Store
    "DuckDBStore",
    # GBFS
    "GBFSClient",
    "GBFSIngestJob",
    # Loaders
    "load_csv_seeds",
    "normalize_zone_snapshot",
    "parse_gbfs_feed_urls",
    # Validators
    "validate_zone",
    "validate_zones",
    "validate_snapshot",
    "validate_snapshots",
    "validate_ride",
    "validate_rider_outcome",
]
