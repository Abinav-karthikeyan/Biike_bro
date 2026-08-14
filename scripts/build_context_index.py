"""
Build the 128-D HNSW context index from zone_snapshots.

Run once (or after ingesting significant new snapshot data):

    python -m scripts.build_context_index

Populates context_index.bin and persists context_ids into zone_id_map.json
alongside the existing zone_ids / zone_meta entries.

What this produces
──────────────────
One 128-D embedding per (zone_id × hour_of_day × day_of_week) bucket.
60 zones × 24 hours × 7 days = up to 10 080 vectors (fewer if snapshot
coverage is sparse).  Each vector captures the typical occupancy,
weather conditions, and event activity for that zone × time slot, so
historical_pattern_search() and intent_search() return real records
instead of an empty index.
"""

import sys
from pathlib import Path

# Allow running from the repo root as `python -m scripts.build_context_index`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.duckdb_store import DuckDBStore
from backend.services.hnsw_search import HNSWSearchService


def main() -> None:
    print("Loading DuckDB + HNSW service …")
    db = DuckDBStore()
    hnsw = HNSWSearchService()

    zones = db.get_zones()
    print(f"  Zones: {len(zones)}")
    print(f"  Zone index: {hnsw._zone_index.get_current_count() if hnsw._zone_index else 0} items")

    print("Building context index from 7-day snapshot window …")
    n = hnsw.build_context_index(db)
    print(f"  Context vectors written: {n}")
    print("Done — context_index.bin and zone_id_map.json updated.")


if __name__ == "__main__":
    main()
