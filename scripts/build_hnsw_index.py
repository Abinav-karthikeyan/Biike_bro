"""
Build HNSW zone index from DuckDB synthetic data.

Usage:
    python scripts/build_hnsw_index.py

Reads zone metadata from synthetic_seed/zones.csv via DuckDB, builds
a 256-D zone embedding index (hnswlib L2, M=16, ef_construction=200),
and saves it to data/hnsw_indices/zone_index.bin + zone_id_map.json.

Re-run whenever zones change (new city, updated venue metadata).
On startup, main.py loads the saved index automatically.
"""

import sys
from pathlib import Path

# Project root on sys.path so backend imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.duckdb_store import DuckDBStore
from backend.services.hnsw_search import HNSWSearchService


def main() -> None:
    print("Loading zones from DuckDB...")
    db = DuckDBStore()
    zones = db.get_zones()
    print(f"  {len(zones)} zones loaded")

    print("Building HNSW zone index...")
    svc = HNSWSearchService()
    svc.build_from_zones(zones)

    count = svc._zone_index.get_current_count() if svc._zone_index else 0
    print(f"  Index built: {count} zones")
    print("  Saved to data/hnsw_indices/")

    # Quick sanity check
    test_zone = zones[0]["zone_id"]
    emb = svc.get_zone_embedding(test_zone)
    if emb is not None:
        results = svc.zone_semantic_search(emb, k=3)
        similar = [r.zone_id for r in results if r.zone_id != test_zone]
        print(f"  Sanity check — zones similar to {test_zone}: {similar}")
    else:
        print("  WARNING: embedding not found for test zone")

    print("Done.")


if __name__ == "__main__":
    main()
