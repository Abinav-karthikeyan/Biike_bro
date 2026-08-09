"""
HNSW Vector Search Service — hnswlib-based semantic zone similarity search.

PRD §5 — HNSW Integration:
  - Zone embeddings: 256-D [lat, lon, venue_type_embedding, avg_daily_turnover, transit_score]
  - Historical day-context embeddings: 128-D [day_of_week, hour, weather_vec, events_vec]
  - Intent embeddings: 128-D (from SLM)

  Use Cases:
    5.1 Historical Pattern Matching (128-D, M=16, ef_construction=200)
    5.2 Zone Semantic Similarity    (256-D, M=16, ef_construction=200)
    5.3 Intent-to-Prediction Mapping (128-D)

  Query cost target: <10ms per search (M=16, ef=50)

Replace the stub logic with your real index loading / nightly-sync code.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.config import get_settings
from backend.models.schemas import ZoneOccupancyProfile, ZoneSemanticSearchResult

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Dimension constants (PRD §5)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_EMBEDDING_DIM = 256      # §5.2 zone semantic similarity
CONTEXT_EMBEDDING_DIM = 128   # §5.1 historical pattern / §5.3 intent

_VENUE_TYPES = ["transit", "retail", "park", "residential", "university", "mixed"]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _zone_to_embedding(
    zone: dict,
    lat_center: float,
    lat_scale: float,
    lon_center: float,
    lon_scale: float,
) -> np.ndarray:
    """
    Build a 256-D zone embedding from structured attributes.

    Encoding strategy
    ─────────────────
    • Lat/lon → sinusoidal encoding across 4 frequencies each (16 dims).
      Low frequencies capture neighbourhood proximity; high frequencies
      distinguish nearby zones.
    • Venue type → one-hot (6 dims).
    • transit_score, capacity (normalised), their interaction (3 dims).
    • Total meaningful dims: 25.  Remaining 231 dims padded with zeros so
      the HNSW L2 distance is dominated by the structured signal.

    Why sinusoidal for coordinates?
    ────────────────────────────────
    A raw (lat, lon) pair gives only 2 dims of spatial signal.  Sinusoidal
    encoding at multiple frequencies (inspired by Transformer positional
    encoding) expands that into a richer representation where both coarse
    proximity and fine-grained distinction are captured without any learned
    parameters.
    """
    lat_norm = (zone.get("lat", lat_center) - lat_center) / max(lat_scale, 1e-6)
    lon_norm = (zone.get("lon", lon_center) - lon_center) / max(lon_scale, 1e-6)

    # 8 dims lat, 8 dims lon (4 frequencies × sin+cos each)
    lat_enc = [fn(lat_norm * np.pi * (2.0 ** i)) for i in range(4) for fn in (np.sin, np.cos)]
    lon_enc = [fn(lon_norm * np.pi * (2.0 ** i)) for i in range(4) for fn in (np.sin, np.cos)]

    # 6-dim one-hot venue type
    vtype = zone.get("venue_type", "mixed")
    venue_enc = [1.0 if vtype == vt else 0.0 for vt in _VENUE_TYPES]

    # 3 scalar dims
    transit = float(zone.get("transit_score", 0.5))
    capacity_norm = min(float(zone.get("capacity", 15)), 60.0) / 60.0
    interaction = transit * capacity_norm  # captures "busy transit hubs"

    features = lat_enc + lon_enc + venue_enc + [transit, capacity_norm, interaction]  # 25 dims

    vec = np.zeros(ZONE_EMBEDDING_DIM, dtype=np.float32)
    vec[: len(features)] = features
    return vec


class HNSWSearchService:
    """
    Manages two HNSW indices:
      - zone_index  : 256-D zone characteristic embeddings
      - context_index: 128-D historical day-context + intent embeddings

    On startup, attempts to load pre-built indices from HNSW_INDEX_PATH.
    Falls back to an empty in-memory index (stub mode) if files not found.
    """

    def __init__(self):
        self._zone_index = None
        self._context_index = None
        self._zone_id_map: List[str] = []       # index position → zone_id
        self._context_id_map: List[str] = []    # index position → record_id
        self._zone_embeddings: Dict[str, np.ndarray] = {}  # zone_id → embedding (for lookup)
        self._zone_meta: Dict[str, dict] = {}   # zone_id → {lat, lon, venue_type} for filters

        self._try_load_indices()

    # ── Index construction ────────────────────────────────────────────────

    def build_from_zones(self, zones: List[dict]) -> None:
        """
        Populate the zone HNSW index from a list of zone attribute dicts.

        Called once at startup after DuckDB is loaded.  Normalises coordinates
        against the dataset's own bounding box so embeddings are scale-invariant
        across cities.

        After building, the index is saved to HNSW_INDEX_PATH so subsequent
        restarts can skip this step via _try_load_indices().
        """
        if not zones:
            logger.warning("build_from_zones called with empty zone list — skipping")
            return

        lats = [z.get("lat", 55.86) for z in zones]
        lons = [z.get("lon", -4.25) for z in zones]
        lat_center = float(np.mean(lats))
        lon_center = float(np.mean(lons))
        lat_scale = float(max(max(lats) - lat_center, 1e-6))
        lon_scale = float(max(max(lons) - lon_center, 1e-6))

        self._initialise_zone_index()

        for zone in zones:
            emb = _zone_to_embedding(zone, lat_center, lat_scale, lon_center, lon_scale)
            self._zone_embeddings[zone["zone_id"]] = emb
            self._zone_meta[zone["zone_id"]] = {
                "lat": zone.get("lat", lat_center),
                "lon": zone.get("lon", lon_center),
                "venue_type": zone.get("venue_type", "mixed"),
            }
            self.add_zone(zone["zone_id"], emb)

        self.save_indices()
        logger.info(
            "Built zone HNSW index: %d zones, lat_center=%.4f, lon_center=%.4f",
            len(zones), lat_center, lon_center,
        )

    def get_zone_embedding(self, zone_id: str) -> Optional[np.ndarray]:
        """Return the pre-computed embedding for a zone_id, or None if unknown."""
        return self._zone_embeddings.get(zone_id)

    # ── Public API ────────────────────────────────────────────────────────

    def zone_semantic_search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        max_distance_m: Optional[float] = None,
        venue_type: Optional[str] = None,
        db_store=None,
        query_zone_id: Optional[str] = None,
    ) -> List[ZoneSemanticSearchResult]:
        """
        §5.2 — Zone Semantic Similarity search.
        Returns k zones most similar to query_embedding (256-D).

        Fetches real occupancy_profile (last 24 h, hourly avg) from db_store
        when provided.  Supports venue_type and max_distance_m post-filters.
        Pass query_zone_id when using max_distance_m so the filter has an anchor.
        """
        if self._zone_index is None or self._zone_index.get_current_count() == 0:
            logger.warning("zone_index empty — returning stub results")
            return self._stub_zone_results(k)

        # Resolve anchor lat/lon for distance filtering
        anchor_lat: Optional[float] = None
        anchor_lon: Optional[float] = None
        if max_distance_m is not None and query_zone_id:
            anchor = self._zone_meta.get(query_zone_id, {})
            anchor_lat = anchor.get("lat")
            anchor_lon = anchor.get("lon")

        # Over-fetch so post-filters don't hollow out the result set
        fetch_k = min(k * 4 + 4, self._zone_index.get_current_count())
        labels, distances = self._zone_index.knn_query(
            query_embedding.reshape(1, -1), k=fetch_k
        )

        results: List[ZoneSemanticSearchResult] = []
        for label, dist in zip(labels[0], distances[0]):
            zone_id = self._zone_id_map[label]

            # venue_type post-filter
            if venue_type:
                meta = self._zone_meta.get(zone_id, {})
                if meta.get("venue_type") != venue_type:
                    continue

            # max_distance_m post-filter (Haversine)
            if max_distance_m is not None and anchor_lat is not None:
                meta = self._zone_meta.get(zone_id, {})
                d = _haversine_m(
                    anchor_lat, anchor_lon,
                    meta.get("lat", anchor_lat), meta.get("lon", anchor_lon),
                )
                if d > max_distance_m:
                    continue

            # Fetch hourly occupancy profile from DuckDB
            occupancy_profile: List[ZoneOccupancyProfile] = []
            if db_store is not None:
                try:
                    snaps = db_store.get_zone_snapshots(zone_id, hours=24)
                    hourly: Dict[int, List[float]] = {}
                    for s in snaps:
                        h = int(s.get("hour_of_day", 0))
                        hourly.setdefault(h, []).append(float(s.get("occupancy_pct", 0)))
                    occupancy_profile = [
                        ZoneOccupancyProfile(
                            hour=h,
                            avg_occupancy_pct=round(sum(v) / len(v), 1) if v else 0.0,
                        )
                        for h in range(24)
                        for v in [hourly.get(h, [])]
                    ]
                except Exception:
                    pass  # non-fatal; return empty profile

            results.append(
                ZoneSemanticSearchResult(
                    zone_id=zone_id,
                    similarity=float(1.0 / (1.0 + dist)),
                    occupancy_profile=occupancy_profile,
                )
            )
            if len(results) >= k:
                break

        return results

    def historical_pattern_search(
        self,
        context_embedding: np.ndarray,
        k: int = 5,
    ) -> List[str]:
        """
        §5.1 — Historical Pattern Matching.
        Returns k most similar historical day-context record IDs (128-D query).
        Reduces cold-start error by ~15% vs XGBoost alone.
        """
        if self._context_index is None or self._context_index.get_current_count() == 0:
            logger.warning("context_index empty — returning empty results")
            return []

        labels, _ = self._context_index.knn_query(
            context_embedding.reshape(1, -1),
            k=min(k, self._context_index.get_current_count()),
        )
        return [self._context_id_map[lbl] for lbl in labels[0]]

    def intent_search(
        self,
        intent_embedding: np.ndarray,
        k: int = 10,
    ) -> List[str]:
        """
        §5.3 — Intent-to-Prediction Mapping.
        Queries historical (intent, zone, time_to_arrival) tuples.
        """
        return self.historical_pattern_search(intent_embedding, k=k)

    # ── Index management ─────────────────────────────────────────────────

    def add_zone(self, zone_id: str, embedding: np.ndarray) -> None:
        """Add or update a zone embedding in the zone index."""
        if self._zone_index is None:
            self._initialise_zone_index()
        idx = len(self._zone_id_map)
        self._zone_id_map.append(zone_id)
        self._zone_index.add_items(embedding.reshape(1, -1), [idx])
        self._zone_embeddings[zone_id] = embedding

    def save_indices(self) -> None:
        """Persist indices and zone_id maps to HNSW_INDEX_PATH."""
        path = Path(settings.HNSW_INDEX_PATH)
        path.mkdir(parents=True, exist_ok=True)
        if self._zone_index:
            self._zone_index.save_index(str(path / "zone_index.bin"))
        if self._context_index:
            self._context_index.save_index(str(path / "context_index.bin"))
        with open(path / "zone_id_map.json", "w") as f:
            json.dump({"zone_ids": self._zone_id_map, "zone_meta": self._zone_meta}, f)
        logger.info("HNSW indices saved to %s", str(path))

    # ── Private helpers ───────────────────────────────────────────────────

    def _try_load_indices(self) -> None:
        try:
            import hnswlib  # type: ignore

            path = Path(settings.HNSW_INDEX_PATH)
            zone_path = path / "zone_index.bin"
            ctx_path = path / "context_index.bin"
            map_path = path / "zone_id_map.json"

            if zone_path.exists():
                try:
                    self._zone_index = hnswlib.Index(space="l2", dim=ZONE_EMBEDDING_DIM)
                    self._zone_index.load_index(str(zone_path), max_elements=200_000)
                    self._zone_index.set_ef(settings.HNSW_EF_SEARCH)

                    if map_path.exists():
                        with open(map_path) as f:
                            data = json.load(f)
                        # Support both old (list) and new (dict with meta) formats
                        if isinstance(data, list):
                            self._zone_id_map = data
                        else:
                            self._zone_id_map = data.get("zone_ids", [])
                            self._zone_meta = data.get("zone_meta", {})

                        labels = list(range(len(self._zone_id_map)))
                        if labels:
                            vecs = self._zone_index.get_items(labels)
                            for zone_id, vec in zip(self._zone_id_map, vecs):
                                self._zone_embeddings[zone_id] = np.array(vec, dtype=np.float32)

                    logger.info(
                        "Loaded zone HNSW index (%d zones) from %s",
                        self._zone_index.get_current_count(), str(zone_path),
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "zone_index.bin corrupt or incompatible (%s) — will rebuild on startup",
                        exc,
                    )
                    zone_path.unlink(missing_ok=True)
                    self._initialise_zone_index()
            else:
                logger.info("No zone index found — will build on first startup")
                self._initialise_zone_index()

            if ctx_path.exists():
                try:
                    self._context_index = hnswlib.Index(space="l2", dim=CONTEXT_EMBEDDING_DIM)
                    self._context_index.load_index(str(ctx_path), max_elements=500_000)
                    self._context_index.set_ef(settings.HNSW_EF_SEARCH)
                    logger.info("Loaded context HNSW index from %s", str(ctx_path))
                except RuntimeError as exc:
                    logger.warning("context_index.bin corrupt (%s) — starting empty", exc)
                    ctx_path.unlink(missing_ok=True)
                    self._initialise_context_index()
            else:
                logger.info("No context index found — starting empty")
                self._initialise_context_index()

        except ImportError:
            logger.warning(
                "hnswlib not installed — HNSW search running in stub mode. "
                "Install with: pip install hnswlib"
            )

    def _initialise_zone_index(self) -> None:
        try:
            import hnswlib  # type: ignore

            self._zone_index = hnswlib.Index(space="l2", dim=ZONE_EMBEDDING_DIM)
            self._zone_index.init_index(
                max_elements=200_000,
                M=settings.HNSW_M,
                ef_construction=settings.HNSW_EF_CONSTRUCTION,
            )
            self._zone_index.set_ef(settings.HNSW_EF_SEARCH)
        except ImportError:
            pass

    def _initialise_context_index(self) -> None:
        try:
            import hnswlib  # type: ignore

            self._context_index = hnswlib.Index(space="l2", dim=CONTEXT_EMBEDDING_DIM)
            self._context_index.init_index(
                max_elements=500_000,
                M=settings.HNSW_M,
                ef_construction=settings.HNSW_EF_CONSTRUCTION,
            )
            self._context_index.set_ef(settings.HNSW_EF_SEARCH)
        except ImportError:
            pass

    @staticmethod
    def _stub_zone_results(k: int) -> List[ZoneSemanticSearchResult]:
        stubs = [
            ("zone-stub-001", 0.91),
            ("zone-stub-002", 0.84),
            ("zone-stub-003", 0.76),
        ]
        return [
            ZoneSemanticSearchResult(zone_id=z, similarity=s, occupancy_profile=[])
            for z, s in stubs[:k]
        ]
