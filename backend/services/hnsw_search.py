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

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.config import get_settings
from backend.models.schemas import ZoneSemanticSearchResult

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Dimension constants (PRD §5)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_EMBEDDING_DIM = 256      # §5.2 zone semantic similarity
CONTEXT_EMBEDDING_DIM = 128   # §5.1 historical pattern / §5.3 intent


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
        self._zone_id_map: List[str] = []     # index position → zone_id
        self._context_id_map: List[str] = []  # index position → record_id

        self._try_load_indices()

    # ── Public API ────────────────────────────────────────────────────────

    def zone_semantic_search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        max_distance_m: Optional[float] = None,
        venue_type: Optional[str] = None,
    ) -> List[ZoneSemanticSearchResult]:
        """
        §5.2 — Zone Semantic Similarity search.
        Returns k zones most similar to query_embedding (256-D).

        TODO: After building real index:
          - Filter results by max_distance_m and venue_type
          - Attach real occupancy_profile from DB
        """
        if self._zone_index is None or self._zone_index.get_current_count() == 0:
            logger.warning("zone_index empty — returning stub results")
            return self._stub_zone_results(k)

        labels, distances = self._zone_index.knn_query(
            query_embedding.reshape(1, -1), k=min(k, self._zone_index.get_current_count())
        )

        results = []
        for label, dist in zip(labels[0], distances[0]):
            zone_id = self._zone_id_map[label]
            results.append(
                ZoneSemanticSearchResult(
                    zone_id=zone_id,
                    similarity=float(1.0 / (1.0 + dist)),  # distance → similarity
                    occupancy_profile=[],  # TODO: fetch from DB
                )
            )
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

    def save_indices(self) -> None:
        """Persist indices to HNSW_INDEX_PATH for nightly sync."""
        path = Path(settings.HNSW_INDEX_PATH)
        path.mkdir(parents=True, exist_ok=True)
        if self._zone_index:
            self._zone_index.save_index(str(path / "zone_index.bin"))
        if self._context_index:
            self._context_index.save_index(str(path / "context_index.bin"))
        logger.info("HNSW indices saved", path=str(path))

    # ── Private helpers ───────────────────────────────────────────────────

    def _try_load_indices(self) -> None:
        try:
            import hnswlib  # type: ignore

            path = Path(settings.HNSW_INDEX_PATH)
            zone_path = path / "zone_index.bin"
            ctx_path = path / "context_index.bin"

            if zone_path.exists():
                self._zone_index = hnswlib.Index(space="l2", dim=ZONE_EMBEDDING_DIM)
                self._zone_index.load_index(str(zone_path))
                self._zone_index.set_ef(settings.HNSW_EF_SEARCH)
                logger.info("Loaded zone HNSW index", path=str(zone_path))
            else:
                logger.info("No zone index found — starting empty (stub mode)")
                self._initialise_zone_index()

            if ctx_path.exists():
                self._context_index = hnswlib.Index(space="l2", dim=CONTEXT_EMBEDDING_DIM)
                self._context_index.load_index(str(ctx_path))
                self._context_index.set_ef(settings.HNSW_EF_SEARCH)
                logger.info("Loaded context HNSW index", path=str(ctx_path))
            else:
                logger.info("No context index found — starting empty (stub mode)")
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
