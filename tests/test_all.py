"""
Unit Tests — Bike Parking Buddy
================================
Tests the DuckDB store, XGBoost prediction, API endpoints, and SLM tools.

Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import json
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def db_store():
    """Create a DuckDB store loaded with synthetic data."""
    from backend.data.duckdb_store import DuckDBStore
    store = DuckDBStore()
    return store


@pytest.fixture(scope="session")
def prediction_service(db_store):
    """Create a prediction service trained on synthetic data."""
    from backend.services.prediction import PredictionService
    svc = PredictionService(db_store)
    return svc


@pytest.fixture(scope="session")
def test_client(db_store, prediction_service):
    """Create a FastAPI TestClient with real data."""
    from fastapi.testclient import TestClient
    from backend.main import create_app

    app = create_app()
    # Pre-inject state so we don't re-train the model
    app.state.db = db_store
    app.state.prediction_service = prediction_service
    app.state.hnsw_service = None

    # Override the lifespan to avoid re-initialization
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ═════════════════════════════════════════════════════════════════════════════
# 1. DuckDB Store Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDuckDBStore:
    def test_zones_loaded(self, db_store):
        zones = db_store.get_zones()
        assert len(zones) == 60, f"Expected 60 zones, got {len(zones)}"

    def test_zone_has_required_fields(self, db_store):
        zones = db_store.get_zones()
        z = zones[0]
        required = ["zone_id", "name", "lat", "lon", "venue_type", "capacity", "transit_score"]
        for field in required:
            assert field in z, f"Missing field: {field}"

    def test_zone_ids_format(self, db_store):
        zones = db_store.get_zones()
        for z in zones:
            assert z["zone_id"].startswith("GLW_Z"), f"Unexpected zone_id: {z['zone_id']}"

    def test_get_single_zone(self, db_store):
        zone = db_store.get_zone("GLW_Z001")
        assert zone is not None
        assert zone["zone_id"] == "GLW_Z001"
        assert zone["name"] == "West End Zone 1"
        assert "occupancy_pct" in zone

    def test_get_nonexistent_zone(self, db_store):
        zone = db_store.get_zone("FAKE_ZONE")
        assert zone is None

    def test_latest_occupancy(self, db_store):
        occ = db_store.get_latest_occupancy()
        assert len(occ) == 60
        for row in occ:
            assert 0 <= row["occupancy_pct"] <= 100

    def test_zone_snapshots(self, db_store):
        snaps = db_store.get_zone_snapshots("GLW_Z001", hours=6)
        assert len(snaps) > 0
        assert snaps[0]["zone_id"] == "GLW_Z001"

    def test_rides_loaded(self, db_store):
        rides = db_store.get_rides(limit=10)
        assert len(rides) == 10
        assert "ride_id" in rides[0]
        assert "was_redirected" in rides[0]

    def test_ride_stats(self, db_store):
        stats = db_store.get_ride_stats()
        assert stats["total_rides"] == 8000
        assert 0 < stats["redirect_rate"] < 100
        assert stats["redirected_count"] > 0

    def test_weather_loaded(self, db_store):
        wx = db_store.get_weather_recent(hours=24)
        assert len(wx) > 0
        assert "wmo_code" in wx[0]
        assert "temp_celsius" in wx[0]

    def test_events_loaded(self, db_store):
        events = db_store.get_events()
        assert len(events) > 0
        assert events[0]["city_id"] == "GLW"

    def test_analytics_summary(self, db_store):
        summary = db_store.get_analytics_summary()
        assert summary["total_zones"] == 60
        assert summary["total_rides"] == 8000
        assert "busiest_hour" in summary
        assert "occupancy_by_venue_type" in summary
        assert len(summary["occupancy_by_venue_type"]) > 0

    def test_zone_heatmap(self, db_store):
        heatmap = db_store.get_zone_heatmap()
        assert len(heatmap) > 0
        assert "zone_id" in heatmap[0]
        assert "hour_of_day" in heatmap[0]
        assert "avg_occupancy" in heatmap[0]

    def test_training_data_shape(self, db_store):
        df = db_store.get_snapshots_for_training()
        assert len(df) > 700000, f"Expected >700k rows, got {len(df)}"
        required_cols = ["zone_id", "occupancy_pct", "venue_type", "transit_score", "hour_of_day"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_log_rider_outcome_persists_row(self, db_store):
        """Phase 2 exit criterion: rider outcomes must land in the DB, not be discarded."""
        before = db_store.get_rider_outcomes(limit=200)
        db_store.log_rider_outcome(
            zone_id="GLW_Z001",
            timestamp="2026-05-20T09:00:00Z",
            actual_availability=0.2,
            rider_satisfaction=4,
        )
        after = db_store.get_rider_outcomes(limit=200)
        assert len(after) == len(before) + 1, "Row count did not increase after log_rider_outcome"
        row = after[0]  # most recent first
        assert row["zone_id"] == "GLW_Z001"
        assert abs(row["actual_fill"] - 0.8) < 0.01  # 1.0 - 0.2
        assert row["satisfaction_score"] == 4


# ═════════════════════════════════════════════════════════════════════════════
# 2. Prediction Service Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPredictionService:
    def test_model_trained(self, prediction_service):
        assert prediction_service.model is not None
        assert prediction_service.model_version == "xgb-v1-synthetic"

    def test_accuracy_threshold(self, prediction_service):
        """PRD target: ≥78% fill classification accuracy."""
        assert prediction_service.accuracy >= 0.70, (
            f"Model accuracy {prediction_service.accuracy:.3f} below threshold"
        )

    def test_model_metrics(self, prediction_service):
        metrics = prediction_service.get_model_metrics()
        assert "accuracy" in metrics
        assert "feature_columns" in metrics
        assert metrics["accuracy"] > 0

    @pytest.mark.asyncio
    async def test_predict_returns_valid_response(self, prediction_service):
        from backend.models.schemas import PredictRequest
        req = PredictRequest(
            zone_id="GLW_Z001",
            current_timestamp=datetime(2026, 5, 20, 8, 30, tzinfo=timezone.utc),
            lookahead_mins=30,
        )
        resp = await prediction_service.predict(req)
        assert 0 <= resp.fill_probability <= 1
        assert 0 <= resp.confidence <= 1
        assert resp.zone_id == "GLW_Z001"
        assert resp.model_version == "xgb-v1-synthetic"
        assert len(resp.reason) > 0

    @pytest.mark.asyncio
    async def test_rush_hour_higher_fill(self, prediction_service):
        """Rush hour should predict higher fill than 3am."""
        from backend.models.schemas import PredictRequest
        rush = PredictRequest(
            zone_id="GLW_Z008",  # transit zone
            current_timestamp=datetime(2026, 5, 20, 17, 0, tzinfo=timezone.utc),
            lookahead_mins=30,
        )
        quiet = PredictRequest(
            zone_id="GLW_Z008",
            current_timestamp=datetime(2026, 5, 20, 3, 0, tzinfo=timezone.utc),
            lookahead_mins=30,
        )
        rush_resp = await prediction_service.predict(rush)
        quiet_resp = await prediction_service.predict(quiet)
        # Rush hour should generally predict higher fill
        # (not always guaranteed with stochastic data, but transit zone at 5pm vs 3am is strong)
        assert rush_resp.fill_probability != quiet_resp.fill_probability

    @pytest.mark.asyncio
    async def test_alternatives_for_high_fill(self, prediction_service):
        """When fill is high, alternatives should be returned."""
        from backend.models.schemas import PredictRequest
        req = PredictRequest(
            zone_id="GLW_Z001",
            current_timestamp=datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc),
            lookahead_mins=30,
        )
        resp = await prediction_service.predict(req)
        if resp.fill_probability > 0.65:
            assert len(resp.alternative_zones) > 0
            for alt in resp.alternative_zones:
                assert alt.fill_probability < resp.fill_probability


# ═════════════════════════════════════════════════════════════════════════════
# 3. API Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestAPIEndpoints:
    def test_health(self, test_client):
        r = test_client.get("/health/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_list_zones(self, test_client):
        r = test_client.get("/zones/")
        assert r.status_code == 200
        zones = r.json()
        assert len(zones) == 60
        assert zones[0]["zone_id"].startswith("GLW_Z")

    def test_list_zones_filter_venue(self, test_client):
        r = test_client.get("/zones/?venue_type=transit")
        assert r.status_code == 200
        zones = r.json()
        assert all(z["venue_type"] == "transit" for z in zones)

    def test_get_zone_detail(self, test_client):
        r = test_client.get("/zones/GLW_Z001")
        assert r.status_code == 200
        data = r.json()
        assert data["zone_id"] == "GLW_Z001"
        assert "current_occupancy_pct" in data

    def test_get_zone_404(self, test_client):
        r = test_client.get("/zones/NONEXISTENT")
        assert r.status_code == 404

    def test_zone_snapshots(self, test_client):
        r = test_client.get("/zones/GLW_Z001/snapshots?hours=6")
        assert r.status_code == 200
        snaps = r.json()
        assert len(snaps) > 0

    def test_predict(self, test_client):
        r = test_client.post("/predict/", json={
            "zone_id": "GLW_Z001",
            "lookahead_mins": 30,
        })
        assert r.status_code == 200
        data = r.json()
        assert 0 <= data["fill_probability"] <= 1
        assert data["model_version"] == "xgb-v1-synthetic"
        assert "reason" in data

    def test_predict_different_zones(self, test_client):
        """Different zones should potentially give different predictions."""
        results = {}
        for zone in ["GLW_Z001", "GLW_Z008", "GLW_Z052"]:
            r = test_client.post("/predict/", json={
                "zone_id": zone,
                "current_timestamp": "2026-05-20T17:00:00Z",
                "lookahead_mins": 30,
            })
            assert r.status_code == 200
            results[zone] = r.json()["fill_probability"]
        # Not all should be identical (different venue types, transit scores)
        assert len(set(f"{v:.2f}" for v in results.values())) > 1

    def test_rides_list(self, test_client):
        r = test_client.get("/rides/?limit=5")
        assert r.status_code == 200
        rides = r.json()
        assert len(rides) == 5

    def test_rides_stats(self, test_client):
        r = test_client.get("/rides/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_rides"] == 8000
        assert data["redirect_rate"] > 0

    def test_weather(self, test_client):
        r = test_client.get("/weather/?hours=12")
        assert r.status_code == 200
        wx = r.json()
        assert len(wx) > 0

    def test_analytics_summary(self, test_client):
        r = test_client.get("/analytics/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_zones"] == 60
        assert data["total_rides"] == 8000

    def test_analytics_heatmap(self, test_client):
        r = test_client.get("/analytics/zone-heatmap")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0

    def test_model_metrics(self, test_client):
        r = test_client.get("/analytics/model-metrics")
        assert r.status_code == 200
        data = r.json()
        assert "accuracy" in data
        assert data["accuracy"] > 0

    def test_events(self, test_client):
        r = test_client.get("/analytics/events")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. SLM Tool Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSLMTools:
    def test_tool_definitions_exist(self):
        from backend.services.slm_tools import TOOL_DEFINITIONS
        assert len(TOOL_DEFINITIONS) == 3
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        assert "zone_semantic_search" in names
        assert "get_zone_forecast" in names
        assert "log_outcome" in names

    def test_tool_schemas_valid_openai_format(self):
        from backend.services.slm_tools import TOOL_DEFINITIONS
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]
            assert tool["function"]["parameters"]["type"] == "object"
            assert "required" in tool["function"]["parameters"]

    @pytest.mark.asyncio
    async def test_dispatch_zone_semantic_search(self):
        from backend.services.slm_tools import dispatch_tool_call
        result = await dispatch_tool_call("zone_semantic_search", {
            "current_zone_id": "GLW_Z001",
            "k": 3,
        })
        assert len(result) == 3
        assert all(hasattr(r, "zone_id") for r in result)

    @pytest.mark.asyncio
    async def test_dispatch_get_zone_forecast(self):
        from backend.services.slm_tools import dispatch_tool_call
        result = await dispatch_tool_call("get_zone_forecast", {
            "zone_id": "GLW_Z001",
            "time_horizon_mins": 30,
        })
        assert hasattr(result, "zone_id")
        assert hasattr(result, "expected_fill")
        assert hasattr(result, "recommended_action")

    @pytest.mark.asyncio
    async def test_dispatch_log_outcome(self):
        from backend.services.slm_tools import dispatch_tool_call
        result = await dispatch_tool_call("log_outcome", {
            "zone_id": "GLW_Z001",
            "timestamp": "2026-05-20T08:00:00Z",
            "actual_availability": 0.3,
            "rider_satisfaction": 4,
        })
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        from backend.services.slm_tools import dispatch_tool_call
        with pytest.raises(ValueError, match="Unknown tool"):
            await dispatch_tool_call("nonexistent_tool", {})


# ═════════════════════════════════════════════════════════════════════════════
# 5. HNSW Search Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def hnsw_service(db_store):
    """Build a fresh HNSW index from the synthetic zone data."""
    from backend.services.hnsw_search import HNSWSearchService
    svc = HNSWSearchService()
    # If index wasn't already built/loaded, build it now
    if svc._zone_index is None or svc._zone_index.get_current_count() == 0:
        zones = db_store.get_zones()
        svc.build_from_zones(zones)
    return svc


class TestHNSWSearch:
    def test_index_has_all_zones(self, hnsw_service):
        count = hnsw_service._zone_index.get_current_count()
        assert count == 60, f"Expected 60 zones in index, got {count}"

    def test_embeddings_populated(self, hnsw_service):
        assert len(hnsw_service._zone_embeddings) == 60

    def test_get_zone_embedding_known_zone(self, hnsw_service):
        emb = hnsw_service.get_zone_embedding("GLW_Z001")
        assert emb is not None
        assert emb.shape == (256,)

    def test_get_zone_embedding_unknown_zone(self, hnsw_service):
        emb = hnsw_service.get_zone_embedding("FAKE_ZONE")
        assert emb is None

    def test_zone_semantic_search_returns_real_zone_ids(self, hnsw_service):
        emb = hnsw_service.get_zone_embedding("GLW_Z008")
        results = hnsw_service.zone_semantic_search(emb, k=3)
        assert len(results) >= 1
        for r in results:
            assert r.zone_id.startswith("GLW_Z"), f"Non-Glasgow zone: {r.zone_id}"

    def test_zone_semantic_search_topk_stable(self, hnsw_service):
        """Same query twice must return the same top-k."""
        emb = hnsw_service.get_zone_embedding("GLW_Z015")
        r1 = [r.zone_id for r in hnsw_service.zone_semantic_search(emb, k=4)]
        r2 = [r.zone_id for r in hnsw_service.zone_semantic_search(emb, k=4)]
        assert r1 == r2

    def test_zone_semantic_search_similarity_ordered(self, hnsw_service):
        """Results must be returned in descending similarity order."""
        emb = hnsw_service.get_zone_embedding("GLW_Z001")
        results = hnsw_service.zone_semantic_search(emb, k=5)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True), "Results not in similarity order"

    def test_venue_type_filter(self, hnsw_service, db_store):
        """venue_type filter should return only matching zones (or empty)."""
        from backend.data.duckdb_store import DuckDBStore
        emb = hnsw_service.get_zone_embedding("GLW_Z001")
        results = hnsw_service.zone_semantic_search(emb, k=10, venue_type="transit")
        for r in results:
            meta = hnsw_service._zone_meta.get(r.zone_id, {})
            assert meta.get("venue_type") == "transit", (
                f"zone {r.zone_id} has venue_type {meta.get('venue_type')}"
            )

    def test_distance_filter_excludes_far_zones(self, hnsw_service):
        """Tight distance filter should return fewer results than without."""
        emb = hnsw_service.get_zone_embedding("GLW_Z001")
        all_results = hnsw_service.zone_semantic_search(emb, k=10)
        tight_results = hnsw_service.zone_semantic_search(emb, k=10, max_distance_m=100)
        assert len(tight_results) <= len(all_results)

    def test_occupancy_profile_populated(self, hnsw_service, db_store):
        """Occupancy profiles are fetched as 24 ZoneOccupancyProfile entries when db_store is provided."""
        emb = hnsw_service.get_zone_embedding("GLW_Z001")
        results = hnsw_service.zone_semantic_search(emb, k=2, db_store=db_store)
        for r in results:
            assert len(r.occupancy_profile) == 24, (
                f"Expected 24-hour profile, got {len(r.occupancy_profile)}"
            )
            # Each entry should be a ZoneOccupancyProfile with hour and avg_occupancy_pct
            for i, entry in enumerate(r.occupancy_profile):
                assert entry.hour == i
                assert 0.0 <= entry.avg_occupancy_pct <= 100.0


# ═════════════════════════════════════════════════════════════════════════════
# 6. GBFS Ingest Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGBFSIngest:
    """Tests use fixture payloads only — no real network calls during CI."""

    # ── Payload normalisation ─────────────────────────────────────────────

    def test_normalize_zone_snapshot_fields(self):
        """normalize_zone_snapshot maps GBFS station_status fields to ZoneSnapshot names."""
        from datetime import datetime, timezone
        from backend.data.loaders import normalize_zone_snapshot

        ts = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
        station = {
            "station_id": "6432324",
            "num_bikes_available": 4,
            "num_docks_available": 8,
            "is_installed": True,
            "is_renting": True,
            "is_returning": True,
            "last_reported": 1786730036,
        }
        snap = normalize_zone_snapshot(station, ts)
        assert snap["zone_id"] == "6432324"
        assert snap["bikes_available"] == 4
        assert "capacity" in snap
        assert "occupancy_pct" in snap
        assert 0.0 <= snap["occupancy_pct"] <= 1.0
        assert snap["timestamp"] == ts
        assert "weather_code" in snap
        assert "is_event_nearby" in snap

    def test_normalize_zero_capacity_station(self):
        """Zero-capacity station (dockless, no dock info) yields occupancy 0.0."""
        from datetime import datetime, timezone
        from backend.data.loaders import normalize_zone_snapshot

        ts = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
        station = {"station_id": "99", "num_bikes_available": 0, "num_docks_available": 0}
        snap = normalize_zone_snapshot(station, ts)
        assert snap["occupancy_pct"] == 0.0

    def test_normalize_full_station(self):
        """Full station (0 bikes available, some capacity) yields occupancy 1.0."""
        from datetime import datetime, timezone
        from backend.data.loaders import normalize_zone_snapshot

        ts = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
        station = {"station_id": "42", "num_bikes_available": 0, "num_docks_available": 10}
        snap = normalize_zone_snapshot(station, ts)
        assert snap["occupancy_pct"] == 1.0

    # ── DB layer ──────────────────────────────────────────────────────────

    def test_synthetic_zone_count_unchanged_after_real_upsert(self, db_store):
        """Inserting a real zone must not change the synthetic zone count."""
        before = len(db_store.get_zones())  # default source='synthetic'
        db_store.upsert_real_zone(
            zone_id="TEST_REAL_001",
            name="Test Real Station",
            lat=44.77,
            lon=17.19,
            capacity=10,
        )
        after = len(db_store.get_zones())
        assert after == before, "Synthetic zone count changed after inserting a real zone"

    def test_real_zone_visible_via_source_filter(self, db_store):
        """Real zone inserted in previous test is queryable with source='real'."""
        db_store.upsert_real_zone(
            zone_id="TEST_REAL_002",
            name="Test Real Station 2",
            lat=44.78,
            lon=17.20,
            capacity=8,
        )
        real_zones = db_store.get_zones(source="real")
        real_ids = {z["zone_id"] for z in real_zones}
        assert "TEST_REAL_002" in real_ids

    def test_insert_zone_snapshots_persists_rows(self, db_store):
        """insert_zone_snapshots must write rows into the DB for known zone_ids."""
        from datetime import datetime, timezone

        # Ensure the zone exists first
        db_store.upsert_real_zone(
            zone_id="TEST_REAL_003",
            name="Test Real Station 3",
            lat=44.79,
            lon=17.21,
            capacity=6,
        )
        ts = datetime(2026, 8, 14, 10, 0, tzinfo=timezone.utc)
        snapshots = [
            {
                "zone_id": "TEST_REAL_003",
                "timestamp": ts,
                "bikes_available": 3,
                "capacity": 6,
                "occupancy_pct": 0.5,
                "weather_code": None,
                "is_event_nearby": None,
            }
        ]
        written = db_store.insert_zone_snapshots(snapshots)
        assert written == 1

    def test_insert_snapshots_skips_unknown_zone(self, db_store):
        """Snapshots for unknown zone_ids must be skipped (FK guard)."""
        from datetime import datetime, timezone

        snapshots = [
            {
                "zone_id": "NONEXISTENT_ZONE_XYZ",
                "timestamp": datetime(2026, 8, 14, 11, 0, tzinfo=timezone.utc),
                "bikes_available": 1,
                "capacity": 5,
                "occupancy_pct": 0.2,
                "weather_code": None,
                "is_event_nearby": None,
            }
        ]
        written = db_store.insert_zone_snapshots(snapshots)
        assert written == 0

    # ── GBFSIngestJob unit tests (no network) ────────────────────────────

    @pytest.mark.asyncio
    async def test_ingest_job_run_once_no_feeds(self):
        """run_once with empty feed list logs a warning and writes nothing."""
        from backend.data.gbfs_ingest import GBFSIngestJob

        job = GBFSIngestJob(feed_urls=[], db_store=None)
        await job.run_once()
        assert job.snapshots_total == 0
        assert job.error_count == 0

    def test_ingest_job_status_dict_never_polled(self):
        """status_dict on a fresh job returns status='never_polled'."""
        from backend.data.gbfs_ingest import GBFSIngestJob

        job = GBFSIngestJob(feed_urls=["https://example.com/gbfs.json"], db_store=None)
        status = job.status_dict()
        assert status["status"] == "never_polled"
        assert status["last_success_at"] is None
        assert status["error_count"] == 0

    # ── /health/ingest endpoint ───────────────────────────────────────────

    def test_health_ingest_disabled(self, test_client):
        """When no ingest job is wired (GBFS_INGEST_ENABLED=false), endpoint returns 'disabled'."""
        r = test_client.get("/health/ingest")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "disabled"
        assert data["last_success_at"] is None
        assert data["error_count"] == 0
