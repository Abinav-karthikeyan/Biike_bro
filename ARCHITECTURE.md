# Architecture

## Repository layout

```
Cycle_Buddy/
├── backend/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── config.py            # Pydantic Settings (all env vars)
│   ├── requirements.txt
│   ├── data/
│   │   ├── duckdb_store.py  # SQLAlchemy-backed data layer (DuckDB / Postgres)
│   │   ├── gbfs_ingest.py   # Async GBFS feed polling (gated by GBFS_INGEST_ENABLED)
│   │   └── loaders.py       # CSV seed loader
│   ├── models/
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── db_models.py     # SQLAlchemy ORM (5 core tables)
│   ├── routers/
│   │   ├── health.py        # GET /health/
│   │   ├── predict.py       # POST /predict/
│   │   ├── recommend.py     # GET /recommend/  (retrieval-first, no LLM)
│   │   ├── zones.py
│   │   ├── rides.py
│   │   ├── weather.py
│   │   ├── analytics.py
│   │   └── slm.py           # POST /slm/query, /slm/predict
│   └── services/
│       ├── prediction.py    # XGBoost zone-fill classifier
│       ├── hnsw_search.py   # hnswlib zone index (256-D) + context index (128-D)
│       ├── geofence.py      # GeofenceRuleEngine — GBFS ride_end_rules evaluation
│       ├── rag_context.py   # RAGContextAssembler — multi-signal context for LLM
│       ├── slm_service.py   # Groq/Llama inference with tool-calling
│       └── slm_tools.py     # OpenAI-compatible tool schemas + dispatcher
├── synthetic_seed/          # 60-zone Glasgow dataset (CSVs + geofencing JSON)
├── scripts/
│   └── build_context_index.py  # One-shot: populate 128-D HNSW context index
├── tests/
│   └── test_all.py          # 103 tests
├── models/
│   └── xgb_model.joblib     # Persisted XGBoost model
├── data/
│   └── hnsw_indices/        # zone_index.bin, context_index.bin, zone_id_map.json
├── Dockerfile               # Root Dockerfile (used by Render)
├── render.yaml              # Render.com deploy config
└── docker-compose.yml       # Local Postgres + Redis (dev only)
```

---

## Data pipeline

```
GBFS Feeds (optional, gated)
    │  async poll every 5 min
    ▼
zone_snapshots (DuckDB in-memory / Postgres)
    │
    ├──▶ XGBoost Classifier  ──▶  fill_probability, confidence, alternatives
    │
    ├──▶ HNSW Zone Index     ──▶  256-D semantic zone similarity
    │         (lat/lon sinusoidal, venue type one-hot, transit score, capacity)
    │
    └──▶ HNSW Context Index  ──▶  128-D (zone × hour × day_of_week) buckets
                                    │
                                    ▼
                              Groq / Llama-3.1-8b
                              Tool-calling + RAG narration
```

---

## Inference tiers

| Tier | Model | RAG budget | Use |
|------|-------|-----------|-----|
| cloud | `llama-3.1-8b-instant` | ~500 tokens | Default |
| edge | `llama-3.2-1b-preview` | ~200 tokens | Compressed context |

Both served via Groq API. Falls back to XGBoost-only stub if `GROQ_API_KEY` is unset.

---

## Retrieval-first fast path (`GET /recommend/`)

No LLM in the critical path:

```
HNSW zone_semantic_search(k+1)
    │  exclude anchor zone
    ▼
GeofenceRuleEngine.evaluate(zone_id)  — O(1) dict lookup
    │
    ▼
XGBoost batch predict_proba(X)  — single call for all candidates
    │
    ▼
Sort: fill ascending, rule-violating zones pushed to back
    │
    ▼
RecommendResponse { candidates, latency_ms }
```

---

## RAG context signals

| Signal | Tier |
|--------|------|
| (a) Current occupancy | both |
| (b) 7-day same-hour average | both |
| (c) HNSW similar zones | cloud only |
| (d) Weather WMO code | both |
| (e) Nearby events ≤1km / ≤2h | cloud only |
| (f) Geofence rule violations | both |

---

## SLM tool-calling flow

```
"Will GLW_Z008 be full in 20 min?"
    │
    ▼
RAG context injected into system prompt
    │
    ▼
Groq chat.completions.create(tools=TOOL_DEFINITIONS)
    ├── get_zone_forecast(zone_id, time_horizon_mins)  → XGBoost
    ├── zone_semantic_search(current_zone_id, k)       → HNSW
    └── log_outcome(zone_id, actual_availability)      → DuckDB
    │
    ▼
Second pass: tool results → natural-language summary
    "72% full in 20 min — try GLW_Z015 (42% full, 350m away)."
```

---

## Database schema

| Table | Purpose | Backend |
|-------|---------|---------|
| `zone_metadata` | Semi-static zone reference | DuckDB (seed CSV) |
| `zone_snapshots` | 5-min occupancy time-series (725k rows) | DuckDB |
| `rider_outcomes` | Opt-in telemetry (`intent_embedding` JSON col) | DuckDB / ORM |
| `model_artifacts` | Versioned ML model registry | ORM (defined, unused) |
| `zone_embeddings_hnsw` | Pre-computed embeddings | ORM (defined, unused) |

Production target: Postgres via `alembic upgrade head`. Default deploy uses DuckDB in-memory re-seeded from CSVs on startup.

---

## ML model

- **Algorithm**: XGBoost binary classifier
- **Target**: `zone_full` (occupancy_pct > 85%)
- **Features**: hour_of_day, day_of_week, weather_code, temperature, event_proximity, venue_type, transit_score, capacity, is_raining, is_rush_hour, is_weekend
- **Accuracy**: ~80% on holdout set
- **Latency**: <10ms per prediction
- **Alternatives**: top-3 zones within 3km with lower predicted fill

---

## Environment variables

| Key | Default | Required |
|-----|---------|----------|
| `GROQ_API_KEY` | `""` | Yes (for LLM; falls back to stub) |
| `GROQ_CLOUD_MODEL` | `llama-3.1-8b-instant` | No |
| `GROQ_EDGE_MODEL` | `llama-3.2-1b-preview` | No |
| `DATABASE_URL` | `duckdb:///:memory:` | No |
| `GBFS_INGEST_ENABLED` | `false` | No |
| `APP_ENV` | `development` | No |
| `CORS_ORIGINS` | `["*"]` (production) | No |
