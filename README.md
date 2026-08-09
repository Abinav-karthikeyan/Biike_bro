# Bike Parking Buddy

> Hybrid ML + SLM predictive parking intelligence for dockless bike-sharing in Glasgow.
> **Synthetic data prototype** — all zone/ride/weather data is generated; no live GBFS feed yet.

**Version**: v1.0-prototype — XGBoost + HNSW + Qwen2.5 (Ollama), DuckDB-backed, 41 tests passing.

[![CI](https://github.com/Abinav-karthikeyan/Biike_bro/actions/workflows/ci.yml/badge.svg)](https://github.com/Abinav-karthikeyan/Biike_bro/actions/workflows/ci.yml)

---

## Project Structure

```
Biike_bro/
│
├── backend/                     # FastAPI application
│   ├── main.py                  # App factory, CORS, lifespan
│   ├── config.py                # Pydantic Settings (env vars)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/
│   │   ├── duckdb_store.py      # In-memory DuckDB data store
│   │   ├── gbfs_ingest.py       # Async GBFS feed polling client
│   │   └── mock_data.py         # Dev seed data
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── db_models.py         # SQLAlchemy ORM (5 core tables)
│   ├── routers/
│   │   ├── health.py            # GET /health/
│   │   ├── predict.py           # POST /predict/
│   │   ├── zones.py             # GET /zones/, GET /zones/{id}/snapshots
│   │   ├── rides.py             # GET /rides/, GET /rides/stats
│   │   ├── weather.py           # GET /weather/
│   │   ├── analytics.py         # GET /analytics/summary, /zone-heatmap, /model-metrics, /events
│   │   └── slm.py               # POST /slm/query, /slm/predict, GET /slm/status, /slm/tools
│   └── services/
│       ├── prediction.py        # XGBoost zone fill prediction
│       ├── hnsw_search.py       # hnswlib zone/context/intent search
│       ├── slm_service.py       # Ollama/Qwen2.5 inference with tool-calling
│       └── slm_tools.py         # SLM function-calling tool definitions
│
├── dashboard/                   # Developer web dashboard
│   ├── index.html               # 5-panel dark glassmorphism UI
│   ├── test_interface.html      # API test interface with run-all
│   ├── css/style.css
│   └── js/app.js
│
├���─ archive/
│   └── finetune_v1/             # Fine-tuning pipeline (archived — V2 uses RAG instead)
│
├── qwen/                        # Qwen2.5 model integration
│   ├── ollama_qwen.py           # Minimal Ollama chat test
│   └── qwen_demo.py             # Transformers inference demo
│
├── synthetic_seed/              # Synthetic Glasgow dataset (60 zones, 725k+ snapshots, 8k rides)
│   ├── generate_synthetic_data.py
│   ├── zones.csv, bikes.csv, rides.csv, weather.csv, etc.
│   └── geofencing_zones.json, free_bike_status.json
│
├── scripts/
│   └── build_hnsw_index.py     # One-shot: build HNSW index from DuckDB zones
│
├── tests/
│   └── test_all.py             # 51 tests — DuckDB, XGBoost, API, SLM tools, HNSW
│
├── data/
│   ├── hnsw_indices/            # HNSW index binaries (built by scripts/build_hnsw_index.py)
│   └── finetune/                # Training data JSONL
│
├── models/
│   └── xgb_model.joblib         # Persisted XGBoost model (loaded on startup)
│
├── plan_v2/                     # V2 implementation plan
├── docker-compose.yml           # PostgreSQL 15 + Redis 7 + API [stub — V2]
├── GOVERNANCE.md                # Honest gap list — what's missing and why
├── .env.example                 # Configuration template
├── .gitignore
└── run.py                       # Dev server launcher
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed with `qwen2.5:7b` pulled:
  ```bash
  ollama pull qwen2.5
  ```

### 1. Clone and set up environment

```bash
git clone https://github.com/Abinav-karthikeyan/Biike_bro && cd Biike_bro
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local dev)
```

### 3. Run the API

```bash
python run.py
# or: uvicorn backend.main:app --reload --port 8000
```

**Open these in your browser:**
- API & interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Developer dashboard: open `dashboard/index.html` (use Live Server or VS Code Go Live)

### 4. Run tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/` | Liveness check |
| GET | `/health/ready` | Readiness probe |
| | | |
| POST | `/predict/` | Zone fill probability prediction |
| | | |
| GET | `/zones/` | List all zones (with optional `venue_type`, `min_transit_score` filters) |
| GET | `/zones/{zone_id}` | Zone detail with latest occupancy |
| GET | `/zones/{zone_id}/snapshots` | Time-series snapshots (`?hours=24`) |
| GET | `/zones/{zone_id}/occupancy-history` | 7-day hourly-downsampled chart data |
| | | |
| GET | `/rides/` | Recent rides (`?limit=50`) |
| GET | `/rides/stats` | Aggregate ride statistics |
| | | |
| GET | `/weather/` | Recent weather records (`?hours=48`) |
| | | |
| GET | `/analytics/summary` | KPI dashboard data |
| GET | `/analytics/zone-heatmap` | Occupancy by zone/hour |
| GET | `/analytics/model-metrics` | XGBoost performance metrics |
| GET | `/analytics/events` | Local events list |
| | | |
| POST | `/slm/query` | Natural-language query to Qwen2.5 + tool dispatch |
| POST | `/slm/predict` | Hybrid XGBoost + SLM narrative prediction |
| GET | `/slm/status` | Ollama connectivity and model info |
| GET | `/slm/tools` | OpenAI-compatible tool function schemas |

---

## Architecture Overview

### Data Pipeline

```
GBFS Feeds (Lime/Voi/Tier)
    │  5-min polling
    ▼
zone_snapshots (PostgreSQL — V2 target)
    │
    ├──▶ XGBoost Classifier  ──▶  fill_probability, confidence, alternatives
    │
    ├──▶ HNSW Zone Index    ──▶  semantic zone similarity (256-D embeddings)
    │
    └──▶ HNSW Context Index ──▶  historical pattern matching (128-D)
                                    │
                                    ▼
                              Qwen2.5 (Ollama)
                              Tool-calling + Narrative
```

### SLM Tool-Calling Flow

```
"Will GLW_Z008 be full in 20 min?"
    │
    ▼
Qwen2.5 decides via tool_calling:
    ├── get_zone_forecast(zone_id="GLW_Z008", time_horizon_mins=20)
    │       → XGBoost returns fill_probability=0.72
    │
    ├── zone_semantic_search(current_zone_id="GLW_Z008", k=3)
    │       → HNSW returns similar zones with occupancy profiles
    │
    └── log_outcome(...)
            → Telemetry recording (opt-in)
    │
    ▼
Qwen2.5 second pass:
    "It's predicted to be 72% full — might be tight. Try GLW_Z012 instead (42% full, 350m away)."
```

---

## V1 Core Components

### ML Prediction (`backend/services/prediction.py`)
- **Model**: XGBoost classifier trained on 725k+ synthetic snapshots
- **Target**: Binary classification of "zone full" (occupancy_pct > 85%)
- **Features**: hour_of_day, day_of_week, weather_code, temperature, event proximity, venue type, transit score, capacity, rain, rush hour, weekend
- **Performance**: ~80% accuracy on holdout set
- **Alternatives**: Returns top 3 nearby zones (within 3km) with lower predicted fill

### HNSW Vector Search (`backend/services/hnsw_search.py`)
- **Zone index**: 256-D embeddings `[lat, lon (sinusoidal), venue_type (one-hot), transit_score, capacity]`
- **Context index**: 128-D day-context embeddings [stub — V2]
- **Search cost**: <10ms per query (M=16, ef=50)
- **Build**: `python scripts/build_hnsw_index.py` — deterministic from DuckDB seed data
- **Persistence**: Save/load from `data/hnsw_indices/`

### SLM + Tool-Calling (`backend/services/slm_service.py` + `slm_tools.py`)
- **Model**: Qwen2.5 (7B, Q4_K_M via Ollama)
- **Tools**: 3 OpenAI-compatible function schemas
  - `zone_semantic_search` — similarity search via HNSW
  - `get_zone_forecast` — XGBoost prediction
  - `log_outcome` — telemetry recording
- **Fallback**: If Ollama unreachable, returns XGBoost-only stub response
- **Latency SLO** (V2 target): <200ms for structured prediction, <2s for SLM narrative

### Data Connectors (`backend/data/gbfs_ingest.py`) [stub — V2]
- **GBFS v2.3 client**: Auto-discovery manifest parsing, station_status feed
- **Ingest**: Upsert path stubbed — no operator feed URL wired yet
- **Enrichment** (V2): Weather + local events cross-reference
- **Operators**: Target Lime, Voi, Tier (any GBFS-compatible operator)

---

## Database Schema (PRD §7)

| Table | Purpose | Status |
|-------|---------|--------|
| `zone_metadata` | Semi-static zone reference data | DuckDB in-memory (seed CSV) |
| `zone_snapshots` | 5-min occupancy time-series | DuckDB in-memory (725k rows) |
| `model_artifacts` | Versioned ML model registry | SQLAlchemy model defined, unused [stub — V2] |
| `rider_outcomes` | Opt-in telemetry for retraining | Written to DuckDB in-memory; lost on restart [stub — V2] |
| `zone_embeddings_hnsw` | Pre-computed 256-D zone embeddings | SQLAlchemy model defined, unused [stub — V2] |

**Current**: DuckDB in-memory; PostgreSQL migration is Phase 2 (see `roadmap_new.md`).
**Migration**: `alembic upgrade head` once Postgres is configured [stub — V2].

---

## What's Next

See [`roadmap_new.md`](roadmap_new.md) for the full sequenced plan. Summary:

| Phase | Status | Goal |
|-------|--------|------|
| 0 — Display-ready pass | **Done** | Reconcile docs with code, CI badge, governance file |
| 1 — Wire HNSW | **Done** | Real zone similarity search, no stubs |
| 2 — Persistence | Planned | SQLAlchemy ORM writes → Alembic → Postgres |
| 3 — GBFS live ingest | Planned | Real operator feed, drop "synthetic-only" |
| 4 — RAG context | Planned | Historical context injected into SLM prompts |
| 5 — Observability | Planned | Prometheus metrics, SLM eval harness |
| 6 — Deployment | Planned | Single VPS + Docker Compose + public URL |

---

## Success Metrics (PRD §3)

- **Prediction accuracy**: ≥78% fill classification (30-min lookahead)
- **API latency**: <200ms structured prediction
- **SLM latency**: <2s end-to-end narrative (V2 target)
- **Wasted minutes**: 25–35% fewer rider redirects

---

## API Example

**Request:**
```bash
curl -X POST http://localhost:8000/slm/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Will GLW_Z008 be full in 20 minutes?", "zone_context": "GLW_Z008"}'
```

**Response (Ollama available):**
```json
{
  "content": "GLW_Z008 is predicted 72% full in 20 minutes — it could be tight. GLW_Z015 and GLW_Z016 are nearby with similar transit connectivity and lower predicted fill.",
  "tool_calls": ["get_zone_forecast", "zone_semantic_search"],
  "tool_results": [
    {"tool": "get_zone_forecast", "result": {"fill_probability": 0.72, "confidence": 0.85}},
    {"tool": "zone_semantic_search", "result": [{"zone_id": "GLW_Z015", "similarity": 0.29}]}
  ],
  "latency_ms": 1840.2,
  "model": "qwen2.5",
  "ollama_used": true
}
```

**Response (Ollama offline — graceful fallback):**
```json
{
  "content": "Parking Buddy is running in offline mode. (Ollama unavailable — stub response)",
  "tool_calls": [],
  "tool_results": [],
  "latency_ms": 0.3,
  "model": "qwen2.5",
  "ollama_used": false
}
```

---

## Getting Help

- **Issues**: [github.com/Abinav-karthikeyan/Biike_bro/issues](https://github.com/Abinav-karthikeyan/Biike_bro/issues)
- **Roadmap**: See `roadmap_new.md` for full phase plan with exit criteria
- **Governance**: See `GOVERNANCE.md` for a transparent gap list

---

## License

Proprietary — internal development prototype.
