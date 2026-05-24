# Bike Parking Buddy

> On-device SLM-powered predictive parking intelligence for dockless bike-sharing in Glasgow.

**Version**: v1.0-prototype — synthetic data, DuckDB-backed, Ollama/Qwen2.5, fine-tuning pipeline archived for V2.

---

## Project Structure

```
Cycle_Buddy/
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
├── qwen/                        # Qwen2.5 model integration
│   ├── ollama_qwen.py           # Minimal Ollama chat test
│   ├── qwen_demo.py             # Transformers inference demo
│   └── finetune/                # Fine-tuning pipeline (archived for V2)
│
├── synthetic_seed/              # Synthetic Glasgow dataset (60 zones, 725k+ snapshots, 8k rides)
│   ├── generate_synthetic_data.py
│   ├── zones.csv, bikes.csv, rides.csv, weather.csv, etc.
│   └── geofencing_zones.json, free_bike_status.json
│
├── tests/
│   └── test_all.py             # 31 tests — DuckDB, XGBoost, API, SLM tools
│
├── data/                        # Runtime data (git-kept, populated at runtime)
│   ├── hnsw_indices/            # HNSW index binaries
│   └── finetune/                # Training data JSONL
│
├── models/                      # Model artifacts (git-kept)
│   └── parking_buddy_qwen/      # LoRA adapter / GGUF exports
│
├── plan_v2/                     # V2 implementation plan (agentic RAG, GKE, Airflow)
├── docker-compose.yml           # PostgreSQL 15 + Redis 7 + API
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
git clone <repo-url> && cd Cycle_Buddy
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
- **Zone index**: 256-D embeddings `[lat, lon, venue_type, transit_score, turnover]`
- **Context index**: 128-D day-context embeddings for historical pattern matching
- **Search cost**: <10ms per query (M=16, ef=50)
- **Persistence**: Save/load from `data/hnsw_indices/`; nightly rebuild in production

### SLM + Tool-Calling (`backend/services/slm_service.py` + `slm_tools.py`)
- **Model**: Qwen2.5 (7B, Q4_K_M via Ollama)
- **Tools**: 3 OpenAI-compatible function schemas
  - `zone_semantic_search` — similarity search via HNSW
  - `get_zone_forecast` — XGBoost prediction
  - `log_outcome` — telemetry recording
- **Fallback**: If Ollama unreachable, returns XGBoost-only stub response
- **Latency SLO** (V2 target): <200ms for structured prediction, <2s for SLM narrative

### Data Connectors (`backend/data/gbfs_ingest.py`)
- **GBFS v2.3 client**: Auto-discovery manifest parsing, station_status feed
- **Ingest**: Async polling designed for 5-min intervals (APScheduler / Airflow)
- **Enrichment** (V2 target): Weather + local events cross-reference
- **Operators**: Target Lime, Voi, Tier (any GBFS-compatible operator)

---

## Database Schema (PRD §7)

| Table | Purpose | Status |
|-------|---------|--------|
| `zone_metadata` | Semi-static zone reference data | SQLAlchemy model defined |
| `zone_snapshots` | 5-min occupancy time-series | SQLAlchemy model defined |
| `model_artifacts` | Versioned ML model registry | SQLAlchemy model defined |
| `rider_outcomes` | Opt-in telemetry for retraining | SQLAlchemy model defined |
| `zone_embeddings_hnsw` | Pre-computed 256-D zone embeddings | SQLAlchemy model defined |

**Current**: V1 uses DuckDB in-memory. PostgreSQL migration is V2 target.
**Migration**: Run `alembic upgrade head` once PostgreSQL is configured.

---

## V2 — What's Next

V2 pivots from fine-tuning to **agentic RAG + tool-calling**, with deployment on **GKE + Airflow**. See [`plan_v2/V2_IMPLEMENTATION_PLAN.md`](plan_v2/V2_IMPLEMENTATION_PLAN.md) for full detail.

### Key V2 Changes
- **Drop fine-tuning**: Archived in `archive/finetune_v1/`. The SLM's job is narrative, not prediction.
- **Add RAG context**: Historical patterns from HNSW injected into every SLM prompt.
- **Wire HNSW**: Zone semantic search connected to real HNSW indices.
- **PostgreSQL production**: Migrate from DuckDB to Cloud SQL.
- **GKE deployment**: Autopilot cluster + GPU pods for Ollama.
- **Airflow DAGs**: 6 orchestrated pipelines (GBFS ingest, embeddings, HNSW rebuild, model retrain, quality checks).

### V2 Phase Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Foundation | Week 1-2 | Terraform, CI/CD, PostgreSQL migration |
| 2. Data Pipeline | Week 2-3 | GBFS upsert, Airflow DAGs, embedding generation |
| 3. Core Services | Week 3-4 | HNSW wiring, RAG context, tool-call hardening |
| 4. Deployment | Week 4-5 | GKE deployment, Cloud Composer, monitoring |
| 5. Polish | Week 5-6 | Load testing, prompt tuning, archive fine-tuning |

---

## Success Metrics (PRD §3)

- **Prediction accuracy**: ≥78% fill classification (30-min lookahead)
- **API latency**: <200ms structured prediction
- **SLM latency**: <2s end-to-end narrative (V2 target)
- **Wasted minutes**: 25–35% fewer rider redirects

---

## Getting Help

- **Issues**: Report at [opencode issue tracker](https://github.com/anomalyco/opencode/issues)
- **V2 Plan**: See `plan_v2/V2_IMPLEMENTATION_PLAN.md` for detailed step-by-step

---

## License

Proprietary — internal development prototype.
