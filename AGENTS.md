# Bike Parking Buddy — Agent Context

## Project Identity
- **Product**: Predictive parking intelligence for dockless bike-sharing
- **Version**: v1.0-prototype (tagged)
- **City**: Glasgow (synthetic data — 60 zones, 725k+ snapshots, 8k rides)
- **Remote**: `https://github.com/Abinav-karthikeyan/Biike_bro.git`

## V1 Architecture (committed baseline)

### Tech Stack
- **API**: FastAPI + Uvicorn (Python 3.11)
- **Data**: DuckDB in-memory (dev) / PostgreSQL + SQLAlchemy (models defined, target for prod)
- **ML**: XGBoost classifier (binary zone fill), scikit-learn, numpy, pandas
- **Vector**: hnswlib (256-D zone index + 128-D context index)
- **SLM**: Qwen2.5 (7B Q4_K_M via Ollama) with OpenAI-compatible tool-calling
- **Cache**: Redis 7 (Memorystore target)
- **Orchestration**: Docker Compose (local) / GKE + Cloud Composer (V2 target)
- **Dashboard**: Vanilla JS + Leaflet maps (5-panel dark UI)

### Key Files

| File | What it does |
|------|-------------|
| `backend/main.py` | App factory, lifespan (DuckDB init, XGBoost train, SLM connect) |
| `backend/config.py` | All env vars via Pydantic Settings |
| `backend/data/duckdb_store.py` | DuckDB queries — zones, snapshots, rides, weather, analytics, ML training data |
| `backend/services/prediction.py` | XGBoost train/predict — 11 features, alternatives, human-readable reasons |
| `backend/services/hnsw_search.py` | HNSW indices — zone semantic search, historical pattern matching, intent search |
| `backend/services/slm_service.py` | Ollama Qwen2.5 client — two-pass tool-calling, fallback to stub |
| `backend/services/slm_tools.py` | 3 tool schemas — `zone_semantic_search`, `get_zone_forecast`, `log_outcome` |
| `backend/routers/slm.py` | SLM endpoints — query, predict (hybrid), status, tool schemas |
| `backend/models/schemas.py` | All Pydantic request/response models |
| `backend/models/db_models.py` | SQLAlchemy ORM — 5 tables (Postgres target) |
| `synthetic_seed/generate_synthetic_data.py` | Full Glasgow synthetic dataset generator |
| `qwen/finetune/finetune_unsloth.py` | Unsloth QLoRA fine-tuning (ARCHIVED — not used in V2) |

## V2 Plan (next phase)

**What changed from V1 to V2:**
- **Dropped**: Fine-tuning pipeline (moved to archive) — SLM's job is narrative, not prediction
- **Added**: Agentic RAG context injected into every SLM prompt (historical patterns from HNSX + weather + events)
- **Wired**: HNSW service into tool dispatch (was `None` in V1)
- **Migrated**: DuckDB → PostgreSQL (Cloud SQL) for production
- **Deployed**: GKE Autopilot + Cloud Composer (Airflow) + Cloud Monitoring

See `plan_v2/V2_IMPLEMENTATION_PLAN.md` for step-by-step.

## Common Commands

```bash
# Run API (dev)
python run.py
# or: uvicorn backend.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Install deps
pip install -r backend/requirements.txt

# Start infra
docker-compose up postgres redis -d
```

## V2 Deployment Targets

- **GKE**: Autopilot cluster, europe-west2 (Glasgow)
- **Ollama pods**: 1× T4 GPU, nvidia-tesla-t4 node selector
- **Database**: Cloud SQL PostgreSQL 15
- **Cache**: Memorystore Redis 7 (1GB)
- **Orchestration**: Cloud Composer 2 (Airflow 2.x)
- **CI/CD**: Cloud Build or GitHub Actions
- **Infra-as-code**: Terraform in `infra/`

## Architecture Principles

1. **Hybrid intelligence**: XGBoost for fast/reliable scores, SLM for natural-language UX — never ask the SLM to predict occupancy
2. **Tool-calling over fine-tuning**: The SLM queries live structured data via tools; it doesn't memorize patterns
3. **Fresh data always**: Real-time GBFS ingestion + periodic model retraining, not static training datasets
4. **Graceful degradation**: Every component has a fallback — Ollama down → XGBoost only, prediction fails → stub, no HNSW → semantic search falls back to geo-distance
5. **Observability first**: All latency, accuracy, and throughput metrics exported to Cloud Monitoring
