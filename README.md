# Cycle Buddy

Predictive bike-parking intelligence for dockless bike-sharing in Glasgow.
XGBoost fill-probability forecasting + HNSW semantic zone search + Groq LLM narration, served as a FastAPI backend on Render.com.

**Synthetic data prototype** — 60 zones, 725k+ snapshots generated from realistic Glasgow distributions.

[![CI](https://github.com/Abinav-karthikeyan/Biike_bro/actions/workflows/ci.yml/badge.svg)](https://github.com/Abinav-karthikeyan/Biike_bro/actions/workflows/ci.yml)

---

## Quick start

```bash
git clone https://github.com/Abinav-karthikeyan/Biike_bro && cd Biike_bro
python -m venv venv && venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env          # set GROQ_API_KEY for LLM narration (optional)
uvicorn backend.main:app --reload --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

```bash
pytest tests/ -v              # 103 tests
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/` | Liveness |
| POST | `/predict/` | Zone fill-probability (XGBoost) |
| GET | `/recommend/` | Retrieval-first alternatives (HNSW → XGBoost, no LLM) |
| GET | `/zones/` | Zone list with filters |
| GET | `/zones/{id}` | Zone detail + latest occupancy |
| GET | `/analytics/summary` | KPI dashboard data |
| POST | `/slm/query` | Natural-language query (Groq/Llama + tool-calling) |
| POST | `/slm/predict` | XGBoost prediction + LLM narrative |
| GET | `/slm/status` | Groq connectivity and model info |

Full endpoint list at `/docs` once running.

---

## Deployment

Hosted on Render.com (free tier). Config in [`render.yaml`](render.yaml).

Required secret: `GROQ_API_KEY` — set in Render dashboard → Environment.
Everything else is wired via env defaults in `render.yaml`.

---

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full component breakdown, data pipeline, ML details, and database schema.
