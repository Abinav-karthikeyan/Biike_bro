# Bike Parking Buddy — System Architecture

> Comprehensive design documentation for the predictive bike-parking assistant system.

**Version**: v1.0-prototype
**Last Updated**: August 2026

---

## Executive Overview

Bike Parking Buddy is a **Small Language Model (SLM)-powered predictive intelligence system** designed to help riders find available dockless bike parking in Glasgow. The system combines:

1. **Statistical ML** (XGBoost) for zone-fill probability prediction
2. **Vector Search** (HNSW) for semantic zone similarity and historical pattern matching
3. **Language-driven Tool-Calling** (Qwen2.5 via Ollama) for conversational prediction and natural-language reasoning

All processing runs **on-device** — no cloud inference, no API latency penalties.

---

## System Layers

### Layer 1: Data Ingestion & Storage

```
┌─────────────────────────────────────────────────────────────┐
│  GBFS v2.3 Operator Feeds (Lime, Voi, Tier, etc.)         │
│  • station_status (real-time occupancy)                    │
│  • station_information (zone metadata)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    5-minute polling
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Ingest Pipeline (backend/data/gbfs_ingest.py)            │
│  • Auto-discovery manifest parsing                         │
│  • Feed validation & schema normalization                  │
│  • Upsert to zone_snapshots table                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    Persisted in:
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  V1: DuckDB (in-memory, dev/test)                         │
│  V2: PostgreSQL 15 + Cloud SQL (production)               │
│  • zone_metadata       (60 zones, semi-static)             │
│  • zone_snapshots      (725k+ time-series)                │
│  • rider_outcomes      (telemetry)                         │
│  • zone_embeddings_hnsw (256-D vectors)                   │
│  • model_artifacts     (versioned XGBoost)                │
└─────────────────────────────────────────────────────────────┘
```

**Key Decisions:**
- **Why GBFS?** Standard protocol across micro-mobility operators (Lime, Voi, Tier, Jump). Reduces vendor lock-in.
- **Why 5-min polling?** Operator rate limits + occupancy change granularity trade-off. Real-time ingest via webhooks is V2 enhancement.
- **Why async ingest?** Non-blocking I/O for concurrent feed polling; FastAPI lifespan background task.

---

### Layer 2: Feature Computation & Embeddings

#### 2A. XGBoost Feature Pipeline

```
zone_snapshots
    │
    ├─ Temporal features    → hour_of_day, day_of_week, is_rush_hour, is_weekend
    ├─ Weather features     → weather_code, temperature, rain, wind_speed
    ├─ Zone features        → capacity, venue_type, transit_score
    ├─ Demand features      → recent_turnover (bikes taken/returned in last 30 min)
    │                          event_proximity (local events nearby)
    │
    ▼
XGBoost Classifier (backend/services/prediction.py)
    • Input: 11 engineered features
    • Output: fill_probability (logits), confidence interval
    • Model: Trained on 725k+ synthetic snapshots, ~80% accuracy
    • Latency: <20ms inference
```

**Model Training Pipeline (V2 retraining):**
1. Sample 80/20 train/test split (stratified by venue_type, time-of-day)
2. Fit XGBoost with early stopping on validation AUC
3. Persist binary model artifact (joblib) to `models/xgb_model.joblib`
4. Log performance metrics (AUC, precision, recall, feature importance) to telemetry

#### 2B. Zone Embeddings & HNSW Indexing

```
zone_metadata
    │
    ├─ lat, lon             (geolocation)
    ├─ venue_type           (one-hot)
    ├─ transit_score        (normalized 0-100)
    ├─ capacity             (bikes)
    ├─ turnover_avg         (bikes/hour, 30-day rolling)
    │
    ▼
Zone Embedding (256-D)
    │
    └─ Indexed in HNSW
       • M=16 (fanout for graph)
       • ef_construction=200 (build precision)
       • ef_search=50 (query precision)
       • Latency: <10ms per query
       • Index persistence: data/hnsw_indices/zones.bin
```

**Why HNSW?** Approximate nearest-neighbor search trades 1-2% recall loss for 100x speedup vs. exact kNN. Critical for <200ms SLM latency SLO.

**Context Index (128-D):** Historical patterns per day-of-week + hour + venue_type. Enables retrieval of "typical occupancy evolution" for cold zones.

---

### Layer 3: SLM Tool-Calling & Reasoning

```
User Query
    │
    "Will GLW_Z008 be full in 20 minutes?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Qwen2.5-0.5B (Ollama, on-device)                         │
│  • Receives query + tool schemas + context                 │
│  • Decides which tools to invoke                           │
│  • Tool-calling compatible w/ OpenAI format               │
└──────────────────────────┬──────────────────────────────────┘
                           │
            Invokes one or more tools:
            │
            ├─ get_zone_forecast(zone_id, time_horizon_mins)
            │    └─▶ XGBoost inference → fill_probability, confidence
            │
            ├─ zone_semantic_search(zone_id, k=3)
            │    └─▶ HNSW kNN search → similar zones + occupancy
            │
            └─ log_outcome(query, prediction, feedback)
                 └─▶ Telemetry upsert (opt-in)
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Qwen2.5 Second Pass (Narrative Generation)               │
│  • Synthesizes tool results                                │
│  • Generates conversational response                       │
│  • May suggest alternatives or caveats                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
Response to User:
"GLW_Z008 is predicted 72% full (moderate confidence).
Try GLW_Z012 instead — 42% full, 350m away, better weather exposure."
```

**Latency Budget (V2 SLO):**
- HNSW zone search: <10ms
- XGBoost prediction: <20ms
- Qwen2.5 tool-calling + generation: <1800ms (Ollama on GPU)
- Total: <2s end-to-end

**Fallback:** If Ollama unavailable, return XGBoost-only stub (prediction + top-3 alternatives).

---

### Layer 4: REST API & Query Routing

```
FastAPI Application (backend/main.py)
    │
    ├─ CORS middleware       → Allow dashboard, local dev
    ├─ Lifespan manager      → Start GBFS ingest, load HNSW indices
    ├─ Error handling        → Standardized 4xx/5xx responses
    │
    └─ Routers:
       │
       ├─ health.py          → GET /health/, /health/ready
       │
       ├─ zones.py           → GET /zones/, /zones/{id}, /zones/{id}/snapshots
       │                        GET /zones/{id}/occupancy-history (7-day chart)
       │
       ├─ predict.py         → POST /predict/ (XGBoost-only)
       │
       ├─ slm.py             → POST /slm/query, /slm/predict
       │                        GET /slm/status, /slm/tools
       │
       ├─ rides.py           → GET /rides/, /rides/stats
       ├─ weather.py         → GET /weather/
       │
       └─ analytics.py       → GET /analytics/summary, /zone-heatmap,
                               /model-metrics, /events
```

**Request Pipeline:**
1. Validate request schema (Pydantic models in `backend/models/schemas.py`)
2. Fetch context from DuckDB/PostgreSQL
3. Invoke prediction service (XGBoost or SLM)
4. Serialize response with timings metadata
5. Log to telemetry (if opt-in enabled)

---

## Data Flow Examples

### Example 1: Prediction Query

```
POST /slm/query
Body: { "query": "Where can I park near the Clyde?" }
│
▼
1. Extract context from zone_snapshots (last 1h)
2. Embed query (Qwen2.5 embeddings)
3. Search HNSW context index → historical patterns
4. Call Qwen2.5 with tools:
   - Suggest zones near Clyde (geographic filter)
   - Invoke get_zone_forecast(zone_id) for each
   - Invoke zone_semantic_search() for alternatives
5. Qwen2.5 synthesizes response
│
▼
Response:
{
  "prediction": "Clyde zone GLW_Z042 is 55% full, improving after 8pm.",
  "alternatives": [
    { "zone_id": "GLW_Z041", "occupancy_pct": 40, "distance_m": 250 }
  ],
  "confidence": 0.78,
  "reasoning": "Clear trend based on 7-day occupancy pattern."
}
```

### Example 2: Zone Occupancy Snapshot

```
GET /zones/GLW_Z008/snapshots?hours=24
│
▼
1. Query zone_snapshots WHERE zone_id='GLW_Z008'
           AND timestamp >= now() - interval '24 hours'
2. Sort by timestamp ascending
3. Aggregate every 15 minutes (chart downsampling)
4. Include weather, event, demand context
│
▼
Response:
[
  {
    "timestamp": "2026-08-09T10:00:00Z",
    "occupancy_pct": 65,
    "available_bikes": 21,
    "weather": { "code": "partly_cloudy", "temp_c": 18 },
    "event_proximity": null
  },
  ...
]
```

---

## Deployment Architecture

### V1 (Current)

```
┌─────────────────────────────────────────────────────────────┐
│  Developer Laptop / Docker Compose                        │
│                                                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ FastAPI Backend  │  │   DuckDB     │  │  Ollama Qwen │  │
│  │ (Python 3.11+)   │  │  (in-memory) │  │  (7B local)  │  │
│  │                  │  │              │  │              │  │
│  │ • Prediction.py  │  │ 725k+        │  │ GPU optional │  │
│  │ • HNSW Search    │  │ snapshots    │  │              │  │
│  │ • Tool-calling   │  │              │  │ Port 11434   │  │
│  └──────────────────┘  └──────────────┘  └──────────────┘  │
│         ▲                     ▲                   ▲          │
│         └─────────────────────┼───────────────────┘          │
│                         Lifespan tasks                       │
└─────────────────────────────────────────────────────────────┘
         │
         ├─ API Docs: http://localhost:8000/docs
         └─ Dashboard: http://localhost:5500/dashboard/index.html
```

**Deployment Command:**
```bash
docker-compose up  # PostgreSQL 15 + Redis 7 (template ready)
# or
python run.py      # Dev server (Ollama + DuckDB)
```

### V2 (Planned)

```
┌─────────────────────────────────────────────────────────────┐
│  Google Cloud Platform (GKE Autopilot)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FastAPI Workload (3 replicas, CPU-optimized nodes)  │  │
│  │ • Cloud Run (auto-scaling)                          │  │
│  │ • Istio for traffic shaping                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Cloud SQL   │  │   Cloud      │  │ GPU Workload     │  │
│  │ PostgreSQL15 │  │ Memorystore  │  │ (Ollama + K8s)   │  │
│  │              │  │  (Redis)     │  │ • Job scheduling │  │
│  │ 100GB+ daily │  │              │  │ • Cost-optimized │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Cloud Composer (Airflow) — 6 DAGs                   │  │
│  │ • GBFS ingest (5-min polling)                       │  │
│  │ • Embedding generation (hourly)                     │  │
│  │ • HNSW rebuild (nightly)                            │  │
│  │ • XGBoost retraining (weekly)                        │  │
│  │ • Data quality checks (6-hourly)                    │  │
│  │ • Model evaluation (daily)                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Monitoring & Observability                          │  │
│  │ • Cloud Logging (structured, searchable)            │  │
│  │ • Cloud Monitoring (metrics, SLOs, alerting)        │  │
│  │ • Cloud Trace (distributed tracing)                 │  │
│  │ • Model Card registry (model metadata, lineage)     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Deployment Tool:** Terraform + Cloud Build (CI/CD).

---

## Key Design Decisions & Rationale

| Decision | Rationale | Trade-Off |
|----------|-----------|-----------|
| **XGBoost** for fill prediction | Fast inference (<20ms), interpretable feature importance, handles non-linear interactions | Requires labeled training data; no sequence modeling |
| **HNSW** for zone search | 100x faster than exact kNN, <10ms latency | ~1-2% recall loss (acceptable for recommendations) |
| **Ollama + Qwen2.5** for SLM | No cloud dependency, zero inference latency penalties, on-device privacy | Requires GPU; 7B model ~14GB VRAM |
| **Tool-calling** over fine-tuning | Generalization to unseen queries, decoupled reasoning from knowledge | Requires well-designed tool schema + validation |
| **DuckDB→PostgreSQL migration** | DuckDB fine for dev/test; PostgreSQL essential for concurrent writes, backups, audit trails | Schema migration effort; operational complexity |
| **5-min GBFS polling** | Operator rate limits, balances freshness vs. cost | Not real-time; webhooks are V2 enhancement |
| **GBFS over proprietary APIs** | Standardized, operator-agnostic, reduces vendor lock-in | Delayed data (5-min latency), feed quality variability |

---

## Error Handling & Resilience

### Circuit Breaker (HNSW Index)

If HNSW indices corrupt or missing on startup:
1. Log warning
2. Rebuild indices from zone_metadata on-the-fly
3. Continue serving with slightly degraded search quality (~50ms instead of <10ms)

### Ollama Fallback

If Ollama unavailable (unreachable, OOM, crash):
1. Catch connection error in `slm_service.py`
2. Return XGBoost-only response (fill_probability, top-3 alternatives)
3. Log telemetry event for alerting
4. Health check endpoint reports `slm_status: unavailable`

### Prediction Confidence Intervals

XGBoost returns both point estimate and confidence:
- High confidence (>0.75): "Strongly recommend"
- Medium (0.6-0.75): "Likely, but check live occupancy"
- Low (<0.6): "Uncertain; recommend backup zone"

---

## Testing Strategy

### Unit Tests (`tests/test_all.py`)

- 31 tests covering:
  - DuckDB schema & upsert logic
  - XGBoost model loading & inference
  - HNSW index search (exact kNN verification)
  - SLM tool schemas & validation
  - API endpoint 4xx/5xx error handling
  - Pydantic model serialization

### Integration Tests (V2 TODO)

- End-to-end SLM query → prediction flow
- Ollama connectivity + fallback
- GBFS ingest pipeline (mock feeds)
- Database transaction rollback on error

### Load Testing (V2 TODO)

- 1000 concurrent prediction requests
- SLM query latency under sustained load
- Database connection pool exhaustion recovery

---

## Metrics & Observability

### Application Metrics

| Metric | Threshold | Alert |
|--------|-----------|-------|
| `/slm/query` latency (p95) | <2s | >3s |
| XGBoost inference latency | <20ms | >50ms |
| HNSW index search latency | <10ms | >25ms |
| Ollama availability | >95% | <90% uptime |
| Zone snapshot ingestion lag | <5 min | >10 min |

### Model Metrics

| Metric | Target | Tracking |
|--------|--------|----------|
| Fill prediction accuracy | ≥78% | Daily evaluation on holdout set |
| Feature importance drift | <10% shift | Weekly comparison to baseline |
| Prediction calibration | Predicted vs. actual occupancy within 5% | Per-zone bucketed analysis |

### Telemetry (Opt-In)

When enabled, log:
- Query text + prediction result
- User feedback (binary: correct/incorrect)
- Zone visited by rider (if provided)
- Feature values at prediction time (for retraining)

---

## Security Considerations

### Data Privacy

- No PII collection (rider-specific: only zone visited, timestamp)
- All computation on-device (no third-party inference APIs)
- CORS restricted to localhost + dashboard origin
- Database credentials in `.env`, never committed to Git

### API Security

- FastAPI CORS middleware restricts origins
- Request validation via Pydantic (rejects malformed payloads)
- No SQL injection (ORM + parameterized queries)
- Rate limiting planned for V2 (Redis + APScheduler)

### Model Security

- XGBoost artifact signed (hash verified on load)
- HNSW indices validated on startup
- SLM tool schemas whitelisted (no dynamic tool creation)

---

## Future Enhancements (V2 & Beyond)

### Short-term (V2)

- [ ] Real-time webhook ingest (Lime/Voi/Tier partnerships)
- [ ] PostgreSQL migration + Cloud SQL
- [ ] GKE deployment + Cloud Composer Airflow DAGs
- [ ] Distributed tracing (OpenTelemetry)
- [ ] API rate limiting + auth token validation
- [ ] Multi-language SLM support (French, Mandarin)

### Medium-term

- [ ] Demand forecasting (rider volume prediction)
- [ ] Multi-modal embeddings (images of zones, rider photos)
- [ ] Reinforcement learning (retraining from rider feedback loops)
- [ ] Live occupancy integration (bike sensor networks, computer vision)

### Long-term

- [ ] Autonomous bike redistribution (fleet optimization)
- [ ] Predictive maintenance (bike fault prediction)
- [ ] Micro-mobility market dynamics modeling

---

## Getting Started for Contributors

1. **Read:** This document → architecture overview
2. **Setup:** See `README.md` → Quick Start section
3. **Explore:** `backend/services/prediction.py` → XGBoost logic
4. **Explore:** `backend/services/slm_service.py` → Tool-calling logic
5. **Run tests:** `pytest tests/ -v`
6. **Run API:** `python run.py` then open `http://localhost:8000/docs`
7. **Iterate:** Modify features, retrain, compare metrics

---

## References

- **Bike Parking Buddy PRD:** `Parking_Buddy_PRD.docx` (8 sections, success metrics)
- **V2 Implementation Plan:** `plan_v2/V2_IMPLEMENTATION_PLAN.md` (5 phases, 6-week timeline)
- **Agent Context:** `AGENTS.md` (for AI agents & new developers)
- **GBFS Spec:** https://gbfs.mobilitydata.org/
- **HNSW Paper:** Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search with Hierarchical Navigable Small World Graphs" (IEEE TPAMI 2018)
- **XGBoost Docs:** https://xgboost.readthedocs.io/

---

## Contact & Support

- **Issues:** Report on GitHub or internal tracker
- **Architecture Questions:** See AGENTS.md for context passing to Claude
- **Deployment Support:** See plan_v2/ for detailed step-by-step

---

**End of ARCHITECTURE.md**
