# Governance — Bike Parking Buddy

> This is a WIP personal project; production governance is deliberately out of scope for V1 — see `roadmap_new.md` for sequencing.

This document names every governance gap honestly so that an interviewer or collaborator
can assess the project's maturity without being surprised by reading the code.

---

## What's in place

| Area | Status |
|------|--------|
| Test suite | 41 pytest tests covering data layer, ML service, API contract, tool dispatcher |
| Graceful degradation | XGBoost-only fallback when Ollama is unreachable |
| Synthetic data labelling | README, CV copy, and docs all say "synthetic" explicitly |
| HNSW index rebuild | Deterministic: `scripts/build_hnsw_index.py` — same input → same index |
| Rider outcome logging | `log_rider_outcome` in DuckDB (in-memory, not persisted across restarts) |

---

## Known gaps — deferred to V2 phases

### Model versioning
`model_artifacts` SQLAlchemy table is defined in `db_models.py` but not used.
XGBoost model retrained fresh on each startup from seed CSVs; no artifact registry.
**Risk:** can't roll back to a prior model if accuracy regresses.
**Planned fix:** Phase 2 (Persistence) — SQLAlchemy ORM write path + Alembic migrations.

### Prompt versioning
System prompts live inline in `slm_service.py::_build_system_prompt`.
No changelog, no rollback, no A/B test harness.
**Risk:** prompt tweaks can silently change tool-calling behaviour.
**Planned fix:** Phase 5 (Observability) — snapshot tests + eval harness.

### Data lineage
No record linking which snapshot rows trained which model version.
Training data is deterministic (fixed seed CSVs) so this is low-risk for V1.
**Planned fix:** Phase 2 — `model_artifacts` table tracks training dataset hash.

### Audit trail on SLM outputs
No `slm_query_log` table — every query/response is ephemeral.
Makes prompt debugging and retraining hard.
**Planned fix:** Phase 5 — log query + response + token count + latency per request.

### PII / consent policy for `log_outcome` telemetry
`log_outcome` tool records zone_id + timestamp + availability + satisfaction.
No consent flow, no data retention policy, no anonymisation.
**Current mitigation:** telemetry is opt-in (rider must call the tool explicitly);
data is in-memory only and lost on restart.
**Planned fix:** Phase 3 (GBFS ingest) — add `zone_source` column + explicit consent flag.

### Documented failure modes
Known SLM failure modes not yet tested in the eval harness:
- SLM hallucinates a `zone_id` that doesn't exist in the index (HNSW returns empty).
- SLM calls `get_zone_forecast` with a negative `time_horizon_mins`.
- Ollama returns malformed JSON in `tool_calls`.
All three are handled defensively in `slm_service.py::_dispatch`, but not test-covered.
**Planned fix:** Phase 5 — 20–30 adversarial queries in the eval harness.

### Bias / fairness
Zone fill predictions are trained on synthetic data with uniform coverage across all 60
Glasgow zones. Under-sampled real-world neighbourhoods (e.g. lower-income areas with fewer
GBFS operator deployments) could have higher prediction error if real data were used.
Not yet analysed.
**Planned fix:** Phase 5 — precision/recall breakdown by neighbourhood in `/analytics/model-metrics`.

### Postgres migration
`db_models.py` defines 5 SQLAlchemy ORM models and `docker-compose.yml` provisions a
Postgres container, but the application currently reads/writes DuckDB in-memory only.
`alembic upgrade head` is not yet wired.
**Planned fix:** Phase 2 — DuckDB → SQLAlchemy ORM writes → Alembic → Postgres.
