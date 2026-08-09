# Biike_bro — Rework Roadmap

> Working document for planning the path from V1 prototype to a defensible V2.
> Assessment baseline: 41/41 tests passing, XGBoost + FastAPI + DuckDB + SLM tool-calling loop functional; HNSW, GBFS persistence, Postgres, and deployment all stubbed or missing.

---

## Guiding principles

Before the task list, three principles to sanity-check every change against:

1. **Close the honesty gap first.** The README currently claims more than the code delivers (HNSW wired, `plan_v2/` exists, fine-tuning archived, Postgres migration ready). Either ship the claim or downgrade the wording. Every stub that stays should be labelled as such in the README.
2. **Every new dependency earns its place.** Airflow, GKE, Cloud SQL, Terraform, Memorystore — the AGENTS.md V2 stack is enterprise-shaped and probably 10x what a solo prototype needs. Ask "does this project actually need this, or does it sound impressive?" before adopting. A boring, small V2 (single VPS + Postgres + cron) is often the more honest engineering answer.
3. **Keep the test suite green through every change.** 41 passing tests is your safety net. Any refactor that breaks them without a plan to restore is a regression, not progress.

---

## Phase 0 — Housekeeping (½–1 day)

Small, high-leverage cleanup to stabilise the base before real work.

- [ ] Fix README directory name (`Cycle_Buddy/` → `Biike_bro/`).
- [ ] Either create `plan_v2/V2_IMPLEMENTATION_PLAN.md` or remove the reference. Same for `archive/finetune_v1/`.
- [ ] Move `qwen/finetune/` to `archive/finetune_v1/` since the plan is to drop fine-tuning. Keeps the main tree honest.
- [ ] Add `.env.example` at the repo root (README references it).
- [ ] Replace deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)` — pytest warned about this.
- [ ] Add ruff + black to `pyproject.toml`; run once and commit the formatting churn as a single "style" commit.
- [ ] Add a GitHub Actions workflow that runs `pytest tests/ -v` on push. Cheap credibility signal for anyone browsing the repo.
- [ ] Pin dependency versions in `requirements.txt` (currently uses `>=`, which will drift).

**Exit criteria:** CI badge green in README, no broken internal links, no dead code paths in the main tree.

---

## Phase 1 — Wire HNSW for real (2–3 days)

**Why first:** the SLM's "semantic zone alternatives" story is a headline feature. Right now the tool returns `zone-stub-001`. This is the single biggest gap between the pitch and the code.

- [ ] Decide the zone embedding: 256-D from `[lat, lon, venue_type_one_hot, transit_score, capacity, turnover_rate, hour_profile...]`. Document the choice in `hnsw_search.py` — future-you will thank present-you.
- [ ] Build a `scripts/build_hnsw_index.py` that reads the synthetic zones from DuckDB and writes `data/hnsw_indices/zone_index.bin` + a sidecar JSON with the `zone_id_map`.
- [ ] Remove `app.state.hnsw_service = None` in `main.py`; instantiate the service and let it load the built index.
- [ ] Wire the `zone_semantic_search` tool dispatcher to actually call `HNSWSearchService` instead of falling through to the stub in `slm_tools.py`.
- [ ] Attach real `occupancy_profile` to results by joining against the latest snapshots in DuckDB (currently a `TODO` in `hnsw_search.py:88`).
- [ ] Add tests: index builds deterministically, top-k results are stable, filter-by-venue-type works, distance filter works.

**Exit criteria:** an end-to-end SLM query for "somewhere similar to GLW_Z008" returns real Glasgow zones, not stubs. Test suite still green.

---

## Phase 2 — Persistence: DuckDB → SQLAlchemy → Postgres (3–5 days)

**Why now:** without persistence, `log_outcome` throws data away, GBFS can't ingest anything, and the "5 SQLAlchemy models defined" claim is technically true but functionally meaningless. Also un-blocks Phase 3.

Sequence matters here — don't jump straight to Postgres.

- [ ] **Step 1:** Point SQLAlchemy at DuckDB using `duckdb_engine`. This exercises the ORM path against the existing data with zero infra changes. Every route that reads via raw DuckDB SQL keeps working; you're just enabling ORM writes.
- [ ] **Step 2:** Rewrite `log_outcome` to actually insert into `rider_outcomes` via SQLAlchemy. Add a test that confirms the row lands.
- [ ] **Step 3:** Add Alembic. Generate the initial migration from `db_models.py`. Verify it runs cleanly against DuckDB.
- [ ] **Step 4:** Add a `DATABASE_URL` config; parameterise the engine so `sqlite:///...`, `duckdb:///...`, and `postgresql://...` all work.
- [ ] **Step 5:** Bring up the Postgres container from `docker-compose.yml`; run `alembic upgrade head`; run the test suite against Postgres. Fix what breaks.
- [ ] **Step 6:** Write a `scripts/seed_postgres.py` that loads the synthetic CSVs into Postgres so the app has data outside of DuckDB.

**Exit criteria:** test suite passes against both DuckDB and Postgres (parameterised via env var). `rider_outcomes` actually gets rows written.

---

## Phase 3 — GBFS live ingestion (2–4 days)

**Why now:** you have persistence, so ingested snapshots have somewhere to go. Also the "fresh data always" principle in `AGENTS.md` is aspirational until this ships.

- [ ] Pick one real GBFS operator to target first. Glasgow's actual bike-share (currently OVO Bikes / previously Nextbike) may or may not expose GBFS — check before committing. If not, fall back to a Lime/Voi/Tier feed from another city as a proof-of-concept and document the caveat.
- [ ] Fill in `GBFS_FEED_URLS` in config with the chosen operator.
- [ ] Replace the "Upsert stub" in `gbfs_ingest.py` with a real SQLAlchemy async upsert into `zone_snapshots`.
- [ ] Add a startup task (or a simple APScheduler job — full Airflow is overkill for one feed) that polls every 5 minutes.
- [ ] Add a `/health/ingest` endpoint that reports last successful poll timestamp and error count.
- [ ] Decide what to do when synthetic and real zones don't line up — probably a `zone_source` column so real and synthetic zones can coexist without confusion.

**Exit criteria:** the API can answer "what's happening in Zone X right now" using data less than 10 minutes old.

---

## Phase 4 — RAG context for the SLM (2–3 days)

**Why here:** the AGENTS.md V2 pivot says "Agentic RAG context injected into every SLM prompt." Once HNSW + persistence are live, the pieces exist. Doing this before Phase 1 would be building on stubs.

- [ ] Design a small context assembler: given a zone_id and query time, fetch (a) latest occupancy, (b) same-hour occupancy from the past 7 days, (c) 3 most similar zones via HNSW, (d) current weather, (e) nearby events in the next 2 hours.
- [ ] Inject that context as a system message before the SLM's first pass. Keep it under ~500 tokens — long contexts hurt Qwen2.5-7B latency more than they help.
- [ ] Add a `/slm/query?include_context=true` flag so you can A/B the answers with and without RAG context and see whether it actually improves them. If it doesn't, kill it — don't cargo-cult RAG.
- [ ] Add prompt-response snapshot tests (not accuracy tests — just "does the shape stay stable across changes").

**Exit criteria:** you can point to a concrete rider query where the RAG-augmented answer is measurably better than the tool-calling-only answer.

---

## Phase 5 — Observability + evaluation (2–3 days)

**Why:** you can't tune what you can't measure, and "78% accuracy on synthetic data" is a claim you shouldn't lean on for a live system.

- [ ] Add Prometheus metrics for: request latency per route, XGBoost predict latency, SLM tool-call latency, Ollama up/down state, cache hit rates.
- [ ] Log every SLM query + response to a table (`slm_query_log`) with token counts and latency. This is your dataset for future prompt-tuning.
- [ ] Build a tiny eval harness: 20–30 hand-written rider queries with expected tool-call sequences. Run on every SLM change. This catches regressions from prompt tweaks that unit tests miss.
- [ ] Add a `/analytics/model-metrics` extension that reports precision/recall broken down by hour-of-day and weather condition, not just aggregate accuracy. That's the interview-differentiating story.

**Exit criteria:** you can answer "is the system getting better?" with numbers, not vibes.

---

## Phase 6 — Deployment (open-ended)

**Read the room before starting this phase.** GKE Autopilot + Cloud Composer + Memorystore + Cloud SQL + Terraform is a **£300+/month baseline** for a personal project and takes weeks to set up correctly. The V2 plan as written in AGENTS.md optimises for looking enterprise, not for shipping.

Two paths — pick honestly based on your goals:

**Path A — Cheap and honest (recommended for a portfolio project):**
- [ ] Single small VPS (Hetzner, Fly.io, Railway — £5–20/month).
- [ ] Docker Compose with Postgres + Redis + API + Ollama on the same box.
- [ ] Caddy or Traefik for TLS.
- [ ] GitHub Actions deploys on push to `main`.
- [ ] Uptime monitoring via a free tier (BetterStack, UptimeRobot).
- **Total setup:** 2–3 days. **Total cost:** pocket change. **What you can say in interviews:** "I deployed it, it runs, here's the URL."

**Path B — GKE stack from AGENTS.md:**
- Only worth it if the target role explicitly needs GKE/Terraform/Airflow experience *and* you're prepared for the bill.
- If you go this route, do it as a separate `infra/` module *after* Path A is proven — never deploy something to GKE that hasn't first run reliably on a single box.

**Exit criteria:** there's a URL. Someone other than you can hit it. It stays up for a week without your intervention.

---

## Explicitly deferred / probably-don't-do

Things worth naming so they don't sneak back onto the list:

- **Fine-tuning Qwen2.5.** The V2 plan correctly drops this. Resist re-adding it unless you have a specific tool-calling failure mode that RAG + prompt engineering can't fix. Fine-tuning is expensive and rarely the right first move.
- **Custom embedding models.** hnswlib on hand-crafted feature vectors is fine for 60 zones. Don't reach for sentence-transformers or a bespoke encoder unless the feature-vector approach demonstrably fails.
- **Multi-city support.** The synthetic dataset is Glasgow-only and that's fine. Adding "we support any city" as a feature is scope creep until Glasgow works end-to-end.
- **A native mobile app.** The dashboard is enough to demo the system. A rider-facing app is a separate product, not V2 of this one.

---

## Rough timeline

Assuming ~8–10 focused hours per week:

| Phase | Focused weeks |
|-------|---------------|
| 0 — Housekeeping | 0.5 |
| 1 — HNSW wiring | 1 |
| 2 — Persistence | 1.5 |
| 3 — GBFS ingest | 1.5 |
| 4 — RAG context | 1 |
| 5 — Observability | 1 |
| 6 — Deployment (Path A) | 1 |
| **Total to a defensible V2** | **~7.5 weeks** |

Path B deployment adds 3–4 more weeks and real money. Budget accordingly.

---

## Decision log to keep

Start a `DECISIONS.md` alongside this roadmap. Every time you pick one thing over another (DuckDB vs Postgres timing, HNSW vs FAISS, APScheduler vs Airflow, VPS vs GKE) — write down the choice and the reasoning in 3–5 lines. In six months you'll have forgotten why, and in interviews this becomes the best material you have.
