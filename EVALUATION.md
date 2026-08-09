# Cycle Buddy — Honest Portfolio Evaluation

> Written: June 2026 | Scope: V1 Prototype

---

## TL;DR

This is a genuinely interesting project with a real problem statement, a thoughtful hybrid ML architecture, and solid prototype foundations. But right now it reads more like a well-planned project than a finished one. The gap between "what's described" and "what's actually wired together" is the single biggest risk to your portfolio impression. Close that gap — even partially — and this becomes a strong differentiator.

---

## What You Actually Built (V1)

| Component | Status | Honest Assessment |
|-----------|--------|-------------------|
| FastAPI backend (7 routers) | ✅ Done | Clean, fully async, well-structured |
| XGBoost fill prediction | ✅ Done | ~80% accuracy, proper feature engineering |
| DuckDB in-memory data layer | ✅ Done | 725k+ rows, fast queries, realistic synthetic data |
| SLM tool-calling framework | ⚠️ Partial | Correct architecture, tool *implementations* are stubs |
| HNSW vector search | ❌ Skeleton | Index structure defined, but zero embeddings generated |
| GBFS real-data ingest | ❌ Stub | `_upsert_snapshots()` is a TODO comment |
| PostgreSQL wiring | ❌ Unused | ORM models defined, never instantiated |
| Redis caching | ❌ Unused | Configured, never wired to any router |
| Prometheus / observability | ❌ Missing | structlog present but no metrics endpoint |
| Dashboard | ⚠️ Unknown | HTML files exist, not tested |

**31 pytest tests pass.** That's a real positive — most side projects skip tests entirely.

---

## The Core Idea: Is It Compelling?

**Yes, and here's why it works as a portfolio piece:**

- The problem is real. Dockless bike-sharing has a genuine "ghost zone" problem — riders arrive, zone is full, they circle. That's minutes wasted and churn.
- The hybrid intelligence model (XGBoost for fast prediction + SLM for human-readable narrative) is architecturally correct. Fine-tuning an SLM on a rapidly-changing rideshare dataset is the wrong call — you made the right call by keeping the SLM as a narrator and XGBoost as the predictor.
- Tool-calling over RAG is the right framing for this use case. A single-shot retrieval wouldn't know to ask "what's the weather right now" and "is there an event nearby" — tools compose that naturally.
- The decision to archive the fine-tuning pipeline and switch to tool-calling shows you course-corrected based on domain reasoning, not just following a tutorial.

**What weakens the pitch:**

- The SLM can call tools, but the tools return hardcoded fake data. If a recruiter or interviewer runs the app and asks "find me a nearby zone," they get a stub response. That's the demo-killer.
- HNSW is listed prominently in the README and repo name (Biike_bro), but the indices are empty — no embeddings have ever been generated. It's an architectural decision with zero implementation behind it.
- The V2 plan is detailed and impressive, but it also signals to a careful reader that most of the interesting parts aren't done yet.

---

## What a Hiring Manager Actually Sees

If a senior ML engineer or backend engineer reviews this repo for 20 minutes, here's what they'll notice:

**Positive signals:**
- You understand feature engineering (rush hours, WMO weather codes, venue types, haversine distance)
- You know when NOT to use a big model (prediction is XGBoost, not LLM)
- You've written 31 tests and they pass
- The code is clean, async, well-organized — not notebook-only
- You've thought about productionization (Pydantic settings, Docker, health endpoints, structlog)
- The V2 plan shows systems thinking: Airflow DAGs, GKE, Cloud SQL, Terraform

**Yellow flags:**
- HNSW is a headline feature with no embeddings
- Tool implementations are stubs — the AI part of "AI-powered" isn't actually executing
- Model trains from scratch on every startup (no persistence, no versioning)
- GBFS ingest can't pull real data — the whole "live" angle is synthetic-only

**What they WON'T say:** "nice idea" — because the idea IS nice. What they'll ask is: "Can you show me the SLM actually recommending a zone based on a real tool call?"

---

## Where You're Stuck (And Why)

The "stuck" feeling is almost certainly because V1 feels done but V2 feels too big to start. Here's the reframe:

**V1 is not "done" — it's "structurally complete but hollow."**

The skeleton is right. The bones are in the right place. But the things that make it breathe — real embeddings, real tool dispatch, real data — aren't there yet. That's not a failure, it's exactly the gap between prototype and product.

The V2 plan is 16 steps, 5-6 weeks. That's intimidating. But you don't need all 16 steps to transform this portfolio piece. You need maybe 3.

---

## The 3 Changes That Would Transform This Portfolio Piece

### 1. Wire one real tool call end-to-end (1-2 days)

Right now `zone_semantic_search` returns a hardcoded list. Replace it with a real HNSW lookup — even with simple TF-IDF or a tiny sentence-transformer embedding (not a GPU model). Generate embeddings for all 60 zones at startup. Store them. Return actual nearest neighbors.

This one change makes the demo real. The SLM calls a tool, the tool returns real zones, the response is grounded.

**Why this matters:** It turns "I built a tool-calling framework" into "I built a tool-calling agent that works."

### 2. Persist the XGBoost model (half a day)

Save the trained model to disk with `joblib`. Load it at startup if it exists. Add a `/analytics/model-metrics` response that includes the training timestamp and version.

**Why this matters:** Model persistence is table-stakes ML engineering. Right now the model retrains from 725k rows on every cold start — that's a red flag to any ML engineer reviewing the code.

### 3. Add one real dashboard interaction (1-2 days)

Make the dashboard actually call the `/predict` endpoint and show a recommendation with a zone map. Even a simple Leaflet map with the top 3 zone markers. Something a recruiter can open in a browser and understand in 10 seconds.

**Why this matters:** Visual demos carry more weight than API responses in a portfolio context.

---

## Technology Choices: Verdict

| Choice | Verdict | Notes |
|--------|---------|-------|
| FastAPI | ✅ Right | Industry standard for ML backends |
| XGBoost | ✅ Right | Correct tool for tabular prediction at this scale |
| DuckDB for dev | ✅ Right | Fast iteration without a DB server |
| Qwen2.5 via Ollama | ✅ Right | Local inference, avoids API cost in demos |
| Tool-calling over fine-tuning | ✅ Right | Domain changes fast, tools compose cleanly |
| hnswlib | ✅ Right | Correct library for approximate nearest neighbor at this scale |
| SQLAlchemy + PostgreSQL for prod | ✅ Right | Standard, scalable |
| GKE + Terraform for V2 | ✅ Right | Shows cloud-native thinking |
| Synthetic data with fixed seed | ✅ Acceptable | Valid for prototype, just be upfront about it |
| 256-D zone embeddings | ⚠️ Unvalidated | Good choice on paper, but no embeddings exist to validate against |

---

## Honest Scores (1-5)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| Problem framing | 5/5 | Real problem, clear user impact, specific geography |
| Architecture design | 4/5 | Hybrid XGBoost+SLM is correct; HNSW design is good but unimplemented |
| Code quality | 4/5 | Clean, async, well-tested — minor config issues |
| ML engineering | 3/5 | XGBoost is solid; no model persistence, no HNSW, no embeddings |
| AI/SLM integration | 3/5 | Tool-calling framework correct; all tools are stubs |
| Data engineering | 3/5 | Synthetic data good; GBFS ingest incomplete; no real pipeline |
| Portfolio readiness | 2.5/5 | Strong bones, but demo is hollow — tools return fake data |
| V2 planning | 4.5/5 | Detailed, realistic, shows systems thinking |

**Overall: 3.5/5** — Solidly above average for a side project. Would rank higher if one real E2E flow worked.

---

## What This Project Says About You (Currently)

- You understand ML system design, not just model training
- You know when to use which tool (XGBoost vs. LLM vs. vector search)
- You've thought about production concerns (async, config, Docker, health checks)
- You course-correct based on reasoning (archived fine-tuning, chose tool-calling)
- You can write clean FastAPI code with proper separation of concerns

**What it doesn't yet say:** That you can ship a working AI feature end-to-end.

Close the HNSW + tool-call gap and it will say that too.

---

## Recommended Next Steps (Prioritized)

1. **[2 days] Make zone_semantic_search real** — Generate embeddings at startup (sentence-transformers, small model), populate HNSW index, return real nearest zones. This is the highest-leverage change.
2. **[0.5 days] Persist the XGBoost model** — `joblib.dump` on train, `joblib.load` on startup. Add version/timestamp to `/analytics/model-metrics`.
3. **[1 day] Wire log_outcome to DuckDB** — Even without PostgreSQL, write outcomes to a `rider_outcomes` DuckDB table. Shows the feedback loop works.
4. **[2 days] Build a working demo UI** — One-page dashboard: zone map + SLM query box + prediction result. Something showable in a browser.
5. **[Then] Follow V2 plan** — PostgreSQL migration, GBFS ingest, Airflow DAGs, GCP deployment. In that order.

---

## Bottom Line

This is not a "nice try" project — the architecture is legitimately interesting and the problem space is real. The risk is that it currently reads as "impressive planning with a hollow core." Three focused days of implementation work would flip that to "working AI system with a clear production path." That's a meaningfully different portfolio story.

The V2 plan is your roadmap. You don't need to finish it — you need to finish enough of it that the demo works.
