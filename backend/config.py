"""
Application configuration — all values sourced from environment variables.
Copy .env.example → .env and fill in your secrets before running.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5500"]

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "duckdb:///:memory:"  # override with postgresql+asyncpg://... for Postgres
    DATABASE_POOL_SIZE: int = 10

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    ZONE_CACHE_TTL_SECONDS: int = 300  # 5 min occupancy cache

    # ── GBFS ─────────────────────────────────────────────────────────────
    # Glasgow OVO Bikes / Nextbike Glasgow does not expose a public GBFS endpoint
    # (absent from MobilityData systems.csv registry as of 2026-08-14).
    # Proof-of-concept: Nextbike Banja Luka (nextbike_bj) — public, no auth,
    # docked stations with station_status and station_information feeds.
    # Replace with a Glasgow operator feed when one becomes available.
    GBFS_INGEST_ENABLED: bool = False     # set true in .env to start the poller
    GBFS_POLL_INTERVAL_SECONDS: int = 300  # 5 min
    GBFS_FEED_URLS: List[str] = [
        "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_bj/gbfs.json",
    ]

    # ── ML / Prediction ───────────────────────────────────────────────────
    MODEL_ARTIFACTS_S3_BUCKET: str = ""
    MODEL_ARTIFACTS_PATH: str = "models/"
    XGB_MODEL_PATH: str = "models/xgb_model.joblib"
    PREDICTION_LOOKAHEAD_30_MIN_THRESHOLD: float = 0.78  # ≥78% accuracy target

    # ── HNSW ─────────────────────────────────────────────────────────────
    HNSW_INDEX_PATH: str = "data/hnsw_indices/"
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 50

    # ── SLM / On-device ───────────────────────────────────────────────────
    # SLM_MODEL_NAME was previously used for a Transformers/HuggingFace path
    # that was dropped in V2 (Ollama serves both tiers instead).
    # Kept here to avoid breaking .env files that reference it.
    SLM_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"  # unused in V2 — kept for .env backwards compat
    SLM_MAX_LATENCY_MS: int = 200  # <200ms requirement per PRD

    # ── Groq (cloud inference API) ────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_CLOUD_MODEL: str = "llama-3.1-8b-instant"   # cloud tier — 8B, comparable to Qwen2.5 7B
    GROQ_EDGE_MODEL: str = "llama-3.2-1b-preview"     # edge tier — 1B, comparable to Qwen2.5 0.5B
    GROQ_TIMEOUT_SECONDS: int = 30

    # ── RAG context token budgets (per inference tier) ────────────────────
    # ~4 chars/token rule-of-thumb for Qwen2.5's BPE vocabulary.
    RAG_CLOUD_TOKEN_BUDGET: int = 500   # full context: occupancy + history + similar zones + weather + events
    RAG_EDGE_TOKEN_BUDGET: int = 200    # compressed: occupancy + history + weather only


@lru_cache
def get_settings() -> Settings:
    return Settings()
