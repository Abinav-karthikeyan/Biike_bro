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
    DATABASE_URL: str = "postgresql+asyncpg://buddy:buddy@localhost:5432/cycle_buddy"
    DATABASE_POOL_SIZE: int = 10

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    ZONE_CACHE_TTL_SECONDS: int = 300  # 5 min occupancy cache

    # ── GBFS ─────────────────────────────────────────────────────────────
    # TODO: Add operator-specific GBFS feed URLs here
    GBFS_POLL_INTERVAL_SECONDS: int = 300  # 5 min as per PRD
    GBFS_FEED_URLS: List[str] = []        # e.g. ["https://data.lime.bike/api/partners/v1/gbfs/..."]

    # ── ML / Prediction ───────────────────────────────────────────────────
    MODEL_ARTIFACTS_S3_BUCKET: str = ""
    MODEL_ARTIFACTS_PATH: str = "models/"
    PREDICTION_LOOKAHEAD_30_MIN_THRESHOLD: float = 0.78  # ≥78% accuracy target

    # ── HNSW ─────────────────────────────────────────────────────────────
    HNSW_INDEX_PATH: str = "data/hnsw_indices/"
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 50

    # ── SLM / On-device ───────────────────────────────────────────────────
    # Model: Qwen2.5-0.5B or Phi-3.5-mini (on-device; no cloud calls)
    SLM_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    SLM_MAX_LATENCY_MS: int = 200  # <200ms requirement per PRD

    # ── Ollama (local inference daemon) ───────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "qwen2.5"  # tag in `ollama list`; override after fine-tuning
    OLLAMA_TIMEOUT_SECONDS: int = 60    # generous for 7B on CPU


@lru_cache
def get_settings() -> Settings:
    return Settings()
