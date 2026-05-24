"""
SQLAlchemy ORM models — maps to the core tables defined in PRD §7.
Uses async-compatible SQLAlchemy 2.x declarative syntax.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# zone_metadata  (PRD §7 — semi-static zone reference data)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneMetadata(Base):
    """Semi-static zone reference data; used for zone embeddings."""

    __tablename__ = "zone_metadata"

    zone_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    venue_type_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    transit_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neighborhood: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    avg_daily_turnover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    snapshots: Mapped[list["ZoneSnapshot"]] = relationship(
        back_populates="zone", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# zone_snapshots  (PRD §7 — 5-min occupancy time-series)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneSnapshot(Base):
    """
    5-minute zone occupancy snapshots from GBFS feeds.
    Indexed on (zone_id, timestamp); retained for 6 months.
    Exported to Delta Lake weekly.
    """

    __tablename__ = "zone_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    zone_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("zone_metadata.zone_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    available_bikes: Mapped[int] = mapped_column(Integer, nullable=False)
    docks_used: Mapped[int] = mapped_column(Integer, nullable=False)
    occupancy_pct: Mapped[float] = mapped_column(Float, nullable=False)
    weather_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    local_events_mask: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    zone: Mapped["ZoneMetadata"] = relationship(back_populates="snapshots")


# ─────────────────────────────────────────────────────────────────────────────
# model_artifacts  (PRD §7 — versioned ML model registry)
# ─────────────────────────────────────────────────────────────────────────────


class ModelArtifact(Base):
    """
    One record per trained model per city per day.
    Versioned for rollback; s3_path points to XGBoost/LightGBM binary.
    """

    __tablename__ = "model_artifacts"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    xgb_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    holdout_mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    city: Mapped[str] = mapped_column(String(128), nullable=False)


# ─────────────────────────────────────────────────────────────────────────────
# rider_outcomes  (PRD §7 — opt-in telemetry for retraining)
# ─────────────────────────────────────────────────────────────────────────────


class RiderOutcome(Base):
    """
    Opt-in telemetry for SLM fine-tuning and prediction validation.
    intent_embedding stored as JSON array (128-D).
    """

    __tablename__ = "rider_outcomes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rider_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    zone_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    predicted_fill: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_fill: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    arrival_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    satisfaction_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    intent_embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ─────────────────────────────────────────────────────────────────────────────
# zone_embeddings_hnsw  (PRD §7 — pre-computed nightly; pushed to mobile)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneEmbeddingHNSW(Base):
    """
    256-D zone embeddings for HNSW similarity search.
    Pre-computed nightly; pushed to mobile app as binary blob.
    """

    __tablename__ = "zone_embeddings_hnsw"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    zone_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embedding_vec: Mapped[list] = mapped_column(
        JSON, nullable=False, comment="256-D float vector"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
