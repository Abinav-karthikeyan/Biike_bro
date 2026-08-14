"""
SQLAlchemy ORM models — aligned with synthetic seed CSV schema and PRD §7.
Table names match CSV-loaded DuckDB tables exactly so the same models work
against both DuckDB (via duckdb-engine) and PostgreSQL.

Note on autoincrement PKs: Integer (→ SERIAL in Postgres) is used instead
of BigInteger (→ BIGSERIAL) because duckdb-engine 0.13.x does not support
BIGSERIAL.  For Postgres production use, a future Alembic migration can
widen these to BIGINT if row counts demand it.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Double,
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
# zones  (semi-static zone reference data, matches zones.csv exactly)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneMetadata(Base):
    """Semi-static zone reference data; matches zones.csv column layout."""

    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    lat: Mapped[float] = mapped_column(Double, nullable=False)
    lon: Mapped[float] = mapped_column(Double, nullable=False)
    radius_m: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    zone_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    venue_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    capacity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    transit_score: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    neighborhood: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    gbfs_region_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    avg_daily_turnover: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    zone_source: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True, server_default="synthetic"
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    snapshots: Mapped[list["ZoneSnapshot"]] = relationship(
        back_populates="zone", lazy="noload"
    )


# ─────────────────────────────────────────────────────────────────────────────
# zone_snapshots  (5-min occupancy time-series, matches zone_snapshots.csv)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneSnapshot(Base):
    """
    5-minute zone occupancy snapshots from GBFS feeds.
    Indexed on (zone_id, timestamp); retained for 6 months.
    """

    __tablename__ = "zone_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    zone_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("zones.zone_id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    bikes_available: Mapped[int] = mapped_column(Integer, nullable=False)
    capacity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    occupancy_pct: Mapped[float] = mapped_column(Double, nullable=False)
    weather_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    temp_celsius: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    is_event_nearby: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    day_of_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hour_of_day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    zone: Mapped["ZoneMetadata"] = relationship(back_populates="snapshots")


# ─────────────────────────────────────────────────────────────────────────────
# rides  (historical ride records, matches rides.csv)
# ─────────────────────────────────────────────────────────────────────────────


class Ride(Base):
    """Historical ride records; read-only seed data from MDS / operator API."""

    __tablename__ = "rides"

    ride_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    bike_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rider_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    start_zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    end_zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    start_lat: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    start_lon: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    end_lat: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    end_lon: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_secs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_pence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    was_redirected: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    redirect_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    minutes_wasted: Mapped[Optional[float]] = mapped_column(Double, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# weather  (hourly weather records, matches weather.csv)
# ─────────────────────────────────────────────────────────────────────────────


class WeatherRecord(Base):
    """Hourly weather observations joined on (city, hour)."""

    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    temp_celsius: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    precip_mm: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    wind_kmh: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    wmo_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_daylight: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# local_events  (event records, matches local_events.csv)
# ─────────────────────────────────────────────────────────────────────────────


class LocalEvent(Base):
    """Local events used as a feature signal for occupancy prediction."""

    __tablename__ = "local_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    city_id: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    event_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    venue_lat: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    venue_lon: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    expected_attendance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# model_artifacts  (versioned ML model registry)
# ─────────────────────────────────────────────────────────────────────────────


class ModelArtifact(Base):
    """One record per trained model per city per day."""

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
# rider_outcomes  (opt-in telemetry for retraining)
# ─────────────────────────────────────────────────────────────────────────────


class RiderOutcome(Base):
    """
    Opt-in telemetry: actual parking outcomes reported by riders.
    rider_id and session_id are nullable — the SLM telemetry stub has no
    rider identity; they'll be populated once a rider auth layer exists.
    """

    __tablename__ = "rider_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    zone_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    predicted_fill: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    actual_fill: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    arrival_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    satisfaction_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    intent_embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ─────────────────────────────────────────────────────────────────────────────
# zone_embeddings_hnsw  (pre-computed nightly zone embeddings)
# ─────────────────────────────────────────────────────────────────────────────


class ZoneEmbeddingHNSW(Base):
    """256-D zone embeddings for HNSW similarity search."""

    __tablename__ = "zone_embeddings_hnsw"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    embedding_vec: Mapped[list] = mapped_column(
        JSON, nullable=False, comment="256-D float vector"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
