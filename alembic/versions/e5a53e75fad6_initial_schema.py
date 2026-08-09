"""Initial schema

Revision ID: e5a53e75fad6
Revises:
Create Date: 2026-08-09

Creates all ORM-managed tables.  Seed data (zones, zone_snapshots, rides,
weather, local_events) is loaded separately via scripts/seed_postgres.py.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5a53e75fad6"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── zones ────────────────────────────────────────────────────────────
    op.create_table(
        "zones",
        sa.Column("zone_id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("lat", sa.Double(), nullable=False),
        sa.Column("lon", sa.Double(), nullable=False),
        sa.Column("radius_m", sa.Double(), nullable=True),
        sa.Column("zone_type", sa.String(64), nullable=True),
        sa.Column("venue_type", sa.String(64), nullable=True),
        sa.Column("capacity", sa.Integer(), nullable=True),
        sa.Column("transit_score", sa.Double(), nullable=True),
        sa.Column("neighborhood", sa.String(255), nullable=True),
        sa.Column("gbfs_region_id", sa.String(64), nullable=True),
        sa.Column("avg_daily_turnover", sa.Double(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
    )

    # ── zone_snapshots ───────────────────────────────────────────────────
    op.create_table(
        "zone_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_id", sa.Integer(), nullable=True),
        sa.Column(
            "zone_id",
            sa.String(64),
            sa.ForeignKey("zones.zone_id"),
            nullable=False,
            index=True,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("bikes_available", sa.Integer(), nullable=False),
        sa.Column("capacity", sa.Integer(), nullable=True),
        sa.Column("occupancy_pct", sa.Double(), nullable=False),
        sa.Column("weather_code", sa.Integer(), nullable=True),
        sa.Column("temp_celsius", sa.Double(), nullable=True),
        sa.Column("is_event_nearby", sa.Boolean(), nullable=True),
        sa.Column("day_of_week", sa.Integer(), nullable=True),
        sa.Column("hour_of_day", sa.Integer(), nullable=True),
    )

    # ── rides ────────────────────────────────────────────────────────────
    op.create_table(
        "rides",
        sa.Column("ride_id", sa.String(64), primary_key=True),
        sa.Column("bike_id", sa.String(64), nullable=True),
        sa.Column("rider_hash", sa.String(128), nullable=True),
        sa.Column("start_zone_id", sa.String(64), nullable=True),
        sa.Column("end_zone_id", sa.String(64), nullable=True),
        sa.Column("start_lat", sa.Double(), nullable=True),
        sa.Column("start_lon", sa.Double(), nullable=True),
        sa.Column("end_lat", sa.Double(), nullable=True),
        sa.Column("end_lon", sa.Double(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_secs", sa.Integer(), nullable=True),
        sa.Column("distance_m", sa.Integer(), nullable=True),
        sa.Column("cost_pence", sa.Integer(), nullable=True),
        sa.Column("was_redirected", sa.Boolean(), nullable=True),
        sa.Column("redirect_reason", sa.String(64), nullable=True),
        sa.Column("minutes_wasted", sa.Double(), nullable=True),
    )

    # ── weather ──────────────────────────────────────────────────────────
    op.create_table(
        "weather",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("city_id", sa.String(16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("temp_celsius", sa.Double(), nullable=True),
        sa.Column("precip_mm", sa.Double(), nullable=True),
        sa.Column("wind_kmh", sa.Double(), nullable=True),
        sa.Column("wmo_code", sa.Integer(), nullable=True),
        sa.Column("is_daylight", sa.Boolean(), nullable=True),
    )

    # ── local_events ─────────────────────────────────────────────────────
    op.create_table(
        "local_events",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("city_id", sa.String(16), nullable=True),
        sa.Column("event_date", sa.Date(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("venue_lat", sa.Double(), nullable=True),
        sa.Column("venue_lon", sa.Double(), nullable=True),
        sa.Column("expected_attendance", sa.Integer(), nullable=True),
        sa.Column("category", sa.String(64), nullable=True),
    )

    # ── model_artifacts ──────────────────────────────────────────────────
    op.create_table(
        "model_artifacts",
        sa.Column("model_id", sa.String(64), primary_key=True),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("xgb_params", sa.JSON(), nullable=True),
        sa.Column("holdout_mae", sa.Float(), nullable=True),
        sa.Column("s3_path", sa.Text(), nullable=True),
        sa.Column("city", sa.String(128), nullable=False),
    )

    # ── rider_outcomes ───────────────────────────────────────────────────
    op.create_table(
        "rider_outcomes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("rider_id", sa.String(64), nullable=True, index=True),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("zone_id", sa.String(64), nullable=False, index=True),
        sa.Column("predicted_fill", sa.Double(), nullable=True),
        sa.Column("actual_fill", sa.Double(), nullable=True),
        sa.Column("arrival_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("satisfaction_score", sa.Integer(), nullable=True),
        sa.Column("intent_embedding", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    # ── zone_embeddings_hnsw ─────────────────────────────────────────────
    op.create_table(
        "zone_embeddings_hnsw",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("zone_id", sa.String(64), nullable=False, index=True),
        sa.Column("embedding_vec", sa.JSON(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("model_version", sa.String(32), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("zone_embeddings_hnsw")
    op.drop_table("rider_outcomes")
    op.drop_table("model_artifacts")
    op.drop_table("local_events")
    op.drop_table("weather")
    op.drop_table("rides")
    op.drop_table("zone_snapshots")
    op.drop_table("zones")
