"""Add zone_source column to zones for synthetic/real GBFS differentiation

Revision ID: a1b2c3d4e5f6
Revises: e5a53e75fad6
Create Date: 2026-08-14

Adds zone_source VARCHAR(16) DEFAULT 'synthetic' to the zones table.
This lets real GBFS station records coexist with the synthetic seed data
without breaking the existing zone counts the test suite asserts.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "e5a53e75fad6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "zones",
        sa.Column(
            "zone_source",
            sa.String(16),
            nullable=True,
            server_default="synthetic",
        ),
    )
    op.execute("UPDATE zones SET zone_source = 'synthetic' WHERE zone_source IS NULL")


def downgrade() -> None:
    op.drop_column("zones", "zone_source")
