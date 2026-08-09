import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Wire in the ORM models so autogenerate can diff them
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.models.db_models import Base  # noqa: E402

target_metadata = Base.metadata

# DATABASE_URL: environment variable wins, then alembic.ini, then Postgres default
DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or config.get_main_option("sqlalchemy.url")
    or "postgresql+psycopg2://buddy:buddy@localhost:5432/cycle_buddy"
)
# Alembic doesn't support asyncpg — swap asyncpg driver for psycopg2
DATABASE_URL = DATABASE_URL.replace("+asyncpg", "+psycopg2")


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": DATABASE_URL},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
