"""
SQLAlchemy engine and session factory.

make_engine() is the single place that translates a DATABASE_URL string
into a configured Engine.  The rest of the app only calls make_engine()
once (in main.py lifespan or DuckDBStore.__init__) and then passes the
engine around.
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def make_engine(database_url: str) -> Engine:
    """
    Return a SQLAlchemy Engine from any supported DATABASE_URL.

    Supported schemes:
      duckdb:///:memory:              in-memory DuckDB (tests / dev)
      duckdb:///path/to/db.duckdb    file-backed DuckDB
      postgresql+psycopg2://...       PostgreSQL via psycopg2

    In-memory DuckDB requires StaticPool so every engine.connect() call
    reuses the same underlying connection.  Without it, each call opens a
    fresh empty in-memory database and the seeded data is invisible.
    File-backed DuckDB and Postgres use the default QueuePool.
    """
    if database_url == "duckdb:///:memory:":
        return create_engine(database_url, poolclass=StaticPool)
    if database_url.startswith("duckdb"):
        return create_engine(database_url)
    # Postgres and anything else
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session(session_factory: sessionmaker) -> Session:
    """Return a bare Session; caller is responsible for commit/close."""
    return session_factory()
