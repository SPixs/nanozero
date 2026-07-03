"""Pytest fixtures for the PostgreSQL migration smoke test.

A REAL PostgreSQL (testcontainers, ``postgres:16-alpine``) is started ONCE per
session. Each test gets a clean schema via the function-scoped ``pg`` fixture:
the global psycopg3 pool (``storage/db.py``) is closed + re-opened against the
container, the schema is (idempotently) created, and every table is truncated
(``RESTART IDENTITY CASCADE``) so ids/sequences start fresh.

⚠️ This conftest is additive — it does NOT touch the legacy SQLite-era fixtures
embedded inside the existing ``tests/test_*.py`` files (those will be backfilled
separately). Only tests that explicitly request the ``pg`` fixture spin up the
container (fixtures are lazy), so the legacy suite is unaffected by collection.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from nanozero_jobserver.storage import db
from testcontainers.postgres import PostgresContainer

# Every table created by db.init_schema() — truncated between tests for a clean
# slate. CASCADE covers the FKs (positions.batch_id → batches, models self-ref).
_TABLES = (
    "batches",
    "positions",
    "jobs",
    "models",
    "server_control",
    "sprt_results",
    "contributors",
)


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start a real PostgreSQL 16 once for the whole test session."""
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture()
def pg(postgres_container: PostgresContainer) -> Iterator[str]:
    """Per-test clean DB on the shared container.

    Yields the normalized ``postgresql://`` conninfo so API tests can build a
    matching :class:`ServerConfig` (``create_app`` re-calls ``create_pool`` with
    the same URL → idempotent, reuses this pool).
    """
    # testcontainers renders a SQLAlchemy-style URL (postgresql+psycopg2://...);
    # psycopg3 / libpq wants a plain postgresql:// conninfo.
    url = postgres_container.get_connection_url().replace(
        "postgresql+psycopg2://", "postgresql://"
    )
    db.close_pool()
    db.create_pool(url, min_size=1, max_size=4)
    db.init_schema()
    with db.connect() as conn:
        conn.execute(
            "TRUNCATE " + ", ".join(_TABLES) + " RESTART IDENTITY CASCADE"
        )
    try:
        yield url
    finally:
        db.close_pool()
