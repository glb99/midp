"""Shared fixtures for the Postgres-backed suites (tests/integration and tests/e2e).

As in orkid, the suite owns its own database: it drops and recreates `midp_test`
on the same server, applies the migrations to it, and gives each test a session
inside a transaction that is rolled back at the end. The development database
is never touched.

The tests are skipped, not failed, when Postgres does not answer.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session

from alembic import command
from midp.db import make_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_URL = os.environ.get("MIDP_SERVER_URL", "postgresql+psycopg://midp:local-dev-only@localhost:5432/midp")
TEST_DATABASE = "midp_test"


def _test_url(server_url: str) -> str:
    return server_url.rsplit("/", 1)[0] + f"/{TEST_DATABASE}"


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    admin = make_engine(SERVER_URL).execution_options(isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:  # noqa: BLE001 - any failure to reach the server means "not available"
        pytest.skip(f"Postgres not reachable at {SERVER_URL}: {error.__class__.__name__}")

    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE}"))
        connection.execute(text(f"CREATE DATABASE {TEST_DATABASE}"))
    admin.dispose()

    test_url = _test_url(SERVER_URL)
    previous = os.environ.get("MIDP_DATABASE_URL")
    os.environ["MIDP_DATABASE_URL"] = test_url
    try:
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        command.upgrade(config, "head")
        test_engine = make_engine(test_url)
        yield test_engine
        test_engine.dispose()
    finally:
        if previous is None:
            os.environ.pop("MIDP_DATABASE_URL", None)
        else:
            os.environ["MIDP_DATABASE_URL"] = previous


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session inside an outer transaction that is rolled back after the test.

    join_transaction_mode="create_savepoint" makes the code's commit() release a
    savepoint instead of committing the outer transaction, so nothing survives.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()
