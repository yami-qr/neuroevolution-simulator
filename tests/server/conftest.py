"""Disposable migrated databases; PostgreSQL tests own an isolated schema."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from neuroevolution.server.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def migration_config(connection):
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


@pytest.fixture
def database(tmp_path):
    external = os.environ.get("TEST_DATABASE_URL")
    admin = None
    schema = "test_" + uuid4().hex
    if external:
        admin = Database(external)
        if admin.engine.dialect.name != "postgresql":
            raise ValueError("TEST_DATABASE_URL must point to PostgreSQL.")
        with admin.engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        url = admin.engine.url.update_query_dict({"options": f"-csearch_path={schema}"})
    else:
        url = f"sqlite:///{(tmp_path / 'backend.sqlite').as_posix()}"
    database = Database(url)
    try:
        with database.engine.begin() as connection:
            command.upgrade(migration_config(connection), "head")
        yield database
    finally:
        database.close()
        if admin is not None:
            with admin.engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            admin.close()
