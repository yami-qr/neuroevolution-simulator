"""SQLAlchemy resources with explicit PostgreSQL configuration and UTC timestamps."""

import json
import os
from datetime import UTC, datetime

from sqlalchemy import DateTime, Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://neuroevolution:neuroevolution@localhost:5432/neuroevolution"
)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Store timezone-aware times and normalize SQLite's timezone-less results."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Database timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return (
            value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        )


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str | URL | None = None) -> None:
        parsed = make_url(url if url is not None else database_url())
        if parsed.drivername in ("postgres", "postgresql"):
            parsed = parsed.set(drivername="postgresql+psycopg")
        if parsed.drivername not in ("postgresql+psycopg", "sqlite", "sqlite+pysqlite"):
            raise ValueError("Use a PostgreSQL psycopg URL or an explicit SQLite URL.")
        options = {
            "pool_pre_ping": True,
            "json_serializer": lambda value: json.dumps(value, allow_nan=False),
        }
        if parsed.get_backend_name() == "postgresql":
            options["connect_args"] = {"connect_timeout": 5}
        if parsed.get_backend_name() == "sqlite":
            options["connect_args"] = {"check_same_thread": False}
            if parsed.database in (None, "", ":memory:"):
                options["poolclass"] = StaticPool
        self.engine: Engine = create_engine(parsed, **options)
        if parsed.get_backend_name() == "sqlite":
            event.listen(self.engine, "connect", self._enable_foreign_keys)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @staticmethod
    def _enable_foreign_keys(connection, record) -> None:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def close(self) -> None:
        self.engine.dispose()
