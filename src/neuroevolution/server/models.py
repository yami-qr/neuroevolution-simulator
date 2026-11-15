"""Persistent experiments, queued runs, and per-generation results."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from neuroevolution.server.database import Base, UTCDateTime, utcnow

JSON_VALUE = JSON().with_variant(JSONB(), "postgresql")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    settings: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, index=True
    )


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_runs_status",
        ),
        CheckConstraint("requested_generations > 0", name="ck_runs_requested"),
        CheckConstraint(
            "completed_generations >= 0 "
            "AND completed_generations <= requested_generations",
            name="ck_runs_progress",
        ),
        Index("ix_runs_queue", "status", "created_at"),
        Index("ix_runs_expiry", "status", "lease_expires_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    settings: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    requested_generations: Mapped[int]
    completed_generations: Mapped[int] = mapped_column(default=0)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    lease_token: Mapped[UUID | None] = mapped_column(Uuid)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    environment: Mapped[dict[str, Any] | None] = mapped_column(JSON_VALUE)
    error: Mapped[str | None] = mapped_column(Text)
    champion: Mapped[dict[str, Any] | None] = mapped_column(JSON_VALUE)
    champion_fitness: Mapped[float | None] = mapped_column(Float)
    champion_generation: Mapped[int | None]
    champion_replay: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_VALUE, deferred=True
    )


class Generation(Base):
    __tablename__ = "generations"

    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
    )
    generation: Mapped[int] = mapped_column(primary_key=True)
    population_size: Mapped[int]
    best_fitness: Mapped[float] = mapped_column(Float)
    mean_fitness: Mapped[float] = mapped_column(Float)
    completed_count: Mapped[int]
    best_progress: Mapped[float] = mapped_column(Float)
    best_checkpoints: Mapped[int]
    champion: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    fitness: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    episode: Mapped[dict[str, Any]] = mapped_column(JSON_VALUE)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
