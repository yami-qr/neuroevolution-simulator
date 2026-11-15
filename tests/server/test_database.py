"""Actual migrations, JSON/UUID persistence, transactions, and foreign keys."""

from datetime import UTC
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from neuroevolution.server.database import Base
from neuroevolution.server.models import Experiment, Run

from .conftest import migration_config


def test_migration_matches_models_and_can_downgrade_and_upgrade(database):
    with database.engine.begin() as connection:
        assert (
            compare_metadata(MigrationContext.configure(connection), Base.metadata)
            == []
        )
        command.downgrade(migration_config(connection), "base")
        assert "experiments" not in inspect(connection).get_table_names()
        command.upgrade(migration_config(connection), "head")
        assert {"experiments", "runs", "generations"} <= set(
            inspect(connection).get_table_names()
        )


def test_json_uuid_and_utc_timestamps_roundtrip_across_sessions(database):
    with database.sessions.begin() as session:
        experiment = Experiment(
            name="Saved experiment", settings={"seed": 42, "values": [1.25, -2]}
        )
        session.add(experiment)
        session.flush()
        run = Run(
            experiment_id=experiment.id,
            settings=experiment.settings,
            requested_generations=3,
        )
        session.add(run)
        session.flush()
        experiment_id, run_id = experiment.id, run.id
    with database.sessions() as session:
        experiment = session.get(Experiment, experiment_id)
        run = session.get(Run, run_id)
        assert experiment.settings == {"seed": 42, "values": [1.25, -2]}
        assert run.experiment_id == experiment_id
        assert run.status == "queued"
        assert run.completed_generations == 0
        assert experiment.created_at.tzinfo == UTC
        assert run.created_at.tzinfo == UTC
        if database.engine.dialect.name == "postgresql":
            assert (
                session.scalar(
                    text("SELECT pg_typeof(settings)::text FROM experiments")
                )
                == "jsonb"
            )


def test_foreign_keys_are_enforced(database):
    with pytest.raises(IntegrityError), database.sessions.begin() as session:
        session.add(Run(experiment_id=uuid4(), settings={}, requested_generations=1))


def test_failed_transactions_do_not_leave_partial_experiments(database):
    with pytest.raises(RuntimeError), database.sessions.begin() as session:
        session.add(Experiment(name="Rolled back", settings={}))
        session.flush()
        raise RuntimeError("Abort the transaction")
    with database.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Experiment)) == 0
