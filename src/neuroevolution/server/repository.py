"""Short database transactions for experiment management and leased run execution."""

from dataclasses import asdict
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import case, func, select, update
from sqlalchemy.orm import undefer

from neuroevolution._validation import positive_float
from neuroevolution.evolution import GenerationResult
from neuroevolution.server.database import Database, utcnow
from neuroevolution.server.models import Experiment, Generation, Run
from neuroevolution.server.schemas import (
    ExperimentCreate,
    ExperimentSettings,
    RunCreate,
)


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class LeaseLost(Exception):
    pass


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _get(session, model, identifier):
        row = session.get(model, identifier)
        if row is None:
            raise NotFound(f"{model.__name__} not found.")
        return row

    def create_experiment(self, request: ExperimentCreate) -> Experiment:
        with self.database.sessions.begin() as session:
            row = Experiment(
                name=request.name,
                description=request.description,
                settings=request.settings.model_dump(mode="json"),
            )
            session.add(row)
            session.flush()
            return row

    def get_experiment(self, identifier: UUID) -> Experiment:
        with self.database.sessions() as session:
            return self._get(session, Experiment, identifier)

    def list_experiments(self, *, offset=0, limit=20):
        with self.database.sessions() as session:
            total = session.scalar(select(func.count()).select_from(Experiment))
            rows = session.scalars(
                select(Experiment)
                .order_by(Experiment.created_at.desc(), Experiment.id)
                .offset(offset)
                .limit(limit)
            ).all()
            return rows, total

    def create_run(self, experiment_id: UUID, request: RunCreate) -> Run:
        with self.database.sessions.begin() as session:
            experiment = self._get(session, Experiment, experiment_id)
            settings = ExperimentSettings.model_validate(
                {**experiment.settings, **request.model_dump(exclude_none=True)}
            )
            row = Run(
                experiment_id=experiment_id,
                settings=settings.model_dump(mode="json"),
                requested_generations=settings.generations,
            )
            session.add(row)
            session.flush()
            return row

    def get_run(self, identifier: UUID) -> Run:
        with self.database.sessions() as session:
            return self._get(session, Run, identifier)

    def list_runs(self, *, experiment_id: UUID | None = None, offset=0, limit=20):
        with self.database.sessions() as session:
            query = select(Run)
            count = select(func.count()).select_from(Run)
            if experiment_id is not None:
                self._get(session, Experiment, experiment_id)
                query = query.where(Run.experiment_id == experiment_id)
                count = count.where(Run.experiment_id == experiment_id)
            rows = session.scalars(
                query.order_by(Run.created_at.desc(), Run.id)
                .offset(offset)
                .limit(limit)
            ).all()
            return rows, session.scalar(count)

    def list_generations(self, identifier: UUID, *, offset=0, limit=100):
        with self.database.sessions() as session:
            self._get(session, Run, identifier)
            query = select(Generation).where(Generation.run_id == identifier)
            rows = session.scalars(
                query.order_by(Generation.generation).offset(offset).limit(limit)
            ).all()
            total = session.scalar(
                select(func.count())
                .select_from(Generation)
                .where(Generation.run_id == identifier)
            )
            return rows, total

    def get_replay(self, identifier: UUID) -> dict:
        with self.database.sessions() as session:
            row = session.scalar(
                select(Run)
                .where(Run.id == identifier)
                .options(undefer(Run.champion_replay))
            )
            if row is None:
                raise NotFound("Run not found.")
            if row.champion_replay is None:
                raise Conflict(
                    "Champion replay is available after the worker finishes."
                )
            return row.champion_replay

    def cancel(self, identifier: UUID, *, now: datetime | None = None) -> Run:
        now = now or utcnow()
        with self.database.sessions.begin() as session:
            row = session.scalar(
                select(Run).where(Run.id == identifier).with_for_update()
            )
            if row is None:
                raise NotFound("Run not found.")
            if row.status == "cancelled":
                return row
            if row.status not in ("queued", "running"):
                raise Conflict("Only queued or running runs can be cancelled.")
            row.cancel_requested = True
            row.updated_at = now
            if row.status == "queued":
                row.status, row.finished_at = "cancelled", now
            session.flush()
            return row

    def claim_next(
        self, *, lease_seconds=30, environment=None, now: datetime | None = None
    ) -> Run | None:
        now = now or utcnow()
        duration = positive_float(lease_seconds, "Lease duration")
        with self.database.sessions.begin() as session:
            session.execute(
                update(Run)
                .where(
                    Run.status == "running",
                    Run.lease_expires_at <= now,
                )
                .values(
                    status=case((Run.cancel_requested, "cancelled"), else_="failed"),
                    error=case(
                        (Run.cancel_requested, None),
                        else_="Worker lease expired; saved generations retained.",
                    ),
                    finished_at=now,
                    updated_at=now,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            identifier = session.scalar(
                select(Run.id)
                .where(Run.status == "queued")
                .order_by(Run.created_at, Run.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if identifier is None:
                return None
            return session.scalar(
                update(Run)
                .where(
                    Run.id == identifier,
                    Run.status == "queued",
                )
                .values(
                    status="running",
                    started_at=now,
                    updated_at=now,
                    lease_token=uuid4(),
                    lease_expires_at=now + timedelta(seconds=duration),
                    environment=environment,
                )
                .returning(Run)
            )

    def renew(
        self,
        identifier: UUID,
        token: UUID,
        *,
        lease_seconds=30,
        now: datetime | None = None,
    ) -> bool:
        now = now or utcnow()
        duration = positive_float(lease_seconds, "Lease duration")
        with self.database.sessions.begin() as session:
            identifier = session.scalar(
                update(Run)
                .where(
                    Run.id == identifier,
                    Run.status == "running",
                    Run.lease_token == token,
                    Run.lease_expires_at > now,
                )
                .values(
                    updated_at=now, lease_expires_at=now + timedelta(seconds=duration)
                )
                .returning(Run.id)
            )
            return identifier is not None

    @staticmethod
    def _owned(session, identifier, token, now):
        row = session.scalar(
            select(Run)
            .where(
                Run.id == identifier,
                Run.status == "running",
                Run.lease_token == token,
                Run.lease_expires_at > now,
            )
            .with_for_update()
        )
        if row is None:
            raise LeaseLost("The worker no longer owns this run.")
        return row

    def save_generation(
        self,
        identifier: UUID,
        token: UUID,
        result: GenerationResult,
        *,
        now: datetime | None = None,
    ) -> None:
        now = now or utcnow()
        with self.database.sessions.begin() as session:
            row = self._owned(session, identifier, token, now)
            if (
                result.stats.generation != row.completed_generations
                or row.completed_generations >= row.requested_generations
            ):
                raise Conflict("Generations must be saved once, in contiguous order.")
            best = max(range(len(result.scores)), key=lambda i: result.scores[i].total)
            controller = result.champion.to_dict()
            session.add(
                Generation(
                    run_id=identifier,
                    **asdict(result.stats),
                    champion=controller,
                    fitness={
                        **asdict(result.scores[best]),
                        "total": result.scores[best].total,
                    },
                    episode={
                        "final_state": asdict(result.episodes[best].final_state),
                        "elapsed_seconds": result.episodes[best].elapsed_seconds,
                    },
                )
            )
            row.completed_generations += 1
            row.updated_at = now
            if (
                row.champion_fitness is None
                or result.stats.best_fitness > row.champion_fitness
            ):
                row.champion = controller
                row.champion_fitness = result.stats.best_fitness
                row.champion_generation = result.stats.generation

    def finish(
        self,
        identifier: UUID,
        token: UUID,
        replay: dict | None,
        *,
        now: datetime | None = None,
    ) -> Run:
        now = now or utcnow()
        with self.database.sessions.begin() as session:
            row = self._owned(session, identifier, token, now)
            if (
                not row.cancel_requested
                and row.completed_generations != row.requested_generations
            ):
                raise Conflict(
                    "All requested generations must finish before completion."
                )
            row.status = "cancelled" if row.cancel_requested else "completed"
            row.champion_replay = replay
            row.finished_at = row.updated_at = now
            row.lease_token = row.lease_expires_at = None
            session.flush()
            return row

    def fail(self, identifier: UUID, token: UUID, error: str) -> None:
        now = utcnow()
        with self.database.sessions.begin() as session:
            row = self._owned(session, identifier, token, now)
            row.status, row.error = "failed", error[:2000]
            row.finished_at = row.updated_at = now
            row.lease_token = row.lease_expires_at = None
