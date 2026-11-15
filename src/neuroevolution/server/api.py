"""FastAPI experiment management; CPU-heavy evolution runs in separate workers."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from alembic.migration import MigrationContext
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from neuroevolution.server.database import Database
from neuroevolution.server.models import Run
from neuroevolution.server.repository import Conflict, NotFound, Repository
from neuroevolution.server.schemas import (
    ChampionRead,
    ExperimentCreate,
    ExperimentRead,
    ExperimentSettings,
    GenerationRead,
    Page,
    RunCreate,
    RunRead,
)

logger = logging.getLogger(__name__)
Offset = Annotated[int, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=100)]
HistoryLimit = Annotated[int, Query(ge=1, le=1000)]


def create_app(database: Database | None = None) -> FastAPI:
    database = database if database is not None else Database()
    repository = Repository(database)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            database.close()

    app = FastAPI(
        title="Neuroevolution Simulator API", version="0.1.0", lifespan=lifespan
    )

    @app.exception_handler(NotFound)
    async def not_found(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, error):
        logger.error("Database operation failed: %s", type(error).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Database unavailable; check the connection and migrations."
            },
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Keep errors JSON-safe even for invalid NaN input or exception contexts.
        details = [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health", tags=["Health"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["Health"])
    def ready():
        with database.sessions() as session:
            if (
                MigrationContext.configure(session.connection()).get_current_revision()
                != "0001"
            ):
                raise HTTPException(503, "Database migrations are pending.")
            session.scalar(select(Run.id).limit(1))
        return {"status": "ready"}

    @app.get(
        "/api/v1/defaults", response_model=ExperimentSettings, tags=["Experiments"]
    )
    def defaults():
        return ExperimentSettings()

    @app.post(
        "/api/v1/experiments",
        response_model=ExperimentRead,
        status_code=201,
        tags=["Experiments"],
    )
    def create_experiment(request: ExperimentCreate):
        return repository.create_experiment(request)

    @app.get(
        "/api/v1/experiments", response_model=Page[ExperimentRead], tags=["Experiments"]
    )
    def list_experiments(offset: Offset = 0, limit: Limit = 20):
        rows, total = repository.list_experiments(offset=offset, limit=limit)
        return {"items": rows, "total": total, "offset": offset, "limit": limit}

    @app.get(
        "/api/v1/experiments/{identifier}",
        response_model=ExperimentRead,
        tags=["Experiments"],
    )
    def get_experiment(identifier: UUID):
        return repository.get_experiment(identifier)

    @app.post(
        "/api/v1/experiments/{identifier}/runs",
        response_model=RunRead,
        status_code=202,
        tags=["Runs"],
    )
    def create_run(identifier: UUID, request: RunCreate | None = None):
        return repository.create_run(identifier, request or RunCreate())

    @app.get("/api/v1/runs", response_model=Page[RunRead], tags=["Runs"])
    def list_runs(
        experiment_id: UUID | None = None, offset: Offset = 0, limit: Limit = 20
    ):
        rows, total = repository.list_runs(
            experiment_id=experiment_id, offset=offset, limit=limit
        )
        return {"items": rows, "total": total, "offset": offset, "limit": limit}

    @app.get("/api/v1/runs/{identifier}", response_model=RunRead, tags=["Runs"])
    def get_run(identifier: UUID):
        return repository.get_run(identifier)

    @app.post(
        "/api/v1/runs/{identifier}/cancel",
        response_model=RunRead,
        status_code=202,
        tags=["Runs"],
    )
    def cancel_run(identifier: UUID):
        return repository.cancel(identifier)

    @app.get(
        "/api/v1/runs/{identifier}/generations",
        response_model=Page[GenerationRead],
        tags=["Results"],
    )
    def generations(identifier: UUID, offset: Offset = 0, limit: HistoryLimit = 100):
        rows, total = repository.list_generations(
            identifier, offset=offset, limit=limit
        )
        return {"items": rows, "total": total, "offset": offset, "limit": limit}

    @app.get(
        "/api/v1/runs/{identifier}/champion",
        response_model=ChampionRead,
        tags=["Results"],
    )
    def champion(identifier: UUID):
        return repository.get_champion(identifier)

    @app.get("/api/v1/runs/{identifier}/replay", response_model=dict, tags=["Results"])
    def replay(identifier: UUID):
        return repository.get_replay(identifier)

    return app
