# Experiment API and PostgreSQL persistence

The API manages immutable experiment definitions and separate execution runs.
Evolution runs in a standalone worker process. PostgreSQL stores the queue,
settings, generation metrics, controller snapshots, and champion replays.

## PostgreSQL quick start

Install the project dependencies from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

With Docker installed and running, start the development database:

```powershell
docker compose -f compose.postgres.yml up -d --wait
```

This database-only Compose file uses PostgreSQL 17 and a named data volume.
The development credentials match `.env.example`. An existing PostgreSQL server
can also be used by setting its connection URL below.

In the API terminal, set the URL, apply migrations, and start the server:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://neuroevolution:neuroevolution@localhost:5432/neuroevolution"
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn neuroevolution.server.api:create_app --factory --host 127.0.0.1 --port 8000
```

In a second terminal in the same repository, start the worker with the same URL:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://neuroevolution:neuroevolution@localhost:5432/neuroevolution"
.\.venv\Scripts\python.exe -m neuroevolution.server.worker
```

Python reads `DATABASE_URL` from the process environment; `.env.example` is a
configuration reference. Each terminal needs its own environment setting.
The API binds to localhost. Interactive API documentation is available at
[localhost:8000/docs](http://127.0.0.1:8000/docs).

## Local development without PostgreSQL

An explicit SQLite URL supports local API testing. In each terminal, use:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
$env:DATABASE_URL = "sqlite:///artifacts/local-api.sqlite"
```

Then run the same migration, API, and worker commands above. PostgreSQL remains
the default backend and the target for concurrent worker operation. SQLite
development should use a single worker.

## Create and queue an experiment

These commands create a small run suitable for a quick API check:

```powershell
$experimentBody = @{
    name = "First API experiment"
    settings = @{
        seed = 42
        generations = 3
        evolution = @{ population_size = 8; elite_count = 1; tournament_size = 3 }
        agent = @{ max_steps = 60 }
    }
} | ConvertTo-Json -Depth 10

$experiment = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/experiments" -ContentType "application/json" -Body $experimentBody
$queueUrl = "http://127.0.0.1:8000/api/v1/experiments/" + $experiment.id + "/runs"
$run = Invoke-RestMethod -Method Post -Uri $queueUrl -ContentType "application/json" -Body '{}'
$runUrl = "http://127.0.0.1:8000/api/v1/runs/" + $run.id
Invoke-RestMethod -Uri $runUrl
```

The queue endpoint returns HTTP 202 immediately. The worker evaluates the job
and saves progress after each generation. Omitting `settings` uses the full
engine defaults. The small example above has a one-second episode limit and
is intended to exercise the API; longer episodes are needed to solve the slalom.

## Endpoints

All resource endpoints use the `/api/v1` prefix.

| Method and path | Behaviour |
| --- | --- |
| `GET /defaults` | Complete default experiment settings, including the course |
| `POST /experiments` | Validate and save an experiment; HTTP 201 |
| `GET /experiments` | Paginated experiment list |
| `GET /experiments/{id}` | Experiment definition and full configuration |
| `POST /experiments/{id}/runs` | Queue a run; optional seed and generation-count overrides |
| `GET /runs` | Paginated runs; optional `experiment_id` filter |
| `GET /runs/{id}` | Status, progress, timestamps, environment, and best score |
| `POST /runs/{id}/cancel` | Cancel a queued job or request cancellation of an active job |
| `GET /runs/{id}/generations` | Ordered, paginated generation metrics |
| `GET /runs/{id}/champion` | Best controller snapshot, fitness, generation, replay availability |
| `GET /runs/{id}/replay` | Persisted champion trajectory and fitness breakdown |

List responses contain `items`, `total`, `offset`, and `limit`. Resource lists
allow up to 100 items per page; generation histories allow up to 1000.
Unknown IDs return 404, invalid requests return 422, and results unavailable
at the current lifecycle stage return 409.

`GET /health` checks API liveness. `GET /ready` checks database connectivity
and migration readiness, returning 503 when unavailable. Application startup
does not create or alter tables; migrations are explicit.

## Execution and persistence

Runs transition from `queued` to `running`, then `completed`, `failed`, or
`cancelled`. Each run stores its own immutable configuration snapshot. New runs
can vary the seed and generation count while preserving the experiment definition.
Changing other settings creates a new experiment.

Workers claim jobs atomically. PostgreSQL row locks and ownership tokens prevent
two workers from executing the same queued job. The default heartbeat interval
is five seconds and the lease lasts thirty seconds. Evolution executes outside
database transactions; each generation and its progress update commit together.

Queued cancellation is immediate. Running cancellation is checked between
generations, preserving the completed generation. Healthy completion or
cancellation also saves the best controller's replay and verifies its fitness.

If a worker crashes, a later worker poll marks its expired lease failed and
retains saved generations and champion parameters. A cancelled job with an
expired lease becomes cancelled. Retries are new runs from generation zero;
existing results remain available. A stale worker cannot overwrite results.

Use `--once` to process at most one job, or configure `--poll-interval`,
`--lease-seconds`, and `--heartbeat-seconds` on the worker command.

PostgreSQL uses UUID identifiers, timezone-aware timestamps, JSONB documents,
foreign keys, and a unique run/generation key. Large replay payloads are loaded
only when requested. Environment metadata records Python, NumPy, and simulator
versions. The saved controller can be loaded with `FeedForwardNetwork.from_dict`.

## Verification

The regular test matrix covers the engine and backend on Windows and Linux.
An additional CI job runs all backend tests against a real PostgreSQL 17
service, including migrations, JSONB round-trips, concurrent claiming, worker
execution, cancellation, retained results after failure, and champion replay.
Each PostgreSQL test owns a separate temporary schema.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

To run the PostgreSQL cases locally, set `TEST_DATABASE_URL` to an accessible
test database whose role can create schemas. Tests create and remove only their
own uniquely named schemas.
