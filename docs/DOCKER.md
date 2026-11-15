# Run the complete lab with Docker Compose

Install Docker with the Compose v2 plugin (Docker Desktop on Windows), start its
engine, and use Linux containers. No host Python or Node installation is needed.
From this repository's root:

```powershell
docker compose up --build --detach --wait
```

Open [Evolution lab](http://127.0.0.1:8000/). Create an experiment and choose
**Create & run**; the worker is already running. The dashboard, assets, and API
share this address. The [dashboard guide](DASHBOARD.md) explains its controls.

The default `compose.yml` starts PostgreSQL 17, a one-shot Alembic migration job,
the API/dashboard, and a worker. PostgreSQL must be healthy and migrations must
succeed before the application services start. The API health check uses `/ready`.
The worker's actual execution is verified by a queued run, rather than an HTTP
health endpoint. API, migration job, and worker use the same non-root Python image.

Only the API port is published, bound to the host loopback interface. PostgreSQL
is reachable within the Compose network using `postgres`, not `localhost`.
The application services explicitly set their database URL; a host `DATABASE_URL`
or the development `.env.example` does not override it. The included credentials
are for this local lab, which has no authentication; this configuration is not a
public hosting setup.

If port 8000 is occupied, set `LAB_PORT=8001` in a local `.env` file or run:

```powershell
$env:LAB_PORT = '8001'
docker compose up --build --detach --wait
```

Then open `http://127.0.0.1:8001/`. Compose uses a separate `lab-postgres-data`
volume from the older database-only configuration; it does not import SQLite or
that existing database's experiments automatically. Do not combine the two
Compose files. For the database-only workflow, continue using
`docker compose -f compose.postgres.yml up --detach` and the [API guide](API.md).

## Inspect and stop

```powershell
docker compose ps --all
docker compose logs --follow api worker
docker compose logs migrate
docker compose down
```

`down` removes containers and their network but keeps PostgreSQL data. Running
`up` again recreates the services, checks migrations, and restores saved runs.
To **permanently delete all experiments in this Compose database**, use
`docker compose down --volumes` instead. This is not needed for ordinary shutdown
or upgrades. Avoid removing the data volume when troubleshooting an image build.

## Upgrade the application

After pulling repository changes, stop the services and rebuild:

```powershell
docker compose down
docker compose up --build --detach --wait
```

This retains the database and ensures the migration job runs before the new
application. Image dependencies use the version ranges in `pyproject.toml`; builds
are not locked byte-for-byte. Run the CI smoke test after dependency updates.

Stopping a worker interrupts an active generation. Saved generations remain;
the unfinished run becomes failed when its ownership lease expires (normally
30 seconds) and the worker polls again. It does not silently resume the random
evolution sequence. Start another run from the saved experiment when ready.

## Validate the stack

GitHub Actions builds the image and starts the actual Compose services. It queues
a short experiment, verifies completion, checks saved populations and replay,
downloads a Matplotlib PNG, then recreates the containers and verifies persisted
results. The same check can be run locally with Python 3.11 or newer:

```powershell
python scripts/compose_smoke.py --run-file artifacts/compose-run-id
docker compose down
docker compose up --detach --wait
python scripts/compose_smoke.py --run-file artifacts/compose-run-id --verify
```

Create the ignored `artifacts` directory first if necessary. Pass `--url` when
using a different API port. A queued run that never progresses calls for checking
worker logs; migration or readiness failures call for checking `migrate` and
`postgres` logs. `docker compose config --quiet` checks configuration syntax
without starting services.
