# Evolution lab dashboard

The dark lab dashboard is served by FastAPI at `/`. It uses browser-native HTML,
CSS, JavaScript modules, canvas, and SVG. The Python package includes the static
assets; there is no frontend build step or separate development server.

## Start the local lab

Install the Python package from this source checkout as described in the README,
then run:

```powershell
.\.venv\Scripts\python.exe -m neuroevolution.server.dev --sqlite
```

Open [the dashboard](http://127.0.0.1:8000/). This explicit development option
creates `artifacts/dashboard.sqlite`, applies migrations, and starts the API and
worker. Experiments persist between restarts. Ctrl+C stops both child processes;
interrupted jobs retain saved generations and become failed when their lease
expires. Use `--port 8001` if port 8000 is occupied.

For PostgreSQL, start the database and configure `DATABASE_URL` using the
[API guide](API.md), then run the same launcher **without `--sqlite`**. It applies
migrations to the configured database and starts the services. The API and worker
can also run separately. Existing databases must be migrated to revision `0002`.

The launcher binds to `127.0.0.1`. This is a local research tool without
authentication; hosted deployment and full application containers are a later
milestone. Node.js is only required for frontend development checks:

```powershell
npm ci
npm test
npm run format:check
```

## Run an experiment

1. Choose **New experiment**. Set a name, seed, population, generation count,
   episode limit, and mutation settings. Advanced settings expose elites,
   tournament size, and hidden-layer width.
2. **Save only** stores the configuration. **Create & run** also queues its first
   run. **Start run** repeats the selected experiment with its saved settings;
   previous runs stay in the run picker.
3. The worker evaluates complete generations. The dashboard polls every 1.5
   seconds during training and every eight seconds for terminal runs. A
   persistent **Queued** status means a worker needs to be started.
4. **Cancel run** requests cancellation after the current generation. Saved
   results remain inspectable for cancelled and failed runs.

Generation zero is the initial population: thirty saved generations contain 29
breeding cycles. Improved fitness does not imply a solved course; check the
completion counts. Try seed **19**, population **24**, elites **1**, and **30**
generations with the other defaults. This produced goal-reaching controllers in
the recorded environment; results can differ between NumPy environments.

## Inspect the results

- **Follow latest** selects each newly saved generation and its champion.
  Choosing a chart point or generation disables this so historical inspection
  stays put during training. Recheck it to return to the latest result.
- Dots show agents' **final outcomes**. Click a dot or use **Selected agent** to
  replay its controller. The generation champion is selected by default; ★ marks
  it in the picker.
- **Play replay**, **Pause replay**, speed, and the seek slider control the saved
  attempt locally. They do not pause or accelerate training. Playback uses
  simulation time. The arena preserves the engine's upward-positive Y axis.
- Toggle sensor rays and trails. The inspector shows the replay frame's position,
  speed, heading, checkpoints, obstacle distances, and movement outputs. Throttle
  maps the signed output to 0–100%. Terminal frames have no movement outputs.
  Fitness is the agent's final episode score.
- Charts show generation best/mean fitness and completion rate. Hover for values;
  click to inspect. The generation picker provides keyboard selection. **CSV**
  exports all saved metrics; **PNG** exports a standalone Matplotlib chart of
  fitness and completion on the backend.
- **Controller JSON** exports the selected agent's actual network parameters.
  **Experiments** reopens saved configurations and runs. The last selected
  experiment/run is remembered in browser storage.

## Storage and replay limits

Migration `0002` adds a nullable population snapshot to each generation. New
worker runs save every controller, score, and final state for inspection. Storage
grows with population, network size, and generations; large experiments can use
substantial database space. Metric listings exclude the snapshots. Older
generations retain champion inspection but cannot recover unsaved agents.

Inspection reruns the stored controller and verifies its score against saved
fitness using the current engine and NumPy environment. A mismatched result is
reported as unavailable. Responses contain at most 1,200 frames by default (up to
2,000 on request), preserving first and terminal states. Long episodes are
labelled **Sampled replay**; playback holds sampled states without interpolating
a path through obstacles. Each API process caches four recent inspected replays;
restarting the API clears that cache, not saved results.

## Additional result endpoints

- `GET /api/v1/runs/{id}/generations/{generation}/population`: indexed final
  outcomes, fitness, and champion index; no controller arrays.
- `GET /api/v1/runs/{id}/generations/{generation}/replay`: champion replay with
  inputs/outputs per frame. `?agent=0` selects a population member; `max_frames`
  accepts 2–2,000.
- `GET /api/v1/runs/{id}/fitness.png`: downloadable Matplotlib chart, requiring
  at least one saved generation.

The [interactive API reference](http://127.0.0.1:8000/docs) documents validation
and the existing experiment/run endpoints.
