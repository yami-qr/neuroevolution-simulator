# Project plan

## Agreed direction

Build a 2D obstacle-course neuroevolution simulator. Agents attempt to reach a
goal while avoiding obstacles. Build the project in this repository and make
regular commits. Discuss and agree on implementation scope before coding.

Use a dark lab-style dashboard, as selected during the design discussion.

The project description supplied as a reference names Python, NumPy,
Matplotlib, FastAPI, PostgreSQL, and Docker. These are the intended technologies.
The Python engine now includes the NumPy network, course geometry, agents,
sensors, bounded episode evaluation, fitness scoring, and a seeded genetic
algorithm. FastAPI experiment management, PostgreSQL persistence, and standalone
workers and the dark lab dashboard are implemented. Full application containers
are available through the [complete Compose stack](DOCKER.md).

## Proposed first version

### Simulation and controller

- Use a fixed neural-network architecture with evolvable weights and biases.
- Give each agent inputs describing nearby obstacles, the goal direction, and
  its current movement.
- Convert network outputs into movement commands.
- Run a population through a bounded episode on a shared obstacle course.
- Score goal completion and progress, with penalties for collisions and wasted
  time. Review the fitness design against observed agent behaviour.

### Evolution

- Preserve strong performers through elitism.
- Select parents using fitness, combine their parameters, and apply mutation.
- Record best and average fitness and goal-completion rates per generation.
- Use recorded random seeds so experiments can be reproduced.

### Interface and persistence

- Provide an interactive browser view of agents and the obstacle course.
- Expose experiment configuration and results through FastAPI.
- Store run configurations, generation statistics, and evolved controllers in
  PostgreSQL.
- Export fitness progression charts using Matplotlib.
- Provide Docker Compose configuration for the application and database.

### Dashboard design

The visual direction is agreed: a dark lab-style interface with neutral
surfaces, restrained colour, and a clear highlight for the selected agent.
The simulation arena should be the main focus.

Proposed layout and interactions:

- Top bar: experiment and run selection, status, Start run, and Cancel run.
- Main arena: population, obstacles, goal, optional trails, and sensor rays.
- Right sidebar: experiment settings and a selected-agent inspector.
- Below the arena: best and mean fitness, plus goal-completion rate over time.
- Clicking an agent highlights it and reveals its sensors and movement outputs.
- Follow latest selects each saved generation and its champion automatically.
- A separate Experiments view holds saved runs and results.

Playback speed should preserve simulation behaviour. Experiment configuration
changes should create a new run so results remain interpretable.
Pause and speed apply to recorded replay. Training cancellation follows the
backend's generation-boundary lifecycle.

## Proposed milestones

1. Complete: implement the NumPy network and verify forward propagation and
   parameter serialization. Includes configurable layers and activations,
   seeded initialization, single/batch inference, independent genomes, and
   versioned JSON-compatible snapshots.
2. Complete: implement a fixed slalom course, immutable geometry and agent
   states, fixed-step movement, continuous body collision checks, seven-ray
   observations, ordered checkpoint tracking, and bounded episode evaluation.
   A scripted reference verifies that the default course is traversable.
3. Complete: implement explainable checkpoint fitness, tournament selection,
   uniform crossover, Gaussian mutation, elitism, and an incremental seeded
   generation runner. Three short seeded experiments improved fitness and
   checkpoint progress over their initial populations. A 30-generation seed-19
   run evolved a goal-reaching controller; see [experiment results](EXPERIMENTS.md).
   JSON reports preserve configuration,
   history, champion parameters, and replay trajectories.
4. Complete: add validated experiment management, a durable run queue, generation
   metrics and champion replay endpoints, PostgreSQL JSONB storage, and Alembic
   migrations. Standalone workers use ownership leases, heartbeats, cooperative
   cancellation, and retained results on failure. Local HTTP/worker/restart
   checks and PostgreSQL integration CI verify the full lifecycle.
5. Complete: serve the responsive dark lab dashboard from FastAPI. Experiment
   controls queue and cancel runs; polling updates fitness/completion charts.
   Generation/agent selection, sensor inspection, timed replay, and CSV,
   Matplotlib PNG, and controller JSON exports use persisted data. See the
   [dashboard guide](DASHBOARD.md).
6. Complete: package the API, dashboard, and worker in a non-root image. Compose
   starts PostgreSQL, runs migrations, and gates application startup on readiness.
   Container lifecycle CI verifies execution, exports, and database persistence
   after container recreation. See the [Docker guide](DOCKER.md).

Build each milestone through small, reviewable commits rather than collecting
the entire application into one commit. Each commit should cover one coherent
increment. Run the relevant checks before committing, then push completed
increments regularly to GitHub.

## Decisions to discuss before implementation

- Dashboard uses browser-native modules, canvas, and SVG served by FastAPI.
- Experiment controls and comparisons for the first release. Current engine
  and evolution defaults are documented in the README and can be configured
  per experiment.
- Side-by-side experiment comparisons can follow the single-run inspector.

## Later possibilities

Custom course editing, multiple course presets, and comparisons across several
seeds can follow the first complete simulator.
