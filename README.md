# Neuroevolution Simulator

An obstacle-course simulator in development where populations of autonomous
agents evolve neural-network controllers to reach a goal and avoid obstacles.

## Status

Neural controllers, deterministic course geometry, fixed-step movement, sensors,
bounded controller episodes, fitness scoring, generation evolution, a FastAPI
experiment service, PostgreSQL persistence, and the dark lab dashboard are
implemented and tested. The dashboard includes agent replay, sensor inspection,
fitness/completion charts, and result exports.

The obstacle-course setting and dark lab-style interface are agreed. Remaining
implementation details in the [project plan](docs/PROJECT_PLAN.md) are proposals
to review before development.

## Project scope

- Implement a feed-forward neural network from scratch using NumPy, including
  activation functions, matrix operations, and forward propagation.
- Evolve network parameters using fitness-based selection, crossover, and
  mutation across generations.
- Visualize agents navigating an obstacle course and track fitness progression.
- Use FastAPI and PostgreSQL to configure experiments and persist results.
- Export fitness charts with Matplotlib and package the application with
  Docker Compose.

## Proposed experience

Watch a population attempt an obstacle course, inspect the strongest agent, and
compare performance across generations. Experiment controls and saved results
will help explain how changes to evolution settings affect behaviour.

## Development workflow

Work will be committed and pushed regularly in small, meaningful changes. Each
commit will cover a coherent increment, such as network inference, collision
detection, or a dashboard control. Implementation branches will use the `codex/`
prefix. Relevant checks will run before committing implemented changes.

## Development setup

Use Python 3.11 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS or Linux, use `.venv/bin/python` instead of
`.\.venv\Scripts\python.exe`.

Development checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

GitHub Actions runs these checks and all three examples on every push and pull request.
The matrix covers Windows and Linux, Python 3.11 and 3.13, and NumPy 1.26 and 2.x.

## Neural network

```python
import numpy as np

from neuroevolution import FeedForwardNetwork

network = FeedForwardNetwork((10, 12, 2), seed=42)
actions = network.forward(np.zeros(10))  # One observation: shape (2,).
batch = network.forward(np.zeros((100, 10)))  # Shared network: shape (100, 2).
```

Hidden and output layers default to `tanh`; `relu`, `sigmoid`, and `linear` are
also available. Each layer computes `activation(inputs @ weights + biases)`.
The example controller has 158 evolvable weights and biases. Initialization
uses a local seed; inference leaves the network and inputs unchanged.

The batch evaluates many observations using **one network**. A population of
agents with different genomes will use separate controllers.

### Genomes and snapshots

```python
import json

genome = network.to_genome()  # Independent float64 array, shape (158,).
child = network.copy()
genome[0] += 0.1
child.set_genome(genome)  # The original controller is unchanged.

encoded = json.dumps(child.to_dict(), allow_nan=False)
restored = FeedForwardNetwork.from_dict(json.loads(encoded))
```

The genome order is each layer's weights flattened in row-major order, followed
by that layer's biases. Import and export copy parameters so parent controllers
and offspring cannot accidentally share mutable arrays. Invalid lengths,
shapes, and non-finite parameters are rejected before updating a controller.

Snapshots include a format version, layer sizes, activation names, and the
complete genome. Saving only a seed is not enough to preserve an evolved
controller. Seeds reproduce initialization in the same NumPy environment;
snapshots store the actual parameters.

Run the example:

```powershell
.\.venv\Scripts\python.exe examples/network_demo.py
```

## Obstacle course

```python
from neuroevolution import slalom_course

course = slalom_course()
distance = course.raycast(course.spawn, angle=0, max_distance=200)  # 170 units.
```

The starting arena is 900 by 600 units, with three staggered walls, a goal, and
three ordered checkpoint regions. Coordinates use x to the right and y upward;
headings are in radians, with zero pointing right and positive turns going
counterclockwise. Courses and their geometry are immutable.

Circular bodies collide on contact with walls or arena boundaries. Movement
queries sweep the entire segment, including rounded rectangle corners, so thin
walls cannot be skipped and clear paths beside corners remain usable. Ray
queries return distance from their origin to the nearest wall or boundary,
capped at the requested range.

### Agent movement

```python
from neuroevolution import Action, spawn_agent, step_agent

agent = spawn_agent(course)
agent = step_agent(course, agent, Action(steering=0, throttle=1))
```

Defaults are an 8-unit body radius, a maximum speed of 240 units/second, a
maximum turn rate of pi radians/second, and a fixed 1/60-second timestep. An
attempt lasts at most 900 steps (15 simulated seconds). Throttle immediately
sets forward speed, and agents can turn in place. Agents do not collide with
one another.

`Action` accepts steering in [-1, 1] and throttle in [0, 1].
`Action.from_network(outputs)` maps two signed outputs to these controls,
saturating values outside [-1, 1]. Each step returns an immutable state. Goal
entry, collision, or the time limit ends the attempt; subsequent steps leave a
terminal state unchanged. Contact events are resolved along the movement
segment, with collisions winning ties. Checkpoints count only in course order
along the distance actually travelled before an attempt ends.

### Sensors and controller inputs

```python
from neuroevolution import FeedForwardNetwork, observe, sense

readings = sense(course, agent)  # Raw distances and endpoints for visualization.
observation = observe(course, agent)
controller = FeedForwardNetwork((10, 12, 2), seed=42)
action = Action.from_network(controller.forward(observation))
agent = step_agent(course, agent, action)
```

The default observation has ten values, in this order:

1. Seven wall distances at -90, -60, -30, 0, 30, 60, and 90 degrees relative
   to the agent's heading, from right to left. Rays start at the body center
   and measure up to 200 units; values are divided by this range into [0, 1].
2. Forward and leftward displacement to the final goal, rotated into the
   agent's frame and divided by the arena diagonal into [-1, 1].
3. Current speed divided by maximum speed into [0, 1].

Rays see obstacles and arena boundaries. Agents, the goal, and scoring
checkpoints do not block them. Checkpoints are not navigation inputs. Custom
ray fans are supported; their observation size is the ray count plus three.

### Running an episode

```python
from neuroevolution import run_episode

result = run_episode(course, controller, record=True)
print(result.final_state.status, result.elapsed_seconds)
```

Each tick reads sensors, runs the unchanged controller, and applies its action.
`record=True` retains immutable states for later replay; recording is off by
default. Elapsed time uses whole simulation ticks, including the final contact
tick. Running an episode does not train or mutate the controller.

Run the headless course demonstration:

```powershell
.\.venv\Scripts\python.exe examples/course_demo.py
```

It shows a scripted reference route through all checkpoints, then evaluates an
untrained seeded network. The reference demonstrates that the course is
traversable; it is not a learned solution. The separate evolution runner breeds
neural controllers using these same episodes.

## Fitness scoring

`score_episode(course, result)` returns an explainable `FitnessBreakdown`.
The default score is `1000 * progress + completion_bonus - time_cost - crash_cost`.

- Progress divides the route into ordered checkpoint stages and a final goal
  stage. Partial progress measures closeness to the next region, relative to
  spawn or the previous checkpoint, clipped to [0, 1]. Reaching the goal sets
  full progress, including valid alternate routes that bypass checkpoints.
- Reaching the goal adds 1000 points. With defaults, any completion outranks
  any failed attempt, and faster completions score higher.
- Time costs up to 10 points over the episode limit; crashing costs 25 points.
- Survival and total distance travelled do not earn points.

`FitnessConfig` exposes the four weights for experiments. Changed weights can
change the relative ranking of success and failure. Pass the same `AgentConfig`
used for the episode so tick limits and elapsed time match.

## Genetic operators

`breed_population(population, fitness, rng, config=EvolutionConfig())` preserves
elite controller copies and creates the remaining offspring from selected
parents. Default settings are 100 individuals, five elites, tournaments of
three sampled without replacement, an 80% crossover probability per child,
and independent Gaussian mutation on 10% of genes with standard deviation 0.2.
Uniform crossover chooses each gene from either parent with equal probability.

All parents must share architecture and activation names. Negative and tied
fitness scores are supported. Elites use stable ranking, and parent arrays
are never changed or shared with offspring. Every random operator requires
an explicit NumPy `Generator`; no global random state is used.

## Evolving controllers

```python
from neuroevolution import EvolutionConfig, EvolutionRunner, slalom_course

runner = EvolutionRunner(
    slalom_course(),
    config=EvolutionConfig(population_size=24, elite_count=2),
    seed=42,
)
for _ in range(5):
    result = runner.step()
    print(
        result.stats.generation,
        result.stats.best_fitness,
        result.stats.mean_fitness,
        result.stats.completion_rate,
    )
champion = runner.champion
```

Generation zero evaluates the initial population; each subsequent step breeds
and evaluates one new generation. Settings stay fixed for a run, and each
individual receives the same course and episode limit. Population accessors
and champions return independent controller copies. An optional
`initial_population` supports controlled comparisons or warm starts.

`runner.history` retains generation statistics. `runner.last_result` includes
per-agent fitness breakdowns and final episode states. Its champion belongs to
that generation; `runner.champion` is the best controller seen across the run.
Elitism preserves best fitness on this deterministic course, while average
fitness and completion rate can fluctuate. `runner.run(n)` evaluates n more
generations, preserving the same seeded sequence as n calls to `step()`.

Reproducing a run requires the same seed, course, initial population settings,
configuration, and NumPy environment. Controller snapshots preserve the actual
evolved parameters for replay without rerunning evolution.

### Evolution demo and reports

```powershell
.\.venv\Scripts\python.exe examples/evolution_demo.py --generations 10 --population 24 --seed 42 --output artifacts/evolution-42.json
```

The demo uses the real slalom course and prints each generation's metrics. Its
smaller default population of 24 keeps experiments manageable; the engine
default remains 100. Generation zero is the baseline initial population, so
ten evaluated generations contain nine breeding cycles. Mutation rate, noise
scale, and episode step limit are configurable through the demo's arguments.

The optional JSON report records the seed, Python and NumPy versions, complete
course and configuration, generation history, champion controller snapshot,
fitness breakdown, and replay trajectory. The demo replays the champion and
checks that its score matches the recorded best. Reports under `artifacts/`
are ignored by Git. A short run can improve fitness without solving the course;
check the recorded completion counts as well as the score.

See [seeded experiment results](docs/EXPERIMENTS.md) for measured baseline
comparisons and reproduction commands. In the recorded 30-generation seed-19
run, the first successful controller appeared in generation 20 and the final
champion reached the goal in 9.60 simulated seconds.

## Experiment API

The backend stores experiment configurations, queued execution runs, per-generation
metrics, evolved controllers, and replay trajectories. A separate worker runs
evolution while the API handles requests. Run state and completed generations
survive API restarts; worker leases retain results after interrupted execution.

See the [API and PostgreSQL guide](docs/API.md) for startup commands, endpoint
details, worker operation, and a local SQLite development option. A
database-only [Compose file](compose.postgres.yml) provisions PostgreSQL 17;
full application containerization follows in a later milestone.

Backend tests run against both isolated SQLite databases and a real PostgreSQL
service in GitHub Actions.

## Dashboard

Start the API and worker together with an explicit local development database:

```powershell
.\.venv\Scripts\python.exe -m neuroevolution.server.dev --sqlite
```

Open [Evolution lab](http://127.0.0.1:8000/). For PostgreSQL, configure
`DATABASE_URL` and omit `--sqlite`. The launcher applies migrations and retains
saved results between restarts. See the [dashboard guide](docs/DASHBOARD.md) for
controls, exports, replay limits, and frontend checks. No Node.js runtime or
frontend build is needed to use the dashboard.
