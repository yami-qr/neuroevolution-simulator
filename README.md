# Neuroevolution Simulator

An obstacle-course simulator in development where populations of autonomous
agents evolve neural-network controllers to reach a goal and avoid obstacles.

## Status

Implementation has started with the Python package foundation. The simulation
and dashboard are planned features.

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

Development checks, once tests have been added:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```
