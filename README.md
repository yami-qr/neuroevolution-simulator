# Neuroevolution Simulator

An obstacle-course simulator in development where populations of autonomous
agents evolve neural-network controllers to reach a goal and avoid obstacles.

## Status

A configurable NumPy feed-forward network is implemented and tested. The
simulation and dashboard are planned features.

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
