# Neuroevolution Simulator

A planned interactive obstacle-course simulator where populations of autonomous
agents evolve neural-network controllers to reach a goal and avoid obstacles.

## Status

Project planning and repository setup. The application has not been implemented.

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

Setup and run instructions will be added when the first runnable version exists.
