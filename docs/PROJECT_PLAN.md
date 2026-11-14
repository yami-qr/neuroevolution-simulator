# Project plan

## Agreed direction

Build a 2D obstacle-course neuroevolution simulator. Agents attempt to reach a
goal while avoiding obstacles. Build the project in this repository and make
regular commits. Discuss and agree on implementation scope before coding.

The project description supplied as a reference names Python, NumPy,
Matplotlib, FastAPI, PostgreSQL, and Docker. These are the intended technologies;
no application components have been implemented yet.

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

## Proposed milestones

1. Implement the NumPy network and verify forward propagation and parameter
   serialization.
2. Implement deterministic course geometry, movement, sensors, collisions, and
   episode evaluation.
3. Implement evolution and demonstrate improvement over an initial population
   using repeatable experiments.
4. Add FastAPI experiment management and PostgreSQL persistence.
5. Add the interactive simulation dashboard and result charts.
6. Add Docker Compose and document a complete local run.

Each milestone should produce reviewable commits and appropriate validation.

## Decisions to discuss before implementation

- The first course layout and agent movement model.
- The interface layout and frontend approach.
- Initial population size, network shape, episode length, and fitness formula.
- Which controls and experiment comparisons belong in the first release.

## Later possibilities

Custom course editing, multiple course presets, and comparisons across several
seeds can follow the first complete simulator.
