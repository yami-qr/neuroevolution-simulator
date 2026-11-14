# Project plan

## Agreed direction

Build a 2D obstacle-course neuroevolution simulator. Agents attempt to reach a
goal while avoiding obstacles. Build the project in this repository and make
regular commits. Discuss and agree on implementation scope before coding.

Use a dark lab-style dashboard, as selected during the design discussion.

The project description supplied as a reference names Python, NumPy,
Matplotlib, FastAPI, PostgreSQL, and Docker. These are the intended technologies.
Implementation is starting with the NumPy neural network in a standalone Python
package; the simulation and dashboard will follow in later milestones.

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

- Top bar: experiment name, course, run status, Start/Pause, and playback speed.
- Main arena: population, obstacles, goal, optional trails, and sensor rays.
- Right sidebar: experiment settings and a selected-agent inspector.
- Below the arena: best and mean fitness, plus goal-completion rate over time.
- Clicking an agent highlights it and reveals its sensors and movement outputs.
- A Follow champion option selects the strongest performer automatically.
- A separate Experiments view holds saved runs and results.

Playback speed should preserve simulation behaviour. Experiment configuration
changes should create a new run so results remain interpretable.

## Proposed milestones

1. Complete: implement the NumPy network and verify forward propagation and
   parameter serialization. Includes configurable layers and activations,
   seeded initialization, single/batch inference, independent genomes, and
   versioned JSON-compatible snapshots.
2. Implement deterministic course geometry, movement, sensors, collisions, and
   episode evaluation.
3. Implement evolution and demonstrate improvement over an initial population
   using repeatable experiments.
4. Add FastAPI experiment management and PostgreSQL persistence.
5. Add the interactive simulation dashboard and result charts.
6. Add Docker Compose and document a complete local run.

Build each milestone through small, reviewable commits rather than collecting
the entire application into one commit. Each commit should cover one coherent
increment. Run the relevant checks before committing, then push completed
increments regularly to GitHub.

## Decisions to discuss before implementation

- The first course layout and agent movement model.
- Final dashboard controls, interaction details, and frontend approach.
- Initial population size, network shape, episode length, and fitness formula.
- Which controls and experiment comparisons belong in the first release.

## Later possibilities

Custom course editing, multiple course presets, and comparisons across several
seeds can follow the first complete simulator.
