# Seeded evolution experiments

## Setup

These experiments were run on 14 November 2026 using Python 3.11.9, NumPy 2.4.6,
and implementation commit `f1b733918fc6339f46016dd3fbc074079764f32b`.

- Course: the default 900 x 600 slalom, three ordered checkpoints.
- Controller: 10 inputs, 12 hidden neurons, 2 outputs; tanh activations.
- Population: 24; one elite; tournament size 3.
- Crossover probability: 0.8; per-gene mutation probability: 0.1; Gaussian
  mutation standard deviation: 0.2.
- Episode limit: 900 fixed steps at 60 Hz, or 15 simulated seconds.
- Fitness: default progress weight 1000, completion bonus 1000, maximum time
  cost 10, collision cost 25.
- No scripted controllers or warm starts were supplied to evolution.

## Ten evaluated generations

Generation zero evaluates the initial population, followed by nine breeding
cycles. Progress below is the scoring fraction across checkpoint and goal
stages, rather than a physical distance percentage.

| Seed | Initial best fitness | Final best fitness | Initial mean | Final mean | Best progress, initial to final | Champion checkpoints | Final goal completions |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 7 | 223.20 | 411.34 | 28.94 | 166.93 | 25.0% to 44.2% | 1 / 3 | 0 / 24 |
| 19 | 223.66 | 581.78 | 50.52 | 178.08 | 25.0% to 59.2% | 2 / 3 | 0 / 24 |
| 42 | 324.05 | 506.43 | 88.40 | 188.32 | 35.4% to 51.6% | 2 / 3 | 0 / 24 |

All three runs improved both best and mean fitness, with improved checkpoint
stage progress. None completed the course within these short runs. Each saved
champion was replayed and reproduced its measured fitness exactly in this
environment.

## Longer seed-19 run

Extending the same seed and settings to 30 evaluated generations produced the
first goal-reaching controller in generation 20. That controller passed all
three checkpoint regions. The final population in generation 29 contained
four goal-reaching agents out of 24.

| Metric | Initial population | Generation 29 |
| --- | ---: | ---: |
| Best fitness | 223.66 | 1993.60 |
| Mean fitness | 50.52 | 598.57 |
| Goal completions | 0 / 24 | 4 / 24 |

The final champion replay reached the goal in 576 ticks, or 9.60 simulated
seconds. It used a faster valid route outside the circular checkpoint regions.
Checkpoints provide intermediate fitness shaping; successful goal completion
also rewards valid alternate routes. The champion's saved score was reproduced
exactly on replay.

```powershell
.\.venv\Scripts\python.exe examples/evolution_demo.py --generations 30 --population 24 --seed 19 --output artifacts/evolution-19-30.json
```

## Reproduction

From the repository root in PowerShell, after installing development dependencies:

```powershell
foreach ($experimentSeed in 7, 19, 42) {
    .\.venv\Scripts\python.exe examples/evolution_demo.py --generations 10 --population 24 --seed $experimentSeed --output "artifacts/evolution-$experimentSeed.json"
}
```

The reports contain the course, settings, environment versions, generation
history, controller parameters, fitness breakdown, and champion trajectory.
Generated reports stay in the ignored `artifacts/` directory. Reproducing exact
numbers requires the same implementation and NumPy environment.
