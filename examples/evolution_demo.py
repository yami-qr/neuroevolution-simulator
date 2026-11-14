"""Evolve controllers, replay the champion, and optionally save a run report."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from neuroevolution import (
    AgentConfig,
    EvolutionConfig,
    EvolutionRunner,
    run_episode,
    score_episode,
    slalom_course,
)


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=positive_integer,
        default=10,
        help="Evaluated generations, including the initial population.",
    )
    parser.add_argument("--population", type=positive_integer, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=positive_integer, default=900)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-scale", type=float, default=0.2)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    try:
        config = EvolutionConfig(
            population_size=args.population,
            elite_count=max(1, args.population // 20),
            tournament_size=min(3, args.population),
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
        )
        runner = EvolutionRunner(
            slalom_course(),
            config=config,
            seed=args.seed,
            agent_config=AgentConfig(max_steps=args.max_steps),
        )
    except ValueError as error:
        parser.error(str(error))

    print(
        f"Seed {runner.seed}; population {config.population_size}; "
        f"{args.generations} evaluated generations",
        flush=True,
    )
    for _ in range(args.generations):
        stats = runner.step().stats
        print(
            f"Generation {stats.generation:3d}: best={stats.best_fitness:8.2f} "
            f"mean={stats.mean_fitness:8.2f} reached={stats.completed_count:3d} "
            f"progress={stats.best_progress:.1%} "
            f"checkpoints={stats.best_checkpoints}/3",
            flush=True,
        )

    champion = runner.champion
    replay = run_episode(
        runner.course,
        champion,
        agent_config=runner.agent_config,
        sensor_config=runner.sensor_config,
        record=args.output is not None,
    )
    fitness = score_episode(
        runner.course,
        replay,
        agent_config=runner.agent_config,
        config=runner.fitness_config,
    )
    if fitness.total != max(stats.best_fitness for stats in runner.history):
        raise RuntimeError("Champion replay did not reproduce its measured fitness.")
    print(
        f"Initial best: {runner.history[0].best_fitness:.2f}; "
        f"final best: {runner.history[-1].best_fitness:.2f}"
    )
    print(
        f"Champion replay: {replay.final_state.status}, "
        f"{replay.elapsed_seconds:.2f}s, "
        f"{replay.final_state.checkpoints_passed}/3 checkpoints"
    )
    if args.output is not None:
        report = {
            "format_version": 1,
            "seed": runner.seed,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "course": asdict(runner.course),
            "evolution": asdict(runner.config),
            "agent": asdict(runner.agent_config),
            "sensors": asdict(runner.sensor_config),
            "fitness_config": asdict(runner.fitness_config),
            "history": [
                {**asdict(stats), "completion_rate": stats.completion_rate}
                for stats in runner.history
            ],
            "champion": {
                "controller": champion.to_dict(),
                "score": {**asdict(fitness), "total": fitness.total},
                "final_state": asdict(replay.final_state),
                "elapsed_seconds": replay.elapsed_seconds,
                "trajectory": [asdict(state) for state in replay.trajectory],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
