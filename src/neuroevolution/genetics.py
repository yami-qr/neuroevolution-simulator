"""Fitness-based genetic operators using an explicitly supplied random generator."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neuroevolution._validation import nonnegative_float, nonnegative_int
from neuroevolution.network import FeedForwardNetwork


def _probability(value: float, name: str) -> float:
    result = nonnegative_float(value, name)
    if result > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return result


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size == 0 or vector.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a nonempty real numeric vector.")
    vector = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values.")
    return vector


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    population_size: int = 100
    elite_count: int = 5
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_scale: float = 0.2

    def __post_init__(self) -> None:
        for name in ("population_size", "elite_count", "tournament_size"):
            object.__setattr__(self, name, nonnegative_int(getattr(self, name), name))
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot exceed population_size.")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 1 and population_size.")
        for name in ("crossover_rate", "mutation_rate"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        object.__setattr__(
            self,
            "mutation_scale",
            nonnegative_float(self.mutation_scale, "mutation_scale"),
        )


def tournament_select(
    fitness: ArrayLike, rng: np.random.Generator, *, size: int = 3
) -> int:
    """Choose the fittest of k sampled individuals, without replacement.

    Negative and tied scores are valid. Tournament selection does not require
    normalizing scores into probabilities or shifting them above zero.
    """
    scores = _vector(fitness, "Fitness")
    size = nonnegative_int(size, "Tournament size")
    if not 1 <= size <= scores.size:
        raise ValueError("Tournament size must be between 1 and population size.")
    contestants = rng.choice(scores.size, size=size, replace=False)
    return int(contestants[np.argmax(scores[contestants])])


def uniform_crossover(
    first: ArrayLike, second: ArrayLike, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Copy each child gene from either parent with equal probability."""
    a, b = _vector(first, "First parent"), _vector(second, "Second parent")
    if a.shape != b.shape:
        raise ValueError("Parent genomes must have the same shape.")
    return np.where(rng.random(a.size) < 0.5, a, b)


def mutate(
    genome: ArrayLike,
    rng: np.random.Generator,
    *,
    rate: float = 0.1,
    scale: float = 0.2,
) -> NDArray[np.float64]:
    """Return a copy with independent per-gene Gaussian mutations.

    ``rate`` is the mutation probability for each gene; ``scale`` is the noise
    standard deviation. Zero rate or scale returns an unchanged copy.
    """
    child = _vector(genome, "Genome").copy()
    rate, scale = _probability(rate, "Mutation rate"), nonnegative_float(scale, "Scale")
    if rate == 0 or scale == 0:
        return child
    mask = rng.random(child.size) < rate
    try:
        with np.errstate(over="raise", invalid="raise"):
            child[mask] += rng.normal(0.0, scale, size=int(mask.sum()))
    except FloatingPointError as error:
        raise ValueError("Mutation produced non-finite parameters.") from error
    if not np.isfinite(child).all():
        raise ValueError("Mutation produced non-finite parameters.")
    return child


def breed_population(
    population: Sequence[FeedForwardNetwork],
    fitness: ArrayLike,
    rng: np.random.Generator,
    *,
    config: EvolutionConfig,
) -> tuple[FeedForwardNetwork, ...]:
    """Preserve elite copies, then select, cross, and mutate the remaining slots."""
    scores = _vector(fitness, "Fitness")
    if len(population) != config.population_size or scores.size != len(population):
        raise ValueError(
            "Population and fitness must match the configured population size."
        )
    prototype = population[0]
    signature = (
        prototype.layer_sizes,
        prototype.hidden_activation,
        prototype.output_activation,
    )
    if any(
        (network.layer_sizes, network.hidden_activation, network.output_activation)
        != signature
        for network in population
    ):
        raise ValueError("All parents must use the same architecture and activations.")
    ranking = np.argsort(-scores, kind="stable")
    children = [
        population[int(index)].copy() for index in ranking[: config.elite_count]
    ]
    genomes = tuple(network.to_genome() for network in population)
    while len(children) < config.population_size:
        first = tournament_select(scores, rng, size=config.tournament_size)
        genome = genomes[first]
        if rng.random() < config.crossover_rate:
            second = tournament_select(scores, rng, size=config.tournament_size)
            genome = uniform_crossover(genome, genomes[second], rng)
        genome = mutate(
            genome, rng, rate=config.mutation_rate, scale=config.mutation_scale
        )
        children.append(
            FeedForwardNetwork.from_genome(
                prototype.layer_sizes,
                genome,
                hidden_activation=prototype.hidden_activation,
                output_activation=prototype.output_activation,
            )
        )
    return tuple(children)
