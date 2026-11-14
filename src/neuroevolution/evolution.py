"""Incremental population evolution over deterministic course episodes."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from neuroevolution._validation import nonnegative_int
from neuroevolution.agent import AgentConfig
from neuroevolution.course import Course
from neuroevolution.episode import EpisodeResult, run_episode
from neuroevolution.fitness import FitnessBreakdown, FitnessConfig, score_episode
from neuroevolution.genetics import EvolutionConfig, breed_population
from neuroevolution.network import ActivationName, FeedForwardNetwork
from neuroevolution.sensors import SensorConfig

_DEFAULT_CONFIG = EvolutionConfig()
_DEFAULT_AGENT_CONFIG = AgentConfig()
_DEFAULT_SENSOR_CONFIG = SensorConfig()
_DEFAULT_FITNESS_CONFIG = FitnessConfig()


@dataclass(frozen=True, slots=True)
class GenerationStats:
    generation: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    completed_count: int
    best_progress: float
    best_checkpoints: int

    @property
    def completion_rate(self) -> float:
        return self.completed_count / self.population_size


@dataclass(frozen=True, slots=True)
class GenerationResult:
    stats: GenerationStats
    scores: tuple[FitnessBreakdown, ...]
    episodes: tuple[EpisodeResult, ...]
    _champion: FeedForwardNetwork = field(repr=False, compare=False)

    @property
    def fitness(self) -> tuple[float, ...]:
        return tuple(score.total for score in self.scores)

    @property
    def champion(self) -> FeedForwardNetwork:
        """An independent copy of this generation's strongest controller."""
        return self._champion.copy()


class EvolutionRunner:
    """Evaluate one generation per step and breed before the next evaluation.

    Generation zero is the unchanged initial population. Configuration stays
    fixed for the run. Public population and champion accessors return copies;
    histories retain compact statistics rather than all generation trajectories.
    """

    def __init__(
        self,
        course: Course,
        *,
        config: EvolutionConfig = _DEFAULT_CONFIG,
        agent_config: AgentConfig = _DEFAULT_AGENT_CONFIG,
        sensor_config: SensorConfig = _DEFAULT_SENSOR_CONFIG,
        fitness_config: FitnessConfig = _DEFAULT_FITNESS_CONFIG,
        layer_sizes: Sequence[int] = (10, 12, 2),
        hidden_activation: ActivationName = "tanh",
        output_activation: ActivationName = "tanh",
        seed: int = 42,
        initial_population: Sequence[FeedForwardNetwork] | None = None,
    ) -> None:
        self._seed = nonnegative_int(seed, "Seed")
        prototype = FeedForwardNetwork(
            layer_sizes,
            seed=0,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        if (
            prototype.layer_sizes[0] != len(sensor_config.ray_angles) + 3
            or prototype.layer_sizes[-1] != 2
        ):
            raise ValueError(
                "Controller architecture must match the sensors and 2 outputs."
            )
        self._course, self._config = course, config
        self._agent_config, self._sensor_config = agent_config, sensor_config
        self._fitness_config = fitness_config
        self._rng = np.random.default_rng(self._seed)
        if initial_population is None:
            seeds = self._rng.integers(0, 2**32, size=config.population_size)
            self._population = tuple(
                FeedForwardNetwork(
                    prototype.layer_sizes,
                    seed=int(network_seed),
                    hidden_activation=hidden_activation,
                    output_activation=output_activation,
                )
                for network_seed in seeds
            )
        else:
            if len(initial_population) != config.population_size:
                raise ValueError("Initial population must match population_size.")
            signature = (prototype.layer_sizes, hidden_activation, output_activation)
            if any(
                (
                    network.layer_sizes,
                    network.hidden_activation,
                    network.output_activation,
                )
                != signature
                for network in initial_population
            ):
                raise ValueError(
                    "Initial controllers must match the configured architecture."
                )
            self._population = tuple(network.copy() for network in initial_population)
        self._history: list[GenerationStats] = []
        self._last_result: GenerationResult | None = None
        self._champion: FeedForwardNetwork | None = None
        self._best_fitness = -math.inf

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def course(self) -> Course:
        return self._course

    @property
    def config(self) -> EvolutionConfig:
        return self._config

    @property
    def agent_config(self) -> AgentConfig:
        return self._agent_config

    @property
    def sensor_config(self) -> SensorConfig:
        return self._sensor_config

    @property
    def fitness_config(self) -> FitnessConfig:
        return self._fitness_config

    @property
    def population(self) -> tuple[FeedForwardNetwork, ...]:
        return tuple(network.copy() for network in self._population)

    @property
    def history(self) -> tuple[GenerationStats, ...]:
        return tuple(self._history)

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    @property
    def champion(self) -> FeedForwardNetwork:
        if self._champion is None:
            raise RuntimeError("Evaluate a generation before requesting its champion.")
        return self._champion.copy()

    def step(self) -> GenerationResult:
        """Evaluate the next generation without modifying any parent controller."""
        population = self._population
        if self._last_result is not None:
            population = breed_population(
                population, self._last_result.fitness, self._rng, config=self.config
            )
        episodes = tuple(
            run_episode(
                self.course,
                controller,
                agent_config=self.agent_config,
                sensor_config=self.sensor_config,
            )
            for controller in population
        )
        scores = tuple(
            score_episode(
                self.course,
                episode,
                agent_config=self.agent_config,
                config=self.fitness_config,
            )
            for episode in episodes
        )
        best_index = max(range(len(scores)), key=lambda i: scores[i].total)
        best = scores[best_index]
        stats = GenerationStats(
            generation=len(self._history),
            population_size=len(population),
            best_fitness=best.total,
            mean_fitness=math.fsum(score.total / len(scores) for score in scores),
            completed_count=sum(
                episode.final_state.status == "reached" for episode in episodes
            ),
            best_progress=best.progress,
            best_checkpoints=episodes[best_index].final_state.checkpoints_passed,
        )
        result = GenerationResult(
            stats, scores, episodes, population[best_index].copy()
        )
        if best.total > self._best_fitness:
            self._champion = population[best_index].copy()
            self._best_fitness = best.total
        self._population = population
        self._last_result = result
        self._history.append(stats)
        return result

    def run(self, generations: int) -> tuple[GenerationStats, ...]:
        """Evaluate a positive number of generations, retaining prior history."""
        count = nonnegative_int(generations, "Generation count")
        if count == 0:
            raise ValueError("Generation count must be positive.")
        start = len(self._history)
        for _ in range(count):
            self.step()
        return self.history[start:]
