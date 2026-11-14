"""Integrated evolution, exact replay, and isolation of external controllers."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuroevolution.agent import AgentConfig
from neuroevolution.course import Circle, Course, slalom_course
from neuroevolution.evolution import EvolutionRunner
from neuroevolution.genetics import EvolutionConfig
from neuroevolution.network import FeedForwardNetwork

CONFIG = EvolutionConfig(population_size=6, elite_count=1)
AGENT = AgentConfig(max_steps=20)


def test_generation_zero_evaluates_initial_population_and_reports_real_metrics():
    runner = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT)
    before = [network.to_dict() for network in runner.population]
    result = runner.step()
    assert result.stats.generation == 0
    assert result.stats.population_size == 6
    assert len(result.episodes) == len(result.fitness) == 6
    assert result.stats.best_fitness == max(result.fitness)
    assert result.stats.mean_fitness == pytest.approx(sum(result.fitness) / 6)
    assert result.stats.completed_count == sum(
        e.final_state.status == "reached" for e in result.episodes
    )
    assert result.stats.completion_rate == result.stats.completed_count / 6
    assert [network.to_dict() for network in runner.population] == before


def test_two_runs_with_same_seed_reproduce_stats_episodes_and_champions():
    first = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT, seed=7)
    second = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT, seed=7)
    for _ in range(3):
        a, b = first.step(), second.step()
        assert a == b
        assert a.champion.to_dict() == b.champion.to_dict()
    assert first.history == second.history
    assert first.champion.to_dict() == second.champion.to_dict()


def test_elitism_keeps_best_fitness_nondecreasing_on_the_same_course():
    runner = EvolutionRunner(
        slalom_course(), config=CONFIG, agent_config=AGENT, seed=19
    )
    history = runner.run(4)
    assert all(
        b.best_fitness >= a.best_fitness
        for a, b in zip(history[:-1], history[1:], strict=True)
    )
    assert [stats.generation for stats in history] == [0, 1, 2, 3]


def test_running_in_chunks_matches_running_all_generations_together():
    first = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT)
    second = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT)
    assert len(first.run(2)) == len(first.run(2)) == 2
    assert first.history == second.run(4)
    assert first.champion.to_dict() == second.champion.to_dict()


def test_population_and_returned_champions_cannot_mutate_the_runner():
    runner = EvolutionRunner(slalom_course(), config=CONFIG, agent_config=AGENT)
    expected = runner.population[0].to_genome()
    outsider = runner.population[0]
    outsider.set_genome(np.zeros(outsider.parameter_count))
    assert_array_equal(runner.population[0].to_genome(), expected)
    result = runner.step()
    expected = runner.champion.to_dict()
    result.champion.set_genome(np.zeros(result.champion.parameter_count))
    runner.champion.set_genome(np.zeros(runner.champion.parameter_count))
    assert runner.champion.to_dict() == result.champion.to_dict() == expected


def test_mutation_can_improve_a_real_episode_from_independent_warm_start_copies():
    course = Course(100, 100, (10, 50), Circle((90, 50), 3))
    network = FeedForwardNetwork.from_genome((10, 2), np.zeros(22))
    initial = [network] * 12
    config = EvolutionConfig(
        population_size=12, elite_count=2, mutation_rate=0.3, mutation_scale=0.2
    )
    runner = EvolutionRunner(
        course,
        config=config,
        seed=13,
        layer_sizes=(10, 2),
        initial_population=initial,
        agent_config=AgentConfig(radius=2, max_speed=100, timestep=0.1, max_steps=10),
    )
    initial[0].set_genome(np.ones(22))
    history = runner.run(5)
    assert history[-1].best_fitness > history[0].best_fitness


def test_full_elitism_is_a_no_evolution_control():
    config = EvolutionConfig(population_size=6, elite_count=6)
    runner = EvolutionRunner(slalom_course(), config=config, agent_config=AGENT)
    first, second = runner.run(2)
    assert first.best_fitness == second.best_fitness
    assert first.mean_fitness == second.mean_fitness


def test_champion_requires_an_evaluated_generation_and_configuration_is_read_only():
    runner = EvolutionRunner(slalom_course(), config=CONFIG)
    assert runner.history == ()
    assert runner.last_result is None
    with pytest.raises(RuntimeError, match="Evaluate"):
        _ = runner.champion
    with pytest.raises(AttributeError):
        runner.seed = 1


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_generation_count_is_rejected(count):
    runner = EvolutionRunner(slalom_course(), config=CONFIG)
    with pytest.raises(ValueError):
        runner.run(count)


@pytest.mark.parametrize(
    "kwargs", [{"seed": -1}, {"layer_sizes": (9, 2)}, {"initial_population": []}]
)
def test_invalid_runner_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EvolutionRunner(slalom_course(), config=CONFIG, **kwargs)
