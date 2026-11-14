"""Genetic operator invariants, parent ownership, and seeded reproduction."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuroevolution.genetics import (
    EvolutionConfig,
    breed_population,
    mutate,
    tournament_select,
    uniform_crossover,
)
from neuroevolution.network import FeedForwardNetwork


def parents():
    return tuple(FeedForwardNetwork.from_genome((1, 1), [i, i + 0.1]) for i in range(4))


def test_tournament_selects_best_from_full_sample_even_with_negative_scores():
    assert tournament_select([-100, -1, -10], np.random.default_rng(42), size=3) == 1


def test_tied_tournaments_are_reproducible_and_return_valid_indices():
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    first = [tournament_select([0, 0, 0, 0], a) for _ in range(10)]
    assert first == [tournament_select([0, 0, 0, 0], b) for _ in range(10)]
    assert all(0 <= index < 4 for index in first)


def test_crossover_copies_only_parent_genes_without_modifying_parents():
    a, b = np.zeros(50), np.ones(50)
    child = uniform_crossover(a, b, np.random.default_rng(42))
    assert np.isin(child, [0, 1]).all()
    assert (child == 0).any() and (child == 1).any()
    child[:] = 99
    assert_array_equal(a, np.zeros(50))
    assert_array_equal(b, np.ones(50))


@pytest.mark.parametrize(("rate", "scale"), [(0, 1), (1, 0)])
def test_disabled_mutation_still_returns_independent_genome(rate, scale):
    parent = np.array([1.0, 2.0, 3.0])
    child = mutate(parent, np.random.default_rng(42), rate=rate, scale=scale)
    assert_array_equal(child, parent)
    child[:] = 99
    assert_array_equal(parent, [1, 2, 3])


def test_full_mutation_is_seeded_finite_and_does_not_change_parent():
    parent = np.ones(20)
    first = mutate(parent, np.random.default_rng(42), rate=1, scale=0.2)
    second = mutate(parent, np.random.default_rng(42), rate=1, scale=0.2)
    assert_array_equal(first, second)
    assert not np.array_equal(first, parent)
    assert np.isfinite(first).all()
    assert_array_equal(parent, np.ones(20))


def test_elites_preserve_top_parameters_and_stable_tie_order():
    population = parents()
    config = EvolutionConfig(population_size=4, elite_count=2)
    children = breed_population(
        population, [-5, 10, 10, 0], np.random.default_rng(42), config=config
    )
    assert len(children) == 4
    assert_array_equal(children[0].to_genome(), population[1].to_genome())
    assert_array_equal(children[1].to_genome(), population[2].to_genome())
    children[0].set_genome([99, 99])
    assert_array_equal(population[1].to_genome(), [1, 1.1])
    assert_array_equal(children[1].to_genome(), [2, 2.1])


def test_disabled_variation_produces_only_independent_parent_copies():
    population = parents()
    config = EvolutionConfig(
        population_size=4, elite_count=1, crossover_rate=0, mutation_rate=0
    )
    children = breed_population(
        population, [0, 1, 2, 3], np.random.default_rng(42), config=config
    )
    for child in children:
        assert all(child is not parent for parent in population)
        assert any(np.array_equal(child.to_genome(), p.to_genome()) for p in population)


def test_complete_breeding_is_reproducible_and_does_not_modify_any_parent():
    population = parents()
    config = EvolutionConfig(population_size=4, elite_count=1)
    before = [p.to_dict() for p in population]
    first = breed_population(
        population, [-5, 0, 2, 10], np.random.default_rng(42), config=config
    )
    second = breed_population(
        population, [-5, 0, 2, 10], np.random.default_rng(42), config=config
    )
    assert [child.to_dict() for child in first] == [child.to_dict() for child in second]
    assert [p.to_dict() for p in population] == before


def test_breeding_rejects_incompatible_parent_activations():
    population = list(parents())
    population[1] = FeedForwardNetwork((1, 1), output_activation="sigmoid")
    with pytest.raises(ValueError, match="architecture and activations"):
        breed_population(
            population,
            [0, 1, 2, 3],
            np.random.default_rng(42),
            config=EvolutionConfig(population_size=4, elite_count=1),
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda r: tournament_select([], r),
        lambda r: tournament_select([0, math.nan], r),
        lambda r: tournament_select([0, 1], r, size=3),
        lambda r: uniform_crossover([0, 1], [0], r),
        lambda r: mutate([math.inf], r),
        lambda r: mutate([0], r, rate=2),
        lambda r: mutate([0], r, scale=-1),
        lambda r: breed_population(
            parents(),
            [0, 1],
            r,
            config=EvolutionConfig(population_size=4, elite_count=1),
        ),
    ],
)
def test_invalid_operator_inputs_are_rejected(operation):
    with pytest.raises(ValueError):
        operation(np.random.default_rng(42))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"population_size": True},
        {"elite_count": 101},
        {"tournament_size": 0},
        {"tournament_size": 101},
        {"mutation_rate": -0.1},
        {"mutation_rate": 1.1},
        {"crossover_rate": math.nan},
        {"mutation_scale": math.inf},
    ],
)
def test_invalid_evolution_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EvolutionConfig(**kwargs)
