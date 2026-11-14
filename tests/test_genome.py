"""Genome layout, ownership, and controller snapshot contracts."""

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuroevolution import FeedForwardNetwork


def test_genome_layout_is_row_major_weights_then_biases_for_each_layer():
    network = FeedForwardNetwork((2, 2, 1), output_activation="linear")
    network.set_parameters(weights=[[[1, 2], [3, 4]], [[7], [8]]], biases=[[5, 6], [9]])
    assert_array_equal(network.to_genome(), np.arange(1, 10))


def test_genome_import_restores_known_weights_and_output():
    network = FeedForwardNetwork.from_genome(
        (2, 2, 1),
        np.arange(1, 10),
        hidden_activation="linear",
        output_activation="linear",
    )
    assert_array_equal(network.weights[0], [[1, 2], [3, 4]])
    assert_array_equal(network.biases[0], [5, 6])
    assert_array_equal(network.weights[1], [[7], [8]])
    assert_array_equal(network.biases[1], [9])
    # Hidden: [1 + 3 + 5, 2 + 4 + 6] = [9, 12]. Output: 9*7 + 12*8 + 9.
    assert_array_equal(network.forward([1, 1]), [168])


def test_exported_genome_is_independent_from_network():
    network = FeedForwardNetwork((2, 3, 1), seed=42)
    before = network.forward([0.1, 0.2])
    genome = network.to_genome()
    genome[:] = 0
    assert_array_equal(network.forward([0.1, 0.2]), before)


def test_imported_genome_is_copied_and_updates_weights_and_biases():
    genome = np.array([2.0, 3.0])
    network = FeedForwardNetwork.from_genome((1, 1), genome, output_activation="linear")
    genome[:] = 99
    assert_array_equal(network.forward([2]), [7])
    network.set_genome([4, 5])
    assert_array_equal(network.forward([2]), [13])


def test_copy_preserves_configuration_and_can_change_without_affecting_parent():
    parent = FeedForwardNetwork(
        (2, 3, 1), seed=42, hidden_activation="relu", output_activation="sigmoid"
    )
    child = parent.copy()
    assert child.to_dict() == parent.to_dict()
    child.set_genome(np.zeros(child.parameter_count))
    assert not np.array_equal(child.to_genome(), parent.to_genome())
    assert_array_equal(child.forward([1, 1]), [0.5])


@pytest.mark.parametrize(
    "genome",
    [[], [1], [1, 2, 3], [[1, 2]], [1, np.nan], [1, np.inf], [1 + 2j, 0]],
)
def test_invalid_genome_does_not_change_existing_controller(genome):
    network = FeedForwardNetwork((1, 1), seed=42)
    before = network.to_genome()
    with pytest.raises(ValueError):
        network.set_genome(genome)
    assert_array_equal(network.to_genome(), before)


@pytest.mark.parametrize("sizes", [(3, 2), (3, 4, 2), (3, 5, 4, 2)])
def test_json_roundtrip_preserves_all_parameters_configuration_and_outputs(sizes):
    original = FeedForwardNetwork(
        sizes, seed=19, hidden_activation="relu", output_activation="sigmoid"
    )
    snapshot = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    restored = FeedForwardNetwork.from_dict(snapshot)
    assert restored.to_dict() == original.to_dict()
    inputs = [[0.25, -0.5, 0.75], [1, 0, -1]]
    assert_array_equal(restored.forward(inputs), original.forward(inputs))
    snapshot["genome"][0] = 12345
    assert_array_equal(restored.to_genome(), original.to_genome())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("format_version", 2),
        ("format_version", True),
        ("layer_sizes", "2,3,1"),
        ("layer_sizes", [2, 0, 1]),
        ("hidden_activation", "unknown"),
        ("output_activation", []),
        ("genome", []),
        ("genome", "invalid"),
        ("genome", [np.nan] * 13),
    ],
)
def test_invalid_snapshots_are_rejected(key, value):
    snapshot = FeedForwardNetwork((2, 3, 1), seed=42).to_dict()
    snapshot[key] = value
    with pytest.raises(ValueError):
        FeedForwardNetwork.from_dict(snapshot)


@pytest.mark.parametrize("state", [{}, None, [], {"format_version": 1}])
def test_missing_or_non_mapping_snapshots_are_rejected(state):
    with pytest.raises(ValueError, match="required fields"):
        FeedForwardNetwork.from_dict(state)
