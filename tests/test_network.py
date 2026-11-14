"""Numerical and input-contract checks for the neural controller."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuroevolution import FeedForwardNetwork


def test_forward_matches_hand_calculated_relu_network():
    network = FeedForwardNetwork(
        (2, 2, 1), hidden_activation="relu", output_activation="linear"
    )
    network.set_parameters(
        weights=[[[1, -1], [0.5, 2]], [[2], [-1]]],
        biases=[[0.1, -0.2], [0.3]],
    )
    # Hidden values: relu([1.6, -4.2]) = [1.6, 0]. Output: 1.6 * 2 + 0.3.
    assert_allclose(network.forward([2, -1]), [3.5])


def test_tanh_hidden_and_output_layers_match_scalar_reference():
    network = FeedForwardNetwork((2, 2, 1))
    network.set_parameters(
        weights=[[[1, -1], [2, 0.5]], [[0.25], [-0.75]]],
        biases=[[0.5, -0.5], [0.1]],
    )
    expected = math.tanh(0.25 * math.tanh(5.5) - 0.75 * math.tanh(-0.5) + 0.1)
    assert_allclose(network.forward([1, 2]), [expected])


def test_batch_matches_known_individual_results():
    network = FeedForwardNetwork((2, 1), output_activation="linear")
    network.set_parameters(weights=[[[2], [-3]]], biases=[[0.5]])
    assert_allclose(network.forward([[1, 2], [-1, 0], [0, 0]]), [[-3.5], [-1.5], [0.5]])
    assert network.forward([1, 2]).shape == (1,)
    assert network.forward([[1, 2]]).shape == (1, 1)
    assert network.forward(np.empty((0, 2))).shape == (0, 1)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("tanh", [-1, 0, 1]),
        ("sigmoid", [0, 0.5, 1]),
        ("relu", [0, 0, 1000]),
        ("linear", [-1000, 0, 1000]),
    ],
)
def test_activations_handle_large_positive_and_negative_values(activation, expected):
    network = FeedForwardNetwork((1, 1), output_activation=activation)
    network.set_parameters(weights=[[[1]]], biases=[[0]])
    with np.errstate(all="raise"):
        result = network.forward([[-1000], [0], [1000]])
    assert_allclose(result[:, 0], expected)


def test_seed_reproduces_initialization_without_changing_global_rng():
    global_state = np.random.get_state()
    first = FeedForwardNetwork((10, 12, 2), seed=42)
    second = FeedForwardNetwork((10, 12, 2), seed=42)
    different = FeedForwardNetwork((10, 12, 2), seed=43)
    for a, b in zip(first.weights, second.weights, strict=True):
        assert_array_equal(a, b)
    assert not np.array_equal(first.weights[0], different.weights[0])
    current_state = np.random.get_state()
    assert current_state[0] == global_state[0]
    assert_array_equal(current_state[1], global_state[1])
    assert current_state[2:] == global_state[2:]
    assert first.parameter_count == 158


def test_parameter_inputs_and_inspection_arrays_do_not_alias_network():
    network = FeedForwardNetwork((2, 1), output_activation="linear")
    weights = np.array([[2.0], [3.0]])
    biases = np.array([1.0])
    network.set_parameters([weights], [biases])
    weights[:] = 99
    biases[:] = 99
    network.weights[0][:] = 99
    network.biases[0][:] = 99
    assert_array_equal(network.forward([1, 1]), [6])


def test_forward_leaves_inputs_and_parameters_unchanged():
    network = FeedForwardNetwork((2, 3, 1), seed=7)
    inputs = np.array([0.25, -0.5])
    weights, biases = network.weights, network.biases
    first = network.forward(inputs)
    assert_array_equal(network.forward(inputs), first)
    assert_array_equal(inputs, [0.25, -0.5])
    for actual, expected in zip(network.weights, weights, strict=True):
        assert_array_equal(actual, expected)
    for actual, expected in zip(network.biases, biases, strict=True):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("sizes", [(), (2,), (0, 2), (2, -1), (2, 1.5), (True, 2)])
def test_invalid_architectures_are_rejected(sizes):
    with pytest.raises(ValueError):
        FeedForwardNetwork(sizes)


@pytest.mark.parametrize("keyword", ["hidden_activation", "output_activation"])
def test_unknown_activations_are_rejected(keyword):
    with pytest.raises(ValueError, match="activation"):
        FeedForwardNetwork((2, 1), **{keyword: "unknown"})


@pytest.mark.parametrize(
    "inputs", [1, [], [1], [[1, 2, 3]], [[[1, 2]]], [np.nan, 0], [np.inf, 0]]
)
def test_invalid_input_shapes_and_nonfinite_values_are_rejected(inputs):
    with pytest.raises(ValueError):
        FeedForwardNetwork((2, 1)).forward(inputs)


@pytest.mark.parametrize("inputs", [[1 + 2j, 0], ["1", "2"], [None, 0]])
def test_non_real_or_non_numeric_inputs_are_rejected(inputs):
    with pytest.raises(ValueError, match="real numeric"):
        FeedForwardNetwork((2, 1)).forward(inputs)


@pytest.mark.parametrize(
    ("weights", "biases"),
    [
        ([], []),
        ([[[1, 2], [3, 4]]], [[0, 0]]),
        ([[[1], [2]], [[3]]], [[0, 0], [0]]),
        ([[[1, 2], [3, 4]], [[1], [2]]], [[0, 0], [[0]]]),
        ([[[1, 2], [3, 4]], [[1], [np.nan]]], [[0, 0], [0]]),
        ([[[1, 2], [3, 4]], [[1], [2]]], [[0, 0], [np.inf]]),
    ],
)
def test_invalid_parameter_update_leaves_every_layer_unchanged(weights, biases):
    network = FeedForwardNetwork((2, 2, 1), seed=42)
    before = network.forward([0.5, -0.5])
    with pytest.raises(ValueError):
        network.set_parameters(weights, biases)
    assert_array_equal(network.forward([0.5, -0.5]), before)


def test_overflow_raises_a_clear_error():
    network = FeedForwardNetwork((1, 1))
    network.set_parameters(weights=[[[1e308]]], biases=[[0]])
    with pytest.raises(ValueError, match="overflowed"):
        network.forward([1e308])
