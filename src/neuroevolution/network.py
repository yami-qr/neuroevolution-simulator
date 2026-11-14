"""A feed-forward controller implemented with NumPy matrix operations."""

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ActivationName = Literal["tanh", "relu", "sigmoid", "linear"]
FloatArray = NDArray[np.float64]


def _sigmoid(values: FloatArray) -> FloatArray:
    """Evaluate sigmoid without exponentiating large positive numbers."""
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


_ACTIVATIONS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "tanh": np.tanh,
    "relu": lambda values: np.maximum(values, 0.0),
    "sigmoid": _sigmoid,
    "linear": lambda values: values,
}


def _as_finite_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Accept real numeric data, without silently dropping complex components."""
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


class FeedForwardNetwork:
    """A configurable dense network whose parameters will evolve externally.

    ``layer_sizes`` includes the input and output sizes, for example (10, 12, 2).
    Weights have shape (inputs, outputs), so each layer computes
    ``activation(inputs @ weights + biases)``. Hidden and output activations
    default to tanh, giving bounded, signed controller outputs.

    Initialization uses a local random generator, fan-in-scaled normal weights,
    and zero biases. A fixed seed reproduces initialization within the same
    NumPy environment. Forward propagation never changes the parameters.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        seed: int | None = None,
        hidden_activation: ActivationName = "tanh",
        output_activation: ActivationName = "tanh",
    ) -> None:
        sizes = tuple(layer_sizes)
        if len(sizes) < 2:
            raise ValueError("Provide at least an input layer and an output layer.")
        if any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in sizes
        ):
            raise ValueError("Layer sizes must be positive integers.")
        if hidden_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown hidden activation: {hidden_activation!r}.")
        if output_activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown output activation: {output_activation!r}.")

        self._layer_sizes = tuple(int(size) for size in sizes)
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        rng = np.random.default_rng(seed)
        self._weights = tuple(
            rng.normal(0.0, np.sqrt(1.0 / inputs), size=(inputs, outputs))
            for inputs, outputs in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:], strict=True
            )
        )
        self._biases = tuple(np.zeros(size) for size in self.layer_sizes[1:])

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Input, hidden, and output layer widths."""
        return self._layer_sizes

    @property
    def hidden_activation(self) -> ActivationName:
        return self._hidden_activation

    @property
    def output_activation(self) -> ActivationName:
        return self._output_activation

    @property
    def parameter_count(self) -> int:
        """Total number of weights and biases."""
        return sum(
            weights.size + biases.size
            for weights, biases in zip(self._weights, self._biases, strict=True)
        )

    @property
    def weights(self) -> tuple[FloatArray, ...]:
        """Copies of layer weights; use set_parameters to update the network."""
        return tuple(weights.copy() for weights in self._weights)

    @property
    def biases(self) -> tuple[FloatArray, ...]:
        """Copies of layer biases, isolated from the live parameters."""
        return tuple(biases.copy() for biases in self._biases)

    def set_parameters(
        self, weights: Sequence[ArrayLike], biases: Sequence[ArrayLike]
    ) -> None:
        """Replace all layers with validated copies, or leave all unchanged."""
        expected = len(self.layer_sizes) - 1
        if len(weights) != expected or len(biases) != expected:
            raise ValueError(f"Expected weights and biases for {expected} layers.")

        new_weights = []
        new_biases = []
        for index, (inputs, outputs) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        ):
            layer_weights = _as_finite_array(weights[index], name="Weights")
            layer_biases = _as_finite_array(biases[index], name="Biases")
            if layer_weights.shape != (inputs, outputs):
                raise ValueError(
                    f"Layer {index} weights must have shape {(inputs, outputs)}."
                )
            if layer_biases.shape != (outputs,):
                raise ValueError(f"Layer {index} biases must have shape {(outputs,)}.")
            new_weights.append(layer_weights.copy())
            new_biases.append(layer_biases.copy())

        self._weights = tuple(new_weights)
        self._biases = tuple(new_biases)

    def forward(self, inputs: ArrayLike) -> FloatArray:
        """Evaluate one input vector or a batch of vectors using shared weights.

        Shapes (input_size,) and (batch_size, input_size) return (output_size,)
        and (batch_size, output_size), respectively. A batch represents multiple
        observations for this one network, not multiple independent genomes.
        """
        values = _as_finite_array(inputs, name="Inputs")
        if values.ndim not in (1, 2) or values.shape[-1] != self.layer_sizes[0]:
            raise ValueError(
                "Inputs must have shape "
                f"({self.layer_sizes[0]},) or (batch_size, {self.layer_sizes[0]})."
            )

        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                for index, (weights, biases) in enumerate(
                    zip(self._weights, self._biases, strict=True)
                ):
                    activation = (
                        self.output_activation
                        if index == len(self._weights) - 1
                        else self.hidden_activation
                    )
                    values = _ACTIVATIONS[activation](values @ weights + biases)
        except FloatingPointError as error:
            raise ValueError(
                "Forward propagation overflowed; scale inputs or parameters."
            ) from error
        return values
