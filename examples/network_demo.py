"""Run one controller and round-trip it through a JSON-compatible snapshot."""

import json

import numpy as np
from numpy.testing import assert_array_equal

from neuroevolution import FeedForwardNetwork


def main() -> None:
    network = FeedForwardNetwork((10, 12, 2), seed=42)
    # Seven obstacle distances, two local goal coordinates, and current speed.
    observation = np.array([1.0] * 7 + [0.8, 0.2, 0.25])
    outputs = network.forward(observation)

    encoded = json.dumps(network.to_dict(), allow_nan=False)
    restored = FeedForwardNetwork.from_dict(json.loads(encoded))
    assert_array_equal(restored.forward(observation), outputs)

    print(f"Architecture: {network.layer_sizes}")
    print(f"Parameters: {network.parameter_count}")
    print(f"Raw tanh controller outputs: {outputs}")
    print("JSON round-trip preserved the controller's outputs.")


if __name__ == "__main__":
    main()
