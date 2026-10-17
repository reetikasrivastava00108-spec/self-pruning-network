
import numpy as np

from tensor import Tensor
from layers import Linear


def test_linear_forward():
    # Set a seed so the test gives repeatable results.
    np.random.seed(42)

    # Create a layer with 2 input features and 3 output features.
    layer = Linear(
        in_features=2,
        out_features=3
    )

    # Create 4 sample inputs, each having 2 features.
    inputs = Tensor(
        np.random.randn(4, 2)
    )

    # Pass the inputs through the layer.
    output = layer.forward(inputs)

    # Check that the output has 4 rows and 3 columns.
    assert output.data.shape == (4, 3)

    # Check that no NaN or infinity values exist.
    assert np.all(np.isfinite(output.data))

    # Check that weights and biases are returned.
    parameters = layer.parameters()

    assert len(parameters) == 2
    assert parameters[0] is layer.weights
    assert parameters[1] is layer.bias

    print("Linear forward test passed!")


if __name__ == "__main__":
    test_linear_forward()
