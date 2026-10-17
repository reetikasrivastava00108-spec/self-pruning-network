
import numpy as np

from tensor import Tensor
from model import MLP


def test_mlp_forward_and_backward():
    np.random.seed(42)

    # Create a model with 2 input features, 16 hidden neurons,
    # and 3 output classes.
    model = MLP(
        input_size=2,
        hidden_size=16,
        output_size=3
    )

    # Create 4 sample inputs.
    inputs = Tensor(np.random.randn(4, 2))

    # Class labels for the 4 examples.
    targets = np.array([0, 1, 2, 1])

    # Forward pass: obtain class scores.
    logits = model.forward(inputs)

    assert logits.data.shape == (4, 3)

    # Calculate cross-entropy loss.
    loss = logits.cross_entropy(targets)

    assert loss.data.shape == ()
    assert np.isfinite(loss.data)

    # Backward pass: calculate gradients.
    loss.backward()

    # Verify that every trainable parameter has finite gradients.
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert np.all(np.isfinite(parameter.grad))

    print("MLP forward and backward test passed!")


if __name__ == "__main__":
    test_mlp_forward_and_backward()
