
import numpy as np

from tensor import Tensor
from optim import Adam


def test_adam_updates_parameter():
    # Create a trainable scalar parameter.
    weight = Tensor([2.0], requires_grad=True)

    # Create Adam for this parameter.
    optimizer = Adam(
        [weight],
        learning_rate=0.01
    )

    initial_value = weight.data.copy()

    # Test a simple loss: loss = weight * weight.
    loss = (weight * weight).sum()

    # Calculate the gradient.
    loss.backward()

    # For loss = weight^2, gradient = 2 * weight = 4.
    np.testing.assert_allclose(
        weight.grad,
        np.array([4.0])
    )

    # Update the parameter.
    optimizer.step()

    # The positive gradient should reduce this positive weight.
    assert weight.data[0] < initial_value[0]

    # Check that the updated parameter is finite.
    assert np.all(np.isfinite(weight.data))

    # Check that Adam has performed one update.
    assert optimizer.step_count == 1

    print("Adam parameter update test passed!")


if __name__ == "__main__":
    test_adam_updates_parameter()
