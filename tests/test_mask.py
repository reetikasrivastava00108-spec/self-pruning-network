
import numpy as np

from tensor import Tensor
from layers import Linear
from optim import Adam


def test_pruned_weight_has_zero_effective_weight():
    layer = Linear(2, 2)

    layer.weights.data[:] = np.array([
        [0.5, 0.8],
        [0.3, 0.6],
    ])

    layer.mask[0, 1] = 0

    effective_weights = layer.weights.data * layer.mask

    assert effective_weights[0, 1] == 0.0


def test_pruned_weight_has_zero_gradient():
    layer = Linear(2, 2)

    layer.weights.data[:] = np.array([
        [0.5, 0.8],
        [0.3, 0.6],
    ])

    layer.mask[0, 1] = 0

    inputs = Tensor(
        np.array([[1.0, 2.0]]),
        requires_grad=False,
    )

    output = layer.forward(inputs)
    output.sum().backward()

    assert layer.weights.grad[0, 1] == 0.0


def test_pruned_weight_and_adam_state_stay_zero():
    layer = Linear(2, 2)

    layer.weights.data[:] = np.array([
        [0.5, 0.8],
        [0.3, 0.6],
    ])

    layer.mask[0, 1] = 0

    optimizer = Adam(
        layer.parameters(),
        learning_rate=0.01,
    )

    weight_index = layer.parameters().index(layer.weights)

    # Seed nonzero optimizer state to simulate a weight
    # that was pruned after previous training steps.
    optimizer.first_moments[weight_index][0, 1] = 0.4
    optimizer.second_moments[weight_index][0, 1] = 0.2

    optimizer.zero_grad()

    # Simulate gradients, including a nonzero gradient
    # at the pruned connection.
    layer.weights.grad[:] = np.ones_like(layer.weights.data)
    layer.bias.grad[:] = np.ones_like(layer.bias.data)

    optimizer.step()

    assert layer.weights.data[0, 1] == 0.0
    assert optimizer.first_moments[weight_index][0, 1] == 0.0
    assert optimizer.second_moments[weight_index][0, 1] == 0.0
