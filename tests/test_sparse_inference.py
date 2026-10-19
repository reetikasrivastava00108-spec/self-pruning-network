
import numpy as np

from tensor import Tensor
from layers import Linear
from model import MLP


def test_sparse_linear_matches_dense_forward():
    np.random.seed(42)

    layer = Linear(3, 2)

    # Prune two connections.
    layer.mask[0, 1] = 0.0
    layer.mask[2, 0] = 0.0

    inputs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    # Existing dense forward path.
    dense_output = layer.forward(Tensor(inputs)).data

    # New sparse-aware forward path.
    sparse_output = layer.forward_sparse(inputs)

    np.testing.assert_allclose(
        sparse_output,
        dense_output,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sparse_mlp_matches_dense_forward():
    np.random.seed(42)

    model = MLP(
        input_size=2,
        hidden_size=4,
        output_size=3,
    )

    # Prune some connections in both layers.
    model.layer1.mask[0, 1] = 0.0
    model.layer1.mask[1, 2] = 0.0
    model.layer2.mask[0, 1] = 0.0
    model.layer2.mask[2, 2] = 0.0

    inputs = np.array([
        [0.2, 0.8],
        [-0.4, 0.5],
        [1.0, -0.3],
    ])

    # Existing dense model forward.
    dense_output = model.forward(Tensor(inputs)).data

    # New sparse model forward.
    sparse_output = model.forward_sparse(inputs)

    np.testing.assert_allclose(
        sparse_output,
        dense_output,
        rtol=1e-10,
        atol=1e-10,
    )
