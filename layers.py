
import numpy as np
from tensor import Tensor


class Linear:
    """
    A fully connected neural network layer.

    Formula:
        output = inputs @ weights + bias
    """

    def __init__(self, in_features, out_features):
        # Save the number of input and output features.
        self.in_features = in_features
        self.out_features = out_features

        # Use He initialization for weights.
        # This is suitable for layers followed by ReLU.
        scale = np.sqrt(2.0 / in_features)

        weights = np.random.randn(
            in_features,
            out_features
        ) * scale

        # Create trainable weights.
        self.weights = Tensor(
            weights,
            requires_grad=True
        )

        # Initialize trainable biases to zero.
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

    def forward(self, inputs):
        """
        Calculate the output of the layer.

        inputs:  (batch_size, in_features)
        weights: (in_features, out_features)
        bias:    (out_features,)

        output:  (batch_size, out_features)
        """

        # Multiply inputs by weights and add bias.
        output = inputs.matmul(self.weights) + self.bias

        return output

    def parameters(self):
        # Return all trainable parameters.
        return [self.weights, self.bias]