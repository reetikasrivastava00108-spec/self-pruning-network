
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
        # 1 means the connection is active.
        # 0 means the connection has been pruned.
        self.mask = np.ones(
            (in_features, out_features),
            dtype=np.float64
        )
        # Attach the mask to the weight Tensor
        self.weights.mask = self.mask

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
        effective_weights = self.weights * Tensor(
            self.mask,
            requires_grad=False
        )

        output = inputs.matmul(effective_weights) + self.bias

        return output
    
    def forward_sparse(self, inputs):
        """
        Run inference using only active weight connections.

        inputs: NumPy array of shape (batch_size, in_features)
        output: NumPy array of shape (batch_size, out_features)
        """

        # Convert inputs to a NumPy array.
        inputs = np.asarray(inputs, dtype=np.float64)

        # Check that the input has the expected shape.
        if inputs.ndim != 2:
            raise ValueError("inputs must be a 2D array.")

        if inputs.shape[1] != self.in_features:
            raise ValueError("Unexpected number of input features.")

        # Start each output with its bias.
        output = np.broadcast_to(
            self.bias.data,
            (inputs.shape[0], self.out_features)
        ).copy()

        # Find only connections that have not been pruned.
        active_inputs, active_outputs = np.nonzero(self.mask)

        # Process each active connection.
        for input_index, output_index in zip(
            active_inputs, active_outputs
        ):
            weight = self.weights.data[input_index, output_index]

            output[:, output_index] += (
                inputs[:, input_index] * weight
            )

        return output

    
    def forward_dense_inference(self, inputs):
        """
        Run dense inference using NumPy matrix multiplication.
        This method does not build an autodiff graph.
        """

        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2:
            raise ValueError("inputs must be a 2D array.")

        if inputs.shape[1] != self.in_features:
            raise ValueError("Unexpected number of input features.")

        # Apply the mask so pruned weights remain zero.
        effective_weights = self.weights.data * self.mask

        # NumPy performs the dense matrix multiplication.
        output = inputs @ effective_weights + self.bias.data

        return output



    def parameters(self):
        # Return all trainable parameters.
        return [self.weights, self.bias]