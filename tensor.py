

import numpy as np


def _unbroadcast(grad, original_shape):
    # Remove extra leading dimensions.
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # Sum over dimensions that were originally size 1.
    for axis, size in enumerate(original_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(original_shape)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad

        self.grad = np.zeros_like(self.data) if requires_grad else None

        self.parents = tuple(parents)
        self._backward = lambda: None

    
    def backward(self):
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a tensor "
                "that does not require gradients."
            )

        if self.data.size != 1:
            raise RuntimeError(
                "backward() requires a scalar output."
            )

        topo_order = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)

                for parent in tensor.parents:
                    build_graph(parent)

                topo_order.append(tensor)

        build_graph(self)

        # Reset intermediate gradients for this backward pass.
        for tensor in topo_order:
            if tensor.parents and tensor.requires_grad:
                tensor.grad = np.zeros_like(tensor.data)

        # The derivative of the scalar output with respect to itself is 1.
        self.grad = np.ones_like(self.data)

        # Propagate gradients in reverse topological order.
        for tensor in reversed(topo_order):
            tensor._backward()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other)
        )

        
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad, self.data.shape
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    result.grad, other.data.shape
                )
        result._backward = _backward

        return result

    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other)
        )

        
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    other.data * result.grad,
                    self.data.shape
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    self.data * result.grad,
                    other.data.shape
                )


        result._backward = _backward

        return result

    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other)
        )

        
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad, self.data.shape
                )

            if other.requires_grad:
                other.grad -= _unbroadcast(
                    result.grad, other.data.shape
                )
        result._backward = _backward

        return result

    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other)
        )

        
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    result.grad / other.data,
                    self.data.shape
                )

            if other.requires_grad:
                other.grad += _unbroadcast(
                    -(self.data / (other.data ** 2))
                    * result.grad,
                    other.data.shape
                )
        result._backward = _backward

        return result

    
    def sum(self):
        result = Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * result.grad

        result._backward = _backward

        return result

    
    def mean(self):
        result = Tensor(
            np.mean(self.data),
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            if self.requires_grad:
                self.grad += (
                    np.ones_like(self.data)
                    * result.grad
                    / self.data.size
                )

        result._backward = _backward

        return result

    
    def relu(self):
        result = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            if self.requires_grad:
                self.grad += (
                    (self.data > 0) * result.grad
                )

        result._backward = _backward

        return result

    
    def tanh(self):
        result = Tensor(
            np.tanh(self.data),
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            if self.requires_grad:
                self.grad += (
                    (1 - result.data ** 2) * result.grad
                )

        result._backward = _backward

        return result
    
    def softmax(self, axis=-1):
        # Subtract the maximum value to avoid numerical overflow.
        shifted_data = self.data - np.max(
            self.data, axis=axis, keepdims=True
        )

        # Calculate the exponential of each shifted value.
        exp_values = np.exp(shifted_data)

        # Divide each exponential by the sum to get probabilities.
        probabilities = exp_values / np.sum(
            exp_values, axis=axis, keepdims=True
        )

        # Create a new Tensor and connect it to the computation graph.
        result = Tensor(
            probabilities,
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            # Calculate the gradient of Softmax.
            if self.requires_grad:
                # Multiply probabilities by the adjusted output gradient.
                weighted_sum = np.sum(
                    result.grad * result.data,
                    axis=axis,
                    keepdims=True
                )

                self.grad += result.data * (
                    result.grad - weighted_sum
                )

        # Save the backward operation in the computation graph.
        result._backward = _backward

        return result
    
    def cross_entropy(self, targets):
        # Cross-entropy expects logits with shape (batch, classes).
        if self.data.ndim != 2:
            raise ValueError(
                "Cross-entropy expects a 2D logits tensor."
            )

        # Convert labels to a NumPy array.
        targets = np.asarray(targets)

        batch_size, num_classes = self.data.shape

        # Check that there is one label for each example.
        if targets.shape != (batch_size,):
            raise ValueError(
                "Targets must have shape (batch_size,)."
            )

        # Class labels must be integers in the valid range.
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Targets must be integer class labels.")

        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError("Target class index is out of range.")

        # Subtract the maximum logit in each row for stability.
        shifted = self.data - np.max(
            self.data, axis=1, keepdims=True
        )

        # Calculate log(sum(exp(logits))) in a stable way.
        log_sum_exp = np.log(
            np.sum(np.exp(shifted), axis=1)
        )

        # Get the logit for the correct class of each example.
        correct_logits = shifted[
            np.arange(batch_size), targets
        ]

        # Calculate the mean cross-entropy loss.
        loss_value = np.mean(log_sum_exp - correct_logits)

        # Create a scalar loss connected to the computation graph.
        result = Tensor(
            loss_value,
            requires_grad=self.requires_grad,
            parents=(self,)
        )

        def _backward():
            if self.requires_grad:
                # Calculate Softmax probabilities.
                probabilities = np.exp(shifted)
                probabilities /= np.sum(
                    probabilities, axis=1, keepdims=True
                )

                # For each example, subtract 1 from the correct class.
                probabilities[
                    np.arange(batch_size), targets
                ] -= 1.0

                # Average the gradient across the batch.
                probabilities /= batch_size

                # Send the gradient back to the logits.
                self.grad += probabilities * result.grad

        # Register the backward operation.
        result._backward = _backward

        return result

    
    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other)
        )

        def _backward():
            if self.requires_grad:
                self.grad += result.grad @ other.data.T

            if other.requires_grad:
    
                other.grad += self.data.T @ result.grad

        result._backward = _backward

        return result






