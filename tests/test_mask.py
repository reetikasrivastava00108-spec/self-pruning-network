import numpy as np
from tensor import Tensor
from layers import Linear

# Create a layer with 2 inputs and 2 outputs
layer = Linear(2, 2)

# Set known weights
layer.weights.data[:] = np.array([
    [0.5, 0.8],
    [0.3, 0.6]
])

# Remove the connection with weight 0.8
layer.mask[0, 1] = 0

# Create input data
inputs = Tensor(
    np.array([[1.0, 2.0]]),
    requires_grad=False
)

# Calculate output
output = layer.forward(inputs)

# Display results
print("Mask:")
print(layer.mask)

print("\nEffective weights:")
print(layer.weights.data * layer.mask)

print("\nOutput:")
print(output.data)

# Calculate gradients
output.sum().backward()

# Display weight gradients
print("\nWeight gradients:")
print(layer.weights.grad)

# Check that the pruned connection has zero gradient
print("\nPruned connection gradient:")
print(layer.weights.grad[0, 1])