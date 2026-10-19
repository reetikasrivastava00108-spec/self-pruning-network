
import numpy as np
from tensor import Tensor


def numerical_gradient(function, x, epsilon=1e-5):
    """Calculate gradient using finite differences."""
    gradient = np.zeros_like(x, dtype=np.float64)

    for index in np.ndindex(x.shape):
        original_value = x[index]

        x[index] = original_value + epsilon
        positive_value = function(x)

        x[index] = original_value - epsilon
        negative_value = function(x)

        gradient[index] = (
            positive_value - negative_value
        ) / (2 * epsilon)

        x[index] = original_value

    return gradient


def check_gradient(name, tensor_gradient, function, x_data):
    """Compare analytical and numerical gradients."""
    numerical_grad = numerical_gradient(
        function, x_data.copy()
    )

    print(f"\n{name}")
    print("Analytical gradient:", tensor_gradient)
    print("Numerical gradient: ", numerical_grad)

    assert np.allclose(
        tensor_gradient, numerical_grad, atol=1e-5
    ), f"{name} gradient check failed!"

    print("Gradient check passed!")


def test_multiplication_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    y = (x * x).sum()
    y.backward()

    check_gradient(
        "Multiplication",
        x.grad,
        lambda values: np.sum(values * values),
        x_data
    )

def test_sum_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    loss = x.sum()
    loss.backward()

    numerical_grad = numerical_gradient(
        lambda values: np.sum(values),
        x_data.copy()
    )

    print("\nSum")
    print("Analytical gradient:", x.grad)
    print("Numerical gradient: ", numerical_grad)

    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "Sum gradient check failed!"

    print("Sum gradient check passed!")


def test_mean_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    loss = x.mean()
    loss.backward()

    numerical_grad = numerical_gradient(
        lambda values: np.mean(values),
        x_data.copy()
    )

    print("\nMean")
    print("Analytical gradient:", x.grad)
    print("Numerical gradient: ", numerical_grad)

    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "Mean gradient check failed!"

    print("Mean gradient check passed!")


def test_addition_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    y = (x + 5.0).sum()
    y.backward()

    check_gradient(
        "Addition",
        x.grad,
        lambda values: np.sum(values + 5.0),
        x_data
    )


def test_subtraction_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    y = (x - 5.0).sum()
    y.backward()

    check_gradient(
        "Subtraction",
        x.grad,
        lambda values: np.sum(values - 5.0),
        x_data
    )


def test_division_gradient():
    x_data = np.array([2.0, 3.0, 4.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    y = (x / 5.0).sum()
    y.backward()

    check_gradient(
        "Division",
        x.grad,
        lambda values: np.sum(values / 5.0),
        x_data
    )

def test_broadcasting_addition_gradient():
    a_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    b_data = np.array([10.0, 20.0, 30.0])

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)

    loss = (a + b).sum()
    loss.backward()

    numerical_grad_a = numerical_gradient(
        lambda values: np.sum(values + b_data),
        a_data.copy()
    )

    numerical_grad_b = numerical_gradient(
        lambda values: np.sum(a_data + values),
        b_data.copy()
    )

    print("\nBroadcasting Addition")
    print("Analytical gradient A:\n", a.grad)
    print("Numerical gradient A:\n", numerical_grad_a)
    print("Analytical gradient b:", b.grad)
    print("Numerical gradient b:", numerical_grad_b)

    assert np.allclose(a.grad, numerical_grad_a, atol=1e-5)
    assert np.allclose(b.grad, numerical_grad_b, atol=1e-5)

    print("Broadcasting gradient check passed!")


def test_broadcasting_subtraction_gradient():
    a_data = np.array([[5., 6., 7.], [8., 9., 10.]])
    b_data = np.array([1., 2., 3.])

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)

    loss = (a - b).sum()
    loss.backward()

    grad_a = numerical_gradient(
        lambda values: np.sum(values - b_data),
        a_data.copy()
    )
    grad_b = numerical_gradient(
        lambda values: np.sum(a_data - values),
        b_data.copy()
    )

    assert np.allclose(a.grad, grad_a, atol=1e-5)
    assert np.allclose(b.grad, grad_b, atol=1e-5)
    print("Broadcasting subtraction gradient check passed!")


def test_broadcasting_multiplication_gradient():
    a_data = np.array([[1., 2., 3.], [4., 5., 6.]])
    b_data = np.array([10., 20., 30.])

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)

    loss = (a * b).sum()
    loss.backward()

    grad_a = numerical_gradient(
        lambda values: np.sum(values * b_data),
        a_data.copy()
    )
    grad_b = numerical_gradient(
        lambda values: np.sum(a_data * values),
        b_data.copy()
    )

    assert np.allclose(a.grad, grad_a, atol=1e-5)
    assert np.allclose(b.grad, grad_b, atol=1e-5)
    print("Broadcasting multiplication gradient check passed!")


def test_broadcasting_division_gradient():
    a_data = np.array([[6., 8., 12.], [10., 12., 18.]])
    b_data = np.array([2., 4., 6.])

    a = Tensor(a_data.copy(), requires_grad=True)
    b = Tensor(b_data.copy(), requires_grad=True)

    loss = (a / b).sum()
    loss.backward()

    grad_a = numerical_gradient(
        lambda values: np.sum(values / b_data),
        a_data.copy()
    )
    grad_b = numerical_gradient(
        lambda values: np.sum(a_data / values),
        b_data.copy()
    )

    assert np.allclose(a.grad, grad_a, atol=1e-5)
    assert np.allclose(b.grad, grad_b, atol=1e-5)
    print("Broadcasting division gradient check passed!")


def test_matmul_gradient():
    x_data = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    w_data = np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    x = Tensor(x_data.copy(), requires_grad=True)
    w = Tensor(w_data.copy(), requires_grad=True)

    # Forward pass: matrix multiplication, then scalar loss
    loss = x.matmul(w).sum()

    # Backward pass: calculate analytical gradients
    loss.backward()

    # Numerical gradient for input matrix X
    numerical_grad_x = numerical_gradient(
        lambda values: np.sum(values @ w_data),
        x_data.copy()
    )

    # Numerical gradient for weight matrix W
    numerical_grad_w = numerical_gradient(
        lambda values: np.sum(x_data @ values),
        w_data.copy()
    )

    print("\nMatrix Multiplication (matmul)")
    print("Analytical gradient X:\n", x.grad)
    print("Numerical gradient X:\n", numerical_grad_x)
    print("Analytical gradient W:\n", w.grad)
    print("Numerical gradient W:\n", numerical_grad_w)

    assert np.allclose(
        x.grad, numerical_grad_x, atol=1e-5
    ), "Input matrix gradient check failed!"

    assert np.allclose(
        w.grad, numerical_grad_w, atol=1e-5
    ), "Weight matrix gradient check failed!"

    print("Matmul gradient check passed!")


def test_shared_parent_gradient_accumulation():
    x_data = np.array([2.0])

    x = Tensor(x_data.copy(), requires_grad=True)

    a = x * x
    b = x + x
    loss = (a + b).sum()

    loss.backward()

    numerical_grad = numerical_gradient(
        lambda values: np.sum(values * values + 2 * values),
        x_data.copy()
    )

    print("\nShared-parent gradient accumulation")
    print("Analytical gradient:", x.grad)
    print("Numerical gradient: ", numerical_grad)

    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "Shared-parent gradient accumulation failed!"

    print("Shared-parent gradient check passed!")


def test_repeated_backward_accumulates_gradients():
    x = Tensor(np.array([3.0]), requires_grad=True)
    loss = (x * x).sum()

    loss.backward()
    first_gradient = x.grad.copy()

    loss.backward()
    second_gradient = x.grad.copy()

    print("\nRepeated backward")
    print("After first backward:", first_gradient)
    print("After second backward:", second_gradient)

    assert np.allclose(first_gradient, [6.0])
    assert np.allclose(second_gradient, [12.0])

    print("Repeated backward accumulation test passed!")

def test_softmax_gradient():
    # Create three input logits.
    x_data = np.array([1.2, -0.7, 2.1])

    # Give each output probability a different weight.
    weights = np.array([0.2, -0.3, 0.7])

    # Create a Tensor that tracks gradients.
    x = Tensor(x_data.copy(), requires_grad=True)

    # Apply Softmax.
    probabilities = x.softmax()

    # Create a scalar loss so backward() can run.
    loss = (probabilities * weights).sum()
    loss.backward()

    # Calculate Softmax using NumPy for numerical checking.
    def numpy_softmax(values):
        # Subtract the maximum for numerical stability.
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    # Calculate the numerical gradient using finite differences.
    numerical_grad = numerical_gradient(
        lambda values: np.sum(
            numpy_softmax(values) * weights
        ),
        x_data.copy()
    )

    print("\nSoftmax")
    print("Probabilities:", probabilities.data)
    print("Probability sum:", np.sum(probabilities.data))
    print("Analytical gradient:", x.grad)
    print("Numerical gradient:", numerical_grad)

    # Probabilities must add up to 1.
    assert np.allclose(
        np.sum(probabilities.data), 1.0, atol=1e-7
    ), "Softmax probabilities do not sum to 1!"

    # Check the analytical gradient against the numerical gradient.
    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "Softmax gradient check failed!"

    # Check that very large logits do not cause overflow.
    large_logits = Tensor(np.array([1000.0, 1001.0, 1002.0]))
    large_probabilities = large_logits.softmax()

    assert np.all(np.isfinite(large_probabilities.data))
    assert np.allclose(
        np.sum(large_probabilities.data), 1.0, atol=1e-7
    )

    print("Softmax gradient check passed!")
    print("Large-logit stability check passed!")


def test_cross_entropy_gradient():
    # Create logits for two examples and three classes.
    logits_data = np.array([
        [1.2, -0.7, 2.1],
        [0.3, 1.5, -0.2]
    ])

    # The correct class for each example.
    targets = np.array([2, 1])

    # Create a Tensor that tracks gradients.
    logits = Tensor(logits_data.copy(), requires_grad=True)

    # Calculate cross-entropy directly from logits.
    loss = logits.cross_entropy(targets)
    loss.backward()

    # Calculate the numerical gradient for checking.
    def numpy_cross_entropy(values):
        # Stable calculation of log-sum-exp.
        shifted = values - np.max(
            values, axis=1, keepdims=True
        )
        log_sum_exp = np.log(
            np.sum(np.exp(shifted), axis=1)
        )

        # Select the correct class score for each example.
        correct_logits = shifted[
            np.arange(len(targets)), targets
        ]

        # Return the average loss.
        return np.mean(log_sum_exp - correct_logits)

    numerical_grad = numerical_gradient(
        numpy_cross_entropy,
        logits_data.copy()
    )

    print("\nCross-Entropy")
    print("Loss:", loss.data)
    print("Analytical gradient:\n", logits.grad)
    print("Numerical gradient:\n", numerical_grad)

    # Check analytical gradients against numerical gradients.
    assert np.allclose(
        logits.grad, numerical_grad, atol=1e-5
    ), "Cross-entropy gradient check failed!"

    # Check that the loss is finite.
    assert np.isfinite(loss.data).all()

    print("Cross-entropy gradient check passed!")

def test_relu_gradient():
    x_data = np.array([-2.0, -0.5, 0.5, 2.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    loss = x.relu().sum()
    loss.backward()

    numerical_grad = numerical_gradient(
        lambda values: np.sum(np.maximum(values, 0.0)),
        x_data.copy()
    )

    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "ReLU gradient check failed!"

    print("ReLU gradient check passed!")


def test_tanh_gradient():
    x_data = np.array([-2.0, -0.5, 0.5, 2.0])

    x = Tensor(x_data.copy(), requires_grad=True)
    loss = x.tanh().sum()
    loss.backward()

    numerical_grad = numerical_gradient(
        lambda values: np.sum(np.tanh(values)),
        x_data.copy()
    )

    assert np.allclose(
        x.grad, numerical_grad, atol=1e-5
    ), "Tanh gradient check failed!"

    print("Tanh gradient check passed!")


if __name__ == "__main__":
    test_broadcasting_addition_gradient()
    test_broadcasting_subtraction_gradient()
    test_broadcasting_multiplication_gradient()
    test_broadcasting_division_gradient()
    test_matmul_gradient()
    test_shared_parent_gradient_accumulation()
    test_repeated_backward_accumulates_gradients()
    test_addition_gradient()
    test_subtraction_gradient()
    test_multiplication_gradient()
    test_division_gradient()
    test_sum_gradient()
    test_mean_gradient()
    test_softmax_gradient()
    test_cross_entropy_gradient()

    print("\nAll gradient checks passed!")
