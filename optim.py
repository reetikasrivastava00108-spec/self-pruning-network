
import numpy as np


class Adam:
    """
    Adam optimizer implemented using NumPy.

    It updates custom Tensor parameters using their gradients.
    """

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        # Store the trainable parameters.
        self.parameters = list(parameters)

        # Store the optimizer hyperparameters.
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Count how many optimizer steps have been performed.
        self.step_count = 0

        # Initialize the first and second moment for each parameter.
        self.first_moments = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

        self.second_moments = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

    def zero_grad(self):
        """Reset gradients before the next backward pass."""

        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.fill(0.0)

    def step(self):
        """Update all parameters using their gradients."""

        self.step_count += 1

        for index, parameter in enumerate(self.parameters):
            # Skip parameters that do not have gradients.
            if parameter.grad is None:
                continue

            gradient = parameter.grad

            # Fail clearly if training produces invalid gradients.
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError(
                    "Non-finite gradient detected."
                )

            # Update the first moment: moving average of gradients.
            self.first_moments[index] = (
                self.beta1 * self.first_moments[index]
                + (1.0 - self.beta1) * gradient
            )

            # Update the second moment: moving average of squared gradients.
            self.second_moments[index] = (
                self.beta2 * self.second_moments[index]
                + (1.0 - self.beta2) * (gradient ** 2)
            )

            # Correct the initialization bias in both moments.
            corrected_first = (
                self.first_moments[index]
                / (1.0 - self.beta1 ** self.step_count)
            )

            corrected_second = (
                self.second_moments[index]
                / (1.0 - self.beta2 ** self.step_count)
            )

            # Calculate the adaptive parameter update.
            update = (
                self.learning_rate * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )

            # Update the underlying parameter values.
            parameter.data -= update

            # Make sure the parameter values remain finite.
            if not np.all(np.isfinite(parameter.data)):
                raise FloatingPointError(
                    "Non-finite parameter detected."
                )
