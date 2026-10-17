
from layers import Linear


class MLP:
    """
    A small Multi-Layer Perceptron.

    Architecture:
        Input -> Linear -> ReLU -> Linear -> Logits
    """

    def __init__(self, input_size, hidden_size, output_size):
        # First layer converts input features into hidden features.
        self.layer1 = Linear(input_size, hidden_size)

        # Second layer converts hidden features into class scores.
        self.layer2 = Linear(hidden_size, output_size)

    def forward(self, inputs):
        # Calculate the output of the first linear layer.
        hidden = self.layer1.forward(inputs)

        # Apply ReLU to introduce nonlinearity.
        hidden = hidden.relu()

        # Calculate the final class scores (logits).
        logits = self.layer2.forward(hidden)

        return logits

    def parameters(self):
        # Collect parameters from both layers.
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
        )
