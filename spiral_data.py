
import numpy as np


def make_spiral_data(
    samples_per_class=300,
    num_classes=3,
    noise=0.2,
    seed=42
):
    """
    Generate a synthetic spiral classification dataset.

    Each example has 2 input features.
    Each example belongs to one of 3 classes.
    """

    # Use a fixed random generator for reproducibility.
    rng = np.random.default_rng(seed)

    total_samples = samples_per_class * num_classes

    # Store the two input features for every example.
    X = np.zeros((total_samples, 2), dtype=np.float64)

    # Store the integer class label for every example.
    y = np.zeros(total_samples, dtype=np.int64)

    for class_id in range(num_classes):
        # Find the rows belonging to this class.
        start = class_id * samples_per_class
        end = start + samples_per_class

        # Spread points along a spiral.
        radius = np.linspace(0.0, 1.0, samples_per_class)

        # Each class starts at a different angle.
        angle = (
            np.linspace(0.0, 4.0 * np.pi, samples_per_class)
            + class_id * (2.0 * np.pi / num_classes)
        )

        # Add random noise so the problem is not perfectly clean.
        angle += rng.normal(0.0, noise, samples_per_class)

        # Convert polar coordinates into two input features.
        X[start:end, 0] = radius * np.sin(angle)
        X[start:end, 1] = radius * np.cos(angle)

        # Assign the class label.
        y[start:end] = class_id

    # Shuffle examples so batches contain mixed classes.
    permutation = rng.permutation(total_samples)

    X = X[permutation]
    y = y[permutation]

    return X, y
