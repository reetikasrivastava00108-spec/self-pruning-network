
import copy
import csv

import matplotlib.pyplot as plt
import numpy as np

from tensor import Tensor
from model import MLP
from optim import Adam
from spiral_data import make_spiral_data
from pruning import prune_weights


SPARSITY_LEVELS = [0.0, 0.50, 0.75, 0.90, 0.95]


def evaluate_accuracy(model, X, y):
    """Return classification accuracy on the supplied dataset."""
    logits = model.forward(Tensor(X))
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == y))


def train_dense_baseline(X_train, y_train, seed=42):
    """Train one dense model without pruning."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 123)

    model = MLP(
        input_size=2,
        hidden_size=64,
        output_size=3,
    )

    optimizer = Adam(
        model.parameters(),
        learning_rate=0.01,
    )

    epochs = 700
    batch_size = 32

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()

            logits = model.forward(Tensor(X_batch))
            loss = logits.cross_entropy(y_batch)

            if not np.isfinite(loss.data):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}"
                )

            loss.backward()
            optimizer.step()

    return model


def main():
    # Generate a reproducible dataset and split.
    X, y = make_spiral_data(
        samples_per_class=300,
        num_classes=3,
        noise=0.2,
        seed=42,
    )

    split_rng = np.random.default_rng(123)
    indices = split_rng.permutation(len(X))
    split_index = int(0.8 * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("Training dense baseline...")
    baseline = train_dense_baseline(
        X_train,
        y_train,
        seed=42,
    )

    results = []

    for target_sparsity in SPARSITY_LEVELS:
        # Each experiment starts from the same trained baseline.
        model = copy.deepcopy(baseline)

        if target_sparsity > 0:
            actual_sparsity = prune_weights(
                model,
                target_sparsity,
                X_train,
                y_train,
                batch_size=32,
            )
        else:
            actual_sparsity = 0.0

        accuracy = evaluate_accuracy(model, X_test, y_test)

        row = {
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sparsity,
            "test_accuracy": accuracy,
        }
        results.append(row)

        print(
            f"Target: {target_sparsity:.0%} | "
            f"Actual: {actual_sparsity:.2%} | "
            f"Test accuracy: {accuracy:.2%}"
        )

    # Save the raw measured results.
    with open(
        "pareto_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_sparsity",
                "actual_sparsity",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Plot actual sparsity against measured test accuracy.
    sparsities = [
        row["actual_sparsity"] * 100 for row in results
    ]
    accuracies = [
        row["test_accuracy"] * 100 for row in results
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(sparsities, accuracies, marker="o")
    plt.xlabel("Actual sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Sparsity vs Test Accuracy — Taylor Pruning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_curve.png", dpi=200)
    plt.close()

    print("\nSaved raw results: pareto_results.csv")
    print("Saved graph: pareto_curve.png")


if __name__ == "__main__":
    main()
