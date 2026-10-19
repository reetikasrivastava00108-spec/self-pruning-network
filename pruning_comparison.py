
import copy
import csv

import matplotlib.pyplot as plt
import numpy as np

from tensor import Tensor
from model import MLP
from optim import Adam
from spiral_data import make_spiral_data
from pruning import prune_weights, prune_weights_by_magnitude


SPARSITY_LEVELS = [0.0, 0.50, 0.75, 0.90, 0.95]
RANDOM_SEEDS = [42, 43, 44]
METHODS = ["Taylor", "Magnitude"]


def evaluate_accuracy(model, X, y):
    """Calculate classification accuracy."""
    logits = model.forward(Tensor(X))
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == y))


def train_dense_baseline(X_train, y_train, seed):
    """Train one unpruned model for a given seed."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 123)

    model = MLP(input_size=2, hidden_size=64, output_size=3)
    optimizer = Adam(model.parameters(), learning_rate=0.01)

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
    # Use the same dataset and fixed split as the Pareto experiment.
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

    raw_results = []

    for seed in RANDOM_SEEDS:
        print(f"\nTraining baseline with seed {seed}...")

        baseline = train_dense_baseline(X_train, y_train, seed)

        for target_sparsity in SPARSITY_LEVELS:
            for method in METHODS:
                # Both methods start from an independent copy
                # of exactly the same trained baseline.
                model = copy.deepcopy(baseline)

                if target_sparsity == 0.0:
                    actual_sparsity = 0.0
                elif method == "Taylor":
                    actual_sparsity = prune_weights(
                        model,
                        target_sparsity,
                        X_train,
                        y_train,
                        batch_size=32,
                    )
                else:
                    actual_sparsity = prune_weights_by_magnitude(
                        model,
                        target_sparsity,
                    )

                accuracy = evaluate_accuracy(model, X_test, y_test)

                row = {
                    "seed": seed,
                    "method": method,
                    "target_sparsity": target_sparsity,
                    "actual_sparsity": actual_sparsity,
                    "test_accuracy": accuracy,
                }
                raw_results.append(row)

                print(
                    f"Seed: {seed} | Method: {method} | "
                    f"Sparsity: {actual_sparsity:.0%} | "
                    f"Accuracy: {accuracy:.2%}"
                )

    # Save each individual measurement.
    with open(
        "pruning_comparison_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "seed",
            "method",
            "target_sparsity",
            "actual_sparsity",
            "test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_results)

    # Summarize results across seeds for each method and sparsity.
    summary_results = []

    for method in METHODS:
        for target_sparsity in SPARSITY_LEVELS:
            matching_rows = [
                row for row in raw_results
                if row["method"] == method
                and row["target_sparsity"] == target_sparsity
            ]

            accuracies = np.array([
                row["test_accuracy"] for row in matching_rows
            ])
            actual_sparsities = np.array([
                row["actual_sparsity"] for row in matching_rows
            ])

            summary_results.append({
                "method": method,
                "target_sparsity": target_sparsity,
                "mean_actual_sparsity": float(
                    np.mean(actual_sparsities)
                ),
                "mean_test_accuracy": float(np.mean(accuracies)),
                "std_test_accuracy": float(
                    np.std(accuracies, ddof=1)
                ),
            })

    with open(
        "pruning_comparison_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "method",
            "target_sparsity",
            "mean_actual_sparsity",
            "mean_test_accuracy",
            "std_test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)

    # Plot mean accuracy and standard deviation for both methods.
    plt.figure(figsize=(9, 5))

    for method in METHODS:
        method_rows = [
            row for row in summary_results
            if row["method"] == method
        ]

        x_values = [
            row["mean_actual_sparsity"] * 100
            for row in method_rows
        ]
        means = [
            row["mean_test_accuracy"] * 100
            for row in method_rows
        ]
        stds = [
            row["std_test_accuracy"] * 100
            for row in method_rows
        ]

        plt.errorbar(
            x_values,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Actual sparsity (%)")
    plt.ylabel("Mean test accuracy (%)")
    plt.title("Taylor vs Magnitude Pruning (3 seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pruning_comparison.png", dpi=200)
    plt.close()

    print("\nSaved raw results: pruning_comparison_results.csv")
    print("Saved summary: pruning_comparison_summary.csv")
    print("Saved graph: pruning_comparison.png")

    print("\nMean test accuracy +/- standard deviation:")
    for row in summary_results:
        print(
            f"{row['method']:9s} | "
            f"{row['target_sparsity']:>5.0%} sparsity | "
            f"{row['mean_test_accuracy']:.2%} +/- "
            f"{row['std_test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
