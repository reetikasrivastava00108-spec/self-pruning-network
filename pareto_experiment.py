
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
RANDOM_SEEDS = [42, 43, 44]


def evaluate_accuracy(model, X, y):
    """Calculate classification accuracy."""
    logits = model.forward(Tensor(X))
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == y))


def train_dense_baseline(X_train, y_train, seed):
    """Train a fresh dense model without pruning."""
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
    # Keep the dataset and train/test split fixed for fair comparison.
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
        print(f"\nTraining dense baseline with seed {seed}...")

        baseline = train_dense_baseline(
            X_train,
            y_train,
            seed=seed,
        )

        for target_sparsity in SPARSITY_LEVELS:
            # Every sparsity level starts from this seed's same baseline.
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
                "seed": seed,
                "target_sparsity": target_sparsity,
                "actual_sparsity": actual_sparsity,
                "test_accuracy": accuracy,
            }
            raw_results.append(row)

            print(
                f"Seed: {seed} | "
                f"Target: {target_sparsity:.0%} | "
                f"Actual: {actual_sparsity:.2%} | "
                f"Accuracy: {accuracy:.2%}"
            )

    # Save every individual measurement.
    with open(
        "pareto_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "seed",
            "target_sparsity",
            "actual_sparsity",
            "test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_results)

    # Calculate mean and standard deviation across seeds.
    summary_results = []

    for target_sparsity in SPARSITY_LEVELS:
        matching_rows = [
            row for row in raw_results
            if row["target_sparsity"] == target_sparsity
        ]

        accuracies = np.array(
            [row["test_accuracy"] for row in matching_rows]
        )
        actual_sparsities = np.array(
            [row["actual_sparsity"] for row in matching_rows]
        )

        summary_results.append({
            "target_sparsity": target_sparsity,
            "mean_actual_sparsity": float(np.mean(actual_sparsities)),
            "mean_test_accuracy": float(np.mean(accuracies)),
            "std_test_accuracy": float(np.std(accuracies, ddof=1)),
        })

    with open(
        "pareto_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "target_sparsity",
            "mean_actual_sparsity",
            "mean_test_accuracy",
            "std_test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)

    # Plot mean accuracy with standard-deviation error bars.
    x_values = [
        row["mean_actual_sparsity"] * 100
        for row in summary_results
    ]
    means = [
        row["mean_test_accuracy"] * 100
        for row in summary_results
    ]
    standard_deviations = [
        row["std_test_accuracy"] * 100
        for row in summary_results
    ]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x_values,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=5,
    )
    plt.xlabel("Mean actual sparsity (%)")
    plt.ylabel("Mean test accuracy (%)")
    plt.title("Sparsity vs Accuracy — Taylor Pruning (3 seeds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_curve.png", dpi=200)
    plt.close()

    print("\nSaved raw results: pareto_results.csv")
    print("Saved summary statistics: pareto_summary.csv")
    print("Saved graph: pareto_curve.png")

    print("\nSummary (mean accuracy ± standard deviation):")
    for row in summary_results:
        print(
            f"Sparsity {row['target_sparsity']:.0%}: "
            f"{row['mean_test_accuracy']:.2%} "
            f"+/- {row['std_test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()

