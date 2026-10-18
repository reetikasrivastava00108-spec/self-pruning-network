
import numpy as np

from tensor import Tensor
from model import MLP
from optim import Adam
from spiral_data import make_spiral_data
from pruning import prune_weights, cubic_sparsity_schedule


def evaluate(model, X, y):
    """Measure loss and accuracy without updating parameters."""

    # Convert all examples into a Tensor.
    inputs = Tensor(X)

    # Predict class scores.
    logits = model.forward(inputs)

    # Calculate classification loss.
    loss = logits.cross_entropy(y)

    # Select the class with the highest score.
    predictions = np.argmax(logits.data, axis=1)

    # Calculate the fraction of correct predictions.
    accuracy = np.mean(predictions == y)

    # Check for invalid loss or predictions.
    if not np.isfinite(loss.data):
        raise FloatingPointError("Non-finite evaluation loss.")

    if not np.all(np.isfinite(logits.data)):
        raise FloatingPointError("Non-finite model predictions.")

    return float(loss.data), float(accuracy)


def train():
    # Make initialization and shuffling reproducible.
    np.random.seed(42)

    # Generate the three-class spiral dataset.
    X, y = make_spiral_data(
        samples_per_class=300,
        num_classes=3,
        noise=0.2,
        seed=42
    )

    # Split the dataset into training and test examples.
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(X))

    split_index = int(0.8 * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Create a neural network: 2 inputs, 64 hidden neurons, 3 classes.
    model = MLP(
        input_size=2,
        hidden_size=64,
        output_size=3
    )

    # Create our own Adam optimizer.
    optimizer = Adam(
        model.parameters(),
        learning_rate=0.01
    )

    # Training configuration.
    epochs = 600
    batch_size = 32

    # Pruning settings
    target_sparsity = 0.75
    pruning_start = 200
    pruning_end = 580
    pruning_frequency = 20
    history = []

    print("Starting training...")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Trainable tensors: {len(model.parameters())}")
    print()

    for epoch in range(1, epochs + 1):

        # Shuffle training examples at the start of every epoch.
        shuffled_indices = rng.permutation(len(X_train))

        X_shuffled = X_train[shuffled_indices]
        y_shuffled = y_train[shuffled_indices]

        # Process the dataset in mini-batches.
        for start in range(0, len(X_train), batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Convert the batch to our custom Tensor.
            inputs = Tensor(X_batch)

            # Clear gradients from the previous batch.
            optimizer.zero_grad()

            # Forward pass: calculate class scores.
            logits = model.forward(inputs)

            # Calculate the loss for this mini-batch.
            loss = logits.cross_entropy(y_batch)

            # Stop immediately if the loss becomes invalid.
            if not np.isfinite(loss.data):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}."
                )

            # Backward pass: calculate parameter gradients.
            loss.backward()

            # Update weights and biases using Adam.
            optimizer.step()
        # Prune low-importance connections at scheduled epochs.
        if (
            epoch >= pruning_start
            and epoch <= pruning_end
            and epoch % pruning_frequency == 0
        ):
            current_sparsity = cubic_sparsity_schedule(
                epoch,
                pruning_start,
                pruning_end,
                target_sparsity
            )

            actual_sparsity = prune_weights(
                model,
                current_sparsity,
                X_train,
                y_train,
                batch_size=batch_size
            )

            print(
                f"Epoch {epoch}: "
                f"Target sparsity: {current_sparsity:.2%} | "
                f"Actual sparsity: {actual_sparsity:.2%}"
            )    
        # Measure performance after completing this epoch.
        train_loss, train_accuracy = evaluate(
            model, X_train, y_train
        )

        test_loss, test_accuracy = evaluate(
            model, X_test, y_test
        )

        # Store the metrics for later analysis.
        history.append([
            epoch,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy
        ])

        # Print progress every 10 epochs and at the end.
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2%} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {test_accuracy:.2%}"
            )

    # Save the actual measured learning curve.
    history = np.asarray(history)

    np.savetxt(
        "learning_curve.csv",
        history,
        delimiter=",",
        header=(
            "epoch,train_loss,train_accuracy,"
            "test_loss,test_accuracy"
        ),
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"]
    )

    # Final numerical-stability check.
    for parameter in model.parameters():
        if not np.all(np.isfinite(parameter.data)):
            raise FloatingPointError(
                "Non-finite parameter detected after training."
            )

    print()
    print("Training completed successfully.")
    print(f"Final training accuracy: {train_accuracy:.2%}")
    print(f"Final test accuracy:     {test_accuracy:.2%}")
    print("Learning curve saved to learning_curve.csv")

    return model, history


if __name__ == "__main__":
    train()
