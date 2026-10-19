import numpy as np
from tensor import Tensor


def calculate_importance_scores(model, X, y, batch_size=32):
    # Select weight parameters that have masks
    weights = [
        parameter
        for parameter in model.parameters()
        if hasattr(parameter, "mask")
    ]

    # Create an importance score array for each weight matrix
    scores = [
        np.zeros_like(parameter.data)
        for parameter in weights
    ]

    # Calculate the number of training examples
    total_examples = len(X)

    # Process the data in mini-batches
    for start in range(0, total_examples, batch_size):
        end = start + batch_size

        X_batch = X[start:end]
        y_batch = y[start:end]

        # Clear gradients before calculating new ones
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.fill(0.0)

        # Run the forward pass
        inputs = Tensor(X_batch)
        logits = model.forward(inputs)

        # Calculate the loss
        loss = logits.cross_entropy(y_batch)

        # Calculate gradients using backpropagation
        loss.backward()

        # Accumulate importance scores
        batch_weight = len(X_batch) / total_examples

        for index, parameter in enumerate(weights):
            gradient = parameter.grad

            # Calculate the Taylor-based importance score
            batch_scores = np.abs(
                parameter.data * gradient
            )
            # Ignore connections that were already pruned
            batch_scores[parameter.mask == 0] = 0.0
            
            scores[index] += batch_scores * batch_weight

    # Clear gradients after scoring
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.fill(0.0)

    return list(zip(weights, scores))

def prune_weights(model, target_sparsity, X, y, batch_size=32):
    # Calculate importance scores
    results = calculate_importance_scores(
        model, X, y, batch_size
    )

    # Count all weight connections
    total_weights = sum(
        weights.data.size for weights, scores in results
    )

    # Calculate how many connections should remain active
    target_active = int(round(
        total_weights * (1.0 - target_sparsity)
    ))

    # Collect importance scores of currently active connections
    active_scores = []

    for weights, scores in results:
        active = weights.mask.astype(bool)

        for index in np.ndindex(weights.data.shape):
            if active[index]:
                active_scores.append(
                    (scores[index], weights, index)
                )

    # Keep the least-important connections eligible for pruning
    active_scores.sort(key=lambda item: item[0])

    # Count currently active connections
    currently_active = len(active_scores)

    # Number of connections to remove
    to_prune = max(0, currently_active - target_active)

    # Remove the least-important connections
    for _, weights, index in active_scores[:to_prune]:
        weights.mask[index] = 0.0
        weights.data[index] = 0.0

    # Report pruning results
    remaining = sum(
        np.count_nonzero(weights.mask)
        for weights, scores in results
    )

    actual_sparsity = 1.0 - remaining / total_weights

    return actual_sparsity

def prune_weights_by_magnitude(model, target_sparsity):
    """
    Prune weights with the smallest absolute values.

    Unlike Taylor pruning, this method does not need gradients.
    """

    # Select parameters that represent weight matrices.
    weights = [
        parameter
        for parameter in model.parameters()
        if hasattr(parameter, "mask")
    ]

    # Count all weight connections, including already-pruned ones.
    total_weights = sum(
        parameter.data.size for parameter in weights
    )

    if total_weights == 0:
        raise ValueError("Model has no weights to prune.")

    if not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity must be between 0 and 1.")

    # Calculate the desired number of active connections.
    target_active = int(round(
        total_weights * (1.0 - target_sparsity)
    ))

    # Collect magnitudes of currently active connections.
    active_scores = []

    for parameter in weights:
        active = parameter.mask.astype(bool)

        for index in np.ndindex(parameter.data.shape):
            if active[index]:
                importance = abs(parameter.data[index])

                active_scores.append(
                    (importance, parameter, index)
                )

    # Smallest absolute values have the lowest importance.
    active_scores.sort(key=lambda item: item[0])

    # Do not restore connections that were previously pruned.
    currently_active = len(active_scores)
    to_prune = max(0, currently_active - target_active)

    # Prune the least-important active connections.
    for _, parameter, index in active_scores[:to_prune]:
        parameter.mask[index] = 0.0
        parameter.data[index] = 0.0

    # Calculate the actual sparsity after pruning.
    remaining = sum(
        np.count_nonzero(parameter.mask)
        for parameter in weights
    )

    return 1.0 - remaining / total_weights

def cubic_sparsity_schedule(
    epoch,
    start_epoch,
    end_epoch,
    target_sparsity
):
    # Before pruning starts, keep all connections
    if epoch < start_epoch:
        return 0.0

    # After the schedule ends, reach target sparsity
    if epoch >= end_epoch:
        return target_sparsity

    # Calculate progress from 0 to 1
    progress = (
        (epoch - start_epoch) /
        (end_epoch - start_epoch)
    )

    # Increase sparsity gradually using a cubic curve
    return target_sparsity * (1 - (1 - progress) ** 3)