
import time
from tensor import Tensor
import numpy as np
from model import MLP


def count_dense_macs(model, batch_size):
    """Count MACs performed by dense linear layers."""

    total_macs = 0

    for layer in (model.layer1, model.layer2):
        total_macs += (
            batch_size
            * layer.in_features
            * layer.out_features
        )

    return total_macs


def count_sparse_macs(model, batch_size):
    """Count MACs for active connections only."""

    total_macs = 0

    for layer in (model.layer1, model.layer2):
        active_connections = np.count_nonzero(layer.mask)
        total_macs += batch_size * active_connections

    return total_macs



def benchmark_inference(model, inputs, sparse=False, repeats=100):
    """Measure average inference time per call."""

    # Dense forward expects our custom Tensor.
    # Sparse forward expects a NumPy array.
    dense_inputs = Tensor(inputs)

    # Warm up before timing.
    for _ in range(5):
        if sparse:
            model.forward_sparse(inputs)
        else:
            model.forward(dense_inputs)

    start = time.perf_counter()

    for _ in range(repeats):
        if sparse:
            model.forward_sparse(inputs)
        else:
            model.forward(dense_inputs)

    elapsed = time.perf_counter() - start

    return elapsed / repeats

def main():
    np.random.seed(42)

    model = MLP(
        input_size=2,
        hidden_size=64,
        output_size=3,
    )

    batch_size = 128

    inputs = np.random.randn(batch_size, 2)

    # Prune 75% of weight connections globally.
    weight_layers = (model.layer1, model.layer2)

    all_connections = []

    for layer in weight_layers:
        for index in np.ndindex(layer.mask.shape):
            all_connections.append((layer, index))

    number_to_prune = int(0.75 * len(all_connections))

    for layer, index in all_connections[:number_to_prune]:
        layer.mask[index] = 0.0
        layer.weights.data[index] = 0.0

    dense_macs = count_dense_macs(model, batch_size)
    sparse_macs = count_sparse_macs(model, batch_size)

    dense_seconds = benchmark_inference(
        model, inputs, sparse=False
    )

    sparse_seconds = benchmark_inference(
        model, inputs, sparse=True
    )

    print(f"Batch size: {batch_size}")
    print(f"Dense MACs per batch: {dense_macs}")
    print(f"Active sparse MACs per batch: {sparse_macs}")

    reduction = 100.0 * (1.0 - sparse_macs / dense_macs)

    print(f"MAC reduction: {reduction:.2f}%")
    print(f"Dense inference: {dense_seconds * 1000:.3f} ms/call")
    print(f"Sparse inference: {sparse_seconds * 1000:.3f} ms/call")

    if sparse_seconds > 0:
        speedup = dense_seconds / sparse_seconds
        print(f"Dense/sparse time ratio: {speedup:.3f}x")


if __name__ == "__main__":
    main()
