import numpy as np

from model import MLP
from spiral_data import make_spiral_data
from pruning import calculate_importance_scores, prune_weights


def test_importance_scores():
    X, y = make_spiral_data(
        samples_per_class=10,
        num_classes=3,
        noise=0.2,
        seed=42
    )

    model = MLP(input_size=2, hidden_size=8, output_size=3)
    results = calculate_importance_scores(model, X, y, batch_size=10)

    assert len(results) == 2

    for weights, scores in results:
        assert scores.shape == weights.data.shape
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))

    print("PASS: Importance scores calculated correctly.")


def test_pruning():
    X, y = make_spiral_data(
        samples_per_class=10,
        num_classes=3,
        noise=0.2,
        seed=42
    )

    model = MLP(input_size=2, hidden_size=8, output_size=3)

    sparsity = prune_weights(
        model,
        target_sparsity=0.5,
        X=X,
        y=y,
        batch_size=10
    )

    weights = [
        parameter for parameter in model.parameters()
        if hasattr(parameter, "mask")
    ]

    total = sum(parameter.mask.size for parameter in weights)
    active = sum(np.count_nonzero(parameter.mask) for parameter in weights)

    assert np.isclose(sparsity, 0.5)
    assert active == total // 2

    for parameter in weights:
        assert np.all(parameter.data[parameter.mask == 0] == 0)

    print("PASS: Pruning removes the expected connections.")
    print(f"Total connections: {total}")
    print(f"Active connections: {active}")
    print(f"Actual sparsity: {sparsity:.0%}")

def test_cubic_schedule():
    from pruning import cubic_sparsity_schedule

    start = 80
    end = 320
    target = 0.9

    # Before pruning starts
    assert cubic_sparsity_schedule(50, start, end, target) == 0.0

    # At the start, sparsity is zero
    assert cubic_sparsity_schedule(80, start, end, target) == 0.0

    # At the end, target sparsity is reached
    assert cubic_sparsity_schedule(320, start, end, target) == 0.9

    # During pruning, sparsity increases gradually
    early = cubic_sparsity_schedule(140, start, end, target)
    later = cubic_sparsity_schedule(240, start, end, target)

    assert 0.0 < early < later < target

    print("PASS: Cubic pruning schedule works correctly.")


if __name__ == "__main__":
    test_importance_scores()
    test_pruning()
    test_cubic_schedule()