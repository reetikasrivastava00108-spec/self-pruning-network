
import numpy as np

from spiral_data import make_spiral_data


def test_spiral_data():
    X, y = make_spiral_data()

    # Check the number of examples and input features.
    assert X.shape == (900, 2)

    # Check that there is one label per example.
    assert y.shape == (900,)

    # Check that all three classes are present.
    assert set(np.unique(y)) == {0, 1, 2}

    # Check that the generated data contains no NaNs or infinities.
    assert np.all(np.isfinite(X))

    # Check that the labels are valid.
    assert np.all((y >= 0) & (y < 3))

    print("Spiral dataset test passed!")


if __name__ == "__main__":
    test_spiral_data()
