# Self-Pruning Neural Network

A NumPy-based neural network built from scratch, featuring a custom reverse-mode automatic differentiation engine, Adam optimizer, and Taylor-based weight pruning.

## Project Overview

This project explores how a neural network can learn while progressively removing low-importance connections under a target sparsity constraint.

The implementation uses Python and NumPy for tensor operations, gradient computation, optimization, training, and pruning. It does not rely on deep-learning frameworks such as PyTorch, TensorFlow, or JAX.

## Key Features

- Custom reverse-mode automatic differentiation engine.
- Tensor operations with gradient propagation and accumulation.
- Broadcasting support and matrix multiplication.
- ReLU and tanh activation functions.
- Stable softmax and cross-entropy loss.
- Custom Adam optimizer.
- Weight masks to disable pruned connections.
- Taylor-based connection importance estimation.
- Gradual pruning schedule.
- Spiral-dataset classification experiment.
- Training metrics saved to CSV and plotted for analysis.

## Project Structure

```text
self-pruning-network/
├── tensor.py
├── layers.py
├── model.py
├── optim.py
├── pruning.py
├── spiral_data.py
├── train.py
├── plot_learning_curve.py
├── learning_curve.csv
├── accuracy_curve.png
├── loss_curve.png
└── tests/
```

## Architecture

The classifier is a small multilayer perceptron:

Input features → Linear layer → ReLU → Linear layer → Class logits

The model receives two-dimensional spiral-dataset inputs and predicts one of three classes.

The neural-network operations and gradient engine are implemented using NumPy rather than a deep-learning framework.

## Automatic Differentiation

The custom `Tensor` class tracks values, parent operations, and gradients.

During the forward pass, operations build a computational graph. During the backward pass, the engine traverses that graph in reverse topological order and propagates gradients using the chain rule.

Gradients from multiple paths are accumulated so shared intermediate tensors receive contributions from every dependent operation.

Broadcasted operations reduce gradients back to the original operand shapes.

## Taylor-Based Pruning

### Connection importance

Magnitude-only pruning ranks connections using the absolute value of their weights. However, a large weight is not necessarily important to the model's loss, and a small weight can still be important.

This project uses a first-order Taylor approximation to estimate the effect of removing a connection.

For a weight \(w\), removing the connection changes it to zero:

\[
\Delta w = -w
\]

Using a first-order approximation of the loss:

\[
\Delta L \approx \frac{\partial L}{\partial w}\Delta w
\]

Therefore:

\[
\Delta L \approx -w\frac{\partial L}{\partial w}
\]

The absolute importance score is:

\[
I(w)=\left|w\frac{\partial L}{\partial w}\right|
\]

A lower score suggests that removing the connection may have a smaller first-order effect on the loss.

This is an approximation, not a guarantee: interactions between weights and higher-order effects can change the actual loss after pruning.

### Gradual pruning

Rather than removing many connections at the beginning of training, the pruning schedule progressively increases sparsity.

The target sparsity is 75%. The schedule starts at epoch 200 and reaches its target by epoch 580. Training then continues after the pruning schedule ends.

Gradual pruning gives the model time to learn useful representations before and during structural reduction.

### Masks and optimizer behavior

A binary mask identifies active and pruned weights. Masked weights should contribute zero to the forward computation and should not regrow through optimizer updates.

The implementation also handles gradients and Adam optimizer state for masked weights. These invariants should be verified by automated tests.

## Experimental Results

The spiral-dataset experiment uses a deterministic train/test split and reports training loss, training accuracy, test loss, and test accuracy.

| Experiment | Training epochs | Target sparsity | Test accuracy |
|---|---:|---:|---:|
| Dense baseline | 400 | 0% | 76.67% |
| Pruned model | 600 | 75% | 61.11% |
| Pruned model | 700 | 75% | 66.11% |

For the 700-epoch pruned experiment:

- Total weight connections: 320
- Active connections: 80
- Pruned connections: 240
- Actual weight sparsity: 75%
- Test accuracy: 66.11%

The 700-epoch pruned experiment improved test accuracy by 5 percentage points compared with the 600-epoch pruned experiment.

Compared with the dense baseline, its test accuracy was 10.56 percentage points lower. This demonstrates an accuracy-versus-sparsity trade-off in the current experiment.

These results represent individual runs; multiple seeds and repeated experiments would be needed to evaluate robustness.

## Running the Project

Activate your Python virtual environment, then install the required dependencies if they are not already installed.

Run the training experiment:

```bash
python train.py
```

Run the test suite:

```bash
python -m pytest -q
```

The training process saves learning metrics to `learning_curve.csv`. The plots `accuracy_curve.png` and `loss_curve.png` can be used to inspect model behavior.

## Limitations

- The Taylor score is a first-order approximation of loss change.
- Results may vary with random initialization and training settings.
- Unstructured weight masks demonstrate connection sparsity, but do not automatically guarantee faster inference or reduced physical memory usage without sparse execution support.
- Further evaluation across random seeds and sparsity targets is needed.

## Technologies

- Python
- NumPy
- pytest
- Matplotlib
- scikit-learn, if used for dataset utilities

## License

This project was created as an engineering take-home exercise.
