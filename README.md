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
â”œâ”€â”€ tensor.py
â”œâ”€â”€ layers.py
â”œâ”€â”€ model.py
â”œâ”€â”€ optim.py
â”œâ”€â”€ pruning.py
â”œâ”€â”€ spiral_data.py
â”œâ”€â”€ train.py
â”œâ”€â”€ plot_learning_curve.py
â”œâ”€â”€ learning_curve.csv
â”œâ”€â”€ accuracy_curve.png
â”œâ”€â”€ loss_curve.png
â””â”€â”€ tests/
```

## Architecture

The classifier is a small multilayer perceptron:

Input features â†’ Linear layer â†’ ReLU â†’ Linear layer â†’ Class logits

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

A binary mask identifies active and pruned weights. Masked weights contribute zero to the effective weight matrix and remain zero after optimizer updates.

The Adam optimizer masks gradients and clears the first and second moment states for pruned connections. This prevents stale optimizer momentum from updating weights that have been removed.

Automated tests verify that a masked connection has zero effective weight, receives zero gradient through the layer's forward computation, and remains zero after an Adam update. The tests also verify that the corresponding Adam moment states are cleared.

The full test suite currently passes 29 tests (latest verified run).


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

## Taylor vs Magnitude Pruning Comparison

Both pruning methods were evaluated using the same trained dense baseline for each seed. Each baseline was trained for 700 epochs. The methods were tested at 0%, 50%, 75%, 90%, and 95% target sparsity, using seeds 42, 43, and 44 and the same deterministic train/test split.

| Target sparsity | Taylor accuracy | Magnitude accuracy |
|---:|---:|---:|
| 0% | 82.78% +/- 7.26% | 82.78% +/- 7.26% |
| 50% | 52.96% +/- 5.56% | 56.48% +/- 3.90% |
| 75% | 39.26% +/- 4.46% | 42.41% +/- 2.25% |
| 90% | 32.04% +/- 4.72% | 34.81% +/- 4.32% |
| 95% | 34.63% +/- 3.90% | 33.15% +/- 4.49% |

Values are mean test accuracy +/- sample standard deviation across three seeds.

Magnitude pruning achieved higher mean accuracy at 50%, 75%, and 90% sparsity in this experiment. Taylor pruning performed slightly better at 95%. Neither method consistently outperformed the other at every sparsity level. Accuracy dropped substantially as sparsity increased.

The comparison graph is saved as `pruning_comparison.png`. Per-seed results are stored in `pruning_comparison_results.csv`, and aggregated results are stored in `pruning_comparison_summary.csv`.

These findings are limited to three seeds and one synthetic spiral dataset; they are observations from this experiment, not a universal ranking of pruning methods.

## Sparse Inference Cost Measurement

At 75% sparsity, active multiply-accumulate operations (MACs) decreased from 40,960 to 10,240 per batch, a 75% reduction.

However, dense NumPy inference took approximately 0.095 ms per call, while the Python-loop sparse implementation took approximately 0.744 ms per call. Thus, sparse inference was slower despite performing fewer active MACs.

Fewer active connections do not automatically mean faster inference. Efficient sparse execution and further benchmarking are needed before claiming a real speedup.

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


