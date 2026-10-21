# Design Document --- Self-Pruning Network

## Scope and implementation note

This document describes the mathematical design and engineering
considerations for a NumPy-based neural network with a custom
reverse-mode autodiff engine and connection pruning. The formulas below
define the intended behavior. Before submission, the implementation
should be checked against these definitions---especially the exact
importance-score formula, the point at which the mask is applied, and
optimizer-state handling. Do not claim a test or benchmark passed unless
it was run against the submitted commit.

## 1. Derive the importance criterion and explain why it approximates the loss change

### Objective

Pruning removes a connection by setting its weight to zero. We want to
rank connections by the estimated damage that removing each connection
would cause, then remove low-impact connections subject to a target
sparsity or compute budget.

Let the model's loss be (L(`\mathbf{w}`{=tex})), where (w_i) is one
scalar connection weight. Removing this connection changes the weight
from (w_i) to zero:

\[ `\Delta `{=tex}w_i = 0-w_i=-w_i. \]

A first-order Taylor expansion around the current parameters gives

\[ L(`\mathbf{w}`{=tex}+`\Delta `{=tex}`\mathbf{w}`{=tex})
`\approx`{=tex}
L(`\mathbf{w}`{=tex})+`\nabla`{=tex}\_{`\mathbf{w}`{=tex}}L(`\mathbf{w}`{=tex})\^`\top`{=tex}`\Delta`{=tex}`\mathbf{w}`{=tex}.
\]

If only parameter (i) changes, the estimated loss change is

\[ `\Delta `{=tex}L_i `\approx`{=tex}
`\frac{\partial L}{\partial w_i}`{=tex}(-w_i)
=-w_i`\frac{\partial L}{\partial w_i}`{=tex}. \]

Because a ranking score should measure the size of the estimated change
regardless of sign, the first-order Taylor importance score is

\[
I_i=`\left`{=tex}\|w_i`\frac{\partial L}{\partial w_i}`{=tex}`\right`{=tex}\|.
\]

A small (I_i) means that removing the connection is predicted to have a
small first-order effect on the loss at the current parameter values.
The pruning procedure can rank weights by (I_i) and consider the
smallest scores first. If the implementation averages scores over
batches, it should use a representative calibration/training sample and
document how the aggregation is done.

### Why the approximation is useful---and limited

The criterion uses both the weight and the loss gradient. The weight
gives the size of the parameter change caused by deletion; the gradient
estimates local sensitivity of the loss to that change. Therefore, two
weights with equal magnitude can receive different importance scores if
the loss is more sensitive to one of them.

This is a local, first-order approximation, not an exact guarantee. It
omits second- and higher-order terms, interactions between multiple
deleted weights, and later recovery through retraining. Scores may also
be noisy if calculated from a small or unrepresentative batch. For these
reasons, prune incrementally where practical and evaluate accuracy and
loss on held-out data after pruning.

A magnitude-pruning baseline ranks weights using

\[ I_i\^{`\mathrm{mag}`{=tex}}=\|w_i\|. \]

Magnitude pruning is cheap and does not need gradients, but it ignores
loss sensitivity. Taylor importance is more informed by the current loss
landscape, at the cost of gradient computation and potentially noisier
scores. For a set of simultaneously removed weights, the total loss
change is not generally equal to the sum of individual first-order
scores because interactions and higher-order terms are omitted.

### Compute-budget interpretation

Sparsity is not identical to speed. Removing a fraction of weights
reduces the number of nonzero parameters, but a dense matrix
multiplication containing zeros may still perform dense arithmetic. A
compute-budget claim should therefore distinguish parameter sparsity,
estimated multiply-accumulate operations (MACs), and measured end-to-end
latency. Any reported Pareto curve should state which quantities are
measured and which are estimated.

## 2. What is the gradient of a masked weight, and why is this the right choice?

### Hard-mask definition

Let (w_i) be a stored parameter and (m_i`\in`{=tex}{0,1}) be a binary
mask. The effective weight used in the forward computation is

\[ `\widetilde{w}`{=tex}\_i=m_iw_i. \]

The loss depends on the stored weight through this effective weight.
Applying the chain rule,

\[ `\frac{\partial L}{\partial w_i}`{=tex}
=`\frac{\partial L}{\partial \widetilde{w}_i}`{=tex}
`\frac{\partial \widetilde{w}_i}{\partial w_i}`{=tex}
=m_i`\frac{\partial L}{\partial \widetilde{w}_i}`{=tex}. \]

Therefore, with a hard mask treated as fixed during this backward pass,

\[ `\frac{\partial L}{\partial w_i}`{=tex}=
```{=tex}
\begin{cases}
0, & m_i=0,\\
\frac{\partial L}{\partial \widetilde{w}_i}, & m_i=1.
\end{cases}
```
\]

This is the chain-rule gradient for the parameterization
(`\widetilde{w}`{=tex}=m`\odot `{=tex}w). A masked connection is absent
from the effective model, so changing its stored value cannot change the
current loss; its gradient with respect to the stored value must be
zero. Active weights receive the ordinary gradient.

This conclusion assumes that the mask is actually applied in the
differentiable forward path, or that the implementation explicitly
enforces the equivalent gradient rule. Merely setting a stored weight to
zero once is not equivalent to maintaining a hard mask.

### Optimizer updates and invariants

A robust implementation should preserve these invariants after every
optimizer step:

1.  The mask is the source of truth for whether a connection is active.
2.  The forward pass uses the effective masked weight, so inactive
    connections contribute zero.
3.  Inactive weights receive zero gradients under the hard-mask rule.
4.  Inactive stored weights remain exactly zero after optimizer updates.
5.  Optimizer state for inactive weights is also handled consistently.

For Adam, the first- and second-moment state associated with inactive
entries should be masked or cleared, and the parameter mask should be
reapplied after the update. Otherwise, stale momentum or weight decay
may change a stored zero even though the connection is intended to
remain disabled. The exact implementation must also account for the
optimizer's update order and any weight-decay option it supports.

A masked-weight correctness test should check the forward effect,
gradient, and post-update state: the connection contributes zero, its
gradient is zero, its stored value remains zero after optimization, and
its optimizer state cannot reactivate it. The test should also check
that unmasked weights continue to learn.

### Alternative gradient definitions

A straight-through estimator or a trainable continuous gate would define
a different optimization problem and could use a different gradient
rule. This document specifies a fixed binary hard mask; it does not
assume a straight-through estimator. The submitted code and tests must
use the same definition.

## 3. Where can the autodiff engine bottleneck, and how could it be optimized?

### Engine behavior

A reverse-mode autodiff engine records tensor operations and their
parent relationships during the forward pass. During `backward()`, it
traverses the computation graph in reverse topological order and
accumulates each node's gradient into its parents. Gradient accumulation
is necessary when a tensor contributes to more than one downstream
operation.

For this small NumPy engine, likely bottlenecks include:

-   **Python overhead:** creating Tensor objects and invoking a Python
    backward function for each small operation can dominate the cost of
    small workloads.
-   **Graph traversal:** building a topological ordering and visiting
    graph nodes adds overhead, especially when the graph has many small
    operations.
-   **Intermediate memory:** parent references and saved forward values
    must remain alive until backward uses them, increasing peak memory.
-   **Temporary arrays and copies:** chained NumPy expressions can
    allocate intermediate arrays and increase memory bandwidth.
-   **Broadcast-gradient reduction:** gradients must be reduced along
    broadcast dimensions to match the original operand shape. Incorrect
    or repeated shape logic can hurt both correctness and performance.
-   **Dense arithmetic with masks:** multiplying a dense matrix by a
    mask does not automatically make matrix multiplication sparse. The
    engine may still compute operations for zeroed connections.

These are plausible bottlenecks for this design; the actual dominant
bottleneck should be established with profiling rather than asserted
without measurements.

### Optimization strategy

1.  **Profile representative workloads first.** Measure forward and
    backward time separately, graph construction, peak memory, and
    individual matrix operations. Use repeatable runs and report shapes,
    batch sizes, warm-up, and summary statistics.
2.  **Vectorize operations.** Use NumPy's optimized array and matrix
    kernels rather than Python loops over individual examples or
    weights.
3.  **Reduce graph overhead.** Avoid unnecessary Tensor nodes and
    retained values; use compact operation records where appropriate;
    release graph references once backward is complete and they are no
    longer needed.
4.  **Centralize broadcasting logic.** Implement a tested `unbroadcast`
    helper that sums extra leading dimensions and dimensions that were
    size one in the original operand.
5.  **Avoid unsafe in-place changes.** Buffer reuse may reduce
    allocations, but must not overwrite values required by backward or
    introduce aliasing bugs.
6.  **Optimize only measured hotspots.** Compare each change against a
    fixed benchmark and rerun finite-difference gradient checks and
    masked-weight tests after changes.
7.  **Evaluate sparse inference separately.** Compare dense masked
    matrix multiplication with a compact sparse representation or
    suitable sparse kernel. Sparse indexing overhead can make sparse
    inference slower for small matrices or unsuitable batch sizes, so
    measure latency and throughput on target hardware.

Correctness and reproducibility come first. A custom Python/NumPy engine
is useful for understanding autodiff mechanics, but it should not be
expected to match the performance of highly optimized deep-learning
runtimes without substantial additional engineering.

## 4. How would you serve a self-pruned model in a real multi-tenant inference service at scale?

### A. Offline build and immutable model artifacts

Training, importance calculation, pruning, and evaluation should run
offline---not inside the online request path. After pruning, evaluate
the model on a held-out set and export a versioned, immutable artifact
containing:

-   model architecture and tensor shapes;
-   trained parameters and, where needed, masks or compact sparse
    indices;
-   preprocessing and postprocessing configuration;
-   model version, metrics, and artifact checksum;
-   runtime compatibility information.

Validate that the exported artifact produces outputs consistent with the
evaluated model. Store it in a versioned model registry or artifact
store, and retain a rollback-capable previous version.

### B. Request path

A typical request flow is:

1.  Authenticate the caller and resolve the tenant identity.
2.  Validate the input schema, payload size, and requested model
    version.
3.  Apply the versioned preprocessing pipeline.
4.  Route the request to a worker that has the required model loaded.
5.  Execute inference using the immutable model artifact.
6.  Apply output validation and any tenant-specific response policy.
7.  Return the response with a request ID and appropriate latency/error
    metadata.

Pruning and training remain separate offline workflows. The serving
layer should not mutate shared model parameters in response to a tenant
request.

### C. Multi-tenant isolation and fairness

-   **Authentication and authorization:** check that each tenant is
    permitted to access the requested model and data.
-   **Quotas:** enforce per-tenant request-rate, concurrency,
    payload-size, and compute limits.
-   **Fair scheduling:** use bounded queues, deadlines, backpressure,
    and admission control so one noisy tenant cannot monopolize workers.
-   **Data isolation:** avoid mixing tenant-specific inputs, caches,
    outputs, or logs; scope caches by tenant and relevant model/version
    keys where their contents could reveal tenant data.
-   **Resource isolation:** separate workloads or worker pools when
    security, latency, or regulatory requirements justify it.
-   **Safe caching:** load immutable model versions once per worker and
    reuse them; avoid loading an artifact for every request.

### D. Scaling and reliability

Scale workers based on observed load, queue depth, resource utilization,
and service-level objectives. Use timeouts, bounded retries for safe
operations, health checks, graceful shutdown, and overload protection.
Roll out new model versions gradually using canaries or staged traffic;
compare quality, error rate, and latency before full deployment. Keep a
tested rollback path.

### E. Observability and security

Track p50, p95, and p99 latency, throughput, error rate, queue time,
CPU/GPU and memory usage, model version, and per-tenant quota
consumption. Use request IDs and structured logs, while avoiding
sensitive payload logging by default. Protect traffic and stored
artifacts, apply least-privilege access, validate artifacts before
loading, and audit model publication and access.

### F. Measure the actual sparsity--latency trade-off

A model with fewer nonzero weights is not necessarily faster. Dense
kernels can still process zero entries, while sparse kernels introduce
indexing and scheduling overhead. Benchmark end-to-end latency and
throughput for representative input shapes, batch sizes, and target
hardware. Include preprocessing, batching, routing, and queueing
overhead, and compare dense and sparse implementations. Choose the
serving representation based on measured performance, not sparsity
percentage alone.

Batching can improve throughput but increase tail latency; sharing model
instances can reduce memory use but requires strong tenant isolation;
aggressive autoscaling can improve responsiveness but raise cost. Define
explicit latency, availability, and error-rate objectives, then validate
the service under realistic mixed-tenant load.

## 5. Submission validation checklist

Before treating the repository as ready for submission, verify the
following against the final committed code:

-   [ ] Gradient-check tests cover every required operation and
    broadcasting behavior.
-   [ ] Shared-parent gradient accumulation is tested.
-   [ ] Masked-weight tests verify forward behavior, the documented
    gradient, and persistence of the mask through optimizer updates.
-   [ ] The exact importance formula in `pruning.py` matches Section 1,
    or Section 1 is updated to describe the actual formula.
-   [ ] README installation instructions work in a clean environment.
-   [ ] README provides separate reproducible commands for Part 1
    gradient checks, Part 2 training, Part 3 self-pruning, and Part 4
    Pareto sweep.
-   [ ] Part 4 raw numbers and plots are committed and can be
    regenerated from the experiment code.
-   [ ] Reported accuracy, sparsity, MAC estimates, and timing numbers
    specify how they were measured.
-   [ ] All reported test and benchmark results were produced from the
    final submitted commit.
