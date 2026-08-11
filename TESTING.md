# TESTING.md

## Purpose

This file defines the validation procedure for the current NumPy reference
implementation, the original Work Packages, EWP1 backend abstraction, and
optional CuPy runtime-compatibility checks. It also records which existing
tests should later serve as the NumPy reference for broader CuPy
numerical-equivalence testing.

`tests/test_backend.py` is NumPy-only smoke coverage for the backend
abstraction. `tests/test_cupy_backend_runtime.py` contains optional CuPy
runtime tests and must skip cleanly when CuPy, CUDA runtime access, or a
visible CUDA GPU is unavailable.

## Current Test State

Latest verified local state:

```text
Non-CuPy local regression: 220 passed, 19 deselected
Full local suite: 220 passed, 19 skipped
EWP2-B local slice on this machine: 2 skipped because cupy is not installed
EWP2-C local slice on this machine: 2 skipped because cupy is not installed
EWP2-D local slice on this machine: 3 skipped because cupy is not installed
CuPy runtime/equivalence slices on this machine: 19 skipped because cupy is not installed
```

Latest verified real GPU EWP1-B state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
CuPy runtime slice: 6 passed
Non-data cluster regression: 223 passed, 3 deselected in 8.20s
```

Compatibility is recorded only for the tested GPU/CUDA/CuPy/Python
configuration above.

Latest verified real GPU EWP2-A state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
EWP2-A layer/loss equivalence: 6 passed in 0.67s
Non-data cluster regression: 229 passed, 3 deselected in 7.86s
```

NumPy remains the authoritative correctness reference. Compatibility is
recorded only for the tested GPU/CUDA/CuPy/Python configuration above.

Latest verified real GPU EWP2-B state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
EWP2-B full-model training equivalence: 2 passed
Non-data cluster regression: 231 passed, 3 deselected in 9.23s
```

NumPy remains the authoritative correctness reference. Compatibility is
recorded only for the tested GPU/CUDA/CuPy/Python configuration above.

Latest verified real GPU EWP2-C state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
EWP2-C input-gradient/FGSM equivalence: 2 passed in 1.14s
Non-data cluster regression: 233 passed, 3 deselected in 8.29s
```

NumPy remains the authoritative correctness reference. Compatibility is
recorded only for the tested GPU/CUDA/CuPy/Python configuration above.

Standard offline CI command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
```

Full local command when CIFAR-10 data is available:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

Data-only command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m requires_data
```

CuPy runtime-compatibility command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_backend_runtime.py -rs
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the offline
subset on Python 3.13.

## General Validation Rules

Before and after important implementation changes:

```bash
python --version
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
git diff --check
git status -sb
```

For documentation-only changes, run the full test suite when feasible and
always run `git diff --check`.

## WP0: Project Orientation and Planning Validation

Goal:

Check that planning files and documentation deliverables exist and reflect the
current implementation.

Suggested checks:

```bash
ls AGENTS.md WP_PLAN.md TESTING.md
ls deliverables/WP0/
```

Expected results:

* `AGENTS.md`, `WP_PLAN.md`, and `TESTING.md` exist.
* WP status in `WP_PLAN.md` matches the current repository.
* WP0 deliverables record method selection and metrics from repository
  evidence, with TODO markers only where external citation details are still
  missing.

## WP1: Data Pipeline Validation

Goal:

Check CIFAR-10 loading, preprocessing, batching, and reproducibility.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_data_pipeline.py -v
.venv/bin/python -m experiments.check_data_pipeline
```

Expected results:

* CIFAR-10 train/test arrays have shapes `(50000, 3, 32, 32)` and
  `(10000, 3, 32, 32)` when local data is present.
* Image dtype is `float32` and values are in `[0, 1]`.
* Labels are `int64` class IDs in `[0, 9]`.
* Fixed seeds reproduce the first shuffled batch.
* The data sanity script writes `results/figures/cifar10_sample_batch.png`.

Tests:

```text
tests/test_data_pipeline.py
```

## WP2: Compact CNN Forward Pass Validation

Goal:

Check the manual forward implementation for layers, model, and loss.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_forward.py tests/test_losses.py -v
```

Expected results:

* `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, and `Linear` match deterministic
  references.
* `CompactCNN.forward` accepts `(N, 3, 32, 32)` inputs and returns `(N, 10)`.
* Outputs are finite.
* Fixed-seed initialization is reproducible.
* Invalid shapes are rejected.
* Softmax Cross-Entropy forward is numerically stable.

Tests:

```text
tests/test_forward.py
tests/test_losses.py
```

## WP3: Manual Backward Validation

Goal:

Check layer-level and model-level manual backward propagation.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_layers.py tests/test_backward.py tests/test_integration.py tests/test_losses.py -v
```

Expected results:

* `Linear.backward`, `ReLU.backward`, `Flatten.backward`,
  `MaxPool2D.backward`, and `Conv2D.backward` return expected shapes and
  deterministic values.
* `Conv2D.backward` supports stride and padding behavior covered by tests.
* `SoftmaxCrossEntropyLoss.backward` matches deterministic references.
* `CompactCNN.backward` returns finite input gradients.
* Parameter gradients match parameter shapes and are finite.

Tests:

```text
tests/test_layers.py
tests/test_backward.py
tests/test_integration.py
tests/test_losses.py
```

## WP4: Gradient Check and Input-Gradient Validation

Goal:

Compare selected manual gradients against finite differences and validate
loss-to-input gradients.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_gradient_check.py tests/test_input_gradients.py -v
```

Expected results:

* `Linear`, `Conv2D`, and `SoftmaxCrossEntropyLoss` gradients match centered
  finite-difference checks.
* Relative-error threshold remains `1e-4` for the existing checks.
* Input gradients are finite, deterministic, and do not update parameters.
* Input-gradient maps are normalized per image and handle zero gradients.

Tests:

```text
tests/test_gradient_check.py
tests/test_input_gradients.py
```

## WP5: Baseline Training Validation

Goal:

Validate optimizer, training/evaluation helpers, checkpointing, metrics,
plotting, and baseline runners.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py tests/test_training.py tests/test_checkpointing.py tests/test_metrics.py tests/test_plotting.py tests/test_config.py tests/test_baseline_runner.py tests/test_cifar10_baseline_runner.py tests/test_portfolio_baseline_runner.py -v
```

Expected results:

* SGD updates match deterministic hand-computed values.
* `train_step`, `train_batches`, `evaluate_batch`, and `evaluate_batches`
  produce finite sample-weighted metrics.
* Checkpoint save/load restores model parameters and logits.
* JSON metrics are deterministic and human-readable.
* Baseline plots and confusion-matrix plots are generated.
* Synthetic and controlled real-data runners create expected artifacts.
* The stronger 4096/1024/1024 baseline runner is tested with synthetic
  monkeypatched data, not by running a full expensive training job in tests.

Tests:

```text
tests/test_optimizer.py
tests/test_training.py
tests/test_checkpointing.py
tests/test_metrics.py
tests/test_plotting.py
tests/test_config.py
tests/test_baseline_runner.py
tests/test_cifar10_baseline_runner.py
tests/test_portfolio_baseline_runner.py
```

## WP6: Runtime Bottleneck Validation

Goal:

Validate the focused `Conv2D.backward` optimization and record runtime
measurements.

Profiling command:

```bash
.venv/bin/python -m experiments.runtime.profile_wp6
```

Current audit profiler output:

```text
conv2d_forward_seconds=0.000115542
conv2d_backward_seconds=0.000345306
train_step_seconds=0.003370375
```

Correctness commands:

```bash
.venv/bin/python -m pytest tests/test_layers.py -v -k conv2d_backward
.venv/bin/python -m pytest tests/test_gradient_check.py -v -k "conv2d or compact_cnn_input_gradient"
.venv/bin/python -m pytest tests/test_backward.py tests/test_integration.py -v
```

Expected results:

* `Conv2D.backward` remains correct after optimization.
* Runtime measurements are treated as local profiling data, not broad hardware
  benchmarks.
* CuPy/GPU work is not part of WP6 validation.

## WP7: FGSM Attack and Input-Gradient Visualization

Goal:

Validate minimal FGSM behavior and small qualitative visualization generation.

Suggested commands:

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_input_gradients.py tests/test_fgsm.py tests/test_visualization.py tests/test_fgsm_examples.py -v
```

Expected results:

* FGSM preserves shape and clips images to `[0, 1]`.
* `epsilon=0` leaves inputs unchanged.
* Perturbations satisfy the `L_inf` bound.
* Attack generation does not update model parameters.
* Qualitative PNG helpers create deterministic non-empty files.

Tests:

```text
tests/test_input_gradients.py
tests/test_fgsm.py
tests/test_visualization.py
tests/test_fgsm_examples.py
```

## WP8: FGSM Robustness Evaluation Validation

Goal:

Validate clean-vs-FGSM evaluation, epsilon sweeps, aggregation, plots,
representative metadata, and quantitative runner artifacts.

Suggested commands:

```bash
MPLCONFIGDIR=/tmp/cnn-wp8-matplotlib .venv/bin/python -m pytest tests/test_fgsm_evaluation.py tests/test_fgsm_robustness_runner.py tests/test_fgsm_quantitative_runner.py tests/test_plotting.py -v
```

Expected results:

* Single-batch metrics match hand-computed references.
* Multi-batch aggregation uses raw sample counts and handles different batch
  sizes.
* `epsilon=0` preserves adversarial predictions and accuracy.
* Evaluation does not update model parameters.
* Epsilon sweep preserves input epsilon order.
* Historical WP8 config defaults to 32 samples, batch size 8, and epsilon
  values `0/255` through `16/255`.
* Quantitative FGSM config defaults to 1024 samples, batch size 32, and
  epsilon values `[0, 2/255, 4/255, 8/255, 16/255]`.

Tests:

```text
tests/test_fgsm_evaluation.py
tests/test_fgsm_robustness_runner.py
tests/test_fgsm_quantitative_runner.py
tests/test_plotting.py
```

## WP9-WP12: Deferred Attack Work

PGD and black-box attacks are intentionally deferred.

Current expected result:

* No PGD implementation exists.
* No PGD evaluation exists.
* No black-box attack implementation exists.
* No query-count evaluation exists.

Do not add PGD, PGD tests, black-box attacks, or black-box tests during the
current CuPy preparation cycle.

## WP13: Grad-CAM Implementation Validation

Goal:

Validate the implemented Grad-CAM core and visualization helpers.

Suggested commands:

```bash
MPLCONFIGDIR=/tmp/cnn-gradcam-matplotlib .venv/bin/python -m pytest tests/test_gradcam.py tests/test_gradcam_visualization.py -v
```

Expected results:

* `compute_gradcam` returns heatmaps with expected shape and finite `[0, 1]`
  range.
* Default target class uses the predicted class.
* Explicit target classes are accepted and validated.
* Grad-CAM is deterministic for fixed inputs and targets.
* Model parameters and classifier gradient buffers are restored.
* `CompactCNN.gradcam_activation` requires a preceding forward call and returns
  a copy.
* Heatmap resizing and overlays produce finite display-ready arrays.
* Grad-CAM figure helpers create non-empty PNG files.

Tests:

```text
tests/test_gradcam.py
tests/test_gradcam_visualization.py
```

## WP14: FGSM-Only Grad-CAM Analysis Validation

Goal:

Validate clean-vs-FGSM Grad-CAM analysis helpers.

Current scope:

* FGSM only.
* PGD and black-box Grad-CAM comparisons are absent because WP9-WP12 are
  deferred.

Suggested command:

```bash
MPLCONFIGDIR=/tmp/cnn-gradcam-matplotlib .venv/bin/python -m pytest tests/test_gradcam.py tests/test_gradcam_visualization.py -v
```

The full runner:

```bash
MPLCONFIGDIR=/tmp/cnn-gradcam-matplotlib .venv/bin/python -m experiments.gradcam.generate_adversarial_comparisons
```

The full runner requires local CIFAR-10 data and the local ignored checkpoint
`results/baseline/portfolio_baseline_best.npz`. It should not be run as part
of standard offline CI.

## WP15: Integration and Reproducibility Validation

Goal:

Check that documentation, tests, CI, and result artifacts remain consistent.

Suggested commands:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
git diff --check
git status -sb
```

Expected results:

* Full suite passes locally.
* Offline CI subset passes without local CIFAR-10.
* Documentation links match tracked or explicitly documented local artifacts.
* Large ignored artifacts are clearly documented as regenerable or external.

## CuPy Runtime Compatibility Tests

The optional EWP1-B runtime tests are:

```text
tests/cupy_test_utils.py
tests/conftest.py
tests/test_cupy_backend_runtime.py
```

They validate:

* CuPy import, CUDA runtime query, visible GPU count, GPU name, and simple
  allocation/computation.
* Backend primitive compatibility for the tensor operations used by the
  migrated path.
* `sliding_window_view`, `einsum(..., optimize=True)`, `cupy.add.at`, and
  `divide_where(...)`.
* First Conv2D forward/backward NumPy/CuPy equivalence slice with `rtol=1e-5`
  and `atol=1e-6`, matching the expected float32 scale of the tested
  operations.

These tests are not cluster integration and do not run large CIFAR-10
experiments.

Verified EWP1-B GPU results:

```text
python -m pytest -q tests/test_cupy_backend_runtime.py -rs
6 passed
```

The tested environment was `NVIDIA GeForce RTX 2080 Ti`, CUDA Toolkit `12.5`,
CuPy `14.1.1`, Python `3.12`, with one GPU allocated through Slurm.

## Cluster CIFAR-10 Dataset Staging

The current cluster CIFAR-10 archive is not a valid data-test signal:

```text
path: data/raw/cifar-10-python.tar.gz
size: 37M
observed MD5: 352dcf059b8b606c932d1db9b8c351a9
expected MD5: c58f30108f718f92721af3b95e74349a
```

The `requires_data` tests fail on the cluster before any CuPy numerical path is
exercised. Do not change the expected checksum, disable archive validation,
weaken data tests, or modify the CIFAR-10 loader to hide this issue. Restage
the dataset before treating data-dependent cluster tests or experiments as
valid.

## NumPy Reference Tests for Future CuPy Equivalence

The following existing tests should serve as the NumPy reference before broader
EWP2 CuPy equivalence tests are introduced:

```text
tests/test_forward.py
tests/test_layers.py
tests/test_losses.py
tests/test_backward.py
tests/test_integration.py
tests/test_gradient_check.py
tests/test_input_gradients.py
tests/test_fgsm.py
tests/test_fgsm_evaluation.py
tests/test_fgsm_quantitative_runner.py
tests/test_gradcam.py
tests/test_backend.py
tests/test_cupy_layer_loss_equivalence.py
tests/test_cupy_model_training_equivalence.py
tests/test_cupy_fgsm_equivalence.py
tests/test_cupy_robustness_equivalence.py
```

Recommended future equivalence coverage:

* Forward logits.
* Layer backward gradients.
* Loss values and logits gradients.
* CompactCNN training-step logits, loss, parameter gradients, and one SGD
  update.
* Model loss-to-input gradients and FGSM adversarial examples.
* FGSM robustness metrics and epsilon sweeps.
* Checkpoint load into NumPy and CuPy model instances.

## EWP2-A Layer and Loss Equivalence Tests

The optional EWP2-A tests are:

```text
tests/test_cupy_layer_loss_equivalence.py
```

Status: COMPLETE.

They validate NumPy/CuPy equivalence for:

* `ReLU.forward` and `ReLU.backward`.
* `MaxPool2D.forward` and `MaxPool2D.backward`, including `add.at` and
  first-maximum tie semantics.
* `Flatten.forward` and `Flatten.backward`.
* `Linear.forward` and `Linear.backward`, including `dx`, `dw`, and `db`.
* `SoftmaxCrossEntropyLoss.forward` and `SoftmaxCrossEntropyLoss.backward`.

The tests use `rtol=1e-5` and `atol=1e-6` for float32 tensor comparisons, and
`rtol=1e-6` and `atol=1e-7` for the scalar softmax cross-entropy loss.

Validated on the GPU cluster with:

```text
python -m pytest -q tests/test_cupy_layer_loss_equivalence.py -rs
6 passed in 0.67s

python -m pytest -q -m "not requires_data"
229 passed, 3 deselected in 7.86s
```

EWP2 is not complete. CompactCNN training-path equivalence is complete in
EWP2-B, input-gradient and FGSM equivalence are complete in EWP2-C, and
robustness metrics, epsilon sweeps, checkpoint, and broader equivalence remain
future EWP2 work.

## EWP2-B CompactCNN Training-Path Equivalence Tests

The optional EWP2-B tests are:

```text
tests/test_cupy_model_training_equivalence.py
```

Status: COMPLETE.

They validate NumPy/CuPy equivalence for:

* Deterministic `CompactCNN` parameter synchronization before comparison.
* Identical deterministic input tensors, labels, and optimizer settings.
* Full-model logits from `CompactCNN.forward`.
* `SoftmaxCrossEntropyLoss.forward` scalar loss and `backward` logits
  gradient.
* `CompactCNN.backward` gradients for `conv1.weights`, `conv1.bias`,
  `conv2.weights`, `conv2.bias`, `classifier.weights`, and
  `classifier.bias`.
* One real `SGD.step()` update for every trainable parameter.
* The public `train_step(...)` helper for the same single-batch
  `forward -> loss -> backward -> SGD` path.

The tests use deterministic synthetic inputs and labels. They do not require
CIFAR-10 data, checkpoint files, input-gradient equivalence, FGSM, robustness
sweeps, PGD, or cluster-runner infrastructure.

The tests use `rtol=1e-5` and `atol=1e-6` for float32 tensor comparisons, and
`rtol=1e-6` and `atol=1e-7` for scalar softmax cross-entropy losses.

Local non-CuPy validation command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_model_training_equivalence.py -rs
```

Expected local result on machines without CuPy/CUDA:

```text
2 skipped
```

GPU validation command:

```bash
python -m pytest -q tests/test_cupy_model_training_equivalence.py -rs
```

Validated on the GPU cluster with:

```text
python -m pytest -q tests/test_cupy_model_training_equivalence.py -rs
2 passed

python -m pytest -q -m "not requires_data"
231 passed, 3 deselected in 9.23s
```

The validated environment was `NVIDIA GeForce RTX 2080 Ti`, CUDA Toolkit
`12.5`, CuPy `14.1.1`, Python `3.12`, with one GPU allocated through Slurm.
Input-gradient and FGSM equivalence are covered by EWP2-C.

## EWP2-C Input-Gradient and FGSM Equivalence Tests

The optional EWP2-C tests are:

```text
tests/test_cupy_fgsm_equivalence.py
```

Status: COMPLETE.

They validate NumPy/CuPy equivalence for:

* Deterministic `CompactCNN` parameter synchronization before attack-path
  comparison.
* Identical deterministic clean inputs, labels, epsilon, and clipping bounds.
* Production `compute_input_gradient(...)` loss-to-input gradients.
* Production `fgsm_attack(...)` with `epsilon=0`.
* Production `fgsm_attack(...)` with a nonzero epsilon.
* FGSM shape, `L_inf` perturbation bound, and `[0, 1]` clipping semantics.
* Adversarial images.
* Adversarial `CompactCNN.forward` logits and exact predicted classes.
* End-to-end attack path:
  `clean input -> input gradient -> FGSM image -> adversarial forward`.
* Parameter preservation after input-gradient computation, FGSM generation,
  and adversarial forward evaluation.

The tests use deterministic synthetic inputs and labels. They do not require
CIFAR-10 data, robustness sweeps, PGD, checkpoint files, or cluster-runner
infrastructure.

The tests use `rtol=1e-5` and `atol=1e-6` for input-gradient,
adversarial-image, and adversarial-logit comparisons. Predicted classes are
compared exactly.

Local non-CuPy validation command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_fgsm_equivalence.py -rs
```

Expected local result on machines without CuPy/CUDA:

```text
2 skipped
```

GPU validation command:

```bash
python -m pytest -q tests/test_cupy_fgsm_equivalence.py -rs
```

Validated on the GPU cluster with:

```text
python -m pytest -q tests/test_cupy_fgsm_equivalence.py -rs
2 passed in 1.14s

python -m pytest -q -m "not requires_data"
233 passed, 3 deselected in 8.29s
```

The validated environment was `NVIDIA GeForce RTX 2080 Ti`, CUDA Toolkit
`12.5`, CuPy `14.1.1`, Python `3.12`, with one GPU allocated through Slurm.
This is not robustness metric or epsilon-sweep equivalence; those remain
future EWP2-D work.

## EWP2-D Robustness and Epsilon-Sweep Equivalence Tests

The optional EWP2-D tests are:

```text
tests/test_cupy_robustness_equivalence.py
```

Status: IMPLEMENTED LOCALLY / GPU VALIDATION PENDING.

They validate NumPy/CuPy equivalence for:

* Production `evaluate_fgsm_batch(...)` single-batch metrics and raw counts.
* Production `evaluate_fgsm_batches(...)` multi-batch raw-count aggregation
  across different batch sizes.
* Production `evaluate_fgsm_epsilon_sweep(...)` using ordered epsilons
  `0/255`, `4/255`, and `8/255`.
* Exact sample counts, clean-correct counts, adversarial-correct counts,
  clean-correct sample counts, successful-attack counts, and epsilon ordering.
* Clean accuracy, adversarial accuracy, accuracy drop, and attack success rate.
* `epsilon=0` invariants for clean/adversarial accuracy, zero successful
  attacks, clean/adversarial images, and clean/adversarial predictions.
* Parameter preservation after single-batch evaluation, multi-batch
  aggregation, and epsilon sweep.

The tests use deterministic synthetic inputs, labels, and batch partitioning.
They do not require CIFAR-10 data, large-scale runs, checkpoint files, PGD, or
cluster-runner infrastructure. Representative-example metadata is not coupled
to the evaluated metric/sweep API and is not covered by this slice.

Count fields, predicted classes, and epsilon ordering are compared exactly.
Scalar robustness metrics use `rtol=1e-6` and `atol=1e-7`. Tensor comparisons
used for epsilon-zero invariants use `rtol=1e-5` and `atol=1e-6`.

Local non-CuPy validation command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_robustness_equivalence.py -rs
```

Expected local result on machines without CuPy/CUDA:

```text
3 skipped
```

GPU validation command:

```bash
python -m pytest -q tests/test_cupy_robustness_equivalence.py -rs
```

EWP2-D must not be marked complete until this command passes on the validated
RTX 2080 Ti cluster environment and the non-data cluster regression remains
green.

## Cluster / GPU Validation Boundary

Cluster/GPU execution is future extension work. The current repository has no
Slurm scripts and no CuPy dependency.

Before large-scale data processing, expanded evaluation subsets, repeated-seed
runs, GPU experiments, or cluster runs, ask the user whether to use the
university-provided ZITI cluster. Do not introduce GPU, CUDA, CuPy, Slurm, or
ZITI workflows without explicit approval.
