# WP_PLAN.md

## Project Title

Compact CNN on CIFAR-10 with Adversarial Robustness and Grad-CAM Analysis

## Purpose

This file records the current status of the original Work Package sequence and
the extension work planned after the verified NumPy implementation. The
original Work Packages remain historical project milestones. The Extended Work
Packages at the end of this file describe the current GPU-scaling direction and
do not rewrite the original definitions.

## Current Verified State

Latest local validation:

```text
Offline CI-compatible suite: 212 passed, 3 deselected
Data-marked suite: 3 passed, 212 deselected
Full local suite: 215 passed
```

Current implementation summary:

* The reference implementation remains NumPy-based.
* WP1-WP4 are complete.
* WP5 is functionally implemented for controlled and reproducible subset
  baselines, including a stronger 4096/1024/1024 CIFAR-10 baseline run, but no
  full 50k/10k training run has been completed.
* WP6, WP7, and WP8 are complete for their documented scopes.
* WP8 includes the validated FGSM pipeline and a 1024-sample quantitative FGSM
  evaluation. The historical WP8 smoke runner currently uses 17 epsilon values
  from `0/255` through `16/255`.
* WP9-WP12 are intentionally deferred.
* WP13 implementation exists and needs formal documentation/closeout.
* WP14 currently covers clean-vs-FGSM Grad-CAM analysis only.
* WP15 remains incomplete because final integration is not finished.

## Status Overview

| WP | Title | Status | Current Notes |
| --- | --- | --- | --- |
| WP0 | Focused Literature Review and Final Method Selection | NEEDS DOCUMENTATION | Scope exists; summary, selection, and metrics docs have now been filled from repository evidence. Remaining TODOs are limited to adding exact literature citations if required. |
| WP1 | Project setup and CIFAR-10 pipeline | COMPLETE | CIFAR-10 loading, NCHW preprocessing, deterministic batching, and reproducibility checks are implemented and tested. |
| WP2 | Compact CNN forward implementation | COMPLETE | Manual forward layers, `CompactCNN.forward`, and loss forward path are implemented and tested. |
| WP3 | Manual Backward Implementation | COMPLETE | Manual layer backward passes, loss backward, and full model backward integration are implemented and tested. |
| WP4 | Gradient Checks and Input-Gradient Support | COMPLETE | Numerical gradient checks and input-gradient pipeline sanity checks pass. |
| WP5 | Baseline training and clean evaluation | PARTIALLY COMPLETE | Training/evaluation/checkpointing/metrics/plots are implemented. Controlled subset and 4096/1024/1024 baseline runs exist. Full 50k/10k training remains unfinished. |
| WP6 | Focused Runtime Bottleneck Handling | COMPLETE | `Conv2D.backward` was profiled, selected, optimized, benchmarked, and tested. |
| WP7 | FGSM Attack and Input-Gradient Visualization | COMPLETE | Input gradients, FGSM, qualitative visualizations, and controlled example generation are implemented and tested. |
| WP8 | FGSM robustness evaluation | COMPLETE | Original FGSM robustness scope is complete, including batch evaluation, epsilon sweeps, plots, representative metadata, and 1024-sample quantitative evaluation. Larger GPU runs are an extension, not missing WP8 work. |
| WP9 | PGD Attack Implementation | DEFERRED | Intentionally not active. No PGD implementation exists. |
| WP10 | PGD Robustness Evaluation and Comparison | DEFERRED | Intentionally not active. No PGD evaluation exists. |
| WP11 | Non-Gradient Black-Box Attack Implementation | DEFERRED | Intentionally not active. No black-box attack implementation exists. |
| WP12 | Black-Box Attack Evaluation | DEFERRED | Intentionally not active. Query-count evaluation is not implemented. |
| WP13 | Grad-CAM Implementation | NEEDS DOCUMENTATION | Core Grad-CAM implementation exists and is tested; formal WP13 closeout remains to be written. |
| WP14 | Grad-CAM Analysis Before and After Attacks | PARTIALLY COMPLETE | Clean-vs-FGSM Grad-CAM analysis exists. PGD/black-box Grad-CAM analysis is absent because WP9-WP12 are deferred. |
| WP15 | Final integration, reproducibility and result organization | PARTIALLY COMPLETE | README, CI, tests, and result artifacts exist, but final integration is incomplete while extension work is pending. |

## Original Work Packages

### WP0: Focused Literature Review and Final Method Selection

Goal:

Review introductory adversarial attack and explainability material, select
methods, and define metrics.

Implemented functionality and documentation:

* Project scope draft exists in `deliverables/WP0/project_scope_draft.md`.
* Method summary, final method selection, and evaluation metrics are documented
  in `deliverables/WP0/`.
* Current selected implemented methods are NumPy CompactCNN, FGSM, and
  Grad-CAM.
* PGD and simplified square-based black-box attacks remain selected historical
  candidates but are not active in the current development cycle.

Remaining work:

* Add exact external citation details if required by a final report.

Relevant files:

```text
README.md
WP_PLAN.md
TESTING.md
WorkPackagePlan.txt
deliverables/WP0/
```

Validation:

```bash
ls AGENTS.md WP_PLAN.md TESTING.md
ls deliverables/WP0/
```

Status:

NEEDS DOCUMENTATION, because the technical implementation has moved ahead and
the WP0 deliverables now summarize repository evidence but do not yet include
full literature citation details.

---

### WP1: Project setup and CIFAR-10 pipeline

Goal:

Set up the project structure, environment, data folders, result directories,
reproducibility settings, CIFAR-10 loading, preprocessing, batching, and basic
dataset utilities.

Implemented functionality:

* CIFAR-10 archive download/extraction helper.
* CIFAR-10 batch loading via Python/NumPy.
* NCHW image tensors with `float32` values in `[0, 1]`.
* Integer labels in `[0, 9]`.
* Deterministic mini-batch iterator.
* Seed helper.
* Data-pipeline sanity-check script and sample-batch figure.

Remaining work:

* None for the original WP1 scope.

Relevant files:

```text
configs/default_config.py
src/data/cifar10_loader.py
src/data/batching.py
src/utils/seed.py
experiments/check_data_pipeline.py
tests/test_data_pipeline.py
deliverables/WP1/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_data_pipeline.py -v
```

Current validation:

```text
tests/test_data_pipeline.py: 5 tests collected
requires_data tests pass when local CIFAR-10 data is present
```

Status:

PARTIALLY COMPLETE.

---

### WP2: Compact CNN forward implementation

Goal:

Implement the from-scratch forward pass for `Conv2D`, `ReLU`, `MaxPool2D`,
`Flatten`, `Linear`, `CompactCNN`, and Softmax Cross-Entropy loss.

Implemented functionality:

* Forward layers in `src/layers/forward.py`.
* Compact CIFAR-10 architecture in `src/models/compact_cnn.py`.
* Softmax Cross-Entropy forward path in `src/losses/cross_entropy.py`.
* Shape checks, finite-output checks, and reproducible initialization.

Remaining work:

* None for the original WP2 scope.

Relevant files:

```text
src/layers/forward.py
src/models/compact_cnn.py
src/losses/cross_entropy.py
tests/test_forward.py
tests/test_losses.py
deliverables/WP2/manual_review/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_forward.py tests/test_losses.py -v
```

Status:

COMPLETE.

---

### WP3: Manual Backward Implementation

Goal:

Implement manual backward passes for all model layers and the loss.

Implemented functionality:

* `Linear.backward`
* `ReLU.backward`
* `Flatten.backward`
* `MaxPool2D.backward`
* `Conv2D.backward`
* `SoftmaxCrossEntropyLoss.backward`
* `CompactCNN.backward`
* `CompactCNN.named_parameters_and_gradients`

Remaining work:

* None for the original WP3 scope.

Relevant files:

```text
src/layers/forward.py
src/models/compact_cnn.py
src/losses/cross_entropy.py
tests/test_layers.py
tests/test_backward.py
tests/test_integration.py
deliverables/WP3/manual_review/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_layers.py tests/test_backward.py tests/test_integration.py -v
```

Status:

COMPLETE.

---

### WP4: Gradient Checks and Input-Gradient Support

Goal:

Validate selected manual gradients numerically and support gradients with
respect to input images.

Implemented functionality:

* Numerical gradient checks for `Linear`, `Conv2D`, and
  `SoftmaxCrossEntropyLoss`.
* `CompactCNN` loss-to-input gradient sanity check.
* Input-gradient helper and normalized input-gradient maps.

Remaining work:

* None for the original WP4 scope.

Relevant files:

```text
src/input_gradients.py
tests/test_gradient_check.py
tests/test_input_gradients.py
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_gradient_check.py tests/test_input_gradients.py -v
```

Status:

COMPLETE.

---

### WP5: Baseline training and clean evaluation

Goal:

Implement optimizer updates, training loops, clean evaluation, checkpointing,
metrics, and plots for the CompactCNN baseline.

Implemented functionality:

* Minimal SGD optimizer.
* Single-batch and multi-batch training/evaluation helpers.
* Checkpoint save/load.
* Deterministic JSON metrics persistence.
* Loss/accuracy and confusion-matrix plotting.
* Synthetic baseline smoke runner.
* Controlled CIFAR-10 64/32 subset baseline runner.
* Reproducible stronger CIFAR-10 baseline runner using:

```text
train_samples: 4096
validation_samples: 1024
test_samples: 1024
batch_size: 32
epochs: 15
learning_rate: 0.03
seed: 42
```

Recorded stronger baseline result:

```text
best_epoch: 15
best_validation_accuracy: 0.470703125
final_test_accuracy: 0.447265625
```

Remaining work:

* A full 50k/10k CIFAR-10 training run has not been completed.
* Large or repeated training runs should be done through the later cluster
  extension path after explicit user approval.

Relevant files:

```text
src/optimizers/sgd.py
src/training.py
src/checkpointing.py
src/metrics.py
src/plotting.py
experiments/baseline/train_baseline.py
experiments/baseline/train_portfolio_baseline.py
tests/test_training.py
tests/test_optimizer.py
tests/test_checkpointing.py
tests/test_baseline_runner.py
tests/test_cifar10_baseline_runner.py
tests/test_portfolio_baseline_runner.py
results/baseline/
deliverables/WP5/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py tests/test_training.py tests/test_checkpointing.py tests/test_baseline_runner.py tests/test_cifar10_baseline_runner.py tests/test_portfolio_baseline_runner.py -v
```

Status:

PARTIALLY COMPLETE, because all functional infrastructure exists but full
50k/10k training has not been run.

---

### WP6: Focused Runtime Bottleneck Handling

Goal:

Identify and improve one measured runtime bottleneck without broad backend
comparison.

Implemented functionality:

* Inspection-only runtime profiling.
* `Conv2D.backward` selected as the focused bottleneck.
* `Conv2D.backward` optimized with `np.einsum`-based gradient accumulation.
* Before/after benchmark and runtime figure documented.

Recorded benchmark:

```text
Conv2D.forward: 0.000068569 -> 0.000066375 seconds
Conv2D.backward: 0.043458736 -> 0.000209222 seconds
train_step: 0.070350708 -> 0.001886028 seconds
```

Current local profiler output from the latest audit:

```text
conv2d_forward_seconds=0.000115542
conv2d_backward_seconds=0.000345306
train_step_seconds=0.003370375
```

Remaining work:

* None for the original WP6 scope.
* GPU acceleration is handled as an Extended Work Package, not as missing WP6
  work.

Relevant files:

```text
src/layers/forward.py
experiments/runtime/profile_wp6.py
deliverables/WP6/
results/WP6/conv2d_backward_runtime_comparison.png
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_layers.py tests/test_gradient_check.py tests/test_backward.py tests/test_integration.py -v
```

Status:

COMPLETE.

---

### WP7: FGSM Attack and Input-Gradient Visualization

Goal:

Implement FGSM using input gradients and save qualitative clean/adversarial
visualizations.

Implemented functionality:

* Deterministic input-gradient computation.
* Normalized input-gradient maps.
* Untargeted FGSM:

```python
np.clip(images + epsilon * np.sign(grad_input), 0.0, 1.0)
```

* `L_inf` perturbation behavior and clipping validation.
* Clean/adversarial/gradient/perturbation PNG saving.
* Controlled one-example CIFAR-10 runner.

Remaining work:

* None for the original WP7 scope.

Relevant files:

```text
src/input_gradients.py
src/attacks/fgsm.py
src/visualization.py
experiments/fgsm/generate_examples.py
tests/test_input_gradients.py
tests/test_fgsm.py
tests/test_visualization.py
tests/test_fgsm_examples.py
results/WP7/qualitative/
deliverables/WP7/wp7_summary.md
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_input_gradients.py tests/test_fgsm.py tests/test_visualization.py tests/test_fgsm_examples.py -v
```

Status:

COMPLETE.

---

### WP8: FGSM robustness evaluation

Goal:

Evaluate clean and FGSM predictions across epsilon values, aggregate metrics,
plot robustness curves, and select representative examples.

Implemented functionality:

* `evaluate_fgsm_batch`
* `evaluate_fgsm_batches`
* `evaluate_fgsm_epsilon_sweep`
* `select_fgsm_representative_examples`
* Controlled WP8 smoke runner with persisted metrics and plot.
* 1024-sample quantitative FGSM evaluation with a stronger baseline checkpoint.
* FGSM plots for adversarial accuracy, attack success rate, and accuracy drop.

Current FGSM quantitative configuration:

```text
checkpoint: results/baseline/portfolio_baseline_best.npz
eval_samples: 1024
batch_size: 32
seed: 42
epsilons: [0, 2/255, 4/255, 8/255, 16/255]
```

Current historical WP8 smoke configuration:

```text
eval_samples: 32
batch_size: 8
seed: 42
epsilon_values: [0/255, 1/255, ..., 16/255]
```

Remaining work:

* None for the original WP8 scope.
* Larger GPU/cluster FGSM evaluation is an extension and should not be treated
  as incomplete WP8 work.

Relevant files:

```text
src/robustness.py
src/plotting.py
experiments/fgsm/evaluate_robustness.py
experiments/fgsm/evaluate_quantitative.py
tests/test_fgsm_evaluation.py
tests/test_fgsm_robustness_runner.py
tests/test_fgsm_quantitative_runner.py
results/WP8/
results/fgsm/
deliverables/WP8/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_fgsm_evaluation.py tests/test_fgsm_robustness_runner.py tests/test_fgsm_quantitative_runner.py tests/test_plotting.py -v
```

Status:

COMPLETE.

---

### WP9: PGD Attack Implementation

Goal:

Implement a small-scale PGD white-box attack.

Implemented functionality:

* None.

Remaining work:

* PGD implementation, projection, clipping, parameter handling, examples, and
  documentation are not part of the current active development cycle.

Status:

DEFERRED.

---

### WP10: PGD Robustness Evaluation and Comparison

Goal:

Evaluate PGD and compare it with FGSM.

Implemented functionality:

* None.

Remaining work:

* PGD evaluation, comparison plots, and PGD representative examples are not
  part of the current active development cycle.

Status:

DEFERRED.

---

### WP11: Non-Gradient Black-Box Attack Implementation

Goal:

Implement one non-gradient black-box attack, historically scoped as a
simplified square-based random-search attack.

Implemented functionality:

* None.

Remaining work:

* Black-box attack implementation is not part of the current active
  development cycle.

Status:

DEFERRED.

---

### WP12: Black-Box Attack Evaluation

Goal:

Evaluate the deferred black-box attack using attack success rate and query
count.

Implemented functionality:

* None.

Remaining work:

* Query-count evaluation and black-box comparisons are not part of the current
  active development cycle.

Status:

DEFERRED.

---

### WP13: Grad-CAM Implementation

Goal:

Implement Grad-CAM heatmap computation for the CompactCNN.

Implemented functionality:

* `CompactCNN` stores the `relu2` activation before `pool2`.
* `compute_gradcam` computes target-class gradients with respect to that
  activation path.
* Channel weights are produced by global average pooling of activation
  gradients.
* Heatmaps are ReLU-filtered and normalized per sample.
* Helper functions resize heatmaps and create overlays.

Remaining work:

* Write formal WP13 closeout documentation under `deliverables/WP13/`.
* Record the chosen target activation and limitations explicitly.

Relevant files:

```text
src/models/compact_cnn.py
src/gradcam.py
src/gradcam_visualization.py
tests/test_gradcam.py
tests/test_gradcam_visualization.py
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_gradcam.py tests/test_gradcam_visualization.py -v
```

Status:

NEEDS DOCUMENTATION.

---

### WP14: Grad-CAM Analysis Before and After Attacks

Goal:

Use Grad-CAM qualitatively to inspect focus changes before and after attacks.

Implemented functionality:

* Clean-vs-FGSM adversarial Grad-CAM comparison runner.
* Deterministic scan of CIFAR-10 test samples.
* Selection of clean-correct FGSM successes and attack-resisted controls.
* Prediction-aligned Grad-CAM figures.
* Fixed-original-target comparison figures.
* Metadata persistence.

Current boundary:

* WP14 currently covers FGSM only.
* PGD and black-box Grad-CAM analysis are absent because WP9-WP12 are deferred.

Remaining work:

* Document WP14 as FGSM-only analysis.
* If PGD/black-box work is ever resumed, extend WP14 only after those attack
  work packages are implemented and validated.

Relevant files:

```text
experiments/gradcam/generate_adversarial_comparisons.py
src/gradcam.py
src/gradcam_visualization.py
results/gradcam/
```

Tests:

```bash
.venv/bin/python -m pytest tests/test_gradcam.py tests/test_gradcam_visualization.py -v
```

Status:

PARTIALLY COMPLETE.

---

### WP15: Final integration, reproducibility and result organization

Goal:

Organize final code, documentation, tests, results, and reproducibility
instructions.

Implemented functionality:

* README summarizes current implementation, tests, results, and run commands.
* GitHub Actions runs the offline test suite.
* Result figures and metrics are organized under `results/`.
* Deliverables exist for completed historical WPs.

Remaining work:

* Finish documentation for WP13 and WP14.
* Decide the checkpoint artifact policy for ignored `.npz` files.
* Add extension documentation after CuPy/cluster work is completed.
* Final report/poster tables remain outside the current repository state.

Status:

COMPLETE.

## Current Active Development Direction

The active development cycle is:

```text
NumPy reference
-> CuPy acceleration
-> CPU/GPU numerical equivalence
-> cluster execution
-> large-scale FGSM evaluation
-> runtime/robustness analysis
```

PGD and black-box attacks are intentionally outside this active cycle. They
remain in the original plan as deferred historical Work Packages.

Before any large-scale data processing, expanded evaluation subset,
repeated-seed run, or cluster experiment, ask the user whether to use the
university-provided ZITI cluster. Do not introduce GPU, CUDA, CuPy, Slurm, or
ZITI workflows without explicit user approval.

## Extended Work Packages

### EWP1: CuPy Backend

Goal:

Add optional GPU acceleration while preserving NumPy as the reference backend.

Scope:

* Keep NumPy behavior as the default and reference implementation.
* Introduce a minimal backend boundary only where it improves correctness,
  performance, or maintainability.
* Do not replace every `numpy` import mechanically.
* Keep plotting, JSON metrics, and public artifact generation CPU-side.

Status:

COMPLETE.

EWP1-A: Backend abstraction

Status: COMPLETE.

* Minimal backend module for optional NumPy/CuPy dispatch.
* NumPy remains the default and reference backend.
* Tensor-path backend plumbing is implemented for layers, model
  forward/backward, loss, SGD, training metrics, input gradients, FGSM,
  robustness evaluation, and checkpoint CPU/device boundaries.

EWP1-B: CuPy runtime compatibility validation

Status: COMPLETE.

* Optional CuPy test infrastructure is available and skips cleanly when CuPy,
  CUDA runtime access, or a visible CUDA GPU is unavailable.
* Primitive compatibility tests cover array creation/conversion, zeros,
  zeros_like, maximum, max, argmax, sum, mean, abs, exp, log, divide, clip,
  sign, matmul, `einsum(..., optimize=True)`, `add.at`, finite checks, scalar
  conversion helpers, and `sliding_window_view`.
* First equivalence slice covers `Conv2D.forward` and `Conv2D.backward`
  outputs/gradients with explicit float32 tolerances.
* Real GPU validation passed on the tested environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

* `sliding_window_view`, `einsum(..., optimize=True)`, `cupy.add.at`,
  `divide_where(...)`, `Conv2D.forward`, and `Conv2D.backward` `dx/dw/db`
  equivalence are validated on that environment.
* No CPU fallback was introduced in the tested tensor path.
* Compatibility is claimed only for the tested environment above, not for
  untested GPU/CUDA/CuPy configurations.
* Broader EWP2 numerical-equivalence validation has not started yet.

Relevant folders/files:

```text
src/backend.py
src/layers/forward.py
src/models/compact_cnn.py
src/losses/cross_entropy.py
src/optimizers/sgd.py
src/training.py
src/input_gradients.py
src/attacks/fgsm.py
src/robustness.py
src/checkpointing.py
tests/test_backend.py
tests/cupy_test_utils.py
tests/conftest.py
tests/test_cupy_backend_runtime.py
deliverables/EWP1/backend_migration_report.md
```

Validation:

* Preserve all NumPy reference behavior.
* Run the full local NumPy test suite after each slice.
* CuPy-specific tests must skip cleanly when the GPU backend is unavailable.
* Do not require CuPy in the local development environment.
* GPU cluster validation:

```text
python -m pytest -q tests/test_cupy_backend_runtime.py -rs
6 passed

python -m pytest -q -m "not requires_data"
223 passed, 3 deselected in 8.20s
```

Dependencies:

* NumPy remains required.
* CuPy is optional and must not be added as a hard dependency in EWP1.
* CIFAR-10 loading, plotting, JSON artifacts, and image output remain CPU-side.

Known cluster issue:

Cluster CIFAR-10 dataset staging remains unresolved. The current cluster
archive `data/raw/cifar-10-python.tar.gz` has size `37M` and MD5
`352dcf059b8b606c932d1db9b8c351a9`, but the project expects
`c58f30108f718f92721af3b95e74349a`. Replace or restage the archive before
using data-dependent cluster tests or experiments. This is not an EWP1 backend
failure.

---

### EWP2: NumPy/CuPy Numerical Equivalence

Goal:

Verify that the CuPy path matches the NumPy reference for the computation paths
used by training, FGSM, and robustness evaluation.

Candidate reference tests:

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
tests/test_cupy_layer_loss_equivalence.py
tests/test_cupy_model_training_equivalence.py
tests/test_cupy_fgsm_equivalence.py
tests/test_cupy_robustness_equivalence.py
```

Scope:

* Forward outputs.
* Backward gradients.
* Loss values and loss gradients.
* Input gradients.
* FGSM adversarial examples.
* Robustness metrics.

Status:

COMPLETE.

Validated coverage summary:

* EWP2-A: `ReLU`, `MaxPool2D`, `Flatten`, `Linear`, and
  `SoftmaxCrossEntropyLoss` forward/backward numerical equivalence.
* EWP2-B: `CompactCNN` forward logits, scalar loss, loss gradient, full model
  backward, all trainable parameter gradients, one real `SGD.step()`, and
  `train_step(...)` numerical equivalence.
* EWP2-C: input gradients, `epsilon=0` FGSM, nonzero-epsilon FGSM, clipping,
  adversarial images, adversarial logits, predictions, and the complete
  clean-input-to-adversarial-forward path.
* EWP2-D: single-batch robustness metrics, raw counts, multi-batch
  aggregation, epsilon-zero invariants, epsilon ordering, epsilon-sweep
  metrics, and parameter preservation.

Validated environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
```

NumPy remains the authoritative correctness reference. Compatibility is
claimed only for the tested environment above, not for untested
GPU/CUDA/CuPy/Python configurations.

EWP2-A: Remaining layer and loss numerical equivalence

Status: COMPLETE.

Implemented coverage:

* `ReLU.forward` and `ReLU.backward`.
* `MaxPool2D.forward` and `MaxPool2D.backward`, including `add.at` backward
  execution and first-maximum tie semantics.
* `Flatten.forward` and `Flatten.backward`.
* `Linear.forward` and `Linear.backward`, including `dx`, `dw`, and `db`.
* `SoftmaxCrossEntropyLoss.forward` and `SoftmaxCrossEntropyLoss.backward`.

Validation boundary:

* The tests are implemented in `tests/test_cupy_layer_loss_equivalence.py`.
* They skip cleanly on systems without CuPy/CUDA.
* Real GPU validation passed on the tested environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

* GPU validation result:

```text
python -m pytest -q tests/test_cupy_layer_loss_equivalence.py -rs
6 passed in 0.67s

python -m pytest -q -m "not requires_data"
229 passed, 3 deselected in 7.86s
```

* NumPy remains the authoritative correctness reference.
* Compatibility is claimed only for the tested environment above, not for
  untested GPU/CUDA/CuPy configurations.
* EWP2 is complete for the planned NumPy/CuPy numerical-equivalence scope.

EWP2-B: CompactCNN end-to-end training-path numerical equivalence

Status: COMPLETE.

Validated coverage:

* Deterministic in-memory synchronization of NumPy and CuPy `CompactCNN`
  parameters without relying on independent random initialization.
* Identical deterministic input tensors, labels, and optimizer settings.
* Full `CompactCNN.forward` logits comparison on a deterministic synthetic
  batch.
* `SoftmaxCrossEntropyLoss.forward` scalar loss comparison and loss
  `backward` logits-gradient comparison.
* Full `CompactCNN.backward` gradient comparison for every trainable
  parameter exposed by `named_parameters_and_gradients()`:
  `conv1.weights`, `conv1.bias`, `conv2.weights`, `conv2.bias`,
  `classifier.weights`, and `classifier.bias`.
* One real `SGD.step()` update comparison for every trainable parameter.
* Public `train_step(...)` helper coverage for the same
  `forward -> loss -> backward -> SGD` path.

Validation boundary:

* The tests are implemented in
  `tests/test_cupy_model_training_equivalence.py`.
* They use deterministic synthetic inputs and do not require CIFAR-10 data.
* They skip cleanly on systems without CuPy/CUDA.
* Tensor comparisons use `rtol=1e-5` and `atol=1e-6`; scalar loss comparisons
  use `rtol=1e-6` and `atol=1e-7`, matching the EWP2-A tolerance policy.
* Local non-CuPy validation passed with `220 passed, 14 deselected` for
  `-m "not requires_cupy"` and `220 passed, 14 skipped` for the full suite;
  the EWP2-B module skipped cleanly with `2 skipped` because CuPy is not
  installed locally.
* Real GPU validation passed on the tested environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

* GPU validation result:

```text
python -m pytest -q tests/test_cupy_model_training_equivalence.py -rs
2 passed

python -m pytest -q -m "not requires_data"
231 passed, 3 deselected in 9.23s
```

* NumPy remains the authoritative correctness reference.
* Compatibility is claimed only for the tested environment above, not for
  untested GPU/CUDA/CuPy configurations.
* This slice does not cover model input-gradient, FGSM, robustness,
  checkpoint, or cluster-runner equivalence.

EWP2-C: Input-gradient and FGSM numerical equivalence

Status: COMPLETE.

Validated coverage:

* Deterministic in-memory synchronization of NumPy and CuPy `CompactCNN`
  parameters using the same strategy as EWP2-B.
* Identical deterministic clean NCHW inputs, labels, epsilon, and clipping
  bounds without CIFAR-10 data.
* Production `compute_input_gradient(...)` equivalence for loss-to-input
  gradients.
* Production `fgsm_attack(...)` equivalence for `epsilon=0` and a nonzero
  epsilon.
* FGSM shape, `L_inf` perturbation bound, and `[0, 1]` clipping semantics.
* Adversarial images.
* Adversarial `CompactCNN.forward` logits and predicted classes.
* End-to-end attack path:
  `clean input -> input gradient -> FGSM image -> adversarial forward`.
* Parameter preservation after input-gradient computation, FGSM generation,
  and adversarial forward evaluation.

Validation boundary:

* The tests are implemented in `tests/test_cupy_fgsm_equivalence.py`.
* They use deterministic synthetic inputs and do not require CIFAR-10 data.
* They skip cleanly on systems without CuPy/CUDA.
* Input-gradient, adversarial-image, and adversarial-logit comparisons use
  `rtol=1e-5` and `atol=1e-6`.
* Local non-CuPy validation passed with `220 passed, 16 deselected` for
  `-m "not requires_cupy"` and `220 passed, 16 skipped` for the full suite;
  the EWP2-C module skipped cleanly with `2 skipped` because CuPy is not
  installed locally.
* Real GPU validation passed on the tested environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

* GPU validation result:

```text
python -m pytest -q tests/test_cupy_fgsm_equivalence.py -rs
2 passed in 1.14s

python -m pytest -q -m "not requires_data"
233 passed, 3 deselected in 8.29s
```

* NumPy remains the authoritative correctness reference.
* Compatibility is claimed only for the tested environment above, not for
  untested GPU/CUDA/CuPy configurations.
* This slice does not cover robustness metric or epsilon-sweep equivalence,
  checkpoint equivalence, PGD, or cluster-runner infrastructure.

EWP2-D: Robustness evaluation and epsilon-sweep numerical equivalence

Status: COMPLETE.

Validated coverage:

* Deterministic in-memory synchronization of NumPy and CuPy `CompactCNN`
  parameters using the same strategy as earlier EWP2 slices.
* Deterministic synthetic NCHW batches, labels, batch partitioning, epsilon
  values, and clipping behavior without CIFAR-10 data.
* Production `evaluate_fgsm_batch(...)` single-batch metric equivalence.
* Production `evaluate_fgsm_batches(...)` multi-batch raw-count aggregation
  equivalence across different batch sizes.
* Production `evaluate_fgsm_epsilon_sweep(...)` epsilon-order and metric
  equivalence for `0/255`, `4/255`, and `8/255`.
* Exact comparison for raw counts and epsilon ordering.
* Clean-correct, adversarial-correct, and successful-attack counts.
* Scalar robustness-metric comparison for clean accuracy, adversarial
  accuracy, accuracy drop, and attack success rate.
* `epsilon=0` invariants for clean/adversarial accuracy, zero successful
  attacks, clean/adversarial images, and clean/adversarial predictions.
* Parameter preservation after single-batch evaluation, multi-batch
  aggregation, and epsilon sweep.

Validation boundary:

* The tests are implemented in `tests/test_cupy_robustness_equivalence.py`.
* They use deterministic synthetic inputs and do not require CIFAR-10 data.
* They skip cleanly on systems without CuPy/CUDA.
* Count fields, predicted classes, and epsilon ordering are exact comparisons.
* Scalar robustness metrics use `rtol=1e-6` and `atol=1e-7`.
* Any tensor comparisons used for epsilon-zero invariants use `rtol=1e-5` and
  `atol=1e-6`.
* Local non-CuPy validation passed with `220 passed, 19 deselected` for
  `-m "not requires_cupy"` and `220 passed, 19 skipped` for the full suite;
  the EWP2-D module skipped cleanly with `3 skipped` because CuPy is not
  installed locally.
* Real GPU validation passed on the tested environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

* GPU validation result:

```text
python -m pytest -q tests/test_cupy_robustness_equivalence.py -rs
3 passed in 4.25s

python -m pytest -q -m "not requires_data"
236 passed, 3 deselected in 8.96s
```

* NumPy remains the authoritative correctness reference.
* Compatibility is claimed only for the tested environment above, not for
  untested GPU/CUDA/CuPy configurations.
* This slice does not cover large-scale CIFAR-10 runs, checkpoint equivalence,
  representative-example metadata equivalence, PGD, or cluster-runner
  infrastructure.

---

### EWP3: Cluster Experiment Infrastructure

Goal:

Prepare configurable, reproducible cluster execution for larger experiments.

Scope:

* Add scheduler-neutral Python runners/configuration first.
* Avoid hard-coded local paths.
* Require explicit data and checkpoint paths.
* Save outputs under structured result/log directories.
* Record seeds, backend, device, batch size, sample count, checkpoint, runtime,
  and memory-related metadata.
* Preserve machine-readable experiment outputs so plots and tables can be
  regenerated.
* Add Slurm or scheduler-specific scripts only after the scheduler is
  confirmed.

Status:

PARTIALLY COMPLETE. EWP3-A environment and dataset-staging validation has
local infrastructure in place, but real cluster validation is pending until a
checksum-valid CIFAR-10 archive is staged.

#### EWP3-A: Cluster Environment and CIFAR-10 Dataset Staging

Status:

IMPLEMENTED / REAL CLUSTER DATA VALIDATION PENDING.

Implemented infrastructure:

* Scheduler-neutral validation utility:

```text
scripts/validate_cluster_environment.py
```

* Human-readable environment and dataset report.
* Optional machine-readable JSON validation report.
* Python, NumPy, optional CuPy, CUDA runtime, visible device count, and GPU
  device-name reporting.
* Explicit CIFAR-10 staging validation without starting training, robustness
  evaluation, or benchmark workloads.
* Optional extraction only when the expected archive exists and its checksum
  matches the project checksum.

Dataset architecture:

* Default data directory: `data/raw`.
* Expected archive: `data/raw/cifar-10-python.tar.gz`.
* Expected archive MD5: `c58f30108f718f92721af3b95e74349a`.
* Expected extracted directory: `data/raw/cifar-10-batches-py`.
* Expected extracted files include `data_batch_1` through `data_batch_5`,
  `test_batch`, and `batches.meta`.
* Expected shapes are `(50000, 3, 32, 32)` for training images and
  `(10000, 3, 32, 32)` for test images, with 10 class names.
* `load_cifar10(data_dir=...)` already supports explicit custom data
  directories. The cluster utility reuses the loader's batch parsing but does
  not auto-download data.

Cluster staging policy:

```text
dataset acquisition/staging
-> checksum verification
-> extraction
-> persistent cluster storage
-> read-only experiment consumption
```

GPU jobs should not repeatedly download CIFAR-10. Stage the archive once from
a trusted source or trusted local copy, verify the checksum, extract it in the
persistent project data directory, and have later jobs consume the staged
files. Do not duplicate the dataset unless a cluster storage policy requires a
separate persistent data location.

Known cluster data issue:

```text
path: data/raw/cifar-10-python.tar.gz
observed size: approximately 37 MB
observed MD5: 352dcf059b8b606c932d1db9b8c351a9
expected MD5: c58f30108f718f92721af3b95e74349a
```

Checksum validation is authoritative. The invalid archive should be quarantined
or replaced by a checksum-valid official archive; do not change the expected
checksum, disable validation, or modify the loader to accept the bad file.

Validation before EWP3-A closeout:

```bash
python scripts/validate_cluster_environment.py --backend cupy --data-dir data/raw --extract-if-needed --json-output results/cluster_validation/cifar10_environment.json
python -m pytest -q -m requires_data
```

EWP3-A does not implement the cluster experiment runner, large-scale FGSM
evaluation, benchmarking plots, or PGD.

---

### EWP4: Large-scale FGSM Evaluation

Goal:

Scale the validated WP8 FGSM sweep to larger CIFAR-10 evaluation sets.

Scope:

* Reuse the existing FGSM implementation and robustness metrics.
* Increase evaluation sample count only through explicit configuration.
* Preserve batched evaluation.
* Save raw per-epsilon metrics and run metadata before creating summary plots.
* Treat successful execution alone as insufficient; large-scale runs should
  produce quantitative artifacts that support robustness interpretation.
* Keep PGD and black-box attacks out of scope.

Status:

PLANNED.

---

### EWP5: Runtime and Robustness Analysis

Goal:

Analyze CPU/GPU runtime, scalability, adversarial accuracy, attack success
rate, and memory use.

Scope:

* Compare NumPy and CuPy runtimes for the same architecture and evaluation
  settings.
* Report clean accuracy, adversarial accuracy, accuracy drop, and attack
  success rate.
* Track runtime by sample count, batch size, epsilon count, and backend.
* Document memory limitations and CPU/GPU transfer points.
* Produce human-readable plots or tables from saved quantitative results when
  they materially improve analysis or communication.

Status:

PLANNED.

## Cluster and Large-Scale Experiment Evidence Requirements

Successful execution is not enough for the upcoming cluster and large-scale
experimental phases. Future quantitative work should produce both:

1. Machine-readable quantitative results.
2. Human-readable plots or tables derived from those results when they improve
   interpretation.

Do not generate decorative or redundant plots merely to increase artifact
count. The goal is to make performance, scaling, robustness, resource use, and
model-behavior claims quantitatively verifiable.

Current roadmap:

```text
EWP1 -> COMPLETE
EWP2 -> COMPLETE
Next -> cluster preparation
     -> correct CIFAR-10 staging
     -> cluster experiment runner
     -> small GPU FGSM smoke experiment
     -> medium-scale evaluation
     -> full CIFAR-10 test-set FGSM evaluation
     -> runtime / robustness analysis
     -> visualization / final experiment artifacts
```

PGD remains deferred.

### Planned Visualization and Artifact Matrix

The following artifacts are planned or candidate deliverables for future
cluster and large-scale evaluation work. They should be generated only when
supported by saved numerical results.

| Area | Planned artifacts | Source data |
| --- | --- | --- |
| Performance / systems | CPU vs GPU runtime, CPU vs GPU speedup, throughput in samples/sec, runtime vs sample count, runtime vs batch size, throughput vs batch size, optional GPU memory/resource utilization if reliable measurement is available | `timing.csv`, `summary.json`, `environment.json` |
| Robustness | FGSM epsilon vs adversarial accuracy, FGSM epsilon vs attack success rate, clean vs adversarial accuracy, robustness degradation relative to epsilon `0`, per-epsilon quantitative summary table | `metrics.csv`, `summary.json`, epsilon sweep outputs |
| Model behavior | Clean confusion matrix, adversarial confusion matrix, representative clean/adversarial image pairs, perturbation visualization, Grad-CAM clean vs adversarial comparisons where existing project functionality supports them | saved predictions, selected example metadata, existing visualization outputs |

Do not add new model functionality solely to produce a figure.

### Preferred Experiment Artifact Organization

Future cluster runs should follow a structure close to the existing
`results/` convention without creating unnecessary directories in advance:

```text
results/
  <experiment-id>/
    config.json
    environment.json
    metrics.csv
    summary.json
    timing.csv
    figures/
      accuracy_vs_epsilon.png
      attack_success_rate_vs_epsilon.png
      runtime_comparison.png
      throughput_scaling.png
    tables/
      epsilon_summary.csv
      runtime_summary.csv
```

Use `deliverables/` for concise human-facing summaries when a work package or
extension phase is closed. Large checkpoints and binary arrays should remain
external or ignored unless an explicit artifact policy says otherwise.

### Benchmarking Quality Requirements

Future CPU/GPU performance claims must use defensible measurement methodology:

* Synchronize GPU work around timed regions when required.
* Run warm-up iterations before benchmark measurement.
* Repeat measurements where practical.
* Report median/mean and variability when useful.
* Use identical workloads for CPU/GPU comparisons.
* Record backend, device, CUDA version, CuPy version, Python version, batch
  size, sample count, epsilon values, seed, checkpoint, and timing method.
* Distinguish end-to-end runtime from model-only or kernel-only runtime.
* Avoid timing asynchronous GPU execution incorrectly.
* Derive speedup from recorded raw timings rather than manually entered
  numbers.

Every important plot or table should be reproducible from saved experiment
outputs where practical:

```text
experiment -> structured raw results -> analysis/plotting script -> figure/table
```

## Artifact Policy Note

The repository ignores large binary arrays and checkpoints through `.gitignore`
patterns such as `*.npz`, `*.npy`, and `results/checkpoints/*`. The README
references `results/baseline/portfolio_baseline_best.npz`, but this checkpoint
is a local/regenerable artifact and is not tracked by Git.

Current policy recommendation:

* Do not force-add large checkpoints during documentation cleanup.
* Keep JSON metrics and PNG figures tracked when they are lightweight and
  useful for review.
* For future cluster experiments, store large checkpoints externally or publish
  them through an explicit release/artifact mechanism.
* Document the exact command and configuration needed to regenerate each
  ignored checkpoint.

## Rule for Moving Between Work Packages

Before starting implementation for a new original or extended Work Package:

1. Review `WP_PLAN.md`, `TESTING.md`, relevant source files, tests, and
   existing results.
2. Define relevant files, validation commands, and dependencies.
3. Run focused tests before and after the change.
4. Preserve the NumPy reference behavior unless the task explicitly changes it.
5. Do not start PGD, black-box attacks, or large-scale cluster experiments
   without explicit user approval.
