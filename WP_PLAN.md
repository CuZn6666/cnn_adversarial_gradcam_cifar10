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
Non-CuPy local regression: 293 passed, 23 deselected
Full local suite on this machine: 293 passed, 23 skipped
Data-marked cluster suite after CIFAR-10 staging: 3 passed, 240 deselected
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
* WP9 now has a local EWP4-A L-infinity PGD core implementation, EWP4-B
  NumPy/CuPy PGD equivalence validation on the tested RTX 2080 Ti
  environment, and EWP4-C local PGD runner/curation infrastructure. Real
  cluster PGD smoke validation is pending.
* WP10-WP12 remain intentionally deferred.
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
| WP9 | PGD Attack Implementation | PARTIALLY COMPLETE | EWP4-A implements the local L-infinity PGD core and focused NumPy tests. EWP4-B PGD equivalence is validated on the tested RTX 2080 Ti environment. EWP4-C local PGD runner/curation infrastructure is implemented and awaits real cluster smoke validation. |
| WP10 | PGD Robustness Evaluation and Comparison | DEFERRED | Intentionally not active beyond EWP4-C smoke infrastructure. No full PGD robustness evaluation exists. |
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

* EWP4-A adds `src/attacks/pgd.py` with `pgd_linf_attack(...)`.
* EWP4-B adds a private `_pgd_linf_attack_from_initial(...)` helper so tests
  can validate deterministic shared-initial-state PGD without comparing
  NumPy and CuPy RNG streams or duplicating the PGD algorithm in tests.
* The implementation reuses `compute_input_gradient(...)` for each PGD step
  instead of duplicating model/loss backward logic.
* The attack enforces the valid image range, L-infinity projection around the
  clean input, deterministic NumPy random start when a seed is provided, and
  non-mutating clean-input behavior.
* `steps=0` is explicitly allowed and returns a clean input copy without
  random initialization or gradient computation.
* `tests/test_pgd.py` covers the local NumPy PGD core.

Remaining work:

* PGD experiment-runner support, CIFAR-10 PGD evaluation, representative PGD
  examples, and curated PGD evidence have not started.

Status:

PARTIALLY COMPLETE. EWP4-A and EWP4-B are complete; PGD evaluation,
experiment-runner support, and curated PGD evidence have not started.

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

PARTIALLY COMPLETE. EWP3-A environment and dataset-staging validation is
complete. EWP3-B scheduler-neutral experiment-runner infrastructure is
complete. EWP3-C medium-scale GPU FGSM sanity validation is complete. EWP3-D
CPU/GPU scaling and performance benchmarking is complete. EWP3-E full-test-set
FGSM robustness evaluation is complete. EWP3-F final portfolio evidence and
project presentation is implemented locally and ready for review.

#### EWP3-A: Cluster Environment and CIFAR-10 Dataset Staging

Status:

COMPLETE.

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

Validated Hawaii cluster result:

```text
path: data/raw/cifar-10-python.tar.gz
observed size: 170498071 bytes
expected MD5: c58f30108f718f92721af3b95e74349a
observed MD5: c58f30108f718f92721af3b95e74349a
checksum: PASS
extracted directory: data/raw/cifar-10-batches-py
train shape: (50000, 3, 32, 32)
train labels: (50000,)
test shape: (10000, 3, 32, 32)
test labels: (10000,)
class count: 10
```

The previous cluster archive with MD5
`352dcf059b8b606c932d1db9b8c351a9` has been replaced by the checksum-valid
official CIFAR-10 archive. Checksum validation remains authoritative; do not
change the expected checksum, disable validation, or modify the loader to
accept invalid files.

Validated environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12.13
NumPy: 2.4.6
CUDA-capable device count: 1
Slurm allocation: 1 GPU
```

Validation artifacts generated on the cluster:

```text
results/cluster_validation/cifar10_environment_numpy.json
results/cluster_validation/cifar10_environment_cupy.json
```

Both reports had `status = passed`. The CuPy report recorded environment
status `passed`, dataset status `passed`, CuPy `14.1.1`, device count `1`, and
GPU name `NVIDIA GeForce RTX 2080 Ti`. These JSON files are run-specific
cluster/environment outputs and are intentionally ignored by Git; this plan
records the verified summary.

Validated command results:

```text
python scripts/validate_cluster_environment.py --backend numpy --data-dir data/raw --json-output results/cluster_validation/cifar10_environment_numpy.json
PASS

python scripts/validate_cluster_environment.py --backend cupy --data-dir data/raw --extract-if-needed --json-output results/cluster_validation/cifar10_environment_cupy.json
PASS

python -m pytest -q -m requires_data
3 passed, 240 deselected in 3.34s
```

Compatibility and data-staging validation are claimed only for the tested
cluster environment above, not for untested hardware/software combinations.

EWP3-A does not implement the cluster experiment runner, large-scale FGSM
evaluation, benchmarking plots, or PGD.

#### EWP3-B: Reproducible FGSM Experiment Runner

Status:

COMPLETE.

Implemented infrastructure:

* Scheduler-neutral FGSM experiment CLI:

```text
python -m experiments.fgsm.run_fgsm_experiment
```

* Explicit configuration for backend, data directory, checkpoint, CIFAR-10
  split, maximum sample count, batch size, epsilon list, seed, output root,
  and run identifier.
* Isolated run directory per run:

```text
results/runs/<run_id>/
```

* Stable machine-readable artifacts:

```text
config.json
environment.json
metrics.csv
metrics.json
timing.json
summary.json
status.json
```

* `status.json` records `RUNNING`, `COMPLETED`, or `FAILED` so partial runs
  are identifiable.
* Existing project numerical APIs are reused: `load_cifar10(...)`,
  `load_checkpoint(...)`, `CompactCNN(..., backend=...)`,
  `SoftmaxCrossEntropyLoss(..., backend=...)`, and
  `evaluate_fgsm_epsilon_sweep(...)`.
* No plotting is performed inside the runner.
* No PGD, scheduler-specific Slurm wrapper, full CIFAR-10 sweep, custom CUDA
  kernel, or Conv2D optimization is included in this slice.

Artifact policy:

* `config.json` stores the exact effective workload.
* `environment.json` stores Python, NumPy, optional CuPy/CUDA/GPU, hostname,
  and Git commit/dirty-state metadata.
* `metrics.json` and `metrics.csv` store one row per epsilon with raw counts,
  clean accuracy, adversarial accuracy, accuracy drop, and attack success
  rate using the existing `src.robustness` semantics.
* `timing.json` stores total wall time, evaluation wall time, sample count,
  epsilon count, sample-epsilon pairs, and throughput-like evaluation rate.
* `summary.json` stores a compact run summary and dataset validation metadata.

Timing policy:

* CPU timing uses `time.perf_counter`.
* CuPy timing synchronizes `cupy.cuda.Stream.null` before and after the
  evaluation timed region to avoid measuring asynchronous launch time only.
* The runner records end-to-end wall time and evaluation wall time. It does
  not claim kernel-level timing precision.

Failure and overwrite policy:

* Invalid config values fail before execution.
* Missing checkpoints, invalid CIFAR-10 archive checksum, missing extracted
  data, invalid sample counts, and unavailable CuPy/GPU fail clearly.
* CuPy requests never silently fall back to NumPy.
* Existing run directories are not overwritten.

Local validation:

```text
tests/test_fgsm_experiment_runner.py
```

The local tests cover CLI/config parsing, run-directory collision behavior,
environment metadata, metric serialization, tiny NumPy execution on synthetic
data via monkeypatched CIFAR-10 loading, and failed-run status artifacts.

Required cluster validation before closeout:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 8 --batch-size 2 --epsilons 0,1/255 --output-root results/runs
```

Validated Hawaii cluster smoke:

```text
run_id: 20260811T171313482022Z_fgsm_cupy
hostname: csg-brook01
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
device_count: 1
Slurm allocation: 1 GPU
backend: cupy
split: test
max_samples: 8
batch_size: 2
seed: 42
epsilons: [0.0, 0.00392156862745098]
checkpoint: results/checkpoints/portfolio_baseline_best.npz
data_dir: data/raw
status: COMPLETED
```

Generated artifact schema:

```text
config.json
environment.json
metrics.csv
metrics.json
timing.json
summary.json
status.json
```

Smoke metrics:

```text
epsilon 0.0:
  total_samples: 8
  clean_correct: 3
  adversarial_correct: 3
  clean_accuracy: 0.375
  adversarial_accuracy: 0.375
  successful_attacks: 0
  attack_success_rate: 0.0

epsilon 1/255:
  total_samples: 8
  clean_correct: 3
  adversarial_correct: 3
  clean_accuracy: 0.375
  adversarial_accuracy: 0.375
  successful_attacks: 0
  attack_success_rate: 0.0
```

Timing:

```text
evaluation_wall_seconds: 0.7688142889965093
total_wall_seconds: 1.6952154820028227
sample_epsilon_pairs: 16
evaluation_sample_epsilon_pairs_per_second: 20.811267726155187
```

CuPy `Stream.null` synchronization was performed before and after the timed
evaluation region. This tiny 8-sample run validates runner integration only;
it is not a robustness conclusion and not a performance benchmark.

Artifact policy:

* `results/runs/` contains run-specific experiment outputs and is ignored by
  Git.
* Runner code, schemas, tests, and verified summary documentation are tracked.
* Later curated benchmark summaries or plots may be intentionally committed
  under a separate final/curated artifact convention.

Large-scale FGSM evaluation belongs to later phases.

#### EWP3-C: Medium-Scale GPU FGSM Sanity Experiment

Status:

COMPLETE.

Goal:

Validate that the production EWP3-B runner remains stable on a nontrivial
real CIFAR-10 subset and that its saved artifacts can produce curated
quantitative evidence. This is not the full 10k evaluation, not the CPU/GPU
scaling benchmark, and not PGD.

Implemented local analysis support:

* Curated plotting/analysis script:

```text
python -m experiments.fgsm.plot_fgsm_run
```

* Reads saved runner artifacts from `results/runs/<run_id>/`.
* Validates `status = COMPLETED`, expected sample count, expected epsilon
  order, dataset checksum metadata, finite metrics, positive timing values,
  and artifact readability.
* Reuses existing FGSM plotting helpers from `src.plotting` for robustness
  figures.
* Adds a concise runtime/throughput summary plot derived from `timing.json`.
* Does not load CIFAR-10, run the model, import CuPy, or alter the numerical
  path.

Validated medium-scale workload:

```text
backend: cupy
split: test
sample_count: 1000
batch_size: 32
seed: 42
epsilons: [0, 1/255, 2/255, 4/255, 8/255]
checkpoint: results/checkpoints/portfolio_baseline_best.npz
data_dir: data/raw
```

Validated Hawaii cluster environment:

```text
run_id: 20260811T173256700165Z_fgsm_cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
hostname: csg-brook01
Git commit recorded by run: 729deed8f0c1c2e15b94d70645352eab54d9186f
repository state at runtime: clean
dataset checksum validation: PASS
```

Raw runner outputs remain under the ignored location:

```text
results/runs/<run_id>/
  config.json
  environment.json
  metrics.csv
  metrics.json
  timing.json
  summary.json
  status.json
```

Curated EWP3-C outputs are written separately:

```text
results/curated/ewp3c/<run_id>/
  robustness_summary.csv
  timing_summary.json
  run_metadata.json
  accuracy_vs_epsilon.png
  attack_success_rate_vs_epsilon.png
  accuracy_drop_vs_epsilon.png
  runtime_throughput_summary.png
```

Curated metadata preserves run ID, backend, GPU, CuPy version, Python version,
NumPy version, checkpoint path, dataset split, sample count, batch size,
epsilon values, timing methodology, Git commit/dirty state, and seed.

Reproducible cluster commands:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 1000 --batch-size 32 --epsilons 0,1/255,2/255,4/255,8/255 --output-root results/runs
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3c --expected-sample-count 1000 --expected-epsilons 0,1/255,2/255,4/255,8/255
```

Validated robustness summary:

| Epsilon | Clean Accuracy | Adversarial Accuracy | Accuracy Drop | Attack Success Rate |
| ------- | -------------- | -------------------- | ------------- | ------------------- |
| 0 | 0.481 | 0.481 | 0.000 | 0.000000000000000 |
| 1/255 | 0.481 | 0.317 | 0.164 | 0.340956340956341 |
| 2/255 | 0.481 | 0.202 | 0.279 | 0.580041580041580 |
| 4/255 | 0.481 | 0.090 | 0.391 | 0.812889812889813 |
| 8/255 | 0.481 | 0.009 | 0.472 | 0.981288981288981 |

Validated timing:

```text
evaluation_wall_seconds: 10.670563029998448
total_wall_seconds: 11.60283667499607
sample_epsilon_pairs: 5000
evaluation_sample_epsilon_pairs_per_second: 468.5788356193916
timing method: time.perf_counter
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

Curated evidence tracked for EWP3-C:

```text
results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/
  robustness_summary.csv
  timing_summary.json
  run_metadata.json
  accuracy_vs_epsilon.png
  attack_success_rate_vs_epsilon.png
  accuracy_drop_vs_epsilon.png
  runtime_throughput_summary.png
```

Closeout validation results:

* Runner `status.json` is `COMPLETED`.
* All five epsilon rows are present in order.
* No metric is `NaN` or `Inf`.
* Each metric row reports `total_samples = 1000`.
* CIFAR-10 checksum metadata remains valid.
* Timing values are positive.
* Curated CSV, JSON summaries, and four PNG figures are generated from saved
  artifacts.
* `results/runs/` remains ignored and no raw run directory is accidentally
  committed.
* No CPU fallback occurs in the CuPy run.

This 1000-sample run is medium-scale evidence only. It is not the final full
10k CIFAR-10 robustness evaluation, not a CPU/GPU benchmark, and not the
batch-size/sample-count scaling study.

Local validation:

```text
tests/test_fgsm_run_curation.py
```

The local tests use synthetic artifact directories and validate loading,
epsilon ordering, curated summary generation, plotting, overwrite behavior,
and missing/invalid artifact failures. They do not require CIFAR-10 data or a
GPU.

#### EWP3-D: CPU/GPU Scaling and Performance Benchmark

Status:

COMPLETE.

Goal:

Produce reproducible, defensible CPU/GPU performance evidence for the existing
FGSM robustness evaluation path. This phase measures systems behavior only; it
does not change numerical semantics, model architecture, FGSM definitions, or
robustness metrics.

Implemented local benchmark support:

* Scheduler-neutral benchmark driver:

```text
python -m experiments.fgsm.run_fgsm_benchmark
```

* Launches the existing production FGSM runner for every benchmark point.
* Does not implement a second numerical path.
* Supports sample-count scaling, CuPy batch-size scaling, repeated measured
  runs, excluded warm-up runs, per-point failure recording, aggregation, and
  plotting from saved benchmark artifacts.
* Supports optional matched NumPy/CuPy batch-size scaling through
  `--batch-scaling-backends numpy,cupy`.
* Preserves raw individual runner outputs under ignored `results/runs/`.
* Writes aggregate benchmark artifacts under `results/benchmarks/<benchmark_id>/`.

Default sample-count scaling matrix:

```text
backends: [numpy, cupy]
sample_counts: [100, 250, 500, 1000, 2000]
batch_size: 32
epsilons: [0, 4/255]
repeats: 3 measured repeats
warmup_runs: 1 excluded warm-up per workload
```

Default CuPy batch-size scaling matrix:

```text
backend: cupy
sample_count: 1000
batch_sizes: [8, 16, 32, 64, 128]
epsilons: [0, 4/255]
repeats: 3 measured repeats
warmup_runs: 1 excluded warm-up per workload
```

Matched batch-size extension:

```text
backends: [numpy, cupy]
sample_count: 1000
batch_sizes: [8, 16, 32, 64, 128]
epsilons: [0, 4/255]
repeats: 3 measured repeats
warmup_runs: 1 excluded warm-up per workload
```

The matched extension is intended to quantify the observed CPU/GPU crossover
region after the first full benchmark showed NumPy faster than CuPy at
matched `batch_size = 32`, while CuPy throughput improved strongly at larger
batch sizes.

The default matrix implies 45 measured runner invocations and 15 warm-up
invocations, for 60 total runner invocations:

```text
sample-count scaling: 5 sample counts * 2 backends * (1 warm-up + 3 repeats) = 40
batch-size scaling: 5 batch sizes * 1 backend * (1 warm-up + 3 repeats) = 20
```

Timing methodology:

* Timing comes from the EWP3-B runner's `timing.json`.
* CPU and GPU timing use `time.perf_counter`.
* CuPy timing synchronizes `cupy.cuda.Stream.null` before and after the
  evaluation region through the validated runner path.
* Benchmark claims use `evaluation_wall_seconds` unless explicitly labeled as
  total-wall timing.
* This phase does not claim kernel-only timing.

Speedup definition:

```text
CPU evaluation_wall_seconds / GPU evaluation_wall_seconds
```

Speedups are computed only for matched workloads with the same sample count,
batch size, epsilon workload, seed, checkpoint, dataset split, and repeat
semantics. Total-wall speedup is recorded separately where available.

Statistical summary:

For each measured workload, aggregate artifacts record:

```text
mean
median
sample standard deviation
min
max
completed repeat count
failed repeat count
```

With three repeats, variability is treated as an engineering stability signal,
not a high-confidence statistical claim.

Benchmark artifact schema:

```text
results/benchmarks/<benchmark_id>/
  config.json
  benchmark_runs.csv
  benchmark_runs.json
  benchmark_summary.csv
  benchmark_summary.json
  speedup_summary.csv
  speedup_summary.json
  crossover_analysis.json
  status.json
  plots/
    runtime_vs_sample_count.png
    throughput_vs_sample_count.png
    speedup_vs_sample_count.png
    runtime_vs_batch_size.png
    throughput_vs_batch_size.png
    speedup_vs_batch_size.png
    cupy_runtime_vs_batch_size.png
    cupy_throughput_vs_batch_size.png
```

Raw individual runs remain under:

```text
results/runs/<run_id>/
```

and remain ignored by Git.

Failure handling:

* Individual runner failures are recorded in `benchmark_runs.csv/json` with
  `status = FAILED`, error type, and error message.
* Earlier valid results are preserved after a later failed configuration.
* Aggregate summaries include completed and failed repeat counts.
* CuPy requests still rely on the runner's no-fallback behavior; CuPy is not
  silently replaced by NumPy.
* A benchmark-level `status.json` records `COMPLETED`, `COMPLETED_WITH_FAILURES`,
  or `FAILED`.

Crossover analysis:

* `speedup_summary.json` includes a `crossover_analysis` object.
* `crossover_analysis.json` records the first tested batch size with
  `evaluation_speedup_median > 1`, the associated speedup, the maximum tested
  speedup, and the batch size associated with that maximum.
* These values are computed from matched CPU/GPU benchmark artifacts and must
  not be manually entered.

Required plots:

* CPU vs GPU evaluation runtime vs sample count.
* CPU vs GPU throughput vs sample count.
* GPU speedup vs sample count.
* CPU vs GPU evaluation runtime vs batch size when matched batch-size
  backends are enabled.
* CPU vs GPU throughput vs batch size when matched batch-size backends are
  enabled.
* GPU speedup vs batch size when matched batch-size backends are enabled.
* CuPy evaluation runtime vs batch size.
* CuPy throughput vs batch size.

Local validation:

```text
tests/test_fgsm_benchmark.py
```

The local tests use synthetic timing/artifact data and monkeypatch the runner.
They cover CLI/config parsing, benchmark matrix generation, repeat and warm-up
indexing, run ID uniqueness, aggregation statistics, matched CPU/GPU speedup,
matched batch-size workload generation, batch-size crossover detection,
partial failure preservation, and plotting from synthetic benchmark summaries.
They do not require CIFAR-10 data, CuPy, or a GPU.

Validated cluster benchmark command:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --sample-counts 100,250,500,1000,2000 --sample-scaling-backends numpy,cupy --sample-scaling-batch-size 32 --batch-sizes 8,16,32,64,128 --batch-scaling-backend cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

Planned matched batch-size validation command:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --skip-sample-scaling --batch-sizes 8,16,32,64,128 --batch-scaling-backends numpy,cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

Validated matched batch-size benchmark:

```text
benchmark_id: 20260811T185420645969Z_fgsm_benchmark
status: COMPLETED
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
sample_count: 1000
epsilons: [0, 4/255]
batch_sizes: [8, 16, 32, 64, 128]
repeats: 3
warmup_runs: 1
completed_repeats: 3 for every measured configuration
failed_repeats: 0 for every measured configuration
```

Median evaluation-wall speedup:

| Batch Size | CPU/GPU Speedup |
| ---------- | --------------- |
| 8 | 0.25152078941245753 |
| 16 | 0.4203870515032853 |
| 32 | 0.7614486302511735 |
| 64 | 1.467427603695772 |
| 128 | 2.8822301436224573 |

Throughput summary, in sample-epsilon pairs per second:

| Batch Size | NumPy Mean Throughput | CuPy Mean Throughput |
| ---------- | --------------------- | -------------------- |
| 8 | ~491.67 | ~123.69 |
| 16 | ~577.79 | ~242.77 |
| 32 | ~622.71 | ~474.55 |
| 64 | ~643.74 | ~945.14 |
| 128 | ~644.50 | ~1855.80 |

Engineering interpretation:

* At `batch_size = 32`, CuPy remained slower than NumPy.
* The first tested GPU-faster batch size was `64`, with median evaluation
  speedup `1.467427603695772`.
* The maximum tested speedup was `2.8822301436224573` at batch size `128`.
* Batch size `128` is the best tested batch size in this benchmark, not a
  global optimum claim.
* CuPy throughput scaled strongly through batch size `128`; NumPy throughput
  approached a plateau around the tested higher batch sizes.
* The earlier sample-count benchmark showed no CPU/GPU crossover at matched
  `batch_size = 32` for sample counts `100`, `250`, `500`, `1000`, and
  `2000`. This supports the conclusion that batch size / GPU utilization, not
  sample count alone, drove the crossover in the current implementation.
* Speedup values are evaluation-wall-time speedups, not kernel-only speedups.
* No custom CUDA kernel or GPU-specific Conv2D optimization has been applied.

Curated EWP3-D evidence:

```text
results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/
  benchmark_summary.csv
  speedup_summary.csv
  crossover_analysis.json
  benchmark_metadata.json
  runtime_vs_batch_size.png
  throughput_vs_batch_size.png
  speedup_vs_batch_size.png
```

Raw `results/runs/` and raw `results/benchmarks/` outputs are runtime-specific
and ignored by Git. Curated EWP3-D evidence may be intentionally tracked after
validation.

#### EWP3-E: Full CIFAR-10 Test-Set FGSM Robustness Evaluation

Status:

COMPLETE.

Goal:

Produce the final full CIFAR-10 test-set FGSM robustness evidence using the
validated runner and the best tested GPU configuration from EWP3-D. This is a
robustness evaluation phase, not a new performance benchmark.

Implementation status:

* Reuses the existing production runner:

```text
python -m experiments.fgsm.run_fgsm_experiment
```

* Reuses the existing curation script:

```text
python -m experiments.fgsm.plot_fgsm_run
```

* The curation script supports final-run metadata through `--interpretation`,
  and optional validation of expected backend and GPU metadata through
  `--expected-backend` and `--expected-gpu-name`.
* No FGSM semantics, robustness metrics, model architecture, numerical
  backend behavior, PGD implementation, or CUDA optimization is included.

Full workload:

```text
backend: cupy
dataset: validated CIFAR-10
split: test
sample_count: 10000
batch_size: 128
seed: 42
checkpoint: results/checkpoints/portfolio_baseline_best.npz
epsilons: [0, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255]
```

Batch size `128` is selected because EWP3-D found it to be the best tested
batch size in the current benchmark range. It is not a global optimum claim.

Validated run:

```text
run_id: 20260812T115232600695Z_fgsm_cupy
backend: cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
batch_size: 128
seed: 42
sample_count: 10000
CIFAR-10 checksum: PASS
Git commit at runtime: b5a755b457c4299d9dd1a7c77d195f6fc3d74bc4
status: COMPLETED
```

Validation criteria:

* `status.json` reports `COMPLETED`: passed.
* Metrics contain exactly 7 epsilon rows in the requested order: passed.
* Every row reports `total_samples = 10000`: passed.
* Metrics are finite: passed.
* Clean accuracy, adversarial accuracy, and attack success rate are bounded in
  `[0, 1]`: passed.
* Epsilon `0` metrics are internally consistent: passed.
* Dataset checksum metadata passes: passed.
* Run metadata confirms CuPy backend and `NVIDIA GeForce RTX 2080 Ti`: passed.
* Timing values are positive: passed.
* No CPU fallback occurs: passed through the validated CuPy runner path.

Cluster run command:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 10000 --batch-size 128 --epsilons 0,1/255,2/255,4/255,8/255,12/255,16/255 --seed 42 --output-root results/runs
```

Curated evidence command:

```bash
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3e --expected-sample-count 10000 --expected-epsilons 0,1/255,2/255,4/255,8/255,12/255,16/255 --expected-backend cupy --expected-gpu-name "NVIDIA GeForce RTX 2080 Ti" --interpretation "Full CIFAR-10 test-set FGSM robustness evaluation; final EWP3-E robustness evidence, not a performance benchmark."
```

Curated artifact evidence:

```text
results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/
  robustness_summary.csv
  timing_summary.json
  run_metadata.json
  accuracy_vs_epsilon.png
  attack_success_rate_vs_epsilon.png
  accuracy_drop_vs_epsilon.png
  runtime_throughput_summary.png
```

Full-test-set robustness summary:

| Epsilon | Clean Accuracy | Adversarial Accuracy | Accuracy Drop | Attack Success Rate |
| ------- | -------------- | -------------------- | ------------- | ------------------- |
| 0 | 0.4639 | 0.4639 | 0.0000 | 0.0000000000000000 |
| 1/255 | 0.4639 | 0.3020 | 0.1619 | 0.3489976287993102 |
| 2/255 | 0.4639 | 0.1854 | 0.2785 | 0.6003449019185170 |
| 4/255 | 0.4639 | 0.0743 | 0.3896 | 0.8398361715887045 |
| 8/255 | 0.4639 | 0.0099 | 0.4540 | 0.9786591937917655 |
| 12/255 | 0.4639 | 0.0017 | 0.4622 | 0.9963354171157577 |
| 16/255 | 0.4639 | 0.0004 | 0.4635 | 0.9991377452037077 |

Timing evidence:

```text
sample_epsilon_pairs: 70000
evaluation_wall_seconds: 37.00973283001804
total_wall_seconds: 37.96863090901752
evaluation_sample_epsilon_pairs_per_second: 1891.3943616265192
timing_method: time.perf_counter
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

These timing values document this experiment execution. They are not a
dedicated benchmark result and should not be compared directly against EWP3-D
speedup claims unless workloads are matched.

Comparison with EWP3-C:

The full 10k run follows the same broad trend as the earlier 1000-sample
EWP3-C sanity run: adversarial accuracy decreases sharply as epsilon
increases, and attack success rate approaches `1.0` at larger epsilons. The
full run is the stronger final robustness evidence because it covers the full
CIFAR-10 test split. The EWP3-C sanity run remains a medium-scale runner and
curation validation, not a statistical estimate to combine with EWP3-E.

Scope notes:

* This is FGSM robustness evidence, not PGD robustness evidence.
* The result is validated for the tested RTX 2080 Ti / CuPy `14.1.1` /
  Python `3.12.13` / NumPy `2.4.6` environment only.
* Batch size `128` is the best tested EWP3-D batch size, not a global optimum.
* Raw `results/runs/` outputs remain ignored; curated EWP3-E evidence is
  intentionally tracked.

#### EWP3-F: Final Portfolio Evidence and Project Presentation

Status:

IMPLEMENTED LOCALLY / REVIEW PENDING.

Goal:

Convert validated EWP1 through EWP3-E engineering results into a concise,
technically defensible portfolio presentation. This phase is documentation,
analysis, and visualization only. It does not implement PGD, modify numerical
code, change model architecture, rerun large experiments, or introduce a new
attack path.

Source-of-truth artifacts:

```text
results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/
results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/
```

Implemented artifacts:

```text
experiments/generate_final_portfolio_evidence.py
results/curated/portfolio/portfolio_summary.csv
results/curated/portfolio/portfolio_summary.json
results/curated/portfolio/final_performance_summary.png
results/curated/portfolio/final_robustness_summary.png
deliverables/EWP3F/portfolio_presentation.md
```

The final portfolio evidence script reads only tracked curated EWP3-D/EWP3-E
CSV/JSON artifacts and derives:

* a single-row machine-readable summary table,
* a combined performance figure showing CPU vs GPU throughput and speedup by
  batch size,
* a combined robustness figure showing clean accuracy, adversarial accuracy,
  and attack success rate across the full-test-set FGSM epsilon sweep.

README updates:

* Adds a concise final evidence snapshot for reviewers.
* Adds a Mermaid architecture diagram showing CIFAR-10 loading, backend
  boundary, CompactCNN, loss/input gradients, FGSM, robustness evaluation,
  runner artifacts, and curated evidence.
* Links the final portfolio summary CSV/JSON and combined figures.
* States scope constraints: FGSM-only robustness, no PGD, RTX 2080 Ti / CuPy
  `14.1.1` environment, and batch `128` as best tested rather than globally
  optimal.

Portfolio narrative:

`deliverables/EWP3F/portfolio_presentation.md` records:

* source-of-truth evidence map,
* final engineering narrative,
* systems-focused, ML/robustness-focused, and balanced resume bullet variants,
* concise interview narrative covering project scope, difficulty, GPU
  underutilization diagnosis, batch-size crossover, correctness preservation,
  and next steps.

Validation:

```text
tests/test_final_portfolio_evidence.py
```

The tests use synthetic curated artifacts and do not require CIFAR-10, CuPy,
or a GPU. They validate summary derivation, plot generation, backend metadata
guardrails, and overwrite behavior.

---

### EWP4: PGD Attack Development

Goal:

Extend the validated adversarial-analysis pipeline with a production-quality
L-infinity projected gradient descent attack while preserving the current
NumPy-first numerical architecture.

Scope:

* Keep NumPy as the authoritative reference backend.
* Reuse the existing model/loss/input-gradient path.
* Do not modify FGSM semantics, model architecture, or robustness metric
  definitions.
* Stage PGD work incrementally: core attack first, then GPU equivalence, then
  runner/evaluation support.
* Do not run large CIFAR-10 PGD experiments until the PGD core and equivalence
  slices are validated.

Status:

PARTIALLY COMPLETE.

#### EWP4-A: L-infinity PGD Core Attack

Status:

COMPLETE for local NumPy core validation.

Implemented functionality:

* `src.attacks.pgd_linf_attack(...)` implements iterative untargeted
  L-infinity PGD.
* Parameters: `epsilon`, `alpha`, `steps`, `random_start`, and `seed`.
* Uses `compute_input_gradient(...)` at each step.
* Enforces `[0, 1]` clipping and projection into the L-infinity epsilon-ball
  around the clean input.
* Preserves clean inputs and model parameters.
* Supports deterministic local NumPy random starts when a seed is provided.
* Allows `steps=0` as an explicit no-op returning a clean input copy.
* Validates invalid configurations such as negative epsilon, non-positive
  alpha when steps are positive, negative steps, invalid image shape/range, and
  invalid label shape.

Validation:

```text
tests/test_pgd.py
```

Local EWP4-A validation covers:

* `epsilon=0` behavior.
* `steps=0` no-op behavior.
* One-step PGD / FGSM relationship for `random_start=False`, `steps=1`, and
  `alpha=epsilon`.
* L-infinity projection and `[0, 1]` clipping.
* Deterministic random start with a fixed seed and different results for
  different seeds.
* Multi-step updates and batch support.
* Clean-input and model-parameter preservation.
* Invalid configuration handling.

Not included in EWP4-A:

* CuPy/GPU PGD equivalence validation.
* PGD robustness evaluation or epsilon sweeps.
* PGD experiment-runner integration.
* Full CIFAR-10 PGD experiments.
* CUDA kernel optimization.

#### EWP4-B: NumPy/CuPy PGD Numerical Equivalence

Status:

COMPLETE.

Implemented functionality:

* `tests/test_cupy_pgd_equivalence.py` adds optional `requires_cupy` coverage
  for PGD equivalence.
* Tests use synchronized NumPy/CuPy `CompactCNN` parameters, deterministic
  synthetic inputs, identical labels, and identical PGD hyperparameters.
* No-random-start PGD validates final adversarial images, adversarial logits,
  exact predictions, projection/clipping invariants, and parameter
  preservation.
* Multi-step PGD uses `steps > 1` and `alpha < epsilon` to exercise the
  iterative path.
* Shared-random-start validation uses a deterministic NumPy-generated initial
  adversarial state transferred to CuPy. It does not compare backend-native
  RNG streams.
* `epsilon=0` validates clean-image copy semantics across backends.
* A compact one-step regression checks that PGD with
  `random_start=False`, `steps=1`, and `alpha=epsilon` matches existing FGSM
  semantics on both backends.

Minimal production refactor:

* `_pgd_linf_attack_from_initial(...)` is a private helper used by both the
  public PGD function and equivalence tests.
* Public `pgd_linf_attack(...)` semantics are unchanged.
* FGSM, robustness metrics, model architecture, and runner code are unchanged.

Validation:

```text
tests/test_cupy_pgd_equivalence.py
```

Local non-CUDA result:

```text
4 skipped
```

Real GPU validation:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12
CUDA-capable device count: 1

python -m pytest -q tests/test_cupy_pgd_equivalence.py -rs
4 passed in 6.22s

python -m pytest -q -m "not requires_data"
294 passed, 3 deselected in 16.65s

git diff --check
passed
```

Validated equivalence scope:

* Multi-step no-random-start PGD.
* `steps > 1`.
* `alpha < epsilon`.
* Deterministic shared-initial-state PGD.
* `epsilon=0`.
* Final adversarial image equivalence.
* Adversarial logits equivalence.
* Exact adversarial predictions.
* L-infinity projection invariant.
* `[0, 1]` clipping invariant.
* Parameter preservation.
* CuPy device integrity.
* One-step PGD / FGSM relationship.

Numerical tolerances:

```text
Tensor comparisons: rtol=1e-5, atol=1e-6
Predictions: exact equality
```

RNG strategy:

NumPy and CuPy RNG streams were not required to match. Cross-backend
random-start equivalence used the same deterministic initial adversarial state
transferred to both backends, isolating PGD numerical equivalence from backend
RNG implementation differences.

Refactor safety:

The EWP4-B production refactor preserves public `pgd_linf_attack(...)`
semantics and introduces only the private initial-state helper needed for
equivalence validation. FGSM, robustness metrics, model/loss semantics, and
runner code are unchanged. No CPU fallback was introduced.

Not included in EWP4-B:

* PGD robustness evaluation or epsilon sweeps.
* PGD experiment-runner integration.
* Full CIFAR-10 PGD experiments.
* CUDA kernel optimization.

#### EWP4-C: PGD Experiment Runner, Curation, and Cluster Smoke

Status:

IMPLEMENTED LOCALLY / CLUSTER SMOKE PENDING.

Implemented functionality:

* `experiments/pgd/run_pgd_experiment.py` adds a scheduler-neutral
  `pgd_linf` runner that reuses the existing cluster runner infrastructure for
  environment metadata, staged CIFAR-10 validation, checkpoint loading,
  deterministic subset selection, backend batching, run directories,
  `RUNNING` / `COMPLETED` / `FAILED` status artifacts, and synchronized CuPy
  timing.
* The runner calls the production `src.attacks.pgd_linf_attack(...)`
  implementation. It does not duplicate PGD math, modify FGSM semantics, or
  modify robustness metric definitions.
* Required PGD configuration is recorded in `config.json`: `attack`,
  `backend`, `data_dir`, `checkpoint`, `split`, `max_samples`, `batch_size`,
  `epsilon`, `alpha`, `steps`, `random_start`, `seed`, `output_root`, and
  `run_id`.
* PGD metrics preserve the existing robustness semantics:
  `successful_attacks / clean_correct_samples`, or `0.0` when there are no
  clean-correct samples.
* PGD timing records `evaluation_wall_seconds`, `total_wall_seconds`,
  `sample_count`, `pgd_steps`, `gradient_evaluations`, `samples_per_second`,
  and `sample_steps_per_second`. For CuPy, the runner synchronizes
  `cupy Stream.null` before and after the timed evaluation region.
* `experiments/pgd/plot_pgd_run.py` adds curation for a PGD smoke run. It
  validates the raw artifacts and writes a small curated artifact set under
  `results/curated/ewp4c/<run_id>/`.

PGD raw artifact schema:

```text
results/runs/<run_id>/
  config.json
  environment.json
  metrics.csv
  metrics.json
  timing.json
  summary.json
  status.json
```

Curated EWP4-C smoke artifact schema:

```text
results/curated/ewp4c/<run_id>/
  robustness_summary.csv
  timing_summary.json
  run_metadata.json
  pgd_smoke_summary.png
```

Planned real cluster smoke workload:

```text
backend: cupy
split: test
sample_count: 32
batch_size: 8
epsilon: 8/255
alpha: 2/255
steps: 10
random_start: true
seed: 42
checkpoint: results/checkpoints/portfolio_baseline_best.npz
```

Local validation:

```text
tests/test_pgd_experiment_runner.py
tests/test_pgd_run_curation.py
```

Not included in EWP4-C:

* Full 10k PGD evaluation.
* PGD epsilon sweeps or hyperparameter matrices.
* Multiple PGD restarts.
* PGD-vs-FGSM final robustness comparison.
* CUDA kernel optimization.

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
EWP3-A -> COMPLETE
EWP3-B -> COMPLETE
EWP3-C -> COMPLETE
EWP3-D -> COMPLETE
EWP3-E -> COMPLETE
EWP3-F -> COMPLETE
EWP4-A -> COMPLETE
EWP4-B -> COMPLETE
EWP4-C -> IMPLEMENTED LOCALLY / CLUSTER SMOKE PENDING
Next -> EWP4-C cluster smoke validation
```

Full PGD robustness evaluation, black-box attacks, and adversarial training
remain deferred.

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
