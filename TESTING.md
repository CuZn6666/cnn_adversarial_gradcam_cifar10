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
Non-CuPy local regression: 300 passed, 23 deselected
Full local suite: 300 passed, 23 skipped
EWP3-A local staging validation tests: 4 passed
EWP3-B local runner infrastructure tests: 7 passed
EWP3-C/EWP3-E local run-curation tests: 11 passed
EWP3-C real 1000-sample CuPy sanity run: completed and curated
EWP3-D local benchmark infrastructure tests: 10 passed
EWP3-E full 10k CuPy robustness run: completed and curated
EWP3-F local portfolio evidence tests: 4 passed
EWP4-A local PGD core tests: 18 passed
EWP4-B local PGD CuPy equivalence slice: 4 skipped because cupy is not installed
EWP4-B real GPU PGD equivalence tests: 4 passed
EWP4-C local PGD runner/curation tests: 19 passed
EWP4-C real 32-sample CuPy PGD smoke run: completed and curated
EWP4-D local final PGD curation/comparison preparation tests: 17 passed
EWP4-D full 10k CuPy PGD run: completed and curated
EWP4-D final PGD configuration: epsilon=8/255, alpha=2/255, steps=10, random_start=true, seed=42
EWP4-D final PGD result: clean_accuracy=0.4639, adversarial_accuracy=0.0023, attack_success_rate=0.9950420349213193
EWP4-D final timing: evaluation_wall_seconds=49.57189846501569, total_wall_seconds=50.52436269199825, gradient_evaluations=100000
EWP4-E 32-sample CuPy PGD epsilon-sweep smoke: completed for epsilons 0,4/255,8/255
EWP4-E full 10k CuPy PGD epsilon sweep: completed for 7 epsilons
EWP4-E configuration: alpha=2/255, steps=10, random_start=true, seed=42, batch_size=128
EWP4-E adversarial accuracy: 46.39%, 30.47%, 17.38%, 4.58%, 0.23%, 0.01%, 0.00%
EWP4-E ASR: 0.00%, 34.32%, 62.54%, 90.13%, 99.50%, 99.98%, 100.00%
EWP2-B local slice on this machine: 2 skipped because cupy is not installed
EWP2-C local slice on this machine: 2 skipped because cupy is not installed
EWP2-D local slice on this machine: 3 skipped because cupy is not installed
CuPy runtime/equivalence slices on this machine: 23 skipped because cupy is not installed
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

Latest verified real GPU EWP2-D and EWP2 closeout state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
EWP2-D robustness/sweep equivalence: 3 passed in 4.25s
Non-data cluster regression: 236 passed, 3 deselected in 8.96s
```

NumPy remains the authoritative correctness reference. Compatibility is
recorded only for the tested GPU/CUDA/CuPy/Python configuration above.

Latest verified real cluster EWP3-A state:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12.13
NumPy: 2.4.6
CUDA-capable device count: 1
Slurm allocation: 1 GPU
CIFAR-10 archive size: 170498071 bytes
CIFAR-10 MD5: c58f30108f718f92721af3b95e74349a
CIFAR-10 checksum validation: PASS
NumPy environment/dataset validation: PASS
CuPy environment/dataset validation: PASS
requires_data suite: 3 passed, 240 deselected in 3.34s
```

EWP3-A validation is recorded only for the tested cluster environment above.

Latest verified real cluster EWP3-B state:

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
epsilons: [0.0, 0.00392156862745098]
status: COMPLETED
evaluation_wall_seconds: 0.7688142889965093
total_wall_seconds: 1.6952154820028227
sample_epsilon_pairs: 16
evaluation_sample_epsilon_pairs_per_second: 20.811267726155187
```

The smoke run validates runner integration only. It is not a robustness
conclusion and not a performance benchmark.

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

## WP9-WP12: Attack Work Status

EWP4-A now provides a local NumPy-validated L-infinity PGD core attack. EWP4-B
validates NumPy/CuPy PGD numerical equivalence on the tested RTX 2080 Ti
environment. EWP4-C adds PGD runner and curation infrastructure validated by a
small real cluster smoke run. EWP4-D prepares final PGD curation and
FGSM-vs-PGD comparison infrastructure. Full PGD robustness results and
black-box attacks remain pending.

Current expected result:

* `src.attacks.pgd_linf_attack(...)` exists for local PGD core validation.
* `tests/test_cupy_pgd_equivalence.py` exists for optional PGD GPU
  equivalence validation.
* `experiments/pgd/run_pgd_experiment.py` exists for PGD runner smoke
  validation.
* `experiments/pgd/plot_pgd_run.py` exists for PGD smoke artifact curation.
* Curated EWP4-C smoke evidence exists under
  `results/curated/ewp4c/20260812T134536776276Z_pgd_linf_cupy/`.
* EWP4-D final-run curation and comparison tests exist for preparation.
* No full PGD robustness result exists yet.
* No black-box attack implementation exists.
* No query-count evaluation exists.

Do not add PGD robustness sweeps, PGD cluster experiments, black-box attacks,
or black-box tests unless the active task explicitly opens that phase.

## EWP4-A L-infinity PGD Core Attack

Status: COMPLETE for local NumPy core validation.

Focused tests:

```text
tests/test_pgd.py
```

The local PGD test slice validates:

* `epsilon=0` returns a clean input copy.
* `steps=0` is an explicit no-op and does not apply random initialization.
* One-step PGD matches existing FGSM semantics when `random_start=False`,
  `steps=1`, and `alpha=epsilon`.
* L-infinity projection and `[0, 1]` clipping.
* Deterministic NumPy random start when a seed is provided.
* Different seeds can produce different random starts.
* Multi-step updates and batch support.
* The clean input is not mutated.
* `CompactCNN` trainable parameters are not mutated by the attack.
* Invalid configuration handling for negative epsilon, invalid alpha, negative
  steps, invalid image shape/range, invalid label shape, invalid
  `random_start`, and invalid seed.

EWP4-A intentionally does not validate CuPy/GPU PGD equivalence, PGD
robustness metrics, PGD experiment runners, or full CIFAR-10 PGD execution.

Focused local command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_pgd.py
```

Expected local result:

```text
18 passed
```

## EWP4-B NumPy/CuPy PGD Equivalence Tests

Status: COMPLETE.

Optional GPU tests:

```text
tests/test_cupy_pgd_equivalence.py
```

The tests validate NumPy/CuPy equivalence for:

* Synchronized `CompactCNN` parameters.
* No-random-start multi-step PGD with `steps > 1` and `alpha < epsilon`.
* Shared-initial-state PGD using an identical deterministic initial
  adversarial image transferred from NumPy to CuPy.
* `epsilon=0` clean-image copy semantics.
* Final adversarial images.
* Final adversarial logits and exact predictions.
* Projection/clipping invariants for both backends.
* Parameter preservation on both backends.
* CuPy device integrity for model parameters, adversarial images, logits,
  predictions, and recorded forward/backward tensors.
* One-step PGD / FGSM relationship on both NumPy and CuPy.

The tests use `rtol=1e-5` and `atol=1e-6` for tensor comparisons. Predictions
are compared exactly.

RNG policy:

* Production `pgd_linf_attack(...)` keeps backend-native random-start
  semantics.
* Equivalence tests do not require NumPy and CuPy RNG streams to match.
* Shared-random-start validation uses `_pgd_linf_attack_from_initial(...)`, a
  private helper that lets tests inject the same initial adversarial state
  while preserving the public PGD API.

Local non-CuPy validation command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_pgd_equivalence.py -rs
```

Expected local result on machines without CuPy/CUDA:

```text
4 skipped
```

GPU validation command:

```bash
python -m pytest -q tests/test_cupy_pgd_equivalence.py -rs
python -m pytest -q -m "not requires_data"
```

Validated real GPU environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12
CUDA-capable device count: 1
```

Real GPU result:

```text
python -m pytest -q tests/test_cupy_pgd_equivalence.py -rs
4 passed in 6.22s

python -m pytest -q -m "not requires_data"
294 passed, 3 deselected in 16.65s

git diff --check
passed
```

EWP4-B does not claim compatibility with untested GPU/CUDA/CuPy/Python
configurations.

EWP4-B does not validate PGD robustness metrics, PGD experiment runners, full
CIFAR-10 PGD execution, or CUDA kernel optimization.

## EWP4-C PGD Runner and Curation Tests

Status: COMPLETE.

Local infrastructure tests:

```text
tests/test_pgd_experiment_runner.py
tests/test_pgd_run_curation.py
```

The tests validate:

* PGD runner CLI/config parsing, including `epsilon`, `alpha`, `steps`,
  `random_start`, `seed`, backend, data path, checkpoint, output root, and
  `run_id`.
* Invalid configuration handling for invalid backend, batch size, epsilon,
  alpha, steps, random-start type, run-id, and multi-value single-PGD
  parameters.
* Safe run-directory collision behavior.
* NumPy environment metadata collection without requiring CuPy.
* PGD metrics serialization to `metrics.csv` and `metrics.json`.
* Tiny synthetic NumPy PGD runner execution with staged-data/checkpoint
  monkeypatches.
* `FAILED` status recording for a missing checkpoint.
* PGD curation validation from synthetic raw artifacts.
* Curated PGD smoke outputs: `robustness_summary.csv`,
  `timing_summary.json`, `run_metadata.json`, and `pgd_smoke_summary.png`.
* Missing artifact, invalid status, non-finite metric, out-of-range metric,
  mismatched config, unexpected backend/GPU, and bad checksum rejection.

Focused local commands:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/pgd/run_pgd_experiment.py experiments/pgd/plot_pgd_run.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_pgd_experiment_runner.py tests/test_pgd_run_curation.py
```

Expected local result:

```text
19 passed
```

Validated RTX 2080 Ti cluster smoke command:

```bash
python -m experiments.pgd.run_pgd_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 32 --batch-size 8 --epsilon 8/255 --alpha 2/255 --steps 10 --random-start --seed 42 --output-root results/runs
```

Curate the smoke artifacts with:

```bash
python -m experiments.pgd.plot_pgd_run --run-dir results/runs/<run_id> --output-root results/curated/ewp4c --expected-sample-count 32 --expected-epsilon 8/255 --expected-backend cupy --expected-gpu-name "NVIDIA GeForce RTX 2080 Ti" --interpretation "Small PGD cluster smoke run; use for runner and artifact validation, not as final PGD robustness evidence."
```

Real cluster smoke result:

```text
run_id: 20260812T134536776276Z_pgd_linf_cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
backend: cupy
dataset checksum: PASS
sample_count: 32
batch_size: 8
epsilon: 8/255
alpha: 2/255
steps: 10
random_start: true
seed: 42
status: COMPLETED
```

Validated smoke metrics:

```text
clean_accuracy: 0.40625
adversarial_accuracy: 0.03125
accuracy_drop: 0.375
attack_success_rate: 0.9230769230769231
```

Validated timing:

```text
evaluation_wall_seconds: 3.2447034979995806
total_wall_seconds: 4.805044429987902
gradient_evaluations: 320
sample_steps: 320
samples_per_second: 9.862226246474782
sample_steps_per_second: 98.62226246474782
```

Curated smoke artifacts:

```text
results/curated/ewp4c/20260812T134536776276Z_pgd_linf_cupy/
  robustness_summary.csv
  timing_summary.json
  run_metadata.json
  pgd_smoke_summary.png
```

The curated artifact validation checks attack/backend metadata, sample count,
PGD hyperparameters, dataset checksum metadata, finite and bounded metrics,
positive timing values, `gradient_evaluations = sample_count * steps`,
`sample_steps = sample_count * steps`, and a valid non-empty PNG file.

EWP4-C does not validate a full CIFAR-10 PGD sweep, multiple restarts,
PGD-vs-FGSM final comparisons, black-box attacks, or CUDA kernel optimization.

## EWP4-D Full PGD Evaluation Preparation Tests

Status: PREPARED / FULL 10K CLUSTER RUN PENDING.

Focused preparation tests:

```text
tests/test_pgd_run_curation.py
tests/test_pgd_fgsm_comparison.py
```

The EWP4-D preparation tests validate:

* Strict final PGD curation expectations for sample count, epsilon, alpha,
  steps, random start, seed, split, checkpoint path, backend, and GPU.
* Final PGD output naming via `final_pgd_robustness_summary.png`.
* Internal count consistency for clean/adversarial counts, accuracy drop, and
  attack success rate.
* `gradient_evaluations = sample_count * steps`.
* `sample_steps = sample_count * steps`.
* FGSM-vs-PGD comparison generation from curated artifacts only.
* Matched FGSM/PGD source validation for checkpoint, split, sample count,
  backend, GPU, and epsilon.
* Comparison CSV, JSON, and PNG output generation.
* Rejection of mismatched sample counts, checkpoint mismatches, and unsafe
  overwrite attempts.

Focused local commands:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/pgd/run_pgd_experiment.py experiments/pgd/plot_pgd_run.py experiments/pgd/compare_fgsm_pgd.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_pgd_run_curation.py tests/test_pgd_fgsm_comparison.py
```

Expected local result:

```text
17 passed
```

Planned full RTX 2080 Ti PGD run command:

```bash
python -m experiments.pgd.run_pgd_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 10000 --batch-size 128 --epsilon 8/255 --alpha 2/255 --steps 10 --random-start --seed 42 --output-root results/runs
```

Curate the final PGD artifacts with:

```bash
python -m experiments.pgd.plot_pgd_run --run-dir results/runs/<run_id> --output-root results/curated/ewp4d --expected-sample-count 10000 --expected-epsilon 8/255 --expected-alpha 2/255 --expected-steps 10 --expected-random-start true --expected-seed 42 --expected-split test --expected-checkpoint results/checkpoints/portfolio_baseline_best.npz --expected-backend cupy --expected-gpu-name "NVIDIA GeForce RTX 2080 Ti" --plot-filename final_pgd_robustness_summary.png --plot-title "Final PGD-Linf Robustness" --interpretation "Full CIFAR-10 test-set PGD-Linf robustness evaluation; final EWP4-D PGD evidence, not a CPU/GPU benchmark."
```

After the real PGD curated artifact exists, build the matched FGSM-vs-PGD
comparison with:

```bash
python -m experiments.pgd.compare_fgsm_pgd --fgsm-curated-dir results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy --pgd-curated-dir results/curated/ewp4d/<run_id> --output-dir results/curated/portfolio --epsilon 8/255 --expected-sample-count 10000
```

EWP4-D preparation does not execute the full 10k PGD run locally, does not
change PGD numerical semantics, does not add multi-restart PGD, and does not
add parameter sweeps.

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

The previous cluster CIFAR-10 archive was not a valid data-test signal:

```text
path: data/raw/cifar-10-python.tar.gz
size: 37M
observed MD5: 352dcf059b8b606c932d1db9b8c351a9
expected MD5: c58f30108f718f92721af3b95e74349a
```

This issue is resolved for the validated Hawaii cluster environment. The
current staged archive is:

```text
path: data/raw/cifar-10-python.tar.gz
size: 170498071 bytes
expected MD5: c58f30108f718f92721af3b95e74349a
observed MD5: c58f30108f718f92721af3b95e74349a
checksum: PASS
extracted directory: data/raw/cifar-10-batches-py
```

Validated data shape checks:

```text
train images: (50000, 3, 32, 32)
train labels: (50000,)
test images: (10000, 3, 32, 32)
test labels: (10000,)
class count: 10
```

Checksum validation remains authoritative. Do not change the expected
checksum, disable archive validation, weaken data tests, or modify the CIFAR-10
loader to hide future staging issues.

## EWP3-A Cluster Environment and Dataset Staging Validation

The scheduler-neutral validation utility is:

```text
scripts/validate_cluster_environment.py
```

Status: COMPLETE.

It validates:

* Python and NumPy versions.
* Requested backend availability.
* CuPy version, CUDA runtime query, visible CUDA device count, and GPU name
  when CuPy is installed.
* Staged CIFAR-10 archive presence, size, and MD5 checksum.
* Extracted CIFAR-10 directory presence.
* Train/test split loading through the repository's CIFAR-10 batch parser.
* Expected train/test shapes and class count.
* Human-readable output and optional JSON validation output.

The utility does not auto-download CIFAR-10. It may extract an already staged
archive only when `--extract-if-needed` is provided and the archive matches the
expected checksum.

Focused local tests:

```text
tests/test_cluster_environment_validation.py
```

These tests use temporary filesystem state, do not require CIFAR-10 data, and
do not require a real GPU.

Suggested local validation:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cluster_environment_validation.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_cupy"
```

Validated cluster environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12.13
NumPy: 2.4.6
CUDA-capable device count: 1
Slurm allocation: 1 GPU
```

Cluster validation artifacts generated:

```text
results/cluster_validation/cifar10_environment_numpy.json
results/cluster_validation/cifar10_environment_cupy.json
```

Both reports had `status = passed`. The CuPy report recorded environment
status `passed`, dataset status `passed`, CuPy `14.1.1`, device count `1`, and
GPU name `NVIDIA GeForce RTX 2080 Ti`.

These JSON files are run-specific cluster/environment outputs and are ignored
by Git. Keep the validation utility and tests tracked, but record verified
summary results in documentation unless a specific report artifact is curated
for publication.

Validated cluster commands:

```bash
python scripts/validate_cluster_environment.py --backend numpy --data-dir data/raw --json-output results/cluster_validation/cifar10_environment_numpy.json
python scripts/validate_cluster_environment.py --backend cupy --data-dir data/raw --extract-if-needed --json-output results/cluster_validation/cifar10_environment_cupy.json
python -m pytest -q -m requires_data
```

Validated result:

```text
requires_data suite: 3 passed, 240 deselected in 3.34s
```

## EWP3-B FGSM Experiment Runner Infrastructure

The scheduler-neutral FGSM experiment runner is:

```text
experiments/fgsm/run_fgsm_experiment.py
```

Status: COMPLETE.

It validates and records:

* Backend selection: `numpy` or `cupy`, with no silent CuPy-to-NumPy fallback.
* Explicit data directory, checkpoint path, split, sample count, batch size,
  epsilon list, seed, output root, and run identifier.
* Staged CIFAR-10 archive presence, checksum, and extracted directory before
  loading data.
* Isolated run directory creation and collision rejection.
* Run status artifacts: `RUNNING`, `COMPLETED`, or `FAILED`.
* Environment metadata including Python, NumPy, optional CuPy/CUDA/GPU,
  hostname, Git commit, and dirty state.
* Machine-readable `config.json`, `environment.json`, `metrics.json`,
  `metrics.csv`, `timing.json`, `summary.json`, and `status.json`.
* CPU timing with `time.perf_counter`.
* CuPy timing with CUDA stream synchronization around the timed evaluation
  region.

Focused local tests:

```text
tests/test_fgsm_experiment_runner.py
```

These tests cover CLI/config parsing, epsilon parsing, invalid config
rejection, run-directory collision behavior, environment metadata, metric
serialization, tiny NumPy execution using monkeypatched CIFAR-10 loading, and
failed-run status artifacts. They do not require real CIFAR-10 data, CuPy, or
a GPU.

Suggested local validation:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_fgsm_experiment_runner.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_cupy"
```

Validated cluster smoke command:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 8 --batch-size 2 --epsilons 0,1/255 --output-root results/runs
```

Validated cluster smoke artifacts:

```text
config.json
environment.json
metrics.csv
metrics.json
timing.json
summary.json
status.json
```

Validated smoke metrics:

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

Validated timing:

```text
evaluation_wall_seconds: 0.7688142889965093
total_wall_seconds: 1.6952154820028227
sample_epsilon_pairs: 16
evaluation_sample_epsilon_pairs_per_second: 20.811267726155187
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

`results/runs/` contains run-specific experiment outputs and is ignored by
Git. Keep raw run directories out of normal source commits; commit later
curated benchmark summaries or plots only through an explicit final artifact
policy.

This smoke command is intentionally small. Do not treat it as the large-scale
FGSM evaluation, a robustness conclusion, or a final CPU/GPU benchmark.

## EWP3-C Medium-Scale FGSM Run Curation

The curated analysis script is:

```text
experiments/fgsm/plot_fgsm_run.py
```

Status: COMPLETE.

It reads a completed raw runner directory from:

```text
results/runs/<run_id>/
```

and writes curated outputs under:

```text
results/curated/ewp3c/<run_id>/
```

Curated outputs:

```text
robustness_summary.csv
timing_summary.json
run_metadata.json
accuracy_vs_epsilon.png
attack_success_rate_vs_epsilon.png
accuracy_drop_vs_epsilon.png
runtime_throughput_summary.png
```

Validation checks:

* `status.json` must report `COMPLETED`.
* `config.json`, `environment.json`, `metrics.csv`, `metrics.json`,
  `timing.json`, `summary.json`, and `status.json` must be readable.
* Epsilon order must match the run config and any expected epsilon list.
* Optional expected sample count must match all metric rows.
* Dataset checksum metadata must be valid.
* Metrics must be finite.
* Timing values must be positive.

Focused local tests:

```text
tests/test_fgsm_run_curation.py
```

These tests use synthetic run artifacts and temporary output directories. They
cover artifact loading, epsilon ordering, curated CSV/JSON/PNG generation,
overwrite behavior, missing required artifacts, invalid run status, non-finite
metrics, and invalid checksum metadata. They do not require CIFAR-10 data,
CuPy, or a GPU.

Suggested local validation:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_fgsm_run_curation.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_cupy"
```

Validated 1000-sample cluster run:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 1000 --batch-size 32 --epsilons 0,1/255,2/255,4/255,8/255 --output-root results/runs
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3c --expected-sample-count 1000 --expected-epsilons 0,1/255,2/255,4/255,8/255
```

Validated environment and run:

```text
run_id: 20260811T173256700165Z_fgsm_cupy
backend: cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
sample_count: 1000
batch_size: 32
epsilons: [0, 1/255, 2/255, 4/255, 8/255]
status: COMPLETED
```

Validated metrics:

```text
epsilon 0:     clean_accuracy 0.481, adversarial_accuracy 0.481, accuracy_drop 0.000, attack_success_rate 0.000000000000000
epsilon 1/255: clean_accuracy 0.481, adversarial_accuracy 0.317, accuracy_drop 0.164, attack_success_rate 0.340956340956341
epsilon 2/255: clean_accuracy 0.481, adversarial_accuracy 0.202, accuracy_drop 0.279, attack_success_rate 0.580041580041580
epsilon 4/255: clean_accuracy 0.481, adversarial_accuracy 0.090, accuracy_drop 0.391, attack_success_rate 0.812889812889813
epsilon 8/255: clean_accuracy 0.481, adversarial_accuracy 0.009, accuracy_drop 0.472, attack_success_rate 0.981288981288981
```

Validated timing:

```text
evaluation_wall_seconds: 10.670563029998448
total_wall_seconds: 11.60283667499607
sample_epsilon_pairs: 5000
evaluation_sample_epsilon_pairs_per_second: 468.5788356193916
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

Curated evidence:

```text
results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/
```

The curated directory contains `robustness_summary.csv`,
`timing_summary.json`, `run_metadata.json`, and four PNG figures. The raw
`results/runs/<run_id>/` directory remains ignored and should not be committed.
This medium-scale run validates runner stability and artifact curation; it is
not the final full CIFAR-10 evaluation and not a CPU/GPU benchmark.

## EWP3-D CPU/GPU FGSM Benchmark Infrastructure

The benchmark driver is:

```text
experiments/fgsm/run_fgsm_benchmark.py
```

Status: COMPLETE.

It launches the existing production FGSM runner for each benchmark point and
does not duplicate the numerical evaluation path.

Default benchmark matrix:

```text
sample-count scaling:
  backends: numpy, cupy
  sample_counts: 100, 250, 500, 1000, 2000
  batch_size: 32
  epsilons: 0, 4/255

CuPy batch-size scaling:
  backend: cupy
  sample_count: 1000
  batch_sizes: 8, 16, 32, 64, 128
  epsilons: 0, 4/255

matched batch-size extension:
  backends: numpy, cupy
  sample_count: 1000
  batch_sizes: 8, 16, 32, 64, 128
  epsilons: 0, 4/255

repeats: 3 measured repeats
warmup_runs: 1 excluded warm-up per workload
```

The default matrix contains 45 measured runner invocations and 15 warm-up
invocations. Warm-up runs use the same workload as measured repeats but are
excluded from aggregate benchmark statistics.

Benchmark artifacts:

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

Raw per-repeat runner outputs remain under ignored `results/runs/<run_id>/`.
The benchmark aggregate directory is not automatically committed; curated
benchmark evidence should be reviewed before tracking.

Timing and speedup semantics:

* Timing uses the EWP3-B runner's `time.perf_counter` measurements.
* CuPy timing uses the runner's `cupy.cuda.Stream.null` synchronization before
  and after the evaluation region.
* Benchmark claims should use `evaluation_wall_seconds` unless explicitly
  labeled as total-wall timing.
* Evaluation speedup is defined as:

```text
CPU evaluation_wall_seconds / GPU evaluation_wall_seconds
```

* Speedups are computed only for matched CPU/GPU workloads.
* Aggregate statistics record mean, median, sample standard deviation, min,
  max, completed repeat count, and failed repeat count.
* `crossover_analysis.json` records the first tested batch size where
  `evaluation_speedup_median > 1`, plus the maximum tested speedup and its
  batch size.

Focused local tests:

```text
tests/test_fgsm_benchmark.py
```

These tests use synthetic runner timings and do not require CIFAR-10 data,
CuPy, or a GPU. They cover CLI/config parsing, matrix generation, repeat and
warm-up indexing, run ID uniqueness, aggregation statistics, matched CPU/GPU
speedup calculation, matched batch-size workload generation, crossover
detection, partial failure preservation, and plotting from synthetic benchmark
summaries.

Suggested local validation:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/fgsm/run_fgsm_benchmark.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_fgsm_benchmark.py
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_cupy"
```

Planned real cluster benchmark command:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --sample-counts 100,250,500,1000,2000 --sample-scaling-backends numpy,cupy --sample-scaling-batch-size 32 --batch-sizes 8,16,32,64,128 --batch-scaling-backend cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

Validated matched batch-size command:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --skip-sample-scaling --batch-sizes 8,16,32,64,128 --batch-scaling-backends numpy,cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

Validated real cluster result:

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

Validated median evaluation speedups:

```text
batch 8: 0.25152078941245753
batch 16: 0.4203870515032853
batch 32: 0.7614486302511735
batch 64: 1.467427603695772
batch 128: 2.8822301436224573
```

`crossover_analysis.json` reports first tested GPU-faster batch size `64` and
maximum tested speedup `2.8822301436224573` at batch size `128`. The values
are evaluation-wall-time speedups, not kernel-only speedups.

Curated evidence:

```text
results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/
```

The curated directory contains `benchmark_summary.csv`, `speedup_summary.csv`,
`crossover_analysis.json`, `benchmark_metadata.json`, and three matched
batch-size PNG plots. Raw `results/benchmarks/` and `results/runs/` outputs
remain ignored.

## EWP3-E Full CIFAR-10 FGSM Robustness Evaluation

Status: COMPLETE.

EWP3-E reuses the production FGSM runner and the existing curation script. It
does not introduce a new attack path, modify robustness semantics, or run PGD.

Validated full-run workload on the Hawaii RTX 2080 Ti cluster:

```text
backend: cupy
split: test
sample_count: 10000
batch_size: 128
seed: 42
epsilons: [0, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255]
checkpoint: results/checkpoints/portfolio_baseline_best.npz
data_dir: data/raw
run_id: 20260812T115232600695Z_fgsm_cupy
Git commit at runtime: b5a755b457c4299d9dd1a7c77d195f6fc3d74bc4
```

Batch size `128` is selected because EWP3-D found it to be the best tested
batch size in the current benchmark range. This is not a global optimum claim.

The curation script now supports EWP3-E validation gates:

```bash
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3e --expected-sample-count 10000 --expected-epsilons 0,1/255,2/255,4/255,8/255,12/255,16/255 --expected-backend cupy --expected-gpu-name "NVIDIA GeForce RTX 2080 Ti" --interpretation "Full CIFAR-10 test-set FGSM robustness evaluation; final EWP3-E robustness evidence, not a performance benchmark."
```

Closeout validation checks:

* `status.json` reports `COMPLETED`: passed.
* The metrics contain exactly seven epsilon rows in the requested order:
  passed.
* Every metric row reports `total_samples = 10000`: passed.
* Clean accuracy, adversarial accuracy, attack success rate, and timing values
  are finite: passed.
* Clean accuracy, adversarial accuracy, and attack success rate are bounded in
  `[0, 1]`: passed.
* The epsilon `0` row has matching clean/adversarial correct counts and
  accuracies, with zero successful attacks: passed.
* Dataset checksum metadata passes: passed.
* Run metadata confirms backend `cupy` and GPU `NVIDIA GeForce RTX 2080 Ti`.
* Timing values are positive: passed.
* Curated CSV, JSON, and PNG artifacts are generated from saved runner
  artifacts: passed.

Validated full-test-set robustness summary:

```text
epsilon 0:      clean_accuracy 0.4639, adversarial_accuracy 0.4639, accuracy_drop 0.0000, attack_success_rate 0.0000000000000000
epsilon 1/255:  clean_accuracy 0.4639, adversarial_accuracy 0.3020, accuracy_drop 0.1619, attack_success_rate 0.3489976287993102
epsilon 2/255:  clean_accuracy 0.4639, adversarial_accuracy 0.1854, accuracy_drop 0.2785, attack_success_rate 0.6003449019185170
epsilon 4/255:  clean_accuracy 0.4639, adversarial_accuracy 0.0743, accuracy_drop 0.3896, attack_success_rate 0.8398361715887045
epsilon 8/255:  clean_accuracy 0.4639, adversarial_accuracy 0.0099, accuracy_drop 0.4540, attack_success_rate 0.9786591937917655
epsilon 12/255: clean_accuracy 0.4639, adversarial_accuracy 0.0017, accuracy_drop 0.4622, attack_success_rate 0.9963354171157577
epsilon 16/255: clean_accuracy 0.4639, adversarial_accuracy 0.0004, accuracy_drop 0.4635, attack_success_rate 0.9991377452037077
```

Validated timing:

```text
sample_epsilon_pairs: 70000
evaluation_wall_seconds: 37.00973283001804
total_wall_seconds: 37.96863090901752
evaluation_sample_epsilon_pairs_per_second: 1891.3943616265192
timing_method: time.perf_counter
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

The timing values are experiment execution evidence, not a dedicated
CPU/GPU benchmark.

Curated evidence:

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

Raw `results/runs/` artifacts remain ignored. The curated EWP3-E directory is
intentional final robustness evidence.

Focused local tests:

```text
tests/test_fgsm_run_curation.py
```

The tests now cover full-run interpretation metadata, expected backend/GPU
validation, bounded robustness metrics, and epsilon-zero consistency in
addition to the EWP3-C curation behavior. They use synthetic artifacts and do
not require CIFAR-10 data, CuPy, or a GPU.

## EWP3-F Final Portfolio Evidence

Status: IMPLEMENTED LOCALLY / REVIEW PENDING.

Final portfolio evidence is generated from tracked curated EWP3-D and EWP3-E
artifacts only:

```text
results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/
results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/
```

The generation script is:

```text
experiments/generate_final_portfolio_evidence.py
```

Default outputs:

```text
results/curated/portfolio/portfolio_summary.csv
results/curated/portfolio/portfolio_summary.json
results/curated/portfolio/final_performance_summary.png
results/curated/portfolio/final_robustness_summary.png
```

The script derives the summary table and final figures from existing CSV/JSON
evidence. It does not load CIFAR-10, run a model, import CuPy, rerun
experiments, or modify numerical code.

Suggested regeneration command:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.generate_final_portfolio_evidence --overwrite
```

Focused local tests:

```text
tests/test_final_portfolio_evidence.py
```

The tests use synthetic curated EWP3-D/EWP3-E artifacts and temporary output
directories. They cover source-summary derivation, PNG generation,
unexpected-backend rejection, and overwrite protection. They do not require
CIFAR-10 data, CuPy, or a GPU.

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

EWP2 is complete for the planned NumPy/CuPy numerical-equivalence scope.

Full EWP2 validated equivalence matrix:

* EWP2-A: `ReLU.forward/backward`, `MaxPool2D.forward/backward`,
  `Flatten.forward/backward`, `Linear.forward/backward`, and
  `SoftmaxCrossEntropyLoss.forward/backward`.
* EWP2-B: `CompactCNN.forward`, scalar loss, loss gradient, full model
  backward, all trainable parameter gradients, one real `SGD.step()`, and
  `train_step(...)`.
* EWP2-C: input gradients, `epsilon=0` FGSM, nonzero-epsilon FGSM, clipping,
  adversarial images, adversarial logits, predictions, and the full
  clean-input-to-adversarial-forward path.
* EWP2-D: single-batch robustness metrics, raw counts, multi-batch
  aggregation, epsilon-zero invariants, epsilon ordering, epsilon-sweep
  metrics, and parameter preservation.

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

Status: COMPLETE.

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

Validated on the GPU cluster with:

```text
python -m pytest -q tests/test_cupy_robustness_equivalence.py -rs
3 passed in 4.25s

python -m pytest -q -m "not requires_data"
236 passed, 3 deselected in 8.96s
```

The validated environment was `NVIDIA GeForce RTX 2080 Ti`, CUDA Toolkit
`12.5`, CuPy `14.1.1`, Python `3.12`, with one GPU allocated through Slurm.
This completes the planned EWP2 NumPy/CuPy numerical-equivalence scope.

## Cluster / GPU Validation Boundary

Cluster/GPU execution is future extension work. The current repository has no
Slurm scripts and no CuPy dependency.

Before large-scale data processing, expanded evaluation subsets, repeated-seed
runs, GPU experiments, or cluster runs, ask the user whether to use the
university-provided ZITI cluster. Do not introduce GPU, CUDA, CuPy, Slurm, or
ZITI workflows without explicit approval.
