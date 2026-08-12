# cnn_adversarial_gradcam_cifar10

[![CI](https://github.com/CuZn6666/cnn_adversarial_gradcam_cifar10/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CuZn6666/cnn_adversarial_gradcam_cifar10/actions/workflows/ci.yml?query=branch%3Amain)

A **NumPy-first, from-scratch CompactCNN pipeline for CIFAR-10** with manual
forward/backward propagation, optional CuPy GPU execution, numerical
equivalence validation, FGSM adversarial robustness evaluation, CPU/GPU
benchmarking, and Grad-CAM-based explainability analysis.

This repository is built as an engineering project, not a thin wrapper around
PyTorch or TensorFlow: the CNN layers, gradients, optimizer path, input
gradients, and FGSM attack pipeline are implemented directly with NumPy-style
array operations. NumPy remains the default and authoritative correctness
reference; CuPy is an optional backend for the compute-heavy tensor path.

## Project Highlights

| Area | Verified status |
| ---- | --------------- |
| **From-scratch NumPy CNN** | `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Linear`, and `CompactCNN` implemented manually. |
| **Manual backward propagation** | Layer-level and full-model `backward(...)` pipeline implemented and tested. |
| **NumPy/CuPy backend** | NumPy is the reference backend; CuPy runtime validated on an RTX 2080 Ti with CuPy `14.1.1`. |
| **Numerical equivalence** | Backend primitives, layers, loss, full model training step, input gradients, FGSM, and robustness sweeps validated across NumPy/CuPy. |
| **Automated tests and CI** | Current local suite: `256 passed, 19 skipped`; non-CuPy regression: `256 passed, 19 deselected`. |
| **Clean CIFAR-10 baseline** | Deterministic NumPy baseline checkpoint reached `47.07%` validation accuracy and `44.73%` clean test-subset accuracy. |
| **Reproducible cluster runner** | Scheduler-neutral FGSM runner records config, environment, metrics, timing, status, and curated evidence. |
| **Full FGSM robustness** | Full CIFAR-10 test set evaluated with CuPy on `10,000` samples and seven FGSM epsilons. |
| **GPU performance evidence** | Matched CPU/GPU benchmark found first tested GPU-faster batch size `64` and best tested median speedup `2.88x` at batch `128`. |
| **Grad-CAM explainability** | Clean and adversarial Grad-CAM comparisons generated from the validated `relu2` target activation. |
| **Runtime engineering** | Profiling identified `Conv2D.backward` as the bottleneck; a focused NumPy optimization achieved a documented `207.72x` local speedup. |
| **Structured workflow** | Work-package-based development with tests, deliverables, metrics, plots, and tracked curated artifacts. |

## Final Evidence Snapshot

The final portfolio evidence is generated from tracked curated artifacts under
`results/curated/ewp3d/` and `results/curated/ewp3e/`:

| Claim | Validated result | Source |
| ----- | ---------------- | ------ |
| Full-test clean accuracy | `46.39%` on `10,000` CIFAR-10 test images | [EWP3-E robustness summary](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/robustness_summary.csv) |
| FGSM adversarial accuracy | `7.43%` at `4/255`, `0.99%` at `8/255`, `0.04%` at `16/255` | [EWP3-E robustness summary](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/robustness_summary.csv) |
| Full-run throughput | `1891.39` sample-epsilon pairs/s for `70,000` pairs | [EWP3-E timing summary](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/timing_summary.json) |
| CPU/GPU crossover | First tested GPU-faster batch size: `64` | [EWP3-D crossover analysis](results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/crossover_analysis.json) |
| Best tested speedup | Median `2.88x` evaluation-wall-time speedup at batch `128` | [EWP3-D speedup summary](results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/speedup_summary.csv) |

Summary artifacts:

* [Final portfolio summary CSV](results/curated/portfolio/portfolio_summary.csv)
* [Final portfolio summary JSON](results/curated/portfolio/portfolio_summary.json)

Regenerate the final summary and figures from tracked curated evidence:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.generate_final_portfolio_evidence --overwrite
```

| Performance scaling | Full-test FGSM robustness |
| ------------------- | ------------------------- |
| [![Final performance summary](results/curated/portfolio/final_performance_summary.png)](results/curated/portfolio/final_performance_summary.png) | [![Final robustness summary](results/curated/portfolio/final_robustness_summary.png)](results/curated/portfolio/final_robustness_summary.png) |

The performance speedup is evaluation-wall-time speedup for matched workloads:
`CPU evaluation_wall_seconds / GPU evaluation_wall_seconds`. Batch size `128`
is the best tested configuration in the benchmark range, not a global optimum.
The robustness evidence is FGSM-only; PGD and black-box attacks remain
deferred.

## Architecture Diagram

```mermaid
flowchart TD
    A[CIFAR-10 NCHW batches] --> B[CPU data loading and batching]
    B --> C{Backend boundary}
    C -->|NumPy reference| D[CompactCNN]
    C -->|CuPy GPU backend| D
    D --> E[Softmax cross entropy]
    E --> F[Manual backward pass]
    F --> G[Input gradients]
    G --> H[FGSM perturbation and clipping]
    H --> I[Adversarial forward pass]
    I --> J[Robustness metrics]
    J --> K[Experiment runner]
    K --> L[Raw run artifacts under results/runs]
    L --> M[Curation and plotting scripts]
    M --> N[Tracked portfolio evidence]
```

## Visual Results

### Clean CIFAR-10 baseline training and evaluation

| Training loss | Train vs validation accuracy | Confusion matrix |
| ------------- | ---------------------------- | ---------------- |
| [![Training Loss Curve](results/baseline/training_loss_curve.png)](results/baseline/training_loss_curve.png) | [![Train vs Validation Accuracy](results/baseline/train_validation_accuracy_curve.png)](results/baseline/train_validation_accuracy_curve.png) | [![CIFAR-10 Confusion Matrix](results/baseline/confusion_matrix.png)](results/baseline/confusion_matrix.png) |

Deterministic local baseline configuration:

```text
train_samples: 4096
validation_samples: 1024
test_samples: 1024
batch_size: 32
epochs: 15
learning_rate: 0.03
seed: 42
```

Selected by validation accuracy:

```text
best_epoch: 15
validation_accuracy: 0.4707
clean_test_subset_accuracy: 0.4473
```

Baseline artifacts:

- Local baseline checkpoint: `results/baseline/portfolio_baseline_best.npz`
  (not tracked; regenerate locally or provide through an external artifact
  store for larger runs)
- [Training history JSON](results/baseline/portfolio_training_history.json)
- [Final metrics JSON](results/baseline/portfolio_final_metrics.json)

Checkpoint artifact policy: `.npz` model checkpoints are intentionally ignored
by Git to avoid committing large binary artifacts. The repository tracks
configuration, metrics, plots, and small JSON/PNG result summaries. Checkpoints
should be regenerated from the documented scripts or distributed through a
release artifact, cluster storage path, or another explicit external artifact
store when an experiment needs an exact saved model.

### FGSM quantitative robustness with reproducible baseline

| Accuracy vs epsilon | Attack success rate | Accuracy drop |
| ------------------- | ------------------- | ------------- |
| [![Accuracy vs Epsilon](results/fgsm/accuracy_vs_epsilon.png)](results/fgsm/accuracy_vs_epsilon.png) | [![Attack Success Rate vs Epsilon](results/fgsm/attack_success_rate_vs_epsilon.png)](results/fgsm/attack_success_rate_vs_epsilon.png) | [![Accuracy Drop vs Epsilon](results/fgsm/accuracy_drop_vs_epsilon.png)](results/fgsm/accuracy_drop_vs_epsilon.png) |

This FGSM quantitative evaluation uses the stronger reproducible baseline
checkpoint:

```text
checkpoint: results/baseline/portfolio_baseline_best.npz
eval_samples: 1024
batch_size: 32
seed: 42
epsilons: [0, 2/255, 4/255, 8/255, 16/255]
```

Measured FGSM robustness:

| Epsilon | Clean accuracy | Adversarial accuracy | Accuracy drop | Attack success rate |
| ------- | -------------: | -------------------: | ------------: | ------------------: |
| `0` | `45.80%` | `45.80%` | `0.00 pp` | `0.00%` |
| `2/255` | `45.80%` | `17.38%` | `28.42 pp` | `62.05%` |
| `4/255` | `45.80%` | `7.23%` | `38.57 pp` | `84.22%` |
| `8/255` | `45.80%` | `1.17%` | `44.63 pp` | `97.44%` |
| `16/255` | `45.80%` | `0.00%` | `45.80 pp` | `100.00%` |

FGSM quantitative artifacts:

- [FGSM quantitative metrics (JSON)](results/fgsm/fgsm_quantitative_metrics.json)
- [Accuracy vs epsilon](results/fgsm/accuracy_vs_epsilon.png)
- [Attack success rate vs epsilon](results/fgsm/attack_success_rate_vs_epsilon.png)
- [Accuracy drop vs epsilon](results/fgsm/accuracy_drop_vs_epsilon.png)

### Historical WP8 smoke evaluation

[![FGSM accuracy vs epsilon](results/WP8/fgsm_accuracy_vs_epsilon.png)](results/WP8/fgsm_accuracy_vs_epsilon.png)

Controlled WP8 smoke run over epsilon values from `0/255` through `16/255`.
The pipeline produces clean accuracy, adversarial accuracy, accuracy drop, and
attack success rate metrics.

Important limitation: this historical WP8 smoke plot was generated with the old
tiny subset checkpoint, which achieved `0.0` clean accuracy on the fixed
32-sample subset. It validates the evaluation pipeline rather than proving
final CIFAR-10 robustness. The FGSM quantitative figures above are the current
controlled evaluation results using the stronger baseline checkpoint.

### FGSM qualitative analysis with reproducible baseline

[![FGSM Qualitative Analysis](results/fgsm/fgsm_qualitative_comparison.png)](results/fgsm/fgsm_qualitative_comparison.png)

This figure uses `results/baseline/portfolio_baseline_best.npz` and a
deterministic CIFAR-10 test-subset selection rule: the first clean-correct
sample that becomes incorrect under FGSM at `epsilon = 8/255`. It shows the
clean image, input-gradient map, visualized perturbation magnitude, and
adversarial image.

[![FGSM Epsilon Progression](results/fgsm/epsilon_progression.png)](results/fgsm/epsilon_progression.png)

The epsilon progression keeps the same clean source image and independently
generates FGSM examples for `0`, `2/255`, `4/255`, `8/255`, and `16/255`.

FGSM qualitative artifacts:

- [FGSM qualitative comparison](results/fgsm/fgsm_qualitative_comparison.png)
- [FGSM epsilon progression](results/fgsm/epsilon_progression.png)
- [FGSM qualitative metadata (JSON)](results/fgsm/fgsm_qualitative_metadata.json)

### Clean vs adversarial Grad-CAM

[![Clean vs Adversarial Grad-CAM](results/gradcam/gradcam_hero_presentation.png)](results/gradcam/gradcam_hero_presentation.png)

Clean vs adversarial Grad-CAM under FGSM (`epsilon = 8/255`). The examples are
clean-correct CIFAR-10 test samples where FGSM changes the model prediction.
Grad-CAM maps are independently normalized to `[0, 1]`, so the visualization
compares spatial localization patterns rather than absolute activation
magnitude.

The presentation figure uses a heatmap-weighted overlay with the `turbo`
colormap to keep low-activation regions close to the original image while
making high-activation regions easier to inspect.

Grad-CAM artifacts:

- [Presentation README hero figure](results/gradcam/gradcam_hero_presentation.png)
- [Compact README hero figure](results/gradcam/gradcam_hero.png)
- [Detailed clean vs adversarial comparison](results/gradcam/gradcam_detailed_comparison.png)
- [Fixed-original-target Grad-CAM comparison](results/gradcam/gradcam_fixed_target_comparison.png)
- [Attack success vs control comparison](results/gradcam/gradcam_success_vs_control.png)
- [Grad-CAM comparison metadata (JSON)](results/gradcam/gradcam_comparison_metadata.json)

Historical WP7 smoke qualitative artifacts remain available for traceability:

- [Clean image](results/WP7/qualitative/fgsm_example_000_clean.png)
- [FGSM adversarial image](results/WP7/qualitative/fgsm_example_000_adversarial.png)
- [Input-gradient map](results/WP7/qualitative/fgsm_example_000_input_gradient.png)
- [Perturbation map](results/WP7/qualitative/fgsm_example_000_perturbation.png)
- [Historical combined WP7 figure](results/WP7/qualitative/fgsm_example_000_combined.png)

## Why This Project Is Technically Significant

The core learning mechanics are implemented manually:

* `Conv2D`, including stride and padding support.
* `ReLU`, `MaxPool2D`, `Flatten`, and `Linear`.
* `SoftmaxCrossEntropyLoss`.
* `SGD` parameter updates.
* Explicit `grad_input`, `grad_weight`, and `grad_bias` computation.
* Full `CompactCNN.backward(grad_logits)` reverse pass.
* Loss-to-input gradients required for FGSM.

This makes the project useful for demonstrating neural-network internals,
numerical programming, debugging discipline, and robust test-driven ML
software development.

## System Pipeline

```text
CIFAR-10 input
  -> CompactCNN
  -> clean prediction
  -> loss and input-gradient computation
  -> FGSM perturbation
  -> adversarial prediction
  -> robustness metrics
  -> saved figures and JSON artifacts
```

## CompactCNN Architecture

The model expects CIFAR-10 tensors in NCHW format:

```text
Input: (N, 3, 32, 32)
```

Architecture implemented in `src/models/compact_cnn.py`:

```text
Input
-> Conv2D(3 -> 8, kernel_size=3, padding=1)
-> ReLU
-> MaxPool2D(kernel_size=2, stride=2)
-> Conv2D(8 -> 16, kernel_size=3, padding=1)
-> ReLU
-> MaxPool2D(kernel_size=2, stride=2)
-> Flatten
-> Linear(16 * 8 * 8 -> 10)
-> logits
```

The output shape is:

```text
(N, 10)
```

## Validation and Testing

Latest validation:

```text
Non-CuPy local regression: 256 passed, 19 deselected
Full local suite on this machine: 256 passed, 19 skipped
```

The tests cover:

* layer-level forward behavior,
* manual backward propagation,
* gradient shapes and finite values,
* centered finite-difference numerical gradient checks,
* full model backward integration,
* optimizer and training utilities,
* checkpoint save/load,
* JSON metrics persistence,
* plotting helpers,
* input-gradient computation,
* FGSM behavior,
* robustness evaluation and epsilon sweeps,
* FGSM quantitative runner and plots,
* FGSM qualitative runner and visualizations,
* Grad-CAM core and adversarial Grad-CAM visualization helpers,
* experiment runner smoke tests.
* final portfolio evidence generation from curated artifacts.

Run the full suite:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

Run the standard offline CI subset:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
```

For a fresh environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same test
offline test subset on push and pull requests. Tests marked `requires_data`
exercise the real CIFAR-10 data pipeline and can be run manually when the
dataset is already available locally:

```bash
.venv/bin/python -m pytest -q -m requires_data
```

### Validate staged CIFAR-10 data for cluster use

Cluster experiments should consume an explicitly staged CIFAR-10 copy instead
of downloading data inside GPU jobs. The expected staged layout is:

```text
data/raw/cifar-10-python.tar.gz
data/raw/cifar-10-batches-py/
```

The expected archive MD5 is:

```text
c58f30108f718f92721af3b95e74349a
```

The Hawaii cluster staging has been validated with archive size
`170498071` bytes, matching MD5, extracted train/test splits of
`50000`/`10000` images, and 10 class names. The validated environment was
`NVIDIA GeForce RTX 2080 Ti`, CuPy `14.1.1`, Python `3.12.13`, and NumPy
`2.4.6`.

Validate the Python/CuPy environment and staged dataset before running
data-dependent cluster tests or experiments:

```bash
python scripts/validate_cluster_environment.py --backend numpy --data-dir data/raw --json-output results/cluster_validation/cifar10_environment_numpy.json
python scripts/validate_cluster_environment.py --backend cupy --data-dir data/raw --extract-if-needed --json-output results/cluster_validation/cifar10_environment_cupy.json
python -m pytest -q -m requires_data
```

The validation utility reports environment metadata and dataset status in
human-readable form and can save JSON reports. It does not run training,
robustness sweeps, or benchmarks. Run-specific files under
`results/cluster_validation/` are ignored by Git; keep the verified summary in
documentation unless a report artifact is intentionally curated.

### Run a small reproducible FGSM experiment

The scheduler-neutral experiment runner writes machine-readable artifacts for
later robustness and runtime analysis. It does not generate plots and does not
implement scheduler submission:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend numpy --data-dir data/raw --checkpoint results/baseline/portfolio_baseline_best.npz --split test --max-samples 8 --batch-size 2 --epsilons 0,1/255 --output-root results/runs
```

On the validated Hawaii GPU environment, use `--backend cupy` for a small
smoke run:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 8 --batch-size 2 --epsilons 0,1/255 --output-root results/runs
```

Each run creates an isolated directory:

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

The runner records raw FGSM metrics per epsilon, run configuration,
environment metadata, Git commit/dirty state, and timing information. Existing
run directories are not overwritten. Large-scale sweeps and final plots should
be run only after the small smoke path has been validated.

The Hawaii cluster smoke run has been validated for:

```text
run_id: 20260811T171313482022Z_fgsm_cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
Python: 3.12.13
NumPy: 2.4.6
hostname: csg-brook01
status: COMPLETED
max_samples: 8
batch_size: 2
epsilons: [0.0, 1/255]
```

It produced the complete artifact schema above. The tiny 8-sample smoke run
validates runner integration only; it is not a robustness conclusion or a
performance benchmark. Run-specific files under `results/runs/` are ignored by
Git. Later curated benchmark summaries and plots can be committed separately
when they are intentionally prepared for final reporting.

### Curate a medium-scale FGSM run

After a completed runner execution, create small review-ready artifacts from
the saved `metrics`, `timing`, `summary`, and `environment` files:

```bash
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3c --expected-sample-count 1000 --expected-epsilons 0,1/255,2/255,4/255,8/255
```

The validated EWP3-C cluster sanity workload is:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 1000 --batch-size 32 --epsilons 0,1/255,2/255,4/255,8/255 --output-root results/runs
```

Curated outputs are written to:

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

These figures and summaries are derived from saved runner artifacts. The
1000-sample run is a medium-scale sanity check, not the final full CIFAR-10
evaluation or CPU/GPU scaling benchmark.

Validated curated evidence:

```text
results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/
```

The run used the CuPy backend on an NVIDIA GeForce RTX 2080 Ti with CuPy
`14.1.1`, NumPy `2.4.6`, Python `3.12.13`, batch size `32`, seed `42`, and
1000 CIFAR-10 test samples. CIFAR-10 checksum validation passed and the run
completed successfully.

| Epsilon | Clean Accuracy | Adversarial Accuracy | Accuracy Drop | Attack Success Rate |
| ------- | -------------- | -------------------- | ------------- | ------------------- |
| 0 | 0.481 | 0.481 | 0.000 | 0.000 |
| 1/255 | 0.481 | 0.317 | 0.164 | 0.341 |
| 2/255 | 0.481 | 0.202 | 0.279 | 0.580 |
| 4/255 | 0.481 | 0.090 | 0.391 | 0.813 |
| 8/255 | 0.481 | 0.009 | 0.472 | 0.981 |

Timing summary:

```text
evaluation_wall_seconds: 10.670563029998448
total_wall_seconds: 11.60283667499607
sample_epsilon_pairs: 5000
evaluation_sample_epsilon_pairs_per_second: 468.5788356193916
```

Curated plot artifacts:

* [Accuracy vs epsilon](results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/accuracy_vs_epsilon.png)
* [Attack success rate vs epsilon](results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/attack_success_rate_vs_epsilon.png)
* [Accuracy drop vs epsilon](results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/accuracy_drop_vs_epsilon.png)
* [Runtime and throughput summary](results/curated/ewp3c/20260811T173256700165Z_fgsm_cupy/runtime_throughput_summary.png)

### Run a CPU/GPU FGSM benchmark

The benchmark driver launches the existing FGSM runner for each workload. It
does not implement a second attack or evaluation path:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --sample-counts 100,250,500,1000,2000 --sample-scaling-backends numpy,cupy --sample-scaling-batch-size 32 --batch-sizes 8,16,32,64,128 --batch-scaling-backend cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

The default benchmark matrix is:

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

Matched batch-size extension:
  backends: numpy, cupy
  sample_count: 1000
  batch_sizes: 8, 16, 32, 64, 128
  epsilons: 0, 4/255

measured repeats: 3
excluded warm-up runs: 1 per workload
```

Benchmark aggregate artifacts are written to:

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

Raw per-repeat runner outputs stay under ignored `results/runs/<run_id>/`.
Evaluation speedup is defined as `CPU evaluation_wall_seconds / GPU
evaluation_wall_seconds` for matched workloads. The benchmark records
`evaluation_wall_seconds` and `total_wall_seconds` separately and uses the
runner's synchronized CuPy timing path.

To run only the matched batch-size extension after the sample-count benchmark
has already been completed:

```bash
python -m experiments.fgsm.run_fgsm_benchmark --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --skip-sample-scaling --batch-sizes 8,16,32,64,128 --batch-scaling-backends numpy,cupy --batch-scaling-sample-count 1000 --epsilons 0,4/255 --repeats 3 --warmup-runs 1 --raw-run-output-root results/runs --benchmark-output-root results/benchmarks
```

`crossover_analysis.json` records the first tested batch size where GPU
evaluation speedup exceeds `1.0`, plus the maximum tested speedup and its
batch size.

Validated RTX 2080 Ti benchmark evidence:

```text
benchmark_id: 20260811T185420645969Z_fgsm_benchmark
backend pair: numpy / cupy
sample_count: 1000
epsilons: 0, 4/255
repeats: 3
warmup_runs: 1
```

On the validated RTX 2080 Ti benchmark, CuPy was slower than NumPy at small
batches, crossed the CPU/GPU break-even at the first tested batch size of
`64`, and reached a median `2.88x` evaluation-wall-time speedup at batch size
`128`. Batch size `128` is the best tested batch size in this benchmark, not a
global optimum claim.

| Batch Size | Median CPU/GPU Evaluation Speedup | NumPy Mean Throughput | CuPy Mean Throughput |
| ---------- | --------------------------------- | --------------------- | -------------------- |
| 8 | 0.252x | ~491.67 pairs/s | ~123.69 pairs/s |
| 16 | 0.420x | ~577.79 pairs/s | ~242.77 pairs/s |
| 32 | 0.761x | ~622.71 pairs/s | ~474.55 pairs/s |
| 64 | 1.467x | ~643.74 pairs/s | ~945.14 pairs/s |
| 128 | 2.882x | ~644.50 pairs/s | ~1855.80 pairs/s |

The earlier sample-count benchmark at matched batch size `32` did not show a
CPU/GPU crossover for sample counts `100`, `250`, `500`, `1000`, or `2000`.
Together, these measurements indicate that batch size and GPU utilization are
the key crossover factors for the current implementation. No custom CUDA
kernel or GPU-specific Conv2D optimization has been applied.

Curated EWP3-D evidence is tracked under:

```text
results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/
```

Key plots:

* [Runtime vs batch size](results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/runtime_vs_batch_size.png)
* [Throughput vs batch size](results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/throughput_vs_batch_size.png)
* [Speedup vs batch size](results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/speedup_vs_batch_size.png)

### Run the full CIFAR-10 FGSM evaluation

EWP3-E uses the validated runner and curation pipeline for the final full
CIFAR-10 test-set FGSM robustness run. This is FGSM robustness evidence, not
PGD robustness evidence and not a new performance benchmark.

Validated Hawaii cluster workload:

```bash
python -m experiments.fgsm.run_fgsm_experiment --backend cupy --data-dir data/raw --checkpoint results/checkpoints/portfolio_baseline_best.npz --split test --max-samples 10000 --batch-size 128 --epsilons 0,1/255,2/255,4/255,8/255,12/255,16/255 --seed 42 --output-root results/runs
```

Batch size `128` is used because EWP3-D found it to be the best tested batch
size in the current benchmark range. It is not a global optimum claim.

The validated run was:

```text
run_id: 20260812T115232600695Z_fgsm_cupy
backend: cupy
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
sample_count: 10000
batch_size: 128
seed: 42
CIFAR-10 checksum: PASS
Git commit at runtime: b5a755b457c4299d9dd1a7c77d195f6fc3d74bc4
status: COMPLETED
```

Curate the saved artifacts with:

```bash
python -m experiments.fgsm.plot_fgsm_run --run-dir results/runs/<run_id> --output-root results/curated/ewp3e --expected-sample-count 10000 --expected-epsilons 0,1/255,2/255,4/255,8/255,12/255,16/255 --expected-backend cupy --expected-gpu-name "NVIDIA GeForce RTX 2080 Ti" --interpretation "Full CIFAR-10 test-set FGSM robustness evaluation; final EWP3-E robustness evidence, not a performance benchmark."
```

The curation step validates run completion, epsilon ordering, sample count,
bounded/finite robustness metrics, epsilon-zero consistency, dataset checksum
metadata, expected CuPy backend metadata, expected RTX 2080 Ti metadata, and
positive timing values.

Curated outputs are tracked under:

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
| 0 | 0.4639 | 0.4639 | 0.0000 | 0.000 |
| 1/255 | 0.4639 | 0.3020 | 0.1619 | 0.349 |
| 2/255 | 0.4639 | 0.1854 | 0.2785 | 0.600 |
| 4/255 | 0.4639 | 0.0743 | 0.3896 | 0.840 |
| 8/255 | 0.4639 | 0.0099 | 0.4540 | 0.979 |
| 12/255 | 0.4639 | 0.0017 | 0.4622 | 0.996 |
| 16/255 | 0.4639 | 0.0004 | 0.4635 | 0.999 |

Timing summary:

```text
sample_epsilon_pairs: 70000
evaluation_wall_seconds: 37.00973283001804
total_wall_seconds: 37.96863090901752
evaluation_sample_epsilon_pairs_per_second: 1891.3943616265192
timing_method: time.perf_counter
gpu_synchronization: CuPy Stream.null synchronized before and after evaluation
```

These timing values document the execution of this robustness experiment. Use
EWP3-D for CPU/GPU speedup claims.

Curated plot artifacts:

* [Accuracy vs epsilon](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/accuracy_vs_epsilon.png)
* [Attack success rate vs epsilon](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/attack_success_rate_vs_epsilon.png)
* [Accuracy drop vs epsilon](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/accuracy_drop_vs_epsilon.png)
* [Runtime and throughput summary](results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/runtime_throughput_summary.png)

The full 10k run follows the same broad robustness trend as the earlier
1000-sample EWP3-C sanity run: adversarial accuracy decreases sharply as
epsilon increases, and attack success rate approaches `1.0` at larger
epsilons. The two runs should not be combined into one statistical estimate.
Raw run artifacts remain under ignored `results/runs/<run_id>/`.

## Numerical Gradient Checking

Manual backpropagation is validated against numerical finite differences.
The repository includes checks for:

* `Linear` input, weight, and bias gradients,
* `Conv2D` input, weight, and bias gradients,
* `SoftmaxCrossEntropyLoss.backward()` with respect to logits,
* `CompactCNN` loss-to-input gradient sanity.

This is important because analytical gradients are implemented manually rather
than delegated to an autodiff framework.

## Runtime Profiling and Optimization

WP6 followed a focused performance-engineering workflow:

1. profile the existing NumPy pipeline,
2. identify one measured bottleneck,
3. optimize only that bottleneck,
4. rerun correctness tests and local benchmarks.

`Conv2D.backward` was selected as the bottleneck. The optimization remained
local to that method and uses `np.einsum`-based accumulation while preserving
the public API, forward behavior, stride, padding, and gradient shapes.

[![Conv2D.backward runtime comparison](results/WP6/conv2d_backward_runtime_comparison.png)](results/WP6/conv2d_backward_runtime_comparison.png)

The runtime figure summarizes the engineering workflow:
profile → identify `Conv2D.backward` as the bottleneck → optimize → benchmark.

Documented local benchmark:

| Operation | Before (s) | After (s) | Comparison |
| --------- | ---------: | --------: | ---------: |
| `Conv2D.forward` | `0.000068569` | `0.000066375` | `1.03x` |
| `Conv2D.backward` | `0.043458736` | `0.000209222` | `207.72x` |
| `train_step` | `0.070350708` | `0.001886028` | `37.30x` |

These are controlled local measurements on synthetic data, not broad hardware
benchmarks.

## FGSM Robustness Pipeline

The implemented FGSM pipeline:

1. computes clean logits and predictions,
2. computes loss gradients with respect to input images,
3. generates adversarial examples using:

   ```python
   np.clip(images + epsilon * np.sign(grad_input), 0.0, 1.0)
   ```

4. evaluates adversarial predictions,
5. aggregates clean accuracy, adversarial accuracy, accuracy drop, and attack
   success rate,
6. runs epsilon sweeps,
7. saves metrics and plots.

The controlled WP8 run used:

```text
eval_samples: 32
batch_size: 8
seed: 42
epsilon_values: [0/255, 1/255, 2/255, ..., 16/255]
```

### Experiment artifacts

- [FGSM robustness metrics (JSON)](results/WP8/fgsm_robustness_metrics.json)
- [FGSM accuracy vs. epsilon plot (PNG)](results/WP8/fgsm_accuracy_vs_epsilon.png)
- [WP8 controlled smoke review](deliverables/WP8/wp8_smoke_review.md)

Current result caveat: all clean and adversarial accuracies are `0.0` on the
fixed 32-sample subset because the checkpoint is weak. Treat WP8 as controlled
pipeline validation, not as a final scientific robustness conclusion.

## Reproducibility / How to Run

### Install dependencies

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### Run tests

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
```

### Run the synthetic baseline smoke test

This does not load CIFAR-10:

```bash
.venv/bin/python -c 'from experiments.baseline.train_baseline import run_synthetic_baseline; result = run_synthetic_baseline(); print(result["final_metrics"])'
```

### Train the reproducible CIFAR-10 baseline

Requires existing local CIFAR-10 data:

```bash
MPLCONFIGDIR=/tmp/cnn-baseline-matplotlib .venv/bin/python -m experiments.baseline.train_portfolio_baseline
```

Default outputs:

- Local baseline checkpoint: `results/baseline/portfolio_baseline_best.npz`
  (not tracked by Git)
- [Training history JSON](results/baseline/portfolio_training_history.json)
- [Final metrics JSON](results/baseline/portfolio_final_metrics.json)
- [Training loss curve](results/baseline/training_loss_curve.png)
- [Train vs validation accuracy curve](results/baseline/train_validation_accuracy_curve.png)
- [Confusion matrix](results/baseline/confusion_matrix.png)

### Generate qualitative FGSM examples

Requires an existing local checkpoint and extracted CIFAR-10 test batch:

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m experiments.fgsm.generate_examples
```

Default outputs:

- [Clean image](results/WP7/qualitative/fgsm_example_000_clean.png)
- [FGSM adversarial image](results/WP7/qualitative/fgsm_example_000_adversarial.png)
- [Input-gradient map](results/WP7/qualitative/fgsm_example_000_input_gradient.png)
- [Perturbation map](results/WP7/qualitative/fgsm_example_000_perturbation.png)

### Run controlled FGSM robustness evaluation

Requires an existing local checkpoint and extracted CIFAR-10 data:

```bash
MPLCONFIGDIR=/tmp/cnn-wp8-matplotlib .venv/bin/python -m experiments.fgsm.evaluate_robustness
```

Default outputs:

- [FGSM robustness metrics (JSON)](results/WP8/fgsm_robustness_metrics.json)
- [FGSM accuracy vs. epsilon plot (PNG)](results/WP8/fgsm_accuracy_vs_epsilon.png)

The data loader checks for local CIFAR-10 data and does not silently fabricate
large-scale benchmark results.

### Run FGSM quantitative evaluation

Uses `results/baseline/portfolio_baseline_best.npz` and writes to
`results/fgsm/`:

```bash
MPLCONFIGDIR=/tmp/cnn-fgsm-matplotlib .venv/bin/python -m experiments.fgsm.evaluate_quantitative
```

Default outputs:

- [FGSM quantitative metrics (JSON)](results/fgsm/fgsm_quantitative_metrics.json)
- [Accuracy vs epsilon](results/fgsm/accuracy_vs_epsilon.png)
- [Attack success rate vs epsilon](results/fgsm/attack_success_rate_vs_epsilon.png)
- [Accuracy drop vs epsilon](results/fgsm/accuracy_drop_vs_epsilon.png)

### Generate FGSM qualitative visualizations

Uses `results/baseline/portfolio_baseline_best.npz`, the deterministic
test-subset policy, and writes to `results/fgsm/`:

```bash
MPLCONFIGDIR=/tmp/cnn-fgsm-qualitative-matplotlib .venv/bin/python -m experiments.fgsm.generate_day3_visualizations
```

Default outputs:

- [FGSM qualitative comparison](results/fgsm/fgsm_qualitative_comparison.png)
- [FGSM epsilon progression](results/fgsm/epsilon_progression.png)
- [FGSM qualitative metadata (JSON)](results/fgsm/fgsm_qualitative_metadata.json)

### Generate clean vs adversarial Grad-CAM comparisons

Uses `results/baseline/portfolio_baseline_best.npz`, deterministic CIFAR-10
test-sample selection, and writes to `results/gradcam/`:

```bash
MPLCONFIGDIR=/tmp/cnn-gradcam-matplotlib .venv/bin/python -m experiments.gradcam.generate_adversarial_comparisons
```

Default outputs:

- [Presentation README hero figure](results/gradcam/gradcam_hero_presentation.png)
- [Compact README hero figure](results/gradcam/gradcam_hero.png)
- [Detailed clean vs adversarial comparison](results/gradcam/gradcam_detailed_comparison.png)
- [Fixed-original-target Grad-CAM comparison](results/gradcam/gradcam_fixed_target_comparison.png)
- [Attack success vs control comparison](results/gradcam/gradcam_success_vs_control.png)
- [Grad-CAM comparison metadata (JSON)](results/gradcam/gradcam_comparison_metadata.json)

## Repository Structure

```text
configs/       Project paths, image shape constants, baseline config
src/           NumPy model, layers, losses, optimizer, attacks, metrics
tests/         Unit, integration, numerical-gradient, and runner tests
experiments/   Baseline, runtime profiling, and FGSM experiment entry points
results/       Committed figures and controlled smoke-result artifacts
deliverables/  Work-package summaries, audits, and manual review notes
```

## Engineering Workflow

The project was developed incrementally through work packages. The workflow
emphasized:

* small validated implementation steps,
* deterministic tests,
* manual review notes for forward and backward computations,
* numerical gradient verification,
* profiling before optimization,
* reproducible metrics and figures,
* explicit scope boundaries between completed work and planned work.

## Current Status and Next Steps

### Completed

* CIFAR-10 data pipeline.
* From-scratch CompactCNN forward pass.
* Manual layer and model backward propagation.
* Softmax Cross-Entropy forward/backward.
* Numerical gradient checks.
* Training, evaluation, checkpointing, metrics, and plotting utilities.
* Controlled baseline runners.
* Reproducible clean CIFAR-10 baseline checkpoint and training/evaluation figures.
* Runtime bottleneck profiling and `Conv2D.backward` optimization.
* Input-gradient computation.
* FGSM attack and qualitative visualizations.
* Controlled FGSM robustness evaluation with epsilon sweep.
* FGSM quantitative robustness figures.
* FGSM qualitative comparison and epsilon progression figures.
* Full CIFAR-10 test-set FGSM robustness evaluation on the validated RTX 2080 Ti / CuPy environment.
* CPU/GPU FGSM scaling benchmark and curated performance plots.
* Clean Grad-CAM core for the final `relu2` activation.
* Clean vs adversarial Grad-CAM qualitative comparison figures.
* GitHub Actions CI workflow for automated test validation.

### Planned

* PGD and additional attack evaluation if time and runtime allow.
* Optional adversarial training.
* Final reproducibility packaging.

PGD, black-box attacks, and adversarial training are planned work; they are not
implemented in the current repository state.

## Scope and Limitations

* The core model and learning mechanics are NumPy-based; plotting uses
  Matplotlib and tests use Pytest.
* Historical WP8 CIFAR-10 robustness artifacts are controlled subset or smoke
  validations; use the EWP3-E curated run for full-test-set FGSM evidence.
* The full-test-set robustness result is FGSM-only. It is not PGD robustness,
  black-box robustness, or adversarial training evidence.
* Full CIFAR-10 multi-epoch training remains future work.
