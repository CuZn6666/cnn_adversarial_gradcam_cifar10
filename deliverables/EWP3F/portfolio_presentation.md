# EWP3-F Portfolio Presentation Notes

## Source-of-Truth Evidence Map

All final quantitative claims in this document are derived from tracked curated
artifacts:

| Claim area | Source artifact |
| ---------- | --------------- |
| CPU/GPU throughput by batch size | `results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/benchmark_summary.csv` |
| CPU/GPU speedup by batch size | `results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/speedup_summary.csv` |
| GPU crossover and maximum tested speedup | `results/curated/ewp3d/20260811T185420645969Z_fgsm_benchmark/crossover_analysis.json` |
| Full CIFAR-10 FGSM robustness metrics | `results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/robustness_summary.csv` |
| Full-run timing and throughput | `results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/timing_summary.json` |
| Full-run environment metadata | `results/curated/ewp3e/20260812T115232600695Z_fgsm_cupy/run_metadata.json` |
| Final summary table and figures | `results/curated/portfolio/` |

## Final Engineering Narrative

This project implements a compact CIFAR-10 CNN largely from scratch with
NumPy-style tensor operations. The model includes manual forward and backward
passes for convolution, ReLU, max pooling, flattening, linear classification,
softmax cross entropy, SGD updates, input-gradient computation, and FGSM
adversarial example generation.

NumPy remains the default and authoritative correctness reference. A minimal
backend abstraction allows the compute-heavy tensor path to run on either
NumPy or CuPy while keeping data loading, plotting, JSON serialization, and
public artifacts on the CPU side.

Correctness was preserved through staged numerical-equivalence validation on a
real RTX 2080 Ti GPU. The validated scope includes backend primitives,
Conv2D, ReLU, MaxPool2D, Flatten, Linear, softmax cross entropy, the full
CompactCNN training path, SGD, input gradients, FGSM perturbations,
adversarial logits/predictions, and FGSM robustness metrics.

The cluster workflow uses a reproducible runner that records configuration,
environment metadata, dataset checksum status, Git commit, metrics, timing,
and run status. Raw per-run artifacts stay under ignored `results/runs/`.
Curated summaries and figures are intentionally tracked when they support
final engineering claims.

## Quantitative Results

### Performance

Validated environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CuPy: 14.1.1
NumPy: 2.4.6
Python: 3.12.13
```

EWP3-D measured matched CPU/GPU FGSM evaluation workloads at batch sizes
`8`, `16`, `32`, `64`, and `128`, with `1000` samples and epsilons `{0,
4/255}`. Speedup is defined as:

```text
CPU evaluation_wall_seconds / GPU evaluation_wall_seconds
```

Validated findings:

* First tested GPU-faster batch size: `64`.
* Median evaluation speedup at batch `64`: `1.4674x`.
* Best tested median evaluation speedup: `2.8822x` at batch `128`.
* CuPy throughput at batch `128`: approximately `1855.8` sample-epsilon
  pairs/s.
* NumPy throughput approaches approximately `644` sample-epsilon pairs/s at
  larger tested batch sizes.

Batch size `128` is the best tested configuration in this benchmark range,
not a global optimum claim.

### Robustness

EWP3-E evaluated the full CIFAR-10 test set with `10000` samples, CuPy backend,
batch size `128`, seed `42`, and epsilons `{0, 1/255, 2/255, 4/255, 8/255,
12/255, 16/255}`.

Validated findings:

| Epsilon | Adversarial Accuracy | Attack Success Rate |
| ------- | -------------------: | ------------------: |
| `0` | `46.39%` | `0.00%` |
| `1/255` | `30.20%` | `34.90%` |
| `2/255` | `18.54%` | `60.03%` |
| `4/255` | `7.43%` | `83.98%` |
| `8/255` | `0.99%` | `97.87%` |
| `12/255` | `0.17%` | `99.63%` |
| `16/255` | `0.04%` | `99.91%` |

Clean accuracy at epsilon `0` is `46.39%`. The full run processed `70000`
sample-epsilon pairs in `37.01` evaluation-wall seconds, or approximately
`1891.39` sample-epsilon pairs/s. This is FGSM evidence only, not PGD
robustness evidence.

## Resume Bullet Variants

### Systems-focused

Built a NumPy-first/CuPy-compatible CNN evaluation stack from scratch, validated
CPU/GPU numerical equivalence on an RTX 2080 Ti, and produced reproducible
FGSM benchmarks showing a batch-size-dependent GPU crossover with a best
tested `2.88x` matched evaluation speedup at batch `128`.

### ML/Robustness-focused

Implemented a from-scratch CIFAR-10 CompactCNN with manual backpropagation,
input-gradient computation, FGSM adversarial examples, and full-test-set
robustness evaluation; measured adversarial accuracy falling from `46.39%`
clean to `0.04%` at FGSM epsilon `16/255`.

### Balanced SDE/ML

Engineered and validated a reproducible NumPy/CuPy adversarial-robustness
pipeline for a from-scratch CIFAR-10 CNN, including numerical equivalence
tests, cluster artifact capture, CPU/GPU scaling analysis, and a full
`10000`-sample FGSM evaluation.

## Interview Narrative

### What was the project?

The project was a compact CIFAR-10 CNN implemented largely from scratch with
NumPy-style array operations. It includes manual forward and backward
propagation, SGD, input gradients, FGSM adversarial attacks, robustness
metrics, Grad-CAM analysis, and a reproducible experiment runner.

### What was difficult?

The main difficulty was preserving correctness while moving from a pure NumPy
reference implementation to optional CuPy execution. The tensor path includes
operations such as stride-based convolution windows, `einsum`, `add.at`,
input-gradient propagation, and robustness aggregation. Each needed to behave
the same under NumPy and CuPy.

### Why was the GPU initially slower?

The first matched benchmark at batch size `32` showed CuPy slower than NumPy.
The model and attack are relatively small, and small batches did not provide
enough parallel work to offset GPU launch overhead and memory movement. The
benchmark showed this was an utilization issue, not simply a sample-count
issue.

### How did you diagnose it?

The benchmark runner executed matched CPU/GPU workloads with identical
checkpoint, dataset split, sample count, epsilon workload, seed, and repeated
measurements. It recorded raw timing, throughput, and speedup artifacts. A
batch-size sweep showed NumPy throughput plateauing while CuPy throughput
continued scaling strongly.

### What changed at batch size 64/128?

At batch size `64`, the GPU became faster for the first tested time, with
median evaluation speedup `1.4674x`. At batch size `128`, CuPy reached the
best tested median speedup of `2.8822x` and approximately `1855.8`
sample-epsilon pairs/s. This is the best tested configuration, not a proof of
a global optimum.

### How was correctness preserved?

NumPy stayed the authoritative reference. CuPy support was introduced through a
small backend abstraction and validated in stages: primitives, layers,
losses, full model training step, SGD update, input gradients, FGSM
perturbations, adversarial predictions, and robustness metrics. Optional CuPy
tests skip on machines without CUDA, while the real GPU validation records the
tested RTX 2080 Ti environment.

### What would you do next?

The next engineering step would be targeted performance analysis of the
remaining GPU bottlenecks before any kernel-level optimization. If new attacks
are added later, PGD should remain a separate work package with its own
correctness and runtime validation rather than being folded into the FGSM
pipeline.
