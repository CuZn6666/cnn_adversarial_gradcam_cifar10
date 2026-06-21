# WP6 Runtime Benchmark After Optimization

## Purpose

Compare the Prompt 2 baseline measurements with the current optimized
`Conv2D.backward` implementation and record the required correctness checks.

## Before Baseline Source

The before values come from
`deliverables/WP6/runtime_profile_initial.md`. The old implementation was not
restored or rerun.

## Fixed Profiling Setup

```text
data: synthetic float32 arrays only
seed: 42
input_shape: (2, 3, 32, 32)
labels: [0, 1]
Conv2D: 3 input channels, 8 output channels, 3x3 kernel, padding 1, stride 1
learning_rate: 0.0005
warmup_count: 1
iteration_count: 3
timer: time.perf_counter
reported value: average seconds per measured iteration
```

Command:

```bash
.venv/bin/python -m experiments.runtime.profile_wp6
```

## Before Results

```text
Conv2D.forward: 0.000068569 seconds per iteration
Conv2D.backward: 0.043458736 seconds per iteration
train_step: 0.070350708 seconds per iteration
```

## After Results

```text
Conv2D.forward: 0.000066375 seconds per iteration
Conv2D.backward: 0.000209222 seconds per iteration
train_step: 0.001886028 seconds per iteration
```

## Before/After Comparison

```text
Conv2D.forward: 1.03x faster
Conv2D.backward: 207.72x faster, 99.52% runtime reduction
train_step: 37.30x faster, 97.32% runtime reduction
```

## Interpretation

The focused `Conv2D.backward` NumPy optimization substantially reduced the
selected bottleneck and also reduced the complete `train_step` runtime.
`Conv2D.forward` was not optimized; its small measured difference is treated as
normal local timing variation.

These measurements use a small iteration count and are intended for focused
local comparison, not broad benchmarking.

## Correctness Tests

```text
Conv2D backward tests: 5 passed
Conv2D numerical and input-gradient tests: 2 passed
CompactCNN backward tests: 9 passed
loss-to-model integration tests: 3 passed
git diff --check: passed
```

No further optimization was implemented in Prompt 5. With the benchmark and
correctness checks complete, WP6 is ready for final documentation
synchronization in Prompt 6.
