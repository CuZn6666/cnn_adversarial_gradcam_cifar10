# WP6 Initial Runtime Profile

## Purpose

Record inspection-only runtime measurements before selecting or implementing
any optimization.

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

## Measured Operations

1. `Conv2D.forward`
2. `Conv2D.backward`
3. one complete `train_step`

## Runtime Results

```text
Conv2D.forward: 0.000068569 seconds per iteration
Conv2D.backward: 0.043458736 seconds per iteration
train_step: 0.070350708 seconds per iteration
```

These are local measurements and may vary across machines. Later before/after
measurements must use the same setup and command.

## Initial Observation

The isolated `Conv2D.backward` measurement is substantially slower than the
isolated `Conv2D.forward` measurement and accounts for a large part of the
complete `train_step` runtime. The `train_step` result is an aggregate that
also includes the complete model forward pass, loss computation, all backward
operations, and the optimizer update.

No optimization path has been selected yet. No optimization was implemented
in this profiling step.
