# WP6 Final Summary

## Goal

Identify the main runtime bottleneck, select exactly one optimization target,
make one focused improvement, and validate runtime and numerical correctness
without broad framework comparison.

## Selected Bottleneck

`Conv2D.backward` was selected from the fixed Prompt 2 measurements:

```text
Conv2D.forward: 0.000068569 seconds per iteration
Conv2D.backward: 0.043458736 seconds per iteration
train_step: 0.070350708 seconds per iteration
```

The isolated backward operation was substantially slower than the isolated
forward operation and accounted for a large part of `train_step`.

## Optimization Summary

The optimization remained local to `Conv2D.backward`:

* `grad_weight` uses cached padded-input windows with `np.einsum`.
* `grad_input` accumulation uses `np.einsum` and loops only over kernel
  positions.
* The public API, forward behavior, stride, padding, and gradient shapes remain
  unchanged.

No other layer or training component was optimized.

## Before/After Runtime

| Operation | Before (s) | After (s) | Comparison |
| --- | ---: | ---: | ---: |
| `Conv2D.forward` | 0.000068569 | 0.000066375 | 1.03x |
| `Conv2D.backward` | 0.043458736 | 0.000209222 | 207.72x |
| `train_step` | 0.070350708 | 0.001886028 | 37.30x |

`Conv2D.backward` runtime decreased by 99.52%, and `train_step` runtime
decreased by 97.32%. `Conv2D.forward` was not optimized; its small difference
is treated as local timing noise.

## Correctness Validation

```text
Conv2D backward tests: 5 passed
Conv2D numerical and input-gradient tests: 2 passed
CompactCNN backward tests: 9 passed
loss-to-model integration tests: 3 passed
git diff --check: passed
```

## Limitations

* The benchmark uses synthetic `float32` data with shape `(2, 3, 32, 32)`.
* It uses one warm-up and three measured iterations.
* Results are local measurements and may vary by machine.
* The benchmark is for focused before/after comparison, not general framework
  benchmarking.
* Full CIFAR-10 multi-epoch training was not run.

## Explicit Non-Goals

WP6 did not add or compare:

* adversarial attacks,
* Grad-CAM,
* GPU, CuPy, CUDA, JAX, or PyTorch,
* cluster or SLURM support,
* optimizations outside `Conv2D.backward`.

## Final Status

WP6 is completed. WP7 has not started.
