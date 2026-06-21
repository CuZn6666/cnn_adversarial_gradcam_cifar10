# WP6 Bottleneck Decision

## Purpose

Select exactly one runtime bottleneck for later WP6 optimization based on the
inspection-only profiling completed in Prompt 2.

## Profiling Reference

The decision uses `deliverables/WP6/runtime_profile_initial.md` and its fixed
synthetic setup:

```text
seed: 42
input_shape: (2, 3, 32, 32)
warmup_count: 1
iteration_count: 3
```

## Profiling Results Used

```text
Conv2D.forward: 0.000068569 seconds per iteration
Conv2D.backward: 0.043458736 seconds per iteration
train_step: 0.070350708 seconds per iteration
```

## Selected Bottleneck

`Conv2D.backward` is the single WP6 optimization target.

## Rationale

The isolated `Conv2D.backward` runtime is approximately 634 times the isolated
`Conv2D.forward` runtime. It is also approximately 61.8% of the measured
complete `train_step` runtime, even though `train_step` additionally includes
the full model forward pass, loss computation, all other backward operations,
and the optimizer update.

The current `Conv2D.backward` implementation therefore provides the clearest
focused target supported by the initial measurements.

## Operations Not Selected

* `Conv2D.forward` is not selected because its isolated runtime is much smaller
  in the fixed profiling setup.
* `train_step` is not selected because it is an aggregate pipeline rather than
  one focused computational operation.
* No other layer or framework is selected because Prompt 2 did not profile
  them as independent optimization targets.

The concrete optimization technique will be selected and implemented later in
Prompt 4. No optimization was implemented in Prompt 3.
