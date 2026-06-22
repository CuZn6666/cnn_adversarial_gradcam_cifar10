# WP8 Final Summary

## Status

Completed for the controlled FGSM pipeline-validation scope.

## Goal

Evaluate clean and FGSM predictions across selected epsilon values, aggregate
robustness metrics, plot accuracy against epsilon, and select representative
successful and failed attacks.

## Implemented Components

* `evaluate_fgsm_batch(...)` computes clean and adversarial counts and metrics
  for one batch.
* `evaluate_fgsm_batches(...)` aggregates raw counts across differently sized
  batches and recomputes sample-weighted metrics.
* `evaluate_fgsm_epsilon_sweep(...)` evaluates epsilons in the provided order.
* `plot_fgsm_accuracy_vs_epsilon(...)` saves clean and adversarial accuracy
  curves.
* `select_fgsm_representative_examples(...)` deterministically selects the
  first eligible successful and failed attack metadata.
* `WP8FGSMRobustnessConfig` defines the controlled local settings.
* `run_fgsm_robustness_pipeline(...)` composes evaluation, persistence,
  plotting, and representative selection.
* `run_cifar10_fgsm_robustness(...)` loads an existing local checkpoint and
  local CIFAR-10 data for the controlled runner.

## Controlled Smoke Configuration

```text
eval_samples: 32
batch_size: 8
seed: 42
epsilon_values: [0, 2/255, 4/255, 8/255, 16/255]
representative_epsilon: 8/255
```

The smoke run used existing local CIFAR-10 data and the existing controlled
checkpoint. It did not download data, train the model, or modify the
checkpoint.

## Committed Artifacts

```text
results/WP8/fgsm_robustness_metrics.json
results/WP8/fgsm_accuracy_vs_epsilon.png
deliverables/WP8/wp8_smoke_review.md
```

## Smoke Results

For all evaluated epsilon values:

```text
clean_accuracy: 0.0
adversarial_accuracy: 0.0
accuracy_drop: 0.0
attack_success_rate: 0.0
```

Representative metadata:

```text
successful: []
failed: []
```

The fixed subset contained no clean-correct samples for the current
checkpoint. Therefore attack success rate is defined as `0.0`, and no samples
were eligible for successful or failed representative selection.

## Interpretation

The WP8 evaluation pipeline runs end to end and produces deterministic metrics
and plotting artifacts.

The checkpoint is too weak on this 32-sample subset to support a meaningful
CIFAR-10 robustness conclusion. WP8 therefore closes as controlled pipeline
validation, not as a final model-robustness claim.

## Validation

```bash
MPLCONFIGDIR=/tmp/cnn-wp8-matplotlib .venv/bin/python -m pytest tests/test_fgsm_evaluation.py tests/test_fgsm_robustness_runner.py tests/test_plotting.py -v
MPLCONFIGDIR=/tmp/cnn-wp8-matplotlib .venv/bin/python -m pytest tests/ -q
```

Latest results:

```text
Focused WP8 tests: 27 passed
Full suite: 159 passed
```

The automated tests use synthetic data or temporary output directories and do
not require CIFAR-10 downloads or real checkpoint loading.

## ZITI Decision

ZITI is not needed for the completed WP8 smoke validation or closeout. No ZITI,
GPU, CUDA, CuPy, JAX, PyTorch, Slurm, or cluster workflow was used.

A larger formal evaluation is deferred until a stronger baseline checkpoint
is available and the user explicitly confirms whether to use ZITI.

## Explicit Non-Goals

WP8 did not include:

* a larger or repeated-seed evaluation,
* a denser epsilon sweep,
* model retraining,
* PGD,
* black-box attacks,
* Grad-CAM,
* any later Work Package content.
