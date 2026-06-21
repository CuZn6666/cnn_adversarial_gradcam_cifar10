# WP5 Synthetic Baseline Smoke Run

## Purpose

This run validates that the existing NumPy WP5 components work together and
produce the expected artifacts. It uses deterministic synthetic arrays only.
It is not a real or full CIFAR-10 baseline run.

## Command

Run from the repository root:

```bash
.venv/bin/python -c 'from experiments.baseline.train_baseline import run_synthetic_baseline; result = run_synthetic_baseline(); print(result["final_metrics"]); print(result["checkpoint_path"]); print(result["metrics_path"]); print(result["loss_curve_path"]); print(result["accuracy_curve_path"])'
```

## Configuration

```text
seed: 42
learning_rate: 0.0005
batch_size: 8
epochs: 1
train_samples: 64
eval_samples: 32
```

## Final Metrics

```text
train_loss: 2.3956282436847687
train_accuracy: 0.09375
eval_loss: 2.396390974521637
eval_accuracy: 0.15625
```

## Generated Artifacts

```text
results/checkpoints/synthetic_baseline.npz
results/logs/synthetic_metrics.json
results/figures/loss_curve.png
results/figures/accuracy_curve.png
```

The checkpoint is ignored by the repository's `*.npz` rule. Generated
smoke-run files are local artifacts and should not be force-added. The command
above regenerates all four files.

## Scope and Remaining Work

This smoke run confirms:

* baseline component orchestration,
* deterministic synthetic training and clean evaluation,
* checkpoint creation,
* JSON metrics persistence,
* loss and accuracy plot generation.

This run does not:

* load or train on CIFAR-10,
* establish real clean CIFAR-10 accuracy,
* complete the full CIFAR-10 baseline,
* use CuPy, CUDA, GPU, cluster, or SLURM support,
* start WP6, adversarial attacks, or Grad-CAM.

WP5 remains in progress. The next controlled follow-up is a small real
CIFAR-10 subset baseline. A full CIFAR-10 baseline should only run after that
subset workflow is validated.
