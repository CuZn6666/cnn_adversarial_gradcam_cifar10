# WP5 Controlled CIFAR-10 Subset Baseline

## Scope

This deliverable records the controlled real-data validation of the WP5 NumPy
baseline pipeline. It uses the existing CIFAR-10 loader with a deterministic
64-sample training subset and 32-sample evaluation subset.

This is distinct from:

* the synthetic smoke run, which validates orchestration without CIFAR-10,
* a full CIFAR-10 multi-epoch baseline, which was not run.

## Command

Run from the repository root with an existing extracted CIFAR-10 dataset:

```bash
MPLCONFIGDIR=/tmp/cnn-wp5-matplotlib .venv/bin/python -c 'from experiments.baseline.train_baseline import run_cifar10_subset_baseline; result = run_cifar10_subset_baseline(); print(result["final_metrics"])'
```

The runner requires:

```text
data/raw/cifar-10-batches-py/
```

It checks that this directory exists before loading data and does not
automatically download CIFAR-10.

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
train_loss: 2.4283203780651093
train_accuracy: 0.171875
eval_loss: 2.434686303138733
eval_accuracy: 0.15625
```

These metrics validate a tiny controlled subset only. They must not be reported
as full CIFAR-10 baseline accuracy.

## Generated Artifacts

```text
results/checkpoints/cifar10_subset_baseline.npz
results/logs/cifar10_subset_metrics.json
results/figures/cifar10_subset_loss_curve.png
results/figures/cifar10_subset_accuracy_curve.png
```

Generated artifacts are local and regenerable. They are not force-added to Git.

## Validation

```text
tests/test_cifar10_baseline_runner.py: 2 passed
tests/test_data_pipeline.py: 3 passed
tests/test_baseline_runner.py: 2 passed
tests/test_training.py: 10 passed
tests/test_plotting.py: 4 passed
full test suite: 110 passed
git diff --check: passed
```

## WP5 Completion Boundary

WP5 is completed for the controlled NumPy baseline pipeline:

* SGD parameter updates,
* single- and multi-batch training,
* clean evaluation,
* checkpoint save/load,
* JSON metrics persistence,
* loss and accuracy plotting,
* deterministic synthetic orchestration,
* deterministic real CIFAR-10 subset orchestration.

WP5 remains NumPy-only. No CuPy, CUDA, GPU, cluster, or SLURM support was
added. WP6, adversarial attacks, and Grad-CAM were not started.

Full CIFAR-10 multi-epoch baseline training is an optional deferred follow-up.
It should only be attempted when the current NumPy runtime is acceptable or
after a validated WP6 optimization.
