# TESTING.md

## Purpose

This file defines the validation procedure for each Work Package.

A Work Package is not considered completed until the relevant tests or checks have been run and the result has been reported.

## General Local Checks

Run these commands before and after important code changes when applicable:

```bash
python --version
pytest tests/
```

If `pytest` is not installed:

```bash
pip install pytest
```

If the full test suite is too expensive, run only the relevant test file for the current Work Package.

## WP0: Project Orientation and Planning Validation

Goal:

Check that the project planning files and repository orientation are available.

Suggested checks:

```bash
ls
ls AGENTS.md WP_PLAN.md TESTING.md
```

Expected results:

* `AGENTS.md` exists.
* `WP_PLAN.md` exists.
* `TESTING.md` exists.
* The project status correctly says that WP0 and WP1 are mostly completed.
* The project status correctly reflects the current active Work Package.

## WP1: Data Pipeline Validation

Goal:

Check that CIFAR-10 data loading works correctly.

Suggested commands:

```bash
pytest tests/test_data.py
```

Expected results:

* CIFAR-10 batches can be loaded.
* Image tensors or arrays have the expected shape.
* Labels are valid class indices.
* No file path or dataset loading error occurs.

If no test exists yet:

Create a minimal test that loads one batch and checks image and label shapes.

## WP2: Compact CNN Forward Pass Validation

Goal:

Check that the compact CNN forward pass works before implementing any backward pass.

Suggested commands:

```bash
pytest tests/test_forward.py
pytest tests/
```

Expected results:

* The model accepts a CIFAR-10 shaped batch.
* The expected input shape is compatible with CIFAR-10, usually `(batch_size, 3, 32, 32)` or the project’s chosen equivalent.
* The model output has shape `(batch_size, 10)`.
* The forward pass runs without runtime errors.
* The output contains no NaN or Inf values.
* The loss can be computed if a loss function already exists.

If no forward test exists yet:

Create a minimal test that:

1. creates a small random input batch,
2. passes it through the compact CNN,
3. checks that the output shape is correct,
4. checks that the output values are finite.

Important:

Do not start WP3 manual backward implementation before WP2 forward validation passes.

## WP3: Manual Backward Validation

Status:

Completed. Manual layer backward passes, `CompactCNN.backward`,
Softmax Cross-Entropy forward/backward, and loss-to-model integration are
implemented and validated.

Goal:

Check that manual backward functions return gradients with correct shapes and stable values.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_layers.py -v
.venv/bin/python -m pytest tests/test_losses.py -v
.venv/bin/python -m pytest tests/test_backward.py -v
.venv/bin/python -m pytest tests/test_integration.py -v
.venv/bin/python -m pytest tests/ -v
```

Expected results:

* `Linear.backward` returns correct input, weight, and bias gradients.
* `ReLU.backward` masks zero and negative activations.
* `Flatten.backward` restores the original input shape.
* `MaxPool2D.backward` routes gradients to deterministic maximum locations.
* `Conv2D.backward` returns correct input, weight, and bias gradients,
  including supported stride and padding behavior.
* Softmax Cross-Entropy forward and backward match deterministic NumPy
  references and remain stable for large logits.
* `CompactCNN.backward` executes the complete reverse layer order and returns
  finite input gradients with the expected shape.
* The loss-to-model integration produces finite parameter gradients with
  shapes matching their parameters.
* No NaN or Inf occurs in gradients.

Recommended implementation order:

1. Test Linear backward.
2. Test ReLU backward.
3. Test MaxPool backward.
4. Test Conv2D backward.
5. Test Softmax Cross-Entropy backward.
6. Test full model backward.
7. Test the loss-to-model backward integration.

If a test fails:

* Do not continue to the next layer.
* Explain the failure.
* Fix only the relevant component.

## WP4: Gradient Check and Input-Gradient Validation

Status:

Completed. Numerical gradient checks and the full input-gradient pipeline
sanity check are implemented and validated.

Goal:

Compare manual gradients against numerical gradients and validate gradients
with respect to input images.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_gradient_check.py -v
.venv/bin/python -m pytest tests/ -v
```

Expected results:

* `Linear` input, weight, and bias gradients match centered finite
  differences.
* `Conv2D` input, weight, and bias gradients match centered finite
  differences.
* `SoftmaxCrossEntropyLoss.backward` matches the numerical gradient with
  respect to logits.
* Numerical checks satisfy `relative_error < 1e-4`.
* The `CompactCNN` loss-to-input gradient pipeline returns gradients with the
  same shape as the input batch.
* Input gradients are finite and not entirely zero.
* Model parameter gradients have the expected shapes and contain no NaN or
  Inf values.
* Latest validation result: `tests/test_gradient_check.py` reports 4 passed
  and the full suite reports 53 passed.

Suggested relative error criterion:

```text
relative_error < 1e-4
```

The completed checks use this threshold. `Conv2D` uses a larger finite-
difference epsilon to account for its `float32` forward output.

## WP5: Baseline Training Validation

Status:

Completed for the controlled NumPy baseline pipeline. The optimizer,
training/evaluation helpers, checkpointing, metrics persistence, plotting,
configuration, synthetic runner, and controlled real CIFAR-10 subset runner
are implemented and validated.

Goal:

Check that the model can train and that metrics are saved.

Validation order:

1. Validate deterministic optimizer parameter updates.
2. Validate single- and multi-batch training and clean evaluation.
3. Validate checkpoint, metric, plot, and baseline configuration helpers.
4. Validate the deterministic synthetic baseline runner.
5. Execute the tiny synthetic smoke run and verify its artifacts.
6. Run the full test suite.

Relevant test commands:

```bash
.venv/bin/python -m pytest tests/test_optimizer.py -v
.venv/bin/python -m pytest tests/test_training.py -v
.venv/bin/python -m pytest tests/test_checkpointing.py -v
.venv/bin/python -m pytest tests/test_metrics.py -v
.venv/bin/python -m pytest tests/test_plotting.py -v
.venv/bin/python -m pytest tests/test_config.py -v
.venv/bin/python -m pytest tests/test_baseline_runner.py -v
.venv/bin/python -m pytest tests/test_cifar10_baseline_runner.py -v
.venv/bin/python -m pytest tests/ -v
```

Tiny deterministic synthetic baseline command:

```bash
.venv/bin/python -c 'from experiments.baseline.train_baseline import run_synthetic_baseline; result = run_synthetic_baseline(); print(result["final_metrics"]); print(result["checkpoint_path"]); print(result["metrics_path"]); print(result["loss_curve_path"]); print(result["accuracy_curve_path"])'
```

This command uses `BASELINE_CONFIG` and synthetic arrays generated inside the
runner. It does not load CIFAR-10.

Expected local artifact paths:

```text
results/checkpoints/synthetic_baseline.npz
results/logs/synthetic_metrics.json
results/figures/loss_curve.png
results/figures/accuracy_curve.png
```

The checkpoint is ignored by the repository's `*.npz` rule. Generated smoke-run
artifacts should not be force-added. Regenerate them with the command above.

Expected results:

* Synthetic training orchestration starts without runtime errors.
* Loss decreases in the repeated-batch smoke test.
* Clean loss is finite and accuracy is in `[0, 1]`.
* Checkpoints, metrics, and loss/accuracy curves are saved under `results/`.
* Existing WP1–WP4 tests continue to pass.
* The full suite reports 110 passed.

Latest synthetic artifact-validation run:

```text
seed: 42
batch_size: 8
epochs: 1
train_samples: 64
eval_samples: 32
train_loss: 2.3956282436847687
train_accuracy: 0.09375
eval_loss: 2.396390974521637
eval_accuracy: 0.15625
```

These metrics are only a deterministic synthetic smoke-run result. They are
not a CIFAR-10 accuracy result and must not be presented as the full baseline.

Controlled real CIFAR-10 subset command:

```bash
MPLCONFIGDIR=/tmp/cnn-wp5-matplotlib .venv/bin/python -c 'from experiments.baseline.train_baseline import run_cifar10_subset_baseline; result = run_cifar10_subset_baseline(); print(result["final_metrics"])'
```

The command requires an existing extracted CIFAR-10 dataset under
`data/raw/cifar-10-batches-py/`. The subset runner checks for this directory
before calling the existing loader and does not automatically download data.

Controlled subset configuration and result:

```text
seed: 42
learning_rate: 0.0005
batch_size: 8
epochs: 1
train_samples: 64
eval_samples: 32
train_loss: 2.4283203780651093
train_accuracy: 0.171875
eval_loss: 2.434686303138733
eval_accuracy: 0.15625
```

Expected local subset artifact paths:

```text
results/checkpoints/cifar10_subset_baseline.npz
results/logs/cifar10_subset_metrics.json
results/figures/cifar10_subset_loss_curve.png
results/figures/cifar10_subset_accuracy_curve.png
```

The controlled subset run validates real-data pipeline integration only. It is
not full CIFAR-10 multi-epoch baseline training. Full training is deferred
because the current manual NumPy `Conv2D` runtime is slow.

For quick local testing, prefer a small number of batches or epochs.

## WP6: Focused Runtime Bottleneck Validation

Goal:

Identify and measure the main runtime bottleneck, then validate one selected
optimization path without broad framework benchmarking.

Status:

Completed. Initial profiling, single-bottleneck selection, focused
`Conv2D.backward` optimization, before/after measurement, and scoped
correctness validation are complete.

Initial profiling protocol:

* Profile before choosing or implementing an optimization.
* Measure `Conv2D.forward`, `Conv2D.backward`, and one `train_step`.
* Use a fixed random seed.
* Use documented fixed input and gradient shapes.
* Use a fixed warm-up count and fixed measured iteration count.
* Use identical inputs and timing procedure for later before/after
  comparisons.
* Keep the profiling workload small and independent of full CIFAR-10 training.

Profiling command:

```bash
.venv/bin/python -m experiments.runtime.profile_wp6
```

Final WP6 correctness checks:

```bash
.venv/bin/python -m pytest tests/test_layers.py -v -k conv2d_backward
.venv/bin/python -m pytest tests/test_gradient_check.py -v -k 'conv2d or compact_cnn_input_gradient'
.venv/bin/python -m pytest tests/test_backward.py -v
.venv/bin/python -m pytest tests/test_integration.py -v
```

Expected results:

* `Conv2D.backward` is identified as the single computational bottleneck.
* One focused NumPy optimization path is implemented.
* Conv2D, numerical-gradient, model-backward, and integration tests pass.
* Runtime measurements are used to understand the bottleneck, not to compare
  NumPy, CuPy, JAX, and PyTorch broadly.
* The measurement record includes seed, shapes, warm-up count, iteration
  count, timing summary, selected path, and limitations.

Recorded fixed benchmark:

```text
Conv2D.forward: 0.000068569 -> 0.000066375 seconds
Conv2D.backward: 0.043458736 -> 0.000209222 seconds
Conv2D.backward speedup: 207.72x
train_step: 0.070350708 -> 0.001886028 seconds
train_step speedup: 37.30x
```

The small `Conv2D.forward` difference is treated as local timing noise because
that operation was not optimized.

Out of scope for the initial WP6 profiling step:

* CIFAR-10 loading or full training,
* attacks or Grad-CAM,
* GPU, CuPy, CUDA, cluster, or SLURM support,
* adding dependencies,
* further optimization beyond the selected `Conv2D.backward` target.

## WP7: FGSM Attack and Input-Gradient Validation

Goal:

Validate input gradients, minimal FGSM behavior, and a small number of
qualitative visualizations without performing the WP8 robustness evaluation.

Status:

Completed. Input-gradient computation and maps, minimal FGSM, qualitative
visualization saving, and the controlled one-example runner are implemented
and validated.

Expected results:

* Loss gradients with respect to inputs match the input shape.
* Input gradients and adversarial images contain only finite values.
* FGSM preserves input shape.
* `epsilon=0` leaves inputs unchanged.
* The `L_inf` perturbation does not exceed epsilon within tolerance.
* Adversarial images remain in `[0, 1]`.
* Fixed inputs, labels, model parameters, and epsilon produce deterministic
  outputs.
* Input-gradient, clean-image, adversarial-image, and perturbation
  visualizations can be saved for a small number of examples.
* Model parameters are not updated during attack generation.

Final lightweight commands:

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_input_gradients.py -v
.venv/bin/python -m pytest tests/test_fgsm.py -v
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_visualization.py -v
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_fgsm_examples.py -v
.venv/bin/python -m pytest tests/test_backward.py tests/test_integration.py tests/test_losses.py -v
```

Controlled local example command:

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m experiments.fgsm.generate_examples
```

The command defaults to one deterministic CIFAR-10 test example,
`epsilon=8/255`, the checkpoint
`results/checkpoints/cifar10_subset_baseline.npz`, and the output directory
`results/WP7/qualitative/`. It requires an existing local CIFAR-10
`test_batch` and checkpoint, does not download data, and does not train or
update the model.

Expected qualitative artifact names:

```text
results/WP7/qualitative/fgsm_example_000_clean.png
results/WP7/qualitative/fgsm_example_000_adversarial.png
results/WP7/qualitative/fgsm_example_000_input_gradient.png
results/WP7/qualitative/fgsm_example_000_perturbation.png
```

The automated runner smoke test uses synthetic arrays and a temporary
directory, so the test suite does not depend on CIFAR-10 data, an external
checkpoint, or committed generated images.

Latest WP7 final validation:

```text
WP7 input-gradient, FGSM, visualization, and runner tests: 26 passed
Backward, loss integration, and loss tests: 21 passed
Controlled local CIFAR-10 example: 1 example generated successfully under
results/WP7/qualitative/
```

WP7 boundary:

* WP7 implements FGSM and creates small qualitative examples.
* WP7 does not run an epsilon sweep, aggregate attack success rate, evaluate
  large batches, or produce accuracy-versus-epsilon results.
* PGD, black-box attacks, and Grad-CAM are out of scope.

## WP8: FGSM Robustness Evaluation Validation

Goal:

Evaluate FGSM quantitatively over selected epsilon values and a larger
evaluation subset after WP7 is completed.

Preparation status:

Documentation preparation is in progress. No WP8 helper, runner, or robustness
experiment has been implemented or run.

Controlled local smoke configuration:

```text
eval_samples: 32
batch_size: 8
seed: 42
epsilon_values: [0, 2/255, 4/255, 8/255, 16/255]
output_directory: results/WP8/
deliverable_directory: deliverables/WP8/
```

Metric definitions:

```text
clean_accuracy = clean_correct / total_samples
adversarial_accuracy = adversarial_correct / total_samples
accuracy_drop = clean_accuracy - adversarial_accuracy
attack_success_rate = successful_attacks / clean_correct_samples
```

A successful attack requires a correct clean prediction and an incorrect
adversarial prediction.

Expected results:

* Clean and adversarial accuracy are measured consistently.
* Multiple epsilon values are evaluated.
* Accuracy-versus-epsilon results are produced.
* Attack success rate is aggregated where defined.
* Representative successful and failed attacks are selected.

Planned validation order:

1. Add focused unit tests for a single-batch FGSM evaluation helper.
2. Add focused unit tests for sample-weighted multi-batch aggregation.
3. Add an `epsilon=0` sanity test.
4. Validate all four metrics with controlled synthetic predictions.
5. Verify evaluation does not update model parameters.
6. Run the controlled local 32-sample smoke evaluation.

Reuse the existing FGSM, input-gradient, batching, checkpointing, metrics,
plotting, and CIFAR-10 loader code. Do not reimplement FGSM.

The current checkpoint has approximately `0.15625` controlled-subset
evaluation accuracy. WP8 results from this checkpoint validate the pipeline
only and are not a strong CIFAR-10 robustness conclusion.

Local execution is allowed for documentation, unit tests, helper
implementation, and the tiny smoke evaluation. Before any larger formal
evaluation, expanded subset, or repeated-seed run, pause and ask the user
whether to use the university-provided ZITI cluster. Do not introduce cluster,
GPU, Slurm, or CUDA workflows without explicit approval.

WP8 must not begin before WP7 FGSM implementation and its lightweight
validation are complete.

## WP9: Final Reproducibility Validation

Goal:

Check that another person can reproduce the project.

Suggested commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Then run the main experiment commands listed in the README.

Expected results:

* Dependencies install successfully.
* Tests pass.
* Main scripts run with documented commands.
* Results are saved in documented locations.

## Cluster / GPU Validation

The cluster should be used for longer training or GPU experiments, not for basic local debugging.

Before implementing or running large-scale data processing, full-dataset runs,
many-image evaluation, epsilon sweeps, repeated-seed experiments, PGD-style
multi-step evaluation, or large-batch evaluation, pause and ask the user
whether to use the university-provided ZITI cluster.

Do not introduce GPU, Slurm, CUDA, or ZITI cluster validation workflows
automatically. Use them only after explicit user approval.

Before using the cluster:

1. Ensure the project runs locally on a small test.
2. Ensure dependencies are listed in `requirements.txt` or `pyproject.toml`.
3. Ensure experiment commands are reproducible.
4. Ensure output directories such as `results/` and `logs/` exist.

Known cluster username:

```text
gpu04
```

Do not store the password in this repository.

Manual login pattern:

```bash
ssh gpu04@zitigate.ziti.uni-heidelberg.de
ssh gpu04@csg-headnode
```

Optional local SSH config pattern:

```sshconfig
Host zitigate
    HostName zitigate.ziti.uni-heidelberg.de
    User gpu04

Host headnode
    HostName csg-headnode
    User gpu04
    ProxyJump zitigate
```

Then connect with:

```bash
ssh headnode
```

Possible Slurm commands:

```bash
sinfo
squeue -u gpu04
sbatch scripts/slurm_train.sh
scancel <jobid>
```

Possible interactive GPU command pattern:

```bash
srun -p exercise-gpu --gres=gpu:1 --pty -- bash
```

Do not hard-code passwords in any script.
