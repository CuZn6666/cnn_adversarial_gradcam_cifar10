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

Planned. This is the next target. Input gradients are already returned by
`CompactCNN.backward`, but numerical gradient checking has not started.

Goal:

Compare manual gradients against numerical gradients and validate gradients
with respect to input images.

Suggested commands:

```bash
.venv/bin/python -m pytest tests/test_gradient_check.py -v
.venv/bin/python -m pytest tests/ -v
```

Expected results:

* Numerical and manual gradients have small relative error.
* Input gradients have the same shape as the input image batch.
* Input gradients do not contain NaN or Inf.

Suggested relative error criterion:

```text
relative_error < 1e-4
```

This threshold may be relaxed for convolution or pooling layers if justified.

## WP5: Baseline Training Validation

Goal:

Check that the model can train and that metrics are saved.

Suggested commands:

```bash
python experiments/train_baseline.py --config configs/baseline.yaml
```

Expected results:

* Training starts without runtime errors.
* Loss decreases at least on a small subset or short run.
* Accuracy is above random chance after training.
* Metrics are saved under `results/`.

For quick local testing, prefer a small number of batches or epochs.

## WP6: Adversarial Attack Validation

Goal:

Check that adversarial examples can be generated and evaluated.

Suggested commands:

```bash
python experiments/evaluate_fgsm.py --config configs/fgsm.yaml
```

Expected results:

* FGSM creates perturbed images with the same shape as original inputs.
* Perturbations are bounded by epsilon.
* Pixel values remain in the valid range.
* Accuracy decreases as epsilon increases.

## WP7: Grad-CAM Validation

Goal:

Check that Grad-CAM heatmaps can be generated and saved.

Suggested commands:

```bash
python experiments/run_gradcam.py --config configs/gradcam.yaml
```

Expected results:

* Heatmaps are generated.
* Heatmaps have valid spatial dimensions.
* Visualizations are saved under `results/` or `deliverables/`.
* The script works for at least one clean image and one adversarial image.

## WP8: Analysis Validation

Goal:

Check that robustness and explanation results are summarized correctly.

Suggested checks:

```bash
ls results/
ls deliverables/
```

Expected results:

* Clean accuracy results exist.
* Adversarial accuracy results exist.
* Grad-CAM visualizations exist.
* Plots or tables are saved.
* Written analysis explains what was expected and what was observed.

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
