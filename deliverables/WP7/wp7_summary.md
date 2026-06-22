# WP7 Summary

## Status

Completed.

## Goal

Implement minimal untargeted FGSM support, compute and visualize loss
gradients with respect to input images, and generate a small number of
qualitative clean/adversarial/gradient/perturbation artifacts.

## Implemented Components

* `compute_input_gradient(...)` composes the existing model and loss backward
  pipeline without updating model parameters.
* `input_gradient_map(...)` converts finite NCHW gradients into deterministic
  normalized spatial maps.
* `fgsm_attack(...)` applies:

  ```python
  np.clip(images + epsilon * np.sign(grad_input), 0.0, 1.0)
  ```

* `save_fgsm_visualizations(...)` saves clean, adversarial, input-gradient, and
  perturbation PNG files for one example.
* `generate_fgsm_examples(...)` composes the WP7 helpers for deterministic
  synthetic or already-loaded arrays.
* `run_cifar10_fgsm_examples(...)` loads an existing checkpoint and one or more
  deterministic samples from a local CIFAR-10 test batch. The default remains
  one example with `epsilon=8/255`.

## Controlled Local Command

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m experiments.fgsm.generate_examples
```

Required local inputs:

```text
results/checkpoints/cifar10_subset_baseline.npz
data/raw/cifar-10-batches-py/test_batch
```

The command does not download CIFAR-10, train the model, or update model
parameters.

Default qualitative outputs:

```text
results/WP7/qualitative/fgsm_example_000_clean.png
results/WP7/qualitative/fgsm_example_000_adversarial.png
results/WP7/qualitative/fgsm_example_000_input_gradient.png
results/WP7/qualitative/fgsm_example_000_perturbation.png
```

Generated images are local experiment artifacts and are not required for the
automated test suite.

## Validation

```bash
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_input_gradients.py -v
.venv/bin/python -m pytest tests/test_fgsm.py -v
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_visualization.py -v
MPLCONFIGDIR=/tmp/cnn-wp7-matplotlib .venv/bin/python -m pytest tests/test_fgsm_examples.py -v
.venv/bin/python -m pytest tests/test_backward.py tests/test_integration.py tests/test_losses.py -v
```

The tests verify:

* finite deterministic input gradients with unchanged model parameters,
* deterministic normalized input-gradient maps,
* FGSM shape, clipping, `epsilon=0`, and `L_inf` behavior,
* deterministic non-empty PNG outputs,
* end-to-end helper composition using synthetic arrays and a temporary output
  directory.

Final validation result:

```text
WP7 input-gradient, FGSM, visualization, and runner tests: 26 passed
Backward, loss integration, and loss tests: 21 passed
```

A controlled local CIFAR-10 validation was also run with the existing
checkpoint, the existing test batch, one example, `seed=42`, and
`epsilon=8/255`. It selected label `7` and generated all four expected PNG
files under `results/WP7/qualitative/`.

## Scope Boundary

WP7 contains only minimal FGSM implementation and small qualitative example
generation.

The following remain WP8 or later work:

* epsilon sweeps,
* clean-versus-adversarial accuracy evaluation,
* accuracy-versus-epsilon plots,
* attack success-rate aggregation,
* many-image or large-batch robustness evaluation,
* PGD and black-box attacks,
* Grad-CAM.

Before any large-scale evaluation, repeated-seed experiment, epsilon sweep, or
many-image run, ask whether to use the university-provided ZITI cluster. No
GPU, CUDA, CuPy, JAX, PyTorch, Slurm, or cluster workflow was added in WP7.
