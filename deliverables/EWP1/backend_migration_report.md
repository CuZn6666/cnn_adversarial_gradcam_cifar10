# EWP1 Backend Migration Report

## Status

PARTIALLY COMPLETE.

EWP1-A: COMPLETE.

The first slice introduced a minimal NumPy-first backend boundary and migrated
the main tensor path to use that boundary. NumPy remains the default and
reference backend. CuPy remains optional and is not a required dependency.

EWP1-B: NEEDS GPU RUNTIME.

The second slice adds optional CuPy runtime-compatibility tests and environment
checks. The local development environment does not have CuPy installed and does
not expose CUDA tooling, so actual GPU execution remains unverified locally.
The tests skip cleanly and can be run later on a CUDA GPU node.

No PGD implementation, PGD refactor, or PGD experiment was added.

## Files Modified

Core backend and tensor path:

* `src/backend.py`
* `src/layers/forward.py`
* `src/models/compact_cnn.py`
* `src/losses/cross_entropy.py`
* `src/optimizers/sgd.py`
* `src/training.py`
* `src/input_gradients.py`
* `src/attacks/fgsm.py`
* `src/robustness.py`
* `src/checkpointing.py`

Validation and planning:

* `tests/test_backend.py`
* `tests/cupy_test_utils.py`
* `tests/conftest.py`
* `tests/test_cupy_backend_runtime.py`
* `TESTING.md`
* `WP_PLAN.md`
* `deliverables/EWP1/backend_migration_report.md`

## Backend API Introduced

`src/backend.py` provides the only new backend abstraction:

* `resolve_backend(backend)` resolves `"numpy"`, `"cupy"`, `numpy`, or `cupy`.
* `get_array_module(*arrays)` detects whether tensors are NumPy or CuPy arrays.
* `ensure_same_backend(*arrays)` rejects mixed NumPy/CuPy tensors in one
  operation.
* `ensure_backend_array(array, backend, name=...)` validates a tensor against a
  selected backend.
* `to_backend(array, backend, dtype=...)` performs explicit CPU/GPU transfer.
* `to_numpy(array)` performs explicit host conversion for artifacts and
  serialization.
* `to_python_bool`, `to_python_float`, and `to_python_int` mark scalar
  synchronization points.
* `isfinite_all(array)` centralizes finite-value validation.
* `sliding_window_view(...)` dispatches to NumPy or CuPy stride-trick support.

The model and layer constructors now accept `backend="numpy"` by default.
`CompactCNN(seed=..., backend="cupy")` is the intended future CuPy entry point
when CuPy is installed.

## Migrated Tensor Operations

The following paths now use backend-aware operations:

* `Conv2D.forward` and `Conv2D.backward`
* `ReLU.forward` and `ReLU.backward`
* `MaxPool2D.forward` and `MaxPool2D.backward`
* `Flatten.forward` and `Flatten.backward`
* `Linear.forward` and `Linear.backward`
* `CompactCNN.forward` and `CompactCNN.backward`
* `SoftmaxCrossEntropyLoss.forward` and `SoftmaxCrossEntropyLoss.backward`
* `SGD.step`
* `train_step`, `train_batches`, `evaluate_batch`, and `evaluate_batches`
* `compute_input_gradient` and `input_gradient_map`
* `fgsm_attack`
* `evaluate_fgsm_batch`, `evaluate_fgsm_batches`,
  `evaluate_fgsm_epsilon_sweep`, and representative-example selection
* `save_checkpoint` and `load_checkpoint` CPU/archive boundaries

## Remaining NumPy-Only Operations

These remain intentionally CPU-side:

* CIFAR-10 loading and preprocessing in `src/data/`.
* Mini-batch creation in `src/data/batching.py`.
* JSON metrics and result serialization in `src/metrics.py`.
* Matplotlib and PIL-based plotting/visualization in `src/plotting.py`,
  `src/visualization.py`, and `src/gradcam_visualization.py`.
* Grad-CAM implementation in `src/gradcam.py`.
* Experiment runners under `experiments/`, except where they call migrated
  tensor functions with NumPy arrays.
* `.npz` checkpoint archives, which remain NumPy files by design.

## Host/Device Transfer Boundaries

Explicit CPU/GPU boundaries introduced or preserved:

* Model initialization uses NumPy RNG for deterministic reference-compatible
  parameters, then copies parameters to the selected backend.
* `to_backend(...)` should be used by future runners when moving CIFAR-10
  batches from CPU loaders to GPU tensors.
* `save_checkpoint(...)` converts model parameters to NumPy before `np.savez`.
* `load_checkpoint(...)` loads NumPy arrays with `np.load` and converts them to
  the target model backend before assignment.
* Training and evaluation metrics convert backend scalars to Python scalars for
  logging and JSON compatibility.
* Representative-example selection converts predictions and labels to NumPy
  once per batch before Python metadata loops.
* Plotting and image output should receive NumPy arrays; future CuPy callers
  must convert with `to_numpy(...)` before visualization.

## Operations Requiring CuPy Equivalence Attention

CuPy equivalence testing should pay special attention to:

* `sliding_window_view` availability and view semantics in the installed CuPy
  version.
* `einsum(..., optimize=True)` behavior and dtype preservation.
* `cupy.add.at` behavior in `MaxPool2D.backward`, especially tie handling.
* Scalar synchronization from `to_python_*` helpers.
* Validation checks that call `isfinite_all(...)`.
* Checkpoint load/save transfers between CPU `.npz` archives and device
  tensors.
* Any future experiment runner that mixes CPU data loader output with a CuPy
  model without explicit `to_backend(...)`.

## EWP1-B Runtime Compatibility Checks

The optional CuPy runtime tests cover:

* CuPy import and version visibility.
* CUDA runtime version query.
* Visible CUDA device count and GPU name.
* Simple CuPy allocation, computation, and synchronization.
* Array creation/conversion through `to_backend(...)` and `to_numpy(...)`.
* `zeros`, `zeros_like`, `maximum`, `max`, `argmax`, `sum`, `mean`, `abs`,
  `exp`, `log`, `divide`, `clip`, `sign`, `matmul`, and finite checks.
* Scalar conversions through `to_python_bool`, `to_python_float`, and
  `to_python_int`.
* `einsum(..., optimize=True)`.
* `cupy.add.at` with repeated indices.
* `sliding_window_view(...)` through the backend wrapper.
* `Conv2D.forward` and `Conv2D.backward` equivalence for deterministic small
  tensors.

The Conv2D equivalence test compares forward outputs, input gradients,
weight gradients, and bias gradients with:

```text
rtol = 1e-5
atol = 1e-6
```

These tolerances are chosen for small float32 tensor operations. They are not
weakened to mask runtime differences; they should be revisited only if an
actual CuPy run demonstrates a numerically justified difference.

## Current Local Environment

Observed on the current local development machine:

```text
Python: 3.13.1
CuPy installed: no
nvidia-smi: not available
nvcc: not available
CUDA GPU visible through CuPy: not testable because cupy is not installed
Simple CuPy allocation/computation: not testable because cupy is not installed
```

Because no CuPy runtime is available locally, the following risky primitives
remain runtime-unverified:

* `sliding_window_view`
* `einsum(..., optimize=True)`
* `cupy.add.at`

They are covered by optional tests and should be run on a CUDA GPU node before
claiming EWP1-B complete.

## Validation

Commands run after this slice:

```bash
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_backend.py tests/test_forward.py tests/test_layers.py tests/test_losses.py tests/test_training.py tests/test_input_gradients.py tests/test_fgsm.py tests/test_fgsm_evaluation.py --maxfail=1
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m "not requires_data" --maxfail=1
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -m requires_data
MPLCONFIGDIR=/tmp/cnn-ci-matplotlib PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cupy_backend_runtime.py -rs
```

Observed results:

```text
87 passed
219 passed, 6 skipped
216 passed, 6 skipped, 3 deselected
3 passed, 222 deselected
6 skipped because cupy is not installed
```

Observed EWP1-B results after optional CuPy tests were added:

```text
Pre-change full suite: 219 passed
Post-change full suite: 219 passed, 6 skipped
CuPy runtime slice: 6 skipped because cupy is not installed
```

## Risks Before CuPy Equivalence Testing

* CuPy is not installed or exercised in the current local validation.
* `Conv2D` still uses a stride-trick window representation, which may be memory
  intensive on GPU for larger batches.
* `MaxPool2D.backward` uses `add.at`; correctness should be checked before
  performance tuning.
* Metrics and representative-example selection intentionally synchronize to
  CPU and are not designed as fully device-resident pipelines.
* Grad-CAM remains NumPy-only and should be migrated separately only if GPU
  explainability becomes part of the active scope.
* EWP1-B cannot be marked complete until the optional CuPy runtime tests run on
  an environment with CuPy, CUDA runtime access, and a visible CUDA GPU.
