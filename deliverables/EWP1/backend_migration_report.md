# EWP1 Backend Migration Report

## Status

COMPLETE.

EWP1-A: COMPLETE.

The first slice introduced a minimal NumPy-first backend boundary and migrated
the main tensor path to use that boundary. NumPy remains the default and
reference backend. CuPy remains optional and is not a required dependency.

EWP1-B: COMPLETE.

The second slice adds optional CuPy runtime-compatibility tests and environment
checks. These tests now pass on a real CUDA GPU environment. NumPy remains the
default backend and the authoritative correctness reference.

Validated environment:

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA Toolkit: 12.5
CuPy: 14.1.1
Python: 3.12
Slurm allocation: 1 GPU
```

Compatibility is claimed only for this tested environment, not for untested
GPU/CUDA/CuPy configurations.

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
* `divide_where(numerator, denominator, out=..., where=...)` provides
  NumPy-compatible masked divide semantics across NumPy and CuPy.
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
* Masked division via `divide_where(...)`.
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
  `exp`, `log`, `divide_where`, `clip`, `sign`, `matmul`, and finite checks.
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

GPU validation findings:

* `sliding_window_view`: validated on the real CuPy GPU.
* `einsum(..., optimize=True)`: validated.
* `cupy.add.at`: validated.
* `divide_where(...)`: compatibility fix validated.
* `Conv2D.forward`: NumPy/CuPy equivalence validated.
* `Conv2D.backward`: NumPy/CuPy equivalence validated for `dx`, `dw`, and `db`.
* No CPU fallback was introduced in the tested tensor path.

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
cannot be rerun locally:

* `sliding_window_view`
* `einsum(..., optimize=True)`
* `cupy.add.at`

They are covered by optional tests and should be run on a CUDA GPU node before
changing the backend implementation.

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

Observed real GPU EWP1-B results:

```text
python -m pytest -q tests/test_cupy_backend_runtime.py -rs
6 passed

python -m pytest -q -m "not requires_data"
223 passed, 3 deselected in 8.20s

git diff --check
passed
```

## Cluster CIFAR-10 Dataset Staging

The current cluster CIFAR-10 archive is a separate staging issue and is not an
EWP1 backend failure:

```text
path: data/raw/cifar-10-python.tar.gz
size: 37M
observed MD5: 352dcf059b8b606c932d1db9b8c351a9
expected MD5: c58f30108f718f92721af3b95e74349a
```

The three `requires_data` tests fail on the cluster during CIFAR-10 archive
validation before any CuPy numerical path is exercised. Do not change the
expected checksum, disable archive validation, weaken the data tests, or modify
the CIFAR-10 loader to hide this issue. Restage the dataset before using
data-dependent cluster tests or experiments.

## Risks Before CuPy Equivalence Testing

* `Conv2D` still uses a stride-trick window representation, which may be memory
  intensive on GPU for larger batches.
* Metrics and representative-example selection intentionally synchronize to
  CPU and are not designed as fully device-resident pipelines.
* Grad-CAM remains NumPy-only and should be migrated separately only if GPU
  explainability becomes part of the active scope.
* Broader EWP2 equivalence has not started; EWP1 validates runtime primitives
  and the first Conv2D equivalence slice only.
