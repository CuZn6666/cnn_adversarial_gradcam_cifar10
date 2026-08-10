# WP1 Reproducibility Check

## Scope

This document records the reproducibility checks supported by the current
CIFAR-10 data pipeline. It uses repository evidence and does not claim results
from experiments that were not run.

## Fixed Project Conventions

```text
image_format: NCHW
image_shape: (N, 3, 32, 32)
image_dtype: float32
image_range: [0, 1]
label_dtype: int64
label_range: [0, 9]
default_seed: 42
default_batch_size: 64
```

These constants are defined in `configs/default_config.py`.

## Reproducible Mini-Batches

`src/data/batching.py` implements deterministic shuffling through
`np.random.default_rng(seed)`. With the same images, labels, batch size, and
seed, the first shuffled mini-batch is reproducible.

The behavior is tested in:

```text
tests/test_data_pipeline.py
```

## Dataset Loading

`src/data/cifar10_loader.py` loads the official CIFAR-10 Python batch format
and converts:

* raw flattened image rows to NCHW tensors,
* pixel values from `[0, 255]` to `[0, 1]`,
* labels to `int64`.

When local CIFAR-10 data is available, the data-marked tests validate the full
shape and range expectations.

## Commands

Run the data-pipeline tests:

```bash
.venv/bin/python -m pytest tests/test_data_pipeline.py -v
```

Run the full data sanity-check script:

```bash
.venv/bin/python -m experiments.check_data_pipeline
```

The sanity-check script also saves:

```text
results/figures/cifar10_sample_batch.png
```

## Latest Repository-Level Validation

Latest audit validation:

```text
Offline CI-compatible suite: 212 passed, 3 deselected
Data-marked suite: 3 passed, 212 deselected
Full local suite: 215 passed
```

## Remaining TODO

If this document is used for a final report, record the exact machine,
Python version, and dataset checksum output from the final reproduction run.
