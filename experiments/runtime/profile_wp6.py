"""Inspection-only runtime profiling for the initial WP6 measurements."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from src.layers import Conv2D
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD
from src.training import train_step

SEED = 42
WARMUP_COUNT = 1
ITERATION_COUNT = 3
INPUT_SHAPE = (2, 3, 32, 32)
LEARNING_RATE = 5e-4


def _average_runtime(operation: Callable[[], object]) -> float:
    for _ in range(WARMUP_COUNT):
        operation()

    start = perf_counter()
    for _ in range(ITERATION_COUNT):
        operation()
    elapsed = perf_counter() - start
    return elapsed / ITERATION_COUNT


def profile_wp6() -> dict[str, float]:
    """Measure current Conv2D and train_step runtimes without optimization."""
    rng = np.random.default_rng(SEED)
    conv_inputs = rng.normal(size=INPUT_SHAPE).astype(np.float32)
    conv = Conv2D(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        padding=1,
        stride=1,
        rng=np.random.default_rng(SEED),
    )
    conv_output = conv.forward(conv_inputs)
    conv_grad_out = rng.normal(size=conv_output.shape).astype(np.float32)

    forward_average = _average_runtime(lambda: conv.forward(conv_inputs))
    conv.forward(conv_inputs)
    backward_average = _average_runtime(
        lambda: conv.backward(conv_grad_out)
    )

    model = CompactCNN(seed=SEED)
    loss_function = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=LEARNING_RATE)
    train_images = rng.normal(size=INPUT_SHAPE).astype(np.float32)
    train_labels = np.array([0, 1], dtype=np.int64)
    train_step_average = _average_runtime(
        lambda: train_step(
            model,
            loss_function,
            optimizer,
            train_images,
            train_labels,
        )
    )

    return {
        "conv2d_forward_seconds": forward_average,
        "conv2d_backward_seconds": backward_average,
        "train_step_seconds": train_step_average,
    }


def main() -> None:
    results = profile_wp6()
    print("WP6 inspection-only profiling")
    print(f"seed={SEED}")
    print(f"input_shape={INPUT_SHAPE}")
    print(f"warmup_count={WARMUP_COUNT}")
    print(f"iteration_count={ITERATION_COUNT}")
    for name, average_seconds in results.items():
        print(f"{name}={average_seconds:.9f}")


if __name__ == "__main__":
    main()
