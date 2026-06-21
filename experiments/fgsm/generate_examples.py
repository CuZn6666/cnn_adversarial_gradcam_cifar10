"""Generate a small deterministic set of qualitative FGSM examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from configs.default_config import (
    CHECKPOINT_DIR,
    CIFAR10_EXTRACTED_DIR,
    DATA_DIR,
    NUM_CLASSES,
    PROJECT_ROOT,
    SEED,
)
from src.attacks import fgsm_attack
from src.checkpointing import load_checkpoint
from src.data.cifar10_loader import load_batch
from src.input_gradients import compute_input_gradient, input_gradient_map
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.visualization import save_fgsm_visualizations

DEFAULT_EPSILON = 8.0 / 255.0
DEFAULT_EXAMPLE_COUNT = 1
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "cifar10_subset_baseline.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "WP7" / "qualitative"


def _validate_examples(
    images: np.ndarray,
    labels: np.ndarray,
    example_count: int,
) -> None:
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("images must be a non-empty NCHW tensor.")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must match the image batch size.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class indices.")
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError("labels contain class indices outside the valid range.")
    if not np.isfinite(images).all():
        raise ValueError("images must contain only finite values.")
    if np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("images must contain values in [0, 1].")
    if (
        isinstance(example_count, bool)
        or not isinstance(example_count, int)
        or example_count <= 0
        or example_count > images.shape[0]
    ):
        raise ValueError(
            "example_count must be a positive integer no larger than "
            "the batch size."
        )


def generate_fgsm_examples(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    epsilon: float = DEFAULT_EPSILON,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Generate and save a controlled number of qualitative FGSM examples."""
    _validate_examples(images, labels, example_count)

    results: list[dict[str, Any]] = []
    for index in range(example_count):
        clean_image = images[index : index + 1]
        label = labels[index : index + 1]
        grad_input = compute_input_gradient(
            model,
            loss_function,
            clean_image,
            label,
        )
        gradient_map = input_gradient_map(grad_input)
        adversarial_image = fgsm_attack(
            clean_image,
            grad_input,
            epsilon,
        )
        paths = save_fgsm_visualizations(
            clean_image,
            adversarial_image,
            gradient_map,
            output_dir,
            prefix=f"fgsm_example_{index:03d}",
        )
        results.append(
            {
                "example_index": index,
                "label": int(label[0]),
                "epsilon": float(epsilon),
                "paths": paths,
            }
        )

    return results


def run_cifar10_fgsm_examples(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_dir: str | Path = DATA_DIR,
    epsilon: float = DEFAULT_EPSILON,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Generate qualitative FGSM examples from a local CIFAR-10 test batch."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint is not available at {checkpoint}."
        )

    dataset_dir = Path(data_dir) / CIFAR10_EXTRACTED_DIR
    test_batch_path = dataset_dir / "test_batch"
    if not test_batch_path.is_file():
        raise FileNotFoundError(
            "CIFAR-10 test data is not available at "
            f"{test_batch_path}. Download and extract it explicitly before "
            "running the FGSM example script."
        )

    test_images, test_labels = load_batch(test_batch_path)
    if example_count > test_images.shape[0]:
        raise ValueError(
            "example_count must not exceed the available test samples."
        )

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        test_images.shape[0],
        size=example_count,
        replace=False,
    )
    selected_images = test_images[selected_indices]
    selected_labels = test_labels[selected_indices]

    model = CompactCNN(seed=seed)
    load_checkpoint(model, checkpoint)
    loss_function = SoftmaxCrossEntropyLoss()
    return generate_fgsm_examples(
        model,
        loss_function,
        selected_images,
        selected_labels,
        output_dir,
        epsilon=epsilon,
        example_count=example_count,
    )


if __name__ == "__main__":
    generated_examples = run_cifar10_fgsm_examples()
    for generated_example in generated_examples:
        print(generated_example)
