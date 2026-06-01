from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from configs.default_config import BATCH_SIZE, FIGURE_DIR, SEED
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_cifar10
from src.utils.seed import set_seed


def save_sample_batch(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Save a grid of CIFAR-10 sample images for visual verification.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(4, 4, figsize=(8, 8))

    for index, axis in enumerate(axes.flat):
        image = images[index].transpose(1, 2, 0)
        label_name = class_names[int(labels[index])]

        axis.imshow(image)
        axis.set_title(label_name, fontsize=9)
        axis.axis("off")

    figure.suptitle("CIFAR-10 Sample Batch", fontsize=14)
    figure.tight_layout()

    output_path = FIGURE_DIR / "cifar10_sample_batch.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    print(f"Saved sample visualization to: {output_path}")


def main() -> None:
    set_seed(SEED)

    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    print("=== CIFAR-10 Pipeline Sanity Check ===")
    print(f"Training images shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test images shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Image dtype: {x_train.dtype}")
    print(f"Image value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    print(f"Class names: {class_names}")

    assert x_train.shape == (50000, 3, 32, 32)
    assert y_train.shape == (50000,)
    assert x_test.shape == (10000, 3, 32, 32)
    assert y_test.shape == (10000,)

    assert x_train.dtype == np.float32
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0

    assert y_train.dtype == np.int64
    assert y_train.min() == 0
    assert y_train.max() == 9

    assert len(class_names) == 10

    loader_1 = iterate_minibatches(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    loader_2 = iterate_minibatches(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    first_images, first_labels = next(loader_1)
    second_images, second_labels = next(loader_2)

    assert first_images.shape == (BATCH_SIZE, 3, 32, 32)
    assert first_labels.shape == (BATCH_SIZE,)

    assert np.array_equal(first_images, second_images)
    assert np.array_equal(first_labels, second_labels)

    print(f"Mini-batch shape: {first_images.shape}")
    print("Reproducibility check passed: identical seed produces identical first batch.")

    save_sample_batch(first_images, first_labels, class_names)

    print("All WP1 data pipeline checks passed successfully.")


if __name__ == "__main__":
    main()
