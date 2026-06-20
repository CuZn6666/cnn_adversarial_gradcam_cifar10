from __future__ import annotations

from collections.abc import Generator

import numpy as np


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield mini-batches of images and labels.

    Args:
        images: Input images with shape (N, C, H, W).
        labels: Labels with shape (N,).
        batch_size: Number of samples in one batch.
        shuffle: Whether to shuffle data before producing batches.
        seed: Random seed used for reproducible shuffling.

    Yields:
        batch_images: Array with shape (batch_size, C, H, W),
            except possibly the final batch.
        batch_labels: Array with shape (batch_size,),
            except possibly the final batch.
    """
    if len(images) != len(labels):
        raise ValueError("Images and labels must contain the same number of samples.")

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    indices = np.arange(len(images))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start_index in range(0, len(images), batch_size):
        batch_indices = indices[start_index : start_index + batch_size]
        yield images[batch_indices], labels[batch_indices]

##Select 64 images at a time for model training
##Randomize the order of the samples during training
##The same random seed produces the same batch order
##！！！！It will still return when there are fewer than 64 images remaining,
# ---> and the remaining training data will not be lost.