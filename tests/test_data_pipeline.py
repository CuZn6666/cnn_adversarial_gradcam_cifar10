import pickle
from pathlib import Path

import numpy as np
import pytest

from configs.default_config import BATCH_SIZE, SEED
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_batch, load_cifar10


@pytest.mark.requires_data
def test_cifar10_shapes_and_ranges() -> None:
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

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


@pytest.mark.requires_data
def test_minibatch_shapes() -> None:
    x_train, y_train, _, _, _ = load_cifar10()

    loader = iterate_minibatches(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    batch_images, batch_labels = next(loader)

    assert batch_images.shape == (BATCH_SIZE, 3, 32, 32)
    assert batch_labels.shape == (BATCH_SIZE,)


@pytest.mark.requires_data
def test_minibatch_reproducibility() -> None:
    x_train, y_train, _, _, _ = load_cifar10()

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

    images_1, labels_1 = next(loader_1)
    images_2, labels_2 = next(loader_2)

    assert np.array_equal(images_1, images_2)
    assert np.array_equal(labels_1, labels_2)


def test_load_batch_converts_synthetic_cifar10_pickle(
    tmp_path: Path,
) -> None:
    batch_path = tmp_path / "data_batch_1"
    raw_images = np.arange(2 * 3 * 32 * 32, dtype=np.uint8).reshape(2, -1)
    labels = [3, 7]
    with batch_path.open("wb") as file:
        pickle.dump({b"data": raw_images, b"labels": labels}, file)

    images, loaded_labels = load_batch(batch_path)

    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == np.float32
    assert images.min() >= 0.0
    assert images.max() <= 1.0
    np.testing.assert_array_equal(loaded_labels, np.array(labels, dtype=np.int64))


def test_minibatch_reproducibility_with_synthetic_arrays() -> None:
    images = np.arange(6 * 3 * 4 * 4, dtype=np.float32).reshape(6, 3, 4, 4)
    labels = np.arange(6, dtype=np.int64)

    first_loader = iterate_minibatches(
        images,
        labels,
        batch_size=3,
        shuffle=True,
        seed=SEED,
    )
    second_loader = iterate_minibatches(
        images,
        labels,
        batch_size=3,
        shuffle=True,
        seed=SEED,
    )

    first_images, first_labels = next(first_loader)
    second_images, second_labels = next(second_loader)

    np.testing.assert_array_equal(first_images, second_images)
    np.testing.assert_array_equal(first_labels, second_labels)
