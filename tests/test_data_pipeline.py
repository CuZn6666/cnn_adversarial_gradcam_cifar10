import numpy as np

from configs.default_config import BATCH_SIZE, SEED
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_cifar10


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
