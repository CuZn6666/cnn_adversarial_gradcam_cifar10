import numpy as np

from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD
from src.training import train_step


def _synthetic_batch() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.default_rng(7).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([1, 4], dtype=np.int64)
    return images, labels


def test_train_step_returns_finite_loss_and_updates_parameters() -> None:
    images, labels = _synthetic_batch()
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=1e-3)
    parameters_before = tuple(
        parameter.copy()
        for parameter in (
            model.conv1.weights,
            model.conv1.bias,
            model.conv2.weights,
            model.conv2.bias,
            model.classifier.weights,
            model.classifier.bias,
        )
    )

    loss = train_step(
        model,
        loss_function,
        optimizer,
        images,
        labels,
    )
    named_pairs = model.named_parameters_and_gradients()

    assert np.ndim(loss) == 0
    assert np.isfinite(loss)
    assert any(
        not np.array_equal(before, parameter)
        for before, (_, parameter, _) in zip(
            parameters_before,
            named_pairs,
        )
    )
    assert all(
        np.isfinite(parameter).all()
        for _, parameter, _ in named_pairs
    )
    assert model.forward(images).shape == (2, 10)


def test_train_step_is_deterministic_for_same_seed_and_input() -> None:
    images, labels = _synthetic_batch()
    first_model = CompactCNN(seed=42)
    second_model = CompactCNN(seed=42)

    first_loss = train_step(
        first_model,
        SoftmaxCrossEntropyLoss(),
        SGD(learning_rate=1e-3),
        images,
        labels,
    )
    second_loss = train_step(
        second_model,
        SoftmaxCrossEntropyLoss(),
        SGD(learning_rate=1e-3),
        images,
        labels,
    )

    assert first_loss == second_loss
    first_named_pairs = first_model.named_parameters_and_gradients()
    second_named_pairs = second_model.named_parameters_and_gradients()
    for first_item, second_item in zip(
        first_named_pairs,
        second_named_pairs,
    ):
        assert first_item[0] == second_item[0]
        np.testing.assert_array_equal(first_item[1], second_item[1])


def test_repeated_batch_training_reduces_loss() -> None:
    images, labels = _synthetic_batch()
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=5e-4)

    initial_loss = loss_function.forward(model.forward(images), labels)
    step_losses = [
        train_step(
            model,
            loss_function,
            optimizer,
            images,
            labels,
        )
        for _ in range(3)
    ]
    final_loss = loss_function.forward(model.forward(images), labels)

    assert np.isfinite([initial_loss, *step_losses, final_loss]).all()
    assert final_loss < initial_loss
    assert all(
        np.isfinite(parameter).all()
        for parameter in (
            model.conv1.weights,
            model.conv1.bias,
            model.conv2.weights,
            model.conv2.bias,
            model.classifier.weights,
            model.classifier.bias,
        )
    )
