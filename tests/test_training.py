import numpy as np

from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD
from src.training import evaluate_batch, evaluate_batches, train_step


def _synthetic_batch() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.default_rng(7).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([1, 4], dtype=np.int64)
    return images, labels


def _model_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


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
        for parameter in _model_parameters(model)
    )


def test_evaluate_batch_returns_finite_loss_and_valid_accuracy() -> None:
    images, labels = _synthetic_batch()

    loss, accuracy = evaluate_batch(
        CompactCNN(seed=42),
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
    )

    assert np.ndim(loss) == 0
    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_batch_computes_controlled_accuracy_without_updates() -> None:
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    labels = np.array([3, 1], dtype=np.int64)
    model = CompactCNN(seed=42)
    for parameter in _model_parameters(model):
        parameter.fill(0.0)
    model.classifier.bias[3] = 1.0
    parameters_before = tuple(
        parameter.copy() for parameter in _model_parameters(model)
    )

    loss, accuracy = evaluate_batch(
        model,
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
    )

    assert np.isfinite(loss)
    assert accuracy == 0.5
    for before, parameter in zip(
        parameters_before,
        _model_parameters(model),
    ):
        np.testing.assert_array_equal(parameter, before)


def test_evaluate_batch_is_deterministic() -> None:
    images, labels = _synthetic_batch()
    model = CompactCNN(seed=42)

    first_result = evaluate_batch(
        model,
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
    )
    second_result = evaluate_batch(
        model,
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
    )

    assert first_result == second_result


def test_evaluate_batches_uses_sample_weighted_aggregation_without_updates() -> None:
    model = CompactCNN(seed=42)
    for parameter in _model_parameters(model):
        parameter.fill(0.0)
    model.classifier.bias[3] = 1.0
    batches = (
        (
            np.zeros((1, 3, 32, 32), dtype=np.float32),
            np.array([3], dtype=np.int64),
        ),
        (
            np.zeros((3, 3, 32, 32), dtype=np.float32),
            np.array([1, 1, 3], dtype=np.int64),
        ),
    )
    parameters_before = tuple(
        parameter.copy() for parameter in _model_parameters(model)
    )
    reference_loss_function = SoftmaxCrossEntropyLoss()
    first_loss, _ = evaluate_batch(
        model,
        reference_loss_function,
        *batches[0],
    )
    second_loss, _ = evaluate_batch(
        model,
        reference_loss_function,
        *batches[1],
    )
    expected_weighted_loss = (first_loss + 3.0 * second_loss) / 4.0
    simple_batch_mean = (first_loss + second_loss) / 2.0

    mean_loss, accuracy = evaluate_batches(
        model,
        SoftmaxCrossEntropyLoss(),
        batches,
    )

    assert np.isfinite(mean_loss)
    assert 0.0 <= accuracy <= 1.0
    np.testing.assert_allclose(mean_loss, expected_weighted_loss)
    assert not np.isclose(mean_loss, simple_batch_mean)
    assert accuracy == 0.5
    for before, parameter in zip(
        parameters_before,
        _model_parameters(model),
    ):
        np.testing.assert_array_equal(parameter, before)


def test_evaluate_batches_is_deterministic() -> None:
    images, labels = _synthetic_batch()
    batches = (
        (images[:1], labels[:1]),
        (images[1:], labels[1:]),
    )
    model = CompactCNN(seed=42)

    first_result = evaluate_batches(
        model,
        SoftmaxCrossEntropyLoss(),
        batches,
    )
    second_result = evaluate_batches(
        model,
        SoftmaxCrossEntropyLoss(),
        batches,
    )

    assert first_result == second_result
