import numpy as np

from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def _model_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


def _deterministic_inputs() -> tuple[np.ndarray, np.ndarray]:
    images = np.random.default_rng(17).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([2, 5], dtype=np.int64)
    return images, labels


def test_compute_input_gradient_returns_valid_gradient_without_parameter_updates(
) -> None:
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    images, labels = _deterministic_inputs()
    parameters_before = [
        parameter.copy() for parameter in _model_parameters(model)
    ]

    grad_input = compute_input_gradient(
        model,
        loss_function,
        images,
        labels,
    )

    assert grad_input.shape == images.shape
    assert np.isfinite(grad_input).all()
    assert np.any(grad_input != 0.0)
    for parameter, parameter_before in zip(
        _model_parameters(model),
        parameters_before,
        strict=True,
    ):
        np.testing.assert_array_equal(parameter, parameter_before)


def test_compute_input_gradient_is_deterministic() -> None:
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    images, labels = _deterministic_inputs()

    first_gradient = compute_input_gradient(
        model,
        loss_function,
        images,
        labels,
    ).copy()
    second_gradient = compute_input_gradient(
        model,
        loss_function,
        images,
        labels,
    )

    np.testing.assert_array_equal(first_gradient, second_gradient)
