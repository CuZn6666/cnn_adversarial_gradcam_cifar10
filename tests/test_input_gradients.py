import numpy as np
import pytest

from src.input_gradients import compute_input_gradient, input_gradient_map
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


def test_input_gradient_map_returns_normalized_spatial_maps() -> None:
    grad_input = np.array(
        [
            [
                [[-1.0, 2.0], [3.0, -4.0]],
                [[3.0, -2.0], [1.0, 0.0]],
            ],
            [
                [[1.0, -3.0], [2.0, -4.0]],
                [[1.0, 1.0], [-2.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )

    gradient_map = input_gradient_map(grad_input)

    assert gradient_map.shape == (2, 2, 2)
    assert np.isfinite(gradient_map).all()
    assert gradient_map.min() >= 0.0
    assert gradient_map.max() <= 1.0
    expected_map = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(gradient_map, expected_map)


def test_input_gradient_map_returns_zero_for_zero_gradients() -> None:
    grad_input = np.zeros((2, 3, 4, 5), dtype=np.float32)

    gradient_map = input_gradient_map(grad_input)

    np.testing.assert_array_equal(
        gradient_map,
        np.zeros((2, 4, 5), dtype=np.float32),
    )


def test_input_gradient_map_is_deterministic() -> None:
    grad_input = np.random.default_rng(23).normal(
        size=(2, 3, 4, 5),
    ).astype(np.float32)

    first_map = input_gradient_map(grad_input)
    second_map = input_gradient_map(grad_input)

    np.testing.assert_array_equal(first_map, second_map)


def test_input_gradient_map_rejects_invalid_shape() -> None:
    invalid_gradient = np.zeros((3, 4, 5), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="expects a non-empty NCHW gradient tensor",
    ):
        input_gradient_map(invalid_gradient)
