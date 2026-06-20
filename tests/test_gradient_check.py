from collections.abc import Callable

import numpy as np

from src.layers import Conv2D, Linear
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def _centered_finite_difference(
    values: np.ndarray,
    objective: Callable[[], float],
    epsilon: float = 1e-6,
) -> np.ndarray:
    numerical_gradient = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        original_value = values[index]

        values[index] = original_value + epsilon
        objective_plus = objective()

        values[index] = original_value - epsilon
        objective_minus = objective()

        values[index] = original_value
        numerical_gradient[index] = (
            objective_plus - objective_minus
        ) / (2.0 * epsilon)

    return numerical_gradient


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(
        1e-12,
        np.abs(actual) + np.abs(expected),
    )
    return float(np.max(np.abs(actual - expected) / denominator))


def test_linear_backward_matches_numerical_gradients() -> None:
    rng = np.random.default_rng(42)
    layer = Linear(in_features=3, out_features=2, rng=rng)
    layer.weights = rng.normal(size=(2, 3))
    layer.bias = rng.normal(size=2)
    inputs = rng.normal(size=(2, 3))
    grad_out = rng.normal(size=(2, 2))

    layer.forward(inputs)
    analytical_grad_input = layer.backward(grad_out).copy()
    analytical_grad_weight = layer.grad_weight.copy()
    analytical_grad_bias = layer.grad_bias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * grad_out))

    numerical_grad_input = _centered_finite_difference(inputs, objective)
    numerical_grad_weight = _centered_finite_difference(
        layer.weights,
        objective,
    )
    numerical_grad_bias = _centered_finite_difference(layer.bias, objective)

    assert _relative_error(analytical_grad_input, numerical_grad_input) < 1e-4
    assert _relative_error(analytical_grad_weight, numerical_grad_weight) < 1e-4
    assert _relative_error(analytical_grad_bias, numerical_grad_bias) < 1e-4


def test_conv2d_backward_matches_numerical_gradients() -> None:
    rng = np.random.default_rng(7)
    layer = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        rng=rng,
    )
    layer.weights = rng.uniform(0.5, 1.5, size=(1, 1, 2, 2))
    layer.bias = rng.uniform(0.5, 1.5, size=1)
    inputs = rng.uniform(0.5, 1.5, size=(1, 1, 3, 3))
    grad_out = rng.uniform(0.5, 1.5, size=(1, 1, 2, 2))

    layer.forward(inputs)
    analytical_grad_input = layer.backward(grad_out).copy()
    analytical_grad_weight = layer.grad_weight.copy()
    analytical_grad_bias = layer.grad_bias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * grad_out))

    epsilon = 1e-3
    numerical_grad_input = _centered_finite_difference(
        inputs,
        objective,
        epsilon,
    )
    numerical_grad_weight = _centered_finite_difference(
        layer.weights,
        objective,
        epsilon,
    )
    numerical_grad_bias = _centered_finite_difference(
        layer.bias,
        objective,
        epsilon,
    )

    assert _relative_error(analytical_grad_input, numerical_grad_input) < 1e-4
    assert _relative_error(analytical_grad_weight, numerical_grad_weight) < 1e-4
    assert _relative_error(analytical_grad_bias, numerical_grad_bias) < 1e-4


def test_softmax_cross_entropy_backward_matches_numerical_gradient() -> None:
    logits = np.array(
        [[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]],
        dtype=np.float64,
    )
    labels = np.array([2, 0], dtype=np.int64)
    loss_function = SoftmaxCrossEntropyLoss()

    loss_function.forward(logits, labels)
    analytical_grad_logits = loss_function.backward()

    def objective() -> float:
        return loss_function.forward(logits, labels)

    numerical_grad_logits = _centered_finite_difference(logits, objective)

    assert (
        _relative_error(analytical_grad_logits, numerical_grad_logits)
        < 1e-4
    )


def test_compact_cnn_input_gradient_pipeline_is_finite_and_nonzero() -> None:
    inputs = np.random.default_rng(11).random(
        (1, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([3], dtype=np.int64)
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()

    logits = model.forward(inputs)
    loss = loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()
    grad_input = model.backward(grad_logits)

    assert logits.shape == (1, 10)
    assert np.isfinite(loss)
    assert grad_logits.shape == logits.shape
    assert grad_input.shape == inputs.shape
    assert np.isfinite(grad_input).all()
    assert np.any(grad_input != 0.0)

    parameters_and_gradients = (
        (model.conv1.weights, model.conv1.grad_weight),
        (model.conv1.bias, model.conv1.grad_bias),
        (model.conv2.weights, model.conv2.grad_weight),
        (model.conv2.bias, model.conv2.grad_bias),
        (model.classifier.weights, model.classifier.grad_weight),
        (model.classifier.bias, model.classifier.grad_bias),
    )
    for parameter, gradient in parameters_and_gradients:
        assert gradient.shape == parameter.shape
        assert np.isfinite(gradient).all()
