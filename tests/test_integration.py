import numpy as np
import pytest

from configs.default_config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def _run_model_loss_backward_chain(
    batch_size: int = 2,
) -> tuple[CompactCNN, float, np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.random.default_rng(7).random(
        (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )
    labels = np.arange(batch_size, dtype=np.int64) % 10
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()

    logits = model.forward(inputs)
    loss = loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()
    grad_input = model.backward(grad_logits)

    return model, loss, logits, grad_logits, grad_input


def test_model_loss_backward_integration_runs_end_to_end() -> None:
    _, loss, logits, grad_logits, grad_input = (
        _run_model_loss_backward_chain()
    )

    assert np.ndim(loss) == 0
    assert np.isfinite(loss)
    assert grad_logits.shape == logits.shape
    assert grad_input.shape == (
        2,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    assert np.isfinite(grad_input).all()


def test_model_loss_backward_integration_parameter_gradients_are_finite() -> None:
    model, _, _, _, _ = _run_model_loss_backward_chain()

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


def test_model_loss_backward_integration_rejects_invalid_labels() -> None:
    inputs = np.zeros(
        (2, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )
    invalid_labels = np.array([0, 10], dtype=np.int64)
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    logits = model.forward(inputs)

    with pytest.raises(
        ValueError,
        match="labels are outside the valid class range",
    ):
        loss_function.forward(logits, invalid_labels)
