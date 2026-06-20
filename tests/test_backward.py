import numpy as np
import pytest

from configs.default_config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.models import CompactCNN


def _run_compact_cnn_backward(
    batch_size: int = 1,
) -> tuple[CompactCNN, np.ndarray, np.ndarray]:
    inputs = np.random.default_rng(7).random(
        (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )
    model = CompactCNN(seed=42)
    logits = model.forward(inputs)
    grad_logits = np.linspace(
        -1.0,
        1.0,
        num=logits.size,
        dtype=np.float32,
    ).reshape(logits.shape)
    grad_input = model.backward(grad_logits)
    return model, inputs, grad_input


def test_compact_cnn_backward_returns_input_gradient_shape() -> None:
    _, inputs, grad_input = _run_compact_cnn_backward(batch_size=2)

    assert grad_input.shape == inputs.shape


def test_compact_cnn_backward_produces_finite_gradients() -> None:
    model, _, grad_input = _run_compact_cnn_backward()

    gradients = (
        grad_input,
        model.conv1.grad_weight,
        model.conv1.grad_bias,
        model.conv2.grad_weight,
        model.conv2.grad_bias,
        model.classifier.grad_weight,
        model.classifier.grad_bias,
    )

    assert all(np.isfinite(gradient).all() for gradient in gradients)


def test_compact_cnn_backward_parameter_gradient_shapes() -> None:
    model, _, _ = _run_compact_cnn_backward()

    assert model.conv1.grad_weight.shape == model.conv1.weights.shape
    assert model.conv1.grad_bias.shape == model.conv1.bias.shape
    assert model.conv2.grad_weight.shape == model.conv2.weights.shape
    assert model.conv2.grad_bias.shape == model.conv2.bias.shape
    assert model.classifier.grad_weight.shape == model.classifier.weights.shape
    assert model.classifier.grad_bias.shape == model.classifier.bias.shape


def test_compact_cnn_backward_requires_forward_call() -> None:
    model = CompactCNN(seed=42)
    grad_logits = np.ones((1, 10), dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=r"CompactCNN\.backward requires a preceding forward call\.",
    ):
        model.backward(grad_logits)


@pytest.mark.parametrize("wrong_shape", [(2, 10), (1, 9)])
def test_compact_cnn_backward_rejects_wrong_grad_logits_shape(
    wrong_shape: tuple[int, int],
) -> None:
    model = CompactCNN(seed=42)
    inputs = np.zeros(
        (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )
    model.forward(inputs)
    wrong_grad_logits = np.ones(wrong_shape, dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Logits gradient shape does not match CompactCNN output.",
    ):
        model.backward(wrong_grad_logits)
