import numpy as np
import pytest

from src.layers import Linear


def test_linear_backward_matches_hand_computed_gradients() -> None:
    layer = Linear(
        in_features=2,
        out_features=3,
        rng=np.random.default_rng(0),
    )
    layer.weights = np.array(
        [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]],
        dtype=np.float32,
    )
    layer.bias = np.array([0.5, -1.0, 2.0], dtype=np.float32)

    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    grad_out = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    expected_grad_input = np.array(
        [[-2.0, 7.0], [-2.0, 16.0]],
        dtype=np.float32,
    )
    expected_grad_weight = np.array(
        [[13.0, 18.0], [17.0, 24.0], [21.0, 30.0]],
        dtype=np.float32,
    )
    expected_grad_bias = np.array([5.0, 7.0, 9.0], dtype=np.float32)

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    assert grad_input.shape == inputs.shape
    assert layer.grad_weight.shape == layer.weights.shape
    assert layer.grad_bias.shape == layer.bias.shape
    np.testing.assert_allclose(grad_input, expected_grad_input)
    np.testing.assert_allclose(layer.grad_weight, expected_grad_weight)
    np.testing.assert_allclose(layer.grad_bias, expected_grad_bias)


def test_linear_backward_requires_forward_call() -> None:
    layer = Linear(
        in_features=2,
        out_features=3,
        rng=np.random.default_rng(0),
    )
    grad_out = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=r"Linear\.backward requires a preceding forward call\.",
    ):
        layer.backward(grad_out)


def test_linear_backward_rejects_wrong_grad_out_shape() -> None:
    layer = Linear(
        in_features=2,
        out_features=3,
        rng=np.random.default_rng(0),
    )
    inputs = np.ones((2, 2), dtype=np.float32)
    wrong_grad_out = np.ones((2, 2), dtype=np.float32)

    layer.forward(inputs)

    with pytest.raises(
        ValueError,
        match="Output gradient shape does not match Linear output.",
    ):
        layer.backward(wrong_grad_out)
