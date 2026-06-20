import numpy as np
import pytest

from src.layers import Flatten, Linear, MaxPool2D, ReLU


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


def test_relu_backward_matches_hand_computed_gradient() -> None:
    layer = ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    grad_out = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    expected_grad_input = np.array([[0.0, 0.0, 6.0]], dtype=np.float32)

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    assert grad_input.shape == inputs.shape
    np.testing.assert_array_equal(grad_input, expected_grad_input)


def test_relu_backward_requires_forward_call() -> None:
    layer = ReLU()
    grad_out = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=r"ReLU\.backward requires a preceding forward call\.",
    ):
        layer.backward(grad_out)


def test_relu_backward_rejects_wrong_grad_out_shape() -> None:
    layer = ReLU()
    inputs = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    wrong_grad_out = np.ones((1, 2), dtype=np.float32)

    layer.forward(inputs)

    with pytest.raises(
        ValueError,
        match="Output gradient shape does not match ReLU output.",
    ):
        layer.backward(wrong_grad_out)


def test_flatten_backward_restores_original_shape() -> None:
    layer = Flatten()
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    grad_out = np.arange(24, dtype=np.float32).reshape(2, 12) * 0.5
    expected_grad_input = grad_out.reshape(2, 3, 4)

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    assert grad_input.shape == inputs.shape
    np.testing.assert_array_equal(grad_input, expected_grad_input)


def test_flatten_backward_requires_forward_call() -> None:
    layer = Flatten()
    grad_out = np.ones((2, 12), dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=r"Flatten\.backward requires a preceding forward call\.",
    ):
        layer.backward(grad_out)


def test_flatten_backward_rejects_wrong_grad_out_shape() -> None:
    layer = Flatten()
    inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    wrong_grad_out = np.ones((2, 11), dtype=np.float32)

    layer.forward(inputs)

    with pytest.raises(
        ValueError,
        match="Output gradient shape does not match Flatten output.",
    ):
        layer.backward(wrong_grad_out)


def test_max_pool2d_backward_routes_gradients_to_max_locations() -> None:
    layer = MaxPool2D(kernel_size=2, stride=2)
    inputs = np.array(
        [
            [
                [
                    [1.0, 3.0, 2.0, 4.0],
                    [5.0, 0.0, 6.0, 1.0],
                    [7.0, 8.0, 9.0, 2.0],
                    [0.0, 1.0, 3.0, 10.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    grad_out = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    expected_grad_input = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    np.testing.assert_array_equal(grad_input, expected_grad_input)


def test_max_pool2d_backward_restores_batch_and_channel_shape() -> None:
    layer = MaxPool2D(kernel_size=2, stride=2)
    inputs = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4)
    grad_out = np.arange(8, dtype=np.float32).reshape(2, 2, 1, 2)

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    assert grad_input.shape == inputs.shape
    assert grad_input.sum() == grad_out.sum()


def test_max_pool2d_backward_requires_forward_call() -> None:
    layer = MaxPool2D(kernel_size=2, stride=2)
    grad_out = np.ones((1, 1, 1, 1), dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=r"MaxPool2D\.backward requires a preceding forward call\.",
    ):
        layer.backward(grad_out)


def test_max_pool2d_backward_rejects_wrong_grad_out_shape() -> None:
    layer = MaxPool2D(kernel_size=2, stride=2)
    inputs = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    wrong_grad_out = np.ones((1, 1, 1, 1), dtype=np.float32)

    layer.forward(inputs)

    with pytest.raises(
        ValueError,
        match="Output gradient shape does not match MaxPool2D output.",
    ):
        layer.backward(wrong_grad_out)


def test_max_pool2d_backward_routes_ties_to_first_row_major_maximum() -> None:
    layer = MaxPool2D(kernel_size=2, stride=2)
    inputs = np.array(
        [[[[5.0, 5.0], [1.0, 0.0]]]],
        dtype=np.float32,
    )
    grad_out = np.array([[[[7.0]]]], dtype=np.float32)
    expected_grad_input = np.array(
        [[[[7.0, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )

    layer.forward(inputs)
    grad_input = layer.backward(grad_out)

    np.testing.assert_array_equal(grad_input, expected_grad_input)
