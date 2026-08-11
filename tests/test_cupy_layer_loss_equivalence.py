import numpy as np
import pytest

from src.backend import is_cupy_array, to_numpy
from src.layers import Flatten, Linear, MaxPool2D, ReLU
from src.losses import SoftmaxCrossEntropyLoss


pytestmark = pytest.mark.requires_cupy


FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6
LOSS_RTOL = 1e-6
LOSS_ATOL = 1e-7


def test_relu_forward_backward_matches_numpy(cp) -> None:
    inputs_np = np.array(
        [[-2.0, -0.0, 0.0, 1.5], [3.0, -4.0, 0.25, -0.75]],
        dtype=np.float32,
    )
    grad_out_np = np.array(
        [[0.5, 1.0, -1.5, 2.0], [-0.25, 0.75, 1.25, -2.5]],
        dtype=np.float32,
    )
    numpy_layer = ReLU(backend="numpy")
    cupy_layer = ReLU(backend=cp)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    assert is_cupy_array(outputs_cp)
    np.testing.assert_array_equal(to_numpy(outputs_cp), outputs_np)

    grad_input_np = numpy_layer.backward(grad_out_np)
    grad_input_cp = cupy_layer.backward(cp.asarray(grad_out_np))
    assert is_cupy_array(grad_input_cp)
    np.testing.assert_array_equal(to_numpy(grad_input_cp), grad_input_np)


def test_maxpool2d_forward_backward_matches_numpy_with_unique_maxima(cp) -> None:
    inputs_np = np.array(
        [
            [
                [
                    [1.0, 2.0, 5.0, 4.0],
                    [3.0, 9.0, 6.0, 7.0],
                    [8.0, 1.0, 0.0, 2.0],
                    [4.0, 3.0, 5.0, 10.0],
                ],
                [
                    [-1.0, -2.0, -3.0, -4.0],
                    [0.5, 0.25, 1.5, 1.25],
                    [2.5, 2.0, 3.0, 2.75],
                    [4.5, 4.0, 5.5, 5.0],
                ],
            ],
            [
                [
                    [10.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [1.0, 4.0, 2.0, 3.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 12.0, 11.0],
                    [13.0, 16.0, 15.0, 14.0],
                ],
            ],
        ],
        dtype=np.float32,
    )
    grad_out_np = np.linspace(
        -0.5,
        0.75,
        num=2 * 2 * 2 * 2,
        dtype=np.float32,
    ).reshape(2, 2, 2, 2)
    numpy_layer = MaxPool2D(kernel_size=2, stride=2, backend="numpy")
    cupy_layer = MaxPool2D(kernel_size=2, stride=2, backend=cp)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    assert is_cupy_array(outputs_cp)
    assert outputs_cp.shape == outputs_np.shape
    np.testing.assert_array_equal(to_numpy(outputs_cp), outputs_np)

    grad_input_np = numpy_layer.backward(grad_out_np)
    grad_input_cp = cupy_layer.backward(cp.asarray(grad_out_np))
    assert is_cupy_array(grad_input_cp)
    np.testing.assert_allclose(
        to_numpy(grad_input_cp),
        grad_input_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_maxpool2d_tie_backward_matches_numpy_first_max_semantics(cp) -> None:
    inputs_np = np.array(
        [[[[5.0, 5.0], [5.0, 1.0]]]],
        dtype=np.float32,
    )
    grad_out_np = np.array([[[[3.0]]]], dtype=np.float32)
    numpy_layer = MaxPool2D(kernel_size=2, stride=2, backend="numpy")
    cupy_layer = MaxPool2D(kernel_size=2, stride=2, backend=cp)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    assert is_cupy_array(outputs_cp)
    np.testing.assert_array_equal(to_numpy(outputs_cp), outputs_np)

    grad_input_np = numpy_layer.backward(grad_out_np)
    grad_input_cp = cupy_layer.backward(cp.asarray(grad_out_np))
    assert is_cupy_array(grad_input_cp)
    np.testing.assert_array_equal(to_numpy(grad_input_cp), grad_input_np)
    np.testing.assert_array_equal(
        grad_input_np,
        np.array([[[[3.0, 0.0], [0.0, 0.0]]]], dtype=np.float32),
    )


def test_flatten_forward_backward_matches_numpy(cp) -> None:
    inputs_np = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)
    grad_out_np = np.linspace(
        -1.0,
        1.0,
        num=inputs_np.size,
        dtype=np.float32,
    ).reshape(2, -1)
    numpy_layer = Flatten(backend="numpy")
    cupy_layer = Flatten(backend=cp)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    assert is_cupy_array(outputs_cp)
    assert outputs_cp.shape == (2, 24)
    np.testing.assert_array_equal(to_numpy(outputs_cp), outputs_np)

    grad_input_np = numpy_layer.backward(grad_out_np)
    grad_input_cp = cupy_layer.backward(cp.asarray(grad_out_np))
    assert is_cupy_array(grad_input_cp)
    assert grad_input_cp.shape == inputs_np.shape
    np.testing.assert_array_equal(to_numpy(grad_input_cp), grad_input_np)


def test_linear_forward_backward_matches_numpy(cp) -> None:
    inputs_np = np.array(
        [[-1.0, 0.5, 2.0, -0.25], [1.5, -2.0, 0.75, 3.0]],
        dtype=np.float32,
    )
    weights_np = np.array(
        [
            [0.2, -0.1, 0.5, 0.3],
            [-0.4, 0.25, 0.1, -0.2],
            [0.7, -0.6, 0.05, 0.15],
        ],
        dtype=np.float32,
    )
    bias_np = np.array([0.05, -0.1, 0.2], dtype=np.float32)
    grad_out_np = np.array(
        [[1.0, -0.5, 0.25], [-1.5, 0.75, 2.0]],
        dtype=np.float32,
    )
    numpy_layer = Linear(in_features=4, out_features=3, backend="numpy")
    cupy_layer = Linear(in_features=4, out_features=3, backend=cp)
    numpy_layer.weights[...] = weights_np
    numpy_layer.bias[...] = bias_np
    cupy_layer.weights[...] = cp.asarray(weights_np)
    cupy_layer.bias[...] = cp.asarray(bias_np)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    assert is_cupy_array(outputs_cp)
    np.testing.assert_allclose(
        to_numpy(outputs_cp),
        outputs_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    dx_np = numpy_layer.backward(grad_out_np)
    dx_cp = cupy_layer.backward(cp.asarray(grad_out_np))
    assert is_cupy_array(dx_cp)
    assert is_cupy_array(cupy_layer.grad_weight)
    assert is_cupy_array(cupy_layer.grad_bias)
    np.testing.assert_allclose(
        to_numpy(dx_cp),
        dx_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        to_numpy(cupy_layer.grad_weight),
        numpy_layer.grad_weight,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        to_numpy(cupy_layer.grad_bias),
        numpy_layer.grad_bias,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_softmax_cross_entropy_forward_backward_matches_numpy(cp) -> None:
    logits_np = np.array(
        [[3.5, -1.25, 0.0, 2.0], [-2.0, 4.0, 1.5, 0.25]],
        dtype=np.float32,
    )
    labels_np = np.array([3, 1], dtype=np.int64)
    numpy_loss = SoftmaxCrossEntropyLoss(backend="numpy")
    cupy_loss = SoftmaxCrossEntropyLoss(backend=cp)

    loss_np = numpy_loss.forward(logits_np, labels_np)
    loss_cp = cupy_loss.forward(cp.asarray(logits_np), cp.asarray(labels_np))
    assert loss_cp == pytest.approx(loss_np, rel=LOSS_RTOL, abs=LOSS_ATOL)

    grad_logits_np = numpy_loss.backward()
    grad_logits_cp = cupy_loss.backward()
    assert is_cupy_array(grad_logits_cp)
    np.testing.assert_allclose(
        to_numpy(grad_logits_cp),
        grad_logits_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
