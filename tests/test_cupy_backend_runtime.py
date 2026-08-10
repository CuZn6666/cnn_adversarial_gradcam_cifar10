import numpy as np
import pytest

from src.backend import (
    isfinite_all,
    sliding_window_view,
    to_backend,
    to_numpy,
    to_python_bool,
    to_python_float,
    to_python_int,
)
from src.layers import Conv2D


pytestmark = pytest.mark.requires_cupy


FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6


def test_cupy_runtime_reports_usable_device(cupy_runtime) -> None:
    assert cupy_runtime.version
    assert cupy_runtime.cuda_runtime_version > 0
    assert cupy_runtime.device_count > 0
    assert cupy_runtime.gpu_name
    assert cupy_runtime.simple_computation == pytest.approx(14.0)


def test_cupy_backend_array_primitives_match_numpy(cp) -> None:
    values_np = np.array(
        [[-2.0, -0.5, 0.0], [1.5, 2.0, 4.0]],
        dtype=np.float32,
    )
    values_cp = to_backend(values_np, cp)

    np.testing.assert_array_equal(to_numpy(values_cp), values_np)
    np.testing.assert_array_equal(
        to_numpy(cp.zeros(values_np.shape, dtype=cp.float32)),
        np.zeros(values_np.shape, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        to_numpy(cp.zeros_like(values_cp)),
        np.zeros_like(values_np),
    )
    np.testing.assert_allclose(
        to_numpy(cp.maximum(values_cp, 0.0)),
        np.maximum(values_np, 0.0),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        to_numpy(values_cp.max(axis=1)),
        values_np.max(axis=1),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_array_equal(
        to_numpy(cp.argmax(values_cp, axis=1)),
        np.argmax(values_np, axis=1),
    )
    assert to_python_float(cp.sum(values_cp)) == pytest.approx(
        float(np.sum(values_np))
    )
    assert to_python_float(cp.mean(values_cp)) == pytest.approx(
        float(np.mean(values_np))
    )
    np.testing.assert_allclose(
        to_numpy(cp.abs(values_cp)),
        np.abs(values_np),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    positive_np = np.abs(values_np) + 0.25
    positive_cp = cp.asarray(positive_np)
    np.testing.assert_allclose(
        to_numpy(cp.log(cp.exp(positive_cp))),
        np.log(np.exp(positive_np)),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    numerator_cp = cp.asarray([1.0, 2.0, 0.0], dtype=cp.float32)
    denominator_cp = cp.asarray([2.0, 4.0, 0.0], dtype=cp.float32)
    divided_cp = cp.zeros_like(numerator_cp)
    cp.divide(
        numerator_cp,
        denominator_cp,
        out=divided_cp,
        where=denominator_cp > 0,
    )
    np.testing.assert_allclose(
        to_numpy(divided_cp),
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    np.testing.assert_allclose(
        to_numpy(cp.clip(values_cp, 0.0, 1.0)),
        np.clip(values_np, 0.0, 1.0),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_array_equal(
        to_numpy(cp.sign(values_cp)),
        np.sign(values_np),
    )
    assert isfinite_all(values_cp)
    assert not isfinite_all(cp.asarray([1.0, cp.inf], dtype=cp.float32))
    assert to_python_bool(cp.asarray(True))
    assert to_python_int(cp.asarray(3, dtype=cp.int64)) == 3


def test_cupy_matmul_and_einsum_optimize_match_numpy(cp) -> None:
    left_np = np.arange(6, dtype=np.float32).reshape(2, 3) / 5.0
    right_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    left_cp = cp.asarray(left_np)
    right_cp = cp.asarray(right_np)

    np.testing.assert_allclose(
        to_numpy(left_cp @ right_cp),
        left_np @ right_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        to_numpy(cp.einsum("ij,jk->ik", left_cp, right_cp, optimize=True)),
        np.einsum("ij,jk->ik", left_np, right_np, optimize=True),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_cupy_add_at_matches_numpy_with_repeated_indices(cp) -> None:
    target_np = np.zeros((2, 3), dtype=np.float32)
    rows_np = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    columns_np = np.array([1, 1, 0, 0, 2], dtype=np.int64)
    updates_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    np.add.at(target_np, (rows_np, columns_np), updates_np)

    target_cp = cp.zeros((2, 3), dtype=cp.float32)
    cp.add.at(
        target_cp,
        (cp.asarray(rows_np), cp.asarray(columns_np)),
        cp.asarray(updates_np),
    )

    np.testing.assert_allclose(
        to_numpy(target_cp),
        target_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_cupy_sliding_window_view_wrapper_matches_numpy(cp) -> None:
    inputs_np = np.arange(2 * 1 * 4 * 5, dtype=np.float32).reshape(2, 1, 4, 5)
    inputs_cp = cp.asarray(inputs_np)

    windows_np = np.lib.stride_tricks.sliding_window_view(
        inputs_np,
        (2, 3),
        axis=(2, 3),
    )
    windows_cp = sliding_window_view(
        inputs_cp,
        (2, 3),
        axis=(2, 3),
    )

    np.testing.assert_array_equal(to_numpy(windows_cp), windows_np)


def test_cupy_conv2d_forward_backward_matches_numpy(cp) -> None:
    inputs_np = np.linspace(
        -0.3,
        0.7,
        num=2 * 2 * 4 * 5,
        dtype=np.float32,
    ).reshape(2, 2, 4, 5)
    weights_np = np.linspace(
        -0.2,
        0.25,
        num=3 * 2 * 3 * 3,
        dtype=np.float32,
    ).reshape(3, 2, 3, 3)
    bias_np = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    grad_out_np = np.linspace(
        -0.4,
        0.6,
        num=2 * 3 * 4 * 5,
        dtype=np.float32,
    ).reshape(2, 3, 4, 5)

    numpy_layer = Conv2D(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=1,
        stride=1,
        backend="numpy",
    )
    cupy_layer = Conv2D(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=1,
        stride=1,
        backend=cp,
    )
    numpy_layer.weights[...] = weights_np
    numpy_layer.bias[...] = bias_np
    cupy_layer.weights[...] = cp.asarray(weights_np)
    cupy_layer.bias[...] = cp.asarray(bias_np)

    outputs_np = numpy_layer.forward(inputs_np)
    outputs_cp = cupy_layer.forward(cp.asarray(inputs_np))
    np.testing.assert_allclose(
        to_numpy(outputs_cp),
        outputs_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )

    dx_np = numpy_layer.backward(grad_out_np)
    dx_cp = cupy_layer.backward(cp.asarray(grad_out_np))

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
