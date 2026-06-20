import numpy as np
import pytest

from configs.default_config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from src.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU
from src.models import CompactCNN


def test_conv2d_forward_matches_hand_computed_output() -> None:
    layer = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        rng=np.random.default_rng(0),
    )
    layer.weights = np.array(
        [[[[1.0, 0.0], [0.0, -1.0]]]],
        dtype=np.float32,
    )
    layer.bias = np.array([0.5], dtype=np.float32)

    inputs = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=np.float32,
    )
    expected = np.full((1, 1, 2, 2), -3.5, dtype=np.float32)

    outputs = layer.forward(inputs)

    np.testing.assert_allclose(outputs, expected)


def test_relu_forward_matches_hand_computed_output() -> None:
    inputs = np.array([[-2.0, -0.5, 0.0, 3.0]], dtype=np.float32)
    expected = np.array([[0.0, 0.0, 0.0, 3.0]], dtype=np.float32)

    outputs = ReLU().forward(inputs)

    np.testing.assert_array_equal(outputs, expected)


def test_max_pool2d_forward_matches_hand_computed_output() -> None:
    inputs = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    expected = np.array(
        [[[[6.0, 8.0], [14.0, 16.0]]]],
        dtype=np.float32,
    )

    outputs = MaxPool2D(kernel_size=2, stride=2).forward(inputs)

    np.testing.assert_array_equal(outputs, expected)


def test_flatten_forward_preserves_batch_order() -> None:
    inputs = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    )

    outputs = Flatten().forward(inputs)

    np.testing.assert_array_equal(outputs, expected)


def test_linear_forward_matches_hand_computed_output() -> None:
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
    expected = np.array(
        [[1.5, 3.0, 3.0], [3.5, 7.0, 3.0]],
        dtype=np.float32,
    )

    outputs = layer.forward(inputs)

    np.testing.assert_allclose(outputs, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_compact_cnn_forward_shape_and_finite_output(batch_size: int) -> None:
    rng = np.random.default_rng(42)
    inputs = rng.random(
        (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )

    outputs = CompactCNN(seed=42).forward(inputs)

    assert outputs.shape == (batch_size, 10)
    assert np.isfinite(outputs).all()


def test_compact_cnn_fixed_seed_is_reproducible() -> None:
    inputs = np.random.default_rng(7).random(
        (2, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )

    first_outputs = CompactCNN(seed=42).forward(inputs)
    second_outputs = CompactCNN(seed=42).forward(inputs)

    np.testing.assert_array_equal(first_outputs, second_outputs)


def test_compact_cnn_rejects_invalid_input_shape() -> None:
    invalid_inputs = np.zeros(
        (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"CompactCNN expects input with shape \(N, 3, 32, 32\)\.",
    ):
        CompactCNN(seed=42).forward(invalid_inputs)
