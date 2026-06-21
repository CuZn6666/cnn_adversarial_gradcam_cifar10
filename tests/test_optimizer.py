import numpy as np
import pytest

from src.optimizers import SGD


def test_sgd_matches_hand_computed_update() -> None:
    parameter = np.array([1.0, -2.0], dtype=np.float32)
    gradient = np.array([0.5, -1.5], dtype=np.float32)

    SGD(learning_rate=0.1).step([(parameter, gradient)])

    expected = np.array([0.95, -1.85], dtype=np.float32)
    np.testing.assert_allclose(parameter, expected)


def test_sgd_updates_multiple_parameter_arrays() -> None:
    first_parameter = np.array([1.0, 2.0], dtype=np.float32)
    first_gradient = np.array([0.25, -0.5], dtype=np.float32)
    second_parameter = np.array([[3.0], [-4.0]], dtype=np.float32)
    second_gradient = np.array([[1.0], [2.0]], dtype=np.float32)

    SGD(learning_rate=0.2).step(
        [
            (first_parameter, first_gradient),
            (second_parameter, second_gradient),
        ]
    )

    np.testing.assert_allclose(
        first_parameter,
        np.array([0.95, 2.1], dtype=np.float32),
    )
    np.testing.assert_allclose(
        second_parameter,
        np.array([[2.8], [-4.4]], dtype=np.float32),
    )


def test_sgd_accepts_named_parameter_gradient_triples() -> None:
    parameter = np.array([1.0, -2.0], dtype=np.float32)
    gradient = np.array([0.5, -1.5], dtype=np.float32)

    SGD(learning_rate=0.1).step(
        [("layer.weights", parameter, gradient)]
    )

    np.testing.assert_allclose(
        parameter,
        np.array([0.95, -1.85], dtype=np.float32),
    )


def test_sgd_preserves_parameter_shapes() -> None:
    parameter = np.arange(6, dtype=np.float32).reshape(2, 3)
    gradient = np.ones_like(parameter)
    original_shape = parameter.shape

    SGD(learning_rate=0.01).step([(parameter, gradient)])

    assert parameter.shape == original_shape


@pytest.mark.parametrize(
    "learning_rate",
    [0.0, -0.1, np.inf, np.nan],
)
def test_sgd_rejects_invalid_learning_rate(learning_rate: float) -> None:
    with pytest.raises(
        ValueError,
        match="learning_rate must be a positive finite number",
    ):
        SGD(learning_rate=learning_rate)


def test_sgd_rejects_parameter_gradient_shape_mismatch() -> None:
    parameter = np.ones((2, 2), dtype=np.float32)
    gradient = np.ones((2,), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Parameter and gradient shapes must match",
    ):
        SGD(learning_rate=0.1).step([(parameter, gradient)])


@pytest.mark.parametrize("non_finite_value", [np.inf, np.nan])
def test_sgd_rejects_non_finite_parameters(
    non_finite_value: float,
) -> None:
    parameter = np.array([1.0, non_finite_value], dtype=np.float32)
    gradient = np.ones_like(parameter)

    with pytest.raises(
        ValueError,
        match="Parameters must contain only finite values",
    ):
        SGD(learning_rate=0.1).step([(parameter, gradient)])


@pytest.mark.parametrize("non_finite_value", [np.inf, np.nan])
def test_sgd_rejects_non_finite_gradients(
    non_finite_value: float,
) -> None:
    parameter = np.ones(2, dtype=np.float32)
    gradient = np.array([1.0, non_finite_value], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Gradients must contain only finite values",
    ):
        SGD(learning_rate=0.1).step([(parameter, gradient)])
