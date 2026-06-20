import numpy as np
import pytest

from src.losses import SoftmaxCrossEntropyLoss


def _reference_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def test_softmax_cross_entropy_forward_matches_hand_computed_loss() -> None:
    logits = np.array(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]],
        dtype=np.float64,
    )
    labels = np.array([2, 0], dtype=np.int64)
    probabilities = _reference_probabilities(logits)
    expected_loss = -np.log(
        probabilities[np.arange(labels.size), labels]
    ).mean()

    loss = SoftmaxCrossEntropyLoss().forward(logits, labels)

    np.testing.assert_allclose(loss, expected_loss)


def test_softmax_cross_entropy_backward_matches_expected_gradient() -> None:
    logits = np.array(
        [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]],
        dtype=np.float64,
    )
    labels = np.array([2, 0], dtype=np.int64)
    probabilities = _reference_probabilities(logits)
    expected_gradient = probabilities.copy()
    expected_gradient[np.arange(labels.size), labels] -= 1.0
    expected_gradient /= labels.size

    loss = SoftmaxCrossEntropyLoss()
    loss.forward(logits, labels)
    grad_logits = loss.backward()

    np.testing.assert_allclose(grad_logits, expected_gradient)


def test_softmax_cross_entropy_backward_requires_forward_call() -> None:
    loss = SoftmaxCrossEntropyLoss()

    with pytest.raises(
        RuntimeError,
        match=(
            r"SoftmaxCrossEntropyLoss\.backward requires a preceding "
            r"forward call\."
        ),
    ):
        loss.backward()


def test_softmax_cross_entropy_rejects_wrong_logits_shape() -> None:
    logits = np.ones((2, 3, 4), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="SoftmaxCrossEntropyLoss expects logits with shape",
    ):
        SoftmaxCrossEntropyLoss().forward(logits, labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([[0], [1]], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ],
)
def test_softmax_cross_entropy_rejects_wrong_labels_shape(
    labels: np.ndarray,
) -> None:
    logits = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="SoftmaxCrossEntropyLoss expects labels with shape",
    ):
        SoftmaxCrossEntropyLoss().forward(logits, labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([-1, 1], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    ],
)
def test_softmax_cross_entropy_rejects_out_of_range_labels(
    labels: np.ndarray,
) -> None:
    logits = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="labels are outside the valid class range",
    ):
        SoftmaxCrossEntropyLoss().forward(logits, labels)


def test_softmax_cross_entropy_is_numerically_stable_for_large_logits() -> None:
    logits = np.array(
        [[10000.0, 0.0, -10000.0], [-10000.0, 0.0, 10000.0]],
        dtype=np.float64,
    )
    labels = np.array([2, 0], dtype=np.int64)
    loss_function = SoftmaxCrossEntropyLoss()

    loss = loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()

    assert np.isfinite(loss)
    assert np.isfinite(grad_logits).all()
