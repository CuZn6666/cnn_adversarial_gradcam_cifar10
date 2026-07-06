import numpy as np
import pytest

from src.gradcam import _normalize_per_sample, compute_gradcam
from src.models import CompactCNN


def _deterministic_images(batch_size: int = 2) -> np.ndarray:
    return np.random.default_rng(17).random(
        (batch_size, 3, 32, 32),
        dtype=np.float32,
    )


def _model_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


def test_compute_gradcam_returns_expected_shape_and_finite_range() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)

    heatmaps = compute_gradcam(model, images)

    assert heatmaps.shape == (2, 16, 16)
    assert np.isfinite(heatmaps).all()
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0


def test_compute_gradcam_default_target_uses_predicted_class() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)
    logits = model.forward(images)
    predicted_classes = np.argmax(logits, axis=1)

    default_heatmaps = compute_gradcam(model, images)
    explicit_heatmaps = compute_gradcam(model, images, predicted_classes)

    np.testing.assert_array_equal(default_heatmaps, explicit_heatmaps)


def test_compute_gradcam_accepts_explicit_target_classes() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)
    target_classes = np.array([0, 9], dtype=np.int64)

    heatmaps = compute_gradcam(model, images, target_classes)

    assert heatmaps.shape == (2, 16, 16)
    assert np.isfinite(heatmaps).all()
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0


def test_compute_gradcam_is_deterministic_for_fixed_inputs_and_targets() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)
    target_classes = np.array([1, 4], dtype=np.int64)

    first_heatmaps = compute_gradcam(model, images, target_classes)
    second_heatmaps = compute_gradcam(model, images, target_classes)

    np.testing.assert_array_equal(first_heatmaps, second_heatmaps)


def test_compute_gradcam_does_not_change_parameter_values() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)
    target_classes = np.array([1, 4], dtype=np.int64)
    parameters_before = [
        parameter.copy() for parameter in _model_parameters(model)
    ]

    compute_gradcam(model, images, target_classes)

    for parameter, parameter_before in zip(
        _model_parameters(model),
        parameters_before,
        strict=True,
    ):
        np.testing.assert_array_equal(parameter, parameter_before)


def test_compute_gradcam_restores_classifier_gradient_buffers() -> None:
    model = CompactCNN(seed=42)
    model.classifier.grad_weight = np.full_like(
        model.classifier.grad_weight,
        3.0,
    )
    model.classifier.grad_bias = np.full_like(
        model.classifier.grad_bias,
        -2.0,
    )
    grad_weight_before = model.classifier.grad_weight
    grad_bias_before = model.classifier.grad_bias
    grad_weight_values_before = grad_weight_before.copy()
    grad_bias_values_before = grad_bias_before.copy()

    compute_gradcam(
        model,
        _deterministic_images(batch_size=2),
        np.array([1, 4], dtype=np.int64),
    )

    assert model.classifier.grad_weight is grad_weight_before
    assert model.classifier.grad_bias is grad_bias_before
    np.testing.assert_array_equal(
        model.classifier.grad_weight,
        grad_weight_values_before,
    )
    np.testing.assert_array_equal(
        model.classifier.grad_bias,
        grad_bias_values_before,
    )


def test_compute_gradcam_restores_none_classifier_gradient_buffers() -> None:
    model = CompactCNN(seed=42)
    model.classifier.grad_weight = None  # type: ignore[assignment]
    model.classifier.grad_bias = None  # type: ignore[assignment]

    compute_gradcam(
        model,
        _deterministic_images(batch_size=1),
        np.array([2], dtype=np.int64),
    )

    assert model.classifier.grad_weight is None
    assert model.classifier.grad_bias is None


def test_normalize_per_sample_returns_zero_for_constant_heatmaps() -> None:
    heatmaps = np.full((2, 4, 5), 7.0, dtype=np.float32)

    normalized = _normalize_per_sample(heatmaps)

    assert normalized.shape == heatmaps.shape
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized, np.zeros_like(heatmaps))


def test_compute_gradcam_rejects_invalid_target_shape() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)

    with pytest.raises(ValueError, match=r"target_classes must have shape"):
        compute_gradcam(model, images, np.array([[1], [2]], dtype=np.int64))

    with pytest.raises(ValueError, match=r"target_classes must have shape"):
        compute_gradcam(model, images, np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    "target_classes",
    [
        np.array([-1, 2], dtype=np.int64),
        np.array([0, 10], dtype=np.int64),
    ],
)
def test_compute_gradcam_rejects_invalid_target_range(
    target_classes: np.ndarray,
) -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)

    with pytest.raises(ValueError, match="outside the valid range"):
        compute_gradcam(model, images, target_classes)


def test_compute_gradcam_rejects_non_integer_targets() -> None:
    model = CompactCNN(seed=42)
    images = _deterministic_images(batch_size=2)

    with pytest.raises(ValueError, match="integer class IDs"):
        compute_gradcam(model, images, np.array([1.0, 2.0]))


def test_gradcam_activation_requires_forward_and_returns_copy() -> None:
    model = CompactCNN(seed=42)

    with pytest.raises(RuntimeError, match="requires a preceding forward call"):
        _ = model.gradcam_activation

    model.forward(_deterministic_images(batch_size=1))
    activation = model.gradcam_activation
    activation[...] = -1.0

    assert not np.array_equal(model.gradcam_activation, activation)
