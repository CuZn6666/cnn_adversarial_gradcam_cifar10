"""Core Grad-CAM computation for CompactCNN."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models import CompactCNN


def _normalize_per_sample(heatmaps: np.ndarray) -> np.ndarray:
    """Normalize each heatmap independently to [0, 1]."""
    if heatmaps.ndim != 3:
        raise ValueError("Grad-CAM heatmaps must have shape (N, H, W).")
    if any(size == 0 for size in heatmaps.shape):
        raise ValueError("Grad-CAM heatmaps must be non-empty.")
    if not np.isfinite(heatmaps).all():
        raise ValueError("Grad-CAM heatmaps must contain only finite values.")

    normalized = np.zeros_like(heatmaps, dtype=np.float32)
    minima = heatmaps.min(axis=(1, 2), keepdims=True)
    maxima = heatmaps.max(axis=(1, 2), keepdims=True)
    ranges = maxima - minima
    np.divide(
        heatmaps - minima,
        ranges,
        out=normalized,
        where=ranges > 0,
    )
    return normalized


def _validated_target_classes(
    target_classes: np.ndarray | None,
    logits: np.ndarray,
) -> np.ndarray:
    batch_size, num_classes = logits.shape
    if target_classes is None:
        return np.argmax(logits, axis=1)

    targets = np.asarray(target_classes)
    if targets.shape != (batch_size,):
        raise ValueError("target_classes must have shape (N,).")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("target_classes must contain integer class IDs.")
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("target_classes contains class IDs outside the valid range.")
    return targets.astype(np.int64, copy=False)


def _snapshot_gradient_buffer(buffer: Any) -> tuple[Any, np.ndarray | None]:
    if isinstance(buffer, np.ndarray):
        return buffer, buffer.copy()
    return buffer, None


def _restore_gradient_buffer(
    owner: object,
    attribute_name: str,
    original_buffer: Any,
    original_values: np.ndarray | None,
) -> None:
    if isinstance(original_buffer, np.ndarray):
        if original_values is None:
            raise RuntimeError("Missing gradient-buffer snapshot.")
        original_buffer[...] = original_values
    setattr(owner, attribute_name, original_buffer)


def compute_gradcam(
    model: CompactCNN,
    images: np.ndarray,
    target_classes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalized clean-image Grad-CAM heatmaps for CompactCNN.

    The fixed target activation is the `relu2` output before `pool2`, with
    native CompactCNN shape `(N, 16, 16, 16)` for CIFAR-10 inputs.
    """
    logits = model.forward(images)
    activations = model.gradcam_activation
    targets = _validated_target_classes(target_classes, logits)

    grad_logits = np.zeros_like(logits)
    grad_logits[np.arange(logits.shape[0]), targets] = 1.0

    original_grad_weight, original_grad_weight_values = _snapshot_gradient_buffer(
        model.classifier.grad_weight
    )
    original_grad_bias, original_grad_bias_values = _snapshot_gradient_buffer(
        model.classifier.grad_bias
    )

    try:
        gradients = model.classifier.backward(grad_logits)
        gradients = model.flatten.backward(gradients)
        activation_gradients = model.pool2.backward(gradients)
    finally:
        _restore_gradient_buffer(
            model.classifier,
            "grad_weight",
            original_grad_weight,
            original_grad_weight_values,
        )
        _restore_gradient_buffer(
            model.classifier,
            "grad_bias",
            original_grad_bias,
            original_grad_bias_values,
        )

    if activations.ndim != 4 or activation_gradients.ndim != 4:
        raise RuntimeError("Grad-CAM activations and gradients must be 4D tensors.")
    if activations.shape != activation_gradients.shape:
        raise RuntimeError("Grad-CAM activation and gradient shapes must match.")
    if not np.isfinite(activations).all():
        raise RuntimeError("Grad-CAM activations contain non-finite values.")
    if not np.isfinite(activation_gradients).all():
        raise RuntimeError("Grad-CAM gradients contain non-finite values.")

    weights = activation_gradients.mean(axis=(2, 3))
    cam = np.sum(weights[:, :, None, None] * activations, axis=1)
    cam = np.maximum(cam, 0.0)
    return _normalize_per_sample(cam)
