"""Input-gradient helpers for adversarial analysis."""

from __future__ import annotations

import numpy as np

from src.backend import ensure_backend_array, get_array_module, isfinite_all
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def input_gradient_map(
    grad_input: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert NCHW input gradients to per-image spatial maps."""
    xp = get_array_module(grad_input)
    if grad_input.ndim != 4 or any(size == 0 for size in grad_input.shape):
        raise ValueError(
            "input_gradient_map expects a non-empty NCHW gradient tensor."
        )
    if not isfinite_all(grad_input):
        raise ValueError("Input gradients must contain only finite values.")

    gradient_map = xp.mean(xp.abs(grad_input), axis=1)
    if not normalize:
        return gradient_map

    maxima = gradient_map.max(axis=(1, 2), keepdims=True)
    normalized_map = xp.zeros_like(gradient_map)
    xp.divide(
        gradient_map,
        maxima,
        out=normalized_map,
        where=maxima > 0,
    )
    return normalized_map


def compute_input_gradient(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute the loss gradient with respect to input images."""
    backend = getattr(model, "xp", get_array_module(images))
    ensure_backend_array(images, backend, name="images")
    ensure_backend_array(labels, backend, name="labels")
    logits = model.forward(images)
    loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()
    grad_input = model.backward(grad_logits)

    if grad_input.shape != images.shape:
        raise RuntimeError("Input gradient shape does not match input images.")
    if not isfinite_all(grad_input):
        raise RuntimeError("Input gradient contains non-finite values.")

    return grad_input
