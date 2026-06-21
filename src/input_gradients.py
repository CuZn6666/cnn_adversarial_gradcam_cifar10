"""Input-gradient helpers for adversarial analysis."""

from __future__ import annotations

import numpy as np

from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def compute_input_gradient(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute the loss gradient with respect to input images."""
    logits = model.forward(images)
    loss_function.forward(logits, labels)
    grad_logits = loss_function.backward()
    grad_input = model.backward(grad_logits)

    if grad_input.shape != images.shape:
        raise RuntimeError("Input gradient shape does not match input images.")
    if not np.isfinite(grad_input).all():
        raise RuntimeError("Input gradient contains non-finite values.")

    return grad_input
