"""Fast Gradient Sign Method attack."""

from __future__ import annotations

import math

import numpy as np


def fgsm_attack(
    images: np.ndarray,
    grad_input: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Apply an untargeted FGSM perturbation to NCHW images."""
    if images.ndim != 4 or any(size == 0 for size in images.shape):
        raise ValueError("fgsm_attack expects non-empty NCHW images.")
    if grad_input.shape != images.shape:
        raise ValueError("Input gradients must match the image shape.")
    if not np.isfinite(images).all():
        raise ValueError("Images must contain only finite values.")
    if not np.isfinite(grad_input).all():
        raise ValueError("Input gradients must contain only finite values.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon < 0
    ):
        raise ValueError("epsilon must be a non-negative finite number.")

    adversarial_images = images + epsilon * np.sign(grad_input)
    return np.clip(adversarial_images, 0.0, 1.0)
