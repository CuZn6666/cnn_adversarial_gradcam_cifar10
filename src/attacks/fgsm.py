"""Fast Gradient Sign Method attack."""

from __future__ import annotations

import math

import numpy as np

from src.backend import ensure_same_backend, isfinite_all


def fgsm_attack(
    images: np.ndarray,
    grad_input: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Apply an untargeted FGSM perturbation to NCHW images."""
    xp = ensure_same_backend(images, grad_input)
    if images.ndim != 4 or any(size == 0 for size in images.shape):
        raise ValueError("fgsm_attack expects non-empty NCHW images.")
    if grad_input.shape != images.shape:
        raise ValueError("Input gradients must match the image shape.")
    if not isfinite_all(images):
        raise ValueError("Images must contain only finite values.")
    if not isfinite_all(grad_input):
        raise ValueError("Input gradients must contain only finite values.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon < 0
    ):
        raise ValueError("epsilon must be a non-negative finite number.")

    adversarial_images = images + epsilon * xp.sign(grad_input)
    return xp.clip(adversarial_images, 0.0, 1.0)
