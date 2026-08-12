"""Projected gradient descent attacks."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.backend import (
    ensure_backend_array,
    get_array_module,
    isfinite_all,
    to_python_bool,
)
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


def pgd_linf_attack(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Generate untargeted L-infinity PGD adversarial examples.

    ``steps=0`` is an explicit no-op and returns a copy of ``images`` without
    applying random initialization or computing input gradients.
    """
    _validate_pgd_config(
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        seed=seed,
    )

    xp = _validate_pgd_inputs(model, images, labels)

    if steps == 0 or epsilon == 0.0:
        return images.copy()

    if random_start:
        perturbation = _uniform_perturbation(
            xp,
            images.shape,
            images.dtype,
            epsilon,
            seed,
        )
        initial_adversarial_images = images + perturbation
    else:
        initial_adversarial_images = images.copy()

    return _pgd_linf_attack_from_initial(
        model,
        loss_function,
        images,
        labels,
        initial_adversarial_images=initial_adversarial_images,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
    )


def _pgd_linf_attack_from_initial(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    initial_adversarial_images: np.ndarray,
    epsilon: float,
    alpha: float,
    steps: int,
) -> np.ndarray:
    """Run L-infinity PGD steps from a caller-provided initial state."""
    _validate_pgd_config(
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=False,
        seed=None,
    )
    xp = _validate_pgd_inputs(model, images, labels)
    ensure_backend_array(
        initial_adversarial_images,
        xp,
        name="initial_adversarial_images",
    )
    if initial_adversarial_images.shape != images.shape:
        raise ValueError(
            "initial_adversarial_images must match the image shape."
        )
    if not isfinite_all(initial_adversarial_images):
        raise ValueError(
            "initial_adversarial_images must contain only finite values."
        )

    if steps == 0 or epsilon == 0.0:
        return images.copy()

    lower_bound = xp.maximum(images - epsilon, 0.0)
    upper_bound = xp.minimum(images + epsilon, 1.0)
    adversarial_images = _project_linf(
        initial_adversarial_images,
        lower_bound,
        upper_bound,
        xp,
    )

    for _ in range(steps):
        grad_input = compute_input_gradient(
            model,
            loss_function,
            adversarial_images,
            labels,
        )
        adversarial_images = adversarial_images + alpha * xp.sign(grad_input)
        adversarial_images = _project_linf(
            adversarial_images,
            lower_bound,
            upper_bound,
            xp,
        )

    return adversarial_images


def _validate_pgd_inputs(
    model: CompactCNN,
    images: np.ndarray,
    labels: np.ndarray,
) -> Any:
    backend = getattr(model, "xp", get_array_module(images))
    xp = ensure_backend_array(images, backend, name="images")
    ensure_backend_array(labels, backend, name="labels")
    _validate_images(images, xp)
    _validate_labels(labels, images)
    return xp


def _validate_pgd_config(
    *,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool,
    seed: int | None,
) -> None:
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(float(epsilon))
        or epsilon < 0
    ):
        raise ValueError("epsilon must be a non-negative finite number.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(float(alpha))
    ):
        raise ValueError("alpha must be a finite number.")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")
    if alpha < 0 or (steps > 0 and alpha <= 0):
        raise ValueError(
            "alpha must be positive when steps > 0 and non-negative otherwise."
        )
    if not isinstance(random_start, bool):
        raise ValueError("random_start must be a boolean.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("seed must be None or a non-negative integer.")


def _validate_images(images: np.ndarray, xp: Any) -> None:
    if images.ndim != 4 or any(size == 0 for size in images.shape):
        raise ValueError("pgd_linf_attack expects non-empty NCHW images.")
    if not isfinite_all(images):
        raise ValueError("Images must contain only finite values.")
    if to_python_bool(xp.any(images < 0.0)) or to_python_bool(
        xp.any(images > 1.0)
    ):
        raise ValueError("Images must be in the valid [0, 1] range.")


def _validate_labels(labels: np.ndarray, images: np.ndarray) -> None:
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(
            "pgd_linf_attack expects labels with shape (batch_size,)."
        )


def _uniform_perturbation(
    xp: Any,
    shape: tuple[int, ...],
    dtype: Any,
    epsilon: float,
    seed: int | None,
) -> np.ndarray:
    if hasattr(xp.random, "default_rng"):
        rng = xp.random.default_rng(seed)
    else:  # pragma: no cover - CuPy compatibility guard.
        rng = xp.random.RandomState(seed)
    perturbation = rng.uniform(-epsilon, epsilon, size=shape)
    return perturbation.astype(dtype, copy=False)


def _project_linf(
    values: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    xp: Any,
) -> np.ndarray:
    return xp.minimum(xp.maximum(values, lower_bound), upper_bound)
