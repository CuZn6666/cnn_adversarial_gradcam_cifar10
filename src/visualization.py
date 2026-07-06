"""Visualization helpers for small qualitative adversarial examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _normalize_map(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    if maximum == 0.0:
        return np.zeros_like(values)
    return values / maximum


def _single_gradient_map(
    gradient_map: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    if gradient_map.ndim == 3 and gradient_map.shape[0] == 1:
        gradient_map = gradient_map[0]
    if gradient_map.ndim != 2 or gradient_map.shape != spatial_shape:
        raise ValueError(
            "gradient_map must have shape (H, W) or (1, H, W) matching "
            "the images."
        )
    if not np.isfinite(gradient_map).all():
        raise ValueError("gradient_map must contain only finite values.")
    return _normalize_map(np.abs(gradient_map))


def save_fgsm_visualizations(
    clean_images: np.ndarray,
    adversarial_images: np.ndarray,
    gradient_map: np.ndarray,
    output_dir: str | Path,
    prefix: str = "fgsm_example",
) -> dict[str, Path]:
    """Save four deterministic PNG files for one qualitative FGSM example."""
    if clean_images.ndim != 4 or clean_images.shape[0] != 1:
        raise ValueError("clean_images must have shape (1, C, H, W).")
    if clean_images.shape[1] not in (1, 3):
        raise ValueError("clean_images must have one or three channels.")
    if adversarial_images.shape != clean_images.shape:
        raise ValueError("adversarial_images must match clean_images shape.")
    if not np.isfinite(clean_images).all():
        raise ValueError("clean_images must contain only finite values.")
    if not np.isfinite(adversarial_images).all():
        raise ValueError(
            "adversarial_images must contain only finite values."
        )
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string.")

    height, width = clean_images.shape[2:]
    normalized_gradient = _single_gradient_map(
        gradient_map,
        (height, width),
    )
    perturbation = np.mean(
        np.abs(adversarial_images[0] - clean_images[0]),
        axis=0,
    )
    normalized_perturbation = _normalize_map(perturbation)

    clean_hwc = np.moveaxis(clean_images[0], 0, -1)
    adversarial_hwc = np.moveaxis(adversarial_images[0], 0, -1)
    if clean_hwc.shape[-1] == 1:
        clean_hwc = clean_hwc[..., 0]
        adversarial_hwc = adversarial_hwc[..., 0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": output_path / f"{prefix}_clean.png",
        "adversarial": output_path / f"{prefix}_adversarial.png",
        "gradient": output_path / f"{prefix}_input_gradient.png",
        "perturbation": output_path / f"{prefix}_perturbation.png",
    }

    plt.imsave(paths["clean"], np.clip(clean_hwc, 0.0, 1.0))
    plt.imsave(
        paths["adversarial"],
        np.clip(adversarial_hwc, 0.0, 1.0),
    )
    plt.imsave(
        paths["gradient"],
        normalized_gradient,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        paths["perturbation"],
        normalized_perturbation,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    return paths


def _load_image_file(path: str | Path, label: str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"{label} image file does not exist: {image_path}")
    image = mpimg.imread(image_path)
    if not np.isfinite(image).all():
        raise ValueError(f"{label} image must contain only finite values.")
    return image


def save_combined_fgsm_figure(
    clean_path: str | Path,
    adversarial_path: str | Path,
    gradient_path: str | Path,
    perturbation_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Combine existing qualitative FGSM artifact PNGs into a 2x2 figure."""
    images = [
        _load_image_file(clean_path, "clean"),
        _load_image_file(adversarial_path, "adversarial"),
        _load_image_file(gradient_path, "input-gradient"),
        _load_image_file(perturbation_path, "perturbation"),
    ]
    titles = [
        "(a) Clean Input",
        "(b) Adversarial Input",
        "(c) Input Gradient",
        "(d) Perturbation",
    ]

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    figure.suptitle("FGSM Adversarial Example Analysis", fontsize=15)
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return figure_path
