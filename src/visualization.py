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


def _nchw_single_to_display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape (1, C, H, W).")
    if image.shape[1] not in (1, 3):
        raise ValueError("image must have one or three channels.")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values.")

    display_image = np.moveaxis(image[0], 0, -1)
    if display_image.shape[-1] == 1:
        display_image = display_image[..., 0]
    return np.clip(display_image, 0.0, 1.0)


def _normalized_perturbation_map(
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
) -> np.ndarray:
    if adversarial_image.shape != clean_image.shape:
        raise ValueError("adversarial_image must match clean_image shape.")
    perturbation = np.mean(
        np.abs(adversarial_image[0] - clean_image[0]),
        axis=0,
    )
    return _normalize_map(perturbation)


def save_fgsm_qualitative_comparison(
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
    gradient_map: np.ndarray,
    output_path: str | Path,
    true_label: str,
    clean_prediction: str,
    adversarial_prediction: str,
    epsilon_label: str,
) -> Path:
    """Save a portfolio FGSM comparison: clean, gradient, perturbation, adv."""
    clean_display = _nchw_single_to_display_image(clean_image)
    adversarial_display = _nchw_single_to_display_image(adversarial_image)
    height, width = clean_image.shape[2:]
    normalized_gradient = _single_gradient_map(gradient_map, (height, width))
    normalized_perturbation = _normalized_perturbation_map(
        clean_image,
        adversarial_image,
    )

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    figure.suptitle(
        f"FGSM Qualitative Analysis | True: {true_label} | ε = {epsilon_label}",
        fontsize=14,
    )
    panels = [
        (clean_display, "Clean Image\nPred: " + clean_prediction, None),
        (normalized_gradient, "Input Gradient", "magma"),
        (normalized_perturbation, "Perturbation\nvisualized magnitude", "gray"),
        (
            adversarial_display,
            "Adversarial Image\nPred: " + adversarial_prediction,
            None,
        ),
    ]
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return figure_path


def save_fgsm_epsilon_progression(
    images_by_epsilon: list[np.ndarray],
    epsilon_labels: list[str] | tuple[str, ...],
    predictions: list[str] | tuple[str, ...],
    output_path: str | Path,
    true_label: str,
) -> Path:
    """Save one clean source image under independently generated epsilons."""
    if not images_by_epsilon:
        raise ValueError("images_by_epsilon must not be empty.")
    if len(images_by_epsilon) != len(epsilon_labels):
        raise ValueError("epsilon_labels length must match images_by_epsilon.")
    if len(predictions) != len(images_by_epsilon):
        raise ValueError("predictions length must match images_by_epsilon.")

    display_images = [
        _nchw_single_to_display_image(image)
        for image in images_by_epsilon
    ]

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    panel_count = len(display_images)
    figure, axes = plt.subplots(1, panel_count, figsize=(2.4 * panel_count, 3.2))
    if panel_count == 1:
        axes = [axes]
    figure.suptitle(
        f"FGSM Epsilon Progression | True: {true_label}",
        fontsize=14,
    )
    for axis, image, epsilon_label, prediction in zip(
        axes,
        display_images,
        epsilon_labels,
        predictions,
    ):
        axis.imshow(image)
        axis.set_title(f"ε = {epsilon_label}\nPred: {prediction}", fontsize=10)
        axis.axis("off")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return figure_path
