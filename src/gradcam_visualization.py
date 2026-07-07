"""Visualization helpers for clean-vs-adversarial Grad-CAM figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_GRADCAM_CMAP = "turbo"
DEFAULT_OVERLAY_ALPHA = 0.85


def nchw_to_display_image(image: np.ndarray) -> np.ndarray:
    """Convert one NCHW image with shape (1, C, H, W) to HWC display format."""
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


def resize_heatmap_to_image(
    heatmap: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a 2D Grad-CAM heatmap to image spatial shape for display."""
    if heatmap.ndim != 2 or any(size == 0 for size in heatmap.shape):
        raise ValueError("heatmap must be a non-empty 2D array.")
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError("image_shape must be a positive (H, W) tuple.")
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap must contain only finite values.")

    heatmap_image = Image.fromarray(np.clip(heatmap, 0.0, 1.0).astype(np.float32))
    resized = heatmap_image.resize(
        (image_shape[1], image_shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    cmap: str = DEFAULT_GRADCAM_CMAP,
) -> np.ndarray:
    """Create a heatmap-weighted RGB overlay from an HWC image and heatmap."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3).")
    if heatmap.ndim != 2 or heatmap.shape != image.shape[:2]:
        raise ValueError("heatmap must have shape (H, W) matching the image.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not np.isfinite(alpha)
        or alpha < 0.0
        or alpha > 1.0
    ):
        raise ValueError("alpha must be a finite value in [0, 1].")
    if not np.isfinite(image).all() or not np.isfinite(heatmap).all():
        raise ValueError("image and heatmap must contain only finite values.")

    base = np.clip(image, 0.0, 1.0)
    clipped_heatmap = np.clip(heatmap, 0.0, 1.0)
    colored_heatmap = plt.colormaps[cmap](clipped_heatmap)[..., :3]
    alpha_map = float(alpha) * clipped_heatmap[..., None]
    return np.clip(
        (1.0 - alpha_map) * base + alpha_map * colored_heatmap,
        0.0,
        1.0,
    )


def normalized_perturbation(
    clean_image: np.ndarray,
    adversarial_image: np.ndarray,
) -> np.ndarray:
    """Return scaled |adversarial-clean| perturbation magnitude for display."""
    if adversarial_image.shape != clean_image.shape:
        raise ValueError("adversarial_image must match clean_image shape.")
    if not np.isfinite(clean_image).all() or not np.isfinite(adversarial_image).all():
        raise ValueError("images must contain only finite values.")

    perturbation = np.mean(np.abs(adversarial_image[0] - clean_image[0]), axis=0)
    maximum = float(perturbation.max())
    if maximum == 0.0:
        return np.zeros_like(perturbation)
    return perturbation / maximum


def _example_field(example: Mapping[str, Any], key: str) -> Any:
    if key not in example:
        raise ValueError(f"Grad-CAM figure example is missing {key}.")
    return example[key]


def _prepared_panels(example: Mapping[str, Any]) -> dict[str, np.ndarray]:
    clean_image = _example_field(example, "clean_image")
    adversarial_image = _example_field(example, "adversarial_image")
    clean_cam = _example_field(example, "clean_cam")
    adversarial_cam = _example_field(example, "adversarial_cam")

    clean_display = nchw_to_display_image(clean_image)
    adversarial_display = nchw_to_display_image(adversarial_image)
    clean_resized_cam = resize_heatmap_to_image(clean_cam[0], clean_display.shape[:2])
    adversarial_resized_cam = resize_heatmap_to_image(
        adversarial_cam[0],
        adversarial_display.shape[:2],
    )
    return {
        "clean_image": clean_display,
        "adversarial_image": adversarial_display,
        "clean_heatmap": clean_resized_cam,
        "adversarial_heatmap": adversarial_resized_cam,
        "clean_overlay": heatmap_overlay(clean_display, clean_resized_cam),
        "adversarial_overlay": heatmap_overlay(
            adversarial_display,
            adversarial_resized_cam,
        ),
        "perturbation": normalized_perturbation(clean_image, adversarial_image),
    }


def _row_label(example: Mapping[str, Any]) -> str:
    return (
        f"Index {int(_example_field(example, 'dataset_index'))} | "
        f"True: {_example_field(example, 'true_class')} | "
        f"Pred: {_example_field(example, 'clean_prediction_class')} → "
        f"{_example_field(example, 'adversarial_prediction_class')}"
    )


def _short_row_label(example: Mapping[str, Any]) -> str:
    return (
        f"Idx {int(_example_field(example, 'dataset_index'))} | "
        f"True: {_example_field(example, 'true_class')} | "
        f"{_example_field(example, 'clean_prediction_class')} → "
        f"{_example_field(example, 'adversarial_prediction_class')}"
    )


def _hide_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.axis("off")


def save_gradcam_hero_figure(
    examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    epsilon_label: str,
) -> Path:
    """Save the README hero figure with prediction-aligned Grad-CAM overlays."""
    if len(examples) < 1:
        raise ValueError("At least one example is required for the hero figure.")

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(examples)
    figure, axes = plt.subplots(rows, 5, figsize=(14.5, 3.0 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    figure.suptitle(
        f"Clean vs Adversarial Grad-CAM | FGSM ε = {epsilon_label}",
        fontsize=16,
    )
    column_titles = [
        "Clean Image",
        "Clean Grad-CAM Overlay",
        "Adversarial Image",
        "Adversarial Grad-CAM Overlay",
        "Perturbation\n(visualized)",
    ]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=11)

    for row, example in enumerate(examples):
        panels = _prepared_panels(example)
        images = [
            panels["clean_image"],
            panels["clean_overlay"],
            panels["adversarial_image"],
            panels["adversarial_overlay"],
            panels["perturbation"],
        ]
        cmaps = [None, None, None, None, "magma"]
        for axis, image, cmap in zip(axes[row], images, cmaps):
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            _hide_axis(axis)
        axes[row, 0].set_title(
            f"Clean Image\n{_short_row_label(example)}",
            fontsize=9,
        )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def save_gradcam_presentation_figure(
    examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    epsilon_label: str,
) -> Path:
    """Save a presentation-oriented clean/adversarial Grad-CAM figure."""
    if len(examples) < 1:
        raise ValueError(
            "At least one example is required for the presentation figure."
        )

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(examples)
    figure, axes = plt.subplots(rows, 6, figsize=(17.0, 4.2 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    figure.suptitle(
        f"Clean vs Adversarial Grad-CAM | FGSM ε = {epsilon_label}\n"
        "Heatmaps and overlays use prediction-aligned target classes",
        fontsize=16,
    )

    for row, example in enumerate(examples):
        panels = _prepared_panels(example)
        clean_target = str(_example_field(example, "clean_prediction_class"))
        adversarial_target = str(
            _example_field(example, "adversarial_prediction_class")
        )
        contents = [
            panels["clean_image"],
            panels["clean_heatmap"],
            panels["clean_overlay"],
            panels["adversarial_image"],
            panels["adversarial_heatmap"],
            panels["adversarial_overlay"],
        ]
        titles = [
            f"Clean Image\n{_short_row_label(example)}",
            f"Clean Grad-CAM Heatmap\nTarget: {clean_target}",
            f"Clean Grad-CAM Overlay\nTarget: {clean_target}",
            f"Adversarial Image\nPred: {adversarial_target}",
            f"Adversarial Grad-CAM Heatmap\nTarget: {adversarial_target}",
            f"Adversarial Grad-CAM Overlay\nTarget: {adversarial_target}",
        ]
        cmaps = [
            None,
            DEFAULT_GRADCAM_CMAP,
            None,
            None,
            DEFAULT_GRADCAM_CMAP,
            None,
        ]
        for axis, content, title, cmap in zip(axes[row], contents, titles, cmaps):
            axis.imshow(content, cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_title(title, fontsize=8.5)
            _hide_axis(axis)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), h_pad=3.0, w_pad=0.8)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path


def save_gradcam_detailed_comparison(
    example: Mapping[str, Any],
    output_path: str | Path,
    epsilon_label: str,
) -> Path:
    """Save a 2x3 clean-vs-adversarial Grad-CAM comparison."""
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    panels = _prepared_panels(example)

    figure, axes = plt.subplots(2, 3, figsize=(9.6, 6.5))
    figure.suptitle(
        f"Detailed Clean vs Adversarial Grad-CAM | ε = {epsilon_label}\n"
        f"{_row_label(example)}",
        fontsize=13,
    )
    rows = [
        (
            "Clean",
            panels["clean_image"],
            panels["clean_heatmap"],
            panels["clean_overlay"],
        ),
        (
            "Adversarial",
            panels["adversarial_image"],
            panels["adversarial_heatmap"],
            panels["adversarial_overlay"],
        ),
    ]
    column_suffixes = ["Image", "Grad-CAM Heatmap", "Grad-CAM Overlay"]
    for row_index, (prefix, image, heatmap, overlay) in enumerate(rows):
        for axis, content, suffix, cmap in zip(
            axes[row_index],
            (image, heatmap, overlay),
            column_suffixes,
            (None, DEFAULT_GRADCAM_CMAP, None),
        ):
            axis.imshow(content, cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_title(f"{prefix} {suffix}", fontsize=10)
            _hide_axis(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def save_gradcam_fixed_target_comparison(
    examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    epsilon_label: str,
) -> Path:
    """Save fixed-original-target Grad-CAM comparisons for successful attacks."""
    if len(examples) < 1:
        raise ValueError("At least one example is required for fixed-target figure.")

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(examples)
    figure, axes = plt.subplots(rows, 3, figsize=(10.8, 3.1 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    figure.suptitle(
        f"Fixed-Target Grad-CAM Comparison | FGSM ε = {epsilon_label}",
        fontsize=15,
    )
    column_titles = [
        "Clean Overlay\nTarget: original class",
        "Adversarial Overlay\nTarget: original class",
        "Adversarial Overlay\nTarget: new prediction",
    ]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=10)

    for row, example in enumerate(examples):
        clean_display = nchw_to_display_image(_example_field(example, "clean_image"))
        adversarial_display = nchw_to_display_image(
            _example_field(example, "adversarial_image")
        )
        clean_original = resize_heatmap_to_image(
            _example_field(example, "clean_original_class_cam")[0],
            clean_display.shape[:2],
        )
        adversarial_original = resize_heatmap_to_image(
            _example_field(example, "adversarial_original_class_cam")[0],
            adversarial_display.shape[:2],
        )
        adversarial_new = resize_heatmap_to_image(
            _example_field(example, "adversarial_new_class_cam")[0],
            adversarial_display.shape[:2],
        )
        overlays = [
            heatmap_overlay(clean_display, clean_original),
            heatmap_overlay(adversarial_display, adversarial_original),
            heatmap_overlay(adversarial_display, adversarial_new),
        ]
        for axis, overlay in zip(axes[row], overlays):
            axis.imshow(overlay)
            _hide_axis(axis)
        axes[row, 0].set_title(
            f"Clean Overlay\nTarget: original class\n{_short_row_label(example)}",
            fontsize=8.5,
        )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path


def save_gradcam_success_vs_control(
    examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    epsilon_label: str,
) -> Path:
    """Save attack-success rows next to attack-resisted control rows."""
    if len(examples) < 1:
        raise ValueError("At least one example is required for success/control figure.")

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(examples)
    figure, axes = plt.subplots(rows, 4, figsize=(12.0, 3.0 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    figure.suptitle(
        f"FGSM Success vs Control Grad-CAM | ε = {epsilon_label}",
        fontsize=15,
    )
    column_titles = [
        "Clean Image",
        "Clean Grad-CAM Overlay",
        "Adversarial Image",
        "Adversarial Grad-CAM Overlay",
    ]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=10)

    for row, example in enumerate(examples):
        panels = _prepared_panels(example)
        images = [
            panels["clean_image"],
            panels["clean_overlay"],
            panels["adversarial_image"],
            panels["adversarial_overlay"],
        ]
        for axis, image in zip(axes[row], images):
            axis.imshow(image)
            _hide_axis(axis)
        status = str(_example_field(example, "status")).replace("_", " ").title()
        axes[row, 0].set_title(
            f"Clean Image\n{status}\n{_short_row_label(example)}",
            fontsize=8.5,
        )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return figure_path
