from pathlib import Path

import numpy as np
import pytest

from experiments.gradcam.generate_adversarial_comparisons import (
    GradCAMCandidate,
    _diverse_selection,
)
from src.gradcam_visualization import (
    heatmap_overlay,
    resize_heatmap_to_image,
    save_gradcam_hero_figure,
)


def _candidate(
    dataset_index: int,
    true_label: int,
    status: str = "attack_success",
) -> GradCAMCandidate:
    return GradCAMCandidate(
        dataset_index=dataset_index,
        true_label=true_label,
        true_class=f"class_{true_label}",
        clean_prediction=true_label,
        clean_prediction_class=f"class_{true_label}",
        adversarial_prediction=(true_label + 1) % 10,
        adversarial_prediction_class=f"class_{(true_label + 1) % 10}",
        clean_confidence=0.8,
        adversarial_confidence=0.6,
        adversarial_original_class_confidence=0.2,
        confidence_drop=0.6,
        status=status,  # type: ignore[arg-type]
        clean_image=np.zeros((1, 3, 32, 32), dtype=np.float32),
        adversarial_image=np.ones((1, 3, 32, 32), dtype=np.float32) * 0.1,
    )


def _figure_example(dataset_index: int = 3) -> dict[str, object]:
    clean_image = np.linspace(
        0.0,
        1.0,
        num=3 * 32 * 32,
        dtype=np.float32,
    ).reshape(1, 3, 32, 32)
    adversarial_image = np.clip(clean_image + 0.02, 0.0, 1.0)
    clean_cam = np.linspace(0.0, 1.0, num=16 * 16, dtype=np.float32).reshape(
        1,
        16,
        16,
    )
    adversarial_cam = clean_cam[:, ::-1, :]
    return {
        "dataset_index": dataset_index,
        "true_class": "cat",
        "clean_prediction_class": "cat",
        "adversarial_prediction_class": "dog",
        "clean_image": clean_image,
        "adversarial_image": adversarial_image,
        "clean_cam": clean_cam,
        "adversarial_cam": adversarial_cam,
        "status": "attack_success",
    }


def test_diverse_selection_prefers_distinct_classes_then_scan_order() -> None:
    candidates = [
        _candidate(0, 1),
        _candidate(1, 1),
        _candidate(2, 2),
        _candidate(3, 3),
    ]

    selected = _diverse_selection(candidates, count=4)

    assert [candidate.dataset_index for candidate in selected] == [0, 2, 3, 1]


def test_diverse_selection_returns_available_candidates_without_duplicates() -> None:
    candidates = [_candidate(5, 4), _candidate(6, 4)]

    selected = _diverse_selection(candidates, count=4)

    assert [candidate.dataset_index for candidate in selected] == [5, 6]


def test_resize_heatmap_to_image_returns_expected_shape_and_finite_values() -> None:
    heatmap = np.linspace(0.0, 1.0, num=16 * 16, dtype=np.float32).reshape(16, 16)

    resized = resize_heatmap_to_image(heatmap, (32, 32))

    assert resized.shape == (32, 32)
    assert np.isfinite(resized).all()
    assert resized.min() >= 0.0
    assert resized.max() <= 1.0


def test_heatmap_overlay_returns_rgb_values_in_range() -> None:
    image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    heatmap = np.linspace(0.0, 1.0, num=32 * 32, dtype=np.float32).reshape(32, 32)

    overlay = heatmap_overlay(image, heatmap, alpha=0.4)

    assert overlay.shape == image.shape
    assert np.isfinite(overlay).all()
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0


def test_save_gradcam_hero_figure_creates_non_empty_png(tmp_path: Path) -> None:
    output_path = save_gradcam_hero_figure(
        [_figure_example(3), _figure_example(7), _figure_example(11)],
        tmp_path / "gradcam_hero.png",
        epsilon_label="8/255",
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_gradcam_hero_figure_rejects_empty_examples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one example"):
        save_gradcam_hero_figure(
            [],
            tmp_path / "empty.png",
            epsilon_label="8/255",
        )
