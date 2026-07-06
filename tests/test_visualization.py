from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.visualization import (
    save_combined_fgsm_figure,
    save_fgsm_visualizations,
)


def _synthetic_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_images = np.linspace(
        0.0,
        1.0,
        num=3 * 4 * 5,
        dtype=np.float32,
    ).reshape(1, 3, 4, 5)
    perturbation = np.full_like(clean_images, 0.05)
    adversarial_images = np.clip(
        clean_images + perturbation,
        0.0,
        1.0,
    )
    gradient_map = np.arange(20, dtype=np.float32).reshape(1, 4, 5)
    return clean_images, adversarial_images, gradient_map


def test_save_fgsm_visualizations_creates_expected_png_files(
    tmp_path: Path,
) -> None:
    clean_images, adversarial_images, gradient_map = _synthetic_example()

    paths = save_fgsm_visualizations(
        clean_images,
        adversarial_images,
        gradient_map,
        tmp_path / "figures",
    )

    assert set(paths) == {
        "clean",
        "adversarial",
        "gradient",
        "perturbation",
    }
    assert {path.name for path in paths.values()} == {
        "fgsm_example_clean.png",
        "fgsm_example_adversarial.png",
        "fgsm_example_input_gradient.png",
        "fgsm_example_perturbation.png",
    }
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_fgsm_visualizations_accepts_hw_map_and_is_repeatable(
    tmp_path: Path,
) -> None:
    clean_images, adversarial_images, gradient_map = _synthetic_example()

    first_paths = save_fgsm_visualizations(
        clean_images,
        adversarial_images,
        gradient_map[0],
        tmp_path,
        prefix="example_01",
    )
    second_paths = save_fgsm_visualizations(
        clean_images,
        adversarial_images,
        gradient_map[0],
        tmp_path,
        prefix="example_01",
    )

    assert first_paths == second_paths
    assert all(path.is_file() for path in second_paths.values())


@pytest.mark.parametrize(
    ("clean_shape", "adversarial_shape", "gradient_shape", "message"),
    [
        ((3, 4, 5), (3, 4, 5), (4, 5), "clean_images must have shape"),
        (
            (1, 3, 4, 5),
            (1, 3, 4, 4),
            (4, 5),
            "adversarial_images must match",
        ),
        (
            (1, 3, 4, 5),
            (1, 3, 4, 5),
            (4, 4),
            "gradient_map must have shape",
        ),
    ],
)
def test_save_fgsm_visualizations_rejects_invalid_shapes(
    tmp_path: Path,
    clean_shape: tuple[int, ...],
    adversarial_shape: tuple[int, ...],
    gradient_shape: tuple[int, ...],
    message: str,
) -> None:
    clean_images = np.zeros(clean_shape, dtype=np.float32)
    adversarial_images = np.zeros(adversarial_shape, dtype=np.float32)
    gradient_map = np.zeros(gradient_shape, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        save_fgsm_visualizations(
            clean_images,
            adversarial_images,
            gradient_map,
            tmp_path,
        )


def _write_png(path: Path, value: float) -> None:
    image = np.full((4, 5, 3), value, dtype=np.float32)
    plt.imsave(path, image)


def test_save_combined_fgsm_figure_creates_file_without_modifying_sources(
    tmp_path: Path,
) -> None:
    source_paths = {
        "clean": tmp_path / "clean.png",
        "adversarial": tmp_path / "adversarial.png",
        "gradient": tmp_path / "gradient.png",
        "perturbation": tmp_path / "perturbation.png",
    }
    for index, path in enumerate(source_paths.values(), start=1):
        _write_png(path, index / 5.0)
    before_bytes = {
        label: path.read_bytes()
        for label, path in source_paths.items()
    }

    output_path = save_combined_fgsm_figure(
        source_paths["clean"],
        source_paths["adversarial"],
        source_paths["gradient"],
        source_paths["perturbation"],
        tmp_path / "combined" / "fgsm_combined.png",
    )

    assert output_path == tmp_path / "combined" / "fgsm_combined.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert {
        label: path.read_bytes()
        for label, path in source_paths.items()
    } == before_bytes


def test_save_combined_fgsm_figure_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.png"
    adversarial_path = tmp_path / "adversarial.png"
    gradient_path = tmp_path / "gradient.png"
    for path in (clean_path, adversarial_path, gradient_path):
        _write_png(path, 0.25)

    with pytest.raises(FileNotFoundError, match="perturbation image file"):
        save_combined_fgsm_figure(
            clean_path,
            adversarial_path,
            gradient_path,
            tmp_path / "missing_perturbation.png",
            tmp_path / "combined.png",
        )
