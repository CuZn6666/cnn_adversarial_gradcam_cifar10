from pathlib import Path

import numpy as np
import pytest

from src.visualization import save_fgsm_visualizations


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
