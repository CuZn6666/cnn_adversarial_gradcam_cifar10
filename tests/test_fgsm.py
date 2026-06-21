import numpy as np
import pytest

from src.attacks import fgsm_attack


def _deterministic_inputs() -> tuple[np.ndarray, np.ndarray]:
    images = np.array(
        [
            [
                [[0.0, 0.2], [0.8, 1.0]],
                [[0.1, 0.4], [0.6, 0.9]],
                [[0.3, 0.5], [0.7, 0.95]],
            ]
        ],
        dtype=np.float32,
    )
    grad_input = np.array(
        [
            [
                [[-1.0, 2.0], [3.0, 4.0]],
                [[1.0, -2.0], [-3.0, 4.0]],
                [[-1.0, 2.0], [-3.0, 4.0]],
            ]
        ],
        dtype=np.float32,
    )
    return images, grad_input


def test_fgsm_attack_zero_epsilon_returns_original_images() -> None:
    images, grad_input = _deterministic_inputs()

    adversarial_images = fgsm_attack(images, grad_input, epsilon=0.0)

    np.testing.assert_array_equal(adversarial_images, images)


def test_fgsm_attack_preserves_shape_and_bounds_perturbation() -> None:
    images, grad_input = _deterministic_inputs()
    epsilon = 0.1

    adversarial_images = fgsm_attack(images, grad_input, epsilon)
    perturbation = adversarial_images - images

    assert adversarial_images.shape == images.shape
    assert np.isfinite(adversarial_images).all()
    assert np.max(np.abs(perturbation)) <= epsilon + 1e-7


def test_fgsm_attack_clips_pixels_to_valid_range() -> None:
    images, grad_input = _deterministic_inputs()

    adversarial_images = fgsm_attack(images, grad_input, epsilon=0.2)

    assert adversarial_images.min() >= 0.0
    assert adversarial_images.max() <= 1.0
    assert adversarial_images[0, 0, 0, 0] == 0.0
    assert adversarial_images[0, 0, 1, 1] == 1.0


@pytest.mark.parametrize("epsilon", [-0.1, np.inf, np.nan, True, "0.1"])
def test_fgsm_attack_rejects_invalid_epsilon(epsilon: object) -> None:
    images, grad_input = _deterministic_inputs()

    with pytest.raises(
        ValueError,
        match="epsilon must be a non-negative finite number",
    ):
        fgsm_attack(images, grad_input, epsilon)  # type: ignore[arg-type]


def test_fgsm_attack_rejects_shape_mismatch() -> None:
    images, _ = _deterministic_inputs()
    wrong_gradient = np.zeros((1, 3, 1, 2), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Input gradients must match the image shape",
    ):
        fgsm_attack(images, wrong_gradient, epsilon=0.1)


def test_fgsm_attack_rejects_non_nchw_images() -> None:
    images = np.zeros((3, 4, 5), dtype=np.float32)
    grad_input = np.zeros_like(images)

    with pytest.raises(
        ValueError,
        match="fgsm_attack expects non-empty NCHW images",
    ):
        fgsm_attack(images, grad_input, epsilon=0.1)


@pytest.mark.parametrize(
    ("invalid_target", "expected_message"),
    [
        ("images", "Images must contain only finite values"),
        ("gradients", "Input gradients must contain only finite values"),
    ],
)
def test_fgsm_attack_rejects_non_finite_values(
    invalid_target: str,
    expected_message: str,
) -> None:
    images, grad_input = _deterministic_inputs()
    if invalid_target == "images":
        images[0, 0, 0, 0] = np.nan
    else:
        grad_input[0, 0, 0, 0] = np.inf

    with pytest.raises(ValueError, match=expected_message):
        fgsm_attack(images, grad_input, epsilon=0.1)


def test_fgsm_attack_is_deterministic() -> None:
    images, grad_input = _deterministic_inputs()

    first_result = fgsm_attack(images, grad_input, epsilon=0.1)
    second_result = fgsm_attack(images, grad_input, epsilon=0.1)

    np.testing.assert_array_equal(first_result, second_result)
