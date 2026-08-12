import numpy as np
import pytest

from src.attacks import fgsm_attack, pgd_linf_attack
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


class FixedGradientModel:
    """Small model double that exercises the real input-gradient helper."""

    def __init__(self, gradient: np.ndarray, num_classes: int = 10) -> None:
        self.xp = np
        self.gradient = gradient
        self.num_classes = num_classes
        self.backward_calls = 0
        self._last_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._last_shape = inputs.shape
        logits = np.linspace(
            -0.25,
            0.25,
            num=self.num_classes,
            dtype=inputs.dtype,
        )
        return np.tile(logits, (inputs.shape[0], 1))

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        if self._last_shape is None:
            raise RuntimeError("forward must be called before backward.")
        self.backward_calls += 1
        return np.broadcast_to(self.gradient, self._last_shape).copy()


def _small_images(batch_size: int = 2) -> np.ndarray:
    return np.linspace(
        0.2,
        0.8,
        num=batch_size * 3 * 2 * 2,
        dtype=np.float32,
    ).reshape(batch_size, 3, 2, 2)


def _labels(batch_size: int = 2) -> np.ndarray:
    return np.arange(batch_size, dtype=np.int64) % 10


def _gradient() -> np.ndarray:
    return np.array(
        [
            [
                [[-2.0, 1.0], [0.0, 3.0]],
                [[1.0, -1.0], [2.0, -2.0]],
                [[3.0, 0.5], [-0.5, -3.0]],
            ]
        ],
        dtype=np.float32,
    )


def _trainable_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


def _assert_linf_bound(
    adversarial_images: np.ndarray,
    clean_images: np.ndarray,
    epsilon: float,
) -> None:
    perturbation = adversarial_images - clean_images
    assert np.max(np.abs(perturbation)) <= epsilon + 1e-7


def test_pgd_zero_epsilon_returns_clean_copy() -> None:
    images = _small_images()
    labels = _labels()
    model = FixedGradientModel(_gradient())

    adversarial_images = pgd_linf_attack(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.0,
        alpha=0.1,
        steps=3,
        random_start=True,
        seed=7,
    )

    assert adversarial_images is not images
    np.testing.assert_array_equal(adversarial_images, images)
    assert model.backward_calls == 0


def test_pgd_zero_steps_is_explicit_no_op() -> None:
    images = _small_images()
    labels = _labels()
    model = FixedGradientModel(_gradient())

    adversarial_images = pgd_linf_attack(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.3,
        alpha=0.0,
        steps=0,
        random_start=True,
        seed=7,
    )

    np.testing.assert_array_equal(adversarial_images, images)
    assert model.backward_calls == 0


def test_one_step_pgd_without_random_start_matches_fgsm() -> None:
    images = _small_images()
    labels = _labels()
    epsilon = 0.125
    gradient_model = FixedGradientModel(_gradient())
    pgd_model = FixedGradientModel(_gradient())

    grad_input = compute_input_gradient(
        gradient_model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
    )
    expected_images = fgsm_attack(images, grad_input, epsilon)
    adversarial_images = pgd_linf_attack(
        pgd_model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=epsilon,
        alpha=epsilon,
        steps=1,
        random_start=False,
    )

    np.testing.assert_allclose(adversarial_images, expected_images)


def test_pgd_enforces_linf_projection_and_valid_range_clipping() -> None:
    images = np.array(
        [
            [
                [[0.02, 0.98], [0.5, 0.5]],
                [[0.01, 0.99], [0.5, 0.5]],
                [[0.00, 1.00], [0.5, 0.5]],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([1], dtype=np.int64)
    gradient = np.array(
        [
            [
                [[-1.0, 1.0], [1.0, -1.0]],
                [[-1.0, 1.0], [1.0, -1.0]],
                [[-1.0, 1.0], [1.0, -1.0]],
            ]
        ],
        dtype=np.float32,
    )

    adversarial_images = pgd_linf_attack(
        FixedGradientModel(gradient),  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.1,
        alpha=0.4,
        steps=2,
        random_start=False,
    )

    assert adversarial_images.min() >= 0.0
    assert adversarial_images.max() <= 1.0
    _assert_linf_bound(adversarial_images, images, epsilon=0.1)
    expected = np.array(
        [
            [
                [[0.0, 1.0], [0.6, 0.4]],
                [[0.0, 1.0], [0.6, 0.4]],
                [[0.0, 1.0], [0.6, 0.4]],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(adversarial_images, expected)


def test_pgd_random_start_is_deterministic_with_seed() -> None:
    images = np.full((2, 3, 2, 2), 0.5, dtype=np.float32)
    labels = _labels()
    zero_gradient = np.zeros((1, 3, 2, 2), dtype=np.float32)

    first = pgd_linf_attack(
        FixedGradientModel(zero_gradient),  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.2,
        alpha=0.1,
        steps=1,
        random_start=True,
        seed=123,
    )
    second = pgd_linf_attack(
        FixedGradientModel(zero_gradient),  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.2,
        alpha=0.1,
        steps=1,
        random_start=True,
        seed=123,
    )

    np.testing.assert_array_equal(first, second)
    _assert_linf_bound(first, images, epsilon=0.2)


def test_pgd_random_start_changes_with_different_seeds() -> None:
    images = np.full((2, 3, 2, 2), 0.5, dtype=np.float32)
    labels = _labels()
    zero_gradient = np.zeros((1, 3, 2, 2), dtype=np.float32)

    first = pgd_linf_attack(
        FixedGradientModel(zero_gradient),  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.2,
        alpha=0.1,
        steps=1,
        random_start=True,
        seed=123,
    )
    second = pgd_linf_attack(
        FixedGradientModel(zero_gradient),  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.2,
        alpha=0.1,
        steps=1,
        random_start=True,
        seed=456,
    )

    assert not np.array_equal(first, second)


def test_pgd_multi_step_update_and_batch_support() -> None:
    images = np.full((2, 3, 2, 2), 0.5, dtype=np.float32)
    labels = _labels()
    gradient = _gradient()
    model = FixedGradientModel(gradient)

    adversarial_images = pgd_linf_attack(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.4,
        alpha=0.1,
        steps=3,
        random_start=False,
    )

    expected = images + 0.3 * np.sign(np.broadcast_to(gradient, images.shape))
    assert adversarial_images.shape == images.shape
    assert model.backward_calls == 3
    np.testing.assert_allclose(adversarial_images, expected, atol=1e-7)


def test_pgd_does_not_mutate_clean_inputs_or_model_parameters() -> None:
    model = CompactCNN(seed=42)
    loss_function = SoftmaxCrossEntropyLoss()
    images = np.random.default_rng(11).random(
        (1, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([3], dtype=np.int64)
    images_before = images.copy()
    labels_before = labels.copy()
    parameters_before = [
        parameter.copy() for parameter in _trainable_parameters(model)
    ]

    adversarial_images = pgd_linf_attack(
        model,
        loss_function,
        images,
        labels,
        epsilon=1.0 / 255.0,
        alpha=1.0 / 255.0,
        steps=2,
        random_start=False,
    )

    np.testing.assert_array_equal(images, images_before)
    np.testing.assert_array_equal(labels, labels_before)
    assert adversarial_images is not images
    assert adversarial_images.shape == images.shape
    for parameter, parameter_before in zip(
        _trainable_parameters(model),
        parameters_before,
        strict=True,
    ):
        np.testing.assert_array_equal(parameter, parameter_before)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"epsilon": -0.1}, "epsilon must be"),
        ({"epsilon": np.inf}, "epsilon must be"),
        ({"alpha": 0.0, "steps": 1}, "alpha must be positive"),
        ({"alpha": -0.1, "steps": 0}, "alpha must be positive"),
        ({"alpha": np.nan}, "alpha must be a finite number"),
        ({"steps": -1}, "steps must be"),
        ({"steps": True}, "steps must be"),
        ({"random_start": "yes"}, "random_start must be"),
        ({"seed": -1, "random_start": True}, "seed must be"),
    ],
)
def test_pgd_rejects_invalid_configuration(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    images = _small_images()
    labels = _labels()
    arguments = {
        "epsilon": 0.1,
        "alpha": 0.05,
        "steps": 1,
        "random_start": False,
        "seed": None,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=expected_message):
        pgd_linf_attack(
            FixedGradientModel(_gradient()),  # type: ignore[arg-type]
            SoftmaxCrossEntropyLoss(),
            images,
            labels,
            **arguments,  # type: ignore[arg-type]
        )


def test_pgd_rejects_invalid_images_and_labels() -> None:
    labels = _labels()

    with pytest.raises(ValueError, match="non-empty NCHW images"):
        pgd_linf_attack(
            FixedGradientModel(_gradient()),  # type: ignore[arg-type]
            SoftmaxCrossEntropyLoss(),
            np.zeros((3, 2, 2), dtype=np.float32),
            labels,
            epsilon=0.1,
            alpha=0.05,
            steps=1,
        )

    out_of_range = _small_images()
    out_of_range[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="valid \\[0, 1\\] range"):
        pgd_linf_attack(
            FixedGradientModel(_gradient()),  # type: ignore[arg-type]
            SoftmaxCrossEntropyLoss(),
            out_of_range,
            labels,
            epsilon=0.1,
            alpha=0.05,
            steps=1,
        )

    with pytest.raises(ValueError, match="labels with shape"):
        pgd_linf_attack(
            FixedGradientModel(_gradient()),  # type: ignore[arg-type]
            SoftmaxCrossEntropyLoss(),
            _small_images(),
            np.array([[1, 2]], dtype=np.int64),
            epsilon=0.1,
            alpha=0.05,
            steps=1,
        )
