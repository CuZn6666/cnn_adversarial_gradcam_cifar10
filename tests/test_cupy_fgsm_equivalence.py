import numpy as np
import pytest

from src.attacks import fgsm_attack
from src.backend import is_cupy_array, to_backend, to_numpy
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


pytestmark = pytest.mark.requires_cupy


TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-6
FGSM_EPSILON = 2.0 / 255.0


def _trainable_parameters(model: CompactCNN):
    return (
        ("conv1.weights", model.conv1.weights),
        ("conv1.bias", model.conv1.bias),
        ("conv2.weights", model.conv2.weights),
        ("conv2.bias", model.conv2.bias),
        ("classifier.weights", model.classifier.weights),
        ("classifier.bias", model.classifier.bias),
    )


def _reference_parameter(name: str, shape: tuple[int, ...]) -> np.ndarray:
    ranges = {
        "conv1.weights": (0.001, 0.020),
        "conv1.bias": (0.001, 0.008),
        "conv2.weights": (0.001, 0.012),
        "conv2.bias": (0.001, 0.006),
        "classifier.weights": (-0.010, 0.010),
        "classifier.bias": (-0.050, 0.050),
    }
    start, stop = ranges[name]
    return np.linspace(
        start,
        stop,
        num=int(np.prod(shape)),
        dtype=np.float32,
    ).reshape(shape)


def _synchronized_models(cp) -> tuple[CompactCNN, CompactCNN]:
    numpy_model = CompactCNN(seed=0, backend="numpy")
    cupy_model = CompactCNN(seed=999, backend=cp)
    numpy_parameters = dict(_trainable_parameters(numpy_model))
    cupy_parameters = dict(_trainable_parameters(cupy_model))

    for name, numpy_parameter in numpy_parameters.items():
        reference = _reference_parameter(name, numpy_parameter.shape)
        numpy_parameter[...] = reference
        cupy_parameters[name][...] = to_backend(reference, cp)

    return numpy_model, cupy_model


def _synthetic_batch(cp):
    images_np = np.linspace(
        0.02,
        0.98,
        num=2 * 3 * 32 * 32,
        dtype=np.float32,
    ).reshape(2, 3, 32, 32)
    images_np[0, 0, 0, 0] = 0.0
    images_np[1, 2, -1, -1] = 1.0
    labels_np = np.array([1, 7], dtype=np.int64)
    return (
        images_np,
        labels_np,
        to_backend(images_np, cp),
        to_backend(labels_np, cp),
    )


def _parameter_snapshots(model: CompactCNN) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(
        (name, to_numpy(parameter).copy())
        for name, parameter in _trainable_parameters(model)
    )


def _max_abs_difference(actual, expected) -> float:
    actual_np = to_numpy(actual)
    expected_np = to_numpy(expected)
    if actual_np.size == 0:
        return 0.0
    return float(np.max(np.abs(actual_np - expected_np)))


def _assert_allclose_named(
    name: str,
    actual,
    expected,
    *,
    rtol: float = TENSOR_RTOL,
    atol: float = TENSOR_ATOL,
) -> None:
    max_abs_diff = _max_abs_difference(actual, expected)
    np.testing.assert_allclose(
        to_numpy(actual),
        to_numpy(expected),
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} max_abs_diff={max_abs_diff:.8e}",
    )


def _assert_parameters_match(
    label: str,
    numpy_model: CompactCNN,
    cupy_model: CompactCNN,
) -> None:
    numpy_parameters = _trainable_parameters(numpy_model)
    cupy_parameters = _trainable_parameters(cupy_model)
    assert [name for name, _ in cupy_parameters] == [
        name for name, _ in numpy_parameters
    ]

    for (name, numpy_parameter), (_, cupy_parameter) in zip(
        numpy_parameters,
        cupy_parameters,
    ):
        assert is_cupy_array(cupy_parameter), f"{label} {name} is not a CuPy array"
        _assert_allclose_named(
            f"{label} parameter {name}",
            cupy_parameter,
            numpy_parameter,
        )


def _assert_parameters_unchanged(
    label: str,
    model: CompactCNN,
    snapshots: tuple[tuple[str, np.ndarray], ...],
    *,
    expect_cupy: bool = False,
) -> None:
    current_parameters = _trainable_parameters(model)
    assert [name for name, _ in current_parameters] == [
        name for name, _ in snapshots
    ]

    for (name, parameter), (_, snapshot) in zip(
        current_parameters,
        snapshots,
    ):
        if expect_cupy:
            assert is_cupy_array(parameter), f"{label} {name} is not a CuPy array"
        np.testing.assert_array_equal(
            to_numpy(parameter),
            snapshot,
            err_msg=f"{label} changed parameter {name}",
        )


def _assert_unit_interval(name: str, values) -> None:
    values_np = to_numpy(values)
    assert np.isfinite(values_np).all(), f"{name} contains non-finite values"
    assert values_np.min() >= 0.0, f"{name} min={values_np.min():.8e}"
    assert values_np.max() <= 1.0, f"{name} max={values_np.max():.8e}"


def _assert_perturbation_bound(
    name: str,
    adversarial_images,
    clean_images,
    epsilon: float,
) -> None:
    perturbation = to_numpy(adversarial_images - clean_images)
    max_abs = float(np.max(np.abs(perturbation)))
    assert max_abs <= epsilon + TENSOR_ATOL, (
        f"{name} perturbation max_abs={max_abs:.8e} exceeds "
        f"epsilon={epsilon:.8e}"
    )


def test_fgsm_attack_zero_epsilon_and_clipping_match_numpy(cp) -> None:
    images_np = np.array(
        [
            [
                [[0.0, 1.0], [0.05, 0.95]],
                [[0.2, 0.8], [0.4, 0.6]],
                [[0.3, 0.7], [0.1, 0.9]],
            ],
            [
                [[1.0, 0.0], [0.9, 0.1]],
                [[0.75, 0.25], [0.5, 0.5]],
                [[0.6, 0.4], [0.2, 0.8]],
            ],
        ],
        dtype=np.float32,
    )
    grad_input_np = np.array(
        [
            [
                [[-1.0, 1.0], [2.0, -2.0]],
                [[1.5, -1.5], [0.5, -0.5]],
                [[-3.0, 3.0], [4.0, -4.0]],
            ],
            [
                [[1.0, -1.0], [-2.0, 2.0]],
                [[-1.5, 1.5], [0.25, -0.25]],
                [[3.0, -3.0], [-4.0, 4.0]],
            ],
        ],
        dtype=np.float32,
    )
    images_cp = to_backend(images_np, cp)
    grad_input_cp = to_backend(grad_input_np, cp)

    zero_epsilon_np = fgsm_attack(images_np, grad_input_np, epsilon=0.0)
    zero_epsilon_cp = fgsm_attack(images_cp, grad_input_cp, epsilon=0.0)
    assert is_cupy_array(zero_epsilon_cp)
    np.testing.assert_array_equal(to_numpy(zero_epsilon_cp), images_np)
    _assert_allclose_named(
        "epsilon=0 adversarial images",
        zero_epsilon_cp,
        zero_epsilon_np,
    )

    epsilon = 0.2
    adversarial_np = fgsm_attack(images_np, grad_input_np, epsilon=epsilon)
    adversarial_cp = fgsm_attack(images_cp, grad_input_cp, epsilon=epsilon)
    assert is_cupy_array(adversarial_cp)
    assert adversarial_cp.shape == images_cp.shape
    _assert_allclose_named(
        "nonzero-epsilon adversarial images",
        adversarial_cp,
        adversarial_np,
    )
    _assert_unit_interval("CuPy adversarial images", adversarial_cp)
    _assert_perturbation_bound(
        "CuPy FGSM",
        adversarial_cp,
        images_cp,
        epsilon,
    )

    adversarial_cp_np = to_numpy(adversarial_cp)
    assert adversarial_cp_np[0, 0, 0, 0] == 0.0
    assert adversarial_cp_np[0, 0, 0, 1] == 1.0
    assert adversarial_cp_np[1, 0, 0, 0] == 1.0
    assert adversarial_cp_np[1, 0, 0, 1] == 0.0


def test_input_gradient_fgsm_attack_path_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    numpy_loss = SoftmaxCrossEntropyLoss(backend="numpy")
    cupy_loss = SoftmaxCrossEntropyLoss(backend=cp)

    _assert_parameters_match("initial", numpy_model, cupy_model)
    numpy_parameters_before = _parameter_snapshots(numpy_model)
    cupy_parameters_before = _parameter_snapshots(cupy_model)

    grad_input_np = compute_input_gradient(
        numpy_model,
        numpy_loss,
        images_np,
        labels_np,
    )
    grad_input_cp = compute_input_gradient(
        cupy_model,
        cupy_loss,
        images_cp,
        labels_cp,
    )
    assert is_cupy_array(grad_input_cp)
    assert grad_input_cp.shape == images_cp.shape
    assert grad_input_np.shape == images_np.shape
    assert np.isfinite(grad_input_np).all()
    assert np.isfinite(to_numpy(grad_input_cp)).all()
    assert np.any(grad_input_np != 0.0)
    _assert_allclose_named("input gradients", grad_input_cp, grad_input_np)
    _assert_parameters_unchanged(
        "after input-gradient NumPy",
        numpy_model,
        numpy_parameters_before,
    )
    _assert_parameters_unchanged(
        "after input-gradient CuPy",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )

    adversarial_np = fgsm_attack(images_np, grad_input_np, epsilon=FGSM_EPSILON)
    adversarial_cp = fgsm_attack(images_cp, grad_input_cp, epsilon=FGSM_EPSILON)
    assert is_cupy_array(adversarial_cp)
    assert adversarial_cp.shape == images_cp.shape
    _assert_allclose_named(
        "FGSM adversarial images",
        adversarial_cp,
        adversarial_np,
    )
    _assert_unit_interval("CuPy attack-path adversarial images", adversarial_cp)
    _assert_perturbation_bound(
        "CuPy attack-path FGSM",
        adversarial_cp,
        images_cp,
        FGSM_EPSILON,
    )
    _assert_parameters_unchanged(
        "after FGSM NumPy",
        numpy_model,
        numpy_parameters_before,
    )
    _assert_parameters_unchanged(
        "after FGSM CuPy",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )

    adversarial_logits_np = numpy_model.forward(adversarial_np)
    adversarial_logits_cp = cupy_model.forward(adversarial_cp)
    assert is_cupy_array(adversarial_logits_cp)
    _assert_allclose_named(
        "adversarial logits",
        adversarial_logits_cp,
        adversarial_logits_np,
    )

    predictions_np = np.argmax(adversarial_logits_np, axis=1)
    predictions_cp = cp.argmax(adversarial_logits_cp, axis=1)
    assert is_cupy_array(predictions_cp)
    np.testing.assert_array_equal(to_numpy(predictions_cp), predictions_np)
    _assert_parameters_unchanged(
        "after adversarial forward NumPy",
        numpy_model,
        numpy_parameters_before,
    )
    _assert_parameters_unchanged(
        "after adversarial forward CuPy",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )
