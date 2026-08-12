import numpy as np
import pytest

from src.attacks import fgsm_attack, pgd_linf_attack
from src.attacks.pgd import _pgd_linf_attack_from_initial
from src.backend import is_cupy_array, to_backend, to_numpy
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


pytestmark = pytest.mark.requires_cupy


TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-6
PGD_EPSILON = 4.0 / 255.0
PGD_ALPHA = 1.0 / 255.0
PGD_STEPS = 3


class _RecordingCompactCNN(CompactCNN):
    def __init__(self, *, seed: int, backend) -> None:
        super().__init__(seed=seed, backend=backend)
        self.forward_inputs = []
        self.forward_outputs = []
        self.backward_inputs = []
        self.backward_outputs = []

    def forward(self, inputs):
        self.forward_inputs.append(inputs)
        outputs = super().forward(inputs)
        self.forward_outputs.append(outputs)
        return outputs

    def backward(self, grad_logits):
        self.backward_inputs.append(grad_logits)
        grad_input = super().backward(grad_logits)
        self.backward_outputs.append(grad_input)
        return grad_input


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


def _synchronized_models(cp) -> tuple[CompactCNN, _RecordingCompactCNN]:
    numpy_model = CompactCNN(seed=0, backend="numpy")
    cupy_model = _RecordingCompactCNN(seed=999, backend=cp)
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


def _shared_initial_state(images_np: np.ndarray) -> np.ndarray:
    perturbation = np.linspace(
        -PGD_EPSILON,
        PGD_EPSILON,
        num=images_np.size,
        dtype=np.float32,
    ).reshape(images_np.shape)
    return np.clip(images_np + perturbation, 0.0, 1.0)


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


def _assert_recorded_cupy_tensors(model: _RecordingCompactCNN) -> None:
    assert model.forward_inputs, "expected at least one recorded forward input"
    assert model.forward_outputs, "expected at least one recorded forward output"
    assert model.backward_inputs, "expected at least one recorded backward input"
    assert model.backward_outputs, "expected at least one recorded backward output"

    for index, tensor in enumerate(model.forward_inputs):
        assert is_cupy_array(tensor), f"forward input {index} is not CuPy"
    for index, tensor in enumerate(model.forward_outputs):
        assert is_cupy_array(tensor), f"forward output {index} is not CuPy"
    for index, tensor in enumerate(model.backward_inputs):
        assert is_cupy_array(tensor), f"backward input {index} is not CuPy"
    for index, tensor in enumerate(model.backward_outputs):
        assert is_cupy_array(tensor), f"backward output {index} is not CuPy"


def _assert_unit_interval(name: str, values) -> None:
    values_np = to_numpy(values)
    assert np.isfinite(values_np).all(), f"{name} contains non-finite values"
    assert values_np.min() >= 0.0, f"{name} min={values_np.min():.8e}"
    assert values_np.max() <= 1.0, f"{name} max={values_np.max():.8e}"


def _assert_linf_bound(
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


def _assert_adversarial_forward_matches(
    numpy_model: CompactCNN,
    cupy_model: _RecordingCompactCNN,
    adversarial_np,
    adversarial_cp,
) -> None:
    logits_np = numpy_model.forward(adversarial_np)
    logits_cp = cupy_model.forward(adversarial_cp)
    assert is_cupy_array(logits_cp)
    _assert_allclose_named("adversarial logits", logits_cp, logits_np)

    predictions_np = np.argmax(logits_np, axis=1)
    predictions_cp = cupy_model.xp.argmax(logits_cp, axis=1)
    assert is_cupy_array(predictions_cp)
    np.testing.assert_array_equal(to_numpy(predictions_cp), predictions_np)


def test_pgd_no_random_start_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    numpy_loss = SoftmaxCrossEntropyLoss(backend="numpy")
    cupy_loss = SoftmaxCrossEntropyLoss(backend=cp)
    numpy_parameters_before = _parameter_snapshots(numpy_model)
    cupy_parameters_before = _parameter_snapshots(cupy_model)

    _assert_parameters_match("initial", numpy_model, cupy_model)
    adversarial_np = pgd_linf_attack(
        numpy_model,
        numpy_loss,
        images_np,
        labels_np,
        epsilon=PGD_EPSILON,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
        random_start=False,
    )
    adversarial_cp = pgd_linf_attack(
        cupy_model,
        cupy_loss,
        images_cp,
        labels_cp,
        epsilon=PGD_EPSILON,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
        random_start=False,
    )

    assert is_cupy_array(adversarial_cp)
    _assert_allclose_named("no-random-start PGD", adversarial_cp, adversarial_np)
    _assert_unit_interval("CuPy no-random-start PGD", adversarial_cp)
    _assert_linf_bound(
        "CuPy no-random-start PGD",
        adversarial_cp,
        images_cp,
        PGD_EPSILON,
    )
    _assert_adversarial_forward_matches(
        numpy_model,
        cupy_model,
        adversarial_np,
        adversarial_cp,
    )
    _assert_recorded_cupy_tensors(cupy_model)
    _assert_parameters_unchanged("NumPy PGD", numpy_model, numpy_parameters_before)
    _assert_parameters_unchanged(
        "CuPy PGD",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )


def test_pgd_shared_initial_state_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    initial_np = _shared_initial_state(images_np)
    initial_cp = to_backend(initial_np, cp)
    numpy_loss = SoftmaxCrossEntropyLoss(backend="numpy")
    cupy_loss = SoftmaxCrossEntropyLoss(backend=cp)
    numpy_parameters_before = _parameter_snapshots(numpy_model)
    cupy_parameters_before = _parameter_snapshots(cupy_model)

    adversarial_np = _pgd_linf_attack_from_initial(
        numpy_model,
        numpy_loss,
        images_np,
        labels_np,
        initial_adversarial_images=initial_np,
        epsilon=PGD_EPSILON,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
    )
    adversarial_cp = _pgd_linf_attack_from_initial(
        cupy_model,
        cupy_loss,
        images_cp,
        labels_cp,
        initial_adversarial_images=initial_cp,
        epsilon=PGD_EPSILON,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
    )

    assert is_cupy_array(initial_cp)
    assert is_cupy_array(adversarial_cp)
    _assert_allclose_named("shared-initial PGD", adversarial_cp, adversarial_np)
    _assert_unit_interval("CuPy shared-initial PGD", adversarial_cp)
    _assert_linf_bound(
        "CuPy shared-initial PGD",
        adversarial_cp,
        images_cp,
        PGD_EPSILON,
    )
    _assert_adversarial_forward_matches(
        numpy_model,
        cupy_model,
        adversarial_np,
        adversarial_cp,
    )
    _assert_recorded_cupy_tensors(cupy_model)
    _assert_parameters_unchanged(
        "NumPy shared-initial PGD",
        numpy_model,
        numpy_parameters_before,
    )
    _assert_parameters_unchanged(
        "CuPy shared-initial PGD",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )


def test_pgd_epsilon_zero_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    cupy_parameters_before = _parameter_snapshots(cupy_model)

    adversarial_np = pgd_linf_attack(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
        epsilon=0.0,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
        random_start=True,
        seed=123,
    )
    adversarial_cp = pgd_linf_attack(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        images_cp,
        labels_cp,
        epsilon=0.0,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
        random_start=True,
        seed=123,
    )

    assert is_cupy_array(adversarial_cp)
    np.testing.assert_array_equal(adversarial_np, images_np)
    np.testing.assert_array_equal(to_numpy(adversarial_cp), images_np)
    _assert_parameters_unchanged(
        "CuPy epsilon-zero PGD",
        cupy_model,
        cupy_parameters_before,
        expect_cupy=True,
    )


def test_one_step_pgd_matches_fgsm_on_both_backends(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    epsilon = 2.0 / 255.0

    grad_np = compute_input_gradient(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
    )
    grad_cp = compute_input_gradient(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        images_cp,
        labels_cp,
    )
    assert is_cupy_array(grad_cp)

    fgsm_np = fgsm_attack(images_np, grad_np, epsilon=epsilon)
    fgsm_cp = fgsm_attack(images_cp, grad_cp, epsilon=epsilon)
    pgd_np = pgd_linf_attack(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
        epsilon=epsilon,
        alpha=epsilon,
        steps=1,
        random_start=False,
    )
    pgd_cp = pgd_linf_attack(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        images_cp,
        labels_cp,
        epsilon=epsilon,
        alpha=epsilon,
        steps=1,
        random_start=False,
    )

    assert is_cupy_array(fgsm_cp)
    assert is_cupy_array(pgd_cp)
    _assert_allclose_named("NumPy PGD/FGSM", pgd_np, fgsm_np)
    _assert_allclose_named("CuPy PGD/FGSM", pgd_cp, fgsm_cp)
    _assert_allclose_named("one-step PGD", pgd_cp, pgd_np)
