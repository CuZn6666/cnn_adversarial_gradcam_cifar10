import numpy as np
import pytest

from src.backend import is_cupy_array, to_backend, to_numpy
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD
from src.training import train_step


pytestmark = pytest.mark.requires_cupy


TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-6
LOSS_RTOL = 1e-6
LOSS_ATOL = 1e-7
LEARNING_RATE = 7e-4


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
        0.05,
        0.95,
        num=2 * 3 * 32 * 32,
        dtype=np.float32,
    ).reshape(2, 3, 32, 32)
    labels_np = np.array([1, 7], dtype=np.int64)
    return (
        images_np,
        labels_np,
        to_backend(images_np, cp),
        to_backend(labels_np, cp),
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


def _assert_named_gradients_match(
    numpy_model: CompactCNN,
    cupy_model: CompactCNN,
) -> None:
    numpy_pairs = numpy_model.named_parameters_and_gradients()
    cupy_pairs = cupy_model.named_parameters_and_gradients()
    assert [name for name, _, _ in cupy_pairs] == [
        name for name, _, _ in numpy_pairs
    ]

    for (name, _, numpy_gradient), (_, cupy_parameter, cupy_gradient) in zip(
        numpy_pairs,
        cupy_pairs,
    ):
        assert is_cupy_array(cupy_parameter), f"{name} parameter is not a CuPy array"
        assert is_cupy_array(cupy_gradient), f"{name} gradient is not a CuPy array"
        _assert_allclose_named(
            f"gradient {name}",
            cupy_gradient,
            numpy_gradient,
        )


def test_compact_cnn_training_step_path_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)
    numpy_loss = SoftmaxCrossEntropyLoss(backend="numpy")
    cupy_loss = SoftmaxCrossEntropyLoss(backend=cp)

    _assert_parameters_match("initial", numpy_model, cupy_model)

    logits_np = numpy_model.forward(images_np)
    logits_cp = cupy_model.forward(images_cp)
    assert is_cupy_array(logits_cp)
    assert is_cupy_array(cupy_model.gradcam_activation)
    _assert_allclose_named("logits", logits_cp, logits_np)

    loss_np = numpy_loss.forward(logits_np, labels_np)
    loss_cp = cupy_loss.forward(logits_cp, labels_cp)
    loss_diff = abs(loss_cp - loss_np)
    assert loss_cp == pytest.approx(
        loss_np,
        rel=LOSS_RTOL,
        abs=LOSS_ATOL,
    ), f"loss max_abs_diff={loss_diff:.8e}"

    grad_logits_np = numpy_loss.backward()
    grad_logits_cp = cupy_loss.backward()
    assert is_cupy_array(grad_logits_cp)
    _assert_allclose_named("grad_logits", grad_logits_cp, grad_logits_np)

    numpy_model.backward(grad_logits_np)
    cupy_model.backward(grad_logits_cp)
    _assert_named_gradients_match(numpy_model, cupy_model)

    SGD(learning_rate=LEARNING_RATE).step(
        numpy_model.named_parameters_and_gradients()
    )
    SGD(learning_rate=LEARNING_RATE).step(
        cupy_model.named_parameters_and_gradients()
    )
    _assert_parameters_match("updated", numpy_model, cupy_model)


def test_train_step_helper_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _synthetic_batch(cp)

    _assert_parameters_match("helper initial", numpy_model, cupy_model)
    loss_np = train_step(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        SGD(learning_rate=LEARNING_RATE),
        images_np,
        labels_np,
    )
    loss_cp = train_step(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        SGD(learning_rate=LEARNING_RATE),
        images_cp,
        labels_cp,
    )

    loss_diff = abs(loss_cp - loss_np)
    assert loss_cp == pytest.approx(
        loss_np,
        rel=LOSS_RTOL,
        abs=LOSS_ATOL,
    ), f"train_step loss max_abs_diff={loss_diff:.8e}"
    _assert_named_gradients_match(numpy_model, cupy_model)
    _assert_parameters_match("helper updated", numpy_model, cupy_model)
