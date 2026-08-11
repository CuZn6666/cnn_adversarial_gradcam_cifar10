import numpy as np
import pytest

from src.attacks import fgsm_attack
from src.backend import is_cupy_array, to_backend, to_numpy
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.robustness import (
    evaluate_fgsm_batch,
    evaluate_fgsm_batches,
    evaluate_fgsm_epsilon_sweep,
)


pytestmark = pytest.mark.requires_cupy


METRIC_RTOL = 1e-6
METRIC_ATOL = 1e-7
TENSOR_RTOL = 1e-5
TENSOR_ATOL = 1e-6
EPSILONS = (0.0, 4.0 / 255.0, 8.0 / 255.0)

COUNT_FIELDS = (
    "total_samples",
    "clean_correct",
    "adversarial_correct",
    "clean_correct_samples",
    "successful_attacks",
)
METRIC_FIELDS = (
    "clean_accuracy",
    "adversarial_accuracy",
    "accuracy_drop",
    "attack_success_rate",
)


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


def _synthetic_images(batch_size: int, start: float, stop: float) -> np.ndarray:
    images = np.linspace(
        start,
        stop,
        num=batch_size * 3 * 32 * 32,
        dtype=np.float32,
    ).reshape(batch_size, 3, 32, 32)
    images[0, 0, 0, 0] = 0.0
    images[-1, 2, -1, -1] = 1.0
    return images


def _single_batch(cp):
    images_np = _synthetic_images(batch_size=2, start=0.02, stop=0.98)
    labels_np = np.array([9, 0], dtype=np.int64)
    return (
        images_np,
        labels_np,
        to_backend(images_np, cp),
        to_backend(labels_np, cp),
    )


def _numpy_batches():
    return (
        (
            _synthetic_images(batch_size=1, start=0.10, stop=0.90),
            np.array([9], dtype=np.int64),
        ),
        (
            _synthetic_images(batch_size=3, start=0.03, stop=0.87),
            np.array([9, 0, 1], dtype=np.int64),
        ),
    )


def _cupy_batches(cp):
    return tuple(
        (to_backend(images, cp), to_backend(labels, cp))
        for images, labels in _numpy_batches()
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


def _assert_batch_results_match(label: str, cupy_result, numpy_result) -> None:
    assert set(cupy_result) == set(numpy_result)

    for field in COUNT_FIELDS:
        assert cupy_result[field] == numpy_result[field], (
            f"{label} {field}: {cupy_result[field]} != {numpy_result[field]}"
        )

    for field in METRIC_FIELDS:
        assert cupy_result[field] == pytest.approx(
            numpy_result[field],
            rel=METRIC_RTOL,
            abs=METRIC_ATOL,
        ), (
            f"{label} {field}: {cupy_result[field]:.8e} != "
            f"{numpy_result[field]:.8e}"
        )


def _assert_sweep_results_match(cupy_results, numpy_results) -> None:
    assert len(cupy_results) == len(numpy_results)
    assert [result["epsilon"] for result in cupy_results] == [
        result["epsilon"] for result in numpy_results
    ]

    for index, (cupy_result, numpy_result) in enumerate(
        zip(cupy_results, numpy_results),
    ):
        assert cupy_result["epsilon"] == numpy_result["epsilon"]
        _assert_batch_results_match(
            f"epsilon index {index} ({cupy_result['epsilon']})",
            cupy_result,
            numpy_result,
        )


def _assert_zero_epsilon_invariants(
    numpy_model: CompactCNN,
    cupy_model: _RecordingCompactCNN,
    images_np,
    labels_np,
    images_cp,
    labels_cp,
) -> None:
    grad_input_np = compute_input_gradient(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
    )
    grad_input_cp = compute_input_gradient(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cupy_model.xp),
        images_cp,
        labels_cp,
    )
    assert is_cupy_array(grad_input_cp)
    _assert_allclose_named("epsilon=0 input gradients", grad_input_cp, grad_input_np)

    adversarial_np = fgsm_attack(images_np, grad_input_np, epsilon=0.0)
    adversarial_cp = fgsm_attack(images_cp, grad_input_cp, epsilon=0.0)
    assert is_cupy_array(adversarial_cp)
    np.testing.assert_array_equal(adversarial_np, images_np)
    np.testing.assert_array_equal(to_numpy(adversarial_cp), images_np)

    clean_logits_np = numpy_model.forward(images_np)
    clean_logits_cp = cupy_model.forward(images_cp)
    adversarial_logits_np = numpy_model.forward(adversarial_np)
    adversarial_logits_cp = cupy_model.forward(adversarial_cp)
    assert is_cupy_array(clean_logits_cp)
    assert is_cupy_array(adversarial_logits_cp)
    _assert_allclose_named(
        "epsilon=0 clean logits",
        clean_logits_cp,
        clean_logits_np,
    )
    _assert_allclose_named(
        "epsilon=0 adversarial logits",
        adversarial_logits_cp,
        adversarial_logits_np,
    )

    clean_predictions_np = np.argmax(clean_logits_np, axis=1)
    adversarial_predictions_np = np.argmax(adversarial_logits_np, axis=1)
    clean_predictions_cp = cupy_model.xp.argmax(clean_logits_cp, axis=1)
    adversarial_predictions_cp = cupy_model.xp.argmax(
        adversarial_logits_cp,
        axis=1,
    )
    assert is_cupy_array(clean_predictions_cp)
    assert is_cupy_array(adversarial_predictions_cp)
    np.testing.assert_array_equal(clean_predictions_np, adversarial_predictions_np)
    np.testing.assert_array_equal(
        to_numpy(clean_predictions_cp),
        to_numpy(adversarial_predictions_cp),
    )
    np.testing.assert_array_equal(to_numpy(clean_predictions_cp), clean_predictions_np)


def test_evaluate_fgsm_batch_metrics_and_zero_epsilon_match_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    images_np, labels_np, images_cp, labels_cp = _single_batch(cp)
    numpy_before = _parameter_snapshots(numpy_model)
    cupy_before = _parameter_snapshots(cupy_model)

    _assert_parameters_match("initial", numpy_model, cupy_model)

    zero_numpy = evaluate_fgsm_batch(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
        epsilon=0.0,
    )
    zero_cupy = evaluate_fgsm_batch(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        images_cp,
        labels_cp,
        epsilon=0.0,
    )
    _assert_batch_results_match("epsilon=0 batch", zero_cupy, zero_numpy)
    assert zero_numpy["adversarial_accuracy"] == zero_numpy["clean_accuracy"]
    assert zero_cupy["adversarial_accuracy"] == zero_cupy["clean_accuracy"]
    assert zero_numpy["successful_attacks"] == 0
    assert zero_cupy["successful_attacks"] == 0

    epsilon = EPSILONS[1]
    nonzero_numpy = evaluate_fgsm_batch(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        images_np,
        labels_np,
        epsilon=epsilon,
    )
    nonzero_cupy = evaluate_fgsm_batch(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        images_cp,
        labels_cp,
        epsilon=epsilon,
    )
    _assert_batch_results_match("nonzero-epsilon batch", nonzero_cupy, nonzero_numpy)

    _assert_zero_epsilon_invariants(
        numpy_model,
        cupy_model,
        images_np,
        labels_np,
        images_cp,
        labels_cp,
    )
    _assert_recorded_cupy_tensors(cupy_model)
    _assert_parameters_unchanged(
        "single-batch NumPy",
        numpy_model,
        numpy_before,
    )
    _assert_parameters_unchanged(
        "single-batch CuPy",
        cupy_model,
        cupy_before,
        expect_cupy=True,
    )


def test_evaluate_fgsm_batches_aggregation_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    numpy_batches = _numpy_batches()
    cupy_batches = _cupy_batches(cp)
    numpy_before = _parameter_snapshots(numpy_model)
    cupy_before = _parameter_snapshots(cupy_model)
    epsilon = EPSILONS[2]

    numpy_result = evaluate_fgsm_batches(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        numpy_batches,
        epsilon=epsilon,
    )
    cupy_result = evaluate_fgsm_batches(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        cupy_batches,
        epsilon=epsilon,
    )

    _assert_batch_results_match("multi-batch aggregation", cupy_result, numpy_result)
    assert numpy_result["total_samples"] == 4
    assert cupy_result["total_samples"] == 4

    first_batch_clean_accuracy = 1.0
    second_batch_clean_accuracy = 1.0 / 3.0
    naive_clean_accuracy = (
        first_batch_clean_accuracy + second_batch_clean_accuracy
    ) / 2.0
    assert not np.isclose(numpy_result["clean_accuracy"], naive_clean_accuracy)
    assert cupy_result["clean_accuracy"] == pytest.approx(
        numpy_result["clean_accuracy"],
        rel=METRIC_RTOL,
        abs=METRIC_ATOL,
    )

    _assert_recorded_cupy_tensors(cupy_model)
    _assert_parameters_unchanged(
        "multi-batch NumPy",
        numpy_model,
        numpy_before,
    )
    _assert_parameters_unchanged(
        "multi-batch CuPy",
        cupy_model,
        cupy_before,
        expect_cupy=True,
    )


def test_evaluate_fgsm_epsilon_sweep_matches_numpy(cp) -> None:
    numpy_model, cupy_model = _synchronized_models(cp)
    numpy_batches = _numpy_batches()
    cupy_batches = _cupy_batches(cp)
    numpy_before = _parameter_snapshots(numpy_model)
    cupy_before = _parameter_snapshots(cupy_model)

    numpy_results = evaluate_fgsm_epsilon_sweep(
        numpy_model,
        SoftmaxCrossEntropyLoss(backend="numpy"),
        numpy_batches,
        EPSILONS,
    )
    cupy_results = evaluate_fgsm_epsilon_sweep(
        cupy_model,
        SoftmaxCrossEntropyLoss(backend=cp),
        cupy_batches,
        EPSILONS,
    )

    _assert_sweep_results_match(cupy_results, numpy_results)
    assert [result["epsilon"] for result in numpy_results] == list(EPSILONS)
    zero_result = numpy_results[0]
    assert zero_result["epsilon"] == 0.0
    assert zero_result["clean_accuracy"] == zero_result["adversarial_accuracy"]
    assert zero_result["accuracy_drop"] == 0.0
    assert zero_result["attack_success_rate"] == 0.0
    assert zero_result["successful_attacks"] == 0

    _assert_recorded_cupy_tensors(cupy_model)
    _assert_parameters_unchanged(
        "epsilon-sweep NumPy",
        numpy_model,
        numpy_before,
    )
    _assert_parameters_unchanged(
        "epsilon-sweep CuPy",
        cupy_model,
        cupy_before,
        expect_cupy=True,
    )
