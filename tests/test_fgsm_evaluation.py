import numpy as np

from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.robustness import evaluate_fgsm_batch


class _ControlledModel:
    def __init__(
        self,
        clean_predictions: list[int],
        adversarial_predictions: list[int],
    ) -> None:
        self.clean_logits = self._logits(clean_predictions)
        self.adversarial_logits = self._logits(adversarial_predictions)
        self._input_shape: tuple[int, ...] | None = None

    @staticmethod
    def _logits(predictions: list[int]) -> np.ndarray:
        logits = np.zeros((len(predictions), 10), dtype=np.float32)
        logits[np.arange(len(predictions)), predictions] = 1.0
        return logits

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_shape = inputs.shape
        if np.any(inputs != 0.0):
            return self.adversarial_logits
        return self.clean_logits

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward.")
        return np.ones(self._input_shape, dtype=np.float32)


def _model_parameters(model: CompactCNN) -> tuple[np.ndarray, ...]:
    return (
        model.conv1.weights,
        model.conv1.bias,
        model.conv2.weights,
        model.conv2.bias,
        model.classifier.weights,
        model.classifier.bias,
    )


def test_evaluate_fgsm_batch_returns_hand_computed_metrics() -> None:
    images = np.zeros((4, 3, 32, 32), dtype=np.float32)
    labels = np.array([0, 1, 9, 3], dtype=np.int64)
    model = _ControlledModel(
        clean_predictions=[0, 1, 2, 3],
        adversarial_predictions=[9, 1, 2, 8],
    )

    result = evaluate_fgsm_batch(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.1,
    )

    assert set(result) == {
        "total_samples",
        "clean_correct",
        "adversarial_correct",
        "clean_correct_samples",
        "successful_attacks",
        "clean_accuracy",
        "adversarial_accuracy",
        "accuracy_drop",
        "attack_success_rate",
    }
    assert result == {
        "total_samples": 4,
        "clean_correct": 3,
        "adversarial_correct": 1,
        "clean_correct_samples": 3,
        "successful_attacks": 2,
        "clean_accuracy": 0.75,
        "adversarial_accuracy": 0.25,
        "accuracy_drop": 0.5,
        "attack_success_rate": 2.0 / 3.0,
    }


def test_evaluate_fgsm_batch_epsilon_zero_matches_clean_predictions() -> None:
    images = np.zeros((3, 3, 32, 32), dtype=np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)
    model = _ControlledModel(
        clean_predictions=[0, 4, 2],
        adversarial_predictions=[9, 9, 9],
    )

    result = evaluate_fgsm_batch(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.0,
    )

    assert result["clean_correct"] == 2
    assert result["adversarial_correct"] == 2
    assert result["successful_attacks"] == 0
    assert result["clean_accuracy"] == result["adversarial_accuracy"]
    assert result["accuracy_drop"] == 0.0
    assert result["attack_success_rate"] == 0.0


def test_evaluate_fgsm_batch_zero_clean_correct_has_zero_success_rate() -> None:
    images = np.zeros((2, 3, 32, 32), dtype=np.float32)
    labels = np.array([4, 5], dtype=np.int64)
    model = _ControlledModel(
        clean_predictions=[0, 0],
        adversarial_predictions=[1, 1],
    )

    result = evaluate_fgsm_batch(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.1,
    )

    assert result["clean_correct_samples"] == 0
    assert result["successful_attacks"] == 0
    assert result["attack_success_rate"] == 0.0


def test_evaluate_fgsm_batch_does_not_update_model_parameters() -> None:
    model = CompactCNN(seed=42)
    images = np.random.default_rng(17).random(
        (2, 3, 32, 32),
        dtype=np.float32,
    )
    labels = np.array([2, 5], dtype=np.int64)
    parameters_before = [
        parameter.copy() for parameter in _model_parameters(model)
    ]

    result = evaluate_fgsm_batch(
        model,
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        epsilon=0.05,
    )

    assert result["total_samples"] == images.shape[0]
    for parameter, parameter_before in zip(
        _model_parameters(model),
        parameters_before,
        strict=True,
    ):
        np.testing.assert_array_equal(parameter, parameter_before)
