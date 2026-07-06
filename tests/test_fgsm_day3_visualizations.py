from pathlib import Path

import numpy as np
import pytest

from experiments.fgsm import generate_day3_visualizations as day3
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import load_metrics


class _SampleAwareModel:
    def __init__(
        self,
        base_values: list[float],
        clean_predictions: list[int],
        adversarial_predictions: list[int],
        adversarial_threshold: float = 0.01,
    ) -> None:
        self.base_values = np.array(base_values, dtype=np.float32)
        self.clean_predictions = clean_predictions
        self.adversarial_predictions = adversarial_predictions
        self.adversarial_threshold = adversarial_threshold
        self._input_shape: tuple[int, ...] | None = None

    @staticmethod
    def _logits(predictions: list[int]) -> np.ndarray:
        logits = np.zeros((len(predictions), 10), dtype=np.float32)
        logits[np.arange(len(predictions)), predictions] = 1.0
        return logits

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input_shape = inputs.shape
        predictions: list[int] = []
        for sample in inputs:
            sample_mean = float(sample.mean())
            sample_index = int(np.argmin(np.abs(sample_mean - self.base_values)))
            base_value = float(self.base_values[sample_index])
            is_adversarial = sample_mean > base_value + self.adversarial_threshold
            predictions.append(
                self.adversarial_predictions[sample_index]
                if is_adversarial
                else self.clean_predictions[sample_index]
            )
        return self._logits(predictions)

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward.")
        return np.ones(self._input_shape, dtype=np.float32)


def _constant_images(values: list[float]) -> np.ndarray:
    return np.array(
        [
            np.full((3, 4, 4), value, dtype=np.float32)
            for value in values
        ],
        dtype=np.float32,
    )


def test_day3_config_defaults_match_portfolio_requirements() -> None:
    config = day3.Day3FGSMVisualizationConfig()

    assert config.eval_samples == 1024
    assert config.seed == 42
    assert config.checkpoint_path.name == "portfolio_baseline_best.npz"
    assert config.output_dir.name == "fgsm"
    assert config.comparison_epsilon == 8.0 / 255.0
    assert config.comparison_epsilon_label == "8/255"
    assert config.epsilon_values == (
        0.0,
        2.0 / 255.0,
        4.0 / 255.0,
        8.0 / 255.0,
        16.0 / 255.0,
    )
    assert config.epsilon_labels == ("0", "2/255", "4/255", "8/255", "16/255")


def test_select_representative_example_uses_first_clean_correct_attack() -> None:
    images = _constant_images([0.1, 0.3, 0.5])
    labels = np.array([0, 1, 2], dtype=np.int64)
    original_indices = np.array([10, 11, 12], dtype=np.int64)
    model = _SampleAwareModel(
        base_values=[0.1, 0.3, 0.5],
        clean_predictions=[9, 1, 2],
        adversarial_predictions=[9, 3, 4],
    )
    class_names = [f"class_{index}" for index in range(10)]

    selected = day3.select_representative_fgsm_example(
        model,  # type: ignore[arg-type]
        SoftmaxCrossEntropyLoss(),
        images,
        labels,
        original_indices,
        class_names,
        epsilon=0.1,
        epsilon_label="0.1",
    )

    assert selected["subset_position"] == 1
    assert selected["original_test_index"] == 11
    assert selected["true_label"] == 1
    assert selected["clean_prediction"] == 1
    assert selected["adversarial_prediction"] == 3
    assert selected["true_class"] == "class_1"
    assert selected["clean_prediction_class"] == "class_1"
    assert selected["adversarial_prediction_class"] == "class_3"


def test_select_representative_example_rejects_missing_successful_attack() -> None:
    images = _constant_images([0.1, 0.3])
    labels = np.array([0, 1], dtype=np.int64)
    original_indices = np.array([4, 5], dtype=np.int64)
    model = _SampleAwareModel(
        base_values=[0.1, 0.3],
        clean_predictions=[0, 1],
        adversarial_predictions=[0, 1],
    )

    with pytest.raises(RuntimeError, match="No clean-correct sample"):
        day3.select_representative_fgsm_example(
            model,  # type: ignore[arg-type]
            SoftmaxCrossEntropyLoss(),
            images,
            labels,
            original_indices,
            [f"class_{index}" for index in range(10)],
            epsilon=0.1,
        )


def test_progression_images_start_from_same_clean_image() -> None:
    clean_image = _constant_images([0.3])
    grad_input = np.ones_like(clean_image)
    model = _SampleAwareModel(
        base_values=[0.3],
        clean_predictions=[1],
        adversarial_predictions=[3],
    )

    images_by_epsilon, predictions = day3._progression_images_and_predictions(
        model,  # type: ignore[arg-type]
        clean_image,
        true_label=1,
        grad_input=grad_input,
        epsilon_values=(0.0, 0.05, 0.1),
    )

    np.testing.assert_allclose(images_by_epsilon[0], clean_image)
    np.testing.assert_allclose(images_by_epsilon[1], clean_image + 0.05)
    np.testing.assert_allclose(images_by_epsilon[2], clean_image + 0.1)
    assert predictions == [1, 3, 3]


def test_day3_runner_creates_figures_and_metadata_without_real_cifar10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "portfolio_baseline_best.npz"
    checkpoint_path.write_bytes(b"placeholder")
    data_dir = tmp_path / "data"
    (data_dir / day3.CIFAR10_EXTRACTED_DIR).mkdir(parents=True)
    config = day3.Day3FGSMVisualizationConfig(
        eval_samples=1,
        seed=7,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "fgsm",
        comparison_epsilon=0.1,
        comparison_epsilon_label="0.1",
        epsilon_values=(0.0, 0.05, 0.1),
        epsilon_labels=("0", "0.05", "0.1"),
    )

    def fake_load_cifar10(path: Path):
        del path
        empty_images = np.zeros((0, 3, 4, 4), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int64)
        test_images = _constant_images([0.3])
        test_labels = np.array([1], dtype=np.int64)
        class_names = [f"class_{index}" for index in range(10)]
        return empty_images, empty_labels, test_images, test_labels, class_names

    model = _SampleAwareModel(
        base_values=[0.3],
        clean_predictions=[1],
        adversarial_predictions=[3],
    )

    monkeypatch.setattr(day3, "load_cifar10", fake_load_cifar10)
    monkeypatch.setattr(day3, "CompactCNN", lambda seed: model)
    monkeypatch.setattr(day3, "load_checkpoint", lambda model, path: None)

    result = day3.run_day3_fgsm_visualizations(config, data_dir=data_dir)

    assert result["qualitative_path"].is_file()
    assert result["progression_path"].is_file()
    assert result["metadata_path"].is_file()
    assert result["qualitative_path"].stat().st_size > 0
    assert result["progression_path"].stat().st_size > 0
    metadata = load_metrics(result["metadata_path"])
    assert metadata["checkpoint"] == str(checkpoint_path)
    assert metadata["true_class"] == "class_1"
    assert metadata["clean_prediction_class"] == "class_1"
    assert metadata["adversarial_prediction_class"] == "class_3"
    assert [item["epsilon_label"] for item in metadata["predictions_by_epsilon"]] == [
        "0",
        "0.05",
        "0.1",
    ]
