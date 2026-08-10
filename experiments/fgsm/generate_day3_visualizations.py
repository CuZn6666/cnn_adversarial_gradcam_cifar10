"""Generate FGSM qualitative visualizations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from configs.default_config import (
    CIFAR10_EXTRACTED_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.attacks import fgsm_attack
from src.checkpointing import load_checkpoint
from src.data.cifar10_loader import load_cifar10
from src.input_gradients import compute_input_gradient, input_gradient_map
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN
from src.visualization import (
    save_fgsm_epsilon_progression,
    save_fgsm_qualitative_comparison,
)


DAY3_EPSILON_VALUES = (
    0.0,
    2.0 / 255.0,
    4.0 / 255.0,
    8.0 / 255.0,
    16.0 / 255.0,
)
DAY3_EPSILON_LABELS = ("0", "2/255", "4/255", "8/255", "16/255")
DAY3_TARGET_EPSILON = 8.0 / 255.0
DAY3_TARGET_EPSILON_LABEL = "8/255"


@dataclass(frozen=True)
class Day3FGSMVisualizationConfig:
    """Deterministic qualitative FGSM configuration."""

    eval_samples: int = 1024
    seed: int = SEED
    checkpoint_path: Path | str = (
        PROJECT_ROOT / "results" / "baseline" / "portfolio_baseline_best.npz"
    )
    output_dir: Path | str = PROJECT_ROOT / "results" / "fgsm"
    comparison_epsilon: float = DAY3_TARGET_EPSILON
    comparison_epsilon_label: str = DAY3_TARGET_EPSILON_LABEL
    epsilon_values: tuple[float, ...] = DAY3_EPSILON_VALUES
    epsilon_labels: tuple[str, ...] = DAY3_EPSILON_LABELS

    def __post_init__(self) -> None:
        if (
            isinstance(self.eval_samples, bool)
            or not isinstance(self.eval_samples, int)
            or self.eval_samples <= 0
        ):
            raise ValueError("eval_samples must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if len(self.epsilon_values) != len(self.epsilon_labels):
            raise ValueError(
                "epsilon_values and epsilon_labels must have the same length."
            )
        for epsilon in (*self.epsilon_values, self.comparison_epsilon):
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0.0
            ):
                raise ValueError("epsilon values must be non-negative finite numbers.")
        if not isinstance(self.comparison_epsilon_label, str):
            raise ValueError("comparison_epsilon_label must be a string.")
        for label in self.epsilon_labels:
            if not isinstance(label, str) or not label:
                raise ValueError("epsilon_labels must be non-empty strings.")
        for field_name in ("checkpoint_path", "output_dir"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))


DAY3_FGSM_VISUALIZATION_CONFIG = Day3FGSMVisualizationConfig()


class SelectedFGSMExample(TypedDict):
    """Metadata and arrays for one deterministic FGSM qualitative example."""

    subset_position: int
    original_test_index: int
    true_label: int
    true_class: str
    clean_prediction: int
    clean_prediction_class: str
    adversarial_prediction: int
    adversarial_prediction_class: str
    epsilon: float
    epsilon_label: str
    clean_image: np.ndarray
    adversarial_image: np.ndarray
    grad_input: np.ndarray


def _select_deterministic_subset_with_indices(
    images: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_count > images.shape[0]:
        raise ValueError(
            f"eval_samples {sample_count} exceeds dataset size {images.shape[0]}."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=sample_count, replace=False)
    return images[indices], labels[indices], indices


def _predict_class(model: CompactCNN, image: np.ndarray) -> int:
    logits = model.forward(image)
    return int(np.argmax(logits, axis=1)[0])


def select_representative_fgsm_example(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    class_names: list[str],
    epsilon: float = DAY3_TARGET_EPSILON,
    epsilon_label: str = DAY3_TARGET_EPSILON_LABEL,
) -> SelectedFGSMExample:
    """Select first clean-correct sample that FGSM flips at target epsilon."""
    if images.shape[0] != labels.shape[0] or labels.shape[0] != original_indices.shape[0]:
        raise ValueError("images, labels, and original_indices must align.")
    if not class_names:
        raise ValueError("class_names must not be empty.")

    for subset_position in range(images.shape[0]):
        clean_image = images[subset_position : subset_position + 1]
        label = labels[subset_position : subset_position + 1]
        true_label = int(label[0])
        clean_prediction = _predict_class(model, clean_image)
        if clean_prediction != true_label:
            continue

        grad_input = compute_input_gradient(
            model,
            loss_function,
            clean_image,
            label,
        )
        adversarial_image = fgsm_attack(clean_image, grad_input, epsilon)
        adversarial_prediction = _predict_class(model, adversarial_image)
        if adversarial_prediction == true_label:
            continue

        return {
            "subset_position": subset_position,
            "original_test_index": int(original_indices[subset_position]),
            "true_label": true_label,
            "true_class": class_names[true_label],
            "clean_prediction": clean_prediction,
            "clean_prediction_class": class_names[clean_prediction],
            "adversarial_prediction": adversarial_prediction,
            "adversarial_prediction_class": class_names[adversarial_prediction],
            "epsilon": float(epsilon),
            "epsilon_label": epsilon_label,
            "clean_image": clean_image,
            "adversarial_image": adversarial_image,
            "grad_input": grad_input,
        }

    raise RuntimeError(
        "No clean-correct sample was successfully attacked at the target epsilon."
    )


def _progression_images_and_predictions(
    model: CompactCNN,
    clean_image: np.ndarray,
    true_label: int,
    grad_input: np.ndarray,
    epsilon_values: tuple[float, ...],
) -> tuple[list[np.ndarray], list[int]]:
    images_by_epsilon: list[np.ndarray] = []
    predictions: list[int] = []
    label = np.array([true_label], dtype=np.int64)
    for epsilon in epsilon_values:
        adversarial_image = fgsm_attack(clean_image, grad_input, epsilon)
        if epsilon == 0.0:
            np.testing.assert_allclose(adversarial_image, clean_image)
        perturbation = np.max(np.abs(adversarial_image - clean_image))
        if perturbation > epsilon + 1e-6:
            raise RuntimeError("FGSM perturbation exceeded the L-infinity bound.")
        images_by_epsilon.append(adversarial_image)
        predictions.append(_predict_class(model, adversarial_image))
    # Keep label alive for explicit dtype documentation and future validation.
    _ = label
    return images_by_epsilon, predictions


def run_day3_fgsm_visualizations(
    config: Day3FGSMVisualizationConfig = DAY3_FGSM_VISUALIZATION_CONFIG,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Generate qualitative FGSM comparison and epsilon progression."""
    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint is not available at {config.checkpoint_path}."
        )

    data_path = Path(data_dir)
    dataset_path = data_path / CIFAR10_EXTRACTED_DIR
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            "CIFAR-10 data is not available at "
            f"{dataset_path}. Download and extract it explicitly before "
            "running FGSM qualitative visualizations."
        )

    _, _, test_images, test_labels, class_names = load_cifar10(data_path)
    subset_images, subset_labels, original_indices = (
        _select_deterministic_subset_with_indices(
            test_images,
            test_labels,
            config.eval_samples,
            config.seed,
        )
    )

    model = CompactCNN(seed=config.seed)
    load_checkpoint(model, config.checkpoint_path)
    loss_function = SoftmaxCrossEntropyLoss()
    selected = select_representative_fgsm_example(
        model,
        loss_function,
        subset_images,
        subset_labels,
        original_indices,
        class_names,
        epsilon=config.comparison_epsilon,
        epsilon_label=config.comparison_epsilon_label,
    )

    output_dir = Path(config.output_dir)
    gradient_map = input_gradient_map(selected["grad_input"])
    qualitative_path = save_fgsm_qualitative_comparison(
        selected["clean_image"],
        selected["adversarial_image"],
        gradient_map,
        output_dir / "fgsm_qualitative_comparison.png",
        true_label=selected["true_class"],
        clean_prediction=selected["clean_prediction_class"],
        adversarial_prediction=selected["adversarial_prediction_class"],
        epsilon_label=config.comparison_epsilon_label,
    )

    images_by_epsilon, prediction_ids = _progression_images_and_predictions(
        model,
        selected["clean_image"],
        selected["true_label"],
        selected["grad_input"],
        config.epsilon_values,
    )
    prediction_classes = [class_names[prediction] for prediction in prediction_ids]
    progression_path = save_fgsm_epsilon_progression(
        images_by_epsilon,
        config.epsilon_labels,
        prediction_classes,
        output_dir / "epsilon_progression.png",
        true_label=selected["true_class"],
    )

    try:
        checkpoint_for_metadata = str(
            config.checkpoint_path.resolve().relative_to(PROJECT_ROOT)
        )
    except ValueError:
        checkpoint_for_metadata = str(config.checkpoint_path)

    metadata = {
        "checkpoint": checkpoint_for_metadata,
        "seed": config.seed,
        "eval_samples": config.eval_samples,
        "sample_selection_rule": (
            "first deterministic test-subset sample that is clean-correct "
            "and becomes incorrect under FGSM at epsilon 8/255"
        ),
        "subset_position": selected["subset_position"],
        "original_test_index": selected["original_test_index"],
        "true_label": selected["true_label"],
        "true_class": selected["true_class"],
        "clean_prediction": selected["clean_prediction"],
        "clean_prediction_class": selected["clean_prediction_class"],
        "adversarial_prediction": selected["adversarial_prediction"],
        "adversarial_prediction_class": selected[
            "adversarial_prediction_class"
        ],
        "comparison_epsilon": config.comparison_epsilon,
        "comparison_epsilon_label": config.comparison_epsilon_label,
        "epsilon_values": list(config.epsilon_values),
        "epsilon_labels": list(config.epsilon_labels),
        "predictions_by_epsilon": [
            {
                "epsilon": float(epsilon),
                "epsilon_label": epsilon_label,
                "prediction": int(prediction),
                "prediction_class": prediction_class,
            }
            for epsilon, epsilon_label, prediction, prediction_class in zip(
                config.epsilon_values,
                config.epsilon_labels,
                prediction_ids,
                prediction_classes,
            )
        ],
    }
    metadata_path = save_metrics(
        metadata,
        output_dir / "fgsm_qualitative_metadata.json",
    )

    return {
        "qualitative_path": qualitative_path,
        "progression_path": progression_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def main() -> None:
    result = run_day3_fgsm_visualizations()
    print(result["metadata"])


if __name__ == "__main__":
    main()
