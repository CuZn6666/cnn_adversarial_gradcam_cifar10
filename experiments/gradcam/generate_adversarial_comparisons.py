"""Generate clean-vs-adversarial Grad-CAM comparison figures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

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
from src.gradcam import compute_gradcam
from src.gradcam_visualization import (
    save_gradcam_detailed_comparison,
    save_gradcam_fixed_target_comparison,
    save_gradcam_hero_figure,
    save_gradcam_success_vs_control,
)
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN


GRADCAM_EPSILON = 8.0 / 255.0
GRADCAM_EPSILON_LABEL = "8/255"


@dataclass(frozen=True)
class AdversarialGradCAMConfig:
    """Controlled clean-vs-adversarial Grad-CAM generation settings."""

    seed: int = SEED
    scan_budget: int = 1000
    epsilon: float = GRADCAM_EPSILON
    epsilon_label: str = GRADCAM_EPSILON_LABEL
    checkpoint_path: Path | str = (
        PROJECT_ROOT / "results" / "baseline" / "portfolio_baseline_best.npz"
    )
    output_dir: Path | str = PROJECT_ROOT / "results" / "gradcam"
    max_success_examples: int = 6
    hero_examples: int = 3
    control_examples: int = 2

    def __post_init__(self) -> None:
        for field_name in ("seed", "scan_budget", "max_success_examples", "hero_examples", "control_examples"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(self.epsilon)
            or self.epsilon < 0.0
        ):
            raise ValueError("epsilon must be a non-negative finite number.")
        if not isinstance(self.epsilon_label, str) or not self.epsilon_label:
            raise ValueError("epsilon_label must be a non-empty string.")
        for field_name in ("checkpoint_path", "output_dir"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))


ADVERSARIAL_GRADCAM_CONFIG = AdversarialGradCAMConfig()


@dataclass
class GradCAMCandidate:
    """One clean-correct candidate with its FGSM adversarial result."""

    dataset_index: int
    true_label: int
    true_class: str
    clean_prediction: int
    clean_prediction_class: str
    adversarial_prediction: int
    adversarial_prediction_class: str
    clean_confidence: float
    adversarial_confidence: float
    adversarial_original_class_confidence: float
    confidence_drop: float
    status: Literal["attack_success", "attack_resisted"]
    clean_image: np.ndarray
    adversarial_image: np.ndarray


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _predict_with_probabilities(
    model: CompactCNN,
    image: np.ndarray,
) -> tuple[int, np.ndarray]:
    logits = model.forward(image)
    probabilities = _softmax(logits)
    return int(np.argmax(logits, axis=1)[0]), probabilities[0]


def _scan_candidates(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    config: AdversarialGradCAMConfig,
) -> tuple[list[GradCAMCandidate], list[GradCAMCandidate], dict[str, int]]:
    sample_count = min(config.scan_budget, test_images.shape[0])
    successes: list[GradCAMCandidate] = []
    controls: list[GradCAMCandidate] = []
    clean_correct_count = 0

    for dataset_index in range(sample_count):
        clean_image = test_images[dataset_index : dataset_index + 1]
        label = test_labels[dataset_index : dataset_index + 1]
        true_label = int(label[0])
        clean_prediction, clean_probabilities = _predict_with_probabilities(
            model,
            clean_image,
        )
        if clean_prediction != true_label:
            continue
        clean_correct_count += 1

        grad_input = compute_input_gradient(
            model,
            loss_function,
            clean_image,
            label,
        )
        adversarial_image = fgsm_attack(clean_image, grad_input, config.epsilon)
        adversarial_prediction, adversarial_probabilities = (
            _predict_with_probabilities(model, adversarial_image)
        )
        status: Literal["attack_success", "attack_resisted"] = (
            "attack_success"
            if adversarial_prediction != true_label
            else "attack_resisted"
        )
        clean_confidence = float(clean_probabilities[clean_prediction])
        adversarial_original_confidence = float(
            adversarial_probabilities[clean_prediction]
        )
        candidate = GradCAMCandidate(
            dataset_index=dataset_index,
            true_label=true_label,
            true_class=class_names[true_label],
            clean_prediction=clean_prediction,
            clean_prediction_class=class_names[clean_prediction],
            adversarial_prediction=adversarial_prediction,
            adversarial_prediction_class=class_names[adversarial_prediction],
            clean_confidence=clean_confidence,
            adversarial_confidence=float(
                adversarial_probabilities[adversarial_prediction]
            ),
            adversarial_original_class_confidence=adversarial_original_confidence,
            confidence_drop=float(clean_confidence - adversarial_original_confidence),
            status=status,
            clean_image=clean_image,
            adversarial_image=adversarial_image,
        )
        if status == "attack_success":
            successes.append(candidate)
        else:
            controls.append(candidate)

    return successes, controls, {
        "samples_scanned": sample_count,
        "clean_correct_count": clean_correct_count,
        "successful_attacks_found": len(successes),
        "controls_found": len(controls),
    }


def _diverse_selection(
    candidates: list[GradCAMCandidate],
    count: int,
) -> list[GradCAMCandidate]:
    selected: list[GradCAMCandidate] = []
    selected_indices: set[int] = set()
    selected_classes: set[int] = set()

    for candidate in candidates:
        if candidate.true_label in selected_classes:
            continue
        selected.append(candidate)
        selected_indices.add(candidate.dataset_index)
        selected_classes.add(candidate.true_label)
        if len(selected) == count:
            return selected

    for candidate in candidates:
        if candidate.dataset_index in selected_indices:
            continue
        selected.append(candidate)
        selected_indices.add(candidate.dataset_index)
        if len(selected) == count:
            return selected

    return selected


def _nonconstant_heatmap(heatmap: np.ndarray) -> bool:
    return bool(np.isfinite(heatmap).all() and np.max(heatmap) > np.min(heatmap))


def _with_gradcam_maps(
    model: CompactCNN,
    candidate: GradCAMCandidate,
) -> dict[str, Any] | None:
    clean_target = np.array([candidate.clean_prediction], dtype=np.int64)
    adversarial_target = np.array([candidate.adversarial_prediction], dtype=np.int64)

    clean_cam = compute_gradcam(model, candidate.clean_image, clean_target)
    adversarial_cam = compute_gradcam(
        model,
        candidate.adversarial_image,
        adversarial_target,
    )
    adversarial_original_class_cam = compute_gradcam(
        model,
        candidate.adversarial_image,
        clean_target,
    )
    if not (
        _nonconstant_heatmap(clean_cam)
        and _nonconstant_heatmap(adversarial_cam)
        and _nonconstant_heatmap(adversarial_original_class_cam)
    ):
        return None

    return {
        "dataset_index": candidate.dataset_index,
        "true_label": candidate.true_label,
        "true_class": candidate.true_class,
        "clean_prediction": candidate.clean_prediction,
        "clean_prediction_class": candidate.clean_prediction_class,
        "adversarial_prediction": candidate.adversarial_prediction,
        "adversarial_prediction_class": candidate.adversarial_prediction_class,
        "clean_confidence": candidate.clean_confidence,
        "adversarial_confidence": candidate.adversarial_confidence,
        "adversarial_original_class_confidence": (
            candidate.adversarial_original_class_confidence
        ),
        "confidence_drop": candidate.confidence_drop,
        "status": candidate.status,
        "clean_image": candidate.clean_image,
        "adversarial_image": candidate.adversarial_image,
        "clean_cam": clean_cam,
        "adversarial_cam": adversarial_cam,
        "clean_original_class_cam": clean_cam,
        "adversarial_original_class_cam": adversarial_original_class_cam,
        "adversarial_new_class_cam": adversarial_cam,
    }


def _select_usable_examples(
    model: CompactCNN,
    candidates: list[GradCAMCandidate],
    count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in _diverse_selection(candidates, len(candidates)):
        example = _with_gradcam_maps(model, candidate)
        if example is None:
            continue
        selected.append(example)
        if len(selected) == count:
            break
    return selected


def _metadata_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_index": example["dataset_index"],
        "true_label": example["true_label"],
        "true_class": example["true_class"],
        "clean_prediction": example["clean_prediction"],
        "clean_prediction_class": example["clean_prediction_class"],
        "adversarial_prediction": example["adversarial_prediction"],
        "adversarial_prediction_class": example["adversarial_prediction_class"],
        "clean_confidence": example["clean_confidence"],
        "adversarial_confidence": example["adversarial_confidence"],
        "adversarial_original_class_confidence": (
            example["adversarial_original_class_confidence"]
        ),
        "confidence_drop": example["confidence_drop"],
        "status": example["status"],
    }


def run_adversarial_gradcam_comparisons(
    config: AdversarialGradCAMConfig = ADVERSARIAL_GRADCAM_CONFIG,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Generate clean-vs-adversarial Grad-CAM figures and metadata."""
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
            "running adversarial Grad-CAM comparisons."
        )

    _, _, test_images, test_labels, class_names = load_cifar10(data_path)
    model = CompactCNN(seed=config.seed)
    load_checkpoint(model, config.checkpoint_path)
    loss_function = SoftmaxCrossEntropyLoss()

    successes, controls, counts = _scan_candidates(
        model,
        loss_function,
        test_images,
        test_labels,
        class_names,
        config,
    )
    selected_successes = _select_usable_examples(
        model,
        successes,
        config.max_success_examples,
    )
    selected_controls = _select_usable_examples(
        model,
        controls,
        config.control_examples,
    )
    if len(selected_successes) < config.hero_examples:
        raise RuntimeError(
            "Not enough usable successful FGSM examples for the README hero figure."
        )
    if len(selected_controls) < config.control_examples:
        raise RuntimeError(
            "Not enough usable attack-resisted control examples."
        )

    output_dir = Path(config.output_dir)
    hero_examples = selected_successes[: config.hero_examples]
    fixed_target_examples = selected_successes[: config.hero_examples]
    success_vs_control_examples = (
        selected_successes[:2] + selected_controls[: config.control_examples]
    )

    hero_path = save_gradcam_hero_figure(
        hero_examples,
        output_dir / "gradcam_hero.png",
        config.epsilon_label,
    )
    detailed_path = save_gradcam_detailed_comparison(
        selected_successes[0],
        output_dir / "gradcam_detailed_comparison.png",
        config.epsilon_label,
    )
    fixed_target_path = save_gradcam_fixed_target_comparison(
        fixed_target_examples,
        output_dir / "gradcam_fixed_target_comparison.png",
        config.epsilon_label,
    )
    success_control_path = save_gradcam_success_vs_control(
        success_vs_control_examples,
        output_dir / "gradcam_success_vs_control.png",
        config.epsilon_label,
    )

    try:
        checkpoint_for_metadata = str(
            config.checkpoint_path.resolve().relative_to(PROJECT_ROOT)
        )
    except ValueError:
        checkpoint_for_metadata = str(config.checkpoint_path)

    metadata = {
        "checkpoint": checkpoint_for_metadata,
        "dataset_split": "cifar10_test",
        "seed": config.seed,
        "epsilon": config.epsilon,
        "epsilon_label": config.epsilon_label,
        "scan_budget": config.scan_budget,
        **counts,
        "usable_success_examples": len(selected_successes),
        "usable_control_examples": len(selected_controls),
        "selection_rule": (
            "Scan CIFAR-10 test samples in deterministic dataset order; keep "
            "clean-correct samples; classify FGSM results at epsilon 8/255; "
            "select first valid examples by distinct true class where possible, "
            "then fill remaining slots in scan order; skip constant Grad-CAM maps."
        ),
        "gradcam_target_semantics": {
            "prediction_aligned": (
                "clean Grad-CAM target = clean prediction; adversarial Grad-CAM "
                "target = adversarial prediction"
            ),
            "fixed_original_target": (
                "clean and adversarial original-class maps target the clean "
                "prediction; adversarial new-class map targets adversarial prediction"
            ),
            "normalization_note": (
                "Grad-CAM maps are independently normalized to [0, 1]; compare "
                "spatial localization, not absolute activation magnitude."
            ),
        },
        "selected_successes": [
            _metadata_example(example) for example in selected_successes
        ],
        "selected_controls": [
            _metadata_example(example) for example in selected_controls
        ],
        "artifacts": {
            "hero": str(hero_path.relative_to(PROJECT_ROOT)),
            "detailed_comparison": str(detailed_path.relative_to(PROJECT_ROOT)),
            "fixed_target_comparison": str(fixed_target_path.relative_to(PROJECT_ROOT)),
            "success_vs_control": str(success_control_path.relative_to(PROJECT_ROOT)),
            "metadata": "results/gradcam/gradcam_comparison_metadata.json",
        },
    }
    metadata_path = save_metrics(
        metadata,
        output_dir / "gradcam_comparison_metadata.json",
    )
    metadata["artifacts"]["metadata"] = str(metadata_path.relative_to(PROJECT_ROOT))

    return {
        "hero_path": hero_path,
        "detailed_path": detailed_path,
        "fixed_target_path": fixed_target_path,
        "success_control_path": success_control_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def main() -> None:
    start = perf_counter()
    result = run_adversarial_gradcam_comparisons()
    elapsed = perf_counter() - start
    metadata = result["metadata"]
    print(
        {
            "elapsed_seconds": round(elapsed, 3),
            "samples_scanned": metadata["samples_scanned"],
            "clean_correct_count": metadata["clean_correct_count"],
            "successful_attacks_found": metadata["successful_attacks_found"],
            "controls_found": metadata["controls_found"],
            "usable_success_examples": metadata["usable_success_examples"],
            "usable_control_examples": metadata["usable_control_examples"],
            "selected_success_indices": [
                item["dataset_index"] for item in metadata["selected_successes"]
            ],
            "selected_control_indices": [
                item["dataset_index"] for item in metadata["selected_controls"]
            ],
        }
    )


if __name__ == "__main__":
    main()
