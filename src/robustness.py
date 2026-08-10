"""Small robustness-evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypedDict

import numpy as np

from src.attacks import fgsm_attack
from src.backend import ensure_same_backend, to_numpy, to_python_int
from src.input_gradients import compute_input_gradient
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN


class FGSMBatchResult(TypedDict):
    """Metrics from one clean and adversarial batch evaluation."""

    total_samples: int
    clean_correct: int
    adversarial_correct: int
    clean_correct_samples: int
    successful_attacks: int
    clean_accuracy: float
    adversarial_accuracy: float
    accuracy_drop: float
    attack_success_rate: float


class FGSMSweepResult(FGSMBatchResult):
    """Metrics for one epsilon in an FGSM robustness sweep."""

    epsilon: float


class FGSMRepresentativeExample(TypedDict):
    """Metadata for one representative FGSM example."""

    global_sample_index: int
    batch_index: int
    index_in_batch: int
    true_label: int
    clean_prediction: int
    adversarial_prediction: int
    epsilon: float
    example_type: Literal["successful", "failed"]


class FGSMRepresentativeExamples(TypedDict):
    """Selected successful and failed FGSM examples."""

    successful: list[FGSMRepresentativeExample]
    failed: list[FGSMRepresentativeExample]


def evaluate_fgsm_batch(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
) -> FGSMBatchResult:
    """Evaluate clean and FGSM predictions for one batch."""
    clean_logits = model.forward(images)
    xp = ensure_same_backend(clean_logits, labels)
    clean_predictions = xp.argmax(clean_logits, axis=1)

    grad_input = compute_input_gradient(
        model,
        loss_function,
        images,
        labels,
    )
    adversarial_images = fgsm_attack(images, grad_input, epsilon)
    adversarial_logits = model.forward(adversarial_images)
    adversarial_predictions = xp.argmax(adversarial_logits, axis=1)

    total_samples = labels.shape[0]
    clean_correct_mask = clean_predictions == labels
    adversarial_correct_mask = adversarial_predictions == labels
    clean_correct = to_python_int(xp.sum(clean_correct_mask))
    adversarial_correct = to_python_int(xp.sum(adversarial_correct_mask))
    successful_attacks = to_python_int(
        xp.sum(clean_correct_mask & ~adversarial_correct_mask)
    )

    clean_accuracy = clean_correct / total_samples
    adversarial_accuracy = adversarial_correct / total_samples
    attack_success_rate = (
        successful_attacks / clean_correct
        if clean_correct > 0
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "clean_correct_samples": clean_correct,
        "successful_attacks": successful_attacks,
        "clean_accuracy": float(clean_accuracy),
        "adversarial_accuracy": float(adversarial_accuracy),
        "accuracy_drop": float(clean_accuracy - adversarial_accuracy),
        "attack_success_rate": float(attack_success_rate),
    }


def select_fgsm_representative_examples(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    epsilon: float,
    max_successful: int = 1,
    max_failed: int = 1,
) -> FGSMRepresentativeExamples:
    """Select the first clean-correct successful and failed FGSM examples."""
    for name, value in (
        ("max_successful", max_successful),
        ("max_failed", max_failed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    batch_values = tuple(batches)
    total_samples = sum(
        labels.shape[0] for _, labels in batch_values
    )
    if not batch_values or total_samples == 0:
        raise ValueError(
            "select_fgsm_representative_examples requires at least one sample."
        )

    selected: FGSMRepresentativeExamples = {
        "successful": [],
        "failed": [],
    }
    global_offset = 0

    for batch_index, (images, labels) in enumerate(batch_values):
        clean_logits = model.forward(images)
        xp = ensure_same_backend(clean_logits, labels)
        clean_predictions = xp.argmax(clean_logits, axis=1)
        grad_input = compute_input_gradient(
            model,
            loss_function,
            images,
            labels,
        )
        adversarial_images = fgsm_attack(images, grad_input, epsilon)
        adversarial_logits = model.forward(adversarial_images)
        adversarial_predictions = xp.argmax(adversarial_logits, axis=1)

        labels_cpu = to_numpy(labels)
        clean_predictions_cpu = to_numpy(clean_predictions)
        adversarial_predictions_cpu = to_numpy(adversarial_predictions)

        for index_in_batch, true_label in enumerate(labels_cpu):
            clean_prediction = int(clean_predictions_cpu[index_in_batch])
            adversarial_prediction = int(
                adversarial_predictions_cpu[index_in_batch]
            )
            true_label_value = int(true_label)

            if clean_prediction != true_label_value:
                continue

            if adversarial_prediction != true_label_value:
                example_type: Literal["successful", "failed"] = "successful"
                target = selected["successful"]
                limit = max_successful
            else:
                example_type = "failed"
                target = selected["failed"]
                limit = max_failed

            if len(target) < limit:
                target.append(
                    {
                        "global_sample_index": global_offset + index_in_batch,
                        "batch_index": batch_index,
                        "index_in_batch": index_in_batch,
                        "true_label": true_label_value,
                        "clean_prediction": clean_prediction,
                        "adversarial_prediction": adversarial_prediction,
                        "epsilon": float(epsilon),
                        "example_type": example_type,
                    }
                )

        global_offset += labels.shape[0]
        if (
            len(selected["successful"]) >= max_successful
            and len(selected["failed"]) >= max_failed
        ):
            break

    return selected


def evaluate_fgsm_epsilon_sweep(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    epsilons: Iterable[float],
) -> list[FGSMSweepResult]:
    """Evaluate multiple FGSM epsilon values in the provided order."""
    epsilon_values = list(epsilons)
    if not epsilon_values:
        raise ValueError(
            "evaluate_fgsm_epsilon_sweep requires at least one epsilon."
        )

    batch_values = tuple(batches)
    results: list[FGSMSweepResult] = []
    for epsilon in epsilon_values:
        batch_result = evaluate_fgsm_batches(
            model,
            loss_function,
            batch_values,
            epsilon,
        )
        results.append(
            {
                "epsilon": float(epsilon),
                **batch_result,
            }
        )
    return results


def evaluate_fgsm_batches(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    epsilon: float,
) -> FGSMBatchResult:
    """Aggregate clean and FGSM metrics over multiple batches."""
    total_samples = 0
    clean_correct = 0
    adversarial_correct = 0
    clean_correct_samples = 0
    successful_attacks = 0

    for images, labels in batches:
        batch_result = evaluate_fgsm_batch(
            model,
            loss_function,
            images,
            labels,
            epsilon,
        )
        total_samples += batch_result["total_samples"]
        clean_correct += batch_result["clean_correct"]
        adversarial_correct += batch_result["adversarial_correct"]
        clean_correct_samples += batch_result["clean_correct_samples"]
        successful_attacks += batch_result["successful_attacks"]

    if total_samples == 0:
        raise ValueError("evaluate_fgsm_batches requires at least one sample.")

    clean_accuracy = clean_correct / total_samples
    adversarial_accuracy = adversarial_correct / total_samples
    attack_success_rate = (
        successful_attacks / clean_correct_samples
        if clean_correct_samples > 0
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "clean_correct": clean_correct,
        "adversarial_correct": adversarial_correct,
        "clean_correct_samples": clean_correct_samples,
        "successful_attacks": successful_attacks,
        "clean_accuracy": float(clean_accuracy),
        "adversarial_accuracy": float(adversarial_accuracy),
        "accuracy_drop": float(clean_accuracy - adversarial_accuracy),
        "attack_success_rate": float(attack_success_rate),
    }
