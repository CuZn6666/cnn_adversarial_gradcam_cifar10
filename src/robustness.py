"""Small robustness-evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import numpy as np

from src.attacks import fgsm_attack
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


def evaluate_fgsm_batch(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
) -> FGSMBatchResult:
    """Evaluate clean and FGSM predictions for one batch."""
    clean_logits = model.forward(images)
    clean_predictions = np.argmax(clean_logits, axis=1)

    grad_input = compute_input_gradient(
        model,
        loss_function,
        images,
        labels,
    )
    adversarial_images = fgsm_attack(images, grad_input, epsilon)
    adversarial_logits = model.forward(adversarial_images)
    adversarial_predictions = np.argmax(adversarial_logits, axis=1)

    total_samples = labels.shape[0]
    clean_correct_mask = clean_predictions == labels
    adversarial_correct_mask = adversarial_predictions == labels
    clean_correct = int(np.sum(clean_correct_mask))
    adversarial_correct = int(np.sum(adversarial_correct_mask))
    successful_attacks = int(
        np.sum(clean_correct_mask & ~adversarial_correct_mask)
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
