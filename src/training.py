from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.backend import ensure_same_backend, to_python_float, to_python_int
from src.losses import SoftmaxCrossEntropyLoss
from src.models import CompactCNN
from src.optimizers import SGD


def train_step(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    optimizer: SGD,
    images: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Run one forward, backward, and parameter-update step."""
    loss, _ = _train_step_metrics(
        model,
        loss_function,
        optimizer,
        images,
        labels,
    )
    return loss


def _train_step_metrics(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    optimizer: SGD,
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, int]:
    logits = model.forward(images)
    loss = loss_function.forward(logits, labels)
    xp = ensure_same_backend(logits, labels)
    correct_predictions = to_python_int(
        xp.sum(xp.argmax(logits, axis=1) == labels)
    )
    grad_logits = loss_function.backward()
    model.backward(grad_logits)
    parameter_gradient_pairs = model.named_parameters_and_gradients()
    optimizer.step(parameter_gradient_pairs)
    return loss, correct_predictions


def train_batches(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    optimizer: SGD,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    """Train on multiple batches and return sample-weighted metrics."""
    total_weighted_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in batches:
        batch_loss, batch_correct = _train_step_metrics(
            model,
            loss_function,
            optimizer,
            images,
            labels,
        )
        batch_size = labels.shape[0]
        total_weighted_loss += batch_loss * batch_size
        total_correct += batch_correct
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("train_batches requires at least one sample.")

    return {
        "mean_loss": float(total_weighted_loss / total_samples),
        "accuracy": float(total_correct / total_samples),
        "total_samples": total_samples,
    }


def evaluate_batch(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Evaluate one batch without updating model parameters."""
    logits = model.forward(images)
    loss = loss_function.forward(logits, labels)
    xp = ensure_same_backend(logits, labels)
    predictions = xp.argmax(logits, axis=1)
    accuracy = to_python_float(xp.mean(predictions == labels))
    return loss, accuracy


def evaluate_batches(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Evaluate multiple batches with sample-weighted aggregation."""
    total_weighted_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, labels in batches:
        batch_loss, batch_accuracy = evaluate_batch(
            model,
            loss_function,
            images,
            labels,
        )
        batch_size = labels.shape[0]
        total_weighted_loss += batch_loss * batch_size
        total_correct += batch_accuracy * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("evaluate_batches requires at least one sample.")

    mean_loss = total_weighted_loss / total_samples
    accuracy = total_correct / total_samples
    return float(mean_loss), float(accuracy)
