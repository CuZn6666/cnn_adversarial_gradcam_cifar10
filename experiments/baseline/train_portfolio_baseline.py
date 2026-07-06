"""Portfolio baseline training for Day 1 Bosch application preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configs.default_config import DATA_DIR, PROJECT_ROOT, SEED
from src.checkpointing import load_checkpoint, save_checkpoint
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_cifar10
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN
from src.optimizers import SGD
from src.plotting import (
    plot_confusion_matrix,
    plot_train_validation_accuracy_curve,
    plot_training_loss_curve,
)
from src.training import evaluate_batches, train_batches


@dataclass(frozen=True)
class PortfolioBaselineConfig:
    """Deterministic local portfolio-baseline configuration."""

    train_samples: int = 4096
    validation_samples: int = 1024
    test_samples: int = 1024
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 0.03
    seed: int = SEED
    output_dir: Path | str = PROJECT_ROOT / "results" / "baseline"

    def __post_init__(self) -> None:
        for field_name in (
            "train_samples",
            "validation_samples",
            "test_samples",
            "batch_size",
            "epochs",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a positive finite number.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if not isinstance(self.output_dir, (str, Path)) or (
            isinstance(self.output_dir, str) and not self.output_dir.strip()
        ):
            raise ValueError("output_dir must be a valid path.")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


PORTFOLIO_BASELINE_CONFIG = PortfolioBaselineConfig()


def _deterministic_train_validation_split(
    images: np.ndarray,
    labels: np.ndarray,
    train_samples: int,
    validation_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_samples = train_samples + validation_samples
    if required_samples > images.shape[0]:
        raise ValueError(
            "Requested train and validation samples exceed available data."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(images.shape[0])[:required_samples]
    train_indices = indices[:train_samples]
    validation_indices = indices[train_samples:]
    return (
        images[train_indices],
        labels[train_indices],
        images[validation_indices],
        labels[validation_indices],
    )


def _deterministic_subset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count > images.shape[0]:
        raise ValueError("Requested sample_count exceeds available data.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=sample_count, replace=False)
    return images[indices], labels[indices]


def _evaluate_with_predictions(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    total_weighted_loss = 0.0
    predictions: list[np.ndarray] = []
    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model.forward(batch_images)
        loss = loss_function.forward(logits, batch_labels)
        total_weighted_loss += loss * batch_labels.shape[0]
        predictions.append(np.argmax(logits, axis=1))

    all_predictions = np.concatenate(predictions, axis=0)
    accuracy = float(np.mean(all_predictions == labels))
    return {
        "loss": float(total_weighted_loss / labels.shape[0]),
        "accuracy": accuracy,
        "predictions": all_predictions,
    }


def _confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def run_portfolio_baseline(
    config: PortfolioBaselineConfig = PORTFOLIO_BASELINE_CONFIG,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Train a deterministic clean CIFAR-10 portfolio baseline."""
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10(
        Path(data_dir)
    )
    (
        train_subset_images,
        train_subset_labels,
        validation_images,
        validation_labels,
    ) = _deterministic_train_validation_split(
        train_images,
        train_labels,
        config.train_samples,
        config.validation_samples,
        config.seed,
    )
    test_subset_images, test_subset_labels = _deterministic_subset(
        test_images,
        test_labels,
        config.test_samples,
        config.seed + 2000,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "portfolio_baseline_best.npz"
    history_path = output_dir / "portfolio_training_history.json"
    final_metrics_path = output_dir / "portfolio_final_metrics.json"
    loss_curve_path = output_dir / "training_loss_curve.png"
    accuracy_curve_path = output_dir / "train_validation_accuracy_curve.png"
    confusion_matrix_path = output_dir / "confusion_matrix.png"

    model = CompactCNN(seed=config.seed)
    loss_function = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=config.learning_rate)
    metrics_history: list[dict[str, int | float]] = []
    best_validation_accuracy = -1.0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        training_batches = iterate_minibatches(
            train_subset_images,
            train_subset_labels,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        )
        train_metrics = train_batches(
            model,
            loss_function,
            optimizer,
            training_batches,
        )
        validation_loss, validation_accuracy = evaluate_batches(
            model,
            loss_function,
            iterate_minibatches(
                validation_images,
                validation_labels,
                batch_size=config.batch_size,
                shuffle=False,
            ),
        )

        epoch_metrics = {
            "epoch": epoch,
            "training_loss": train_metrics["mean_loss"],
            "training_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "train_samples": config.train_samples,
            "validation_samples": config.validation_samples,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
        }
        metrics_history.append(epoch_metrics)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            save_checkpoint(model, checkpoint_path)

    load_checkpoint(model, checkpoint_path)
    final_loss_function = SoftmaxCrossEntropyLoss()
    validation_result = _evaluate_with_predictions(
        model,
        final_loss_function,
        validation_images,
        validation_labels,
        config.batch_size,
    )
    test_result = _evaluate_with_predictions(
        model,
        final_loss_function,
        test_subset_images,
        test_subset_labels,
        config.batch_size,
    )
    matrix = _confusion_matrix(
        test_subset_labels,
        test_result["predictions"],
        class_count=len(class_names),
    )

    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_validation_loss": validation_result["loss"],
        "final_validation_accuracy": validation_result["accuracy"],
        "final_test_loss": test_result["loss"],
        "final_test_accuracy": test_result["accuracy"],
        "train_samples": config.train_samples,
        "validation_samples": config.validation_samples,
        "test_samples": config.test_samples,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "checkpoint_path": str(checkpoint_path),
    }

    save_metrics(metrics_history, history_path)
    save_metrics(final_metrics, final_metrics_path)
    plot_training_loss_curve(metrics_history, loss_curve_path)
    plot_train_validation_accuracy_curve(metrics_history, accuracy_curve_path)
    plot_confusion_matrix(matrix, class_names, confusion_matrix_path)

    return {
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "final_metrics_path": final_metrics_path,
        "training_loss_curve_path": loss_curve_path,
        "train_validation_accuracy_curve_path": accuracy_curve_path,
        "confusion_matrix_path": confusion_matrix_path,
        "metrics_history": metrics_history,
        "final_metrics": final_metrics,
        "confusion_matrix": matrix,
    }


def main() -> None:
    result = run_portfolio_baseline()
    print(result["final_metrics"])


if __name__ == "__main__":
    main()
