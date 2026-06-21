"""Minimal deterministic baseline experiment orchestration."""

from pathlib import Path
from typing import Any

import numpy as np

from configs.default_config import (
    BASELINE_CONFIG,
    CIFAR10_EXTRACTED_DIR,
    DATA_DIR,
    BaselineConfig,
)
from src.checkpointing import save_checkpoint
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_cifar10
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models.compact_cnn import CompactCNN
from src.optimizers.sgd import SGD
from src.plotting import plot_metrics
from src.training import evaluate_batches, train_batches


def _create_synthetic_dataset(
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(sample_count, 3, 32, 32),
    ).astype(np.float32)
    labels = np.arange(sample_count, dtype=np.int64) % 10
    return images, labels


def _resolve_output_directories(
    config: BaselineConfig,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path]:
    if output_root is None:
        return config.checkpoint_dir, config.log_dir, config.figure_dir

    root = Path(output_root)
    return root / "checkpoints", root / "logs", root / "figures"


def _select_deterministic_subset(
    images: np.ndarray,
    labels: np.ndarray,
    subset_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if subset_size > images.shape[0]:
        raise ValueError(
            f"subset_size {subset_size} exceeds dataset size {images.shape[0]}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=subset_size, replace=False)
    return images[indices], labels[indices]


def _run_baseline(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    eval_images: np.ndarray,
    eval_labels: np.ndarray,
    config: BaselineConfig,
    output_root: str | Path | None,
    checkpoint_filename: str,
    metrics_filename: str,
    figure_prefix: str,
) -> dict[str, Any]:
    checkpoint_dir, log_dir, figure_dir = _resolve_output_directories(
        config,
        output_root,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = CompactCNN(seed=config.seed)
    loss_function = SoftmaxCrossEntropyLoss()
    optimizer = SGD(learning_rate=config.learning_rate)
    metrics_history: list[dict[str, int | float]] = []

    for epoch in range(1, config.epochs + 1):
        training_batches = iterate_minibatches(
            train_images,
            train_labels,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch - 1,
        )
        train_metrics = train_batches(
            model,
            loss_function,
            optimizer,
            training_batches,
        )

        evaluation_batches = iterate_minibatches(
            eval_images,
            eval_labels,
            batch_size=config.batch_size,
            shuffle=False,
        )
        eval_loss, eval_accuracy = evaluate_batches(
            model,
            loss_function,
            evaluation_batches,
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["mean_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_total_samples": train_metrics["total_samples"],
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "eval_total_samples": eval_labels.shape[0],
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "seed": config.seed,
            }
        )

    checkpoint_path = save_checkpoint(
        model,
        checkpoint_dir / checkpoint_filename,
    )
    metrics_path = save_metrics(
        metrics_history,
        log_dir / metrics_filename,
    )
    loss_curve_path, accuracy_curve_path = plot_metrics(
        metrics_history,
        figure_dir,
        filename_prefix=figure_prefix,
    )

    return {
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "loss_curve_path": loss_curve_path,
        "accuracy_curve_path": accuracy_curve_path,
        "metrics_history": metrics_history,
        "final_metrics": metrics_history[-1],
    }


def run_synthetic_baseline(
    config: BaselineConfig = BASELINE_CONFIG,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a small deterministic baseline experiment on synthetic data."""
    if config.train_subset_size is None or config.eval_subset_size is None:
        raise ValueError(
            "Synthetic baseline requires train_subset_size and eval_subset_size."
        )

    train_images, train_labels = _create_synthetic_dataset(
        config.train_subset_size,
        config.seed,
    )
    eval_images, eval_labels = _create_synthetic_dataset(
        config.eval_subset_size,
        config.seed + 1,
    )

    return _run_baseline(
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        config,
        output_root,
        checkpoint_filename="synthetic_baseline.npz",
        metrics_filename="synthetic_metrics.json",
        figure_prefix="",
    )


def run_cifar10_subset_baseline(
    config: BaselineConfig = BASELINE_CONFIG,
    output_root: str | Path | None = None,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Run a deterministic baseline on an existing CIFAR-10 subset."""
    if config.train_subset_size is None or config.eval_subset_size is None:
        raise ValueError(
            "CIFAR-10 subset baseline requires train_subset_size and "
            "eval_subset_size."
        )

    data_path = Path(data_dir)
    dataset_path = data_path / CIFAR10_EXTRACTED_DIR
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            "CIFAR-10 data is not available at "
            f"{dataset_path}. Download and extract it explicitly before "
            "running the subset baseline."
        )

    train_images, train_labels, eval_images, eval_labels, _ = load_cifar10(
        data_path
    )
    train_subset = _select_deterministic_subset(
        train_images,
        train_labels,
        config.train_subset_size,
        config.seed,
    )
    eval_subset = _select_deterministic_subset(
        eval_images,
        eval_labels,
        config.eval_subset_size,
        config.seed + 1,
    )

    return _run_baseline(
        *train_subset,
        *eval_subset,
        config,
        output_root,
        checkpoint_filename="cifar10_subset_baseline.npz",
        metrics_filename="cifar10_subset_metrics.json",
        figure_prefix="cifar10_subset_",
    )
