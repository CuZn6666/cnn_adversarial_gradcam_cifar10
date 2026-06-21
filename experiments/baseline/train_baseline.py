"""Minimal deterministic baseline orchestration using synthetic data only."""

from pathlib import Path
from typing import Any

import numpy as np

from configs.default_config import BASELINE_CONFIG, BaselineConfig
from src.checkpointing import save_checkpoint
from src.data.batching import iterate_minibatches
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


def run_synthetic_baseline(
    config: BaselineConfig = BASELINE_CONFIG,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a small deterministic baseline experiment on synthetic data."""
    if config.train_subset_size is None or config.eval_subset_size is None:
        raise ValueError(
            "Synthetic baseline requires train_subset_size and eval_subset_size."
        )

    if output_root is None:
        checkpoint_dir = config.checkpoint_dir
        log_dir = config.log_dir
        figure_dir = config.figure_dir
    else:
        root = Path(output_root)
        checkpoint_dir = root / "checkpoints"
        log_dir = root / "logs"
        figure_dir = root / "figures"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels = _create_synthetic_dataset(
        config.train_subset_size,
        config.seed,
    )
    eval_images, eval_labels = _create_synthetic_dataset(
        config.eval_subset_size,
        config.seed + 1,
    )

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
                "eval_total_samples": config.eval_subset_size,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "seed": config.seed,
            }
        )

    checkpoint_path = save_checkpoint(
        model,
        checkpoint_dir / "synthetic_baseline.npz",
    )
    metrics_path = save_metrics(
        metrics_history,
        log_dir / "synthetic_metrics.json",
    )
    loss_curve_path, accuracy_curve_path = plot_metrics(
        metrics_history,
        figure_dir,
    )

    return {
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "loss_curve_path": loss_curve_path,
        "accuracy_curve_path": accuracy_curve_path,
        "metrics_history": metrics_history,
        "final_metrics": metrics_history[-1],
    }
