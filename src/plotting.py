from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _required_series(
    metrics_history: list[dict[str, Any]],
    keys: tuple[str, ...],
    label: str,
) -> list[float]:
    values = []
    for metrics in metrics_history:
        for key in keys:
            if key in metrics:
                value = metrics[key]
                break
        else:
            raise ValueError(f"Metrics history is missing {label}.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} values must be finite numbers.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"{label} values must be finite numbers.")
        values.append(numeric_value)
    return values


def _optional_series(
    metrics_history: list[dict[str, Any]],
    key: str,
) -> list[float] | None:
    present_count = sum(key in metrics for metrics in metrics_history)
    if present_count == 0:
        return None
    if present_count != len(metrics_history):
        raise ValueError(f"Metrics history must include {key} for every epoch.")
    return _required_series(metrics_history, (key,), key)


def _save_curve(
    epochs: list[float],
    train_values: list[float],
    eval_values: list[float] | None,
    ylabel: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(epochs, train_values, marker="o", label="Train")
    if eval_values is not None:
        axes.plot(epochs, eval_values, marker="o", label="Eval")
    axes.set_xlabel("Epoch")
    axes.set_ylabel(ylabel)
    axes.set_title(f"{ylabel} Curve")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_metrics(
    metrics_history: list[dict[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save loss and accuracy curves from an existing metrics history."""
    if not metrics_history:
        raise ValueError("Metrics history must not be empty.")

    epochs = _required_series(metrics_history, ("epoch",), "epoch")
    train_loss = _required_series(
        metrics_history,
        ("train_loss", "mean_loss"),
        "train loss",
    )
    train_accuracy = _required_series(
        metrics_history,
        ("train_accuracy", "accuracy"),
        "train accuracy",
    )
    eval_loss = _optional_series(metrics_history, "eval_loss")
    eval_accuracy = _optional_series(metrics_history, "eval_accuracy")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    loss_path = output_path / "loss_curve.png"
    accuracy_path = output_path / "accuracy_curve.png"

    _save_curve(epochs, train_loss, eval_loss, "Loss", loss_path)
    _save_curve(
        epochs,
        train_accuracy,
        eval_accuracy,
        "Accuracy",
        accuracy_path,
    )
    return loss_path, accuracy_path
