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
    filename_prefix: str = "",
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
    loss_path = output_path / f"{filename_prefix}loss_curve.png"
    accuracy_path = output_path / f"{filename_prefix}accuracy_curve.png"

    _save_curve(epochs, train_loss, eval_loss, "Loss", loss_path)
    _save_curve(
        epochs,
        train_accuracy,
        eval_accuracy,
        "Accuracy",
        accuracy_path,
    )
    return loss_path, accuracy_path


def plot_fgsm_accuracy_vs_epsilon(
    sweep_results: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save clean and adversarial accuracy across FGSM epsilon values."""
    if not sweep_results:
        raise ValueError("FGSM sweep results must not be empty.")

    epsilons = _required_series(
        sweep_results,
        ("epsilon",),
        "epsilon",
    )
    clean_accuracy = _required_series(
        sweep_results,
        ("clean_accuracy",),
        "clean accuracy",
    )
    adversarial_accuracy = _required_series(
        sweep_results,
        ("adversarial_accuracy",),
        "adversarial accuracy",
    )

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.axhline(0.0, color="0.75", linewidth=1.0, zorder=1)
    axes.plot(
        epsilons,
        clean_accuracy,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Clean Accuracy",
        zorder=3,
    )
    axes.plot(
        epsilons,
        adversarial_accuracy,
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="FGSM Accuracy",
        zorder=4,
    )
    axes.set_xlabel("Epsilon")
    axes.set_ylabel("Accuracy")
    axes.set_title("FGSM Accuracy vs Epsilon")
    axes.set_ylim(0.0, 1.0)
    axes.spines["bottom"].set_position(("outward", 6))
    if np.allclose(clean_accuracy, adversarial_accuracy):
        axes.text(
            0.5,
            0.06,
            "Clean and FGSM accuracy overlap in this controlled run.",
            transform=axes.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path
