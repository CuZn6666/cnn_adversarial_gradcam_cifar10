"""FGSM quantitative robustness evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypedDict

import numpy as np

from configs.default_config import (
    CIFAR10_EXTRACTED_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.checkpointing import load_checkpoint
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import load_cifar10
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN
from src.plotting import (
    plot_fgsm_accuracy_drop_vs_epsilon,
    plot_fgsm_attack_success_rate_vs_epsilon,
    plot_fgsm_portfolio_accuracy_vs_epsilon,
)
from src.robustness import FGSMSweepResult, evaluate_fgsm_epsilon_sweep


FGSM_QUANTITATIVE_EPSILON_VALUES = (
    0.0,
    2.0 / 255.0,
    4.0 / 255.0,
    8.0 / 255.0,
    16.0 / 255.0,
)
FGSM_QUANTITATIVE_EPSILON_LABELS = (
    "0",
    "2/255",
    "4/255",
    "8/255",
    "16/255",
)


@dataclass(frozen=True)
class PortfolioFGSMQuantitativeConfig:
    """Deterministic FGSM quantitative evaluation configuration."""

    eval_samples: int = 1024
    batch_size: int = 32
    seed: int = SEED
    epsilon_values: tuple[float, ...] = FGSM_QUANTITATIVE_EPSILON_VALUES
    epsilon_labels: tuple[str, ...] = FGSM_QUANTITATIVE_EPSILON_LABELS
    checkpoint_path: Path | str = (
        PROJECT_ROOT / "results" / "baseline" / "portfolio_baseline_best.npz"
    )
    output_dir: Path | str = PROJECT_ROOT / "results" / "fgsm"

    def __post_init__(self) -> None:
        for field_name in ("eval_samples", "batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if not self.epsilon_values:
            raise ValueError("epsilon_values must not be empty.")
        if len(self.epsilon_values) != len(self.epsilon_labels):
            raise ValueError(
                "epsilon_values and epsilon_labels must have the same length."
            )
        for epsilon in self.epsilon_values:
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0.0
            ):
                raise ValueError(
                    "epsilon_values must be non-negative finite numbers."
                )
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


PORTFOLIO_FGSM_QUANTITATIVE_CONFIG = PortfolioFGSMQuantitativeConfig()


class PortfolioFGSMQuantitativeResult(TypedDict):
    """Artifacts and metrics from one FGSM quantitative run."""

    metrics_path: Path
    accuracy_plot_path: Path
    attack_success_rate_plot_path: Path
    accuracy_drop_plot_path: Path
    sweep_results: list[FGSMSweepResult]


def _validate_epsilon_zero_invariants(
    sweep_results: list[FGSMSweepResult],
) -> None:
    for result in sweep_results:
        if result["epsilon"] != 0.0:
            continue
        if result["clean_correct"] != result["adversarial_correct"]:
            raise RuntimeError("epsilon=0 must preserve adversarial_correct.")
        if result["clean_accuracy"] != result["adversarial_accuracy"]:
            raise RuntimeError("epsilon=0 must preserve adversarial_accuracy.")
        if result["accuracy_drop"] != 0.0:
            raise RuntimeError("epsilon=0 must have zero accuracy_drop.")
        if result["attack_success_rate"] != 0.0:
            raise RuntimeError("epsilon=0 must have zero attack_success_rate.")


def run_portfolio_fgsm_quantitative_pipeline(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    config: PortfolioFGSMQuantitativeConfig = (
        PORTFOLIO_FGSM_QUANTITATIVE_CONFIG
    ),
) -> PortfolioFGSMQuantitativeResult:
    """Run FGSM epsilon sweep, persist metrics, and create figures."""
    batch_values = tuple(batches)
    sweep_results = evaluate_fgsm_epsilon_sweep(
        model,
        loss_function,
        batch_values,
        config.epsilon_values,
    )
    _validate_epsilon_zero_invariants(sweep_results)

    output_dir = Path(config.output_dir)
    try:
        checkpoint_for_metrics = str(
            config.checkpoint_path.resolve().relative_to(PROJECT_ROOT)
        )
    except ValueError:
        checkpoint_for_metrics = str(config.checkpoint_path)

    metrics = {
        "config": {
            "checkpoint_path": checkpoint_for_metrics,
            "eval_samples": config.eval_samples,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "epsilon_values": list(config.epsilon_values),
            "epsilon_labels": list(config.epsilon_labels),
        },
        "sweep_results": sweep_results,
    }
    metrics_path = save_metrics(
        metrics,
        output_dir / "fgsm_quantitative_metrics.json",
    )
    accuracy_plot_path = plot_fgsm_portfolio_accuracy_vs_epsilon(
        sweep_results,
        output_dir / "accuracy_vs_epsilon.png",
        epsilon_labels=config.epsilon_labels,
    )
    attack_success_rate_plot_path = (
        plot_fgsm_attack_success_rate_vs_epsilon(
            sweep_results,
            output_dir / "attack_success_rate_vs_epsilon.png",
            epsilon_labels=config.epsilon_labels,
        )
    )
    accuracy_drop_plot_path = plot_fgsm_accuracy_drop_vs_epsilon(
        sweep_results,
        output_dir / "accuracy_drop_vs_epsilon.png",
        epsilon_labels=config.epsilon_labels,
    )

    return {
        "metrics_path": metrics_path,
        "accuracy_plot_path": accuracy_plot_path,
        "attack_success_rate_plot_path": attack_success_rate_plot_path,
        "accuracy_drop_plot_path": accuracy_drop_plot_path,
        "sweep_results": sweep_results,
    }


def _select_deterministic_subset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count > images.shape[0]:
        raise ValueError(
            f"eval_samples {sample_count} exceeds dataset size "
            f"{images.shape[0]}."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=sample_count, replace=False)
    return images[indices], labels[indices]


def run_portfolio_cifar10_fgsm_quantitative(
    config: PortfolioFGSMQuantitativeConfig = (
        PORTFOLIO_FGSM_QUANTITATIVE_CONFIG
    ),
    data_dir: str | Path = DATA_DIR,
) -> PortfolioFGSMQuantitativeResult:
    """Run FGSM quantitative evaluation on local CIFAR-10 test data."""
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
            "running FGSM quantitative evaluation."
        )

    _, _, test_images, test_labels, _ = load_cifar10(data_path)
    subset_images, subset_labels = _select_deterministic_subset(
        test_images,
        test_labels,
        config.eval_samples,
        config.seed,
    )
    batches = iterate_minibatches(
        subset_images,
        subset_labels,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = CompactCNN(seed=config.seed)
    load_checkpoint(model, config.checkpoint_path)
    return run_portfolio_fgsm_quantitative_pipeline(
        model,
        SoftmaxCrossEntropyLoss(),
        batches,
        config,
    )


def main() -> None:
    result = run_portfolio_cifar10_fgsm_quantitative()
    print(result)


if __name__ == "__main__":
    main()
