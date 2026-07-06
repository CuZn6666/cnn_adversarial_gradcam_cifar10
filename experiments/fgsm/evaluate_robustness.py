"""Controlled orchestration for WP8 FGSM robustness evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypedDict

import numpy as np

from configs.default_config import (
    CHECKPOINT_DIR,
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
from src.plotting import plot_fgsm_accuracy_vs_epsilon
from src.robustness import (
    FGSMRepresentativeExamples,
    FGSMSweepResult,
    evaluate_fgsm_epsilon_sweep,
    select_fgsm_representative_examples,
)

DEFAULT_EPSILON_VALUES = tuple(index / 255.0 for index in range(17))


@dataclass(frozen=True)
class WP8FGSMRobustnessConfig:
    """Controlled local configuration for WP8 robustness evaluation."""

    eval_samples: int = 32
    batch_size: int = 8
    seed: int = SEED
    epsilon_values: tuple[float, ...] = DEFAULT_EPSILON_VALUES
    output_dir: Path | str = PROJECT_ROOT / "results" / "WP8"
    representative_epsilon: float = 8.0 / 255.0
    max_successful_examples: int = 1
    max_failed_examples: int = 1
    checkpoint_path: Path | str = (
        CHECKPOINT_DIR / "cifar10_subset_baseline.npz"
    )

    def __post_init__(self) -> None:
        for field_name in ("eval_samples", "batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if not self.epsilon_values:
            raise ValueError("epsilon_values must not be empty.")
        for epsilon in (*self.epsilon_values, self.representative_epsilon):
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0
            ):
                raise ValueError(
                    "FGSM epsilon values must be non-negative finite numbers."
                )
        for field_name in (
            "max_successful_examples",
            "max_failed_examples",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )
        for field_name in ("output_dir", "checkpoint_path"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))


WP8_FGSM_CONFIG = WP8FGSMRobustnessConfig()


class FGSMRobustnessRunResult(TypedDict):
    """Artifacts and in-memory results from one WP8 pipeline run."""

    metrics_path: Path
    plot_path: Path
    sweep_results: list[FGSMSweepResult]
    representative_examples: FGSMRepresentativeExamples


def run_fgsm_robustness_pipeline(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    config: WP8FGSMRobustnessConfig = WP8_FGSM_CONFIG,
) -> FGSMRobustnessRunResult:
    """Run the testable WP8 evaluation pipeline on existing batches."""
    batch_values = tuple(batches)
    sweep_results = evaluate_fgsm_epsilon_sweep(
        model,
        loss_function,
        batch_values,
        config.epsilon_values,
    )
    representative_examples = select_fgsm_representative_examples(
        model,
        loss_function,
        batch_values,
        config.representative_epsilon,
        max_successful=config.max_successful_examples,
        max_failed=config.max_failed_examples,
    )

    metrics = {
        "config": {
            "eval_samples": config.eval_samples,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "epsilon_values": list(config.epsilon_values),
            "representative_epsilon": config.representative_epsilon,
        },
        "sweep_results": sweep_results,
        "representative_examples": representative_examples,
    }
    metrics_path = save_metrics(
        metrics,
        config.output_dir / "fgsm_robustness_metrics.json",
    )
    plot_path = plot_fgsm_accuracy_vs_epsilon(
        sweep_results,
        config.output_dir / "fgsm_accuracy_vs_epsilon.png",
    )
    return {
        "metrics_path": metrics_path,
        "plot_path": plot_path,
        "sweep_results": sweep_results,
        "representative_examples": representative_examples,
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


def run_cifar10_fgsm_robustness(
    config: WP8FGSMRobustnessConfig = WP8_FGSM_CONFIG,
    data_dir: str | Path = DATA_DIR,
) -> FGSMRobustnessRunResult:
    """Run the controlled WP8 pipeline with existing local CIFAR-10 data."""
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
            "running WP8 evaluation."
        )

    _, _, eval_images, eval_labels, _ = load_cifar10(data_path)
    subset_images, subset_labels = _select_deterministic_subset(
        eval_images,
        eval_labels,
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
    return run_fgsm_robustness_pipeline(
        model,
        SoftmaxCrossEntropyLoss(),
        batches,
        config,
    )


def main() -> None:
    """Run the controlled local WP8 evaluation when explicitly invoked."""
    result = run_cifar10_fgsm_robustness()
    print(result)


if __name__ == "__main__":
    main()
