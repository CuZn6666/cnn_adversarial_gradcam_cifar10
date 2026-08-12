"""Scheduler-neutral PGD experiment runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time
from typing import Any

from configs.default_config import DATA_DIR, PROJECT_ROOT, SEED
from experiments.fgsm.run_fgsm_experiment import (
    DEFAULT_OUTPUT_ROOT,
    RUN_ID_PATTERN,
    _backend_batches,
    _select_deterministic_subset,
    _select_split,
    _validate_staged_cifar10_data,
    collect_environment_metadata,
    parse_epsilon_values,
    prepare_run_directory,
    synchronize_backend,
    utc_timestamp,
    write_status,
)
from src.attacks import pgd_linf_attack
from src.backend import ensure_same_backend, resolve_backend, to_python_int
from src.checkpointing import load_checkpoint
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "results" / "baseline" / "portfolio_baseline_best.npz"
)
DEFAULT_EPSILON = 8.0 / 255.0
DEFAULT_ALPHA = 2.0 / 255.0
DEFAULT_STEPS = 10
DEFAULT_RANDOM_START = True
METRIC_FIELDS = (
    "run_id",
    "attack",
    "backend",
    "split",
    "seed",
    "batch_size",
    "requested_max_samples",
    "epsilon",
    "alpha",
    "steps",
    "random_start",
    "total_samples",
    "clean_correct",
    "adversarial_correct",
    "clean_correct_samples",
    "successful_attacks",
    "clean_accuracy",
    "adversarial_accuracy",
    "accuracy_drop",
    "attack_success_rate",
)


@dataclass(frozen=True)
class PGDExperimentConfig:
    """Effective configuration for one L-infinity PGD robustness run."""

    backend: str = "numpy"
    data_dir: Path | str = DATA_DIR
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH
    split: str = "test"
    max_samples: int = 32
    batch_size: int = 8
    epsilon: float = DEFAULT_EPSILON
    alpha: float = DEFAULT_ALPHA
    steps: int = DEFAULT_STEPS
    random_start: bool = DEFAULT_RANDOM_START
    seed: int = SEED
    output_root: Path | str = DEFAULT_OUTPUT_ROOT
    run_id: str | None = None

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if backend not in {"numpy", "cupy"}:
            raise ValueError("backend must be 'numpy' or 'cupy'.")
        object.__setattr__(self, "backend", backend)

        split = self.split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        object.__setattr__(self, "split", split)

        for field_name in ("max_samples", "batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

        if not isinstance(self.random_start, bool):
            raise ValueError("random_start must be a boolean.")

        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, Real)
            or not math.isfinite(float(self.epsilon))
            or self.epsilon < 0.0
        ):
            raise ValueError("epsilon must be a non-negative finite number.")
        object.__setattr__(self, "epsilon", float(self.epsilon))

        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, Integral)
            or self.steps < 0
        ):
            raise ValueError("steps must be a non-negative integer.")
        object.__setattr__(self, "steps", int(self.steps))

        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, Real)
            or not math.isfinite(float(self.alpha))
        ):
            raise ValueError("alpha must be a finite number.")
        if self.alpha < 0.0 or (self.steps > 0 and self.alpha <= 0.0):
            raise ValueError(
                "alpha must be positive when steps > 0 and non-negative otherwise."
            )
        object.__setattr__(self, "alpha", float(self.alpha))

        for field_name in ("data_dir", "checkpoint_path", "output_root"):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))

        run_id = self.run_id
        if run_id is None:
            run_id = generate_run_id(self.backend)
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id may contain only letters, numbers, '.', '_', and '-'."
            )
        object.__setattr__(self, "run_id", run_id)


def generate_run_id(backend: str) -> str:
    """Create a readable unique PGD run identifier."""
    timestamp = utc_timestamp().replace("-", "").replace(":", "")
    timestamp = timestamp.replace(".", "")
    return f"{timestamp}_pgd_linf_{backend}"


def parse_single_epsilon(value: str, *, name: str) -> float:
    """Parse one float/fraction value using the project epsilon conventions."""
    values = parse_epsilon_values(value)
    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    return float(values[0])


def config_to_json(config: PGDExperimentConfig) -> dict[str, Any]:
    """Serialize the effective PGD config to a stable JSON mapping."""
    return {
        "schema_version": 1,
        "experiment_type": "pgd_linf",
        "attack": "pgd_linf",
        "backend": config.backend,
        "data_dir": str(config.data_dir),
        "checkpoint_path": str(config.checkpoint_path),
        "split": config.split,
        "max_samples": config.max_samples,
        "batch_size": config.batch_size,
        "epsilon": config.epsilon,
        "alpha": config.alpha,
        "steps": config.steps,
        "random_start": config.random_start,
        "seed": config.seed,
        "random_start_seed_strategy": (
            "seed + batch_index for each evaluated batch"
            if config.random_start
            else "not used when random_start is false"
        ),
        "output_root": str(config.output_root),
        "run_id": config.run_id,
    }


def _pgd_seed_for_batch(config: PGDExperimentConfig, batch_index: int) -> int | None:
    if not config.random_start:
        return None
    return config.seed + batch_index


def evaluate_pgd_batches(
    model: CompactCNN,
    loss_function: SoftmaxCrossEntropyLoss,
    batches: Any,
    config: PGDExperimentConfig,
) -> dict[str, Any]:
    """Aggregate clean and PGD metrics over one deterministic batch stream."""
    total_samples = 0
    clean_correct = 0
    adversarial_correct = 0
    clean_correct_samples = 0
    successful_attacks = 0

    for batch_index, (images, labels) in enumerate(batches):
        clean_logits = model.forward(images)
        xp = ensure_same_backend(clean_logits, labels)
        clean_predictions = xp.argmax(clean_logits, axis=1)

        adversarial_images = pgd_linf_attack(
            model,
            loss_function,
            images,
            labels,
            epsilon=config.epsilon,
            alpha=config.alpha,
            steps=config.steps,
            random_start=config.random_start,
            seed=_pgd_seed_for_batch(config, batch_index),
        )
        adversarial_logits = model.forward(adversarial_images)
        adversarial_predictions = xp.argmax(adversarial_logits, axis=1)

        batch_samples = int(labels.shape[0])
        clean_correct_mask = clean_predictions == labels
        adversarial_correct_mask = adversarial_predictions == labels
        batch_clean_correct = to_python_int(xp.sum(clean_correct_mask))
        batch_adversarial_correct = to_python_int(
            xp.sum(adversarial_correct_mask)
        )
        batch_successful_attacks = to_python_int(
            xp.sum(clean_correct_mask & ~adversarial_correct_mask)
        )

        total_samples += batch_samples
        clean_correct += batch_clean_correct
        adversarial_correct += batch_adversarial_correct
        clean_correct_samples += batch_clean_correct
        successful_attacks += batch_successful_attacks

    if total_samples == 0:
        raise ValueError("evaluate_pgd_batches requires at least one sample.")

    clean_accuracy = clean_correct / total_samples
    adversarial_accuracy = adversarial_correct / total_samples
    attack_success_rate = (
        successful_attacks / clean_correct_samples
        if clean_correct_samples > 0
        else 0.0
    )
    return {
        "epsilon": config.epsilon,
        "alpha": config.alpha,
        "steps": config.steps,
        "random_start": config.random_start,
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


def _metric_rows(
    config: PGDExperimentConfig,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": config.run_id,
            "attack": "pgd_linf",
            "backend": config.backend,
            "split": config.split,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "requested_max_samples": config.max_samples,
            "epsilon": result["epsilon"],
            "alpha": result["alpha"],
            "steps": result["steps"],
            "random_start": result["random_start"],
            "total_samples": result["total_samples"],
            "clean_correct": result["clean_correct"],
            "adversarial_correct": result["adversarial_correct"],
            "clean_correct_samples": result["clean_correct_samples"],
            "successful_attacks": result["successful_attacks"],
            "clean_accuracy": result["clean_accuracy"],
            "adversarial_accuracy": result["adversarial_accuracy"],
            "accuracy_drop": result["accuracy_drop"],
            "attack_success_rate": result["attack_success_rate"],
        }
    ]


def write_metrics_artifacts(
    run_dir: Path,
    config: PGDExperimentConfig,
    result: dict[str, Any],
) -> dict[str, str]:
    """Write PGD metrics as JSON and CSV artifacts."""
    rows = _metric_rows(config, result)
    metrics_json_path = save_metrics(
        {
            "schema_version": 1,
            "attack": "pgd_linf",
            "metric_fields": list(METRIC_FIELDS),
            "metric_semantics": {
                "attack_success_rate": (
                    "successful_attacks / clean_correct_samples, or 0.0 "
                    "when there are no clean-correct samples"
                ),
                "accuracy_drop": "clean_accuracy - adversarial_accuracy",
                "sample_steps": (
                    "sample_count * steps; each step computes one input gradient"
                ),
            },
            "results": rows,
        },
        run_dir / "metrics.json",
    )

    metrics_csv_path = run_dir / "metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "metrics_json": str(metrics_json_path),
        "metrics_csv": str(metrics_csv_path),
    }


def _summary_payload(
    config: PGDExperimentConfig,
    result: dict[str, Any],
    timing: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "status": "COMPLETED",
        "experiment_type": "pgd_linf",
        "attack": "pgd_linf",
        "backend": config.backend,
        "split": config.split,
        "sample_count": result["total_samples"],
        "batch_size": config.batch_size,
        "epsilon": config.epsilon,
        "alpha": config.alpha,
        "steps": config.steps,
        "random_start": config.random_start,
        "clean_accuracy": result["clean_accuracy"],
        "adversarial_accuracy": result["adversarial_accuracy"],
        "accuracy_drop": result["accuracy_drop"],
        "attack_success_rate": result["attack_success_rate"],
        "timing": timing,
        "artifacts": artifacts,
    }


def run_pgd_experiment(config: PGDExperimentConfig) -> dict[str, Any]:
    """Run one configured PGD robustness experiment and write artifacts."""
    run_dir = prepare_run_directory(config)
    started_at = utc_timestamp()
    write_status(run_dir, "RUNNING", started_at=started_at)
    total_start = time.perf_counter()

    try:
        config_path = save_metrics(config_to_json(config), run_dir / "config.json")
        environment = collect_environment_metadata(config.backend)
        environment_path = save_metrics(environment, run_dir / "environment.json")

        if not Path(config.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Model checkpoint is not available at {config.checkpoint_path}."
            )

        dataset_metadata = _validate_staged_cifar10_data(Path(config.data_dir))
        images, labels, class_names = _select_split(
            Path(config.data_dir),
            config.split,
        )
        selected_images, selected_labels = _select_deterministic_subset(
            images,
            labels,
            config.max_samples,
            config.seed,
        )

        xp = resolve_backend(config.backend)
        model = CompactCNN(seed=config.seed, backend=xp)
        load_checkpoint(model, config.checkpoint_path)
        loss_function = SoftmaxCrossEntropyLoss(backend=xp)
        batches = _backend_batches(
            selected_images,
            selected_labels,
            batch_size=config.batch_size,
            backend=xp,
        )

        synchronize_backend(xp)
        evaluation_start = time.perf_counter()
        pgd_result = evaluate_pgd_batches(
            model,
            loss_function,
            batches,
            config,
        )
        synchronize_backend(xp)
        evaluation_seconds = time.perf_counter() - evaluation_start

        total_seconds = time.perf_counter() - total_start
        sample_count = pgd_result["total_samples"]
        sample_steps = sample_count * config.steps
        timing = {
            "schema_version": 1,
            "timing_method": "time.perf_counter",
            "gpu_synchronization": (
                "cupy Stream.null synchronized before and after evaluation"
                if config.backend == "cupy"
                else "not required for numpy"
            ),
            "total_wall_seconds": total_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "sample_count": sample_count,
            "pgd_steps": config.steps,
            "gradient_evaluations": sample_steps,
            "sample_steps": sample_steps,
            "samples_per_second": (
                sample_count / evaluation_seconds
                if evaluation_seconds > 0.0
                else None
            ),
            "sample_steps_per_second": (
                sample_steps / evaluation_seconds
                if evaluation_seconds > 0.0
                else None
            ),
        }
        timing_path = save_metrics(timing, run_dir / "timing.json")
        metric_artifacts = write_metrics_artifacts(run_dir, config, pgd_result)

        artifacts = {
            "config": str(config_path),
            "environment": str(environment_path),
            "timing": str(timing_path),
            **metric_artifacts,
        }
        summary = _summary_payload(config, pgd_result, timing, artifacts)
        summary["dataset"] = {
            **dataset_metadata,
            "split": config.split,
            "available_split_samples": int(images.shape[0]),
            "evaluated_samples": sample_count,
            "class_count": len(class_names),
        }
        summary_path = save_metrics(summary, run_dir / "summary.json")
        artifacts["summary"] = str(summary_path)
        status_path = write_status(
            run_dir,
            "COMPLETED",
            started_at=started_at,
            ended_at=utc_timestamp(),
        )
        artifacts["status"] = str(status_path)

        return {
            "run_dir": run_dir,
            "config": config,
            "pgd_result": pgd_result,
            "timing": timing,
            "artifacts": artifacts,
        }
    except Exception as error:
        write_status(
            run_dir,
            "FAILED",
            started_at=started_at,
            ended_at=utc_timestamp(),
            error={
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a scheduler-neutral L-infinity PGD robustness experiment and "
            "write machine-readable artifacts."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="Array backend to use. CuPy never falls back to NumPy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the staged CIFAR-10 archive and extraction.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="CompactCNN .npz checkpoint to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="CIFAR-10 split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Number of deterministic split samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--epsilon",
        default="8/255",
        help="PGD epsilon as a float or fraction.",
    )
    parser.add_argument(
        "--alpha",
        default="2/255",
        help="PGD step size as a float or fraction.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of PGD gradient steps.",
    )
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RANDOM_START,
        help="Enable or disable random initialization inside the epsilon ball.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for deterministic sample selection and PGD random starts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory under which the isolated run directory is created.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run identifier. Must be unique under output-root.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PGDExperimentConfig:
    return PGDExperimentConfig(
        backend=args.backend,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epsilon=parse_single_epsilon(args.epsilon, name="epsilon"),
        alpha=parse_single_epsilon(args.alpha, name="alpha"),
        steps=args.steps,
        random_start=args.random_start,
        seed=args.seed,
        output_root=args.output_root,
        run_id=args.run_id,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = config_from_args(parse_args(argv))
        result = run_pgd_experiment(config)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"run_id: {config.run_id}")
    print(f"run_dir: {result['run_dir']}")
    print(f"summary: {result['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
