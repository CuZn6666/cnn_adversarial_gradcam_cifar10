"""Scheduler-neutral FGSM experiment runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import math
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

from configs.default_config import (
    CIFAR10_ARCHIVE_NAME,
    CIFAR10_EXTRACTED_DIR,
    CIFAR10_MD5,
    DATA_DIR,
    PROJECT_ROOT,
    SEED,
)
from src.backend import backend_name, resolve_backend, to_backend
from src.checkpointing import load_checkpoint
from src.data.batching import iterate_minibatches
from src.data.cifar10_loader import compute_md5, load_cifar10
from src.losses import SoftmaxCrossEntropyLoss
from src.metrics import save_metrics
from src.models import CompactCNN
from src.robustness import FGSMSweepResult, evaluate_fgsm_epsilon_sweep


DEFAULT_EPSILON_VALUES = (
    0.0,
    2.0 / 255.0,
    4.0 / 255.0,
    8.0 / 255.0,
    16.0 / 255.0,
)
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "results" / "baseline" / "portfolio_baseline_best.npz"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "runs"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
METRIC_FIELDS = (
    "run_id",
    "backend",
    "split",
    "seed",
    "batch_size",
    "requested_max_samples",
    "epsilon",
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
class FGSMExperimentConfig:
    """Effective configuration for one FGSM robustness experiment."""

    backend: str = "numpy"
    data_dir: Path | str = DATA_DIR
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH
    split: str = "test"
    max_samples: int = 32
    batch_size: int = 8
    epsilon_values: tuple[float, ...] = DEFAULT_EPSILON_VALUES
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

        if not self.epsilon_values:
            raise ValueError("epsilon_values must not be empty.")
        normalized_epsilons: list[float] = []
        for epsilon in self.epsilon_values:
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0.0
            ):
                raise ValueError("epsilon_values must be non-negative finite numbers.")
            normalized_epsilons.append(float(epsilon))
        object.__setattr__(self, "epsilon_values", tuple(normalized_epsilons))

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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for artifacts."""
    return _utc_now().isoformat().replace("+00:00", "Z")


def generate_run_id(backend: str) -> str:
    """Create a readable unique run identifier."""
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_fgsm_{backend}"


def _parse_epsilon_token(token: str) -> float:
    token = token.strip()
    if not token:
        raise ValueError("epsilon values must not contain empty entries.")

    if "/" in token:
        parts = token.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid epsilon fraction: {token}")
        numerator = float(parts[0])
        denominator = float(parts[1])
        if denominator == 0.0:
            raise ValueError("epsilon fraction denominator must be non-zero.")
        value = numerator / denominator
    else:
        value = float(token)

    if not math.isfinite(value) or value < 0.0:
        raise ValueError("epsilon values must be non-negative finite numbers.")
    return float(value)


def parse_epsilon_values(value: str) -> tuple[float, ...]:
    """Parse comma-separated float or fraction epsilon values."""
    epsilons = tuple(_parse_epsilon_token(token) for token in value.split(","))
    if not epsilons:
        raise ValueError("at least one epsilon value is required.")
    return epsilons


def config_to_json(config: FGSMExperimentConfig) -> dict[str, Any]:
    """Serialize the effective config to a stable JSON-compatible mapping."""
    return {
        "schema_version": 1,
        "experiment_type": "fgsm_epsilon_sweep",
        "backend": config.backend,
        "data_dir": str(config.data_dir),
        "checkpoint_path": str(config.checkpoint_path),
        "split": config.split,
        "max_samples": config.max_samples,
        "batch_size": config.batch_size,
        "epsilon_values": list(config.epsilon_values),
        "seed": config.seed,
        "output_root": str(config.output_root),
        "run_id": config.run_id,
    }


def _git_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _decode_device_name(raw_name: Any) -> str:
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8", errors="replace")
    return str(raw_name)


def collect_environment_metadata(backend: str) -> dict[str, Any]:
    """Collect reproducibility metadata without making NumPy depend on CuPy."""
    errors: list[str] = []
    cupy_info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_runtime_version": None,
        "device_count": None,
        "gpu_name": None,
        "error": None,
    }

    if importlib.util.find_spec("cupy") is None:
        cupy_info["error"] = "cupy is not installed"
    else:  # pragma: no cover - exercised on GPU validation nodes.
        cupy_info["installed"] = True
        try:
            import cupy as cp

            cupy_info["version"] = str(cp.__version__)
            cupy_info["cuda_runtime_version"] = int(
                cp.cuda.runtime.runtimeGetVersion()
            )
            device_count = int(cp.cuda.runtime.getDeviceCount())
            cupy_info["device_count"] = device_count
            if device_count > 0:
                properties = cp.cuda.runtime.getDeviceProperties(0)
                cupy_info["gpu_name"] = _decode_device_name(
                    properties.get("name", "unknown")
                )
        except Exception as error:
            cupy_info["error"] = str(error)

    if backend == "cupy":
        if not cupy_info["installed"]:
            errors.append("CuPy backend requested, but cupy is not installed.")
        elif cupy_info["error"] is not None:
            errors.append(
                "CuPy backend requested, but CUDA/CuPy validation failed: "
                f"{cupy_info['error']}"
            )
        elif not cupy_info["device_count"]:
            errors.append("CuPy backend requested, but no CUDA GPU is visible.")

    git_status = _git_output(["status", "--short"])
    git_metadata = {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(git_status),
        "status_short": git_status,
    }

    if errors:
        raise RuntimeError(" ".join(errors))

    return {
        "schema_version": 1,
        "captured_at": utc_timestamp(),
        "backend_requested": backend,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy_version": np.__version__,
        "cupy": cupy_info,
        "git": git_metadata,
    }


def synchronize_backend(xp: Any) -> None:
    """Synchronize asynchronous GPU work before or after timed regions."""
    if backend_name(xp) == "cupy":  # pragma: no cover - GPU-only branch.
        xp.cuda.Stream.null.synchronize()


def prepare_run_directory(config: FGSMExperimentConfig) -> Path:
    """Create an isolated run directory and reject collisions."""
    run_dir = Path(config.output_root) / str(config.run_id)
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists and will not be overwritten: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    return run_dir


def write_status(
    run_dir: Path,
    status: str,
    *,
    started_at: str,
    ended_at: str | None = None,
    error: dict[str, str] | None = None,
) -> Path:
    """Persist current run status for partial-run inspection."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    if error is not None:
        payload["error"] = error
    return save_metrics(payload, run_dir / "status.json")


def _validate_staged_cifar10_data(data_dir: Path) -> dict[str, Any]:
    archive_path = data_dir / CIFAR10_ARCHIVE_NAME
    extracted_dir = data_dir / CIFAR10_EXTRACTED_DIR

    if not archive_path.is_file():
        raise FileNotFoundError(f"CIFAR-10 archive is missing: {archive_path}")
    observed_md5 = compute_md5(archive_path)
    if observed_md5 != CIFAR10_MD5:
        raise ValueError(
            "CIFAR-10 archive checksum mismatch. "
            f"Expected {CIFAR10_MD5}, observed {observed_md5}."
        )
    if not extracted_dir.is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 extracted directory is missing: {extracted_dir}"
        )

    return {
        "data_dir": str(data_dir),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_expected_md5": CIFAR10_MD5,
        "archive_observed_md5": observed_md5,
        "archive_checksum_matches": True,
        "extracted_dir": str(extracted_dir),
    }


def _select_split(
    data_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_train, y_train, x_test, y_test, class_names = load_cifar10(data_dir)
    if split == "train":
        return x_train, y_train, class_names
    return x_test, y_test, class_names


def _select_deterministic_subset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count > images.shape[0]:
        raise ValueError(
            f"max_samples {sample_count} exceeds split size {images.shape[0]}."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=sample_count, replace=False)
    return images[indices], labels[indices]


def _backend_batches(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    backend: Any,
):
    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size=batch_size,
        shuffle=False,
    ):
        yield (
            to_backend(batch_images, backend, dtype=np.float32),
            to_backend(batch_labels, backend, dtype=np.int64),
        )


def _metric_rows(
    config: FGSMExperimentConfig,
    sweep_results: list[FGSMSweepResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sweep_results:
        rows.append(
            {
                "run_id": config.run_id,
                "backend": config.backend,
                "split": config.split,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "requested_max_samples": config.max_samples,
                "epsilon": result["epsilon"],
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
        )
    return rows


def write_metrics_artifacts(
    run_dir: Path,
    config: FGSMExperimentConfig,
    sweep_results: list[FGSMSweepResult],
) -> dict[str, str]:
    """Write JSON and CSV metric artifacts for later analysis."""
    rows = _metric_rows(config, sweep_results)
    metrics_json_path = save_metrics(
        {
            "schema_version": 1,
            "metric_fields": list(METRIC_FIELDS),
            "metric_semantics": {
                "attack_success_rate": (
                    "successful_attacks / clean_correct_samples, or 0.0 "
                    "when there are no clean-correct samples"
                ),
                "accuracy_drop": "clean_accuracy - adversarial_accuracy",
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
    config: FGSMExperimentConfig,
    sweep_results: list[FGSMSweepResult],
    timing: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    epsilon_zero_result = next(
        (result for result in sweep_results if result["epsilon"] == 0.0),
        None,
    )
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "status": "COMPLETED",
        "backend": config.backend,
        "split": config.split,
        "sample_count": sweep_results[0]["total_samples"],
        "batch_size": config.batch_size,
        "epsilon_values": list(config.epsilon_values),
        "epsilon_count": len(config.epsilon_values),
        "clean_accuracy_at_epsilon_zero": (
            None
            if epsilon_zero_result is None
            else epsilon_zero_result["clean_accuracy"]
        ),
        "final_epsilon": sweep_results[-1]["epsilon"],
        "final_adversarial_accuracy": sweep_results[-1]["adversarial_accuracy"],
        "final_attack_success_rate": sweep_results[-1]["attack_success_rate"],
        "timing": timing,
        "artifacts": artifacts,
    }


def run_fgsm_experiment(config: FGSMExperimentConfig) -> dict[str, Any]:
    """Run one configured FGSM robustness experiment and write artifacts."""
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
        sweep_results = evaluate_fgsm_epsilon_sweep(
            model,
            loss_function,
            batches,
            config.epsilon_values,
        )
        synchronize_backend(xp)
        evaluation_seconds = time.perf_counter() - evaluation_start

        total_seconds = time.perf_counter() - total_start
        sample_epsilon_pairs = config.max_samples * len(config.epsilon_values)
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
            "sample_count": config.max_samples,
            "epsilon_count": len(config.epsilon_values),
            "sample_epsilon_pairs": sample_epsilon_pairs,
            "evaluation_sample_epsilon_pairs_per_second": (
                sample_epsilon_pairs / evaluation_seconds
                if evaluation_seconds > 0.0
                else None
            ),
        }
        timing_path = save_metrics(timing, run_dir / "timing.json")
        metric_artifacts = write_metrics_artifacts(
            run_dir,
            config,
            sweep_results,
        )

        artifacts = {
            "config": str(config_path),
            "environment": str(environment_path),
            "timing": str(timing_path),
            **metric_artifacts,
        }
        summary = _summary_payload(config, sweep_results, timing, artifacts)
        summary["dataset"] = {
            **dataset_metadata,
            "split": config.split,
            "available_split_samples": int(images.shape[0]),
            "evaluated_samples": config.max_samples,
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
            "sweep_results": sweep_results,
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
            "Run a scheduler-neutral FGSM robustness experiment and write "
            "machine-readable artifacts."
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
        "--epsilons",
        default="0,2/255,4/255,8/255,16/255",
        help="Comma-separated epsilon values as floats or fractions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for deterministic sample selection.",
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


def config_from_args(args: argparse.Namespace) -> FGSMExperimentConfig:
    return FGSMExperimentConfig(
        backend=args.backend,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epsilon_values=parse_epsilon_values(args.epsilons),
        seed=args.seed,
        output_root=args.output_root,
        run_id=args.run_id,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = config_from_args(parse_args(argv))
        result = run_fgsm_experiment(config)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"run_id: {config.run_id}")
    print(f"run_dir: {result['run_dir']}")
    print(f"summary: {result['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
