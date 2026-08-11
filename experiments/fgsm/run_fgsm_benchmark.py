"""Benchmark NumPy and CuPy FGSM evaluation using the production runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.default_config import DATA_DIR, PROJECT_ROOT, SEED
from experiments.fgsm.plot_fgsm_run import epsilon_label
from experiments.fgsm.run_fgsm_experiment import (
    FGSMExperimentConfig,
    RUN_ID_PATTERN,
    parse_epsilon_values,
    run_fgsm_experiment,
    utc_timestamp,
)
from src.metrics import save_metrics


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "results" / "checkpoints" / "portfolio_baseline_best.npz"
)
DEFAULT_RAW_RUN_OUTPUT_ROOT = PROJECT_ROOT / "results" / "runs"
DEFAULT_BENCHMARK_OUTPUT_ROOT = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_SAMPLE_COUNTS = (100, 250, 500, 1000, 2000)
DEFAULT_BATCH_SIZES = (8, 16, 32, 64, 128)
DEFAULT_EPSILON_VALUES = (0.0, 4.0 / 255.0)
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_RUNS = 1
RUN_FIELDNAMES = (
    "benchmark_id",
    "suite",
    "backend",
    "sample_count",
    "batch_size",
    "epsilon_values",
    "epsilon_labels",
    "measurement_type",
    "repeat_index",
    "status",
    "run_id",
    "run_dir",
    "evaluation_wall_seconds",
    "total_wall_seconds",
    "sample_epsilon_pairs",
    "evaluation_sample_epsilon_pairs_per_second",
    "git_commit",
    "gpu_name",
    "error_type",
    "error_message",
)
SUMMARY_FIELDNAMES = (
    "benchmark_id",
    "suite",
    "backend",
    "sample_count",
    "batch_size",
    "epsilon_values",
    "epsilon_labels",
    "completed_repeats",
    "failed_repeats",
    "evaluation_wall_seconds_mean",
    "evaluation_wall_seconds_median",
    "evaluation_wall_seconds_std",
    "evaluation_wall_seconds_min",
    "evaluation_wall_seconds_max",
    "total_wall_seconds_mean",
    "total_wall_seconds_median",
    "total_wall_seconds_std",
    "total_wall_seconds_min",
    "total_wall_seconds_max",
    "throughput_mean",
    "throughput_median",
    "throughput_std",
    "throughput_min",
    "throughput_max",
)
SPEEDUP_FIELDNAMES = (
    "benchmark_id",
    "suite",
    "sample_count",
    "batch_size",
    "epsilon_values",
    "epsilon_labels",
    "matched_repeats",
    "cpu_backend",
    "gpu_backend",
    "cpu_evaluation_wall_seconds_mean",
    "gpu_evaluation_wall_seconds_mean",
    "evaluation_speedup_mean",
    "cpu_evaluation_wall_seconds_median",
    "gpu_evaluation_wall_seconds_median",
    "evaluation_speedup_median",
    "paired_evaluation_speedup_mean",
    "paired_evaluation_speedup_median",
    "paired_evaluation_speedup_std",
    "paired_evaluation_speedup_min",
    "paired_evaluation_speedup_max",
    "total_wall_speedup_mean",
    "total_wall_speedup_median",
)
CROSSOVER_ANALYSIS_FILENAME = "crossover_analysis.json"


@dataclass(frozen=True)
class FGSBenchmarkConfig:
    """Effective configuration for one FGSM benchmark matrix."""

    data_dir: Path | str = DATA_DIR
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH
    split: str = "test"
    seed: int = SEED
    raw_run_output_root: Path | str = DEFAULT_RAW_RUN_OUTPUT_ROOT
    benchmark_output_root: Path | str = DEFAULT_BENCHMARK_OUTPUT_ROOT
    benchmark_id: str | None = None
    sample_counts: tuple[int, ...] = DEFAULT_SAMPLE_COUNTS
    sample_scaling_backends: tuple[str, ...] = ("numpy", "cupy")
    sample_scaling_batch_size: int = 32
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    batch_scaling_backends: tuple[str, ...] = ("cupy",)
    batch_scaling_backend: str | None = None
    batch_scaling_sample_count: int = 1000
    epsilon_values: tuple[float, ...] = DEFAULT_EPSILON_VALUES
    repeats: int = DEFAULT_REPEATS
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    run_sample_scaling: bool = True
    run_batch_scaling: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "data_dir",
            "checkpoint_path",
            "raw_run_output_root",
            "benchmark_output_root",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (str, Path)) or (
                isinstance(value, str) and not value.strip()
            ):
                raise ValueError(f"{field_name} must be a valid path.")
            object.__setattr__(self, field_name, Path(value))

        split = self.split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        object.__setattr__(self, "split", split)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        for field_name in (
            "sample_scaling_batch_size",
            "batch_scaling_sample_count",
            "repeats",
        ):
            _require_positive_integer(getattr(self, field_name), field_name)
        _require_non_negative_integer(self.warmup_runs, "warmup_runs")

        sample_counts = _normalize_positive_ints(
            self.sample_counts,
            "sample_counts",
        )
        batch_sizes = _normalize_positive_ints(self.batch_sizes, "batch_sizes")
        object.__setattr__(self, "sample_counts", sample_counts)
        object.__setattr__(self, "batch_sizes", batch_sizes)

        sample_backends = tuple(
            _normalize_backend(backend) for backend in self.sample_scaling_backends
        )
        if not sample_backends:
            raise ValueError("sample_scaling_backends must not be empty.")
        object.__setattr__(self, "sample_scaling_backends", sample_backends)
        raw_batch_backends = self.batch_scaling_backends
        if self.batch_scaling_backend is not None:
            raw_batch_backends = (self.batch_scaling_backend,)
        batch_backends = tuple(
            _normalize_backend(backend) for backend in raw_batch_backends
        )
        if not batch_backends:
            raise ValueError("batch_scaling_backends must not be empty.")
        object.__setattr__(self, "batch_scaling_backends", batch_backends)
        object.__setattr__(
            self,
            "batch_scaling_backend",
            batch_backends[0] if len(batch_backends) == 1 else None,
        )

        if not self.epsilon_values:
            raise ValueError("epsilon_values must not be empty.")
        epsilons = []
        for epsilon in self.epsilon_values:
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon < 0.0
            ):
                raise ValueError("epsilon_values must be non-negative finite numbers.")
            epsilons.append(float(epsilon))
        object.__setattr__(self, "epsilon_values", tuple(epsilons))

        benchmark_id = self.benchmark_id or generate_benchmark_id()
        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string.")
        if not RUN_ID_PATTERN.fullmatch(benchmark_id):
            raise ValueError(
                "benchmark_id may contain only letters, numbers, '.', '_', and '-'."
            )
        object.__setattr__(self, "benchmark_id", benchmark_id)

        if not (self.run_sample_scaling or self.run_batch_scaling):
            raise ValueError("At least one benchmark suite must be enabled.")


@dataclass(frozen=True)
class BenchmarkPoint:
    """One runner invocation in the benchmark matrix."""

    suite: str
    backend: str
    sample_count: int
    batch_size: int
    epsilon_values: tuple[float, ...]
    measurement_type: str
    repeat_index: int


def _require_positive_integer(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")


def _require_non_negative_integer(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")


def _normalize_positive_ints(values: tuple[int, ...], label: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{label} must not be empty.")
    normalized: list[int] = []
    for value in values:
        _require_positive_integer(value, label)
        normalized.append(int(value))
    return tuple(normalized)


def _normalize_backend(value: str) -> str:
    backend = value.lower()
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend values must be 'numpy' or 'cupy'.")
    return backend


def generate_benchmark_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_fgsm_benchmark"


def _parse_int_list(value: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as error:
        raise ValueError(f"{label} must be a comma-separated integer list.") from error
    return _normalize_positive_ints(values, label)


def _parse_backend_list(value: str) -> tuple[str, ...]:
    return tuple(
        _normalize_backend(token.strip())
        for token in value.split(",")
        if token.strip()
    )


def _epsilon_signature(epsilon_values: tuple[float, ...]) -> str:
    return ",".join(f"{epsilon:.17g}" for epsilon in epsilon_values)


def _epsilon_labels(epsilon_values: tuple[float, ...]) -> str:
    return ",".join(epsilon_label(epsilon) for epsilon in epsilon_values)


def config_to_json(config: FGSBenchmarkConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_type": "fgsm_cpu_gpu_scaling",
        "benchmark_id": config.benchmark_id,
        "data_dir": str(config.data_dir),
        "checkpoint_path": str(config.checkpoint_path),
        "split": config.split,
        "seed": config.seed,
        "raw_run_output_root": str(config.raw_run_output_root),
        "benchmark_output_root": str(config.benchmark_output_root),
        "sample_counts": list(config.sample_counts),
        "sample_scaling_backends": list(config.sample_scaling_backends),
        "sample_scaling_batch_size": config.sample_scaling_batch_size,
        "batch_sizes": list(config.batch_sizes),
        "batch_scaling_backends": list(config.batch_scaling_backends),
        "batch_scaling_backend": config.batch_scaling_backend,
        "batch_scaling_sample_count": config.batch_scaling_sample_count,
        "epsilon_values": list(config.epsilon_values),
        "epsilon_labels": [epsilon_label(value) for value in config.epsilon_values],
        "repeats": config.repeats,
        "warmup_runs": config.warmup_runs,
        "run_sample_scaling": config.run_sample_scaling,
        "run_batch_scaling": config.run_batch_scaling,
        "overwrite": config.overwrite,
        "speedup_definition": (
            "CPU evaluation_wall_seconds / GPU evaluation_wall_seconds for "
            "matched workloads"
        ),
        "std_definition": "sample standard deviation; 0.0 for one repeat",
        "warmup_policy": (
            "Warm-up runner invocations use the same workload as measured "
            "repeats and are excluded from aggregate statistics."
        ),
    }


def build_benchmark_plan(config: FGSBenchmarkConfig) -> list[BenchmarkPoint]:
    """Build the ordered runner invocation matrix."""
    points: list[BenchmarkPoint] = []

    def append_point_set(
        *,
        suite: str,
        backend: str,
        sample_count: int,
        batch_size: int,
    ) -> None:
        for index in range(config.warmup_runs):
            points.append(
                BenchmarkPoint(
                    suite=suite,
                    backend=backend,
                    sample_count=sample_count,
                    batch_size=batch_size,
                    epsilon_values=config.epsilon_values,
                    measurement_type="warmup",
                    repeat_index=index,
                )
            )
        for index in range(config.repeats):
            points.append(
                BenchmarkPoint(
                    suite=suite,
                    backend=backend,
                    sample_count=sample_count,
                    batch_size=batch_size,
                    epsilon_values=config.epsilon_values,
                    measurement_type="measured",
                    repeat_index=index,
                )
            )

    if config.run_sample_scaling:
        for sample_count in config.sample_counts:
            for backend in config.sample_scaling_backends:
                append_point_set(
                    suite="sample_count_scaling",
                    backend=backend,
                    sample_count=sample_count,
                    batch_size=config.sample_scaling_batch_size,
                )

    if config.run_batch_scaling:
        for batch_size in config.batch_sizes:
            for backend in config.batch_scaling_backends:
                append_point_set(
                    suite="batch_size_scaling",
                    backend=backend,
                    sample_count=config.batch_scaling_sample_count,
                    batch_size=batch_size,
                )

    return points


def benchmark_run_id(config: FGSBenchmarkConfig, point: BenchmarkPoint) -> str:
    suite_prefix = "samples" if point.suite == "sample_count_scaling" else "batch"
    measurement_prefix = "warmup" if point.measurement_type == "warmup" else "repeat"
    return (
        f"{config.benchmark_id}_{suite_prefix}_{point.backend}_"
        f"n{point.sample_count}_b{point.batch_size}_"
        f"{measurement_prefix}{point.repeat_index}"
    )


def prepare_benchmark_directory(config: FGSBenchmarkConfig) -> Path:
    benchmark_dir = Path(config.benchmark_output_root) / str(config.benchmark_id)
    if benchmark_dir.exists() and not config.overwrite:
        raise FileExistsError(
            "Benchmark directory already exists and will not be overwritten: "
            f"{benchmark_dir}"
        )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "plots").mkdir(exist_ok=True)
    return benchmark_dir


def _load_json_if_present(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return {}
    with artifact_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return payload
    return {}


def _base_row(
    config: FGSBenchmarkConfig,
    point: BenchmarkPoint,
    *,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "benchmark_id": config.benchmark_id,
        "suite": point.suite,
        "backend": point.backend,
        "sample_count": point.sample_count,
        "batch_size": point.batch_size,
        "epsilon_values": _epsilon_signature(point.epsilon_values),
        "epsilon_labels": _epsilon_labels(point.epsilon_values),
        "measurement_type": point.measurement_type,
        "repeat_index": point.repeat_index,
        "status": "PENDING",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "evaluation_wall_seconds": None,
        "total_wall_seconds": None,
        "sample_epsilon_pairs": point.sample_count * len(point.epsilon_values),
        "evaluation_sample_epsilon_pairs_per_second": None,
        "git_commit": None,
        "gpu_name": None,
        "error_type": None,
        "error_message": None,
    }


def run_benchmark_point(
    config: FGSBenchmarkConfig,
    point: BenchmarkPoint,
) -> dict[str, Any]:
    """Run one benchmark point through the production FGSM experiment runner."""
    run_id = benchmark_run_id(config, point)
    run_dir = Path(config.raw_run_output_root) / run_id
    row = _base_row(config, point, run_id=run_id, run_dir=run_dir)

    runner_config = FGSMExperimentConfig(
        backend=point.backend,
        data_dir=config.data_dir,
        checkpoint_path=config.checkpoint_path,
        split=config.split,
        max_samples=point.sample_count,
        batch_size=point.batch_size,
        epsilon_values=point.epsilon_values,
        seed=config.seed,
        output_root=config.raw_run_output_root,
        run_id=run_id,
    )

    try:
        result = run_fgsm_experiment(runner_config)
    except Exception as error:
        row["status"] = "FAILED"
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
        return row

    timing = result["timing"]
    artifacts = result.get("artifacts", {})
    environment = _load_json_if_present(artifacts.get("environment"))
    cupy = environment.get("cupy", {})
    git = environment.get("git", {})

    row.update(
        {
            "status": "COMPLETED",
            "run_dir": str(result["run_dir"]),
            "evaluation_wall_seconds": timing.get("evaluation_wall_seconds"),
            "total_wall_seconds": timing.get("total_wall_seconds"),
            "sample_epsilon_pairs": timing.get("sample_epsilon_pairs"),
            "evaluation_sample_epsilon_pairs_per_second": timing.get(
                "evaluation_sample_epsilon_pairs_per_second"
            ),
            "git_commit": git.get("commit"),
            "gpu_name": cupy.get("gpu_name"),
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def write_run_rows(benchmark_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    json_path = save_metrics(
        {
            "schema_version": 1,
            "rows": rows,
        },
        benchmark_dir / "benchmark_runs.json",
    )
    csv_path = _write_csv(benchmark_dir / "benchmark_runs.csv", rows, RUN_FIELDNAMES)
    return {"benchmark_runs_json": json_path, "benchmark_runs_csv": csv_path}


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    if len(values) == 1:
        std = 0.0
    else:
        std = statistics.stdev(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def _finite_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured completed repeats by workload."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    failed_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        if row["measurement_type"] != "measured":
            continue
        key = (
            row["benchmark_id"],
            row["suite"],
            row["backend"],
            row["sample_count"],
            row["batch_size"],
            row["epsilon_values"],
            row["epsilon_labels"],
        )
        if row["status"] == "COMPLETED":
            groups.setdefault(key, []).append(row)
        else:
            failed_counts[key] = failed_counts.get(key, 0) + 1

    summaries: list[dict[str, Any]] = []
    for key in sorted(set(groups) | set(failed_counts)):
        completed_rows = groups.get(key, [])
        eval_values = [
            value
            for value in (
                _finite_float(row["evaluation_wall_seconds"])
                for row in completed_rows
            )
            if value is not None
        ]
        total_values = [
            value
            for value in (
                _finite_float(row["total_wall_seconds"]) for row in completed_rows
            )
            if value is not None
        ]
        throughput_values = [
            value
            for value in (
                _finite_float(row["evaluation_sample_epsilon_pairs_per_second"])
                for row in completed_rows
            )
            if value is not None
        ]
        eval_stats = _stats(eval_values)
        total_stats = _stats(total_values)
        throughput_stats = _stats(throughput_values)
        (
            benchmark_id,
            suite,
            backend,
            sample_count,
            batch_size,
            epsilon_values,
            epsilon_labels,
        ) = key
        summaries.append(
            {
                "benchmark_id": benchmark_id,
                "suite": suite,
                "backend": backend,
                "sample_count": sample_count,
                "batch_size": batch_size,
                "epsilon_values": epsilon_values,
                "epsilon_labels": epsilon_labels,
                "completed_repeats": len(completed_rows),
                "failed_repeats": failed_counts.get(key, 0),
                "evaluation_wall_seconds_mean": eval_stats["mean"],
                "evaluation_wall_seconds_median": eval_stats["median"],
                "evaluation_wall_seconds_std": eval_stats["std"],
                "evaluation_wall_seconds_min": eval_stats["min"],
                "evaluation_wall_seconds_max": eval_stats["max"],
                "total_wall_seconds_mean": total_stats["mean"],
                "total_wall_seconds_median": total_stats["median"],
                "total_wall_seconds_std": total_stats["std"],
                "total_wall_seconds_min": total_stats["min"],
                "total_wall_seconds_max": total_stats["max"],
                "throughput_mean": throughput_stats["mean"],
                "throughput_median": throughput_stats["median"],
                "throughput_std": throughput_stats["std"],
                "throughput_min": throughput_stats["min"],
                "throughput_max": throughput_stats["max"],
            }
        )
    return summaries


def _summary_lookup(
    summaries: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row["suite"],
            row["backend"],
            row["sample_count"],
            row["batch_size"],
            row["epsilon_values"],
        ): row
        for row in summaries
    }


def compute_speedup_rows(
    run_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute CPU/GPU speedup for matched completed measured workloads."""
    lookup = _summary_lookup(summaries)
    speedups: list[dict[str, Any]] = []
    for key, cpu_summary in sorted(lookup.items()):
        suite, backend, sample_count, batch_size, epsilon_values = key
        if backend != "numpy":
            continue
        gpu_summary = lookup.get(
            (suite, "cupy", sample_count, batch_size, epsilon_values)
        )
        if gpu_summary is None:
            continue
        cpu_mean = _finite_float(cpu_summary["evaluation_wall_seconds_mean"])
        gpu_mean = _finite_float(gpu_summary["evaluation_wall_seconds_mean"])
        cpu_median = _finite_float(cpu_summary["evaluation_wall_seconds_median"])
        gpu_median = _finite_float(gpu_summary["evaluation_wall_seconds_median"])
        cpu_total_mean = _finite_float(cpu_summary["total_wall_seconds_mean"])
        gpu_total_mean = _finite_float(gpu_summary["total_wall_seconds_mean"])
        cpu_total_median = _finite_float(cpu_summary["total_wall_seconds_median"])
        gpu_total_median = _finite_float(gpu_summary["total_wall_seconds_median"])
        if cpu_mean is None or gpu_mean is None or gpu_mean <= 0.0:
            continue
        if cpu_median is None or gpu_median is None or gpu_median <= 0.0:
            continue

        paired_speedups = _paired_evaluation_speedups(
            run_rows,
            suite=suite,
            sample_count=sample_count,
            batch_size=batch_size,
            epsilon_values=epsilon_values,
        )
        paired_stats = _stats(paired_speedups)
        speedups.append(
            {
                "benchmark_id": cpu_summary["benchmark_id"],
                "suite": suite,
                "sample_count": sample_count,
                "batch_size": batch_size,
                "epsilon_values": epsilon_values,
                "epsilon_labels": cpu_summary["epsilon_labels"],
                "matched_repeats": len(paired_speedups),
                "cpu_backend": "numpy",
                "gpu_backend": "cupy",
                "cpu_evaluation_wall_seconds_mean": cpu_mean,
                "gpu_evaluation_wall_seconds_mean": gpu_mean,
                "evaluation_speedup_mean": cpu_mean / gpu_mean,
                "cpu_evaluation_wall_seconds_median": cpu_median,
                "gpu_evaluation_wall_seconds_median": gpu_median,
                "evaluation_speedup_median": cpu_median / gpu_median,
                "paired_evaluation_speedup_mean": paired_stats["mean"],
                "paired_evaluation_speedup_median": paired_stats["median"],
                "paired_evaluation_speedup_std": paired_stats["std"],
                "paired_evaluation_speedup_min": paired_stats["min"],
                "paired_evaluation_speedup_max": paired_stats["max"],
                "total_wall_speedup_mean": (
                    None
                    if cpu_total_mean is None
                    or gpu_total_mean is None
                    or gpu_total_mean <= 0.0
                    else cpu_total_mean / gpu_total_mean
                ),
                "total_wall_speedup_median": (
                    None
                    if cpu_total_median is None
                    or gpu_total_median is None
                    or gpu_total_median <= 0.0
                    else cpu_total_median / gpu_total_median
                ),
            }
        )
    return speedups


def detect_speedup_crossover(
    speedups: list[dict[str, Any]],
    *,
    suite: str = "batch_size_scaling",
    speedup_key: str = "evaluation_speedup_median",
) -> dict[str, Any]:
    """Summarize the first and maximum observed GPU speedups for one suite."""
    rows = sorted(
        [
            row
            for row in speedups
            if row["suite"] == suite and _finite_float(row.get(speedup_key)) is not None
        ],
        key=lambda row: row["batch_size"],
    )
    first_gpu_faster = next(
        (row for row in rows if float(row[speedup_key]) > 1.0),
        None,
    )
    max_row = (
        None
        if not rows
        else max(rows, key=lambda row: float(row[speedup_key]))
    )
    return {
        "schema_version": 1,
        "suite": suite,
        "speedup_metric": speedup_key,
        "break_even_speedup": 1.0,
        "tested_batch_sizes": [row["batch_size"] for row in rows],
        "first_gpu_faster_batch_size": (
            None if first_gpu_faster is None else first_gpu_faster["batch_size"]
        ),
        "first_gpu_faster_speedup": (
            None if first_gpu_faster is None else first_gpu_faster[speedup_key]
        ),
        "max_speedup_batch_size": None if max_row is None else max_row["batch_size"],
        "max_speedup": None if max_row is None else max_row[speedup_key],
    }


def _paired_evaluation_speedups(
    run_rows: list[dict[str, Any]],
    *,
    suite: str,
    sample_count: int,
    batch_size: int,
    epsilon_values: str,
) -> list[float]:
    cpu_by_repeat: dict[int, float] = {}
    gpu_by_repeat: dict[int, float] = {}
    for row in run_rows:
        if (
            row["suite"] != suite
            or row["sample_count"] != sample_count
            or row["batch_size"] != batch_size
            or row["epsilon_values"] != epsilon_values
            or row["measurement_type"] != "measured"
            or row["status"] != "COMPLETED"
        ):
            continue
        value = _finite_float(row["evaluation_wall_seconds"])
        if value is None or value <= 0.0:
            continue
        if row["backend"] == "numpy":
            cpu_by_repeat[int(row["repeat_index"])] = value
        elif row["backend"] == "cupy":
            gpu_by_repeat[int(row["repeat_index"])] = value

    speedups: list[float] = []
    for repeat_index in sorted(set(cpu_by_repeat) & set(gpu_by_repeat)):
        gpu_value = gpu_by_repeat[repeat_index]
        if gpu_value > 0.0:
            speedups.append(cpu_by_repeat[repeat_index] / gpu_value)
    return speedups


def write_summary_artifacts(
    benchmark_dir: Path,
    summaries: list[dict[str, Any]],
    speedups: list[dict[str, Any]],
) -> dict[str, Path]:
    crossover_analysis = detect_speedup_crossover(speedups)
    summary_json = save_metrics(
        {"schema_version": 1, "rows": summaries},
        benchmark_dir / "benchmark_summary.json",
    )
    summary_csv = _write_csv(
        benchmark_dir / "benchmark_summary.csv",
        summaries,
        SUMMARY_FIELDNAMES,
    )
    speedup_json = save_metrics(
        {
            "schema_version": 1,
            "speedup_definition": (
                "CPU evaluation_wall_seconds / GPU evaluation_wall_seconds "
                "for matched workloads"
            ),
            "crossover_analysis": crossover_analysis,
            "rows": speedups,
        },
        benchmark_dir / "speedup_summary.json",
    )
    speedup_csv = _write_csv(
        benchmark_dir / "speedup_summary.csv",
        speedups,
        SPEEDUP_FIELDNAMES,
    )
    crossover_json = save_metrics(
        crossover_analysis,
        benchmark_dir / CROSSOVER_ANALYSIS_FILENAME,
    )
    return {
        "benchmark_summary_json": summary_json,
        "benchmark_summary_csv": summary_csv,
        "speedup_summary_json": speedup_json,
        "speedup_summary_csv": speedup_csv,
        "crossover_analysis_json": crossover_json,
    }


def _plot_metric_by_sample_count(
    summaries: list[dict[str, Any]],
    *,
    metric_key: str,
    error_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    rows = [
        row
        for row in summaries
        if row["suite"] == "sample_count_scaling"
        and row["completed_repeats"] > 0
        and _finite_float(row[metric_key]) is not None
    ]
    if not rows:
        raise ValueError("No sample-count benchmark rows are available for plotting.")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for backend, color in (("numpy", "#4c78a8"), ("cupy", "#f58518")):
        backend_rows = sorted(
            [row for row in rows if row["backend"] == backend],
            key=lambda row: row["sample_count"],
        )
        if not backend_rows:
            continue
        x_values = [row["sample_count"] for row in backend_rows]
        y_values = [float(row[metric_key]) for row in backend_rows]
        y_errors = [
            0.0 if _finite_float(row[error_key]) is None else float(row[error_key])
            for row in backend_rows
        ]
        axes.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            linewidth=2.0,
            capsize=4,
            label=backend,
            color=color,
        )
    axes.set_xlabel("Sample count")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(title="Backend")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_speedup_by_sample_count(
    speedups: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    rows = sorted(
        [
            row
            for row in speedups
            if row["suite"] == "sample_count_scaling"
            and _finite_float(row["evaluation_speedup_median"]) is not None
        ],
        key=lambda row: row["sample_count"],
    )
    if not rows:
        raise ValueError("No matched CPU/GPU speedup rows are available.")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(
        [row["sample_count"] for row in rows],
        [float(row["evaluation_speedup_median"]) for row in rows],
        marker="o",
        linewidth=2.0,
        color="#54a24b",
        label="Median evaluation speedup",
    )
    axes.axhline(1.0, color="0.5", linestyle="--", linewidth=1.2)
    axes.set_xlabel("Sample count")
    axes.set_ylabel("CPU / GPU evaluation-wall speedup")
    axes.set_title("FGSM Benchmark: GPU Speedup vs Sample Count")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_cupy_batch_metric(
    summaries: list[dict[str, Any]],
    *,
    metric_key: str,
    error_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    rows = sorted(
        [
            row
            for row in summaries
            if row["suite"] == "batch_size_scaling"
            and row["backend"] == "cupy"
            and row["completed_repeats"] > 0
            and _finite_float(row[metric_key]) is not None
        ],
        key=lambda row: row["batch_size"],
    )
    if not rows:
        raise ValueError("No CuPy batch-size benchmark rows are available.")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.errorbar(
        [row["batch_size"] for row in rows],
        [float(row[metric_key]) for row in rows],
        yerr=[
            0.0 if _finite_float(row[error_key]) is None else float(row[error_key])
            for row in rows
        ],
        marker="o",
        linewidth=2.0,
        capsize=4,
        color="#f58518",
        label="cupy",
    )
    axes.set_xlabel("Batch size")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(title="Backend")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_metric_by_batch_size(
    summaries: list[dict[str, Any]],
    *,
    metric_key: str,
    error_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    rows = [
        row
        for row in summaries
        if row["suite"] == "batch_size_scaling"
        and row["completed_repeats"] > 0
        and _finite_float(row[metric_key]) is not None
    ]
    if not rows:
        raise ValueError("No batch-size benchmark rows are available for plotting.")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for backend, color in (("numpy", "#4c78a8"), ("cupy", "#f58518")):
        backend_rows = sorted(
            [row for row in rows if row["backend"] == backend],
            key=lambda row: row["batch_size"],
        )
        if not backend_rows:
            continue
        axes.errorbar(
            [row["batch_size"] for row in backend_rows],
            [float(row[metric_key]) for row in backend_rows],
            yerr=[
                0.0
                if _finite_float(row[error_key]) is None
                else float(row[error_key])
                for row in backend_rows
            ],
            marker="o",
            linewidth=2.0,
            capsize=4,
            label=backend,
            color=color,
        )
    axes.set_xlabel("Batch size")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend(title="Backend")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def _plot_speedup_by_batch_size(
    speedups: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    rows = sorted(
        [
            row
            for row in speedups
            if row["suite"] == "batch_size_scaling"
            and _finite_float(row["evaluation_speedup_median"]) is not None
        ],
        key=lambda row: row["batch_size"],
    )
    if not rows:
        raise ValueError("No matched batch-size speedup rows are available.")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(
        [row["batch_size"] for row in rows],
        [float(row["evaluation_speedup_median"]) for row in rows],
        marker="o",
        linewidth=2.0,
        color="#54a24b",
        label="Median evaluation speedup",
    )
    axes.axhline(
        1.0,
        color="0.5",
        linestyle="--",
        linewidth=1.2,
        label="Break-even",
    )
    axes.set_xlabel("Batch size")
    axes.set_ylabel("CPU / GPU evaluation-wall speedup")
    axes.set_title("FGSM Benchmark: GPU Speedup vs Batch Size")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def plot_benchmark_results(
    benchmark_dir: Path,
    summaries: list[dict[str, Any]],
    speedups: list[dict[str, Any]],
) -> dict[str, Path]:
    plots_dir = benchmark_dir / "plots"
    outputs: dict[str, Path] = {}
    plotters = (
        (
            "runtime_vs_sample_count",
            lambda: _plot_metric_by_sample_count(
                summaries,
                metric_key="evaluation_wall_seconds_median",
                error_key="evaluation_wall_seconds_std",
                ylabel="Evaluation wall time (seconds)",
                title="FGSM Benchmark: Runtime vs Sample Count",
                output_path=plots_dir / "runtime_vs_sample_count.png",
            ),
        ),
        (
            "throughput_vs_sample_count",
            lambda: _plot_metric_by_sample_count(
                summaries,
                metric_key="throughput_median",
                error_key="throughput_std",
                ylabel="Sample-epsilon pairs / second",
                title="FGSM Benchmark: Throughput vs Sample Count",
                output_path=plots_dir / "throughput_vs_sample_count.png",
            ),
        ),
        (
            "speedup_vs_sample_count",
            lambda: _plot_speedup_by_sample_count(
                speedups,
                plots_dir / "speedup_vs_sample_count.png",
            ),
        ),
        (
            "runtime_vs_batch_size",
            lambda: _plot_metric_by_batch_size(
                summaries,
                metric_key="evaluation_wall_seconds_median",
                error_key="evaluation_wall_seconds_std",
                ylabel="Evaluation wall time (seconds)",
                title="FGSM Benchmark: Runtime vs Batch Size",
                output_path=plots_dir / "runtime_vs_batch_size.png",
            ),
        ),
        (
            "throughput_vs_batch_size",
            lambda: _plot_metric_by_batch_size(
                summaries,
                metric_key="throughput_median",
                error_key="throughput_std",
                ylabel="Sample-epsilon pairs / second",
                title="FGSM Benchmark: Throughput vs Batch Size",
                output_path=plots_dir / "throughput_vs_batch_size.png",
            ),
        ),
        (
            "speedup_vs_batch_size",
            lambda: _plot_speedup_by_batch_size(
                speedups,
                plots_dir / "speedup_vs_batch_size.png",
            ),
        ),
        (
            "cupy_runtime_vs_batch_size",
            lambda: _plot_cupy_batch_metric(
                summaries,
                metric_key="evaluation_wall_seconds_median",
                error_key="evaluation_wall_seconds_std",
                ylabel="Evaluation wall time (seconds)",
                title="FGSM Benchmark: CuPy Runtime vs Batch Size",
                output_path=plots_dir / "cupy_runtime_vs_batch_size.png",
            ),
        ),
        (
            "cupy_throughput_vs_batch_size",
            lambda: _plot_cupy_batch_metric(
                summaries,
                metric_key="throughput_median",
                error_key="throughput_std",
                ylabel="Sample-epsilon pairs / second",
                title="FGSM Benchmark: CuPy Throughput vs Batch Size",
                output_path=plots_dir / "cupy_throughput_vs_batch_size.png",
            ),
        ),
    )
    for name, plotter in plotters:
        try:
            outputs[name] = plotter()
        except ValueError:
            continue
    return outputs


def write_benchmark_status(
    benchmark_dir: Path,
    status: str,
    *,
    started_at: str,
    ended_at: str | None = None,
    error: dict[str, str] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    if error is not None:
        payload["error"] = error
    return save_metrics(payload, benchmark_dir / "status.json")


def run_fgsm_benchmark(config: FGSBenchmarkConfig) -> dict[str, Any]:
    """Run the configured benchmark matrix and write aggregate artifacts."""
    benchmark_dir = prepare_benchmark_directory(config)
    started_at = utc_timestamp()
    write_benchmark_status(benchmark_dir, "RUNNING", started_at=started_at)
    config_path = save_metrics(config_to_json(config), benchmark_dir / "config.json")
    rows: list[dict[str, Any]] = []
    run_artifacts: dict[str, Path] = {}

    try:
        for point in build_benchmark_plan(config):
            row = run_benchmark_point(config, point)
            rows.append(row)
            run_artifacts = write_run_rows(benchmark_dir, rows)

        summaries = aggregate_benchmark_rows(rows)
        speedups = compute_speedup_rows(rows, summaries)
        summary_artifacts = write_summary_artifacts(
            benchmark_dir,
            summaries,
            speedups,
        )
        plot_artifacts = plot_benchmark_results(benchmark_dir, summaries, speedups)
        status = "COMPLETED_WITH_FAILURES" if any(
            row["status"] == "FAILED" for row in rows
        ) else "COMPLETED"
        status_path = write_benchmark_status(
            benchmark_dir,
            status,
            started_at=started_at,
            ended_at=utc_timestamp(),
        )
    except Exception as error:
        write_benchmark_status(
            benchmark_dir,
            "FAILED",
            started_at=started_at,
            ended_at=utc_timestamp(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise

    return {
        "benchmark_dir": benchmark_dir,
        "config": config,
        "config_path": config_path,
        "rows": rows,
        "summaries": summaries,
        "speedups": speedups,
        "artifacts": {
            **run_artifacts,
            **summary_artifacts,
            **plot_artifacts,
            "status": status_path,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable CPU/GPU FGSM benchmarks by launching the existing "
            "FGSM experiment runner for each matrix point."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--raw-run-output-root",
        type=Path,
        default=DEFAULT_RAW_RUN_OUTPUT_ROOT,
        help="Ignored raw runner output root, normally results/runs.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_ROOT,
        help="Benchmark aggregate output root, normally results/benchmarks.",
    )
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument(
        "--sample-counts",
        default="100,250,500,1000,2000",
        help="Comma-separated sample counts for NumPy/CuPy sample scaling.",
    )
    parser.add_argument(
        "--sample-scaling-backends",
        default="numpy,cupy",
        help="Comma-separated backends for sample-count scaling.",
    )
    parser.add_argument("--sample-scaling-batch-size", type=int, default=32)
    parser.add_argument(
        "--batch-sizes",
        default="8,16,32,64,128",
        help="Comma-separated batch sizes for batch-size scaling.",
    )
    parser.add_argument(
        "--batch-scaling-backends",
        default="cupy",
        help="Comma-separated backends for batch-size scaling.",
    )
    parser.add_argument(
        "--batch-scaling-backend",
        choices=("numpy", "cupy"),
        default=None,
        help="Deprecated single-backend alias for --batch-scaling-backends.",
    )
    parser.add_argument("--batch-scaling-sample-count", type=int, default=1000)
    parser.add_argument(
        "--epsilons",
        default="0,4/255",
        help="Comma-separated epsilon workload, using floats or fractions.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument(
        "--skip-sample-scaling",
        action="store_true",
        help="Disable NumPy/CuPy sample-count scaling.",
    )
    parser.add_argument(
        "--skip-batch-scaling",
        action="store_true",
        help="Disable batch-size scaling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing benchmark_id output directory.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FGSBenchmarkConfig:
    batch_scaling_backends = _parse_backend_list(args.batch_scaling_backends)
    if args.batch_scaling_backend is not None:
        batch_scaling_backends = (args.batch_scaling_backend,)
    return FGSBenchmarkConfig(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        seed=args.seed,
        raw_run_output_root=args.raw_run_output_root,
        benchmark_output_root=args.benchmark_output_root,
        benchmark_id=args.benchmark_id,
        sample_counts=_parse_int_list(args.sample_counts, "sample_counts"),
        sample_scaling_backends=_parse_backend_list(args.sample_scaling_backends),
        sample_scaling_batch_size=args.sample_scaling_batch_size,
        batch_sizes=_parse_int_list(args.batch_sizes, "batch_sizes"),
        batch_scaling_backends=batch_scaling_backends,
        batch_scaling_sample_count=args.batch_scaling_sample_count,
        epsilon_values=parse_epsilon_values(args.epsilons),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        run_sample_scaling=not args.skip_sample_scaling,
        run_batch_scaling=not args.skip_batch_scaling,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = config_from_args(parse_args(argv))
        result = run_fgsm_benchmark(config)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"benchmark_id: {config.benchmark_id}")
    print(f"benchmark_dir: {result['benchmark_dir']}")
    print(f"completed_invocations: {len(result['rows'])}")
    print(f"summary: {result['artifacts']['benchmark_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
